#!/usr/bin/env python3
"""Four-GPU browser MVP for persistent LingBot World interaction."""

import argparse
import base64
import json
import logging
import os
import sys
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from generate import create_causal_pipeline
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.interactive import InteractiveSession


HTML = r"""<!doctype html>
<meta charset="utf-8"><title>LingBot World Interactive MVP</title>
<style>
body{margin:0;background:#111;color:#eee;font:15px system-ui;display:grid;place-items:center;min-height:100vh}
#app{width:min(96vw,1000px)} canvas{width:100%;background:#000;border-radius:8px}
#bar{display:flex;gap:16px;align-items:center;margin:10px 0}.ok{color:#72e39a}.busy{color:#ffd166}
kbd{background:#333;padding:2px 6px;border-radius:4px}button{padding:8px 14px}
</style><div id="app"><canvas id="view" width="832" height="464"></canvas>
<div id="bar"><button id="go">开始 / 捕获鼠标</button><span id="state">未启动</span><span id="timing"></span></div>
<div><kbd>WASD</kbd> 移动，<kbd>Q/E</kbd> 上下，鼠标控制视角，<kbd>Esc</kbd> 释放鼠标。</div></div>
<script>
const c=document.querySelector('#view'),ctx=c.getContext('2d'),state=document.querySelector('#state'),timing=document.querySelector('#timing');
const keys=new Set();let dx=0,dy=0,running=false,busy=false,frames=[],frameDelay=250,lastFrameAt=0;
addEventListener('keydown',e=>keys.add(e.key.toLowerCase()));addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));
c.addEventListener('mousemove',e=>{if(document.pointerLockElement===c){dx+=e.movementX;dy+=e.movementY}});
document.querySelector('#go').onclick=()=>{c.requestPointerLock();running=true;step()};
async function step(){if(!running||busy)return;busy=true;state.textContent='生成下一段…';state.className='busy';
 const payload={keys:[...keys],mouse_dx:dx,mouse_dy:dy};dx=dy=0;
 try{const r=await fetch('/step',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});const o=await r.json();
  if(!r.ok)throw Error(o.error||r.statusText);frames.push(...o.frames);const fps=o.timing.frames/o.timing.total_s;frameDelay=1000/fps;timing.textContent=`step ${o.timing.step}: ${fps.toFixed(2)} fps / DiT ${o.timing.dit_s.toFixed(2)}s / VAE ${o.timing.decode_s.toFixed(2)}s`;
  state.textContent=`播放队列 ${frames.length} 帧`;state.className='ok';
 }catch(e){state.textContent=e;running=false}finally{busy=false;if(running)step()}}
function render(now){if(frames.length&&now-lastFrameAt>=frameDelay){lastFrameAt=now;const im=new Image();im.onload=()=>{if(c.width!==im.width){c.width=im.width;c.height=im.height}ctx.drawImage(im,0,0)};im.src='data:image/jpeg;base64,'+frames.shift()}requestAnimationFrame(render)}requestAnimationFrame(render);
</script>"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--image", default="examples/03/image.jpg")
    p.add_argument("--intrinsics", default="examples/03/intrinsics.npy")
    p.add_argument("--prompt", default="A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds.")
    p.add_argument("--size", default="480*832")
    p.add_argument("--max_area", type=int, default=None,
                   help="Override output pixel area; MVP default script uses 240x416.")
    p.add_argument("--chunk_size", type=int, default=4)
    p.add_argument("--ulysses_size", type=int, default=4)
    p.add_argument("--local_attn_size", type=int, default=18)
    p.add_argument("--sink_size", type=int, default=6)
    p.add_argument("--sample_shift", type=float, default=10.0)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no_prewarm", action="store_true",
                   help="Skip the throw-away warmup chunk before accepting users.")
    return p.parse_args()


def frames_to_jpegs(video):
    array = ((video.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    encoded = []
    for frame in array:
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        encoded.append(base64.b64encode(data).decode("ascii"))
    return encoded


def main():
    args = parse_args()
    rank = int(os.getenv("RANK", "0")); world = int(os.getenv("WORLD_SIZE", "1")); local = int(os.getenv("LOCAL_RANK", "0"))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        format="[%(asctime)s] %(levelname)s: %(message)s", stream=sys.stdout)
    if world != args.ulysses_size:
        raise ValueError("WORLD_SIZE must equal --ulysses_size")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world,
                            timeout=timedelta(hours=2))
    init_distributed_group()
    cfg = WAN_CONFIGS["i2v-A14B"]
    model_args = argparse.Namespace(
        ckpt_dir=args.ckpt_dir, t5_fsdp=False, dit_fsdp=False,
        ulysses_size=args.ulysses_size, t5_cpu=False, t5_rank0_cpu=False,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size)
    pipe = create_causal_pipeline(model_args, cfg, local, rank, mode="causal_fast")
    image = Image.open(args.image).convert("RGB")
    intrinsics = np.load(args.intrinsics)[0]

    def make_session():
        return InteractiveSession(
            pipe, args.prompt, image, intrinsics,
            max_area=(args.max_area if args.max_area is not None
                      else MAX_AREA_CONFIGS[args.size]),
            chunk_size=args.chunk_size,
            shift=args.sample_shift, seed=args.base_seed)

    session = make_session()
    if not args.no_prewarm:
        _, warm_timing = session.step({"keys": [], "mouse_dx": 0, "mouse_dy": 0})
        session.close()
        if rank == 0:
            logging.info("Interactive prewarm complete: %s", warm_timing)
        session = make_session()
    if rank != 0:
        while True:
            command = [None]
            dist.broadcast_object_list(command, src=0)
            if command[0].get("type") == "stop":
                break
            session.step(command[0]["action"])
        session.close(); dist.destroy_process_group(); return

    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, data, content_type="application/json"):
            body = data if isinstance(data, bytes) else data.encode("utf-8")
            self.send_response(status); self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == "/": self._send(200, HTML, "text/html; charset=utf-8")
            else: self._send(404, '{"error":"not found"}')
        def do_POST(self):
            if self.path != "/step": return self._send(404, '{"error":"not found"}')
            try:
                length = int(self.headers.get("content-length", "0"))
                action = json.loads(self.rfile.read(length) or b"{}")
                with lock:
                    dist.broadcast_object_list([{"type": "step", "action": action}], src=0)
                    video, timing_data = session.step(action)
                    result = {"timing": timing_data, "frames": frames_to_jpegs(video)}
                self._send(200, json.dumps(result, separators=(",", ":")))
            except Exception as exc:
                logging.exception("interactive step failed")
                self._send(500, json.dumps({"error": str(exc)}))
        def log_message(self, fmt, *values):
            logging.info("http " + fmt, *values)

    server = HTTPServer((args.host, args.port), Handler)
    logging.info("Interactive MVP ready at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        dist.broadcast_object_list([{"type": "stop"}], src=0)
        session.close(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
