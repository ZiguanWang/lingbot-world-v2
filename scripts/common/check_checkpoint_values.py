import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open


def scan_tensor(tensor, chunk_elements):
    flat = tensor.reshape(-1)
    nan_count = 0
    inf_count = 0
    nonzero_count = 0
    abs_max = 0.0
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start:start + chunk_elements]
        if chunk.is_floating_point():
            nan_count += int(torch.isnan(chunk).sum())
            inf_count += int(torch.isinf(chunk).sum())
            finite = chunk[torch.isfinite(chunk)]
        else:
            finite = chunk
        nonzero_count += int(torch.count_nonzero(finite))
        if finite.numel():
            abs_max = max(abs_max, float(finite.abs().max()))
    return nan_count, inf_count, nonzero_count, abs_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--chunk-elements", type=int, default=8_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transformer_dir = args.checkpoint_dir / "transformers"
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    expected = set(index["weight_map"])
    seen = set()
    report = {"files": {}, "summary": {}}
    total_tensors = total_elements = total_nan = total_inf = zero_tensors = 0

    for path in sorted(transformer_dir.glob("*.safetensors")):
        file_report = {"tensors": 0, "elements": 0, "nan": 0, "inf": 0,
                       "zero_tensors": [], "largest_abs": []}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                nan, inf, nonzero, abs_max = scan_tensor(tensor, args.chunk_elements)
                seen.add(key)
                file_report["tensors"] += 1
                file_report["elements"] += tensor.numel()
                file_report["nan"] += nan
                file_report["inf"] += inf
                if nonzero == 0:
                    file_report["zero_tensors"].append(key)
                file_report["largest_abs"].append((abs_max, key, str(tensor.dtype), list(tensor.shape)))
                del tensor
        file_report["largest_abs"] = sorted(file_report["largest_abs"], reverse=True)[:10]
        report["files"][path.name] = file_report
        total_tensors += file_report["tensors"]
        total_elements += file_report["elements"]
        total_nan += file_report["nan"]
        total_inf += file_report["inf"]
        zero_tensors += len(file_report["zero_tensors"])
        print(
            f"CHECK {path.name} tensors={file_report['tensors']} "
            f"elements={file_report['elements']} nan={file_report['nan']} "
            f"inf={file_report['inf']} zero_tensors={len(file_report['zero_tensors'])}",
            flush=True,
        )
        gc.collect()

    report["summary"] = {
        "tensors": total_tensors,
        "elements": total_elements,
        "nan": total_nan,
        "inf": total_inf,
        "zero_tensors": zero_tensors,
        "missing_from_files": sorted(expected - seen),
        "extra_in_files": sorted(seen - expected),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
