import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    if left.shape != right.shape:
        print(f"LATENT_COMPARE equal=False left_shape={tuple(left.shape)} right_shape={tuple(right.shape)}")
        raise SystemExit(1)
    delta = (left - right).abs()
    equal = torch.equal(left, right)
    print(
        f"LATENT_COMPARE equal={equal} shape={tuple(left.shape)} "
        f"max_abs_diff={delta.max().item()} nonzero={int(torch.count_nonzero(delta))}",
        flush=True,
    )
    if not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
