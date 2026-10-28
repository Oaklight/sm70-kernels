"""How much does the KV cache dtype cost in attention accuracy?

Compares e5m2 and e4m3 (and fp16 as a floor) against an fp32 reference computed
from the *same* quantized values the kernel would read. Two numbers matter: the
error in the attention output, and how often a value saturates the format,
because e4m3 tops out at 448 while e5m2 reaches 57344.

The test is deliberately magnitude-aware: `--k-scale` multiplies the key
magnitudes, which is what RoPE does to a long context and what makes e4m3 a
risk in the first place.
"""

import argparse

import torch

DIM = 256
LOG2E = 1.4426950408889634


def quantize(x, fmt):
    return x.to(fmt).float()


def attention(q, k, v, scale):
    """One query row against the whole prefix, fp32 throughout."""
    logits = (k @ q) * scale
    p = torch.softmax(logits, dim=0)
    return p @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    scale = DIM**-0.5
    formats = {
        "fp16": torch.float16,
        "e5m2": torch.float8_e5m2,
        "e4m3": torch.float8_e4m3fn,
    }
    print(f"device={dev} context={args.context} k_scale={args.k_scale}")
    print(f"{'fmt':5s} {'max|err|':>10s} {'mean|err|':>10s} {'rel rms':>9s} "
          f"{'sat K%':>8s} {'sat V%':>8s}")
    for name, fmt in formats.items():
        worst = mean = rel = sat_k = sat_v = 0.0
        for _ in range(args.reps):
            k = torch.randn(args.context, DIM, device=dev) * args.k_scale
            v = torch.randn(args.context, DIM, device=dev)
            q = torch.randn(DIM, device=dev)
            ref = attention(q, k, v, scale)
            kq = quantize(k, fmt)
            vq = quantize(v, fmt)
            got = attention(q, kq, vq, scale)
            d = (got - ref).abs()
            worst = max(worst, d.max().item())
            mean = max(mean, d.mean().item())
            rel = max(rel, (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item())
            fmax = torch.finfo(fmt).max
            sat_k = max(sat_k, (k.abs() > fmax).float().mean().item() * 100)
            sat_v = max(sat_v, (v.abs() > fmax).float().mean().item() * 100)
        print(f"{name:5s} {worst:10.3e} {mean:10.3e} {rel:9.2%} "
              f"{sat_k:8.2f} {sat_v:8.2f}")


if __name__ == "__main__":
    main()
