"""GDN chunk output kernel: TileLang vs the Triton reference.

o = (q @ h^T) * exp(g) + (A * exp(g_i - g_j)) @ v, causal over BT=64 chunks.
Shapes are per TP rank: B=1, T=4096, Hg=4 key heads, H=12 value heads,
K=V=128. The reference is the SGLang FLA `chunk_fwd_o`; when it cannot be
imported or launched, an fp32 torch implementation of the same formula is
used instead, including the A->fp16 cast the Triton kernel performs.

Measured on a V100-SXM2: Triton 24.94 ms, TileLang 0.27 ms. The Triton
kernel spills 5258 registers per thread and cannot launch on a card with
under ~2 GB free.

    python benchmarks/gdn_chunk_o.py --tokens 4096
    python benchmarks/gdn_chunk_o.py --tokens 4096 --triton   # needs a free card

SGLANG_SRC points at the SGLang python tree to import the Triton reference.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

SGLANG_SRC = os.environ.get("SGLANG_SRC", "/workspace/restomod/repo/.build/sglang/python")
if os.path.isdir(SGLANG_SRC):
    sys.path.insert(0, SGLANG_SRC)

from sm70_kernels.gdn_chunk_o import get_chunk_o_kernel


def reference_fp32(q, k, v, h, g, BT):
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    NT = T // BT
    qc = q[0].float().view(NT, BT, Hg, K)
    kc = k[0].float().view(NT, BT, Hg, K)
    vc = v[0].float().view(NT, BT, H, V)
    gc = g[0].view(NT, BT, H)
    out = torch.empty(NT, BT, H, V, device=q.device, dtype=torch.float32)
    idx = torch.arange(BT, device=q.device)
    tril = (idx[:, None] >= idx[None, :]).float()
    groups = H // Hg
    for hh in range(H):
        kh = hh // groups
        bo = torch.einsum("tik,tvk->tiv", qc[:, :, kh], h[:, hh].float())
        A = torch.einsum("tik,tjk->tij", qc[:, :, kh], kc[:, :, kh])
        bo = bo * torch.exp(gc[:, :, hh])[..., None]
        d = gc[:, :, hh][:, :, None] - gc[:, :, hh][:, None, :]
        A = (A * torch.exp(torch.clamp(d, max=0.0)) * tril).half().float()
        out[:, :, hh] = bo + torch.einsum("tij,tjv->tiv", A, vc[:, :, hh])
    return out.view(B, T, H, V) * (K ** -0.5)


def report(name, got, ref):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    big = ref.abs() >= 1e-2
    max_abs = err.max().item()
    max_rel = (err[big] / ref.abs()[big]).max().item() if big.any() else 0.0
    small = err[~big].max().item() if (~big).any() else 0.0
    ulp = ref.abs().max().item() * 2 ** -11
    print(
        "%-8s max_abs %.4e  max_rel %.4e  small_band_abs %.4e  %.2f ULP"
        % (name, max_abs, max_rel, small, max_abs / ulp if ulp else 0.0)
    )


def bench(fn, iters=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--triton", action="store_true", help="also run the Triton reference")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    B, T, H, Hg, K, V, BT, BV = 1, args.tokens, 12, 4, 128, 128, 64, 64
    NT = T // BT
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, T, Hg, K, device=dev, dtype=torch.float16)
    k = torch.randn(B, T, Hg, K, device=dev, dtype=torch.float16)
    v = torch.randn(B, T, H, V, device=dev, dtype=torch.float16)
    h = torch.randn(B * NT, H, V, K, device=dev, dtype=torch.float16)
    g = torch.cumsum(-torch.rand(B, T, H, device=dev, dtype=torch.float32) * 0.05, dim=1).contiguous()

    kernel = get_chunk_o_kernel(B, T, H, Hg, K, V, BT, BV, 128)
    got = kernel(q, k, v, h, g)
    torch.cuda.synchronize()
    report("fp32ref", got, reference_fp32(q, k, v, h, g, BT))

    flop = 2 * NT * H * (2 * BT * K * BV + BT * BT * BV)
    t = bench(lambda: kernel(q, k, v, h, g), args.iters)
    print("tilelang %.3f ms  %.1f GF/s  (%.0f tokens)" % (t, flop / t / 1e9, T))

    if args.triton:
        from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o

        ref = chunk_fwd_o(q, k, v, h, g, scale=K ** -0.5)
        report("triton", got, ref)
        t_ref = bench(lambda: chunk_fwd_o(q, k, v, h, g, scale=K ** -0.5), args.iters)
        print("triton   %.3f ms  %.1f GF/s  speedup %.2fx" % (t_ref, flop / t_ref / 1e9, t_ref / t))


if __name__ == "__main__":
    main()
