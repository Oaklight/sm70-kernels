"""Paged extend-attention tile comparison on SM70.

Two configurations of the same TileLang kernel:

  shared_q_64x32  the old shipped tile: Q occupies 32 KiB of shared memory, so
                  block_N has to stay at 32.
  qfrag_64x64     Q lives in a fragment, so block_N=64 fits (72 KiB) and the
                  KV loop runs half as many iterations.

Times both on the 27B per-rank shape and checks the outputs against the old
tile. Run inside the CUDA image on a V100:

    python benchmarks/attn_prefill.py --prefixes 8192,32768,98304
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from sm70_kernels.tilelang_fa._kernels_paged import _paged_kernel_func

VARIANTS = {
    "shared_q_64x32": dict(block_M=64, block_N=32, threads=256, num_stages=0,
                           num_splits=1, q_in_regs=False),
    "qfrag_64x64": dict(block_M=64, block_N=64, threads=256, num_stages=0,
                        num_splits=1, q_in_regs=True),
}


def build(prefix, extend, heads, heads_kv, dim, page, device):
    total = prefix + extend
    blocks = (total + page - 1) // page
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(blocks, page, heads_kv, dim, dtype=torch.float16,
                    device=device, generator=g)
    v = torch.randn(blocks, page, heads_kv, dim, dtype=torch.float16,
                    device=device, generator=g)
    q = torch.randn(extend, heads, dim, dtype=torch.float16,
                    device=device, generator=g)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).unsqueeze(0)
    seq_lens = torch.tensor([total], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, extend], dtype=torch.int32, device=device)
    prefix_kv_lens = torch.tensor([prefix], dtype=torch.int32, device=device)
    return q, k, v, block_table, seq_lens, query_start_loc, prefix_kv_lens


def compile_kernel(q, k, block_table, cfg):
    nt = q.shape[0]
    return _paged_kernel_func(
        batch=1,
        heads=q.shape[1],
        heads_kv=k.shape[2],
        dim=q.shape[2],
        page_block_size=k.shape[1],
        max_blocks_per_seq=block_table.shape[1],
        num_pages=k.shape[0],
        is_causal=True,
        sliding_window_size=-1,
        **cfg,
    )


def time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = []
    for _ in range(3):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best.append(start.elapsed_time(end) / iters)
    return min(best), statistics.median(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="8192,32768,98304")
    ap.add_argument("--extend", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--heads-kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--page", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--variants", default="shared_q_64x32,qfrag_64x64")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    device = "cuda"
    names = args.variants.split(",")
    print(f"gpu: {torch.cuda.get_device_name(0)} | extend {args.extend} rows, "
          f"heads {args.heads}/{args.heads_kv}, dim {args.dim}, page {args.page}")
    print()
    print(f"{'prefix':>8} " + " ".join(f"{n:>28}" for n in names))
    scale = 1.0 / math.sqrt(args.dim)

    for prefix in (int(x) for x in args.prefixes.split(",")):
        q, k, v, bt, sl, qsl, pk = build(
            prefix, args.extend, args.heads, args.heads_kv, args.dim, args.page, device)
        row = []
        ref = None
        for name in names:
            kern = compile_kernel(q, k, bt, VARIANTS[name])
            call = lambda kern=kern: kern(q, k, v, bt, sl, qsl, pk, args.extend, scale)
            out = call()[: args.extend]
            torch.cuda.synchronize()
            if ref is None:
                ref = out.float()
                note = ""
            else:
                diff = (out.float() - ref).abs()
                note = f" (maxdiff {diff.max().item():.2e})"
            best, _ = time_call(call, args.warmup, args.iters)
            row.append(f"{best:10.3f} ms{note}")
        print(f"{prefix:8d} " + " ".join(f"{c:>28}" for c in row))


if __name__ == "__main__":
    main()
