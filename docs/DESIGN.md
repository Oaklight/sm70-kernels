# Design notes

## What is worth optimizing (measured, not guessed)

At 256k, one decode-attention layer of the 27B costs 2.688 ms before any of this
work. Cutting it open by shortening one loop at a time:

| term | ms | share |
|---|---:|---:|
| memory + E5M2 convert | 0.79 | 29% |
| QK math | 0.83 | 31% |
| PV math | 1.06 | 40% |

The memory path already runs at 683 GB/s against the 776 GB/s a plain `copy`
gets on the same GPU, so the kernel is **instruction-bound, not bandwidth-bound**.
That ruled out the obvious fixes and they stay ruled out:

- more splits: 2.51 ms at 20, 2.29 at 320, 2.50 at 1280 — plateau
- page-table indirection: paged 2.680 vs dense 2.668 ms
- concurrent kernels: two streams overlap 1.06x, no idle GPU
- wider KV: fp16 moves 2x the bytes in 9% more time

Hence the HFMA2 QK and per-tile half2 PV, both of which keep fp32 accumulation
where it matters and leave greedy output byte-identical.

## Where tensor cores would go

What is left is 0.79 ms of memory and ~0.8 ms of math that half2 cannot reach.
QK for 12-24 q heads against 32 keys of dim 256 is a real GEMM; PV likewise. A
TileLang `T.gemm` formulation is the natural next step, and the reason this repo
exists separately: kernel experiments should not need an engine checkout.

The open question is whether SM70's wmma path helps at q=1 shapes, or whether
the arithmetic intensity is simply too low. That is measurable in one afternoon
against `benchmarks/attn_dense.py`, which is why the harness ships here.
