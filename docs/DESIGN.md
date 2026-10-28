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

## Where the code came from, and what to port next

A file-level comparison against the upstream TileLang FlashAttention projects
(2026-10-28) settles what is ours, what is theirs, and what a decode library
should be built from.

**Vendoring chain.** `haohervchb/Tilelang-FA-V100` (upstream, `c6332eb`) →
vendored by GooseLLM as the submodule `3rdparty/tilelang-fa-v100` pinned at
`7a9e958` → vendored into `haohervchb/sglang-V100` (`46daf6279`, three files,
"adapted for tilelang 0.1.8") → inherited by `dg1kjd/sglang-sxm2@73abfe1`.

**What is ours, with no upstream counterpart:**

| file | what |
|---|---|
| `_decode_cuda.py` + `csrc/sm70_longctx_decode.cu` | the CUDA split-KV decode partial (this package) |
| `_kernels_paged_decode.py` | TileLang grouped decode, q=1, partial + combine |
| `_kernels_dense_d256*.py` | gathered dense-KV prefill, 256-dim, incl. the split-KV tail |
| `_kernels_fp8_bridge.py` | fp8 gather/bridge |

**What upstream has that we do not:**

- dense **forward with autotuning** and the whole **training path** (dQ/dKV,
  `torch.autograd.Function`) — irrelevant for inference, useful as a reference
- **three decode variants worth benchmarking**:
  - `_kernels_paged.py::_decode_kernel_func` — single CTA per (head, batch),
    serial over the context. Correct but slow; a good numerical reference.
  - `feat/gemv-decode` — `_decode_gemv_kernel_func`, SIMT FMA, **no tensor
    cores**, `block_N=16`.
  - `feat/decode-hd256-blockN16` — `block_N=16, threads=32` for D=256.
- the **real split-KV decode structure** is in the `rapatel0` fork branch
  `feature/dflash-grouped-verifier` (`_kernels_paged_verify.py`): grid
  `heads_kv * gqa_ctas x max_splits x batch`, partial O + LSE, then a combine
  over splits. That is a decode split-KV library already — specialized to ≤16
  verify rows. Our `_kernels_paged_decode.py` is the same skeleton at q=1.

**So the decode library is a port, not a greenfield design:** start from
`_decode_kernel_func` for the paged loader and online softmax, take the
partial/combine ABI from the grouped-verifier branch, and benchmark the GEMV and
blockN16 tilings as alternatives. What does not exist upstream and we already
have: GQA-grouped CTAs, the split-count policy, the E5M2/E4M3 byte path, and the
combine ABI that matches `sm70_longctx_decode.cu`.

Note that upstream's `block_M=16` decode uses tensor cores on a shape where 15 of
16 rows are padding; the GEMV branch exists precisely because SM70's MMA may lose
to plain FMA at q=1. That question is measurable here in an afternoon, and it is
the first thing to settle before writing more TileLang decode code.
