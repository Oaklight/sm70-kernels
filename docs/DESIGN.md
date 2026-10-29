# Design notes

## What is worth optimizing (measured, not guessed)

At 256k, one decode-attention layer of the 27B costs 2.688 ms before any of this
work. Cutting it open by shortening one loop at a time:

| term | ms | share |
|---|---:|---:|
| memory + E5M2 convert | 0.79 | 29% |
| QK math | 0.83 | 31% |
| PV math | 1.06 | 40% |

The first reading of that table was "instruction-bound, not bandwidth-bound",
and it was wrong in a way worth keeping on the page. Two experiments settled it.

Cutting the instruction count did nothing. The QK dot went from 128 scalar smem
reads per row-tile to 32 vector reads, and PV from four strided reads per token
to one 16-byte read, by giving lane `l` dims `[8l, 8l+8)` - the layout the loader
already writes. Measured at 256k: 1.917 ms before, 1.917 ms after. Zero.

The **fp16 KV cache moves twice the bytes in the same time**: 1.886 ms against
1.246 ms for e5m2, at identical math. A kernel that does not slow down when you
double its bytes is not bandwidth-bound, and one that does not speed up when you
halve its instructions is not instruction-bound either. What both dtypes share
is the number of *requests*: e5m2 reads 16 bytes per request and fp16 reads 32.

So the kernel is **latency-bound on bytes in flight**. Two changes follow, and
they are where all of the gain came from:

1. **Batch the tile's loads** before consuming any of them, instead of
   load-convert-store per iteration (`pipe_load_tile`): 1.917 -> 1.725 ms.
2. **Two CTAs per SM** (`__launch_bounds__(kThreads, 2)`), so the hardware
   interleaves one CTA's loads with the other's math and holds twice the bytes
   in flight: 1.725 -> 1.246 ms.

The software pipeline in `decode_partial_kernel_pipelined` (ten loader warps
stream tile i+1 while six compute warps run tile i) reaches the same idea by
hand and lands within 12% of the occupancy version. Which one wins depends on
KV dtype, so both are kept:

| at 256k, splits 80, lambda12 | e5m2 kvh=4 | e5m2 kvh=2 | fp16 kvh=4 |
|---|---:|---:|---:|
| before this work | 1.917 | 0.902 | 2.010 |
| serial + batched loads | 1.725 | 0.891 | 1.981 |
| warp-specialized pipeline | 1.414 | 0.719 | **1.631** |
| two CTAs per SM | **1.246** | **0.628** | 1.886 |

e5m2 is what long-context serving uses for capacity, and there occupancy wins by
28%; fp16 is already at 658 GB/s of a 776 GB/s ceiling, so it has nothing left
to win and the pipeline's extra overlap is what it can use.

Earlier dead ends that stay dead, all re-checked against the same harness:

- more splits: 2.51 ms at 20, 2.29 at 320, 2.50 at 1280 — plateau
- page-table indirection: paged 2.680 vs dense 2.668 ms
- concurrent kernels: two streams overlap 1.06x, no idle GPU

The HFMA2 QK and per-tile half2 PV from the first round still stand: they keep
fp32 accumulation where it matters and leave greedy output byte-identical. The
vectorized smem reads were kept because they cost nothing, not because they pay.

All three variants agree with an fp32 per-split reference to 1.6e-3 absolute on
a signal of RMS 0.101 (`benchmarks/partial_check.py`), and serial, occ2 and
pipelined agree with each other bit for bit (`benchmarks/attn_variants.py`).

## Pipeline plus occupancy does not add up (2026-10-28)

The two fixes above attack the same stall from opposite ends: batching keeps more
requests in flight inside a CTA, and a second resident CTA overlaps one CTA's
loads with the other's math. If both were buying bytes in flight, doing both
should beat either. It does not.

`decode_partial_kernel_regpipe` is the pipelined kernel with the prefetched tile
held in **registers** instead of a second smem buffer. That is the only thing
keeping the shipped pipeline at one CTA per SM: the double buffer is
`2 * 2 * kSmemKV = 67.6 KB` of dynamic smem. The register version stages four
vectors per loader thread (16 registers on the e5m2 path, 32 on fp16) and stays
at 37 KB, so two CTAs remain resident. At TP2 the grid is
`kv_heads * splits` = two CTAs per SM, which is exactly the residency slot the
shipped pipeline leaves empty; at TP1 it is four, i.e. two waves of two.

| at 256k, splits 80, lambda12 | TP2 e5m2 | TP2 e4m3 | TP1 e5m2 | TP1 e4m3 |
|---|---:|---:|---:|---:|
| serial | 0.862 | 0.870 | 1.699 | 1.715 |
| **two CTAs per SM** | **0.620** | **0.632** | **1.231** | **1.249** |
| warp-specialized pipeline | 0.712 | 0.721 | 1.407 | 1.419 |
| pipeline + two CTAs (registers) | 0.660 | 0.664 | 1.294 | 1.304 |

Trading the second smem buffer for registers recovers 57% of the pipeline's
deficit to the occupancy variant at TP2 and 64% at TP1, and none of the rest:
0.660 against 0.620 ms. Every cell of the table says the same thing, and all four
variants are bitwise identical to serial (`max|do|` and `max|dlse|` both 0).

So this rules out two candidate explanations at once:

- **Overlap is not additive.** A CTA that overlaps its own loads with its own
  math gives the SM nothing that a second resident CTA was not already giving it.
  The two mechanisms are substitutes, not terms in a sum.
- **Per-thread request depth is not the constraint.** An e5m2 loader thread in
  regpipe issues four requests (`kPipeLoadThreads` 320 against
  `kPipeVecPerTile` 1024) where the serial and occupancy kernels issue two -
  which is the depth fp16 reaches by accident, and was the best remaining
  explanation for fp16 sitting at 569 GB/s where e5m2 sat at 431. Same total
  bytes, twice the depth, no movement. What is left is bytes per request.

Register pressure is not the explanation either: `__launch_bounds__(kThreads, 2)`
caps regpipe at 64 registers, and the shipped occupancy kernel already runs at
exactly 64 (`cuobjdump -res-usage`), so the cap takes nothing from the compute
path that it was not already giving up.

The open question is narrower than it was: an e5m2 kernel at 431 GB/s of a
776 GB/s copy ceiling is not short of concurrency, so the remaining gap is in the
request itself - 16 bytes against fp16's 32 - or in the access pattern. The
683 GB/s that the loop-cut measurement above reaches on the serial structure is
the number to explain.

### The harness was charging the clock ramp

Every number in this section comes from `benchmarks/attn_variants.py`, which used
to time each variant straight after 40 warmup launches - about 40 ms of load. A
V100 idles at 1290 MHz and only reaches 1530 under sustained load, so the harness
under-reported by 7% and charged all of it to whichever variant ran first.
`--clock-warmup` (2 s by default) now spins the clock before anything is timed,
and `--passes` reports the best of several timed passes. Serial at TP2 moves from
1.002 to 0.862 ms with no change to the kernel, while the occupancy number, which
had been measured warm, does not move. A `--dtype` list that did not end in fp16
also inflated the reported bandwidth of the earlier dtypes, because the `fp16`
doubling mutated a shared `moved`; that is fixed too.

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

## KV dtype: e5m2 was chosen for the kernel, not for accuracy

The decode partial converts E5M2 to FP16 with a byte shift, which is exact
because E5M2 and FP16 share a sign bit, a five-bit exponent and the same bias.
E4M3 has a four-bit exponent with bias 7 and a three-bit mantissa, so it needs
real conversion and the kernel used to refuse it. That refusal is why every
measurement here is E5M2, and the reason was speed, not quality.

Measured, one 32k prefix against an fp32 reference, same quantized bytes on
both sides:

| fmt | rel rms | max abs err | saturated |
|---|---:|---:|---:|
| fp16 | 0.03% | 9.1e-06 | 0% |
| **e4m3** | **4.05%** | 1.2e-03 | 0% |
| e5m2 | 7.68% | 2.1e-03 | 0% |

and at 8x key magnitude, which is what a long context does to K:

| fmt | rel rms | saturated |
|---|---:|---:|
| **e4m3** | **12.2%** | 0% |
| e5m2 | 41.6% | 0% |

Neither format saturates, so the range argument for E5M2 (448 against 57,344)
does not bite on this data. **E4M3 is about twice as accurate at identical
capacity.**

`e4m3_to_fp16_16` now exists and is selected by passing the cache with its own
dtype rather than viewed as bytes. The conversion is exact - the kernel's
residual error against the reference is 0.00116 for E4M3 against 0.00115 for
FP16 - so E4M3 costs nothing in conversion fidelity.

### It cost time until the conversion was fixed

The first version of `e4m3_to_fp16_16` was branchy per-byte bit manipulation and
cost 41% more than E5M2. That was a property of the code, not of the format:

| at 256k, splits 80, TP2 | branchy | PRMT (shipped) |
|---|---:|---:|
| e5m2 KV | 0.727 ms | 0.629 ms |
| e4m3 KV | 1.014 ms | **0.635 ms** |

E4M3 KV now costs the same as E5M2 to convert - within 1%, which is noise - and
it is twice as accurate. The rewrite is 32 SASS instructions per 16-byte vector
against 480 for the branchy form and 752 for `__nv_cvt_fp8x2_to_halfraw2`,
which is emulated on SM70 and was measured slower than both.

Two things make it that cheap:

- The bias correction is a **multiply by 256, not an add**. Assembling
  `s<<15 | e<<10 | m<<7` is right for normals, and for `e == 0` it yields the
  FP16 subnormal `m<<7 = m*2^-17` where E4M3 means `m*2^-9`; one `HMUL2` fixes
  both cases and covers two values per instruction. Adding `0x2000` per half is
  wrong for all 18 subnormal encodings.
- E5M2 needs no arithmetic at all: it is the high byte of each half, so two
  `PRMT`s convert four values.

The only E4M3 inputs not reproduced are `0x7F` and `0xFF`, which come out as
+-480 rather than NaN. A saturating quantizer cannot produce them.

### So the KV dtype is a free choice

Equal speed, twice the accuracy, same capacity: **E4M3 is the better KV format
on this hardware**, and the reason every earlier measurement used E5M2 was that
the kernel refused E4M3 rather than that E4M3 was worse. What it still cannot
settle is whether the accuracy matters: that needs a perplexity or task metric,
not an attention RMS.
