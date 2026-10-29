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
the number to explain - and the next section explains it, in a way that makes
both candidates moot: that number is not a memory ceiling at all.

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

## The kernel is compute-blocked, and the GB/s question was malformed (2026-10-28)

Everything above reasons about bandwidth. One measurement replaces that with
something simpler: delete the compute phase and time what is left.

`decode_tile_compute` is removed from `decode_partial_kernel` (the serial and
occupancy variants) and replaced with a two-`LDS` sink so ptxas cannot delete the
loads it just staged. Everything else - the loads, the E5M2 conversion, the smem
stores, both barriers per tile - stays. At TP2/256k, splits 80, lambda12:

| | full kernel | compute removed | exposed compute |
|---|---:|---:|---:|
| e5m2, boost 1530 MHz | 0.620 ms (431 GB/s) | **0.333 ms (805 GB/s)** | 0.287 ms, 46% |
| fp16, boost 1530 MHz | 0.944 ms (568 GB/s) | **0.638 ms (842 GB/s)** | 0.306 ms, 32% |
| e5m2, 1300 MHz | 0.723 ms | 0.333 ms | 0.390 ms, 54% |
| fp16, 1300 MHz | 0.974 ms | 0.638 ms | 0.336 ms, 35% |

The two clock rows are not padding, they are the check. The compute-free arm is
DRAM-bound, so its time is set by the bytes and does not move with the SM clock;
the full arm's does, and by very different amounts - e5m2 0.723 -> 0.620 (17%)
against fp16 0.974 -> 0.944 (3%). That split is exactly the compute-bound against
memory-bound signature, measured without any counters.

Two things fall out of the table.

**The memory path was never the problem.** The same access pattern, grid and
byte path reaches 805 GB/s the moment the compute stops blocking it - above the
776 GB/s a copy benchmark manages - and the assembled kernel collects 431. The
kernel is at 54% of the bandwidth its own loads can sustain, and the missing 46%
is the compute phase holding the next tile's load behind a barrier. Nothing in
the memory path (batching, pipelines, occupancy, more splits) can recover time
that is being lost in the compute warps.

**The two dtypes expose the same compute.** 0.287 and 0.306 ms, from kernels
whose memory halves differ by 1.9x. That is what a per-tile fixed cost looks
like, and it means the "e5m2 at 431 against fp16 at 658 GB/s" comparison that
this document has been chasing since it was written was measuring the wrong
thing: on the same shape fp16 reports 568 GB/s only because it has twice the
bytes to hide the same compute behind. Per token, which is what a decode step
pays for, e5m2 is 1.52x faster - 0.620 against 0.944 ms for the same 262144
tokens.

So there is no e5m2 bandwidth problem to solve. The next question is the compute
phase, and the first fact about it is that it runs on six of the sixteen warps:
`is_compute_warp = warp < kGroup` gives one warp per query row and leaves ten
warps idle at the barrier while they wait. PV splits across more warps with no
reduction at all (lane `l` owns dims `[8l, 8l+8)`, so two warps per row can own
half each); QK needs the 256-dim dot split with a cross-warp add instead. That
is the next thing to try, and it is worth up to the 46% above.

### Splitting PV across two warps buys 2%, not 23%

`decode_partial_kernel_pv2` does the PV half of that: warp pair `(2r, 2r+1)`
owns row `r`, the even warp keeps QK, softmax and the m/l state, and both warps
run PV over 128 dims each (lane `l` owning `[half*128 + 4l, +4)`). Disjoint dims,
so no reduction and no change to any lane's accumulation order - the variant is
**bitwise identical to serial**, which `benchmarks/attn_variants.py` checks.

| 256k, splits 80, lambda12 | two CTAs (shipped) | pv2 | |
|---|---:|---:|---|
| e5m2, TP2 | 0.624 ms | **0.610** | -2.2% |
| e4m3, TP2 | 0.633 ms | **0.619** | -2.2% |
| e5m2, TP1 | 1.232 ms | **1.201** | -2.5% |
| fp16, TP2 | 0.952 ms | 0.970 | **+1.9%** |

Doubling the compute warps for PV is worth 2%, so PV is a small part of the
0.287 ms of exposed compute: if the compute phase were PV-bound, halving the PV
chain would have shown up as roughly five times this. The cost side is visible
too - pv2 pays a third barrier per tile and drops the PV loads from `LDS.128` to
`LDS.64` (4 dims per lane instead of 8) - and fp16, whose compute is only 32% of
its runtime and which is memory-bound, comes out **slower**. That is the control
which says the win is real but bounded.

So PV is not *instruction*-bound. What used to follow here - "the exposed compute
is not in PV" - was the wrong inference from those 2%, and the next section
measures the two terms directly and finds PV is half of it. QK is the one that
can be spread over warps without changing shape: two warps over 128 dims each,
one barrier, then an fp32 add of the two halves. That does change the summation
order - each half accumulates 128 dims in fp16 and the halves are added in fp32 -
so it stops being bitwise identical and has to be checked against the fp32
reference in `benchmarks/partial_check.py` instead.

### Correction: that table says PV is not instruction-bound, not that it is small

The paragraph above reads the 2% as evidence that PV is a small part of the
compute. That does not follow, and the next measurement says so.

pv2 halves PV's *instruction count* - two warps, 4 dims a lane instead of 8 - but
it does not shorten PV's dependency chain, which is 32 tokens long either way.
A latency-bound loop does not care how many lanes feed it. So 2% says PV is not
instruction-bound. Removing PV outright answers the other question, by taking
`for (n = 0; n < kBlockN; ++n)` out of `decode_tile_compute` with a per-tile sink
so `vs` cannot go dead (`--arm nopv`), and doing the same to the QK dot
(`--arm noqk`):

| arm at e5m2/TP2/256k | ms | delta |
|---|---:|---:|
| full | 0.623 | |
| no QK | 0.488 | -0.135, 21.7% |
| no PV | 0.490 | -0.133, 21.3% |
| neither | 0.333 | -0.290, 46.5% |

**QK and PV cost the same**, 0.135 and 0.133 ms, about a fifth of the kernel
each. Their sum is 8% short of removing both (0.268 against 0.290); the
difference is interaction, mostly the second barrier that each single-term arm
still keeps. So the exposed compute is half QK and half PV, both latency-bound on
a 32-long chain, and both worth about 11% of the kernel if that chain can be
halved.

The chain is what to attack, and neither split tried so far shortens it: pv2
halved the lanes' work, not the 32 tokens. Two ways that do shorten it, both
changing the summation order and both therefore needing `benchmarks/partial_check.py`
rather than the bitwise check:

- **PV over the token axis**: two warps, 16 tokens each, one fp32 add of the two
  partial accumulators per dim. Exactly halves the chain.
- **QK over the dim axis**: two warps, 128 dims each, one fp32 add of the two
  halves. Halves the chain and changes the fp16 accumulation grouping too.

### All three of those levers are dead. It is instruction issue.

Both splits above were built in scratch trees before being written up, and
neither survived contact. Three probes, all on `decode_partial_kernel` at
e5m2/TP2/256k, splits 80, warm clock:

| probe | what changes | result |
|---|---|---:|
| pv2, already shipped | half the lanes' work, twice the warps | -2.2% |
| two accumulator sets, even and odd tokens | chain 32 -> 16, same instructions | **0.0%** (0.622 vs 0.623) |
| reuse the previous V vector on odd tokens | half the PV `LDS.128`, same FLOPs | -4.5% (0.597 vs 0.625) |

The middle row is the one that matters: **PV is not latency-bound**. Halving its
dependency chain does nothing at all. The first row says spreading the same work
over twice the warps does nothing either, and the third says halving the shared
memory traffic buys 4.5% where pure bandwidth would have bought about 10%. So it
is neither chain, nor warp count, nor shared memory - it is how many instructions
the phase has to issue:

- PV per tile per CTA is 6 rows x 32 tokens x 256 dims = 49,152 multiplies. As
  `HFMA2` that is 24,576 instructions, or 768 warp-instructions. QK is the same
  shape and the same count: 1,536 warp-instructions per tile per CTA.
- Two CTAs per SM make that 3,072 warp-instructions per SM per tile, and an SM
  issues at most 4 warp-instructions per cycle. **Floor: 768 cycles.**
- Measured compute is 0.287 ms over 103 tiles, 2.79 us per tile. At 1530 MHz
  that is 4,264 cycles per tile per CTA, so 2,132 per tile on the SM - **2.8x the
  FMA-only floor**, and the loads, conversions and loop control make up most of
  the difference.

2.8x sounds like headroom, but it is headroom in instructions already being
issued, not in stalls that can be filled. Scheduling work cannot beat an issue
floor; only fewer instructions can, and no rearrangement of warps, chains or
loads changes the number of `HFMA2` a tile needs. That is the argument for
tensor cores, and it is now a measured one rather than an intuition: at these
shapes PV is `32 tokens x 256 dims` per row and QK is `q . k` over 256 dims,
which is what `mma.sync.m16n8k16` exists for.

The alternative to tensor cores is to move the work off the compute warps
entirely - the 10 idle warps are idle for a reason (they wait at a barrier while
6 compute), but a variant where *all 16* warps compute would only help if issue
slots were the limit rather than instructions, and the pv2 row above says they
are not.

### Two ways to mis-measure this, both hit on the way

**Instrumentation is not free.** Bracketing the phases with `clock64` and
accumulating per thread pushed the stack frame from 240 to 288 bytes - the
kernel stays at 64 registers under `__launch_bounds__(kThreads, 2)`, so the
counters spill - and the kernel read 0.836 ms against 0.620. Moving the
accumulators to shared memory and reading the clock on one thread only brought
the frame to 272 and the time to 0.790. Any phase split taken from counters on
this kernel is measured through that perturbation; the differential above needs
no counters and is why it is the one quoted.

**And the clock is worse than it looks.** A memory-heavy kernel does not hold the
V100's boost: whichever block is timed first in a fresh process sits at about
1300 MHz, and it stays there even after a matmul ramp reads 1522 MHz a second
earlier - the clock falls as soon as the decode kernel starts. That is why the
table above carries two clock rows rather than one nominal 1530. It also means
`torch.cuda.clock_rate()` sampled between blocks is not evidence about the block
that just ran, and that any figure quoted for this kernel without the clock it
was measured at is worth about 17%.

The useful consequence is that the clock makes a free second experiment: a
compute-bound kernel slows by the clock ratio (e5m2: 17%) and a memory-bound one
barely moves (fp16: 3%), so one measurement at each clock separates the two terms
without instrumenting anything.

## Where tensor cores would go

What is left is 0.79 ms of memory and ~0.8 ms of math that half2 cannot reach.
QK for 12-24 q heads against 32 keys of dim 256 is a real GEMM; PV likewise. A
TileLang `T.gemm` formulation is the natural next step, and the reason this repo
exists separately: kernel experiments should not need an engine checkout.

The open question is whether SM70's wmma path helps at q=1 shapes, or whether
the arithmetic intensity is simply too low. That is measurable in one afternoon
against `benchmarks/attn_dense.py`, which is why the harness ships here.

Note, after the sections above: the numbers in the first paragraph of this
section are the pre-HFMA2 decomposition and the "0.79 ms of memory" in it is the
old serial kernel's, not the current memory floor. What has changed is that this
section's question is no longer a hunch. The compute phase is 46% of the kernel,
it is instruction-issue-bound rather than latency- or bandwidth-bound, and three
attacks on it have now measured between -4.5% and 0.0% - with the best of the
three being the one that deleted instructions rather than rearranging them.
Tensor cores are the only remaining lever that changes the instruction count per
result, and PV's `32 tokens x 256 dims` and QK's 256-dim dot are exactly the
shapes `mma.sync.m16n8k16` covers.

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
