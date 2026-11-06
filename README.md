# sm70-kernels

GPU kernels for NVIDIA **V100 (SM70)** inference, kept separate from any
inference engine so they can be measured, versioned and vendored without
dragging a fork along.

Two families:

| package | what | source |
|---|---|---|
| `sm70_kernels.longctx_decode` | split-KV decode attention for long context: E5M2 byte KV or fp16, per-split partial + LSE, HFMA2 QK / per-tile half2 PV | ours, evolved from `dg1kjd/sglang-sxm2` |
| `sm70_kernels.tilelang_fa` | TileLang FlashAttention for SM70: paged prefill, fp8 gather, grouped decode | vendored from [GooseLLM](https://github.com/haohervchb/GooseLLM) `3rdparty/tilelang-fa-v100` |

## Why a separate repo

The kernels are the part that has to be tuned against hardware and measured to
the microsecond; the engine integration is the part that has to track an
upstream SGLang pin. Mixing them means every kernel experiment drags a fork
along, and every upstream bump breaks the measurement harness.

- `sm70-kernels` — kernels, loaders, microbenchmarks, provenance.
- `Oaklight/sglang-restomod` — SGLang integration (attention backends,
  registration), pins this package.
- `Oaklight/lambda-v100-lab` — end-to-end benchmarks, receipts, machine notes.

## Usage

```python
from sm70_kernels.longctx_decode import sm70_cuda_decode_partial

partial_o, partial_lse = sm70_cuda_decode_partial(
    q, k_cache, v_cache, page_table, seq_lens,
    max_splits=80, min_tokens_per_split=32, softmax_scale=1 / 16,
)
```

The extension JIT-builds on first use into `$SM70_KERNELS_BUILD_DIR`
(default `/tmp/sm70_kernels`) and is reused while newer than the source.

## Measured

256k context, 1 batch, V100-SXM2, e5m2 KV, page 16 (TP1 shape, 4 kv heads):

| kernel | ms/layer | GB/s |
|---|---:|---:|
| baseline (scalar FP32 QK/PV) | 2.688 | 200 |
| + HFMA2 QK | 2.219 | 242 |
| + per-tile half2 PV | **1.984** | 271 |

At 128k in the served TP2 shape the kernel is 0.498 ms/layer, i.e. 8.0 ms per
token of 16 layers, against a 36 ms/token decode step. `benchmarks/` reproduces
all of it in seconds.

## Status

- `longctx_decode` — working, measured, and shipped in the fork we run.
- `tilelang_fa` — the paged prefill/extend kernel lives here as the source of
  truth (`tilelang_fa/_kernels_paged.py`); `sglang-restomod` syncs it through
  its `UPSTREAM_PIN.toml` submodule. The fp8 gather and grouped decode modules
  still live only in restomod.

## Paged extend: Q in registers, block_N=64

The dim-256 prefill tile was pinned by shared memory: with a 32 KiB Q tile in
shared, `block_N=64` needs 108 KiB against the 96 KiB cap. Holding Q in a
fragment instead lets `block_N=64` fit at 72 KiB (K 32 + V 32 + P 8), which
halves the KV-loop iteration count. Same inputs, one V100, 4096 extend rows,
12 heads / 2 KV heads, page 16 (`benchmarks/attn_prefill.py`):

| prefix | shared Q 64x32 | Q fragment 64x64 | ratio |
|---:|---:|---:|---:|
| 8,192 | 16.09 ms | **13.17 ms** | 0.82 |
| 32,768 | 53.15 ms | **41.35 ms** | 0.78 |
| 98,304 | 155.30 ms | **122.35 ms** | 0.79 |

Max absolute difference against the shared-Q tile is 6.1e-05 on an output of
RMS 0.05. `num_stages=1` and the head-dim-split variants were measured and are
slower; see the lab note.
