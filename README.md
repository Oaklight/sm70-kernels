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
- `tilelang_fa` — not vendored here yet; it currently lives in
  `sglang-restomod`'s `attention/v100` component. Moving it here is tracked as
  the next step.
