# sm70-kernels

GPU kernels for NVIDIA **V100 (SM70)** inference, kept separate from any
inference engine so they can be measured, versioned and vendored without
dragging a fork along.

Two families:

| package | what | source |
|---|---|---|
| `sm70_kernels.longctx_decode` | split-KV decode attention for long context: E5M2 byte KV or fp16, per-split partial + LSE, HFMA2 QK / per-tile half2 PV | ours, evolved from `dg1kjd/sglang-sxm2` |
| `sm70_kernels.tilelang_fa` | TileLang FlashAttention for SM70: paged prefill, fp8 gather, grouped decode | vendored from [GooseLLM](https://github.com/haohervchb/GooseLLM) `3rdparty/tilelang-fa-v100` |
| `sm70_kernels.gdn_chunk_o` | TileLang GDN chunk output `(q h^T + A v)`, causal and gated | ours, new work |
| `sm70_kernels.gdn_chunk_h` | TileLang GDN chunk state pass, state in an fp32 fragment | ours, new work |

## GDN chunked prefill (new)

No linear-attention kernel existed in this repo before these two. They replace
the Triton `chunk_fwd_kernel_o` / `chunk_gated_delta_rule_fwd_kernel_h` pair in
the SGLang GDN prefill. Shapes are per TP rank (B=1, T=4096, H=12, Hg=4,
K=V=128, BT=64).

| kernel | Triton | TileLang | speedup |
|---|---|---|---|
| `gdn_chunk_o` | 25.12 ms | 0.23 ms | 108x |
| `gdn_chunk_h` | 15.4 ms | 0.47 ms | 33x |

Generated CUDA contains `mma_sync_sm70` for both. Accuracy against an fp32
reference: 0.74 ULP for chunk_o (Triton itself is 0.64), 0.75-1.13 ULP for the
chunk_h outputs. Why Triton loses on this target at all: its PTX has zero
`mma.sync` and the kernel spills 5258 registers per thread, so it also cannot
launch alongside a server that leaves under ~2 GB free.

Current restriction: B=1, T a multiple of 64, no varlen. The SGLang dispatch
falls back to Triton for anything else.

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

## License

Apache-2.0. See [LICENSE](LICENSE) for the terms and [NOTICE](NOTICE) for the
upstream attribution: `longctx_decode`'s CUDA source and `tilelang_fa`'s paged
kernel are derivative of `dg1kjd/sglang-sxm2` at `73abfe1`, the latter through
`haohervchb/GooseLLM` `3rdparty/tilelang-fa-v100`.
