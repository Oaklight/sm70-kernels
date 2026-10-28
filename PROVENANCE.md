# Provenance

| Component | Source | Commit | License |
|---|---|---|---|
| `sm70_kernels.longctx_decode` | `dg1kjd/sglang-sxm2` (fork of `haohervchb/sglang-V100`) | `73abfe1dd7e7aae7a62b222a201e48f532fb298e` | Apache-2.0 |
| `sm70_kernels.tilelang_fa` (planned) | [GooseLLM](https://github.com/haohervchb/GooseLLM) `3rdparty/tilelang-fa-v100` | to be pinned | Apache-2.0 |

Local changes to `longctx_decode` relative to the fork:

- E5M2 byte KV path (shift-by-8 exact conversion) and the fp16 path
- two uninitialised reads fixed in the split combine (`probs[]` lanes past
  `tile_tokens`, empty trailing splits returning before writing `o`/`lse`)
- HFMA2 QK accumulation
- per-tile half2 PV accumulation folded into fp32 once per tile
