# Provenance

| Component | Source | Commit | License |
|---|---|---|---|
| `sm70_kernels.longctx_decode` | `dg1kjd/sglang-sxm2` (fork of `haohervchb/sglang-V100`) | `73abfe1dd7e7aae7a62b222a201e48f532fb298e` | Apache-2.0 |
| `sm70_kernels.tilelang_fa` | `dg1kjd/sglang-sxm2` @ `73abfe1dd7e7aae7a62b222a201e48f532fb298e` (`tilelang_fa_v100/_kernels_paged.py`), itself vendored from [GooseLLM](https://github.com/haohervchb/GooseLLM) `3rdparty/tilelang-fa-v100` | `73abfe1` | Apache-2.0 |

Local changes to `longctx_decode` relative to the fork:

- E5M2 byte KV path (shift-by-8 exact conversion) and the fp16 path
- two uninitialised reads fixed in the split combine (`probs[]` lanes past
  `tile_tokens`, empty trailing splits returning before writing `o`/`lse`)
- HFMA2 QK accumulation
- per-tile half2 PV accumulation folded into fp32 once per tile

Local changes to `tilelang_fa` relative to the fork:

- the per-iteration `T.clear` of the K/V shared tiles is hoisted out of the
  loop, the PV gemm reads `P_shared` directly, and softmax uses `exp2`
  (`Oaklight/lambda-v100-lab`, paged-extend inner-loop work);
- the dim-256 tile holds Q in a fragment, so `block_N=64` fits and the KV loop
  runs half as many iterations (`feat/prefill-qfrag`). Measured 0.78-0.82 of
  the shared-Q tile's time at 8k/32k/98k on one V100.
