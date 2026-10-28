"""Kernels for NVIDIA V100 (SM70) inference.

Two families live here:

* ``longctx_decode`` — the split-KV decode-attention kernel for long context,
  hand-written CUDA, with an E5M2 byte-KV path and per-instruction-tuned QK/PV
  math.
* ``tilelang_fa`` — the TileLang FlashAttention kernels for SM70 (paged prefill,
  the fp8 gather, and grouped decode), vendored from GooseLLM.

The package imports without CUDA present; every kernel loads lazily on first use
and raises a clear error if the device or the toolchain does not fit.
"""

from . import longctx_decode

__all__ = ["longctx_decode"]
__version__ = "0.1.0"
