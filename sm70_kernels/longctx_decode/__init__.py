"""Split-KV decode attention for long context (V100 / SM70).

One call returns 4 + lse per (batch, split, head); a second pass combines the
splits. The caller owns the page table and the KV cache layout; see
``sm70_cuda_decode_partial`` for the exact shapes.

Kernel source: ``csrc/sm70_longctx_decode.cu``, JIT-built with torch's
``load_inline`` into a build directory that can be shared across processes.
"""

from ._loader import sm70_cuda_decode_partial, sm70_cuda_decode_available

__all__ = ["sm70_cuda_decode_partial", "sm70_cuda_decode_available"]
