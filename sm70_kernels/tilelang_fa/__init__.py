"""TileLang FlashAttention kernels for SM70.

The kernel module imports ``torch`` and ``tilelang``, so it is loaded on first
use. Importing this package costs nothing and does not need a GPU.
"""

from __future__ import annotations

__all__ = ["get_paged_kernel"]


def get_paged_kernel(*args, **kwargs):
    """Return the compiled paged prefill/extend kernel. See ``_kernels_paged``."""
    from ._kernels_paged import get_paged_kernel as _impl

    return _impl(*args, **kwargs)
