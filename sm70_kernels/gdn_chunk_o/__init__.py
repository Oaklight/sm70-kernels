"""TileLang GDN chunk output kernel (SM70).

o = (q @ h^T) * exp(g) + (A * exp(g_i - g_j)) @ v, causal over BT=64 chunks.
Import is cheap; the kernel module loads torch and tilelang on first use.
"""

from __future__ import annotations

__all__ = ["get_chunk_o_kernel"]


def get_chunk_o_kernel(*args, **kwargs):
    """Return the compiled chunk_o kernel. See ``_chunk_o.chunk_o_kernel``."""
    from ._chunk_o import chunk_o_kernel

    return chunk_o_kernel(*args, **kwargs)
