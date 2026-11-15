"""TileLang GDN chunk state kernel (SM70).

Sequential pass over BT=64 chunks; the state is carried in an fp32 fragment.
v_new = u - w @ h^T;  h_t = exp(g_last) * h + v_new_scaled^T @ k.
Import is cheap; the kernel module loads torch and tilelang on first use.
"""

from __future__ import annotations

__all__ = ["get_chunk_h_kernel"]


def get_chunk_h_kernel(*args, **kwargs):
    """Return the compiled chunk_h kernel. See ``_chunk_h.chunk_h_kernel``."""
    from ._chunk_h import chunk_h_kernel

    return chunk_h_kernel(*args, **kwargs)
