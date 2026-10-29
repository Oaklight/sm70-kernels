"""Build/load the SM70 long-context decode extension."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SRC_PATH = Path(__file__).resolve().parent / "csrc" / "sm70_longctx_decode.cu"
_BUILD_DIR = Path(os.environ.get("SM70_KERNELS_BUILD_DIR", "/tmp/sm70_kernels"))
_EXT = None
_ATTEMPTED = False


def sm70_cuda_decode_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() == (7, 0)
    except Exception:
        return False


def _load_built(build_directory: Path):
    import importlib.util as _ilu

    so = build_directory / "sm70_longctx_decode_v100.so"
    if not so.is_file() or so.stat().st_mtime < _SRC_PATH.stat().st_mtime:
        return None
    try:
        spec = _ilu.spec_from_file_location("sm70_longctx_decode_v100", so)
        module = _ilu.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module if hasattr(module, "sm70_longctx_decode") else None
    except Exception:
        logger.warning("could not load %s directly; rebuilding", so, exc_info=True)
        return None


def _load_ops():
    global _EXT, _ATTEMPTED
    if _EXT is not None:
        return _EXT
    if _ATTEMPTED:
        return None
    _ATTEMPTED = True
    if not _SRC_PATH.is_file():
        logger.warning("kernel source missing: %s", _SRC_PATH)
        return None
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    built = _load_built(_BUILD_DIR)
    if built is not None:
        _EXT = built
        return _EXT
    from torch.utils.cpp_extension import load_inline

    _EXT = load_inline(
        name="sm70_longctx_decode_v100",
        cpp_sources="",
        cuda_sources=_SRC_PATH.read_text(),
        functions=None,
        is_python_module=True,
        verbose=False,
        build_directory=str(_BUILD_DIR),
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
    )
    return _EXT


def sm70_cuda_decode_partial(
    q, k_cache, v_cache, page_table, seq_lens,
    max_splits, min_tokens_per_split, softmax_scale, k_scale=1.0, v_scale=1.0,
    variant="serial",
):
    """Run the CUDA split-KV partial, returning (partial_o, partial_lse).

    q:          [batch, heads, 256] fp16
    k_cache:    [pages, page_size, heads_kv, 256] e5m2 or fp16
    v_cache:    same shape/dtype as k_cache
    page_table: [batch, max_pages] int32, or an empty tensor for a dense cache
    seq_lens:   [batch] int32

    variant:    "serial"    - all warps load, then six warps compute
                "occ2"      - serial, two CTAs per SM so the SM overlaps them
                "pipelined" - ten warps stream tile i+1 while six compute i
                "regpipe"   - as pipelined, but tile i+1 waits in registers
                              instead of a second smem buffer, so two CTAs
                              stay resident per SM
                "pv2"       - PV split across two warps per query row, so the
                              compute phase runs on twelve warps not six
    """
    ext = _load_ops()
    if ext is None:
        raise RuntimeError("SM70 long-context decode extension is unavailable.")
    batch, heads, dim = q.shape
    partial_o = torch.empty(
        (batch, max_splits, heads, dim), dtype=torch.float16, device=q.device
    )
    partial_lse = torch.empty(
        (batch, max_splits, heads), dtype=torch.float32, device=q.device
    )
    # E5M2 is handed over as raw bytes because the kernel converts it with a
    # byte shift. E4M3 is passed with its own dtype so the kernel can tell the
    # two apart and take the general conversion.
    fp8_e5m2 = k_cache.dtype == torch.float8_e5m2
    kv = (lambda t: t.view(torch.uint8)) if fp8_e5m2 else (lambda t: t)
    entry = {
        "serial": ext.sm70_longctx_decode,
        "occ2": ext.sm70_longctx_decode_occ2,
        "pipelined": ext.sm70_longctx_decode_pipelined,
        "regpipe": ext.sm70_longctx_decode_regpipe,
        "pv2": ext.sm70_longctx_decode_pv2,
    }[variant]
    entry(
        q.contiguous(), kv(k_cache).contiguous(), kv(v_cache).contiguous(),
        page_table.to(dtype=torch.int32).contiguous(),
        seq_lens.to(dtype=torch.int32).contiguous(),
        int(max_splits), int(min_tokens_per_split),
        float(softmax_scale), float(k_scale), float(v_scale),
        partial_o, partial_lse,
    )
    return partial_o, partial_lse
