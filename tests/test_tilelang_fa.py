"""tilelang_fa ships with the package, imports lazily, and picks the tuned tile."""

from __future__ import annotations

import ast
from pathlib import Path

import sm70_kernels
import sm70_kernels.tilelang_fa as fa


def _source() -> str:
    return Path(fa.__file__).with_name("_kernels_paged.py").read_text()


def _dict_call(node):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return ast.literal_eval(node)


def _best_configs():
    for node in ast.parse(_source()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "_BEST_CONFIGS" for target in node.targets
        ):
            return {int(ast.literal_eval(k)): _dict_call(v) for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError("_BEST_CONFIGS not found")


def test_exports():
    assert "tilelang_fa" in sm70_kernels.__all__


def test_package_import_is_lazy():
    # CI has torch but no tilelang; importing the package must not load the
    # kernel module, which imports tilelang at module scope.
    import subprocess
    import sys

    code = "import sys, sm70_kernels.tilelang_fa; print('tilelang' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_kernel_source_parses():
    ast.parse(_source())


def test_dim256_tile_uses_registers_for_q():
    cfg = _best_configs()[256]
    assert cfg["q_in_regs"] is True
    assert cfg["block_N"] == 64
    assert cfg["block_M"] == 64
    assert cfg["num_stages"] == 0
