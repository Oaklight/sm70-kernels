"""The package must import and expose its API without a GPU."""

import sm70_kernels


def test_exports():
    assert "longctx_decode" in sm70_kernels.__all__


def test_decode_api():
    from sm70_kernels.longctx_decode import (
        sm70_cuda_decode_available,
        sm70_cuda_decode_partial,
    )

    assert callable(sm70_cuda_decode_partial)
    # No GPU in CI: the availability probe must answer False, not raise.
    assert sm70_cuda_decode_available() is False


def test_source_ships_with_package():
    from pathlib import Path

    import sm70_kernels.longctx_decode._loader as loader

    assert Path(loader._SRC_PATH).is_file()
    assert Path(loader._SRC_PATH).suffix == ".cu"
