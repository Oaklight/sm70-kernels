"""Differential cost of the decode kernel's compute phase.

Runs one arm per invocation and patches a temporary copy of the package, so the
tree is never modified:

    python benchmarks/phase_differential.py --arm full
    python benchmarks/phase_differential.py --arm nocompute

`nocompute` removes `decode_tile_compute` from `decode_partial_kernel` and
replaces it with a two-`LDS` sink, which keeps every load, E5M2 conversion, smem
store and barrier while dropping the math. The difference is what the compute
phase costs the kernel - the number `docs/DESIGN.md` quotes as 0.283 ms of 0.620
at e5m2/TP2/256k.

Why not `clock64` counters: bracketing the phases pushed the kernel's stack frame
from 240 to 288 bytes, because it holds 64 registers under
`__launch_bounds__(kThreads, 2)` and the counters spill. A shared-memory
accumulator still left 272 bytes and the kernel at 0.790 ms against 0.620. The
differential has no such cost.

The sink matters. With nothing reading `ks`/`vs`, ptxas is entitled to drop the
loads that just staged them, and the arm would time a kernel that does no work.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPUTE_OLD = """    // ---- QK / softmax / PV for one 32-token tile ----
    if (is_compute_warp) {
      decode_tile_compute(warp, tile_tokens, ks, vs, qs, scores,
                          probs, m_row[warp], l_row[warp],
                          o_acc[warp], scale_log2);
    }
    __syncthreads();
"""

COMPUTE_NEW = """    // ---- compute removed for the differential: keep the staged tile live ----
    if (tid == 0) {
      dbg_sink += __half2float(ks[tile & 31]) + __half2float(vs[tile & 31]);
    }
    __syncthreads();
"""

DECL_OLD = """  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {"""

DECL_NEW = """  constexpr int kLoadIters = (kBlockN * (kDim / 16) + kThreads - 1) / kThreads;
  float dbg_sink = 0.f;
  for (int tile = 0; tile < num_tiles; ++tile) {"""

FIN_OLD = """  // ---- Finalize partial output for this partition ----
  if (is_compute_warp) {"""

FIN_NEW = """  // ---- Finalize partial output for this partition ----
  if (tid == 0 && dbg_sink == 1.2345e-30f) {
    partial_lse[0] = dbg_sink;
  }
  if (is_compute_warp) {"""


def _prepare(arm):
    """Copy the package to a temp dir and, for the nocompute arm, strip the math."""
    root = tempfile.mkdtemp(prefix="sm70phase_")
    shutil.copytree(os.path.join(ROOT, "sm70_kernels"),
                    os.path.join(root, "sm70_kernels"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.o"))
    if arm == "nocompute":
        path = os.path.join(root, "sm70_kernels", "longctx_decode", "csrc",
                            "sm70_longctx_decode.cu")
        src = open(path).read()
        # The compute block is unique to decode_partial_kernel. The declaration
        # and the sink consumer are not - the other kernels share that boilerplate
        # - so both are applied at their first occurrence, which is this kernel
        # because it is defined before them.
        n = src.count(COMPUTE_OLD)
        assert n == 1, "compute block matched %d times" % n
        src = src.replace(COMPUTE_OLD, COMPUTE_NEW)
        for old, new, what in (
            (DECL_OLD, DECL_NEW, "sink declaration"),
            (FIN_OLD, FIN_NEW, "sink consumer"),
        ):
            n = src.count(old)
            assert n >= 1, "%s not found" % what
            i = src.index(old)
            src = src[:i] + new + src[i + len(old):]
        open(path, "w").write(src)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("full", "nocompute"), default="full")
    ap.add_argument("--contexts", default="262144")
    ap.add_argument("--kv-heads", default="2,4", help="2 = TP2, 4 = TP1")
    ap.add_argument("--dtype", default="e5m2,fp16")
    ap.add_argument("--splits", default="80")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--passes", type=int, default=5)
    args = ap.parse_args()

    root = _prepare(args.arm)
    os.environ["SM70_KERNELS_BUILD_DIR"] = tempfile.mkdtemp(prefix="sm70phasebuild_")
    sys.path.insert(0, root)
    from sm70_kernels.longctx_decode._loader import (
        sm70_cuda_decode_available,
        sm70_cuda_decode_partial,
    )

    if not sm70_cuda_decode_available():
        raise SystemExit("needs a real SM70 device")
    dev = torch.cuda.current_device()
    DIM = 256
    PAGE = 16
    scale = DIM ** -0.5

    def clock_mhz():
        try:
            return torch.cuda.clock_rate()
        except Exception:
            return -1

    print(f"arm={args.arm} gpu={dev} {torch.cuda.get_device_name(dev)}", flush=True)

    # Ramp on something that actually draws power. The decode kernel alone does
    # not always reach the boost - the nocompute arm cannot, since it only moves
    # bytes - and the first block of a fresh process has read 1290 MHz even after
    # 30 s of it. A large fp16 matmul gets there in a couple of seconds.
    ramp = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 60:
        ramp @ ramp
        torch.cuda.synchronize()
        if time.perf_counter() - t0 > 2 and clock_mhz() >= 1500:
            break
    print(f"clock after ramp: {clock_mhz()} MHz", flush=True)

    def clock_ref():
        """A compute-bound reference: its time scales as 1/clock.

        The two arms run in separate processes and can sit at different clocks,
        so the honest comparison is each arm's decode time divided by its own
        reference. Report it rather than trusting a nominal 1530.
        """
        best = 1e9
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                ramp @ ramp
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / 20 * 1e3)
        return best

    for ctx in [int(x) for x in args.contexts.split(",")]:
        for kvh in [int(x) for x in args.kv_heads.split(",")]:
            q = torch.randn(1, kvh * 6, DIM, dtype=torch.float16, device=dev) * scale
            pt = torch.arange(ctx // PAGE, dtype=torch.int32, device=dev).unsqueeze(0)
            sl = torch.tensor([ctx], dtype=torch.int32, device=dev)
            shape = (ctx // PAGE + 1, PAGE, kvh, DIM)
            for dtype in args.dtype.split(","):
                if dtype == "fp16":
                    k = torch.randn(shape, dtype=torch.float16, device=dev)
                    v = torch.randn(shape, dtype=torch.float16, device=dev)
                else:
                    fmt = torch.float8_e5m2 if dtype == "e5m2" else torch.float8_e4m3fn
                    k = torch.randn(shape, dtype=torch.float16, device=dev).to(fmt)
                    v = torch.randn(shape, dtype=torch.float16, device=dev).to(fmt)
                moved = 2 * ctx * kvh * DIM * (2 if dtype == "fp16" else 1)
                for splits in [int(x) for x in args.splits.split(",")]:
                    # Warm the decode path, then boost immediately before
                    # timing. Ordering is the whole story here: the decode
                    # kernel is memory-heavy and does not hold the boost by
                    # itself, so whichever block is timed first in a fresh
                    # process reads 1290 MHz and 16% slow (fp16 shows it too,
                    # at 0.978 against 0.944). A second of matmul right before
                    # the measurement puts the clock at 1530 and the 60 ms of
                    # timing is too short for it to fall back.
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 1.5:
                        sm70_cuda_decode_partial(q, k, v, pt, sl, splits, 32, scale,
                                                 1.0, 1.0, variant="occ2")
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 1.0:
                        ramp @ ramp
                    torch.cuda.synchronize()
                    before = clock_mhz()
                    best = 1e9
                    for _ in range(args.passes):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(args.reps):
                            sm70_cuda_decode_partial(q, k, v, pt, sl, splits, 32,
                                                     scale, 1.0, 1.0, variant="occ2")
                        torch.cuda.synchronize()
                        best = min(best, (time.perf_counter() - t0) / args.reps * 1e3)
                    print(f"  ctx={ctx:>6} kvh={kvh} {dtype:5s} splits={splits:>3} "
                          f"{best:7.3f} ms  {moved / 1e9 / (best / 1e3):6.1f} GB/s  "
                          f"clock={before} MHz  ref={clock_ref():.3f} ms", flush=True)
                del k, v
                torch.cuda.empty_cache()
            del q, pt, sl
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
