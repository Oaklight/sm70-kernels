"""Serial vs pipelined SM70 decode-partial, with a numeric cross-check.

Every row is printed with the host name and the board type, because the two
lab hosts are not interchangeable even though the accelerator is: lambda12
has two chips joined by NV2, lambda4 has eight in two four-chip NVLink
islands. A single-GPU streaming kernel sees the same part on both, so any
difference here is clock/power or host affinity, not board type - but it has
to be measured rather than assumed, hence bench-host.
"""

import argparse
import os
import socket
import time

import torch

from sm70_kernels.longctx_decode._loader import (
    sm70_cuda_decode_available,
    sm70_cuda_decode_partial,
)

DIM = 256
PAGE = 16


def _make(ctx, heads_kv, dtype, device):
    pages = ctx // PAGE + 1
    shape = (pages, PAGE, heads_kv, DIM)
    if dtype in ("e5m2", "e4m3"):
        fmt = torch.float8_e5m2 if dtype == "e5m2" else torch.float8_e4m3fn
        k = torch.randn(shape, dtype=torch.float16, device=device).to(fmt)
        v = torch.randn(shape, dtype=torch.float16, device=device).to(fmt)
    else:
        k = torch.randn(shape, dtype=torch.float16, device=device)
        v = torch.randn(shape, dtype=torch.float16, device=device)
    page_table = torch.arange(ctx // PAGE, dtype=torch.int32, device=device).unsqueeze(0)
    return k, v, page_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="32768,131072,262144")
    ap.add_argument("--splits", default="80,160,320")
    ap.add_argument("--kv-heads", default="4,2", help="4 = TP1, 2 = TP2")
    ap.add_argument("--dtype", default="e5m2,e4m3,fp16")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=40,
                    help="iterations to run before timing; a V100 idles at "
                         "135 MHz and only reaches 1530 under sustained load")
    args = ap.parse_args()

    if not sm70_cuda_decode_available():
        raise SystemExit("needs a real SM70 device")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    host = socket.gethostname()
    print(f"host={host} gpu={device} board={name}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")

    score_scale = DIM**-0.5
    for ctx in [int(x) for x in args.contexts.split(",")]:
        for heads_kv in [int(x) for x in args.kv_heads.split(",")]:
            heads = heads_kv * 6
            q = torch.randn(1, heads, DIM, dtype=torch.float16, device=device) * score_scale
            seq_lens = torch.tensor([ctx], dtype=torch.int32, device=device)
            moved = 2 * ctx * heads_kv * DIM  # K + V bytes, e5m2
            for dtype in args.dtype.split(","):
                k, v, page_table = _make(ctx, heads_kv, dtype, device)
                if dtype == "fp16":
                    moved *= 2
                for splits in [int(x) for x in args.splits.split(",")]:
                    row = {}
                    for variant in ("serial", "occ2", "pipelined"):
                        for _ in range(args.warmup):
                            sm70_cuda_decode_partial(
                                q, k, v, page_table, seq_lens, splits, 32,
                                score_scale, 1.0, 1.0, variant=variant,
                            )
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(args.reps):
                            out = sm70_cuda_decode_partial(
                                q, k, v, page_table, seq_lens, splits, 32,
                                score_scale, 1.0, 1.0, variant=variant,
                            )
                        torch.cuda.synchronize()
                        row[variant] = (time.perf_counter() - t0) / args.reps * 1e3
                        row[variant + "_o"] = out[0]
                        row[variant + "_lse"] = out[1]
                    d_o = max(
                        (row["serial_o"].float() - row[v + "_o"].float()).abs().max().item()
                        for v in ("occ2", "pipelined")
                    )
                    d_l = max(
                        (row["serial_lse"] - row[v + "_lse"]).abs().max().item()
                        for v in ("occ2", "pipelined")
                    )
                    best = min(row["occ2"], row["pipelined"])
                    print(
                        f"ctx={ctx:>6} kvh={heads_kv} {dtype:4s} splits={splits:>3} "
                        f"serial={row['serial']:7.3f} occ2={row['occ2']:7.3f} "
                        f"pipe={row['pipelined']:7.3f} best=x{row['serial'] / best:4.2f} "
                        f"bw={moved / 1e9 / (best / 1e3):6.1f}GB/s "
                        f"max|do|={d_o:.2e} max|dlse|={d_l:.2e}",
                        flush=True,
                    )
                del k, v, page_table
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
