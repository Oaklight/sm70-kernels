"""Check the split-KV partials against an fp32 reference, per split.

The partial ABI is (o, lse) per (batch, split, head) with o normalized by the
split's own softmax denominator, so each split can be validated on its own
without the merge kernel. That keeps this check independent of the TileLang
combine, which lives in the sglang tree rather than here.
"""

import argparse

import torch

from sm70_kernels.longctx_decode._loader import (
    sm70_cuda_decode_available,
    sm70_cuda_decode_partial,
)

DIM = 256
PAGE = 16
LOG2E = 1.4426950408889634


def reference(q, k_h, v_h, split_begin, split_end, scale):
    """fp32 reference partial for one kv head, all six query rows.

    Mirrors the kernel's formulation: the raw dot product is held unscaled, the
    running max is over raw scores, and scale*log2(e) is applied inside exp2.
    The partial output is normalized by this split's own denominator and the
    lse is reported in scaled log2 units.
    """
    kf = k_h[split_begin:split_end].float()
    vf = v_h[split_begin:split_end].float()
    log2s = scale * LOG2E
    rows = []
    lses = []
    for r in range(q.shape[0]):
        raw = kf @ q[r].float()
        m = raw.max()
        p = torch.exp2((raw - m) * log2s)
        denom = p.sum()
        rows.append((p @ vf) / denom)
        lses.append(torch.log2(denom) + m * log2s)
    return torch.stack(rows), torch.stack(lses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--splits", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    args = ap.parse_args()
    if not sm70_cuda_decode_available():
        raise SystemExit("needs a real SM70 device")

    dev = "cuda:0"
    torch.manual_seed(0)
    scale = DIM**-0.5
    heads_kv = args.kv_heads
    heads = heads_kv * 6

    dense_k = torch.randn(args.context, heads_kv, DIM, dtype=torch.float16, device=dev)
    dense_v = torch.randn(args.context, heads_kv, DIM, dtype=torch.float16, device=dev)
    q = torch.randn(1, heads, DIM, dtype=torch.float16, device=dev)
    seq_lens = torch.tensor([args.context], dtype=torch.int32, device=dev)

    split_len = (args.context + args.splits - 1) // args.splits
    active = min(args.splits, max(1, (args.context + 32 - 1) // 32))

    for kv_dtype in ("e5m2", "fp16"):
        if kv_dtype == "e5m2":
            kc = dense_k.to(torch.float8_e5m2)
            vc = dense_v.to(torch.float8_e5m2)
            kq, vq = kc, vc  # kernel dequantizes from the same bytes
        else:
            kq, vq = dense_k, dense_v
        page_table = torch.arange(
            args.context // PAGE + 1, dtype=torch.int32, device=dev
        ).unsqueeze(0)
        paged_k = (
            (dense_k.reshape(-1, PAGE, heads_kv, DIM) if kv_dtype == "fp16"
             else dense_k.to(torch.float8_e5m2).reshape(-1, PAGE, heads_kv, DIM))
        )
        paged_v = (
            (dense_v.reshape(-1, PAGE, heads_kv, DIM) if kv_dtype == "fp16"
             else dense_v.to(torch.float8_e5m2).reshape(-1, PAGE, heads_kv, DIM))
        )
        for variant in ("serial", "occ2", "pipelined"):
            o, lse = sm70_cuda_decode_partial(
                q, paged_k, paged_v, page_table, seq_lens, args.splits, 32,
                scale, 1.0, 1.0, variant=variant,
            )
            worst_o = 0.0
            worst_l = 0.0
            for s in range(active):
                begin = s * split_len
                end = min(args.context, begin + split_len)
                if begin >= end:
                    continue
                for h in range(heads_kv):
                    ref_o, ref_l = reference(
                        q[0, h * 6:(h + 1) * 6], kq[:, h], vq[:, h],
                        begin, end, scale,
                    )
                    got_o = o[0, s, h * 6:(h + 1) * 6].float()
                    got_l = lse[0, s, h * 6:(h + 1) * 6]
                    worst_o = max(worst_o, (got_o - ref_o).abs().max().item())
                    worst_l = max(worst_l, (got_l - ref_l).abs().max().item())
            ref_rms = ref_o.pow(2).mean().sqrt().item()
            print(
                f"{kv_dtype:4s} {variant:9s} max|do|={worst_o:.5f} "
                f"max|dlse|={worst_l:.5f} ref_rms={ref_rms:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
