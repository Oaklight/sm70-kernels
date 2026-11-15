"""GDN chunk state kernel: TileLang vs the Triton reference.

Sequential pass over BT=64 chunks:
    v_new_t = u_t - w_t @ h_{t-1}^T
    h_t     = exp(g_last_t) * h_{t-1} + v_new_t_scaled^T @ k_t
h[t] is the state entering chunk t (stored before the update); the final
state is written back to S0. Shapes are per TP rank: B=1, T=4096, Hg=4,
H=12, K=V=128, BV=32. The state is carried in an fp32 fragment.

Measured on a V100-SXM2: Triton 15.4 ms, TileLang 0.47 ms (33x).

    python benchmarks/gdn_chunk_h.py --tokens 4096
    python benchmarks/gdn_chunk_h.py --tokens 4096 --triton   # needs a free card

Set SGLANG_SRC to the SGLang python tree to enable --triton.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SGLANG_SRC = os.environ.get("SGLANG_SRC")
if SGLANG_SRC:
    sys.path.insert(0, SGLANG_SRC)

from sm70_kernels.gdn_chunk_h import get_chunk_h_kernel


def make_inputs(B, T, H, Hg, K, V, dev="cuda", seed=1):
    """Well-conditioned inputs: l2-normalised keys, chunk-local log-decay."""
    torch.manual_seed(seed)
    BT = 64
    NT = T // BT

    def l2norm(x):
        return x / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)

    k = l2norm(torch.randn(B, T, Hg, K, device=dev, dtype=torch.float16)).to(torch.float16)
    w = (torch.randn(B, T, H, K, device=dev, dtype=torch.float16) * (K ** -0.5)).to(torch.float16)
    u = (torch.randn(B, T, H, V, device=dev, dtype=torch.float16) * 0.5).to(torch.float16)
    logdecay = -torch.nn.functional.softplus(
        torch.randn(B, T, H, device=dev, dtype=torch.float32) * 1.5
    )
    g = logdecay.view(B, NT, BT, H).cumsum(dim=2).view(B, T, H).contiguous()
    s0 = (torch.randn(B, H, V, K, device=dev, dtype=torch.float16) * 0.05).to(torch.float16)
    return k, w, u, g, s0


def reference_fp32(k, w, u, g, s0, BT=64):
    B, T, Hg, K = k.shape
    H, V = w.shape[2], u.shape[3]
    NT = T // BT
    dev = k.device
    state = s0.float().clone()
    hout = torch.empty(B, NT, H, V, K, device=dev, dtype=torch.float32)
    vnew = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
    groups = H // Hg
    for hh in range(H):
        kh = hh // groups
        hstate = state[:, hh]
        for it in range(NT):
            sl = slice(it * BT, (it + 1) * BT)
            hout[:, it, hh] = hstate
            w_t = w[:, sl, hh].float()
            k_t = k[:, sl, kh].float()
            u_t = u[:, sl, hh].float()
            v = u_t - torch.einsum("btk,bvk->btv", w_t, hstate)
            vnew[:, sl, hh] = v
            glast = g[:, it * BT + BT - 1, hh]
            scale = torch.exp(glast[:, None, None] - g[:, sl, hh][:, :, None])
            hstate = hstate * torch.exp(glast)[:, None, None] + torch.einsum(
                "btv,btk->bvk", v * scale, k_t
            )
        state[:, hh] = hstate
    return hout, vnew, state


def report(name, got, ref):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    big = ref.abs() >= 1e-2
    max_abs = err.max().item()
    max_rel = (err[big] / ref.abs()[big]).max().item() if big.any() else 0.0
    small = err[~big].max().item() if (~big).any() else 0.0
    ulp = ref.abs().max().item() * 2 ** -11
    print(
        "%-8s max_abs %.4e  max_rel %.4e  small_band_abs %.4e  %.2f ULP"
        % (name, max_abs, max_rel, small, max_abs / ulp if ulp else 0.0)
    )


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--triton", action="store_true", help="also run the Triton reference")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    B, T, H, Hg, K, V, BT, BV = 1, args.tokens, 12, 4, 128, 128, 64, 32
    NT = T // BT
    k, w, u, g, s0 = make_inputs(B, T, H, Hg, K, V)

    kernel = get_chunk_h_kernel(B, T, H, Hg, K, V, BT, BV, 128)
    state = s0.clone()
    h, vnew = kernel(k, w, u, g, state)
    torch.cuda.synchronize()
    ref_h, ref_v, ref_s = reference_fp32(k, w, u, g, s0, BT)
    report("h", h, ref_h)
    report("v_new", vnew, ref_v)
    report("state", state, ref_s)

    # S0 is mutated, so each timed run starts from a fresh clone.
    t = bench(lambda: kernel(k, w, u, g, s0.clone()), args.iters)
    flop = 2 * NT * H * (4 * BT * K * BV + 4 * BV * BT * K)
    print("tilelang %.3f ms  %.1f GF/s  (%.0f tokens)" % (t, flop / t / 1e9, T))

    if args.triton:
        from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

        idx = torch.arange(B, device=k.device, dtype=torch.int32)

        def run_triton():
            s = s0.clone()
            return chunk_gated_delta_rule_fwd_h(
                k=k, w=w, u=u, g=g, initial_state=s, initial_state_indices=idx,
                cu_seqlens=None, save_new_value=True, inplace_update=True,
            )

        ref_h, ref_v = run_triton()
        report("h vs tri", h, ref_h)
        report("v vs tri", vnew, ref_v)
        t_ref = bench(run_triton, args.iters)
        print("triton   %.3f ms  %.1f GF/s  speedup %.2fx" % (t_ref, flop / t_ref / 1e9, t_ref / t))


if __name__ == "__main__":
    main()
