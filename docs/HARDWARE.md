# The lab hosts, and what transfers between them

Every number this repo publishes is produced on one host. This page says what
that means for the next host.

## Inventory (probed 2026-10-28)

`lambda012` does not resolve from the login/control host; the reachable hosts
are `lambda0`, `lambda1`, `lambda2`, `lambda4` and `lambda12`.

| host | GPUs | board | power | max mem clock | max SM clock | topology |
|---|---:|---|---:|---:|---:|---|
| lambda12 | 2 | Tesla V100-SXM2-32GB | 300 W | 877 MHz | 1530 MHz | one `NV2` peer link |
| lambda4 | 8 | Tesla V100-SXM2-32GB | 300 W | 877 MHz | 1530 MHz | two 4-GPU islands, `NV1`/`NV2` inside, `SYS` across |
| lambda0 | 8 | Tesla V100-SXM2-32GB | 300 W | 877 MHz | 1530 MHz | same as lambda4 |
| lambda1 | 8 | Tesla V100-SXM2-32GB | 300 W | 877 MHz | 1530 MHz | same as lambda4 |
| lambda2 | 8 | Tesla V100-SXM2-32GB | 300 W | 877 MHz | 1530 MHz | same as lambda4 |

All five are SXM2. There is no PCIe V100 in the lab and no NVSwitch, so the
"PCIe vs SXM2" axis does not apply; the axis that does is **how many chips
share an NVLink island**, and that is what changes a result.

## What transfers: a single-GPU kernel

Tested, and the first answer was wrong, so the method is on the record too.

The first cross-host comparison showed lambda2 16% slower than lambda12 on
every variant, with the rankings preserved. That looked like a host effect. It
was a **cold-clock artifact**: a V100 sits at 135 MHz and only reaches 1530 MHz
under sustained load, and the harness warmed up for three iterations (~4 ms) -
enough to warm one host that had already been running benchmarks and not
enough for a host that had not.

With 40 warmup iterations, the same binary on the two hosts:

| at 256k, splits 80, e5m2 | lambda12 kvh=2 | lambda2 kvh=2 |
|---|---:|---:|
| serial + batched loads | 0.861 ms | 0.869 ms |
| two CTAs per SM | **0.630 ms** | **0.628 ms** |
| warp-specialized pipeline | 0.709 ms | 0.718 ms |

Same part, same clock ceiling, same number, to within 0.3%. So a single-GPU
kernel result transfers across the lab as long as the harness controls the
clock. `attn_variants.py` now defaults to 40 warmup iterations for that reason;
anything measured with a shorter warmup is a statement about the clock ramp.

TP1 (4 kv heads, 320 CTAs, two waves of two CTAs per SM) is the noisy one:
1.246 to 1.446 ms across runs where TP2 never moves. If a comparison matters,
run TP1 several times and take the best, not the first.

## What does not transfer: anything with collectives

TP ranks talk over NVLink inside an island and over `SYS` between islands, and
the difference is not a few percent:

- 27B TP2 on lambda12, `NV2` peer link: a 57 µs all-reduce.
- Flash-Next TP8/EP8 on lambda4, five of eight ranks crossing `SYS`: 5.4
  collectives per step at a 2.20 ms median, 12.2 ms of a 29.1 ms step.

138 KB per collective is 0.46 µs on NVLink, so those 2.20 ms are round-trip
latency, not bandwidth, and no all-reduce algorithm fixes them - `1stage` and
`2stage` measured the same. On lambda4 an 8-way TP therefore carries a cost
that a 4-way TP plus 2 pipeline stages would not, and the 4-GPU islands are why.

## Rule

Label every measured number with the host and the rank count. A kernel number
without a host is a rounding of one machine; a throughput number without a
topology is a statement about lambda12's two-chip NVLink, not about V100.
