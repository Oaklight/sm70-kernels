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

The accelerator is the same part with the same clock ceilings, so the shape of
a kernel result transfers and the absolute value does not. Measured with the
same binary, same driver, same 262144-token context, e5m2 KV, splits 80:

| variant | lambda12 kvh=2 | lambda2 kvh=2 | ratio |
|---|---:|---:|---:|
| serial + batched loads | 0.891 ms | 1.033 ms | 1.16x |
| two CTAs per SM | **0.628 ms** | **0.728 ms** | 1.16x |
| warp-specialized pipeline | 0.719 ms | 0.837 ms | 1.16x |

Every variant is 16% slower on lambda2 and the rankings are identical. So
`occ2 > pipelined > serial` is a property of the kernel, while "0.63 ms" is a
property of lambda12 on that afternoon. Report both or neither.

The 16% is not explained by the board: the plausible causes are the neighbours
(lambda2 had a job holding 7.4 GB on GPU 0 at 10%), the shared 2400 W chassis
power budget, and host/PCIe affinity. If a number matters, re-measure it on the
host that will run it rather than scaling.

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
