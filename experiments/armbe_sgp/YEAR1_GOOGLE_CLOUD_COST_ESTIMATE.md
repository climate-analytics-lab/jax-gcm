# Year-1 Google Cloud Credit Estimate

## Recommendation

Request **$35,000 in Google Cloud credits for Year 1**. The modeled conservative
usage is **$20,818** at public on-demand prices. The difference is an explicit
program reserve for gradient-based calibration, higher-resolution runs,
ensembles, failed autoresearch trials, and price or capacity changes; it is not
expected baseline spend.

This estimate covers cloud infrastructure and model API usage for two to three
full-time researchers. It does not include salaries, local hardware, indirect
costs, taxes, or paid support.

## Estimate Summary

| Scenario | Annual cost | Interpretation |
| --- | ---: | --- |
| Baseline development | $18,956 | 1,280 A100 hours, no TPU pilot, Gemini 2.5 Flash |
| Conservative plan | $20,818 | 1,600 A100 hours, TPU pilot, Gemini 2.5 Flash |
| Conservative plan with Gemini 2.5 Pro | $23,268 | Same infrastructure with a higher-capability LLM allowance |
| Recommended credit request | **$35,000** | Conservative plan plus $14,182 unallocated program reserve |

The reserve is 68% of modeled conservative spend. Even relative to the more
expensive Gemini 2.5 Pro scenario, it leaves $11,732, or 50%, for workload
uncertainty. Cloud credits should be monitored against the modeled line items;
the reserve should not be pre-allocated in the pricing calculator to fictitious
resources.

## Calculator-Ready Conservative Plan

Rates are public USD on-demand list prices checked on 2026-08-31. Compute and
storage use Iowa (`us-central1`) unless stated otherwise. Quantities are annual;
monthly equivalents are provided to make entry into the Google Cloud Pricing
Calculator straightforward.

| Calculator item | Configuration or quantity | Monthly equivalent | Public rate | Annual cost |
| --- | --- | ---: | ---: | ---: |
| Interactive CPU | `e2-standard-8`, 8 vCPU, 32 GiB; 3,600 hours/year | 300 hours | $0.268045680/hour | $964.96 |
| Batch CPU | `c2d-standard-32`, 32 vCPU, 128 GiB; 3,000 hours/year | 250 hours | $1.452768000/hour | $4,358.30 |
| GPU | `a2-ultragpu-1g`, one A100 80 GB; 1,600 hours/year | 133.33 hours | $5.068797890/hour | $8,110.08 |
| TPU pilot | TPU v5e; 200 chip-hours/year | 16.67 chip-hours | $1.20/chip-hour | $240.00 |
| Balanced Persistent Disk | 19,200 GiB-month/year | 1,600 GiB | $0.10/GiB-month | $1,920.00 |
| SSD Persistent Disk | 12,000 GiB-month/year | 1,000 GiB | $0.17/GiB-month | $2,040.00 |
| Cloud Storage Standard | 24,000 GiB-month/year | 2,000 GiB | $0.020/GiB-month | $480.00 |
| Cloud Storage Nearline | 96,000 GiB-month/year | 8,000 GiB | $0.010/GiB-month | $960.00 |
| Internet egress | 2,048 GiB/year, spread across the year | 170.67 GiB | $0.12/GiB before free tier | $244.32 |
| Gemini 2.5 Flash | 1.0B input and 200M output tokens/year | 83.33M in, 16.67M out | $0.30/M in, $2.50/M out | $800.00 |
| Cloud Logging | 100 GiB/project/month | 100 GiB | first 50 GiB free, then $0.50/GiB | $300.00 |
| Storage operations and retrieval | Planning allowance | n/a | usage-dependent | $400.00 |
| **Total** |  |  |  | **$20,817.67** |

The A100 rate includes the predefined VM host, 12 vCPUs, 170 GB of host memory,
the GPU, and bundled Local SSD. A separate VM or GPU charge must not be added.
The persistent-disk and Cloud Storage quantities are aggregate GiB-months, not
capacities that should be multiplied by 12 again.

The $400 storage-processing allowance can be represented outside the calculator
or entered as approximately 10 million Standard Class A operations ($50), 10
million Nearline Class A operations ($100), and 24 TiB of Nearline retrieval
($245.76). Class B operations are immaterial at this scale. Nearline has a
30-day minimum duration and a $0.01/GiB retrieval charge.

## Scenario Arithmetic

The baseline uses 1,280 A100 hours, before the conservative 25% retry and
exploration uplift, and omits the 200-hour TPU compatibility pilot:

```text
CPU                         $5,323.27
1,280 A100 VM-hours          6,488.06
Persistent and object data   5,400.00
Egress                         244.32
Gemini 2.5 Flash               800.00
Logging                        300.00
Storage processing             400.00
                            ----------
Baseline                    $18,955.65
```

The conservative plan increases the A100 allocation to 1,600 hours and adds the
TPU pilot. Substituting Gemini 2.5 Pro at $1.25 per million input tokens and
$10 per million output tokens changes the LLM line from $800 to $3,250 and the
total from $20,817.67 to $23,267.67. Requests over 200,000 input tokens have
higher Pro rates and must be budgeted separately if long-context use dominates.

## Evidence Behind the Compute Allowance

The repository contains two relevant measured benchmarks in
`ECHAM_WORKFLOW_COST_EXPERIMENT.md`:

- A prescribed-state full ECHAM 1M plus RRTMGP call at T63L47 took a median
  2.054 seconds without the additional clear-sky CRE solve on a shared NVIDIA
  RTX PRO 6000 Blackwell GPU. A 960-state, one-scheme evaluation extrapolates to
  about 33 minutes before remapping and metric I/O.
- Continuous ECHAM 1M plus RRTMGP evolution on an idle A100 PCIe 40 GB took
  20.51 seconds per simulated day without CRE and 36.40 seconds with CRE. The
  corresponding raw forward-only throughput is about 2.08 or 3.69 GPU-hours per
  simulated year.

The 1,600-hour allowance is therefore not derived from a single production run.
It funds many candidate schemes, prescribed-state evaluations, continuous
forecast campaigns, and retries. Measured timings exclude ERA5 remapping, host
serialization, gradient evaluation, and the greater memory pressure of
differentiating through RRTMGP.

Measured scaling in `docs/source/design/parallelization.md` also affects the
deployment plan. On an 8 x A100 80 GB PCIe host, T63 achieved only 0.95x speedup
on two GPUs, 0.94x on four, and 0.53x on eight. TL255 achieved a useful 1.79x on
two GPUs. Year-1 T63 throughput should therefore come from parallel independent
single-GPU jobs, not a sharded multi-GPU T63 job. Multi-GPU reservations should
only follow resolution- and interconnect-specific scaling measurements.

## Measured Evidence Versus Planning Assumptions

| Item | Status |
| --- | --- |
| T63L47 forward physics and evolution timings | Measured locally on RTX PRO 6000 and A100 40 GB hardware |
| T63 and TL255 multi-GPU scaling | Measured on an 8 x A100 80 GB PCIe host |
| 1,600 A100 80 GB VM-hours | Planning allocation with 25% uplift over the 1,280-hour baseline |
| CPU hours | Planning allocation for development, preprocessing, remapping, orchestration, and analysis |
| Disk and object-storage capacity | Planning allocation for active state caches, scratch data, checkpoints, and retained archives |
| TPU hours | Compatibility and scaling experiment only; no production TPU throughput is assumed |
| LLM tokens | Planning allowance for code, literature, experiment orchestration, and result triage |
| Gradient and higher-resolution cost | Not yet measured; covered by the credit reserve rather than false-precision line items |

## Required Scaling Experiments

Run these early in Year 1 before moving reserve funds into accelerator line
items:

1. Benchmark one forward step, one loss evaluation, and one gradient step on an
   A100 80 GB VM using identical T63L47 states and output materialization.
2. Measure peak memory and throughput for column chunking through RRTMGP, then
   estimate the number of gradient-based trials supportable per GPU-hour.
3. Run identical T63L47 forward and gradient workloads on one TPU v5e slice.
   Continue TPU use only if the code is compatible and cost per accepted trial
   is competitive.
4. Repeat one- versus two-device scaling at TL255 on the actual Google Cloud
   interconnect before requesting multi-GPU capacity.
5. Measure ERA5 remapping, checkpoint, and host serialization costs separately
   so CPU, disk, and network bottlenecks are not misattributed to physics.
6. Review actual spend after 30 and 90 days and reallocate the reserve based on
   accepted-trial cost rather than raw accelerator utilization.

## Pricing Assumptions and Sources

- [Accelerator-optimized VM pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized):
  `a2-ultragpu-1g` in Iowa under Default/on-demand pricing.
- [A2 machine specifications](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a2_vms):
  GPU, vCPU, memory, and Local SSD composition of the A100 80 GB VM.
- [General-purpose VM pricing](https://cloud.google.com/products/compute/pricing/general-purpose#e2-machine-types):
  `e2-standard-8` in Iowa.
- [Compute-optimized VM pricing](https://cloud.google.com/products/compute/pricing/compute-optimized#c2d-machine-types):
  `c2d-standard-32` in Iowa.
- [Compute disk pricing](https://cloud.google.com/compute/disks-image-pricing#persistent-disk-and-hyperdisk-pricing):
  balanced and SSD Persistent Disk prices in Iowa.
- [Cloud Storage pricing](https://cloud.google.com/storage/pricing): regional
  Standard and Nearline storage, operations, retrieval, and network rates.
- [Cloud TPU pricing](https://cloud.google.com/tpu/pricing): TPU v5e on-demand
  pricing in `us-central1`. TPU prices are per chip-hour, not per VM-hour.
- [Vertex AI generative AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
  and [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing): Gemini
  2.5 Flash and Pro standard paid token rates.
- [Google Cloud Observability pricing](https://cloud.google.com/products/observability/pricing#logging-pricing-summary):
  Cloud Logging ingestion and the monthly free allotment.

Pricing pages are dynamic. Recheck region, currency, and the Default/on-demand
consumption model when entering the estimate. The estimate deliberately does
not assume Spot VMs or committed-use discounts: Spot capacity can be preempted,
and commitments create payment obligations even when research demand changes.
Either can extend the credits after workload stability is demonstrated.

## Professor-Facing Summary

> Based on current Google Cloud on-demand pricing and our measured ECHAM-RRTMGP
> workloads, we estimate approximately $21,000 of planned Year-1 cloud usage.
> Allowing for gradient-based optimization, higher-resolution experiments,
> ensembles, failed searches, and pricing uncertainty, I recommend requesting
> $35,000 in Google Cloud credits. This supports CPU development, approximately
> 1,600 A100 80-GB GPU-hours, a small TPU scaling pilot, data storage, and
> LLM-assisted autoresearch for a team of two to three researchers. We will use
> measured cost per accepted experiment to review and reallocate the reserve
> after the first 30 and 90 days.
