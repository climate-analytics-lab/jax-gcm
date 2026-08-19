# ECHAM T63L47 Autoresearch Workflow Cost

## Question

What is the compute cost of reproducing the prescribed-state cloud workflow
with the full ECHAM physics stack, one-moment cloud microphysics, and RRTMGP on
the production T63L47 grid?

This is a short forward-only timing experiment. It is intended to determine
where ECHAM/RRTMGP can sit in an autoresearch evaluation ladder, not to measure
forecast skill or produce publication-quality performance numbers.

## Method

`benchmark_echam_workflow_cost.py` places a reusable JIT around one independent
full-grid physics call and forces all output arrays to materialize. It reports
setup, compile-plus-first-call, and repeated steady-state wall time.

Three configurations were compared:

| Configuration | Grid | State | Physics |
| --- | --- | --- | --- |
| SPEEDY reference | T31L8, 96 x 48 | deterministic plausible state | full SPEEDY physics |
| ECHAM-grey | T63L47, 192 x 96 | bundled spun-up state with `qc` and `qi` | full ECHAM 1M stack with grey radiation |
| ECHAM-RRTMGP | T63L47, 192 x 96 | same spun-up state | full ECHAM 1M stack with RRTMGP |

ECHAM uses bundled real T63 terrain and forcing. The RRTMGP comparison uses the
installed g128 longwave/g112 shortwave coefficients. The primary run disables
cloud-radiative-effect diagnostics (`compute_cre=False`) because the existing
workflow needs all-sky RSUT, not an additional clear-sky solve. A second run
measures the constructor default, `compute_cre=True`.

All reported comparison runs used one visible NVIDIA RTX PRO 6000 Blackwell GPU,
JAX 0.10.2, Python 3.12.12, float32, and forward-only evaluation. The GPU was
shared: utilization was 56% when selected and varied during the runs. Medians
therefore represent realistic shared-machine cost, not isolated peak hardware
performance.

## Results

| Configuration | 3D cells | Compile + first call | Steady median | Observed range | 960-state serial estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPEEDY T31L8 | 36,864 | 1.78 s | 0.00067 s | 0.00066-0.00086 s | 0.65 s |
| ECHAM 1M + grey T63L47 | 866,304 | 7.13 s | 0.0291 s | 0.0191-0.0340 s | 27.9 s |
| ECHAM 1M + RRTMGP, no CRE | 866,304 | 16.27 s | 2.054 s | 1.502-2.159 s | 32.9 min |
| ECHAM 1M + RRTMGP, with CRE | 866,304 | 23.19 s | 3.573 s | 3.014-3.807 s | 57.2 min |

All output leaves were finite. Materialized outputs were approximately 3.5 MB
for SPEEDY, 192 MB for ECHAM-grey, and 494 MB for ECHAM-RRTMGP.

The direct wall-time ratios are:

- ECHAM-grey versus SPEEDY: 43x.
- ECHAM-RRTMGP without CRE versus SPEEDY: 3,051x.
- ECHAM-RRTMGP without CRE versus ECHAM-grey: 70.7x.
- Enabling the additional clear-sky CRE solve: 1.74x.

T63L47 contains 23.5 times as many three-dimensional cells as T31L8. After
normalizing by cell count, ECHAM-grey is about 1.84 times slower per cell than
SPEEDY, while ECHAM-RRTMGP is about 130 times slower per cell. RRTMGP spectral
gas-optics and radiative-transfer work, rather than one-moment microphysics, is
the dominant cost.

## Workflow Implications

The full 240-window ERA5 benchmark contains 960 prescribed states. For one
scheme, the measured forward physics cost is approximately 33 minutes without
CRE or 57 minutes with CRE, before atmospheric remapping and metric I/O. A
three-scheme comparison would therefore require roughly 1.6 GPU-hours without
CRE or 2.9 GPU-hours with CRE under the serial assumption.

These estimates do not include ERA5-to-T63L47 horizontal and hybrid-pressure
remapping. The earlier SPEEDY workflow showed that repeated state preparation
can dominate total runtime. A reusable, labelled T63L47 prescribed-state cache
is therefore a prerequisite rather than an optional optimization.

RRTMGP's normal two-hour radiation cache does not help independent
prescribed-state evaluation: each state has no previous radiation diagnostic and
must execute a full radiation call. Batching many full T63 states is also limited
by memory, since one materialized result is already about 494 MB and RRTMGP has
much larger live intermediates than its final output.

The cost is still practical as an outer evaluation gate. It is not practical as
the inner objective for every symbolic mutation or broad hyperparameter trial.

## Continuous Evolution Benchmark

A second benchmark used `Model.run` followed by continuous `Model.resume`
cycles, so it includes the dynamical core, physics, tendency application, state
updates, diffusion, persistent cloud/radiation carry, and one end-of-cycle
snapshot. It was rerun on an idle NVIDIA A100 PCIe 40 GB GPU with JAX 0.11.1.

SPEEDY used ten 30-minute steps per five-hour block. ECHAM used ten 12-minute
steps per two-hour block, exactly matching its radiation interval: one full
RRTMGP solve followed by nine steps that reuse the stored radiative heating.
Ten post-warm-up blocks were timed for each configuration.

| Configuration | Steady block median | Effective step | Estimated wall time per simulated day |
| --- | ---: | ---: | ---: |
| SPEEDY T31L8 | 0.02824 s per 5 h | 0.00282 s | **0.136 s** |
| ECHAM 1M + RRTMGP, no CRE | 1.70889 s per 2 h | 0.17089 s | 20.51 s |
| ECHAM 1M + RRTMGP, with CRE | 3.03328 s per 2 h | 0.30333 s | 36.40 s |

The ECHAM radiation carry advanced from step 10 after the compile cycle, to 20
after the resume warm-up, to 120 after the ten measured cycles. This verifies
that the expected radiation cadence was executed rather than inferred only from
wall time. All final dycore, physics-carry, and diagnostic leaves were finite.

Normalized by simulated time, ECHAM without CRE is 151 times slower than SPEEDY;
with CRE it is 269 times slower. The extra clear-sky solve increases ECHAM cost
by 1.775 times. Approximate 30-day costs on this idle A100 are 4.1 seconds for
SPEEDY, 10.3 minutes for ECHAM without CRE, and 18.2 minutes for ECHAM with CRE.

These per-day values extrapolate blocks with different snapshot cadence: one
snapshot per five simulated hours for SPEEDY and per two hours for ECHAM. That
matches each ten-step timing block but is not a strictly identical output-I/O
schedule. Device-side snapshot materialization is included; host serialization
is not. RRTMGP dominates ECHAM sufficiently that this difference does not alter
the workflow conclusion, but a publication benchmark should use equal daily
output cadence.

Evolution artifacts are:

- `outputs/evolution_cost_speedy_t31l8_test_run.json`
- `outputs/evolution_cost_echam_rrtmgp_t63l47_test_run.json`
- `outputs/evolution_cost_echam_rrtmgp_cre_t63l47_test_run.json`

## Recommended Staged Workflow

1. Cache ARM and ERA5 states after mapping to the T63L47 hybrid grid.
2. Cache model-available ECHAM diagnostics and observational targets once.
3. Run symbolic or neural searches offline against those cached feature tables.
4. Use ECHAM-grey or selected radiation proxies for broad online screening.
5. Run full RRTMGP without CRE only for finalists and ablations requiring RSUT.
6. Enable CRE only when clear-sky/all-sky decomposition is itself an evaluation
   target.
7. Run short forecasts and coupled tests only after a candidate passes the
   prescribed-state cloud and radiation gates.

This structure preserves the more physically complete ECHAM target while
keeping the autoresearch search loop computationally tractable.

## Scientific Caveats

- This benchmark times a bundled spun-up ECHAM state, not remapped ERA5 or ARM
  states. Runtime should transfer reasonably, but scientific initialization of
  model-consistent `qc` and `qi` remains unresolved for observations.
- SPEEDY and ECHAM use different grids and states. The comparison answers the
  cost of the proposed workflows, not an isolated microphysics-kernel ratio.
- The ECHAM factory includes the complete physics composition, not only 1M
  microphysics and radiation.
- Timings were collected on a shared GPU and should be rerun on an idle device
  before publication or hardware procurement decisions.
- The estimate is forward-only. Gradient-based calibration through RRTMGP has a
  substantially larger memory requirement and needs explicit column chunking.

## Artifacts

- Harness: `benchmark_echam_workflow_cost.py`
- SPEEDY: `outputs/workflow_cost_speedy_t31l8_gpu6.json`
- ECHAM-grey: `outputs/workflow_cost_echam_grey_t63l47_gpu6.json`
- ECHAM-RRTMGP: `outputs/workflow_cost_echam_rrtmgp_t63l47_gpu6.json`
- ECHAM-RRTMGP with CRE:
  `outputs/workflow_cost_echam_rrtmgp_cre_t63l47_gpu6.json`
