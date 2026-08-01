# pySES + JAM performance review — measurements, bottlenecks, and 4-GPU options

*July 2026, derecho A100 campaign (branch `feat/pyses-jam-forcing`). All
measurements at ne30 L47 (21,600 pg2 columns), dt = 900 s, hybrid coupling,
quasi-uniform hyperviscosity, tracer-split 5, RRTMGP + Hines.*

## 1. Measurements

| configuration | 1× A100-40GB | 4× A100 (element-sharded) |
|---|---|---|
| 2m clouds, no JAM | **21.1** sim days/hr | 6.7 (**3.1× slower**) |
| 2m + JAM (mam4_jax, online emissions) | **6.3** | 4.8 (1.3× slower) |
| peak device memory (2m-jam) | 35.4 GiB | 10.9 GiB/GPU |

Two facts fall out immediately:

1. **JAM physics is a 3.3× single-GPU multiplier** (21.1 → 6.3). The dycore —
   including transporting all ~90 JAM tracers — is *not* the single-GPU cost
   center.
2. **Element-sharding one ne30 simulation over 4 GPUs is a net loss for
   every configuration** — the absolute 4-GPU penalty (+3.8 s/step for 2m,
   +1.9 s/step for 2m-jam, partially hidden under physics) is config-independent,
   i.e. it lives in the dynamics, not the physics. The sharding itself is
   *correct* (0 NaN, identical statistics, memory scales) — it is the latency
   that kills it.

## 2. Why strong scaling fails at ne30: the collective census

From the pyses source (`_config.py`, `operations_2d/local_assembly.py`,
`cam_se/time_stepping.py`, `tracer_transport/eulerian_spectral.py`):

- The active multi-device DSS is a **GSPMD assembly gather**
  (`arr.at[...].get(out_sharding=...)` + two resharding reshapes) per
  projected field, synchronous, no compute overlap. A shard_map +
  `lax.ppermute` neighbor-exchange DSS (`dss_ppermute`, edge-colored
  schedule) **exists in pyses but is never called**.
- Dynamics: RK3-Ullrich runs 5 tendency evaluations per substep, each ending
  in `project_dynamics` = 4 field-DSS (u, v, thermo, d_mass) → **20 DSS per
  dynamics substep**; hyperviscosity adds 8, the sponge 4.
- Tracers are the *good news*: transport stacks all ~90 tracers into one
  array and pays ~**11 batched DSS per tracer subcycle** total (≈ 55/step) —
  per-tracer exchanges would have been ~90× worse. The design intent
  ("one batched DSS ... rather than n_tracers skinny ones") is already there.
- Net: at ~45 dynamics substeps/step, **≈ 900 (RK) + ≈ 540 (hypervis+sponge)
  + ≈ 55 (tracers) ≈ 1,000–1,500 synchronous cross-device exchanges per
  900 s step**, each ~1–3 XLA collectives. At NVLink round-trip + kernel
  launch latencies of O(1 ms), that is O(seconds) per step — matching the
  observed +2–4 s/step penalty. Bandwidth is irrelevant; it is pure latency
  serialization, which is why 5,400 columns/GPU (under-occupied A100s)
  cannot hide it.

## 3. Where the single-GPU time goes (2m-jam)

From the jcm chain (13 JAM terms added over plain 2m):

| cost driver | structure | cadence |
|---|---|---|
| `Mam4JaxMicrophysics` | full amicphys box model vmapped over **~1.0 M cells** (47×21,600); gasaerexch→rename→newnuc→coag + length-4 SOA scan; **float64 by default** (jcm's f32 tracers upcast at the boundary) | every step |
| `JamOpticsTerm` | (14 SW + 16 LW) bands × 4 modes × species Mie-LUT interpolations, ragged Python mode/species loops inside a band vmap | every step |
| `RRTMGPRadiation` | per-column vmap over 21,600 cols; internally float32 | every **8th** step, **×2** (`compute_cre=True` runs a second full clear-sky solve) |
| 10 other JAM terms (emissions, gas/aqueous chem, activation, sedimentation, wet/dry dep) | batched/elementwise; wetdep+sedi vmap over stacked tracers | every step |

`checkpoint_terms=True` wraps each term in `jax.checkpoint` — **forward-cost
neutral** (it only adds recompute in reverse-mode), so it does not affect these
throughput numbers.

## 4. Options

### A. Making the most of a 4-GPU node

**A1. Instance parallelism — run 4 independent simulations, one per GPU.**
Zero code: four job chains with `CUDA_VISIBLE_DEVICES` pinned 0–3 on one node
(or four 1-GPU jobs). 4.0× node throughput today. This is the correct mode for
climatology production, spin-up ensembles, perturbed-parameter/emission
scenarios, and seasonal restart farms. *Recommended immediately; a
`derecho_pyses_jam_4x1gpu.pbs` template is trivial.*

**A2. Batched ensemble in one process (member-axis sharding).**
vmap the model step over a stacked member axis and shard *that* axis across
the 4 GPUs: members are embarrassingly parallel, so zero cross-GPU collectives,
one compile, one job, in-run ensemble statistics. Same 4× as A1 with nicer
operations (and gradient-through-ensemble for calibration work). Moderate
wrapper effort in `Model`/driver; the physics/dycore are untouched (per-member
state is just a leading axis). Worth doing when ensembles become routine.

**A3. Cut the DSS latency (upstream pyses work) — the real strong-scaling fix.**
Three compounding levers, in order of leverage:
   1. **Batch `project_dynamics`**: stack (u, v, thermo, d_mass) into one
      array before projection — 4 exchanges → 1 (the tracer path already
      proves the pattern). ≈ 900 → 225 exchanges/step for ~30 lines.
   2. **Activate the dormant `dss_ppermute`** neighbor exchange in place of
      the GSPMD assembly gather — neighbor ppermute on an edge-colored
      schedule has far lower latency than a global gather + 2 reshards, and
      the machinery (comm map, shard_map wrapper) is already written and
      tested upstream.
   3. **Fuse hypervis/sponge projections** into the RK stage projections
      where algebraically valid.
   Realistic outcome: ~4–8× fewer, cheaper collectives — likely bringing
   4-GPU ne30 to parity-or-better, and making ne60+ scale properly. This is
   a pyses PR, not a jcm one.

**A4. Higher resolution.** At ne60 (86,400 columns) per-GPU occupancy
quadruples and compute grows ~8× vs unchanged collective *count* — strong
scaling should cross over even without A3. No action until science needs it.

**A5. Not recommended:** comm/compute overlap (XLA gives little control),
level-axis sharding (vertically coupled physics), radiation-on-its-own-GPU
(architecturally large; revisit only if radiation cadence is ever per-step).

### B. Single-GPU levers (multiply with A1/A2 across the node)

**B1. MAM4 in float32 via a scoped x64-off context — the big one.**
The dominant term runs float64 over 1M cells today. The casper campaign
validated the f32 core (MAM4-JAX #60: substep/coag backends f32-safe) but used
the *global* `MAM4_JAX_ENABLE_X64=0` flag — under pySES that global flip would
break the f64 dynamics. The fix is the same pattern as commit `27bb36f` for
RRTMGP: cast at the wrapper boundary + `jax.enable_x64(False)` scoped around
the core call. Plausible ~1.5–2× on the dominant term (A100 f64 FLOPs are
1/2 f32 even on tensor-free paths, and bandwidth halves). *Highest
value-per-effort of everything here; validate against a 2-day golden run.*

**B2. Gate `JamOpticsTerm` to radiation steps.** Aerosol optics is consumed
by RRTMGP, which runs every 8th step — computing 30-band Mie optics on the
other 7 steps is discarded work. Cache per-band optics in the physics carry
exactly like the radiation cache. ~8× reduction on the second-largest JAM
term.

**B3. `compute_cre=False` for production.** Halves the RRTMGP work on compute
steps; CRE is a diagnostic — keep it for analysis runs only. One flag.

**B4. Radiation interval 2 h → 3 h.** Science trade-off; standard in ECHAM
lineage. Only if B1–B3 are insufficient.

### C. First step before investing in B: one profiled run

A single `jax.profiler` trace of ~4 steps (1-GPU 2m-jam) would confirm the
MAM4 : optics : rest split this review infers structurally. Cheap insurance
(~30 min GPU) before spending effort on B1/B2.

## 5. Verification round (measured, ne30 2m-jam, 1× A100 unless noted)

The B-levers and the collective work were implemented and **measured against
the compiled XLA graph** (jobs `verify_levers_6802639`, `pyses_head_bench_6802910`):

| change | steady sim days/hr | verdict |
|---|---|---|
| baseline (wheel 0.1.3a2, f64 core, CRE, ungated optics) | 6.4 | — |
| + f32 MAM4 core (B1) | 6.4 | **engaged but ~neutral** (−28 ms/step) |
| + optics gate (B2) | 6.5 | −74 ms on non-radiation steps |
| + no-CRE (B3) | 6.7 | −1.05 s on radiation steps only |
| **pyses HEAD** (kernel fusions) + levers | **9.2** | **+40 % — the real single-GPU lever** |
| pyses HEAD + batched projections + live ppermute DSS, 4 GPU | 5.4 | up from 4.8, still < 1 GPU |

Why the physics levers were neutral, from the graph itself: the step
executable is an ~87 MB HLO module of ~230 k ops whose f64 census barely
moved with the f32 core (235 k → 227 k tensor refs — MAM4 is ~3 % of the
graph), and the profiler trace shows **~20,400 kernels/step with device-busy
time ≈ wall time** (5.49 s): the GPU is compute-saturated by **f64
tracer-transport work** (scatter/gather fusions ~0.55 s, f64 GEMM
derivative contractions ~0.6 s, limiter select/compare fusions ~0.9 s, and a
~2.9 s tail of per-tracer pipeline kernels). §3's structural inference
(MAM4 dominant) was wrong; the measurement stands corrected: **the 2m→jam
3.3× is the ~95-tracer transport in the f64 dynamics**, exactly the kernels
upstream's fusion commits attack (Zerroukat-remap vectorization → the
scatters/selects; GLL-contraction unroll → the d884 GEMMs).

4-GPU remains below the crossover at ne30 even with one lane-aware neighbor
exchange per projection site — consistent with upstream's own NX30 2-GPU
62 % efficiency. **2 GPUs are worse than 4** (4.6 vs 5.4): fitting
``wall = compute/N + overhead`` gives a ~570–585 s/day fixed sharding tax
(~6 s/step) nearly independent of device count — the tax, not the split,
is the problem. Instance parallelism (A1) stays the node-throughput answer.

**CPU node** (128-core Milan, same stack): 1 dev × 128 threads ≈ 0.2 sim
days/hr; 8 dev × 16 ≈ 0.3 (best); 120 × 1 worse than the measurement
budget. 30–45× slower than one A100 — usable only as debug capacity.

**Flagged follow-up (science decision, not tonight)**: pyses supports f32
dynamics (`PYSES_USE_DOUBLE=0`, benchmarked upstream at ~2× on NX15). jcm
runs the CAM-SE core f64 by choice; an f32-dynamics jam experiment is the
one remaining ~2× single-GPU lever, gated on thin-lid stability validation.

## 6. Recommended sequence

1. **A1 now** (zero code): 4× node throughput for the ensemble/production use
   case; the running 1-year chain already occupies only 1 of 4 GPUs on its
   node-share.
2. **C then B1**: profile, then scoped-f32 MAM4 — compounds with A1
   (4 × ~1.5–2× ≈ 6–8× node throughput for JAM production).
3. **B2 + B3**: optics gating and CRE flag — small diffs, mechanical wins.
4. **A3 upstream**: batched `project_dynamics` + `dss_ppermute` activation in
   pyses — the durable fix that makes multi-GPU-per-simulation viable and
   ne60+ practical.
5. **A2** when ensemble workflows become routine.

## Where the verification lives now

The one-off scripts used for this review (`verify_perf_levers.py`,
`probe_pyses_sharding.py`) are retired from the repo. Their durable
assertions moved into unit tests:

- optics gating replays cached per-band fields between radiation steps —
  ``optics_term_test.py::test_radiation_gate_replays_cache_between_compute_steps``
- ``compute_cre=False`` drops the clear-sky RRTMGP call —
  covered in ``rrtmgp_test.py`` (term- and scheme-level)

The sharding measurements (GSPMD gather DSS cost, explicit-mesh
requirements, per-step sharding tax) are recorded in the tables above;
re-measure with a scaled-down production run rather than a bespoke probe.
