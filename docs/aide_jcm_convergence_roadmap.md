# JCM ↔ AIDE convergence roadmap

**Status:** proposal, 2026-08-11.
**Scope:** phased development plan for `climate-analytics-lab/jax-gcm` (JCM) and
`reflective-org/aide_sai_core` (AIDE, with `reflective-org/tomas-jax`), starting
from separate tracks and converging on a shared-library end state ("Option D"),
with AIDE persisting long-term as a research harness for plugins and wrappers
that don't belong in JCM.

This follows a full comparative read of both codebases (AIDE @ 7b837a7,
JCM `dev` @ f47ff5a including the JAM aerosol harness). The analysis it rests
on: both projects already share `jax-rrtmgp` and feed it aerosol through the
identical `aerosol_optics_sw/lw` kwargs; JAM was explicitly architected for
interchangeable microphysics cores with a sectional family anticipated (#491);
AIDE's kernels are pure JAX but its orchestration (env-vars-at-import,
monkeypatch composition, host-side numpy loop with per-hour netCDF reads) is
non-differentiable and non-importable as a library.

## End state (the target picture)

- **JCM is the library**: dycore protocol (prognostic, prescribed-met, learned),
  the JAM process harness with swappable aerosol cores (placeholder / MAM4-JAX /
  TOMAS-JAX / trained emulators), radiation backends (grey / RRTMGP / NN),
  calibration machinery.
- **AIDE is the research harness**: campaign configs, host-model readers
  (CESM h1, …), experimental plugins and wrappers, emulator training and
  validation studies — an incubator whose stable pieces graduate into JCM or
  tomas-jax.
- **tomas-jax stays a separate installable package** (LGPL-3.0), consumed by
  JCM through the same lazy-load pattern already used for GPL-3.0 mam4-jax.
- **Physics lives in JCM or tomas-jax, never in AIDE** except while
  experimental. One implementation of every seam.

Each convergence seam is also an **emulator swap point** — this is what makes
the roadmap serve the linked-emulator goal rather than compete with it:

| component | physical backend | emulator backend | seam |
|---|---|---|---|
| dynamics | Dinosaur (or prescribed CESM met) | learned dycore | `DynamicalCore` protocol |
| aerosol microphysics | TOMAS-JAX / MAM4-JAX | trained NN core | JAM microphysics-core contract |
| radiation | jax-rrtmgp | GRU emulator (exists) | radiation scheme signature + `aerosol_optics` kwargs |

---

## Phase 0 — hygiene and contracts (weeks; unblocks everything)

Parallel, low-risk, mostly each org in its own tree.

**JCM**
- Merge JAM to `dev`; land the in-flight physics-review PRs (PR1/PR2 branches).
- Re-verify the MACv2-SP review findings against post-JAM `dev`
  (`macv2_sp.py` was heavily reworked; some findings may already be fixed).

**Shared (`jax-rrtmgp`)**
- Make it pip-installable with version tags and CI.
- Write down the `aerosol_optics_sw/lw = {optical_depth, ssa,
  asymmetry_factor}` kwarg contract as the stable seam, with a contract test
  both downstreams run against pinned versions.

**AIDE / tomas-jax**
- Fix tomas-jax license metadata (`pyproject.toml` says MIT; LICENSE/README say
  LGPL-3.0 — decide and make them agree).
- Merge/publish the `gpu-fast` branch (`tomas_jax.fast`) into tomas-jax main;
  it is currently AIDE's production engine but lives on an unmerged branch.
- AIDE replaces the `sys.path` sibling-repo hacks with pip installs of
  jax-rrtmgp and tomas-jax.

**Gate:** both models consume the same pinned, installable jax-rrtmgp; the
optics contract test passes in both CIs.

---

## Phase 1 — interface convergence (1–2 months)

**JCM**
- Implement the sectional population family (#491): `SectionalAerosolSpec` as
  the sibling of `ModalAerosolSpec`, exposing the same geometry interface so
  the harness terms (emissions, sedimentation, drydep, wetdep, activation,
  optics) stay family-invariant as designed.
- Extend the microphysics-core contract (`ModalMicrophysicsTerm` →
  a family-neutral base) to cover per-bin number+mass tracer layouts.
- Port AIDE's composition-resolved sulfate optics into JAM: Tabazadeh (1997)
  equilibrium wt%, Palmer & Williams RI at tabulated solution strengths,
  wt%-interpolated Mie tables. Benefits JCM's own stratospheric-aerosol
  fidelity independent of any merge (AIDE measured ~47% 550 nm extinction
  error from dry-size/fixed-composition optics).

**AIDE**
- Untangle config from import: env vars resolved once at the entry point into
  a config object passed down (keep the env interface as a shim; the
  self-describing run-header discipline is worth keeping verbatim).
- Move the global `jax_enable_x64` flag from `settling.py` import time to the
  entry points.
- Extract the CESM h1 reader into a standalone module with a documented
  interface (it becomes the first "host-model reader" plugin).

**tomas-jax**
- Adopt `wet_size` / `tang_density` upstream (currently duplicated in AIDE's
  `settling.py` with a documented workaround for the empty-bin density guard —
  fix the guard upstream instead).

**Gate:** TOMAS-JAX runs as a JAM core in a JCM single-column / prescribed-state
test, with a first performance number for the coupling budget.

---

## Phase 2 — components cross the boundary (2–4 months)

**Into JCM**
- `TomasJaxMicrophysics` core, lazy-loaded like mam4-jax (LGPL boundary
  respected). Decisions to make explicitly at the wrapper: f64↔f32 casting
  policy; substep count per physics dt; ICOMP=2 (sulfate-only fast engine)
  vs full-species core.
- Optional implicit (backward-Euler upwind) solver in JAM sedimentation for
  long-substep configurations, following AIDE's `settle_step`.
- AIDE's budget-audit discipline as a JCM diagnostic utility: staged
  per-process burden closure ("if a change breaks that closure, the change is
  wrong") — valuable for every JCM tracer, not just aerosol.

**Into AIDE (as plugins/deps, replacing local code)**
- `jax_solar` for solar geometry (replaces the hand-rolled zenith formula).
- JAM wet scavenging — closes AIDE's own standing caveat ("no wet removal
  anywhere; settling and transport out of the band are the only sinks").
- JAM's differentiable Gaussian injection profile — replaces the
  zero-gradient nearest-level snap (`INJ_HPA`), directly serving SAI
  injection-height calibration.

**Gate:** A/B validation of the TOMAS-in-JAM core against AIDE's box/production
references (sulfur closure, size-distribution error bounds from their
`docs/gpu_fast.md` calibration tables).

---

## Phase 3 — PrescribedMetDycore: AIDE as a JCM configuration (3–6 months)

The structural unification step. AIDE's mode of operation — evolve tracers on
prescribed 3-D meteorology — is a missing middle in JCM (between
`prescribed_state_model`, which has no tracer evolution, and the full GCM).

**JCM**
- A `PrescribedMetDycore` backend implementing the `DynamicalCore` protocol:
  - pluggable offline met source (CESM h1 first; the Phase-1 reader);
  - tracer transport by AIDE's Lin-Rood flux-form scheme (`fct_lr`), ported as
    the backend's advection operator. This is the right operator to own
    regardless: conservative to roundoff for winds that don't satisfy discrete
    continuity, which AIDE's docs correctly note "matters MORE for emulator
    winds". It also sidesteps the open question of spectral advection of ~82
    positive-definite sectional tracers through Dinosaur;
  - subdomain/band support (e.g. 1–150 hPa) and polar-cap handling;
  - `u/v/T` overwritten from files each step; anomaly-mode `dT_rad` carried by
    a physics term (mirrors AIDE's `RAD_MODE=anomaly`).
- Config for the SAI setup: `jam_aerosol_physics(microphysics="tomas_jax")`
  + RRTMGP + injection, on the prescribed dycore.

**AIDE**
- Reproduce the production scenario (10 Tg/yr equatorial ring, 90 d) through
  the JCM path; side-by-side vs the legacy driver on burdens, AOD550, ARF
  within documented tolerances.
- Freeze the legacy driver (tag) for reproducing published runs; new science
  moves to the JCM path.

**Gate (the payoff demo):** a differentiable SAI experiment neither codebase
can do today — gradients of AOD/forcing metrics w.r.t. injection height, rate,
nucleation scale, accommodation coefficient, through the full coupled band
model.

---

## Phase 4 — Option D steady state (6–12 months)

- JCM releases carry the prescribed dycore, sectional core support, and the
  calibration workflow; AIDE's repo shrinks to readers + campaign configs +
  experimental plugins + analysis.
- Graduation rule, to keep the boundary honest: a component moves from AIDE to
  JCM (or tomas-jax) when it has tests, a stable interface, and a second use
  case; it stays in AIDE while it is a single-campaign instrument.
- Emulator program runs on the stabilized seams: train aerosol/radiation
  emulators in AIDE against the JAM-core and radiation contracts, deploy them
  in JCM by construction. Speed-vs-fidelity sweeps (the original AIDE goal)
  become configuration matrices, not code forks.

---

## Standing risks and decision points

| risk | phase | mitigation / decision |
|---|---|---|
| Spectral advection of sectional tracers (Gibbs → negative number) | 1–2 | Prescribed/flux-form path first; decide whether prognostic-dynamics sectional runs wait for a finite-volume dycore backend |
| TOMAS f64 vs JCM f32 | 2 | Explicit casting policy at the core wrapper; measure accuracy cost |
| Microphysics cost in a GCM loop (94% of AIDE runtime) | 1 gate | Early perf number; chunking/stiffness-sort strategies port from `tomas_jax.fast` |
| AIDE campaign reproducibility | 3 | Legacy driver frozen and tagged; published runs never re-run through new code silently |
| Cross-org drift | all | Pinned versions + contract tests in both CIs; physics-single-home rule |
| tomas-jax license metadata inconsistency | 0 | Resolve before anything depends on it contractually |
