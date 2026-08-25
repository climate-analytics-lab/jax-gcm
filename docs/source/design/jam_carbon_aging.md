# JAM carbonaceous ageing (pcarbon → accumulation)

**Issue:** [#721](https://github.com/climate-analytics-lab/jax-gcm/issues/721).
**Where it runs:** inside the mam4-jax core (`mam_pcarbon_aging_1subarea`
port), invoked by `Mam4JaxMicrophysics` after coagulation each step.

## The problem it solves

JAM emits 100 % of BC and POA into the `primary_carbon` (pcm) mode, which is
`can_activate=False` by design — structurally excluded from ARG activation,
stratiform and convective in-cloud scavenging, and the explicit cloud-borne
phase. That is correct for *fresh* hydrophobic carbon, but with no ageing
process the exclusion was permanent: the 2-year closed-budget run (PR #720)
measured a BC lifetime of ~21 days (observed 5–8 d) and a ~2.5× high burden.
Both reference models age their carbon: MAM4 via `mam_pcarbon_aging_1subarea`
(E3SM `modal_aero_amicphys.F90:5111-5285`), ECHAM-HAMMOZ via
`m7_concoag`/`m7_coat` (`mo_ham_m7.f90:3532-3844`).

## Which reference formulation, and why

Both references implement the same physics — a particle counts as aged once
condensed + coagulated sulfate (plus SOA as hygroscopicity-equivalent
"so4") coats its surface with a threshold number of monolayers — but for
different mode structures. JAM's modes **are** MAM4's (pcm → accumulation is
the one age-pair), so the MAM4 formulation ports verbatim while HAMMOZ's
KI/AI/CI→soluble transfers have no direct image here. The HAMMOZ evidence
establishes the process and brackets the one free parameter.

The transferred fraction per step is

```
xferfrac = min( vol_shell · dgn · e^{2.5 ln²σ} / (6 · n · Δr · vol_core),  1 − 10ε )
```

with `Δr = 4.76e-10 m` (one bisulfate monolayer), `vol_shell` = so4 + soa
(soa scaled by κ_soa/κ_so4), `vol_core` = pom + bc + moa. Core species and
mode *number* move by `xferfrac`; shell species move wholesale.

**The monolayer count `n` is exposed** (`Mam4JaxMicrophysics(n_so4_monolayers=…)`)
because the references legitimately disagree — the MAM4 *amicphys* path
receives 3 (via `phys_control`, the CAM5/ACME lineage; the oft-quoted 8 in
`modal_aero_gasaerexch.F90` belongs to E3SM's legacy `modal_aero_coag` ageing
path), and ECHAM-HAM uses 1 — and it directly sets the BC/POA lifetime
(smaller = faster ageing). Default: 3.0 (the amicphys reference value, which
also reproduces the vendored Fortran captures). Each term instance holds its own value and passes it to
the core **per call** as a static jit argument (never via the core's
process-global config, which is read at trace time and would make several
differently-configured instances in one process order-dependent). Static
means it is **not** a differentiable parameter; calibrating it means a
sweep — which the per-instance binding makes safe — not a gradient.

## Why in-core, not a harness term

The coating criterion needs the sulfate condensed onto pcm *within the
step*, which exists only inside `amicphys` (the harness never sees per-mode
condensation increments). It must also run **before** the core's state
repack: pcm has no so4/soa tracer slots, so condensed shell mass on pcm was
previously **silently dropped at repack** — a standing per-step sulfur/SOA
sink proportional to pcm number. Ageing transfers that mass to accumulation
(which has the slots) and thereby closes the leak by construction. A
harness-side approximation would have fixed neither.

## What changes for users

Ageing is ON by default in every JAM run using the `mam4_jax` core (there
is no Fortran toggle to mirror; the core's `mdo_pcarbonaging` exists for
fixture parity only). Expected effects: BC/POA lifetimes drop toward the
observed 5–8 d range, carbonaceous burdens fall accordingly, and SO4 gains
a small source from the closed repack leak. The mam4-jax pin also carries a
float32 fix without which coagulation moved **zero** mass between modes in
f32 runs (`betaij3` underflow) — see the mam4-jax PR for details.
