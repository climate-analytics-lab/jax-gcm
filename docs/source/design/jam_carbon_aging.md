# JAM carbonaceous ageing (pcarbon → accumulation)

**Issue:** [#721](https://github.com/climate-analytics-lab/jax-gcm/issues/721).
**Where it runs:** inside the mam4-jax core (`mam_pcarbon_aging_1subarea`
port), invoked by `Mam4JaxMicrophysics` after coagulation each step.

## Why the process is needed

JAM emits all BC and POA into the `primary_carbon` (pcm) mode, which is
`can_activate=False`: structurally excluded from ARG activation, from
stratiform and convective in-cloud scavenging, and from the explicit
cloud-borne phase. That exclusion is right for *fresh* hydrophobic carbon and
wrong for the same particle once a soluble coating has grown on it. Ageing is
the transfer that ends the exclusion — the only pathway by which carbon
reaches accumulation, where activation and wet removal can act on it. Without
it the exclusion is permanent, and BC lifetime is set by dry deposition and
sedimentation alone: ~21 days against an observed 5–8 d, with a
correspondingly high burden.

Both reference models age their carbon: MAM4 via
`mam_pcarbon_aging_1subarea` (E3SM `modal_aero_amicphys.F90:5111-5285`),
ECHAM-HAMMOZ via `m7_concoag`/`m7_coat` (`mo_ham_m7.f90:3532-3844`).

## Which reference formulation, and why

Both references implement the same physics — a particle counts as aged once
condensed + coagulated sulfate (plus SOA as hygroscopicity-equivalent "so4")
coats its surface with a threshold number of monolayers — but for different
mode structures. JAM's modes **are** MAM4's, and pcm → accumulation is the one
age-pair, so the MAM4 formulation ports verbatim. HAMMOZ's KI/AI/CI → soluble
transfers were considered and have no direct image here: they presuppose a
soluble/insoluble mode pairing JAM does not have. The HAMMOZ evidence is
retained for what it does establish — that the process is real, and where the
one free parameter plausibly sits.

The transferred fraction per step is

```
xferfrac = min( vol_shell · dgn · e^{2.5 ln²σ} / (6 · n · Δr · vol_core),  1 − 10ε )
```

with `Δr = 4.76e-10 m` (one bisulfate monolayer), `vol_shell` = so4 + soa
(soa scaled by κ_soa/κ_so4), `vol_core` = pom + bc + moa. Core species and
mode *number* move by `xferfrac`; shell species move wholesale.

## The monolayer threshold is a parameter, not a constant

`n` is exposed as `Mam4JaxMicrophysics(n_so4_monolayers=…)` because the
references legitimately disagree, though not along the line this page first
drew. E3SM's amicphys receives `n_so4_monolayers_pcage` from the namelist
(`namelist_defaults_eam.xml`, read 2026-10-02): **8.0** for the default
physics (the EAMv1/v2 lineage) and **3.0** for the CLUBB+P3 (v3) physics,
with 3.0 as the bare fallback; CESM's CAM hard-codes 8.0 in both
`modal_aero_gasaerexch.F90` and `modal_aero_coag.F90`, and Liu et al. (2016)
§2 states eight for MAM4 (the sensitivity set MAM4L1/L2/L4/L8, L8 the
production choice). ECHAM-HAM's `m7_coat` uses 1. So 8 is MAM4's published
and default value, 3 is E3SMv3's; the earlier reading that 8 "belongs to the
legacy coag path" was not right (see `jam_modal_microphysics.md`, §9
AER-06/AER-12). The spread matters because `n` sets carbonaceous lifetime
directly — smaller means faster ageing. Across the two MAM4 values, 90-day T63L47 `echam-jam-aerocom` runs
(area-weighted global means over days 45–90, τ = burden / (dry + wet
removal)) give BC τ ≈ 8.4 d and burden 0.22 mg/m² at `n = 8`, and τ ≈ 4.8 d
and 0.15 mg/m² at `n = 3`, with 56 % and 68 % of BC mass in accumulation
respectively. Both bracket the observed 5–8 d; a factor of ~2.7 in the
parameter moves lifetime by nearly a factor of two.

This adapter's default is **3.0** — E3SMv3's value, which also reproduces
the vendored Fortran captures; the in-tree modal core defaults to MAM4's 8.

Each term instance holds its own value as an **`nnx.Param` leaf**, a
differentiable parameter per the repo convention, and passes it to the core
per call as a **traced** `AmicphysParams` field. It is deliberately not routed
through the core's process-global config, which is read at trace time: that
would make several differently-configured instances in one process
order-dependent. Consequences of the traced form: `jax.grad` flows through the
criterion (piecewise — the gradient is exactly zero once the mode saturates),
a calibration sweep over the threshold reuses a single compiled step, and the
gas netprod rates ride the same pytree for future source calibration.

`n` does not subsume the `mdo_pcarbonaging` toggle. Zero monolayers is a
zero-thickness coating requirement — instant full ageing, not off — and no
finite threshold suppresses the wholesale shell-species transfer. The toggle
exists for parity against the `skip_pcarbon_aging` reference build; ageing is
on in every JAM run.

## Why in-core, not a harness term

The coating criterion needs the sulfate condensed onto pcm *within the step*,
which exists only inside `amicphys` — the harness never sees per-mode
condensation increments. It must also run **before** the core's state repack:
pcm has no so4/soa tracer slots, so shell mass condensed onto pcm has nowhere
to be written and would be dropped there, a per-step sulfur/SOA sink
proportional to pcm number. Running ageing first moves that mass to
accumulation, which does have the slots, so the sink cannot arise. A
harness-side approximation satisfies neither requirement.

## Requirements

The core is pinned to **mam4-jax 0.5.1**; version 0.4.0 first carried the
ageing port; `Mam4JaxMicrophysics` refuses to construct against a core
without it.
