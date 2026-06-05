# The Jax Aerosol Module (JAM)

The `jcm.physics.aerosol.jam` package is a HAMMOZ-style, microphysics-agnostic
**process harness** for online aerosol, wrapping an interchangeable
**microphysics core**. It provides emissions, an aerosol microphysics core,
ARG cloud-droplet activation, gravitational sedimentation, and dry/wet
deposition, all as composable `PhysicsTerm`s. "HAMMOZ-style" is used here as a
familiar point of comparison — JAM is *inspired by* HAM and currently shares
several of its parameterizations, but the package name is deliberately
decoupled from any specific scheme so the implementations can be swapped
freely (like the other JCM physics packages). See issue #461.

## Design

JAM mirrors HAMMOZ's split between a microphysics-agnostic process layer and a
swappable microphysics core, expressed in JCM's composable architecture:

- **Population contract** (`population.py`): `ModalAerosolSpec` /
  `AerosolMode` / `AerosolSpecies` are pure-Python frozen dataclasses
  describing the *shape* of the population (classes, per-class species, σ_g,
  density, κ). `ModalAerosolSpec` is the modal realisation (each class a
  log-normal mode); a sectional realisation (each class a size bin) would be a
  sibling spec with the same role (#491). They are static config — never JAX
  pytree leaves — so class counts are known at compose time and no
  data-dependent shapes arise. Field names echo the AMBRS modal vocabulary so
  a future part2pop/AMBRS interop adapter is a thin mapping rather than a
  rewrite.
- **Microphysics core** (`microphysics/`): the single swap point. The default
  `PlaceholderMicrophysics` computes κ-Köhler equilibrium radii/density with
  **zero tendency** on the real MAM4 4-mode population (`mam4_data.py`,
  constants from E3SM `rad_constituents.F90` / `modal_aero_data.F90`,
  cross-checked against MAM4-JAX). Replacing it with the real MAM4-JAX core
  swaps only the per-step microphysics, not the contract (#490).
- **Tracers** (`tracer_layout.py`): aerosol mass and number are ordinary
  `state.tracers` entries (`m_`/`mc_`/`n_`/`nc_` for interstitial/cloud-borne
  mass/number, keyed by aerosol class), so the dynamical core transports them
  and existing diagnostics work. The cloud-borne mirror is carried for cores
  that prognose it (e.g. MAM4; currently inert).
- **Inter-term state** (`jam_state.py`): the core writes a typed
  `JamAerosolState` under the `_jam_state` diagnostic; activation, deposition
  and sedimentation read it. Its leading `n_aer` axis is a generic aerosol-
  *class* index (mode or bin), so the struct is representation-agnostic.

Every harness step is differentiable; scheme choice (placeholder vs real core,
ARG variant) is a compose-time Python decision with no traced branching.

## Term chain

`jam_aerosol_physics()` returns the ordered list:

1. `JamEmissions` — sea salt / dust / DMS / volcanic / biogenic sources →
   lowest-layer tracer tendencies. Emission is computed in gridpoint (nodal)
   space; the split of a source's mass into aerosol classes belongs to the
   microphysics core's population, so the harness works with any dycore (no
   modal representation is assumed on the dynamics side).
2. microphysics core (`PlaceholderMicrophysics`) — writes `_jam_state`.
3. `ArgActivation` — Abdul-Razzak & Ghan (2000); writes `activated_cdnc`
   (the same key the 2M SPA floor produces, so ARG and SPA are
   interchangeable, #342). Optional `ghosh2025` variant (Ghosh et al. 2025,
   GMD 18 4899; coefficients reconstructed from the paper's Table 3, gated
   off by default pending PDF verification).
4. `StokesSedimentation` — Stokes settling + donor-cell vertical transport.
5. `SlinnDryDeposition` — aerodynamic + Slinn & Slinn (1980) over-water
   resistances; reads `surface_friction_velocity` from the
   `vertical_diffusion` diagnostic (previous step).
6. `WetScavenging` — in-cloud nucleation + size-dependent below-cloud
   impaction scavenging, built from the cloud scheme's precip / condensate
   diagnostics.

In `echam_physics` the chain is *split*: terms 1–5 run in the pre-cloud
aerosol block (activation must precede the cloud microphysics term that
consumes `activated_cdnc`), while `WetScavenging` is placed immediately
**after** the cloud microphysics term so it scavenges against the current
step's precipitation and condensate rather than the previous step's.

## Usage

```python
from jcm.physics.echam.echam_terms import echam_physics

physics = echam_physics(
    aerosol_module="jam",      # default "macv2sp"
    cloud_scheme="2m",         # ARG activated_cdnc feeds the 2M scheme
    jam_microphysics="placeholder",
    jam_arg_variant="arg2000", # or "ghosh2025"
)
```

For now the JAM path **augments** MACv2-SP rather than replacing it: MACv2-SP
is kept for the aerosol radiative optics and Twomey factor that radiation and
the cloud schemes currently read — a temporary fudge in lieu of proper JAM
aerosol↔radiation and aerosol↔microphysics coupling (#495). Once JAM supplies
those, MACv2-SP need not be included.

The 2M scheme uses ARG's `activated_cdnc` where it is non-empty and falls back
to the MACv2-SP SPA floor wherever the online source is ≈0 (e.g. before the
prognostic JAM tracers spin up from a zero-seeded initial state), so the
default JAM+2M run always activates droplets.

## Status and caveats

- **Source magnitudes** (emissions) are order-of-magnitude defaults, not
  inventory-calibrated. Prescribed CEDS emissions with HAMMOZ-grounded,
  differentiable per-sector characteristics are tracked in #498.
- **Wet scavenging** currently reconstructs the per-level precip-formation
  rate from column precip; exposing the true per-level formation/evaporation
  rates from the cloud schemes and adding re-evaporation re-injection is
  tracked in #499.
- **Real MAM4-JAX core** (#490) is not yet differentiable upstream; the
  harness itself is fully differentiable.

Out of scope, tracked separately: sectional/bulk families (#491), part2pop
diagnostics adapter (#492), SOA volatility basis (#493), heterogeneous
freezing (#494), aerosol optics in radiation (#495), gas-phase chemistry
coupling (#496).
