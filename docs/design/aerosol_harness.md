# Online aerosol harness (HAM)

The `jcm.physics.aerosol.ham` package is a HAMMOZ-style, microphysics-agnostic
**process harness** for online aerosol, wrapping an interchangeable modal
**microphysics core**. It provides emissions, an aerosol microphysics core,
ARG cloud-droplet activation, gravitational sedimentation, and dry/wet
deposition, all as composable `PhysicsTerm`s. See issue #461.

## Design

The harness mirrors HAMMOZ's split between a microphysics-agnostic process
layer and a swappable microphysics core, expressed in JCM's composable
architecture:

- **Population contract** (`population.py`): `ModalAerosolSpec` /
  `AerosolMode` / `AerosolSpecies` are pure-Python frozen dataclasses
  describing the *shape* of the population (modes, per-mode species, σ_g,
  density, κ). They are static config — never JAX pytree leaves — so class
  counts are known at compose time and no data-dependent shapes arise. Field
  names echo the AMBRS modal vocabulary so a future part2pop/AMBRS interop
  adapter is a thin mapping rather than a rewrite.
- **Microphysics core** (`microphysics/`): the single swap point. The default
  `PlaceholderMicrophysics` computes κ-Köhler equilibrium radii/density with
  **zero tendency** on the real MAM4 4-mode population (`mam4_data.py`,
  constants from E3SM `rad_constituents.F90` / `modal_aero_data.F90`,
  cross-checked against MAM4-JAX). Replacing it with the real MAM4-JAX core
  swaps only the per-step microphysics, not the contract (#490).
- **Tracers** (`tracer_layout.py`): aerosol mass and number are ordinary
  `state.tracers` entries (`m_`/`mc_`/`n_`/`nc_` for interstitial/cloud-borne
  mass/number), so the dynamical core transports them and existing
  diagnostics work. The cloud-borne mirror is carried for the future MAM4
  core (currently inert).
- **Inter-term state** (`ham_state.py`): the core writes a typed
  `HamAerosolState` under the `_ham_state` diagnostic; activation, deposition
  and sedimentation read it.

Every harness step is differentiable; scheme choice (placeholder vs real core,
ARG variant) is a compose-time Python decision with no traced branching.

## Term chain

`ham_aerosol_physics()` returns the ordered list:

1. `HamEmissions` — sea salt / dust / DMS / volcanic / biogenic surface
   sources → lowest-layer modal tracers (mass + implied number).
2. microphysics core (`PlaceholderMicrophysics`) — writes `_ham_state`.
3. `ArgActivation` — Abdul-Razzak & Ghan (2000); writes `activated_cdnc`
   (the same key the 2M SPA floor produces, so ARG and SPA are
   interchangeable, #342). Optional `ghosh2025` variant (Ghosh et al. 2025,
   GMD 18 4899; coefficients reconstructed from the paper's Table 3, gated
   off by default pending PDF verification).
4. `HamSedimentation` — Stokes settling + donor-cell vertical transport.
5. `HamDryDeposition` — aerodynamic + Slinn & Slinn (1980) over-water
   resistances; reads `surface_friction_velocity` from the
   `vertical_diffusion` diagnostic (previous step).
6. `HamWetDeposition` — in-cloud nucleation + size-dependent below-cloud
   impaction scavenging, built from the cloud scheme's existing precip /
   condensate diagnostics.

## Usage

```python
from jcm.physics.echam.echam_terms import echam_physics

physics = echam_physics(
    aerosol_module="ham",      # default "macv2sp"
    cloud_scheme="2m",         # ARG activated_cdnc feeds the 2M scheme
    ham_microphysics="placeholder",
    ham_arg_variant="arg2000", # or "ghosh2025"
)
```

The HAM path **augments** MACv2-SP rather than replacing it: MACv2-SP is kept
for the aerosol radiative optics and Twomey factor that radiation and the
cloud schemes hard-require, while HAM adds prognostic aerosol and ARG
activation. Letting online aerosol fully replace MACv2-SP optics (its direct
radiative effect) is tracked in #495.

## Status and caveats

- **Source magnitudes** (emissions) are order-of-magnitude defaults, not
  inventory-calibrated; the AeroCom-budget acceptance gate needs reference
  data not carried in-repo.
- **Re-evaporation** re-injection in wet deposition is deferred (no per-level
  evaporation diagnostic is exposed).
- **Real MAM4-JAX core** (#490) is not yet differentiable upstream; the
  harness itself is fully differentiable.

Out of scope, tracked separately: sectional/bulk families (#491), part2pop
diagnostics adapter (#492), SOA volatility basis (#493), heterogeneous
freezing (#494), aerosol optics in radiation (#495), gas-phase chemistry
coupling (#496).
