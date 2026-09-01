# ECHAM Observational Autoresearch Roadmap

## Purpose

This document maps ARM observations to compact equations and closures in the
JCM ECHAM physics stack that could be discovered, recalibrated, or evaluated by
autoresearch. Cloud fraction is the first campaign, not the final scope. The
larger objective is a reusable observational dataset that supports multiple
process-specific equation searches without erasing instrument provenance,
causal ordering, or target uncertainty.

The observational catalog and dataset architecture are described in
`ARM_JCM_OBSERVATION_MAP.md`. The Mac-local MICROBASE acquisition and reduction
workflow is described in `MAC_LOCAL_MICROBASE_PIPELINE_PLAN.md`.

## ECHAM Process Order

The default composition in `jcm/physics/echam/echam_terms.py` runs approximately
in this order:

```text
moist atmospheric state
    -> boundary conditions
    -> aerosol and simple chemistry
    -> diagnostic cloud fraction
    -> radiation
    -> TTE-TKE vertical diffusion and surface exchange
    -> surface diagnostics
    -> Tiedtke-Nordeng convection
    -> cloud microphysics
    -> optional aerosol wet removal and chemistry
    -> gravity-wave drag
```

This ordering is part of the scientific contract. A learned replacement may use
only quantities available at its insertion point. For example, cloud fraction
must not use same-step precipitation generated later by microphysics, and a
convection trigger may use the same-step vertical-diffusion moisture supply only
because vertical diffusion runs first.

## Candidate Summary

| Priority | Equation family | Example target | Main ARM evidence | Feasibility with ARM evidence |
| --- | --- | --- | --- | --- |
| 1 | Surface exchange | Drag and heat/moisture transfer coefficients | ECOR/QCECOR, EBBR, towers | High |
| 2 | Boundary-layer mixing | Eddy diffusivity, mixing length, entrainment | Fluxes, profiles, PBL height, Doppler lidar | Medium-high |
| 3 | Warm-rain formation | Autoconversion and accretion rates | MICROBASE, MWR, radar, disdrometers | Medium-high |
| 4 | Rain evaporation | Below-cloud precipitation loss | Radar profiles, humidity, surface rain | Medium-high |
| 5 | Aerosol activation | Activated droplet number or fraction | CCN, size, composition, updraft | Medium-high |
| 6 | Cloud optical properties | Effective radius and optical depth | MICROBASE, MWR, MFRSR, QCRAD | Medium-high |
| 7 | Aerosol optical properties | Humidity growth, extinction, SSA | Nephelometer, absorption, AOD, size | High for local relationships |
| 8 | Convection | Trigger, mass-flux closure, entrainment | Soundings, VARANAL, radar, LASSO LES | Medium-low from routine ARM alone |
| 9 | Mixed-phase microphysics | Freezing, deposition, phase conversion | Radar/lidar, MICROBASE, campaigns | Medium-low |
| 10 | Land-surface response | Moisture stress, heat diffusion, albedo | ECOR, soil, surface radiation | Medium-high locally |

## 1. Surface Exchange

The ECHAM surface layer and lower boundary of TTE-TKE vertical diffusion use
bulk relationships of the form:

```text
H   = rho cp C_H U (T_surface - T_air)
LE  = rho Lv C_E U (q_surface - q_air)
tau = rho C_D U^2
```

The compact empirical target is the stability-dependent transfer relationship:

```text
C_D, C_H, C_E = f(Richardson number, wind, roughness, stability)
```

Relevant implementation:

- `jcm/physics/vertical_diffusion/tte_tke/surface_layer.py`
- `jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion.py`

Relevant observations:

- `30ecor` and `30qcecor` sensible and latent heat fluxes;
- friction velocity and turbulence statistics;
- `30ebbr` as an independent energy-balance method;
- tower wind, temperature, humidity, and pressure;
- surface and soil temperature and moisture; and
- surface radiation for energy-budget context.

This is one of the strongest observational equation-discovery targets because
the primary outputs are measured fluxes rather than unobserved process
tendencies. Footprint mismatch, energy-balance closure corrections, roughness,
and land-cover heterogeneity must remain explicit metadata.

## 2. Boundary-Layer Mixing

TTE-TKE uses relationships resembling:

```text
K_m = c_m l sqrt(TKE)
K_h = c_h l sqrt(TKE)
```

Candidate equation targets include:

- the mixing-length relationship `l`;
- stability dependence of `K_m` and `K_h`;
- the turbulent Prandtl number `K_m / K_h`;
- PBL-top entrainment;
- TKE production and dissipation; and
- a compact mixing-timescale closure.

Relevant implementation is under
`jcm/physics/vertical_diffusion/tte_tke/`, especially
`turbulence_coefficients.py` and `tke_budget.py`.

Relevant observations include ECOR fluxes, tower gradients, soundings, AERI and
MWR thermodynamic profiles, radar wind-profiler winds, Doppler-lidar turbulence,
and method-labelled PBL-height products.

Interior eddy diffusivity is not directly observed. It must be inferred from
flux-gradient relationships, profile evolution, or budget closure. LASSO LES is
valuable process-resolved support, but it is simulated evidence and must remain
distinguishable from instrument observations.

## 3. Warm-Rain Formation

ECHAM 1-moment and optional 2-moment microphysics contain empirical
autoconversion and accretion relationships. A compact target is:

```text
P_auto = f(qc, Nc, cloud_fraction, density, temperature)
```

Relevant implementation:

- `jcm/physics/clouds/echam_1m.py`
- `jcm/physics/clouds/lohmann_2m/precip.py`

Useful evidence includes:

- MICROBASE liquid-water concentration and effective radius;
- MWR liquid-water path;
- cloud radar drizzle onset, reflectivity, and fall streaks;
- CCN, CDNC proxies, or aerosol context;
- disdrometer drop-size distributions; and
- surface precipitation.

Candidate equations must enforce nonnegative rates, zero production at zero
condensate, water conservation, timestep-safe conversion, and physically
defensible monotonicity. Surface rain alone cannot identify autoconversion
because accretion, sedimentation, and evaporation intervene. Vertically
resolved radar evidence is therefore important.

## 4. Rain Evaporation

Below-cloud precipitation loss can be treated separately:

```text
E_rain = f(RH deficit, temperature, pressure, rain flux, density, layer depth)
```

Relevant implementation is the precipitation-evaporation path in
`jcm/physics/clouds/echam_1m.py`.

Useful evidence includes radar precipitation profiles, cloud-base height,
surface rain rate, disdrometers, and thermodynamic profiles from soundings,
AERI, or MWR. Virga cases, where precipitation occurs aloft but weakens or
vanishes before reaching the surface, are particularly informative.

## 5. Aerosol Activation

The simple SPA pathway uses a relationship approximately of the form:

```text
Nc = A (CCN * cloud_fraction)^b
```

The ARG pathway represents activation more physically:

```text
activated_fraction, S_max = f(size, kappa, number, updraft, T, p)
```

Relevant implementation:

- `jcm/physics/aerosol/spa.py`
- `jcm/physics/aerosol/jam/activation/`

ARM can provide CCN concentration at measured supersaturation, CPC number,
SMPS/UHSAS/APS size distributions, ACSM composition, HTDMA hygroscopic growth,
cloud-base velocity, and cloud effective-radius or CDNC retrieval context.

Instrument supersaturation, inlet size cut, dry versus ambient convention,
particle-diameter definition, and cloud selection must be part of the sample
schema. A generic aerosol concentration field would destroy information needed
to identify activation physics.

## 6. Cloud Optical Properties

Once cloud fraction and condensate are credible, candidate relationships are:

```text
r_eff = f(qc, Nc, temperature, phase)
optical_depth = f(LWP, IWP, r_eff, cloud_fraction)
```

Relevant implementation:

- `jcm/physics/radiation/cloud_optics.py`
- `jcm/physics/radiation/mcica.py`

Relevant observations include MICROBASE condensate and effective radius, MWR
LWP, MFRSR cloud optical depth, QCRAD broadband fluxes, and radar/lidar phase.
Cloud-mask profiles can also constrain statistical vertical-overlap and
decorrelation-length relationships.

Radiative fluxes should normally be coupled validation targets rather than the
only supervision. Otherwise optical coefficients can absorb errors in cloud
amount, condensate, surface albedo, atmospheric state, or gas absorption.

## 7. Aerosol Optical Properties

Potential compact relationships include:

```text
f(RH) = wet_scattering / dry_scattering
extinction = f(size_distribution, composition, RH)
SSA = f(scattering, absorption, composition)
CCN = f(size_distribution, composition, kappa)
```

Relevant implementation:

- `jcm/physics/aerosol/macv2_sp.py`
- `jcm/physics/aerosol/jam/optics/`

ARM evidence includes dry/wet nephelometer scattering, PSAP/CLAP absorption,
size distributions, composition, hygroscopicity, and MFRSR AOD. These products
can strongly constrain local microphysical and optical relationships.

One ARM site cannot identify global MACv2 plume placement, transport, or source
strength. Local equation discovery must not be presented as a global aerosol
distribution calibration.

## 8. Convection

Potential Tiedtke-Nordeng targets include:

```text
trigger = f(CAPE, CIN, RH, PBL depth, moisture convergence)
M_cloud_base = f(CAPE, moisture supply, closure timescale)
entrainment = f(height, buoyancy, humidity contrast)
precipitation_efficiency = f(cloud depth, condensate, RH)
```

Relevant implementation is under
`jcm/physics/convection/tiedtke_nordeng/`.

Routine ARM observations do not directly measure convective mass flux,
entrainment, or process tendencies. This campaign requires a distinct evidence
combination:

- ARM soundings and radar for observational constraints;
- VARANAL large-scale forcing and budget products;
- LASSO or CloudBench LES for process-resolved supervision; and
- short forced single-column integrations for coupled evaluation.

Convection should not be framed as ordinary profile-to-tendency supervised
learning. Advection, mesoscale organization, and multiple compensating closure
parameters make the target weakly identifiable without forcing and process
context.

## 9. Mixed-Phase And Ice Microphysics

Potential targets include:

- liquid-to-ice phase partition;
- heterogeneous and homogeneous freezing;
- Wegener-Bergeron-Findeisen conversion;
- vapor deposition;
- aggregation and riming;
- snow production and melt; and
- ice sedimentation.

Relevant implementation is under `jcm/physics/clouds/lohmann_2m/` and the ice
paths in `jcm/physics/clouds/echam_1m.py`.

Useful observations include radar/lidar phase, MICROBASE liquid and ice
retrievals, MWR LWP, precipitation phase, and specialized mixed-phase campaign
measurements. NSA and mixed-phase campaigns are likely more informative than an
SGP-only backbone.

Retrieval uncertainty, particle habit, vertical velocity, nucleation,
sedimentation, and convective detrainment are strongly confounded. This should
follow warm-cloud work rather than precede it.

## 10. Land-Surface Response

Candidate equations include:

- soil-moisture stress on evaporation and transpiration;
- roughness and stability effects on surface exchange;
- soil thermal diffusion;
- ground heat flux;
- snow or soil albedo response; and
- partitioning of available energy into sensible and latent heat.

Relevant implementation is under `jcm/physics/surface/echam/`, but the default
ECHAM composition delivers atmospheric turbulent fluxes through the lower
boundary of vertical diffusion. Any learned surface equation must therefore be
inserted where it actually affects the atmospheric coupling, not merely replace
a diagnostic that republishes fluxes.

ARM ECOR, EBBR, soil, tower, albedo, and radiation products provide strong local
constraints. Footprint and land-cover heterogeneity limit direct interpretation
as a coarse-grid global closure.

## Poor ARM-Only Targets

The following should not be early observation-only autoresearch campaigns:

- Hines non-orographic gravity-wave drag;
- Lott-Miller subgrid-orographic drag;
- global aerosol emissions, transport, and deposition;
- correlated-k gas spectroscopy in RRTMGP;
- vertically resolved atmospheric chemistry; and
- top-of-atmosphere radiation inferred only from surface ARM measurements.

Radiation emulation remains useful, but RRTMGP or line-by-line calculations
should be the teacher. ARM observations should evaluate uncertain cloud,
aerosol, surface, and atmospheric inputs rather than relearn molecular
spectroscopy from surface broadband fluxes.

## Evidence Classes

Every target must be labelled by evidence class:

| Class | Meaning | Examples |
| --- | --- | --- |
| Direct or instrument-proximate | A calibrated measurement or standard instrument reduction | Irradiance, precipitation gauge, CCN count, size spectrum |
| Retrieval or VAP | A multisensor or prior-dependent retrieved quantity | MICROBASE condensate, PBL height, AOD, ARMBE fields |
| Inferred residual | A derivative, flux divergence, or budget residual | Heating rate, eddy diffusivity, convective tendency |
| Simulated process evidence | Process-resolved model output conditioned on forcing | LASSO LES, CloudBench LES |

These classes must not be mixed into an apparently homogeneous target. A model
trained against a budget residual has a different evidentiary meaning from one
trained against a flux sensor.

## Versioned Process-Example Modules

### Meaning

A process-example module is not necessarily a Python package or a new ECHAM
physics implementation. It is an immutable, reproducible dataset recipe for one
specific closure at one causal insertion point.

The shared native and harmonized observation layers contain broadly useful ARM
records. A process-example module turns selected records into the exact rows,
profiles, sequences, inputs, targets, masks, and splits required for one
autoresearch question.

For example, `surface_exchange/v1` and `warm_rain_autoconversion/v1` use some of
the same atmospheric-state observations, but they have different targets,
cadences, QC, collocation rules, and causal inputs. They must not be flattened
into one universal table.

### Required Contents

A module should have a structure similar to:

```text
process_examples/
    surface_exchange/
        v1/
            README.md
            recipe.yaml
            schema.json
            build.py
            splits.json
            manifest.json
    warm_rain_autoconversion/
        v1/
            README.md
            recipe.yaml
            schema.json
            build.py
            splits.json
            manifest.json
```

The exact storage format can be NetCDF, Zarr, or Parquet according to the sample
geometry. The scientific contract matters more than the container format.

Each version pins:

- the scientific question and ECHAM insertion point;
- permitted predictors available at that insertion point;
- target definition and evidence class;
- source datastreams, product versions, and facility coordinates;
- units, sign conventions, and vertical coordinates;
- QC interpretation and missing-data rules;
- temporal averaging windows and minimum valid counts;
- spatial and vertical collocation operators;
- conservative remapping rules where applicable;
- physical constants and derived-variable formulas;
- uncertainty fields and sample weights;
- blockwise train, validation, and untouched outer-holdout splits;
- output schema and validation checks; and
- builder code revision and source/output checksums.

### Why Version It

Changing any of the following can change the scientific meaning of a sample:

- cloud occurrence from strict to inclusive retrieval flags;
- a 60-minute window to a 15-minute window;
- in-cloud condensate to grid/time-mean condensate;
- one PBL-height algorithm to another;
- a surface flux target from ECOR to energy-balance-corrected EBBR;
- adding a predictor generated after the ECHAM insertion point;
- changing vertical remapping or density conversion;
- changing the site/time split; or
- changing a QC bit from accepted to excluded.

Such a change creates a new process-example version rather than silently
overwriting the old dataset. Immutable versions allow an equation, metric, and
paper result to refer to an exact scientific contract.

A simple integer sequence such as `v1`, `v2`, and `v3` is sufficient. A new
version is required when rows, values, target meaning, permitted inputs, or
splits change. Documentation corrections or builder refactors that produce
identical checksums can retain the same data version while recording a new code
revision in the manifest.

### Example Contracts

`surface_exchange/v1` might define:

```text
sample: one quality-controlled 30-minute ECOR interval
inputs: pre-vdiff near-surface state, wind, stability, roughness, soil state
targets: sensible heat, latent heat, and momentum-related flux diagnostics
evidence: direct/instrument-proximate
split: blocked by season and year, with whole years held out
```

`warm_rain_autoconversion/v1` might define:

```text
sample: one cloud layer and matched radar time window
inputs: pre-microphysics qc, cloud fraction, density, temperature, CDNC context
targets: drizzle onset and vertically conditioned precipitation-production proxy
evidence: retrieval/VAP plus instrument-proximate radar and disdrometer evidence
split: whole storm systems and time blocks, never random profile rows
```

`convection_trigger/v1` might define:

```text
sample: one forced single-column analysis window
inputs: post-vdiff state, CAPE/CIN, moisture supply, large-scale forcing
targets: radar-observed convective initiation plus VARANAL/LES process diagnostics
evidence: mixed observation, VAP, inferred residual, and simulated process evidence
split: whole convective events and deployments
```

## Autoresearch Contract

Every equation campaign should proceed through the same gates:

1. Freeze the target, insertion point, permitted inputs, and evidence class.
2. Build and manually audit a bounded process-example version.
3. Reserve site/time/deployment blocks before equation search.
4. Fit simple baselines and published reference equations first.
5. Search compact equations with units, signs, bounds, conservation, and
   monotonicity encoded where physically justified.
6. Rank candidates on validation blocks without accessing the outer holdout.
7. Freeze the candidate and evaluate the outer holdout once.
8. Insert the equation into ECHAM and run short sequential or forced-column
   tests.
9. Evaluate coupled downstream effects and conservation, not only offline RMSE.
10. Promote only equations that remain stable and useful online.

Long free-running evaluations and instantaneous offline fits answer different
questions. Instantaneous fits risk process attribution error; long runs allow
state drift and compensating parameterizations. Short sequential rollouts are
the bridge between them.

## Recommended Program Order

1. Diagnostic cloud fraction.
2. Surface exchange coefficients.
3. Warm-rain autoconversion and rain evaporation.
4. Aerosol activation.
5. Cloud effective radius, optical depth, and overlap.
6. Boundary-layer mixing and entrainment.
7. Aerosol hygroscopicity and optical properties.
8. Convection with VARANAL and LES support.
9. Mixed-phase microphysics using NSA and focused campaigns.

This order balances scientific value, observational identifiability, and the
ECHAM process chain. It also lets later campaigns consume better cloud,
aerosol, and turbulence diagnostics produced by earlier work without allowing
future-process information to leak into an upstream closure.
