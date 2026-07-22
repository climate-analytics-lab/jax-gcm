# Design spike: extending JCM to planetary science

**Status:** feasibility spike — no implementation yet.
**Branch:** `claude/jcm-planetary-science-spike-jw2ew7`

## Motivation and scope

The immediate motivation is the KITP program *clouds28*
(<https://www.kitp.ucsb.edu/activities/clouds28>): cloud formation and
convection across planetary environments — solar-system atmospheres (Mars,
Titan, the giants), exoplanets (from temperate terrestrial worlds to hot
Jupiters), and brown dwarfs.

> **Assumption.** The program page was not reachable from the environment in
> which this spike was written, so the scope above is inferred from the
> program slug and the general framing of recent KITP planetary-atmospheres
> activities. If the actual program emphasizes different regimes (e.g.
> protoplanetary disks, microphysics lab work), the tier boundaries below
> still hold but the phase ordering should be revisited.

The question this spike answers: **how much of JCM is Earth-hardcoded, and
what would it take to run credible planetary simulations?** The answer,
in short: the dynamical core and the constants layer are already fully
planet-parameterized; the Earth-specificity is concentrated in the physics
parameterizations, and it falls into three cleanly separable tiers of
difficulty.

### Why JCM at all, when planetary GCMs exist?

Established planetary GCMs — LMD Generic PCM, Isca, ExoCAM, MITgcm,
ExoPlaSim — all exist and are battle-tested. JCM's differentiator is that it
would be, to our knowledge, **the first fully differentiable planetary GCM**.
That matters for exactly the class of questions a clouds-across-environments
program asks:

- **Parameter inference under sparse data.** Exoplanet and brown-dwarf
  observables (JWST spectra, phase curves, variability) constrain cloud and
  convection parameterizations only indirectly. Gradient-based calibration of
  scheme parameters against observables — the workflow JCM already supports
  for Earth — is qualitatively harder with non-differentiable models
  (ensemble/MCMC over expensive GCM runs).
- **Sensitivity analysis.** ∂(observable)/∂(obliquity, rotation rate, cloud
  parameter) in one backward pass, rather than finite-difference ensembles.
- **Hybrid physics–ML.** Learning cloud microphysics corrections jointly with
  the circulation, e.g. training an emulator of a kinetic cloud model inside
  the GCM loop.

### Precedent: PlaSIM / ExoPlaSim

The closest existing analogue is **PlaSIM** (Planet Simulator, Univ. Hamburg)
and its exoplanet fork **ExoPlaSim** (Paradise et al. 2022, MNRAS 511:3272).
PlaSIM couples the PUMA spectral dynamical core to a simplified physics
package — note it is *not* SPEEDY-based, but it is the same
intermediate-complexity class as JCM's SPEEDY package: spectral core,
T21–T42, empirical broadband radiation, diagnostic clouds. ExoPlaSim
demonstrated that this class of model, suitably parameterized, usefully
simulates tidally-locked terrestrial planets, non-Earth surface pressures,
different stellar spectra, and geological-timescale evolution. Mars and Titan
adaptations of PlaSIM also exist. This is strong evidence that **JCM's
SPEEDY-class physics is scientifically adequate for terrestrial-planet
questions** — the intermediate-complexity level is a feature (fast, tunable,
interpretable), not a blocker. Isca (Vallis et al. 2018, GMD) makes the same
point from the GFDL lineage: a hierarchy from Held-Suarez through grey
radiation to full physics, applied to Mars, Jupiter and tidally-locked
planets.

## What is already planet-ready (audit findings)

The audit examined constants, dycore coupling, insolation, thermodynamics,
radiation, surface, vertical coordinates, and config. Anchors are given as
`file:line` at the audit date; they will drift.

### Constants: single overridable source of truth

`jcm/constants.py` is explicitly designed for planet overrides: all
independent quantities are fields of the `PhysicalConstants` NamedTuple
(radius, Ω, g, cpd, κ, rv, eps, latent heats, solar constant, p₀, …), derived
quantities (`rd = akap·cpd`, `vtmpc1`, …) are recomputed properties, and
`set_constants()` rebinds a process-global singleton that consumers read by
attribute access. Hydra already exposes this: `config.yaml` has a
`constants: {}` group applied in `runners.py` *before* model construction —
`python -m jcm.main +constants.grav=3.71 +constants.rearth=3.39e6` works
today.

One subtlety to document in any planet config: `rd` is derived, so a CO₂
atmosphere is specified by setting `cpd` **and** `akap` consistently
(e.g. Mars: `cpd≈846`, `akap≈0.227` → `rd≈192`), not by setting `rd`
directly.

### Dynamical core: fully parameterized

`jcm/dycore/dinosaur/dycore.py:39` (`physics_specs_from_constants`) is the
single bridge that maps `PhysicalConstants` → dinosaur's
`PrimitiveEquationsSpecs.from_si(radius, angular_velocity, gravity,
ideal_gas_constant, kappa, …)`. Nothing in the dycore path is Earth-locked:
Coriolis comes from `omega`, the grid radius from `rearth` (`utils.py:97`),
the initial isothermal reference atmosphere from `p0`. **Arbitrary rotation
rates (including slow/tidally-locked), radii, gravities and gas constants
work out of the box dynamically.**

### Held-Suarez: a dry planetary benchmark, today

`jcm/physics/held_suarez/` takes its forcing parameters as SI `Quantity`
kwargs and sources κ from the live constants. Combined with the two points
above, **JCM can already run dry planetary-circulation experiments** —
e.g. the classic rotation-rate sweeps (superrotation onset), or an HS-style
benchmark with Mars-like parameters. The only blemish is a cosmetic hardcoded
`101325.0` reference pressure in `cache_coords`.

### Also already fine

- **Vertical coordinates** (sigma/hybrid) are fractions of surface pressure —
  composition-agnostic.
- **Terrain**: `TerrainData.aquaplanet()` gives a uniform surface;
  orography/land-sea mask are data-driven with no Earth values baked into the
  struct.
- **Solar constant** is a plumbed field (`RadiationParameters.solar_constant`)
  — stellar flux scaling is trivial.
- **`SolarGeometry`** (`forcing.py`) is a clean abstraction boundary: physics
  consumes `tyear` / `orbital_phase` / `synodic_phase`, so planetary orbits
  slot in behind one interface (see Tier 1).

## Gap analysis

Earth-hardcoding concentrates in three tiers.

### Tier 1 — moderate: orbit, calendar, grey radiation

These unlock *idealized moist planetary climates* (the ExoPlaSim/Isca
operating point) and are each bounded, local changes.

**1a. Orbital insolation.** Three insolation paths exist and all assume
Earth's orbit:

- SPEEDY shortwave (`speedy_shortwave.py:417` `solar()`): declination and
  star–planet distance come from the Spencer (1971) Fourier series with
  Earth's obliquity and eccentricity baked into ~5 lines of coefficients
  (444–448). Crucially, the surrounding daily-average hour-angle geometry
  (450–462) is planet-general. Replacing the Spencer lines with a
  parameterized computation — declination from obliquity + true anomaly,
  distance from a Kepler solve over eccentricity — makes SPEEDY insolation
  fully planetary. This is a well-understood, ~100-line, testable module.
- The grey/RRTMGP/NN paths use the external `jax_solar` package via
  `OrbitalTime(orbital_phase, synodic_phase)`; `jax_solar` hardcodes Earth
  obliquity/eccentricity/perihelion internally. Proposal: write
  `jcm/orbit.py` with an `OrbitalParameters` struct (obliquity, eccentricity,
  longitude of perihelion, rotation:orbit ratio, solar constant) producing
  declination / distance / zenith angle, and use it behind `SolarGeometry`
  for *all* schemes, dropping the `jax_solar` dependency for geometry. A
  differentiable Kepler solve is a few Newton iterations — no obstacle.
- **Tidally-locked mode** falls out almost for free: fixed substellar point =
  zenith angle from (lat, lon − substellar lon), no synodic phase. This is
  the single most important configuration for the exoplanet community and
  should be a first-class switch, not an edge case.

**1b. Calendar.** `jcm/date.py` hardcodes 86400 s/day and 365-day years
(`SUPPORTED_CALENDARS`, `_FIXED_UNIT_DAYS`, `tyear()`), and
`forcing.py:591` derives `fraction_of_day` as `seconds/86400`. Proposal: a
`planetary` calendar parameterized by `(seconds_per_day, days_per_year)`
feeding the same `tyear`/phase outputs. Mars (sol = 88775 s, year = 668.6
sols) is the acceptance test. This is self-contained but touches dating
throughout I/O, so it needs care around forcing-file time axes.

**1c. Grey / semi-grey radiation with free parameters.** The
`grey_two_stream` package is the right *skeleton* (two-stream solver +
Planck), but despite the name its `gas_optics.py` hardcodes Earth absorbers:
H₂O continuum, CO₂ 15 µm band, O₃ bands, Rayleigh with an 8 km scale height
at 101325 Pa. Proposal: add a genuinely grey/semi-grey/picket-fence gas
optics option — optical depth as a free function of pressure
(τ = τ₀·(p/p₀)^n per band, the Frierson/Isca pattern) with tunable
shortwave/longwave parameters. The two-stream solver and Planck integration
are reusable as-is. This is *the* standard idealized-planetary radiation and,
being low-dimensional and smooth, it is also the natural target for
gradient-based tuning against line-by-line calculations or observations.

**1d. `planet` Hydra config group.** `config/planet/{earth,mars,
tidally_locked_m_earth,...}.yaml` bundling the constants overrides (1 file
per planet) plus the new orbital/calendar/radiation parameters. The
`constants:` plumbing already exists; the new parameters need homes in the
structs above. Also: `forcing.py:_validate_bc_fields` hardcodes Earth-range
sanity checks (SST 220–320 K etc.) which must become per-planet or
warn-only.

### Tier 2 — deep: condensables, composition-aware radiation, surfaces

These unlock *quantitative* solar-system and exoplanet science.

**2a. Generalized condensable species.** Water is hardwired as *the*
condensable: Tetens/Magnus coefficients and 622/0.378 molecular-weight
factors in `speedy_humidity.py:84–99`, water/ice saturation coefficients in
the ECHAM cloud schemes, latent heats applied as `alhc/alhs`, and — the
worst offender — the literal `0.608` (water's `vtmpc1`) scattered through
convection, vertical diffusion, and surface flux code instead of
`c.vtmpc1`. Proposed staging:

  1. *Hygiene sweep (cheap, do early, Earth-neutral):* replace every literal
     `0.608`/`0.622`/`0.378` with the derived constants (`c.vtmpc1`,
     `c.eps`). This is a pure refactor, verifiable against existing
     regression tests, and is prerequisite plumbing for everything else. The
     audit found the codebase already *inconsistent* here (some call sites
     use `c.vtmpc1`, some the literal), so this is worth doing regardless of
     planetary ambitions.
  2. *Condensable struct:* a `Condensable` dataclass (saturation vapor
     pressure function, latent heats, triple point, molecular weight ratio,
     condensate densities) with `water()`, `co2()`, `ch4()` factories,
     threaded through the saturation call sites. The constants layer already
     carries `rv/cpv/eps/alhc/...` — this generalizes their *source*, not
     their use.
  3. *Regime caveat:* Mars' main condensable (CO₂) is the bulk gas itself —
     condensation changes surface pressure — which no Earth-derived moisture
     scheme represents. Titan's methane cycle is closer to Earth's hydrology.
     Realistic targets: CH₄/H₂O cycles first; CO₂ mass condensation is its
     own project.

**2b. Composition-aware radiation.** RRTMGP is wired to Earth k-distribution
tables (H₂O/CO₂/O₃/CH₄/N₂O/O₂); retraining for exotic compositions is
deep-surgery and *should not be attempted in early phases* — the semi-grey
scheme (1c) covers idealized science, and correlated-k tables for specific
atmospheres (e.g. from HELIOS/petitRADTRANS/Exo_k line lists) can be a later
integration. The NN radiation emulator path is interesting here: a
JAX-native emulator trained on planetary line-by-line calculations would be
differentiable by construction and cheaper than porting a full k-table
pipeline.

**2c. Non-water surfaces.** Aquaplanet (uniform prescribed surface) works
today and covers a lot of exoplanet science. The ECHAM/SPEEDY ocean, sea-ice
and land modules assume water thermodynamics at 273.15 K; a Mars-like
regolith (no ocean, low thermal inertia soil) or Titan hydrology would be
new surface schemes. Bounded, but real work; sequence after 2a.

### Tier 3 — out of scope for now: giants and brown dwarfs

The dycore solves the **hydrostatic primitive equations in sigma
coordinates on a shallow spherical shell**, and physics assumes a surface.
Hot Jupiters are marginal (km/s winds stress the spectral core's
hyperdiffusion and timestep; precedent exists — MITgcm/SPARC runs — but it
is a research project), and brown-dwarf convection is non-hydrostatic. Deep,
surface-free atmospheres would need a different bottom boundary treatment
and probably a different vertical coordinate. **Recommendation: declare
terrestrial-planet scope (Tier 0–2) and treat gas giants as a separate
future spike.** JCM can still contribute to brown-dwarf/giant questions at
the *parameterization* level (differentiable cloud microphysics columns)
without global circulation.

## What JCM could contribute to clouds28-type questions, by phase

| Phase | Capability | Example science questions |
|---|---|---|
| **0 (today)** | Dry planetary dynamics: arbitrary radius/Ω/g/R/cp + Held-Suarez | Rotation-rate regime transitions; superrotation onset; differentiable sensitivity of jet structure to planetary parameters |
| **A (Tier 1)** | Moist idealized planets: planetary orbit/calendar, semi-grey radiation, aquaplanet, tidally-locked mode | Cloud/convection feedback on tidally-locked M-dwarf planets; substellar cloud shielding; obliquity/eccentricity seasonal cycles; gradient-tuning grey radiation and convection parameters against observed phase curves |
| **B (Tier 2a/2c)** | Non-water condensables, simple non-ocean surfaces | Titan-like methane hydrology; condensable-dependent convection regimes; comparative "hydrological" cycles |
| **C (Tier 2b)** | Composition-specific radiation (per-planet k-tables or NN emulators) | Quantitative spectra/phase-curve forward modelling in the loop with circulation |

The differentiability thread runs through all phases and is the novel
contribution at each: every new parameterized module (orbit, grey optics,
condensable) is smooth and low-dimensional by design, i.e. built to be
inferred, not just prescribed.

## Proposed implementation order

1. **Phase 0 validation (days):** a `planet` config group + Mars/slow-rotator
   Held-Suarez regression tests proving the existing constants path
   end-to-end. Zero new physics. Also fix the cosmetic `101325.0` in
   Held-Suarez `cache_coords` and do the Tier-2a hygiene sweep
   (`0.608` → `c.vtmpc1`), which is Earth-neutral.
2. **Phase A (weeks):** `jcm/orbit.py` (`OrbitalParameters`, Kepler solve,
   declination/zenith, tidally-locked mode) behind `SolarGeometry`;
   `planetary` calendar in `date.py`; parameterized semi-grey gas optics in
   `grey_two_stream`. Acceptance: Earth defaults reproduce current results
   bit-for-bit-ish (regression), Mars insolation matches published curves,
   tidally-locked aquaplanet reproduces the qualitative ExoPlaSim/THAI
   circulation (substellar convection, eastward jet).
3. **Phase B (months):** `Condensable` abstraction + Titan-like CH₄ cycle;
   simple regolith surface.
4. **Phase C (opportunistic):** composition radiation via NN emulation or
   external k-tables.

## Risks and open questions

- **`jax_solar` and `dinosaur` are external.** Orbit generalization is best
  done by *replacing* jax_solar geometry with `jcm/orbit.py` rather than
  forking it. Dinosaur needs nothing changed (already parameterized), but we
  depend on that remaining true.
- **Calendar/I-O coupling:** forcing files and climatology interpolation are
  keyed to Earth calendars; planetary runs will mostly use synthetic/uniform
  forcing, but the dating code paths need to fail loudly, not silently, when
  an Earth climatology meets a Mars calendar.
- **SPEEDY empirical tunings:** beyond insolation, SPEEDY's shortwave/
  longwave transmissivities and reference lapse/scale-height globals
  (`speedy/physical_constants.py`) are Earth-tuned module globals. For
  planetary work the ECHAM+grey path is the primary vehicle; SPEEDY physics
  stays Earth-scoped unless someone specifically wants it.
- **Validation targets:** THAI (tidally-locked terrestrial intercomparison)
  and published ExoPlaSim/Isca results give concrete benchmark cases for
  Phase A; agreeing on 2–3 named benchmark configurations early will keep
  the work honest.
- **Workshop scope confirmation:** see the assumption note at the top —
  worth confirming the clouds28 program's actual emphasis before committing
  Phase B/C ordering.

## Verdict

**Feasible, and closer than expected.** The two changes that are usually the
expensive ones in an Earth-GCM-to-planetary conversion — parameterizing the
dynamical core and centralizing physical constants — are already done and
config-exposed. Dry planetary dynamics works today; moist idealized planets
(the operating point of ExoPlaSim and Isca, and the level at which most
clouds-across-environments circulation questions are posed) is weeks of
bounded work concentrated in three modules (orbit, calendar, grey optics);
quantitative solar-system atmospheres are a longer but well-precedented road.
The differentiable-GCM angle is unique in the planetary space and aligns
directly with the inference-limited nature of exoplanet cloud observations.
