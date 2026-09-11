# Supported configurations

A *configuration* is a named, one-command composition of the process choices
described in the rest of this document, selected on the CLI as

```bash
python -m jcm.main +configuration=<name>
```

The ``+`` prefix is required — ``configuration`` is not a member of the base
``defaults:`` list, so a bare ``configuration=<name>`` errors. Each configuration
lives as one yaml in ``jcm/config/configuration/`` and is the authoritative record
of what that run composes. This page is the HAM2-style "which option fills each
slot" index; follow the links into the process sections for the science of each
choice.

Every ECHAM spectral configuration shares the same validated stability recipe:
dry Jablonowski–Williamson init (``init=jw``, ``init.rh=0.0``), the production
upper sponge (``run=longrun`` → sponge ``levels=10``, ``target_T_K=250`` — see
{doc}`dynamical_core` and {doc}`gravity_waves`), semi-Lagrangian off-centring 0.2,
and ``run.time_step=12`` min. An isothermal cold start or a shallower sponge NaNs
within days at L47, so these are not interchangeable with a bare ``grid=``
override.

## Which scheme fills each slot

Each column links to the process section that documents the scheme.

| package | [radiation](radiation.md) | [convection](convection.md) | [cloud fraction](clouds_microphysics.md) | [microphysics](clouds_microphysics.md) | [aerosol](aerosol.md) | [vert. diffusion](vertical_diffusion.md) | [surface](surface.md) | [gravity waves](gravity_waves.md) | [chemistry](chemistry.md) |
|---|---|---|---|---|---|---|---|---|---|
| **speedy** | SPEEDY SW/LW | SPEEDY | SPEEDY diagnostic | large-scale condensation | — | SPEEDY | SPEEDY bulk | — | — |
| **held_suarez** | Newtonian relaxation | — | — | — | — | — | Rayleigh friction | — | — |
| **echam** (1M) | RRTMGP | Tiedtke-Nordeng | Sundqvist | ECHAM 1-moment | MACv2-SP | TTE-TKE | ECHAM multi-tile | Hines + Lott-Miller SSO | SimpleChemistry |
| **echam-rrtmgp-2m** | RRTMGP | Tiedtke-Nordeng | Sundqvist | Lohmann 2-moment | MACv2-SP | TTE-TKE | ECHAM multi-tile | Hines + SSO | SimpleChemistry |
| **echam-emulated-2m** | NN emulator | Tiedtke-Nordeng | Sundqvist | Lohmann 2-moment | MACv2-SP | TTE-TKE | ECHAM multi-tile | Hines + SSO | SimpleChemistry |
| **echam-jam** | RRTMGP | Tiedtke-Nordeng | Sundqvist | Lohmann 2-moment | JAM modal (MAM4) | TTE-TKE + JAM tracer transport | ECHAM multi-tile | Hines + SSO | SimpleChemistry + JAM sulfur |
| **echam-jam-aerocom** | RRTMGP + aerosol-free solve | Tiedtke-Nordeng | Sundqvist | Lohmann 2-moment | JAM (+ AeroCom diagnostics incl. per-λ Mie optics) | TTE-TKE + JAM | ECHAM multi-tile | Hines + SSO | SimpleChemistry + JAM sulfur |
| **echam-jam-aerocom-optics** *(alias of the row above)* | RRTMGP + aerosol-free solve | Tiedtke-Nordeng | Sundqvist | Lohmann 2-moment | JAM (+ AeroCom diagnostics) | TTE-TKE + JAM | ECHAM multi-tile | Hines + SSO | SimpleChemistry + JAM sulfur |

The ``echam-jam*`` packages are **factory-built** (``builder: echam_physics`` →
``echam_physics()`` + ``jam_aerosol_physics()``): the JAM aerosol chain is split
around the cloud term, which a flat term list cannot express. The JAM path owns
the aerosol slot itself and does not additionally run MACv2-SP; JAM online optics
provide the direct effect. Two further physics packages
(``echam-rrtmgp-2m-cosp``, ``echam-jam-aci``) add COSP satellite simulators and
are used directly via ``physics=`` rather than through a configuration.

## Tier 1a — release-validated

These seven configurations are in the release-validation matrix
(``tools/release_validation/matrix.yaml``): each is run for one full A100 year
with 5-day means and checked by ``health.py``.

| configuration | package | grid | dycore | Δt | forcing / ozone |
|---|---|---|---|---|---|
| ``speedy-t31`` | speedy | T31 L8 sigma | dinosaur | 15 min | packaged T63 SST (interpolated); ozone auto |
| ``t63-echam-1m`` | echam (1M) | T63 L47 hybrid | dinosaur | 12 min | T63 present-day; ozone auto |
| ``t106-echam-1m`` | echam (1M) | T106 L47 hybrid | dinosaur | 12 min | T106 present-day; ozone auto |
| ``t63-echam-2m`` | echam-rrtmgp-2m | T63 L47 hybrid | dinosaur | 12 min | T63 present-day; ozone auto |
| ``t106-echam-2m`` | echam-rrtmgp-2m | T106 L47 hybrid | dinosaur | 12 min | T106 present-day; ozone auto |
| ``ma-t63-l47`` | echam-jam | T63 L47 hybrid | dinosaur | 12 min | T63 present-day + level-matched ozone |
| ``ma-t63-l95`` | echam-jam | T63 L95 hybrid | dinosaur | 12 min | T63 present-day + level-matched ozone |

The ``speedy-t31`` row is the deliberate outlier: SPEEDY takes the default
``run`` group and ``init=isothermal`` and ``run.time_step=15``, because the ECHAM
longrun sponge spans its entire L8 atmosphere and the dry-JW init is an ECHAM
spin-up device — both NaN SPEEDY within a chunk.

## Tier 1b — configuration-group, benchmark-validated

These twelve compose and run, and each has a throughput/stability benchmark, but
they are not (yet) in the release matrix. They are runnable anywhere the data
mirror is reachable, except where noted.

- ``t63-echam-rrtmgp`` and ``t63-echam-rrtmgp-2m`` — the same physics as
  ``t63-echam-1m`` / ``t63-echam-2m`` respectively, differing only in data
  plumbing; present as historical benchmark ids for the same compositions.
- ``t63-echam-emulated-2m`` — the NN radiation emulator in the 2M composition
  (ships trained weights via ``weights_file: auto``).
- ``t63-echam-jam`` and ``t63-echam-jam-aerocom`` — the JAM aerosol package at
  T63 L47, the latter adding the AeroCom phase-4 diagnostics (including the
  per-species/per-mode spectral optics, which ``echam-jam-aerocom`` enables by
  default). ``t63-echam-jam-aerocom-optics`` is an **alias** for the same
  physics options, kept as a named entry point the benchmark presets reference.
- ``ma-t106-l47``, ``ma-t106-l95`` — the JAM middle-atmosphere sweep at T106.
- ``ma-t119-l47``, ``ma-t119-l95`` — T119; **no mirror bundle** exists, so terrain
  and level-matched ozone are machine-local, and these ship
  **prescribed-emission-free** — the four file-backed inputs are nulled, so only
  online Gong sea salt is active.
- ``ma-ne30-l47``, ``ma-ne30-l95`` — the JAM physics on the pySES CAM-SE ne30
  cubed sphere (``dycore=pyses_ne30l{47,95}``, ``run=pyses_year``,
  ``init=isothermal`` — pySES rejects JW). These configurations set the scalar
  ``physics.cu_lmfmid: false`` so the Tiedtke mid-level trigger does not demand an
  ``omega`` the pySES backend does not provide, and are runnable as shipped —
  but **prescribed-emission-free**: ``forcing.*_file: auto`` resolves to nothing
  on the pySES path, so only the file-independent online natural emissions
  (wind-driven Gong sea salt) are active; sulfur, dust and carbonaceous species
  stay at zero unless explicit on-grid ``forcing.emissions_file`` /
  ``dms_file`` / ``dust_file`` / ``oxidants_file`` overrides are supplied. See
  {doc}`../design/pyses_cam_se_dycore`.

## Tier 2 — composable but unvalidated

These compose and run but have no validation coverage; several have a known
scientific gap, so treat results with care:

- **Grey radiation + online (JAM) aerosol** — the direct effect *is* carried,
  through the broadband 550 nm-band profile ``JamOpticsTerm`` writes for grey
  (see {doc}`radiation`), but at band-centre accuracy and with none of the
  per-band spectral detail RRTMGP uses, and no validation campaign has been run
  on the combination. Reachable from either door: ``echam_physics(
  aerosol_module="jam", radiation_scheme="grey", ...)`` or the CLI's
  factory-backed ``physics=echam-jam physics.radiation_scheme=grey``.
- **Untabulated hybrid level counts with ``diffusion=auto``** — the ECHAM
  ``lmidatm`` hyperdiffusion profiles exist only for L47/L95; any other hybrid
  level count falls back to the uniform SPEEDY del² profile with a warning, which
  is not the ECHAM stability stack such a grid may need (see {doc}`dynamical_core`).
- **2M microphysics with no aerosol coupling** — runs on the SPA activation
  fallback; clouds are known thin and the magnitude is unvalidated.
- The COSP-instrumented packages (``echam-rrtmgp-2m-cosp``, ``echam-jam-aci``),
  the idealized ``held_suarez``, the ``echam-strong-conv`` example, and the
  ``amip`` / ``era5`` / ``init=era5`` bundles that no configuration selects.

## Tier 3 — guarded or invalid

The model refuses these at build time rather than run something meaningless. Each
guard is a ``ValueError`` / ``RuntimeError`` in the file named:

- **JAM requires 2M microphysics** — ``jcm/physics/echam/echam_terms.py``
  (``echam_physics``): the JAM scavenging/resuspension terms read the 2M scheme's
  process-time ledger, which 1M does not publish.
- **AeroCom optics without JAM**, and **aerosol-free interval without RRTMGP** —
  same factory: the per-species optics and the aerosol-free companion solve both
  need the JAM population / RRTMGP optics to exist.
- **Unknown enum values** for ``radiation_scheme`` / ``cloud_scheme`` /
  ``aerosol_module`` / ``gw_scheme`` — same factory.
- **pySES + a dinosaur-specific init** (``jw`` / ``era5`` /
  ``balanced_isothermal``) or **pySES + nudging** — ``jcm/runners.py``: pySES
  initializes from its resting USSA-1976 state (``init=isothermal``) or a saved
  state, and nudging is dinosaur-only.
- **A SL-less dinosaur install** — ``jcm/dycore/dinosaur/dycore.py``
  (``_require_semi_lagrangian``): the Eulerian tracer path was removed, so
  the backend requires the semi-Lagrangian dinosaur.
- **Physics that needs a dycore field the backend cannot provide** —
  ``jcm/model.py``: e.g. Tiedtke's ``cu_lmfmid`` mid-level trigger needs ``omega``,
  which pySES does not publish (hence ``cu_lmfmid: false`` in the ne30
  configurations).

There are also runnable-but-warned traps (JAM on an aquaplanet, MACv2-SP with the
default all-ones weights, transient forcing with present-day emissions,
untabulated ``diffusion=auto`` levels); these warn at startup rather than raise.
