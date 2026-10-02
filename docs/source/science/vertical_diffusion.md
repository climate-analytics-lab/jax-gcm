# Vertical diffusion (boundary-layer turbulence)

**What we do.** Boundary-layer turbulent mixing uses a TKE-based (total-turbulent-
energy) closure ported from ICON/ECHAM ``vdiff``, wrapped as the composable term
``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion.py::TteTkeVerticalDiffusion``.
It carries a prognostic TKE (and θᵥ-variance) budget — shear + buoyancy production
− dissipation + transport — diagnoses exchange coefficients from a mixing length
and √TKE, and solves the diffusion implicitly (backward Euler) with a tridiagonal
Thomas solve (``matrix_solver.py``, following ``mo_vdiff_solver.f90``). The term
owns the whole turbulent column ECHAM-style. Momentum couples to the surface
through one bottom-row Robin term with the fraction-weighted drag. Heat and
moisture couple tile by tile, as ECHAM6.3's ``richtmyer_land``/``_ocean``/``_ice``
and ``blend_zq_zt`` do. After the top-down elimination, each tile relates the
lowest level to its own surface value, ``X̂_K,t = E_t·X̂_s,t + F_t``. The land
tile solves its skin energy balance against its own coefficients (see
{doc}`surface`), the bottom value is the fraction-weighted blend, and each tile's
flux is taken against its own lowest-level value. The delivered fluxes are
diagnosed from the implicit solution (the ``pev_vdiff`` identity: reported
equals delivered by construction). Surface-layer
exchange coefficients use a faithful Louis (1979, unstable) / Mauritsen (2007,
stable) port (``surface_layer.py``, ``mo_turbulence_diag::sfc_exchange_coeff``).
The interior stability — the Richardson number that scales the mixing length
and the buoyancy term of the TKE budget — is ECHAM's moist, cloud-weighted
buoyancy of each interface (see *Interior stability* below).
**K floors** hold minimum diffusivities: exchange coefficients clip to ECHAM's
free-troposphere background, mixing length floors at 1 m, friction velocity at
0.01 m/s, TKE at the ECHAM lower bound. Tracer diffusion is a separate generic
term ``tracer_diffusion.py::TracerVerticalDiffusion`` that mixes an explicit
tracer list with the ``kh`` profile the TTE-TKE term publishes, via one batched
unconditionally-stable backward-Euler solve that conserves each tracer's column
mass exactly.

**What ECHAM/CAM does.** ICON/ECHAM6 ``vdiff`` (Brinkop & Roeckner 1995;
Mauritsen et al. 2007 total-turbulent-energy closure) — prognostic TKE,
Louis/Mauritsen surface-layer stability functions, and an implicit column solve
whose bottom row couples to the surface tiles through the Richtmyer–Morton
elimination (ECHAM6.3 ``vdiff.f90`` with ``mo_surface_land/ocean/ice``;
ICON's ``mo_vdiff_solver.f90``, ``mo_turbulence_diag.f90``). ECHAM diffuses every tracer with the heat exchange
coefficient ``cfh`` (the ``pxtte`` update in ``mo_vdiff_solver``); CAM diffuses
all constituents likewise.

**Why we differ.**
- `science` — the surface-layer exchange uses a Louis (1979) / Mauritsen (2007)
  form matching ECHAM/ICON to order of magnitude across the Richardson-number
  range, not a bit-exact reproduction of every ECHAM branch. Against ECHAM6.3's
  compiled ``precalc_land``, its unstable branch agrees to 0.1 %, and its stable
  branch (Mauritsen) gives 1.5× the heat and 2.1× the momentum exchange at
  Ri ≈ 0.3 (#982).
- `compute` — the TTE-TKE column solve covers only its fixed variable block
  (u, v, T, qᵥ, qc, qi, TKE, θᵥ variance), so aerosol/gas tracers are mixed by the
  separate ``TracerVerticalDiffusion`` term rather than in the same tridiagonal
  ; its boundaries are zero-flux (surface exchange is dry deposition's job).

**Status & known limitations.** ``TracerVerticalDiffusion`` is a no-op on the
first step — it reads the previous step's ``kh`` carry, which is seeded to
zero on step 0 (zero exchange coefficient, zero tendency)
and reads the previous step's ``kh`` carry, because vdiff runs after the aerosol
block in the ECHAM ordering. The closure that the buoyancy feeds is simplified:
the exchange coefficients use constant ``c_m = 0.4`` and ``c_h = 0.5``, the
Richardson number enters only as an ad hoc factor on the mixing length (1 when
unstable, falling to 0.1 as ``Ri`` goes from 0 to 0.25), the mixing length is
capped at a tenth of a fixed 1000 m boundary-layer height, and TKE and the
exchange coefficients are stored on full levels. ECHAM's differs on each count:
Louis stability functions of ``Ri``, the Blackadar mixing length with the
Holtslag-Boville asymptote and a diagnosed boundary-layer extension, and TKE and
the coefficients on the interfaces (*Interior stability*, below). That port is
post-v3 work, to be done with the land fixes (#672), and the #682 retune is done
against the current closure.

**Code pointers.**
- ``jcm/physics/vertical_diffusion/tte_tke/`` — ``vertical_diffusion.py``
  (``TteTkeVerticalDiffusion``, ``vertical_diffusion_column``),
  ``turbulence_coefficients.py`` (``compute_exchange_coefficients`` and the K
  floor), ``matrix_solver.py`` (``setup_matrix_system``,
  ``solve_tridiagonal_system``, ``couple_surface_tiles``),
  ``surface_layer.py`` (``compute_surface_exchange_coefficients_echam_louis``),
  ``tke_budget.py``, ``vertical_diffusion_types.py`` (the floors).
- ``jcm/physics/vertical_diffusion/tracer_diffusion.py`` —
  ``TracerVerticalDiffusion``, ``diffuse_tracers_implicit``.

## Interior stability

**What we do.** The buoyancy of every interior interface is ECHAM's
``zbuoy``: the liquid-water potential temperature ``θ_l = θ − (L/c_p)(θ/T)·x``
(``x = q_c + q_i``) and the total water ``q_t = q + x`` of the two adjacent full
levels, averaged to the interface with the layer masses
(``zsdep1 = Δp_k/(Δp_k + Δp_{k+1})``), weighted by the interface's cloud cover
``cc`` (the same mass-weighted average of the cover)::

    zbuoy = (∂θ_l/∂z · zdus1 + θ · zdus2 · ∂q_t/∂z) · g / θ_v
    zdus1 = cc · zmult5 + (1 − cc) · zmult1
    zdus2 = cc · zmult4 + (1 − cc) · vtmpc1

``zmult1 = 1 + vtmpc1·q_t`` and ``zmult5 = zmult1 − (L/c_p T · zmult1 − rv/rd) ·
(rd/rv)(L/R_d T) q_s / (1 + (rd/rv)(L/c_p T)(L/R_d T) q_s)`` carry the latent
heating of the condensation a displaced parcel undergoes, and ``zmult4 =
(L/c_p T)·zmult5 − 1``. ``q_s`` is the mass-weighted average of the full levels'
saturation humidities from ECHAM's ``ua`` table
(``jcm/physics/thermodynamics.py::saturation_specific_humidity``), not the
saturation at the mean temperature, and ``L`` is the condensation heat at and
above the melting point and the sublimation heat below it, averaged like the
rest. A saturated, cloudy layer therefore mixes on its moist-adiabatic
stability: a layer of uniform ``θ_l`` and ``q_t`` is neutral whatever its dry
static stability.

The Richardson number ``Ri = zbuoy / max(zshear, 10⁻⁵ s⁻²)`` and the TKE
source ``l (c_m·zshear − c_h·zbuoy)`` read this one ``zbuoy``, and the surface
layer's bulk Richardson number takes the lowest level's cover into the same
multipliers, on every tile. The cover is the Sundqvist diagnostic
(``SundqvistCloudFraction``, ECHAM's ``cover``) of the same step, which the
composition requires upstream of the vertical diffusion.

**What ECHAM/CAM does.** ``vdiff.f90::vdiff`` (r7492) l.658-700 forms
``zlteta1``, ``ztvir1``, ``zqss`` (from the ``ua`` table), and the half-level
averages ``zqssm``, ``ztmitte``, ``zfaxen``, ``zccover``; l.777-799 forms
``zbuoy``, ``zshear`` and ``zri = zbuoy/MAX(zshear, zepshr)``, with
``zepshr = 1e-5``. The TKE budget (l.837) uses the same ``zbuoy``. The
surface exchange routines (``mo_surface_land.f90::precalc_land`` l.224-236 and
the ocean and ice analogues) apply the multipliers to the lowest level with
``paclc(klev)``.

**Why we differ.**
- `science` — the closure's stability treatment is not ECHAM's. ECHAM sets the
  momentum and heat exchange from Louis (1979) stability functions ``zsm``,
  ``zsh`` of this Richardson number (l.819-832) and a Blackadar mixing length
  with the Holtslag-Boville asymptote (l.801-817) up to a PBL extension
  (l.737-761); the
  term here uses constant ``c_m``, ``c_h`` and an ad hoc factor on the mixing
  length that falls from 1 to 0.1 as ``Ri`` goes from 0 to 0.25
  (``compute_mixing_length``). The buoyancy that enters is ECHAM's; what is
  done with it is the simplified closure.
- `differentiability` — the latent heat switches at the melting point
  (``FSEL(T − tmelt, alv, als)``, a 13% jump), and the shear is floored under the
  denominator of ``Ri``. Both are ECHAM's exact values and are differentiated as
  they stand: the floor is a floor under a denominator, which stays in the value,
  and the phase switch is the class of jumps the ``ua`` table itself carries
  (see {doc}`../design/surrogate_gradients` and #843). ``cc`` weights the
  multipliers linearly, so the buoyancy is differentiable in the cover.

**Code pointers.**
- ``jcm/physics/vertical_diffusion/tte_tke/moist_buoyancy.py`` —
  ``interior_buoyancy_terms``, ``cloud_weighted_buoyancy_multipliers`` (the one
  implementation both the interior and the surface layer call),
  ``richardson_number``.
- ``jcm/physics/vertical_diffusion/tte_tke/turbulence_coefficients.py`` —
  ``compute_richardson_number``.
- ``jcm/physics/vertical_diffusion/tte_tke/surface_layer.py`` —
  ``surface_bulk_richardson``.
- ``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion.py`` —
  ``vertical_diffusion_column`` (the TKE source), ``TteTkeVerticalDiffusion``
  (reads ``diagnostics["clouds"]``).

**Validation evidence.**
``jcm/physics/vertical_diffusion/tte_tke/moist_buoyancy_test.py`` compares the
interior ``zbuoy``/``zshear``/``zri`` and their ingredients with the compiled
statements of ``vdiff.f90`` for 384 columns, and the surface Richardson number
with ``precalc_land``/``precalc_ocean``/``precalc_ice`` for 96 cells
(``jcm/data/test/echam_vdiff_reference/``): float64 agreement to the tables'
interpolation error (6.5e-14 of the largest value in ``zqss``) and 2e-15 in
the buoyancy and Richardson number; with the tables replaced by the Sonntag fit
they tabulate, to round-off. The limits are pinned: a layer of uniform ``θ_l``,
``q_t`` is neutral while dry-stable, the conditionally unstable layer's
Richardson number falls monotonically with the cover through zero, the dry
clear limit is ``(g/θ)∂θ/∂z``.

## Surface saturation and latent heat

**What we do.** The saturation specific humidity of every surface tile, and of
the air at the lowest level in the surface-layer Richardson number, is ECHAM's
``ua`` table: Sonntag (1990) over ice at and below the melting point and over
water above it (``jcm/physics/thermodynamics.py::saturation_specific_humidity``,
``phase="auto"``; see {doc}`constants`): the sea-ice tile (at
``min(SST, 271.38 K)``) always saturates over ice, frozen land does at and
below 273.15 K. The latent heat in the
surface-layer buoyancy is the condensation heat when the lowest-level air is at
or above the melting point and the sublimation heat below, and the
liquid-water potential temperature subtracts ``(L/c_p)·(θ/T)·q_x`` with the same
``L``. The latent heat of the delivered moisture flux is assembled per tile:
``alhc·E`` over open water, ``alhs·E`` over sea ice, and over land
``alhc·E + (alhs − alhc)·s·E_pot`` with ``s`` the snow-covered fraction (the
prescribed ``snowc_am``, glaciers fully covered) and ``E_pot`` the flux at full
wetness. The land's moisture flux carries JSBACH's humidity factors,
``ρ·C·(csat·q_s − cair·q̂_K)`` (see {doc}`surface`). The snow-covered share
evaporates at the potential rate within them, so the land flux always covers
the snow share the sublimation heat is charged to. The same factors set the
surface humidity the surface-layer buoyancy sees, ``csat·q_s + (1 − cair)·q_a``
(``precalc_land``). Each tile's latent heat comes from its own flux against its
own lowest-level value, and the reported latent heat is their fraction-weighted
sum, so it stays exactly consistent with the delivered moisture flux.

**What ECHAM/CAM does.** ECHAM's ``precalc_ocean``/``precalc_ice``/
``precalc_land`` read the tile saturation from the ``tlucua`` table
(``mo_echam_convect_tables``), which switches from water to ice at the melting
point with no mixed-phase blend, as does the lowest-level ``zqss`` in
``vdiff.f90``; ``zfaxe = FSEL(T − tmelt, alv, als)`` sets the latent heat in the
buoyancy and in ``zlteta1``. ``postproc_ice`` reports ``als·E`` and JSBACH
(``mo_soil.f90``) reports ``alv·E_T + (als − alv)·snow_fract·E_pot``, with the
snow fraction entering the land ``csat``/``cair`` as
``snow_fract + (1 − snow_fract)·(…)`` (``qsat_fact``).

**Why we differ.** The snow-covered fraction is the prescribed climatology
until snow is prognostic (#672), and JSBACH's humidity factors are built on
prescribed soil moisture with no wet skin (see {doc}`surface`).

**Code pointers.**
- ``jcm/physics/vertical_diffusion/tte_tke/surface_layer.py`` —
  ``compute_surface_exchange_coefficients_echam_louis``.
- ``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion.py`` —
  ``vertical_diffusion_column`` (the surface tiles),
  ``TteTkeVerticalDiffusion`` (sublimation fractions, JSBACH factors).
- ``jcm/physics/vertical_diffusion/tte_tke/matrix_solver.py`` —
  ``couple_surface_tiles``.

**Validation evidence.**
``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion_test.py`` —
``TestSurfaceTilePhase``: ``LH/E`` equals ``alhs`` over sea ice, ``alhc`` over
open water and ``alhc + (alhs − alhc)·s/w`` for a tile with factors ``w``; the
term's land follows JSBACH's factors (a dry bare soil does not evaporate, and
snow sublimates at the potential rate); and a column at the ice saturation of a
260 K tile exchanges no moisture. ``land_coupling_test.py``: a single tile
reproduces the Robin row exactly, and mixed tiles close the column budget
against their own fluxes.

## Ten-metre wind diagnostic

**What we do.** The term publishes the 10 m wind as a grid-mean speed and
vector, ``VerticalDiffusionData.wind_10m`` / ``wind_10m_u`` / ``wind_10m_v``,
and per tile (``wind_10m_tile`` / ``wind_10m_u_tile`` / ``wind_10m_v_tile``,
with the ``surface_fraction`` they are weighted by), alongside the exchange
coefficients. It is the ECHAM family's one 10 m wind: the surface-exchange
contract publishes it as ``wind_u``/``wind_v``, and the AeroCom ``uas``/``vas``
apply its grid-mean reduction (``wind_10m_reduction``) to the post-physics
lowest-level wind, the time level of the other AeroCom winds. It is
the surface-layer profile evaluated at 10 m rather than an interpolation
between levels: with ``bn = ln(z₁/z₀ₘ)`` the neutral profile factor and
``bm = bn·√(CMₙ|U| / CM|U|)`` its stability-corrected counterpart,

```
zrat = 10/z₁
cbn  = ln(1 + (e^bn − 1)·zrat)
red  = [cbn + (stable: −(bn − bm)·zrat | unstable: −ln(1 + (e^(bn−bm) − 1)·zrat))] / bm
```

and each tile's 10 m wind is ``red·U(z₁)`` component-wise; the grid mean is
their fraction-weighted sum, ``U(z₁)·Σ f·red``, so it keeps the lowest-level
direction. The
reduction is computed **inside** each surface-layer scheme's ``lax.cond``
branch, from that scheme's own neutral drag: the two schemes differ in
roughness (state-carried vs a hard-coded table), in the bound on ``z/z₀`` and
in whether the wind is ``zepdu2``-floored, and both branches build their
momentum drag from the same helper the reduction uses, so the pair cannot
drift. The profile factor uses the ``zepdu2``-floored speed the coefficients
themselves were built from (``max(|U|, 1 m/s)``), and the resulting reduction
multiplies the true wind.

**What ECHAM/CAM does.** ECHAM5 ``vdiff.f90`` / ICON
``mo_surface_diag::nsurf_diag`` compute the 10 m wind by exactly this
construction, from the ``pbn``/``pbm`` profile factors ``mo_turbulence_diag``
exports for the purpose, from the step-start wind ``pum1`` that
``vdiff.f90::update_surface`` passes to the surface (per tile ``zu10w = zred·pum1`` in
``mo_surface_ocean.f90::postproc_ocean`` and its ice/land analogues, box-averaged
by fraction in ``mo_surface.f90::surface_box_average`` into ``u10``/``v10``/
``wind10``); ``zepdu2 = 1 m²/s²`` is ECHAM's calm-wind floor on the
bulk Richardson number and the drag.

**Why we differ.** We do not. The stable/unstable branch is selected by
``CM|U| < CMₙ|U|`` rather than by the sign of ``Ri``, which is equivalent for
both schemes here (their stability factors are <1 stable, ≥1 unstable, and both
equal 1 — with equal stable and unstable expressions — at ``Ri = 0``) and keeps
the Richardson number inside the coefficient solve.

**Status & known limitations.** The wind is not taken relative to an ocean
surface current: ECHAM's open-water 10 m speed and stress use ``u − ocu``, and
jcm applies a zero current (``ForcingData.ocean_u``/``ocean_v`` are reserved
for it but not yet read, #915; see {doc}`../design/surface_exchange`). The Businger-Dyer
branch's neutral reference uses that scheme's own hard-coded von Kármán constant
rather than the live ``jcm.constants`` value, so a ``set_constants`` override
changes the drag and its neutral reference together.

**Code pointers.**
- ``jcm/physics/vertical_diffusion/tte_tke/surface_layer.py`` —
  ``wind_10m_reduction``, ``echam_louis_neutral_drag``.
- ``jcm/physics/vertical_diffusion/tte_tke/turbulence_coefficients.py`` —
  ``businger_dyer_neutral_drag``, ``compute_turbulence_diagnostics``.
- ``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion_types.py`` —
  ``VerticalDiffusionData``.
- ``jcm/physics/surface/echam/turbulent_fluxes.py`` —
  ``compute_surface_diagnostics`` (the reported 10 m wind and its components).

**Validation evidence.**
``jcm/physics/vertical_diffusion/tte_tke/surface_layer_test.py`` — the neutral
limit against a hand-computed log profile (0.912/0.904/0.894 for
z₀ = 5e-5/1.5e-4/5e-4 m at z₁ = 32.6 m), monotonicity in stability, continuity
across the branch, and that each scheme uses its own neutral drag.

**Validation evidence.**
``jcm/physics/vertical_diffusion/tte_tke/`` test suite —
``vertical_diffusion_test.py``, ``surface_layer_test.py``,
``boundary_layer_cases_test.py``, ``scm_boundary_layer_cases_test.py``;
``tracer_diffusion_test.py`` (column-mass conservation).
