"""``WetScavenging`` — in-cloud and below-cloud aerosol scavenging.

Removal pathways for aerosol, all differentiable, driven by the per-level
precipitation process rates the cloud microphysics schemes actually
integrate (``CloudData.precip_formation_rate`` / ``precip_evaporation_rate``,
#499; the carrier flux is their own cumulative ledger, which deliberately
excludes the sedimenting cloud-ice flux — see ``__call__``). With a prognostic
cloud-borne phase (``spec.cloud_borne``, #602) the stratiform in-cloud
pathway acts on the cloud-borne tracers at the full condensate→precip
conversion rate and the interstitial tracers keep in-cloud and below-cloud
impaction and convective processing; without one, the in-cloud pathway acts
on interstitial aerosol weighted by its per-mode activated fractions plus
the impaction fractions (the implicit M7/TOMAS-style treatment):

* **In-cloud nucleation scavenging** — aerosol residing in cloud droplets is
  removed at the fraction cloud condensate converted to precipitation this
  step, read from the cloud scheme's process-time scavenging ledger
  (``incloud_scavenged_fractions`` — HAMMOZ's ``peffwat``/``peffice``, #708),
  which stays alive in cells the microphysics emptied.
* **In-cloud impaction scavenging** — cloud droplets and ice crystals
  collect interstitial aerosol (ECHAM-HAM's ``ic_scav_imp``,
  ``wetdep.incloud_impaction``): per mode, phase and moment, a fraction from
  Croft et al.'s collision tables at the 2M scheme's own droplet and crystal
  effective radii (``reffl``/``reffi``) and, for ice, its crystal number,
  removed with the same per-phase condensate→precip conversion as the
  in-droplet aerosol. With the explicit phase it acts on the interstitial
  aerosol in the cloud; without it, HAM's nucleation + impaction sum,
  capped at one, acts on the in-cloud aerosol.
* **Below-cloud impaction scavenging** — precipitation falling through a
  layer collects interstitial aerosol at CAM's Slinn impaction coefficient
  (``wetdep.impaction``), separately for the number and mass moments. Both
  contributions use the per-level flux ENTERING each layer — stratiform
  from the microphysics ledger, convective from
  ``ConvectionData.precip_flux`` (the cuflx rain + snow budget) — so the
  collection rate follows the carrier that is actually falling there and
  washout is confined below where precip actually forms. The stratiform
  carrier is not cloud-weighted: CAM's swept precipitating volume cancels
  against the in-precip-area rain rate (see ``below_cloud_rate``). The
  convective carrier acts only in the fraction of the box it falls
  through — HAMMOZ's updraft area ``M_u/(ρ·w_u)`` (``conv_precip_cover``,
  jax-gcm#781) — and removes ``f_cu·(1 − exp(−Λ·Δt))`` of the layer's
  interstitial aerosol (``conv_below_cloud_rate``).
* **Convective in-cloud scavenging** — the convective mirror of the
  stratiform pathway: scavenging ratio × (per-layer updraft precip
  formation / in-updraft condensate), from ``ConvectionData``'s
  ``precip_formation`` (ECHAM ``pdmfup``) and ``qc_conv``/``qi_conv``;
  activatable modes only. With ``in_plume_convective=True`` this
  environment-profile pathway is retired: the composed
  ``ConvectiveTracerTransport`` scavenges inside the updraft instead
  (jax-gcm#621, HAMMOZ ``csr_conv·peff`` in a closed plume budget) and
  this term only folds
  the transport term's surface fluxes into the ``wet_*`` ledger.
* **Re-evaporation re-injection** — aerosol scavenged by the stratiform
  pathways is carried in the falling precip; where that precip evaporates
  or sublimates, the same fraction of the carried aerosol returns to the
  INTERSTITIAL phase (a droplet that evaporates releases its aerosol, so
  cloud-borne-scavenged material also re-enters as interstitial). The
  aerosol-in-precip flux is integrated top to bottom per tracer, exactly
  mirroring HAMMOZ ``mo_ham_wetdep``'s re-evaporation ledger; the flux
  reaching the bottom is the net surface wet deposition. Aerosol the
  convective carrier scavenges below cloud is released the same way by the
  convective precipitation's own evaporation fraction
  (``ConvectionData.precip_evap_fraction``, HAMMOZ ``prevap``).

``ConvectionData`` is read via ``diagnostics.get("convection")`` with a
zero-precip fallback so the term still composes without a convection
scheme. A cloud scheme that does not populate the per-level process rates
(both ECHAM schemes do) yields zero stratiform scavenging altogether — the
in-cloud rate, the impaction carrier and the re-injection ledger all derive
from those two fields.

Mirrors ``mo_hammoz_wetdep``.
"""

from __future__ import annotations

import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import tree_math
from flax import nnx
from jax.scipy.special import ndtr, ndtri

from jcm.physics.aerosol.jam.cloud_borne_store import (
    CARRY_KEY,
    LIQUID_SHARE_KEY,
    apply_updates,
    carry_mode,
    mirror_names,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.removal_split import split_view
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.aerosol.jam.wetdep.incloud_impaction import (
    VARIANTS as INCLOUD_IMPACTION_VARIANTS,
    crystal_impaction_fraction,
    droplet_impaction_fraction,
    impaction_radius_um,
)
from jcm.physics.aerosol.jam.wetdep.impaction import (
    IMPACT_SCALE_DEFAULT,
    MU_WATER_AIR_DEFAULT,
    bcscavcoef,
    build_impaction_table,
    table_log_coefficients,
)
from jcm.physics.convection.tracer_transport import release_scavenged
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

_EPS = 1.0e-30
# Physical floors for the re-injection budget's divisions. Values below
# these are dynamically negligible (a 1e-15 1/s removal rate is a ~30 Myr
# timescale; 1e-12 kg/m²/s is ~1e-4 mm/day), and a *physical* floor — not a
# tiny epsilon — keeps the guarded-division VJPs clear of the squared-
# underflow window in float32 (the double-where NaN class).
_RATE_FLOOR = 1.0e-15   # [1/s]
_FLUX_FLOOR = 1.0e-12   # [kg/m²/s]


@tree_math.struct
class WetDepParameters:
    """Tunable scavenging knobs (differentiable)."""

    incloud_scale: jnp.ndarray     # multiplies in-cloud removal (nucleation and impaction)
    sol_factb: jnp.ndarray         # below-cloud solubility factor [-]
    mu_water_air: jnp.ndarray      # water/air viscosity ratio, interception
    impact_scale: jnp.ndarray      # multiplies below-cloud inertial-impaction efficiency
    conv_scav_ratio: jnp.ndarray   # convective in-cloud scavenging ratio [-]
    conv_updraft_velocity: jnp.ndarray  # sets the convective precip footprint [m/s]

    @classmethod
    def default(cls) -> "WetDepParameters":
        # sol_factb: CAM's ``sol_factb_interstitial`` namelist default —
        # only the soluble part of interstitial aerosol is collected by
        # falling precip. CAM's un-set fallback is the mass-weighted
        # hygroscopicity of the mode, which every supported CAM
        # configuration overrides with this scalar.
        # mu_water_air / impact_scale: the two knobs on the Slinn collection
        # integral itself (interception viscosity ratio; inertial-impaction
        # efficiency). Defaults are CAM as written, so they are inert until
        # tuned; the knob-dependent part of the table is rebuilt per step.
        # conv_scav_ratio: fraction of soluble aerosol removed with the
        # condensate-to-precip conversion (HAMMOZ soluble-mode value).
        # conv_updraft_velocity: HAMMOZ's assumed in-cloud updraft speed;
        # the updraft mass flux divided by ρ·w_u is the updraft area, the
        # fraction of the box the convective precipitation falls through
        # (``conv_precip_cover``).
        return cls(
            incloud_scale=jnp.asarray(1.0),
            sol_factb=jnp.asarray(0.1),
            mu_water_air=jnp.asarray(MU_WATER_AIR_DEFAULT),
            impact_scale=jnp.asarray(IMPACT_SCALE_DEFAULT),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(CONV_UPDRAFT_VELOCITY_DEFAULT),
        )


#: HAMMOZ ``prep_wetdep_hydro``'s pool floor (``zmin = 1e-10``): below it a
#: ledger pool counts as absent and the scavenged fraction comes from the
#: formation-ledger marker instead of the division.
_LEDGER_POOL_MIN = 1.0e-10
#: Cap on a per-step scavenged fraction before converting it to an
#: equivalent decay rate via -log1p(-f)/dt: keeps the rate finite (~14/dt)
#: while the batched ``1 - exp(-rate*dt)`` still removes >0.999999 of the
#: tracer in fully-converting cells.
_FRACTION_CAP = 1.0 - 1.0e-6


def incloud_scavenged_fractions(
    clouds, dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-step in-cloud scavenged fractions from the process-time ledger.

    The port of HAMMOZ ``prep_wetdep_hydro`` (mo_hammoz_wetdep.f90:405-440):

    * ``f_wat = clip((zmratepr + zmsnowacl)·dt / zmlwc, 0, 1)`` — the
      fraction of the in-cloud liquid pool converted to precipitation
      this step (rain formation plus droplets rimed onto falling snow),
    * ``f_ice = clip(zmrateps·dt / zmiwc, 0, 1)`` — same for ice into
      snow (including the sedimenting-ice carrier ECHAM-HAM seeds it
      with),
    * ``pice`` — the ice mass fraction of the in-cloud pool, splitting a
      phase-agnostic tracer between the two fractions.

    Both numerator and denominator are captured at process time (the
    denominator BEFORE formation depletes it), so each fraction is
    bounded by construction — no unbounded rate ratio, no reliance on a
    floor to be "accidentally correct" in near-empty cells.

    One documented deviation from HAMMOZ (#708): where the pool is below
    HAMMOZ's ``zmin`` but the formation ledger is positive, HAMMOZ maps
    the fraction to 0 — missing the removal in exactly the cells the
    step fully converted to precipitation (the assembly zeroes
    zmlwc/zmiwc where the post-write-back cover fell below ``clc_min``).
    Here that marker maps to fraction 1: the droplets became rain, so
    everything they carried went with them. ``pice`` falls back to the
    formation-ledger split in the same cells.
    """
    il = jnp.maximum(clouds.incloud_liquid, 0.0)
    ii = jnp.maximum(clouds.incloud_ice, 0.0)
    form_wat = jnp.maximum(
        clouds.incloud_rain_formation + clouds.incloud_riming, 0.0) * dt
    form_ice = jnp.maximum(clouds.incloud_snow_formation, 0.0) * dt

    live_w = il > _LEDGER_POOL_MIN
    f_wat = jnp.where(
        live_w,
        jnp.clip(form_wat / jnp.maximum(il, _LEDGER_POOL_MIN), 0.0, 1.0),
        jnp.where(form_wat > 0.0, 1.0, 0.0),
    )
    live_i = ii > _LEDGER_POOL_MIN
    f_ice = jnp.where(
        live_i,
        jnp.clip(form_ice / jnp.maximum(ii, _LEDGER_POOL_MIN), 0.0, 1.0),
        jnp.where(form_ice > 0.0, 1.0, 0.0),
    )

    pool = il + ii
    form = form_wat + form_ice
    pice = jnp.where(
        pool > _LEDGER_POOL_MIN,
        ii / jnp.maximum(pool, _LEDGER_POOL_MIN),
        jnp.where(
            form > _LEDGER_POOL_MIN,
            form_ice / jnp.maximum(form, _LEDGER_POOL_MIN),
            0.0,
        ),
    )
    return f_wat, f_ice, pice


#: In-ice number fractions at or below this take a zero mass share, and at or
#: above ``1 - _ICE_SHARE_EDGE_HI`` a full one: the probit diverges at 0 and 1.
#: At 1e-12 the HAM mass share of a σ = 1.8 mode is ~1e-7, so the cut is below
#: anything the exchange or the removal can resolve.
_ICE_SHARE_EDGE_LO = 1.0e-12
_ICE_SHARE_EDGE_HI = 1.0e-6

#: Number mixing ratio [1/kg] below which a mode holds no particles to put in
#: the ice (a physical floor, far below any real population, for the f32 VJP of
#: the division).
_NUMBER_FLOOR = 1.0e-3


def ice_phase_fractions(spec: ModalAerosolSpec, number, icnc):
    """Per-mode share of the in-cloud aerosol inside ice crystals (HAM rule).

    The port of ECHAM-HAM ``mo_ham_wetdep.f90::ic_scav_nuc`` for the ice phase
    (``nwetdep = 3``): every crystal holds one aerosol particle, and the
    crystals are filled from the largest activatable mode down. The number
    share of mode ``m`` is

        f_n(m) = clip((ICNC − Σ_{modes larger than m} N) / N_m, 0, 1),

    and the particles in the ice are the largest of the mode: the mass share
    is the log-normal mass tail beyond the radius whose number tail is
    ``f_n``, ``Φ(Φ⁻¹(f_n) + 3 ln σ)`` — the radius-free form of HAM's
    ``ham_m7_invertlogtail``/``ham_m7_logtail`` pair. Modes that cannot
    activate (HAM's insoluble modes) take none.

    Args:
        spec: the modal population.
        number: per-mode total number mixing ratio [1/kg] (interstitial plus,
            with an explicit phase, cloud-borne), ordered like ``spec.modes``.
        icnc: in-cloud ice crystal number [1/kg].

    Returns:
        ``(f_number, f_mass)``: per-mode lists like ``number``.

    """
    order = sorted(
        (i for i, m in enumerate(spec.modes) if m.can_activate),
        key=lambda i: spec.modes[i].dgnum, reverse=True,
    )
    zeros = jnp.zeros_like(icnc)
    f_number = [zeros for _ in spec.modes]
    f_mass = [zeros for _ in spec.modes]
    remaining = jnp.maximum(icnc, 0.0)
    for i in order:
        n = jnp.maximum(number[i], 0.0)
        has = n > _NUMBER_FLOOR
        f = jnp.where(has, jnp.clip(remaining / jnp.where(has, n, 1.0),
                                    0.0, 1.0), 0.0)
        remaining = jnp.maximum(remaining - n, 0.0)
        inner = (f > _ICE_SHARE_EDGE_LO) & (f < 1.0 - _ICE_SHARE_EDGE_HI)
        f_safe = jnp.where(inner, f, 0.5)
        shift = 3.0 * math.log(spec.modes[i].geom_std_dev)
        tail = ndtr(ndtri(f_safe) + shift)
        f_number[i] = f
        f_mass[i] = jnp.where(inner, tail, jnp.where(f >= 0.5, 1.0, 0.0))
    return f_number, f_mass


def post_microphysics_icnc(state, diagnostics, dt):
    """In-cloud ice crystal number [1/kg] as the cloud microphysics leaves it.

    The two-moment scheme's ``qni`` tracer (ECHAM's ``idt_icnc``, in-cloud
    crystals per kg of air) advanced by the tendency the terms upstream of
    the caller have accumulated this step — the ``t+dt`` value HAM's wet
    deposition reads after the cloud microphysics. Only the two-moment
    scheme carries crystal numbers, and neither the ice share of the
    in-cloud aerosol nor the crystals' impaction can be formed without
    them, so a host without ``qni`` is refused rather than read as
    crystal-free (the ECHAM factory already requires ``cloud_scheme='2m'``
    for JAM). A structural probe with no tracers seeded returns zeros.
    """
    qni = state.tracers.get("qni")
    if qni is None and not state.tracers:
        return jnp.zeros_like(state.temperature)
    if qni is None:
        raise ValueError(
            "JAM's in-cloud aerosol split and in-cloud impaction need the "
            "two-moment cloud scheme's ice crystal number (tracer 'qni'): "
            "the ice share is HAM's one-particle-per-crystal rule and the "
            "crystals collect aerosol by impaction. Compose JAM with "
            "cloud_scheme='2m'."
        )
    run = diagnostics.get("_tendency_run")
    dqni = None if run is None else run.get("tracers", {}).get("qni")
    if dqni is not None:
        qni = qni + dt * dqni
    return jnp.maximum(qni, 0.0)


def incloud_effective_radii(state, diagnostics):
    """``(reffl, reffi)``: the 2M scheme's in-cloud droplet and crystal radii [um].

    ECHAM's ``mo_activ`` ``REFFL``/``REFFI`` streams, which HAM's in-cloud
    impaction reads: the two-moment microphysics' own effective radii of the
    condensate it leaves (``preffl``/``preffi``), 0 where a level holds no
    liquid or ice cloud. They are not the radii the radiation used
    (``clouds.r_eff_*``), which come from the step-start condensate by the
    radiation's own laws and are held between radiation steps. Only the
    two-moment scheme publishes them, so a host without them is refused
    rather than read as cloud-free; a structural probe with no tracers
    seeded returns zeros.
    """
    reffl = diagnostics.get("reffl")
    reffi = diagnostics.get("reffi")
    if reffl is not None and reffi is not None:
        return reffl, reffi
    if not state.tracers:
        zeros = jnp.zeros_like(state.temperature)
        return zeros, zeros
    raise ValueError(
        "JAM's in-cloud impaction scavenging needs the two-moment cloud "
        "scheme's in-cloud effective radii (diagnostics 'reffl' and "
        "'reffi'), which only Lohmann2MMicrophysics publishes. Compose JAM "
        "with cloud_scheme='2m', or build WetScavenging with "
        "incloud_impaction='none'."
    )


def phase_split(clouds, dt):
    """``(f_wat, f_ice, p_ice)``: HAM's in-cloud scavenged fractions and phase split.

    ``incloud_scavenged_fractions`` from the process-time ledger. Where that
    ledger holds neither a pool nor a formation (a cell the microphysics left
    covered but recorded no in-cloud condensate for), the ice share is taken
    from the cloud scheme's grid-mean condensate ``qi / (qc + qi)`` instead of
    defaulting to all-liquid, so an ice cloud is never read as a droplet cloud.
    """
    f_wat, f_ice, pice = incloud_scavenged_fractions(clouds, dt)
    pool = jnp.maximum(clouds.incloud_liquid, 0.0) + jnp.maximum(
        clouds.incloud_ice, 0.0)
    form = dt * (
        jnp.maximum(clouds.incloud_rain_formation + clouds.incloud_riming, 0.0)
        + jnp.maximum(clouds.incloud_snow_formation, 0.0))
    ledger = (pool > _LEDGER_POOL_MIN) | (form > _LEDGER_POOL_MIN)
    qc = jnp.maximum(clouds.qc, 0.0)
    qi = jnp.maximum(clouds.qi, 0.0)
    cond = qc + qi
    has_cond = cond > _LEDGER_POOL_MIN
    pice_cond = jnp.where(has_cond, qi / jnp.where(has_cond, cond, 1.0), 0.0)
    return f_wat, f_ice, jnp.where(ledger, pice, pice_cond)


def fraction_to_rate(fraction: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """Equivalent first-order decay rate removing ``fraction`` over ``dt``.

    The batched scavenging update applies ``1 - exp(-rate·dt)``, so the
    exact inverse is ``-log1p(-f)/dt``; the cap keeps it finite at f = 1
    (fully-converting cells) while still removing >0.999999 of the tracer.
    """
    f = jnp.clip(fraction, 0.0, _FRACTION_CAP)
    return -jnp.log1p(-f) / dt


def below_cloud_rate(
    precip_flux: jnp.ndarray,  # (nlev, ncols) precip falling through [kg/m²/s]
    scav_coef: jnp.ndarray,    # Slinn impaction coefficient [1/mm]
    params: WetDepParameters,
) -> jnp.ndarray:
    """Below-cloud impaction scavenging rate [1/s].

    CAM ``wetdepa_v2``'s below-cloud term reduces to
    ``Λ = sol_factb · Λ₁(D_wet) · R``: the swept volume ``cldv`` cancels
    against the in-precip-area rain rate inside ``odds``, so it acts on the
    whole grid-mean interstitial tracer with no cloud weighting. ``Λ₁`` is
    the Slinn collection-efficiency integral over the raindrop and aerosol
    size distributions (``impaction.bcscavcoef``, per moment) in 1/mm, and
    ``R`` the precipitation flux in kg/m²/s ≡ mm/s. Slinn's efficiency is
    capped at 1, so ``Λ₁`` saturates at the rain's geometric sweep-out rate
    rather than growing without bound with particle size.

    ``precip_flux`` is the local flux entering each layer from above —
    a per-level profile for both the stratiform and convective carriers.
    """
    return params.sol_factb * scav_coef * jnp.maximum(precip_flux, 0.0)


#: HAMMOZ ``prep_wetdep_hydro``'s assumed in-cloud updraft velocity
#: (``zwu = 2.0`` m/s, mo_hammoz_wetdep.f90:410): the updraft mass flux
#: divided by ``ρ·w_u`` is the updraft area, which HAMMOZ takes as the
#: fraction of the grid box the convective precipitation falls through.
CONV_UPDRAFT_VELOCITY_DEFAULT = 2.0


def conv_precip_cover(
    mass_flux_up: jnp.ndarray,   # (nlev, *horiz) updraft flux at each layer's top [kg/m²/s]
    ktype: jnp.ndarray,          # (*horiz) convection type (3 = mid-level)
    layer_mass: jnp.ndarray,     # (nlev, *horiz) ρ·Δz [kg/m²]
    air_density: jnp.ndarray,    # (nlev, *horiz) [kg/m³]
    updraft_velocity: jnp.ndarray,  # [m/s]
) -> jnp.ndarray:
    """Fraction of the grid box the convective precipitation falls through.

    HAMMOZ ``mo_hammoz_wetdep.f90::prep_wetdep_hydro`` takes the updraft
    area as the precipitating fraction of the box for the convective wet
    deposition call, estimating it from the mass flux with a prescribed
    updraft velocity: ``zclceff = pmfu / (zwu·prhou)``. The mass flux it
    sees is the one ECHAM ``cuflx`` hands to ``cuflx_subm``: the plume's
    own profile through the cloud and, below the cloud base, a linear
    decrease in pressure from the cloud-base value to zero at the surface
    (``pmfu(jk) = pmfu(kcbot)·zzp``, ``zzp = (p_s − p_half(jk)) /
    (p_s − p_half(kcbot))``, squared for mid-level convection;
    mo_cufluxdts.f90:233-239) — the updraft draws its air from the whole
    sub-cloud layer, so the shaft below the base keeps its footprint and
    tapers to the surface. ``ConvectionData.mass_flux_up`` already carries
    that taper (the Tiedtke term publishes the flux its ledger uses), so
    the shared helper's reconstruction of it from the layer masses —
    ``p_s − p_half(k) = g·Σ_{j≥k} m_j`` below the lowest interface with a
    non-zero flux — is then a no-op; it keeps a plume-only profile correct
    too. Levels are top-first. With the JAM chain composed (ECHAM's
    ``lham``), the convection scheme's sub-cloud rain evaporation uses this
    same area as its footprint.

    One documented deviation: HAMMOZ divides by the UPDRAFT density
    ``zrhou = p/(rd·ptu)`` (mo_cufluxdts.f90:406-407); ``ConvectionData``
    publishes no updraft temperature, so the environment density stands
    in. ``f_cu`` is therefore low by ``(T_u − T_env)/T_env`` — a few K over
    ~280 K in the plume core, under 2 % — against an assumed ``w_u`` that is
    itself the dominant uncertainty of the estimate.

    The area itself is computed by the shared
    :func:`~jcm.physics.convection.tiedtke_nordeng.flux_tendencies.updraft_area_cover`
    — the single implementation the convection scheme's own sub-cloud rain
    evaporation also uses (jax-gcm#812) — so the wet-deposition and
    evaporation footprints cannot drift apart. That helper floors the
    velocity at a physical 0.01 m/s so the division stays clear of the
    float32 squared-underflow window when the differentiable ``w_u`` is
    driven towards zero (see the ``_RATE_FLOOR`` note above).

    Clipped to [0, 1] here: HAMMOZ does not clip, but an updraft area above
    the whole box is a closure artefact, not a cover. (The evaporation
    caller keeps the raw area, as ``cuflx`` does.)
    """
    from jcm.physics.convection.tiedtke_nordeng.flux_tendencies import (
        updraft_area_cover,
    )
    # HAMMOZ uses the environment density as a stand-in for ``zrhou``; the
    # taper weight is the layer air mass (the 10 m ``layer_thickness`` floor
    # never binds on the supported level sets, so it equals ECHAM's Δp
    # ratio). The shared helper does not clip; a cover fraction is [0, 1].
    return jnp.clip(
        updraft_area_cover(
            mass_flux_up, air_density, ktype, layer_mass, updraft_velocity,
        ),
        0.0, 1.0,
    )


def conv_below_cloud_rate(
    precip_flux: jnp.ndarray,   # (nlev, *horiz) convective precip entering [kg/m²/s]
    precip_cover: jnp.ndarray,  # (nlev, *horiz) fraction of the box it falls through
    scav_coef: jnp.ndarray,     # Slinn impaction coefficient [1/mm]
    params: WetDepParameters,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Convective below-cloud impaction as an equivalent decay rate [1/s].

    HAMMOZ ``ham_wetdep`` (mo_ham_wetdep.f90:434-437) removes
    ``q_ambient · pclc · (1 − exp(−Λ·Δt))`` from a layer: the exponential
    removal happens inside the precipitating fraction ``pclc`` of the box
    and nowhere else, so a step can take at most that fraction of the
    layer's aerosol, and ``Λ`` is looked up at the grid-mean flux itself
    (``bc_rain`` feeds ``pfrain`` straight into the coefficient). For the
    convective carrier ``pclc`` is the updraft area (``conv_precip_cover``,
    jax-gcm#781). Returned as the first-order rate that removes exactly that
    fraction over ``Δt``, so it composes with the other pathways in the
    batched exponential update; for ``Λ·Δt ≪ 1`` it is ``cover · Λ``.

    This is HAMMOZ's form, not CAM's: ``wetdepa_v2`` rescales the rain rate
    to the precipitating area so the area cancels (``below_cloud_rate``,
    kept for the stratiform carrier). The two references differ by the
    factor ``cover`` for the convective carrier; see
    ``docs/source/design/jam_aerosol_removal.md``.
    """
    removed = -jnp.expm1(-below_cloud_rate(precip_flux, scav_coef, params) * dt)
    return fraction_to_rate(precip_cover * removed, dt)


def conv_in_cloud_rate(
    precip_formation: jnp.ndarray,  # (nlev, *horiz) updraft precip gen [kg/m²/s]
    conv_condensate: jnp.ndarray,   # (nlev, *horiz) in-updraft qc+qi [kg/kg]
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    params: WetDepParameters,
) -> jnp.ndarray:
    """Convective in-cloud (nucleation) scavenging rate [1/s].

    The convective mirror of the stratiform in-cloud pathway: scavenging
    ratio × (local
    condensate→precip conversion rate / in-updraft condensate), with the
    per-layer formation flux converted to a mixing-ratio rate by ρ·Δz.
    Zero wherever the updraft carries no condensate.
    """
    local_form = jnp.maximum(precip_formation, 0.0) / (
        air_density * layer_thickness
    )
    qcond = jnp.maximum(conv_condensate, _EPS)
    rate = params.conv_scav_ratio * local_form / qcond
    return jnp.where(conv_condensate > 1.0e-12, rate, 0.0)


def reinjection_budget(
    scavenged_below: jnp.ndarray,   # (K, nlev, ncols) impaction removal [kg/m²/s]
    scavenged_formed: jnp.ndarray,  # (K, nlev, ncols) in-cloud removal [kg/m²/s]
    evap_fraction: jnp.ndarray,     # (nlev, ncols) incoming-precip fraction evaporating
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Aerosol-in-precip ledger: re-injected flux per layer + surface flux.

    Integrates the scavenged-aerosol flux top to bottom, honouring the
    cloud schemes' own flux ordering (evaporation is capped by the
    INCOMING precip, formation is added after): aerosol impacted out by
    the falling precip within a layer (``scavenged_below``) joins the
    incoming carrier BEFORE the release — a fully-evaporating virga layer
    releases it rather than surface-depositing it through dry air — while
    aerosol scavenged into precip NEWLY FORMED in the layer
    (``scavenged_formed``) joins AFTER, because the incoming carrier's
    evaporation cannot touch precip that is only now forming and heading
    down. Returns ``(reinjected, surface_flux)`` with ``reinjected``
    shaped like the inputs and ``surface_flux`` ``(K, ncols)``, satisfying
    column conservation
    ``sum_k(scavenged_below + scavenged_formed - reinjected) =
    surface_flux`` exactly.
    """
    def step(carried, xs):
        s_below_k, s_form_k, e_k = xs        # (K, ncols) x2, (ncols,)
        incoming = carried + s_below_k
        released = incoming * e_k[jnp.newaxis, :]
        carried = incoming - released + s_form_k
        return carried, released

    k, _, ncols = scavenged_below.shape
    surface, released = jax.lax.scan(
        step,
        jnp.zeros((k, ncols), scavenged_below.dtype),
        (
            jnp.moveaxis(scavenged_below, 1, 0),
            jnp.moveaxis(scavenged_formed, 1, 0),
            evap_fraction,
        ),
    )
    return jnp.moveaxis(released, 0, 1), surface


class WetScavenging(PhysicsTerm):
    """In-cloud + below-cloud scavenging with re-evaporation re-injection."""

    name: ClassVar[str] = "jam_wet_deposition"
    category: ClassVar[str] = "aerosol_wetdep"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "activated_fraction", "clouds",
        "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: WetDepParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
        in_plume_convective: bool = False,
        incloud_impaction: str = "ham",
    ):
        """Hold params and the population.

        ``in_plume_convective``: the composed ``ConvectiveTracerTransport``
        scavenges soluble aerosol inside the updraft (jax-gcm#621), so
        this term drops its own environment-profile convective in-cloud
        pathway (``conv_in_cloud_rate``) to avoid double counting —
        keeping convective below-cloud washout, which is a distinct
        (impaction) pathway — and folds the transport term's published
        surface fluxes into the AeroCom ``wet_*`` ledger.

        The stratiform in-cloud pathways read the cloud scheme's
        process-time scavenging ledger from ``CloudData``
        (``incloud_scavenged_fractions`` — the HAMMOZ ``cloud_subm``
        interface), so this term requires a cloud scheme that publishes
        it; the physics factory enforces that at compose time.

        ``incloud_impaction`` selects the in-cloud impaction scavenging of
        interstitial aerosol by droplets and ice crystals (ECHAM-HAM's
        ``ic_scav_imp``, see ``wetdep.incloud_impaction``): ``"ham"`` (the
        default, with the three r7492 table-lookup defects corrected),
        ``"ham_r7492"`` (the compiled r7492 lookup, for like-for-like
        comparison with ECHAM-HAM), or ``"none"`` (no in-cloud impaction, as
        in CAM, whose interstitial in-cloud solubility factor is zero). Any
        but ``"none"`` reads the 2M scheme's effective radii, so the term
        then requires ``reffl``/``reffi``.
        """
        if incloud_impaction not in (*INCLOUD_IMPACTION_VARIANTS, "none"):
            raise ValueError(
                f"incloud_impaction={incloud_impaction!r}: choose one of "
                f"{(*INCLOUD_IMPACTION_VARIANTS, 'none')}."
            )
        self.params = nnx.Param(params or WetDepParameters.default())
        self._in_plume_convective = in_plume_convective
        self._incloud_impaction = incloud_impaction
        self._spec = spec or MAM4_SPEC
        # Per-mode Slinn impaction tables, built once here exactly as CAM
        # builds them in ``modal_aero_bcscavcoef_init`` (the 50x51 double
        # integral is far too costly to evaluate per cell per step). CAM
        # tabulates against the mode's first-species material density.
        self._impaction_tables = tuple(
            build_impaction_table(
                mode.dgnum, mode.geom_std_dev,
                self._spec.species_props(mode.species[0]).density,
            )
            for mode in self._spec.modes
        )
        if carry_mode(self._spec):
            # In carry mode the store term must run upstream each step
            # (name-set fixing + vertical mixing); requiring its key makes
            # _validate_ordering enforce that, instead of apply_updates
            # silently seeding an unmixed, unmanaged dict.
            self.requires = (*self.requires, CARRY_KEY)
        if incloud_impaction != "none":
            self.requires = (*self.requires, "reffl", "reffi")

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        aer = diagnostics["_jam_state"]
        activated_fraction = diagnostics["activated_fraction"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        dm = air_density * dz
        # Timestep for the implicit (exponential) scavenging update below.
        dt = diagnostics.get("_dt_seconds", 1800.0)

        clouds = diagnostics["clouds"]
        # Per-level stratiform process rates from the cloud scheme (#499):
        # the true local condensate→precip conversion and the falling-precip
        # evaporation. The carrier flux for impaction and the re-evap ledger
        # is rebuilt as the cumulative (formation − evaporation) integral of
        # those same rates rather than read from ``clouds.rain_flux +
        # snow_flux``: the schemes fold the sedimenting cloud-ice flux into
        # the frozen profile at interior levels, and that ice is not a
        # scavenging carrier (its sublimation is likewise excluded from
        # ``precip_evaporation_rate``), so using the profiles directly would
        # both drive impaction with a non-carrier flux and dilute the
        # re-evaporation fraction under cirrus. Melt only moves mass between
        # rain and snow, so the summed ledger is exact for the actual
        # precip; the floor guards accumulated round-off. Note the
        # ice-sedimentation flux that reaches the surface as snow therefore
        # carries no aerosol removal at all — a real missing sink, accepted
        # with the non-carrier stance.
        p_form = jnp.maximum(clouds.precip_formation_rate, 0.0)
        p_evap = jnp.maximum(clouds.precip_evaporation_rate, 0.0)
        strat_flux = jnp.maximum(
            jnp.cumsum((p_form - p_evap) * dm, axis=0), 0.0
        )
        flux_in = jnp.concatenate(
            [jnp.zeros_like(strat_flux[:1]), strat_flux[:-1]], axis=0
        )
        # Fraction of the precip entering a layer that evaporates within it,
        # guarded on a physical flux floor (see _FLUX_FLOOR note above).
        evap_fraction = jnp.where(
            flux_in > _FLUX_FLOOR,
            jnp.clip(p_evap * dm / jnp.maximum(flux_in, _FLUX_FLOOR), 0.0, 1.0),
            0.0,
        )

        # Convective precipitation (Tiedtke). Zero-precip fallback keeps the
        # term composable without a convection scheme (see module docstring).
        conv = diagnostics.get("convection")
        if conv is None:
            conv_flux_in = jnp.zeros_like(state.temperature)
            conv_cover = jnp.zeros_like(state.temperature)
            rate_conv_incloud = jnp.zeros_like(state.temperature)
            conv_evap_fraction = jnp.zeros_like(state.temperature)
        else:
            # HAMMOZ ``prevap`` of the convective precipitation: the
            # convectively scavenged aerosol it carries returns to the
            # environment where it evaporates (below).
            conv_evap_fraction = conv.precip_evap_fraction
            # Local carrier flux for impaction — the convective precip
            # falling into this layer, as in ECHAM xtwetdep / CAM. It is
            # zero above the first precip-forming level, which is itself
            # the cloud-top confinement.
            conv_flux_in = conv.precip_flux
            # The fraction of the box that flux falls through: HAMMOZ's
            # updraft area (jax-gcm#781). The impaction below acts on the
            # grid-mean working copy, standing in for HAMMOZ's environment
            # value ``pxtenh``; the two differ by O(cover). ``dm`` is the
            # layer mass this term already integrates with; the taper is a
            # ratio of such masses, and the 10 m floor on layer_thickness
            # never binds on the supported level sets (65 m minimum on
            # L47/L95), so it equals the Δp ratio ECHAM uses.
            conv_cover = conv_precip_cover(
                conv.mass_flux_up, conv.ktype, dm, air_density,
                params.conv_updraft_velocity,
            )
            conv_condensate = conv.qc_conv + conv.qi_conv
            if self._in_plume_convective:
                # Retired here: the transport term removes inside the
                # ascent scan (see __init__ docstring).
                rate_conv_incloud = jnp.zeros_like(state.temperature)
            else:
                rate_conv_incloud = conv_in_cloud_rate(
                    conv.precip_formation, conv_condensate,
                    air_density, dz, params,
                )

        # Stratiform in-cloud (nucleation) scavenging rates from the
        # process-time ledger (#708): the per-step scavenged fraction of an
        # in-droplet tracer is HAMMOZ's peffwat/peffice split by the phase
        # mass fraction — bounded by construction, and alive in cells the
        # microphysics emptied (post-write-back cover 0). The in-droplet
        # SHARE of interstitial aerosol is keyed to the cover the
        # processes actually ran under (process_cloud_fraction), not the
        # post-write-back cover, for the same reason. The removal is
        # linear in the activated fraction, so keep the unit-fraction base
        # and apply per-mode, per-quantity fractions below: ARG's number
        # and mass fractions differ a lot (large particles activate
        # preferentially) and vary by mode. The aggregate fraction is kept
        # only as a fallback for standalone composition without ARG
        # upstream.
        #
        # HAM scavenges the in-cloud aerosol's two phases apart
        # (``mo_ham_wetdep.f90::ham_wetdep``): the water part, by its
        # activated fraction, at the liquid conversion ``f_wat``; the ice
        # part, by the one-particle-per-crystal share (``ic_scav_nuc``), at
        # the ice conversion ``f_ice``. With an explicit phase the exchange
        # term records each reservoir's droplet-held share; without one the
        # same split is formed here from the crystal number.
        f_wat, f_ice, pice = phase_split(clouds, dt)
        f_comb = (1.0 - pice) * f_wat + pice * f_ice
        rate_ledger = fraction_to_rate(f_comb, dt)
        cf_proc = jnp.clip(clouds.process_cloud_fraction, 0.0, 1.0)
        rate_ic_unit = params.incloud_scale * cf_proc * rate_ledger
        jam_act = diagnostics.get("_jam_activation")
        liquid_share = diagnostics.get(LIQUID_SHARE_KEY) or {}

        def phase_rate(share):
            """Return the removal rate of a reservoir with droplet-held ``share``."""
            return params.incloud_scale * fraction_to_rate(
                share * f_wat + (1.0 - share) * f_ice, dt)

        # Build per-tracer scavenging rates and stack with the matching
        # tracers, so the elementwise removal runs as one batched op (rather
        # than an unrolled tendency per mode×species). Stratiform and
        # convective rates are kept separate: only stratiform-scavenged
        # aerosol enters the re-evaporation ledger (its carrier's per-level
        # evaporation is known). ``state.tracers`` is empty during
        # ``Model.get_empty_data``'s structural probe, so fall back to zeros
        # there (real runs have every declared tracer seeded).
        zeros = jnp.zeros_like(state.temperature)
        # Operator splitting: scavenge what sedimentation and dry deposition
        # left, not the step-start state (see ``removal_split``).
        view = split_view(self._spec, state, diagnostics)
        # Removal reads are floored at 0: spectral ringing leaves negative
        # lobes on near-zero tracers, and a removal rate applied to a
        # negative value INJECTS mass — the 30-day storage A/B measured
        # the advected cloud-borne fields being pumped net-negative by
        # exactly this interaction.
        names: list[str] = []
        # Interstitial destination for each stacked tracer's re-injected
        # aerosol: itself for interstitial tracers; the interstitial partner
        # for cloud-borne ones (an evaporated droplet releases its aerosol
        # to the interstitial phase).
        reinject_to: list[str] = []
        q_list: list[jnp.ndarray] = []
        # Stratiform removal splits by carrier relationship (see
        # ``reinjection_budget``): impaction into the INCOMING precip vs
        # in-cloud scavenging into precip FORMED here.
        rate_below_strat: list[jnp.ndarray] = []
        rate_form_strat: list[jnp.ndarray] = []
        rate_conv: list[jnp.ndarray] = []
        # With a prognostic cloud-borne phase (``spec.cloud_borne``, #602) the
        # stratiform in-cloud (nucleation) pathway belongs to the cloud-borne
        # tracers, which sit in the droplets by definition: they are removed
        # at the full condensate→precip conversion rate, and the interstitial
        # tracers keep only impaction and convective processing (activated
        # aerosol first transfers via ``CloudBorneExchange``, then rains
        # out). Without it, the implicit treatment stands — the interstitial
        # tracers are scavenged by their per-mode activated fractions.
        explicit_cb = self._spec.cloud_borne
        implicit_split = not explicit_cb and jam_act is not None
        impaction = self._incloud_impaction != "none"
        if implicit_split or impaction:
            icnc = post_microphysics_icnc(state, diagnostics, dt)
        if implicit_split:
            ice_number, ice_mass = ice_phase_fractions(
                self._spec,
                [jnp.maximum(view.get(number_name(m.short), zeros), 0.0)
                 for m in self._spec.modes],
                icnc)
        if impaction:
            # In-cloud impaction (ECHAM-HAM ``ic_scav_imp``): the 2M scheme's
            # own droplet/crystal effective radii and the crystal number
            # concentration HAM forms as ``pxtp1c(idt_icnc)*prhop1`` (the
            # in-cloud ICNC after the microphysics, not cloud-weighted).
            reffl, reffi = incloud_effective_radii(state, diagnostics)
            icnc_m3 = icnc * air_density

        def impaction_fractions(mode, r_wet, moment):
            """HAM's per-phase impaction fractions of one mode and moment."""
            mr = impaction_radius_um(r_wet, mode.geom_std_dev, moment)
            return (
                droplet_impaction_fraction(
                    reffl, mr, moment, self._incloud_impaction),
                crystal_impaction_fraction(
                    reffi, icnc_m3, mr, dt, self._incloud_impaction),
            )

        def impaction_rate(imp):
            """Return the removal rate of interstitial aerosol by in-cloud impaction alone.

            The interstitial aerosol in the cloudy part of the box
            (``cf_proc`` of the grid mean: the interstitial aerosol is taken
            as uniform across clear and cloudy air, as the exchange takes
            it), split between the phases by ``pice`` as HAM splits its
            in-cloud aerosol, collected at each phase's impaction fraction
            and removed with that phase's condensate→precip conversion
            (``ic_scav``: ``pxt·pfrac·peff``).
            """
            imp_w, imp_i = imp
            return params.incloud_scale * fraction_to_rate(
                cf_proc * ((1.0 - pice) * imp_w * f_wat
                           + pice * imp_i * f_ice), dt)

        for i, mode in enumerate(self._spec.modes):
            # Number and mass ride different moments of the same lognormal,
            # so CAM tabulates and applies a separate impaction coefficient
            # for each (``scavcoefnv`` jnv=1 number / jnv=2 volume).
            table = self._impaction_tables[i]
            ln_num, ln_vol = table_log_coefficients(
                table, params.mu_water_air, params.impact_scale)
            coef_num, coef_mass = bcscavcoef(
                aer.r_wet[i], table.dgnum, ln_num, ln_vol)
            below_strat_num = below_cloud_rate(flux_in, coef_num, params)
            below_strat_mass = below_cloud_rate(flux_in, coef_mass, params)
            below_conv_num = conv_below_cloud_rate(
                conv_flux_in, conv_cover, coef_num, params, dt,
            )
            below_conv_mass = conv_below_cloud_rate(
                conv_flux_in, conv_cover, coef_mass, params, dt,
            )
            # In-cloud impaction (HAM ``ic_scav_imp``) acts on every mode:
            # collision needs no solubility, and HAM computes it for every
            # mode, insoluble ones included.
            if impaction:
                imp_num = impaction_fractions(mode, aer.r_wet[i], "number")
                imp_mass = impaction_fractions(mode, aer.r_wet[i], "mass")
            # In-cloud nucleation only removes from activatable (soluble)
            # modes — and only implicitly (via the activated fraction) when
            # there is no explicit cloud-borne phase to carry it. Convective
            # processing always acts on interstitial (updrafts ingest
            # environment air).
            if mode.can_activate and not explicit_cb:
                if jam_act is not None:
                    # HAM's implicit form of the same split: the cloudy part
                    # of the box, times each phase's in-cloud share times
                    # that phase's conversion, removed this step — the mass
                    # the explicit phase removes at exchange equilibrium.
                    # The in-cloud share is HAM's ``get_icscavfrac`` sum,
                    # nucleation plus impaction capped at one.
                    def form_rate(f_liq, f_in_ice, imp):
                        if imp is not None:
                            f_liq = jnp.minimum(f_liq + imp[0], 1.0)
                            f_in_ice = jnp.minimum(f_in_ice + imp[1], 1.0)
                        return params.incloud_scale * fraction_to_rate(
                            cf_proc * ((1.0 - pice) * f_liq * f_wat
                                       + pice * f_in_ice * f_ice), dt)
                    form_num = form_rate(
                        jam_act.number_frac[i], ice_number[i],
                        imp_num if impaction else None)
                    form_mass = form_rate(
                        jam_act.mass_frac[i], ice_mass[i],
                        imp_mass if impaction else None)
                else:
                    # Standalone composition without ARG upstream: the
                    # aggregate activated fraction, with the impaction rate
                    # added rather than HAM's capped sum (the batched
                    # exponential still bounds the removal by the tracer).
                    form_num = form_mass = activated_fraction * rate_ic_unit
                    if impaction:
                        form_num = form_num + impaction_rate(imp_num)
                        form_mass = form_mass + impaction_rate(imp_mass)
                conv_num = below_conv_num + rate_conv_incloud
                conv_mass = below_conv_mass + rate_conv_incloud
            else:
                # The explicit phase holds the in-droplet and in-crystal
                # aerosol and removes it itself (the cloud-borne entries
                # below), and a non-activating mode has none; what is left
                # in the cloud is interstitial, which impaction collects.
                if impaction:
                    form_num = impaction_rate(imp_num)
                    form_mass = impaction_rate(imp_mass)
                else:
                    form_num = form_mass = zeros
                if mode.can_activate:
                    conv_num = below_conv_num + rate_conv_incloud
                    conv_mass = below_conv_mass + rate_conv_incloud
                else:
                    conv_num = below_conv_num
                    conv_mass = below_conv_mass
            n_nm = number_name(mode.short)
            names.append(n_nm)
            reinject_to.append(n_nm)
            q_list.append(jnp.maximum(view.get(n_nm, zeros), 0.0))
            rate_below_strat.append(below_strat_num)
            rate_form_strat.append(form_num)
            rate_conv.append(conv_num)
            for sp in mode.species:
                nm = mass_name(sp, mode.short)
                names.append(nm)
                reinject_to.append(nm)
                q_list.append(jnp.maximum(view.get(nm, zeros), 0.0))
                rate_below_strat.append(below_strat_mass)
                rate_form_strat.append(form_mass)
                rate_conv.append(conv_mass)
            if explicit_cb:
                # Cloud-borne aerosol is entirely in-droplet: no below-cloud
                # impaction, no activated-fraction weighting, and its
                # re-injected share returns to the interstitial partner.
                pairs = [(number_name(mode.short, cloud_borne=True),
                          number_name(mode.short))] + [
                    (mass_name(sp, mode.short, cloud_borne=True),
                     mass_name(sp, mode.short))
                    for sp in mode.species
                ]
                for nm, partner in pairs:
                    names.append(nm)
                    reinject_to.append(partner)
                    q_list.append(jnp.maximum(view.get(nm, zeros), 0.0))
                    rate_below_strat.append(zeros)
                    rate_form_strat.append(
                        phase_rate(liquid_share.get(nm, 1.0 - pice)))
                    rate_conv.append(zeros)

        # Implicit (exponential) scavenging over the step: q(t+dt) = q·exp(-rate·dt).
        # The first-order-decay rate is unbounded — the in-cloud rate ∝ 1/qc
        # diverges in near-clear cells — so an explicit ``dq = -rate·q`` step
        # removes far more than the available mass when ``rate·dt ≫ 1``,
        # overshooting into a sign-flipped runaway that NaNs the model
        # in a few steps. The analytic exponential of the decay is unconditionally
        # stable and positivity-preserving for any ``rate ≥ 0`` (HAMMOZ
        # ``mo_ham_wetdep`` applies the same ``1 - exp(-Λ·Δt)`` removed fraction).
        # Emitted as a per-second tendency so the operator-split sum + dynamics
        # apply exactly ``q·(exp(-rate·dt) - 1)`` over the step.
        # Clamp the decay rates to ≥0 so the exponential update is always a
        # bounded removal (a scavenging rate is non-negative by construction).
        below_arr = jnp.maximum(jnp.stack(rate_below_strat), 0.0)
        form_arr = jnp.maximum(jnp.stack(rate_form_strat), 0.0)
        conv_arr = jnp.maximum(jnp.stack(rate_conv), 0.0)
        rate_arr = below_arr + form_arr + conv_arr
        removed_frac = -jnp.expm1(-rate_arr * dt)              # 1 - exp(-rate·dt) ∈ [0, 1]
        dq_stack = -(removed_frac * jnp.stack(q_list)) / dt

        # Re-evaporation re-injection (#499): the stratiform shares of each
        # tracer's removal (proportional attribution among the rate
        # families) ride the falling precip; ``reinjection_budget``
        # releases them where the carrier evaporates, with impaction
        # joining the incoming carrier and in-cloud scavenging the newly
        # formed one (see its docstring).
        rate_safe = jnp.maximum(rate_arr, _RATE_FLOOR)
        live = rate_arr > _RATE_FLOOR
        share_below = jnp.where(live, below_arr / rate_safe, 0.0)
        share_form = jnp.where(live, form_arr / rate_safe, 0.0)
        removed_flux = -dq_stack * dm[jnp.newaxis]
        reinjected, _surface = reinjection_budget(
            removed_flux * share_below, removed_flux * share_form,
            evap_fraction,
        )
        # The convective share rides the convective precipitation and is
        # released by its own evaporation fraction, as ``ham_wetdep``
        # releases the below-cloud deposit (``zdxtevapbc``) of the
        # convective call; the in-plume removal of the transport term is
        # released there, so each kilogram is released once.
        share_conv = jnp.where(live, conv_arr / rate_safe, 0.0)
        reinjected_conv, _conv_surface = release_scavenged(
            removed_flux * share_conv, conv_evap_fraction,
        )
        reinject_tend = (reinjected + reinjected_conv) / dm[jnp.newaxis]

        all_tends = {nm: dq_stack[k] for k, nm in enumerate(names)}
        for k, target in enumerate(reinject_to):
            # Targets are always interstitial (cloud-borne re-injects to
            # its partner), so this never touches a cloud-borne entry.
            all_tends[target] = all_tends[target] + reinject_tend[k]

        # Cloud-borne removals go to the active store; in carry mode they
        # integrate sequentially, in tracers mode they rejoin the tendency
        # dict unchanged.
        if carry_mode(self._spec):
            cb_updates = {
                nm: all_tends.pop(nm)
                for nm in mirror_names(self._spec) if nm in all_tends
            }
            diagnostics, passthrough = apply_updates(
                self._spec, diagnostics, cb_updates, dt,
            )
            tracer_tends = {**all_tends, **passthrough}
            flux_tends = {**tracer_tends, **cb_updates}
        else:
            tracer_tends = all_tends
            flux_tends = all_tends

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        # AeroCom deposition fluxes (jax-gcm#581): this term's NET removal
        # (scavenging minus re-injection), column-integrated, accumulated
        # onto the per-step-reset keys.
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            DEPOSITED_SPECIES, _species_of, accumulate_deposition_fluxes)
        diagnostics = accumulate_deposition_fluxes(
            diagnostics, flux_tends,
            diagnostics["air_density"], diagnostics["layer_thickness"],
            kind="wet")
        # In-plume convective scavenging (jax-gcm#621) removes mass in the
        # transport term, whose tendencies never pass through this
        # accumulator — fold its published surface fluxes in here so the
        # ``wet_*`` ledger stays the complete wet sink. The keys all exist
        # after the accumulate call above.
        conv_scav = diagnostics.get("_conv_scav_flux")
        if self._in_plume_convective and conv_scav:
            diagnostics = dict(diagnostics)
            for nm, flx in conv_scav.items():
                species = _species_of(nm)
                if species in DEPOSITED_SPECIES:
                    key = f"wet_{species}"
                    diagnostics[key] = diagnostics[key] + flx
        return tendency, diagnostics
