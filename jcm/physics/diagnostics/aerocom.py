"""AeroCom Phase 4 diagnostics.

Derived, submission-ready diagnostics for the AeroCom phase-4 experiments
(AP4-CTRL, aci-baseline, MMPPE, imo2020). Everything here is a *pure
function of the post-physics state* — nothing feeds back into the model —
so it lives in a single diagnostic-only :class:`PhysicsTerm` placed at the
end of the term list rather than being scattered through the schemes.

Why a term and not a data struct
--------------------------------
The AeroCom request is a flat, terminal name -> array mapping: no other
term consumes it, the requested set changes with each protocol revision,
and a run only wants the subset its experiment asks for. A typed
``@tree_math.struct`` (the pattern used for ``CloudData`` and friends)
exists to carry *scheme state between terms* and would force all-or-nothing
evaluation of ~100 heterogeneous fields. So this term emits a plain dict,
which ``output_averages`` already accumulates, and the ``groups`` argument
selects what is computed at all.

What is NOT here (and cannot be)
--------------------------------
Diagnostics that must be produced *inside* another scheme, because the
information does not survive to the end of the step:

* aerosol-free (``*noa`` / ``*_na``) fluxes — need a second radiation call
  with aerosol optics zeroed;
* per-species / per-mode / per-wavelength optical depth and absorption —
  need the aerosol optics internals;
* microphysical process rates (``autoconv``, ``accretn``, ``wbf``) —
  internal to the cloud microphysics;
* emission fluxes (``emi_*``) — internal to the emission terms.

Those are opt-in flags on their owning terms (see jax-gcm#581).

Vertical convention
-------------------
The physics-internal frame is **TOA-first**: level index 0 is the model
top and index -1 the surface, with the vertical on axis 0 and any trailing
axes horizontal (the repo's broadcasting-native column convention). This
matches the AeroCom cloud-top pseudo-code, whose ``k=1`` is the uppermost
level, so the scan below runs in protocol order without a flip. Note the
*saved output* is surface-first — the flip happens in the writer, not
here.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

# Cloud-top sampling thresholds, from the AeroCom phase-2 Indirect3 Fortran
# reproduced in the phase-4 aci-baseline protocol. A layer contributes only
# if it is both optically visible and not vanishingly thin.
THRES_CLD = 0.001   # minimum cloud fraction [1]
THRES_COD = 0.3     # minimum layer cloud optical depth [1]

# Overlap hypotheses for the cloud-top scan. Should match the overlap the
# radiation scheme uses (the protocol asks for consistency).
OVERLAP_MAXIMUM = "maximum"
OVERLAP_RANDOM = "random"
OVERLAP_MAXIMUM_RANDOM = "maximum-random"

# Effective-radius floor when forming optical depth, so a cloud-free layer
# (r_eff = 0) cannot divide by zero. Only reached where condensate is also
# zero, so the guarded value never affects a physical result.
_MIN_REFF_M = 1e-9

# Physical floor for condensate guards. A guard of ``> 0.0`` is not enough
# in reverse mode: the division VJP forms ``-g x / y^2`` and ``y^2``
# underflows to exactly zero for ``y`` below sqrt(TINY) (~1e-154 in f64,
# ~1e-19 in f32), turning a finite forward into a NaN gradient. Spectral
# ringing puts condensate tails deep inside that window, so guard on a
# physical floor, never on zero (CLAUDE.md). 1e-19 covers both dtypes and
# is still ~11 orders below any physically meaningful condensate.
_MIN_CONDENSATE = 1e-19

# Densities used for the geometric-optics optical depth.
_RHO_WATER = 1000.0  # kg/m^3
_RHO_ICE = 917.0     # kg/m^3

# Dry diameters for the aerosol number-concentration diagnostics [m].
_N_THRESHOLDS = {"N70": 70e-9, "N100": 100e-9}
# Aerodynamic diameters for particulate-matter diagnostics [m].
_PM_THRESHOLDS = {"PM1": 1e-6, "PM10": 10e-6}

# Species emitted as column burdens. AeroCom wants ONE total per species,
# so each entry sums every matching tracer: interstitial ``m_<spec>_<mode>``,
# cloud-borne ``mc_<spec>_<mode>`` and, for gases, ``g_<spec>``. Keeping the
# list static matters — the diagnostics dict is part of the scan carry, so
# key names derived from the live tracer dict (which is empty in the initial
# carry probe) change the pytree and the scan rejects it.
_BURDEN_SPECIES = ("so4", "bc", "oc", "poa", "soa", "ss", "du", "moa",
                   "dms", "so2", "h2so4", "soag")


def _layer_mass(pressure_half: jnp.ndarray) -> jnp.ndarray:
    """Dry-air mass per unit area of each layer, ``dp/g`` [kg/m^2].

    ``pressure_half`` holds the ``nlev+1`` interfaces TOA-first, so the
    forward difference is positive (pressure increases downward).
    """
    return jnp.diff(pressure_half, axis=0) / c.grav


def _column_integral(field: jnp.ndarray, pressure_half: jnp.ndarray) -> jnp.ndarray:
    """Mass-weighted column integral of a mixing ratio [kg/kg -> kg/m^2]."""
    return jnp.sum(field * _layer_mass(pressure_half), axis=0)


def _interp_to_pressure(
    field: jnp.ndarray, pressure_full: jnp.ndarray, target_pa: float
) -> jnp.ndarray:
    """Interpolate a model-level field to one pressure surface.

    Linear in ``log(p)`` — the standard choice for winds and temperature,
    and exact for a field varying with geopotential in a hydrostatic
    isothermal layer. Columns whose surface pressure lies above the target
    (i.e. the level is below ground) return the nearest in-column value
    rather than an extrapolation; AeroCom asks for the field on the
    pressure surface without specifying below-ground treatment, and
    clamping is the convention CMOR tools use for 700/850 hPa over
    orography.

    ``field`` and ``pressure_full`` are ``(nlev, *horiz)``, TOA-first.
    """
    log_p = jnp.log(pressure_full)
    log_t = jnp.log(jnp.asarray(target_pa, dtype=pressure_full.dtype))

    # Locate the layer bracketing the target: the last level (from the top)
    # whose pressure is still below the target. Clipped so the gather stays
    # in range for columns where the target is outside the profile.
    nlev = pressure_full.shape[0]
    below = (log_p < log_t).astype(jnp.int32)
    k_upper = jnp.clip(jnp.sum(below, axis=0) - 1, 0, nlev - 2)

    take = lambda arr, k: jnp.take_along_axis(arr, k[None, ...], axis=0)[0]
    p_up, p_dn = take(log_p, k_upper), take(log_p, k_upper + 1)
    f_up, f_dn = take(field, k_upper), take(field, k_upper + 1)

    # Safe denominator: identical pressures would only arise from a
    # degenerate profile, and the clamp below discards that branch anyway.
    dp = p_dn - p_up
    w = jnp.where(jnp.abs(dp) > 0.0, (log_t - p_up) / jnp.where(jnp.abs(dp) > 0.0, dp, 1.0), 0.0)
    w = jnp.clip(w, 0.0, 1.0)  # clamp instead of extrapolating below ground
    return f_up + w * (f_dn - f_up)


def _cloud_optical_depth(
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    r_eff_liq_m: jnp.ndarray,
    r_eff_ice_m: jnp.ndarray,
    pressure_half: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-layer liquid and ice cloud optical depth (geometric-optics).

    ``tau = 3 W / (2 rho_x r_eff)`` with ``W`` the layer condensate path —
    the large-size-parameter limit with extinction efficiency 2, which is
    what the AeroCom ``cod``/``codliq``/``codice`` request means and what
    the phase-2 cloud-top code assumed. These are *grid-mean* paths, so the
    result is the grid-mean optical depth; the protocol's Q/A is explicit
    that 2-D cloud fields are stored as grid-box means and NOT divided by
    cloud cover.
    """
    dm = _layer_mass(pressure_half)
    lwp_layer, iwp_layer = qc * dm, qi * dm
    r_liq = jnp.maximum(r_eff_liq_m, _MIN_REFF_M)
    r_ice = jnp.maximum(r_eff_ice_m, _MIN_REFF_M)
    tau_liq = 1.5 * lwp_layer / (_RHO_WATER * r_liq)
    tau_ice = 1.5 * iwp_layer / (_RHO_ICE * r_ice)
    return tau_liq, tau_ice


def cloud_top_sample(
    cod3d: jnp.ndarray,
    f3d: jnp.ndarray,
    t3d: jnp.ndarray,
    phase3d: jnp.ndarray,
    cdr3d: jnp.ndarray,
    icr3d: jnp.ndarray,
    cdnc3d: jnp.ndarray,
    overlap: str = OVERLAP_MAXIMUM_RANDOM,
) -> dict[str, jnp.ndarray]:
    """Cloud-top quantities as seen from above, per the AeroCom protocol.

    Direct translation of the phase-2 Indirect3 Fortran reproduced in the
    phase-4 aci-baseline protocol: sweep down from the top accumulating,
    for each visible layer, the *additional* cloud fraction newly seen from
    above, and weight the layer's cloud properties by it. The result is the
    cloud-top value a passive satellite retrieval would see, weighted by
    visible area.

    All inputs are ``(nlev, *horiz)`` TOA-first, matching the protocol's
    ``k=1`` uppermost convention. ``phase3d`` is the liquid fraction
    (1 = all liquid, 0 = all ice). Returns grid-mean (not in-cloud) sums,
    per the protocol Q/A: do **not** divide by ``clt`` here — that is done
    after time-averaging in the analysis.

    Implemented as a ``lax.scan`` over levels: the recurrence is sequential
    (each level's weight depends on the running cover from all levels
    above) and the scan keeps it differentiable and compile-friendly.
    """
    if overlap not in (OVERLAP_MAXIMUM, OVERLAP_RANDOM, OVERLAP_MAXIMUM_RANDOM):
        raise ValueError(
            f"overlap must be one of {OVERLAP_MAXIMUM!r}, {OVERLAP_RANDOM!r}, "
            f"{OVERLAP_MAXIMUM_RANDOM!r}; got {overlap!r}")

    horiz = f3d.shape[1:]
    dtype = f3d.dtype
    # For random / maximum-random the running variable is the *clear-sky*
    # product and starts at 1; for maximum it is the cloud cover itself and
    # starts at 0. flag_max restores the sign of the increment accordingly.
    if overlap == OVERLAP_MAXIMUM:
        clt0 = jnp.zeros(horiz, dtype)
        flag_max = jnp.asarray(-1.0, dtype)
    else:
        clt0 = jnp.ones(horiz, dtype)
        flag_max = jnp.asarray(1.0, dtype)

    zeros = jnp.zeros(horiz, dtype)
    init = {
        "clt": clt0, "ttop": zeros, "icr": zeros, "icc": zeros,
        "cdr": zeros, "cdnc": zeros, "lcc": zeros,
    }

    def step(carry, level):
        cod, f, t, phase, cdr, icr, cdnc, f_above = level
        clt = carry["clt"]

        if overlap == OVERLAP_MAXIMUM:
            ftmp = jnp.maximum(clt, f)
        elif overlap == OVERLAP_RANDOM:
            ftmp = clt * (1.0 - f)
        else:
            # Maximum-random: contiguous cloud blocks overlap maximally,
            # separated blocks randomly. The denominator is bounded away
            # from zero by the 1 - THRES_CLD cap, exactly as in the
            # reference Fortran.
            denom = 1.0 - jnp.minimum(f_above, 1.0 - THRES_CLD)
            ftmp = clt * (1.0 - jnp.maximum(f, f_above)) / denom

        visible = (cod > THRES_COD) & (f > THRES_CLD)
        # Newly-visible cloud fraction contributed by this layer.
        w = jnp.where(visible, (clt - ftmp) * flag_max, 0.0)

        new = {
            "clt": jnp.where(visible, ftmp, clt),
            "ttop": carry["ttop"] + t * w,
            "icr": carry["icr"] + icr * (1.0 - phase) * w,
            "icc": carry["icc"] + (1.0 - phase) * w,
            "cdr": carry["cdr"] + cdr * phase * w,
            "cdnc": carry["cdnc"] + cdnc * phase * w,
            "lcc": carry["lcc"] + phase * w,
        }
        return new, None

    # f_above is the cloud fraction of the level immediately above, needed
    # by maximum-random. The uppermost layer is assumed cloud-free by the
    # protocol ("assumption: uppermost layer is cloud-free (k=1)"), so it
    # is excluded from the scan and its f_above is irrelevant.
    f_above = f3d[:-1]
    levels = (cod3d[1:], f3d[1:], t3d[1:], phase3d[1:],
              cdr3d[1:], icr3d[1:], cdnc3d[1:], f_above)
    out, _ = jax.lax.scan(step, init, levels)

    clt = out["clt"]
    if overlap != OVERLAP_MAXIMUM:
        clt = 1.0 - clt
    return {
        "clt": clt, "ttop": out["ttop"], "icr": out["icr"], "icc": out["icc"],
        "cdr": out["cdr"], "cdnc": out["cdnc"], "lcc": out["lcc"],
    }


def _lognormal_number_above(
    number: jnp.ndarray, r_dry: jnp.ndarray, sigma_g: jnp.ndarray,
    diameter_threshold: float,
) -> jnp.ndarray:
    """Return the number concentration above a dry-diameter threshold.

    For a lognormal mode the fraction above diameter ``D`` is
    ``0.5 erfc(ln(D/D_g) / (sqrt(2) ln(sigma_g)))``. Modes with zero
    number or radius contribute nothing; the guarded logarithm keeps the
    reverse-mode gradient finite there (a bare ``log(0)`` would emit
    ``-inf`` into the cotangent even though the forward value is masked).
    """
    d_g = 2.0 * r_dry
    valid = (d_g > 0.0) & (number > 0.0)
    d_safe = jnp.where(valid, d_g, 1.0)
    ln_sigma = jnp.log(sigma_g)
    z = jnp.log(diameter_threshold / d_safe) / (jnp.sqrt(2.0) * ln_sigma)
    frac = 0.5 * jax.scipy.special.erfc(z)
    return jnp.where(valid, number * frac, 0.0)


def _default_jam_spec():
    """Return the MAM4 spec, imported lazily so jcm works without JAM."""
    from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
    return MAM4_SPEC


def _post_physics(state, diagnostics, field):
    """Return a prognostic FIELD as it will be saved, not as at step start.

    The scalar-field counterpart of :func:`_post_physics_tracer` — same
    operator-splitting argument, applied to u/v/T/q. ``_tendency_run`` is
    preferred over ``thermo_run`` here: ``thermo_run`` is advanced only by
    terms that opt in (convection, the cloud schemes), whereas the running
    accumulator captures every term's returned tendency automatically —
    including vertical diffusion, radiation and gravity-wave drag, which
    never touch ``thermo_run`` but do move the saved winds and temperature.
    """
    x = getattr(state, field)
    run = diagnostics.get("_tendency_run")
    dt = diagnostics.get("_dt_seconds")
    if run is None or dt is None or run.get(field) is None:
        return x
    out = x + run[field] * dt
    # ``verify_tendencies`` clamps the applied humidity so the SAVED field is
    # non-negative; the raw sum here is not clamped, so without this a large
    # same-step sink could give a negative prw that disagrees with the saved
    # hus. Water vapour is the only non-negative scalar among these fields —
    # winds are signed and temperature is far from zero — so the floor is
    # applied only to it.
    if field == "specific_humidity":
        out = jnp.maximum(out, 0.0)
    return out


def _post_physics_tracer(state, diagnostics, name, default=None):
    """Return a tracer as it will be SAVED, not as at step start.

    Operator splitting means ``state.tracers`` is the step-start value for
    every term: schemes return their effect as a tendency and the dycore
    applies the sum once, afterwards. A diagnostic that reads ``state.tracers``
    therefore reports a field one step behind the tracers written at the same
    timestamp — for the 2M number tracers that is every step the microphysics
    does anything, and for aerosol mass it drops the whole step's emissions,
    chemistry, sedimentation, deposition and scavenging.

    ``_tendency_run`` carries the sum over the terms that have already run
    (see ``ComposablePhysics``); these diagnostics run last, so it is the
    full physics tendency. Falls back to the raw tracer when the view is
    absent, which keeps this structural rather than data-dependent.
    """
    q = state.tracers.get(name, default)
    if q is None:
        return None
    run = diagnostics.get("_tendency_run")
    dt = diagnostics.get("_dt_seconds")
    if run is None or dt is None:
        return q
    dq = run.get("tracers", {}).get(name)
    if dq is None:
        return q
    # Mixing ratios and number concentrations are both non-negative.
    return jnp.maximum(q + dq * dt, 0.0)


class AerocomDiagnostics(PhysicsTerm):
    """AeroCom phase-4 derived diagnostics (diagnostic-only, no tendency).

    Groups (select with ``groups=``) so a run pays only for what its
    experiment requests:

    ``cloud``
        Cloud-top sampled quantities (``ttop``, ``cdr``, ``icr``,
        ``cdnc``, ``lcc``, ``icc``, ``clt``) per the protocol algorithm,
        the layer/column optical depths (``cod``, ``codliq``, ``codice``)
        and the condensate paths (``lwp``, ``iwp``, ``clivi``, ``cllvi``).
    ``column``
        Water vapour path ``prw``, column droplet/ice number
        (``cdnum``, ``icnum``) and ``albedo``.
    ``plev``
        Fields on pressure surfaces (``u200``/``v200``/``u700``/``v700``)
        and lower-tropospheric stability ``lts``.
    ``aerosol``
        Column burdens per aerosol tracer plus ``N70``/``N100`` and
        ``PM1``/``PM10`` from the modal state. Requires the JAM aerosol
        module; silently inactive without it.
    ``nearsurface``
        2 m temperature/dew point, 10 m winds (neutral log-profile
        interpolation, see :meth:`_nearsurface_group`), sea-level
        pressure, the convective/stratiform x rain/snow precipitation
        split (``prcr``/``prcs``/``prsn``) and the activation cloud-base
        updraft ``wbase``.

    Everything is emitted as grid-box means, per the protocol Q/A —
    in-cloud values are formed in the analysis, after time-averaging.

    Measured cost at T63L47 on an A100, against the same run with the term
    absent: ``cloud`` +3.3 %, ``column`` +3.3 %, ``plev`` +3.0 %,
    ``aerosol`` +3.2 %, the first three together +4.4 %, all four +9.4 %.
    Most of a single group's ~3 % is the term's fixed per-step overhead, so
    groups cost much less together than their sum; ``aerosol`` is the
    largest true increment (~5 % on top of the other three) because of the
    per-mode lognormal integrals. No group triggers extra physics or an
    extra radiation call — the diagnostics that would (aerosol-free
    radiation, ~2x) are deliberately excluded; see jax-gcm#583.
    """

    name: ClassVar[str] = "aerocom_diagnostics"
    category: ClassVar[str] = "diagnostics"
    requires: ClassVar[tuple[str, ...]] = ("clouds", "pressure_full", "pressure_half")
    # Every key this term can publish. The emitted set must be static (the
    # diagnostics dict is part of the scan carry), so each selected group
    # writes all of its keys, zero-filled where the active configuration
    # cannot supply a value.
    provides: ClassVar[tuple[str, ...]] = (
        # cloud
        "aerocom_clt", "aerocom_ttop", "aerocom_cdr", "aerocom_icr",
        "aerocom_cdnc", "aerocom_lcc", "aerocom_icc", "aerocom_cod",
        "aerocom_codliq", "aerocom_codice", "aerocom_lwp", "aerocom_iwp",
        "aerocom_cllvi", "aerocom_clivi",
        # column
        "aerocom_prw", "aerocom_cdnum", "aerocom_icnum", "aerocom_albedo",
        "aerocom_cdnc3d",
        # plev
        "aerocom_lts", "aerocom_ptp",
        "aerocom_wap", "aerocom_w500", "aerocom_w700",
        # nearsurface
        "aerocom_tas", "aerocom_uas", "aerocom_vas", "aerocom_dew2",
        "aerocom_psl", "aerocom_prsn", "aerocom_prcr", "aerocom_prcs",
        "aerocom_wbase",
        # aerosol (per-tracer burdens are named from the active tracer set)
        "aerocom_N70", "aerocom_N100", "aerocom_PM1", "aerocom_PM10",
        *(f"aerocom_burden_{sp}" for sp in _BURDEN_SPECIES),
    )

    ALL_GROUPS: ClassVar[tuple[str, ...]] = (
        "cloud", "column", "plev", "aerosol", "nearsurface")

    def __init__(
        self,
        groups: tuple[str, ...] | list[str] = ("cloud", "column"),
        overlap: str = OVERLAP_MAXIMUM_RANDOM,
        plev_pa: tuple[float, ...] = (20000.0, 70000.0),
        mode_sigma_g: tuple[float, ...] | None = None,
    ):
        """Configure which diagnostic groups to compute.

        ``overlap`` should match the radiation scheme's overlap
        assumption, as the protocol requests. ``plev_pa`` lists the
        pressure surfaces for the ``plev`` group (default 200 and 700
        hPa, the levels AeroCom asks for). ``mode_sigma_g`` gives the
        geometric standard deviation per aerosol mode for the number
        diagnostics; ``None`` uses the MAM4 defaults.
        """
        unknown = set(groups) - set(self.ALL_GROUPS)
        if unknown:
            raise ValueError(
                f"unknown aerocom diagnostic group(s) {sorted(unknown)}; "
                f"choose from {list(self.ALL_GROUPS)}")
        if overlap not in (OVERLAP_MAXIMUM, OVERLAP_RANDOM, OVERLAP_MAXIMUM_RANDOM):
            raise ValueError(f"unknown overlap {overlap!r}")
        self.groups = tuple(groups)
        self.overlap = str(overlap)
        self.plev_pa = tuple(float(p) for p in plev_pa)
        # MAM4 modal widths (Aitken, accumulation, coarse, primary-carbon).
        self.mode_sigma_g = (tuple(float(s) for s in mode_sigma_g)
                             if mode_sigma_g is not None
                             else (1.6, 1.8, 1.8, 1.6))
        # sigma_g = 1 is a monodisperse delta: ln(sigma) = 0 divides both
        # lognormal integrals by zero, and sigma < 1 is not a width at all.
        # The per-mode COUNT is checked against the live modal state in
        # _aerosol_group, where it is known.
        bad = [s for s in self.mode_sigma_g if s <= 1.0]
        if bad:
            raise ValueError(
                f"mode_sigma_g entries must be > 1 (geometric std dev); got {bad}")

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute the selected diagnostic groups. Emits no tendency."""
        clouds = diagnostics["clouds"]
        p_half = diagnostics["pressure_half"]
        p_full = diagnostics["pressure_full"]
        out: dict[str, jnp.ndarray] = {}

        # Use the running thermodynamic state where an upstream term has
        # published it, so the diagnostics describe the same atmosphere the
        # microphysics saw (matching the CloudSat simulator's convention).
        # The running thermodynamic view carries the POST-microphysics
        # condensate (both cloud schemes advance it), so these diagnostics
        # describe the same atmosphere as the tracers saved at this
        # timestamp. ``CloudData.qc``/``qi`` are the step-START values —
        # the schemes return their condensate change in the tendency only —
        # so using them here would date every cloud product one step.
        thermo = diagnostics.get("thermo_run") or {}
        # Post-physics temperature: thermo_run is advanced only by terms that
        # opt in, so radiation and gravity-wave drag are missing from it even
        # though they move the saved ta. The condensate below still comes
        # from thermo_run, which is the view that carries qc/qi — and which
        # now includes convective detrainment.
        temperature = _post_physics(state, diagnostics, "temperature")
        qc = thermo.get("qc", clouds.qc)
        qi = thermo.get("qi", clouds.qi)

        cdnc_gm, cdnc_ic, qnc, qni = self._number_concentrations(
            state, diagnostics, clouds)
        # Publish the resolved 3-D droplet number: CloudData.droplet_number
        # is zero under the 2-moment scheme (which carries qnc instead), so
        # the CMOR writer needs this rather than the raw CloudData field.
        # GRID-MEAN under both schemes, so the CMOR'd cdnc3d means one thing.
        out["aerocom_cdnc3d"] = cdnc_gm
        if "cloud" in self.groups:
            out.update(self._cloud_group(clouds, temperature, p_half,
                                         cdnc_ic, qc, qi))
        if "column" in self.groups:
            out.update(self._column_group(state, diagnostics, p_half, qnc, qni))
        if "plev" in self.groups:
            out.update(self._plev_group(state, diagnostics, temperature, p_full))
        if "aerosol" in self.groups:
            out.update(self._aerosol_group(state, diagnostics, p_half))
        if "nearsurface" in self.groups:
            out.update(self._nearsurface_group(
                state, diagnostics, terrain, temperature, p_full))

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
        )
        return tendency, {**diagnostics, **out}

    @staticmethod
    def _number_concentrations(state, diagnostics, clouds):
        """Return ``(cdnc_gm_m3, cdnc_ic_m3, qnc_per_kg, qni_per_kg)``.

        The two microphysics schemes publish droplet number differently, in
        different units AND with different in-cloud semantics, so the
        diagnostics must not assume one of them:

        * 2-moment (``lohmann_2m``) carries prognostic ``qnc``/``qni``
          tracers in **kg^-1** as **grid means** (they are advected
          tracers) and leaves ``CloudData.droplet_number`` at its zero
          carry init;
        * 1-moment (``echam_1m``) writes ``CloudData.droplet_number`` in
          **m^-3** as a characteristic **in-cloud** value
          (``base_cdnc * cdnc_factor``), nonzero even in clear sky.

        Both a grid-mean and an in-cloud volumetric field are returned so
        each consumer takes the semantics it needs: the CMOR'd ``cdnc3d``
        and the ``cdnum`` column integral want grid means, the cloud-top
        sampler wants the in-cloud value. Deriving both HERE, per scheme,
        is what stops the 1M in-cloud field being divided by cloud
        fraction a second time (inflating cloud-top CDNC by 1/cf) or
        integrated over clear sky (inflating cdnum). ``qnc`` is returned
        as the grid-mean per-mass form, the exact measure for dp/g.
        """
        rho = diagnostics.get("air_density")
        cf = clouds.cloud_fraction
        cloudy = cf > THRES_CLD
        cf_safe = jnp.where(cloudy, cf, 1.0)
        # Post-physics, not step-start: the 2M scheme returns number changes
        # as tendencies, so the raw tracers lag a step (see
        # _post_physics_tracer).
        qnc = _post_physics_tracer(state, diagnostics, "qnc")
        qni = _post_physics_tracer(state, diagnostics, "qni")
        if qnc is not None and rho is not None:
            cdnc_gm = qnc * rho          # kg^-1 -> m^-3, grid mean
            cdnc_ic = jnp.where(cloudy, cdnc_gm / cf_safe, 0.0)
        else:
            ic_char = clouds.droplet_number  # 1M path: in-cloud, m^-3
            cdnc_ic = jnp.where(cloudy, ic_char, 0.0)
            cdnc_gm = ic_char * cf
            if qnc is None and rho is not None:
                qnc = jnp.where(
                    rho > 0.0, cdnc_gm / jnp.where(rho > 0.0, rho, 1.0), 0.0)
        return cdnc_gm, cdnc_ic, qnc, qni

    def _cloud_group(self, clouds, temperature, p_half, cdnc_ic,
                     qc, qi) -> dict:
        """Cloud-top sampling, optical depths and condensate paths.

        ``cdnc_ic`` is the IN-CLOUD droplet number [m^-3], already resolved
        per scheme by :meth:`_number_concentrations` — the protocol asks for
        in-cloud cdnc3d as input to the sampler (the grid-mean output then
        follows from the area weighting inside it). Do not divide by cloud
        fraction here: for the 1M scheme the field is in-cloud already.
        """
        # jcm carries effective radii in microns; the protocol wants metres.
        r_liq_m = clouds.r_eff_liq * 1e-6
        r_ice_m = clouds.r_eff_ice * 1e-6
        tau_liq, tau_ice = _cloud_optical_depth(
            qc, qi, r_liq_m, r_ice_m, p_half)
        cod3d = tau_liq + tau_ice

        # Liquid fraction of the condensate; a condensate-free layer is
        # assigned phase 0 but is masked out by the visibility test anyway.
        # The guard floor excludes the squared-underflow window, not just
        # zero (see _MIN_CONDENSATE): ringing tails feed that window, where
        # the division VJP would emit NaN into any gradient taken through
        # these diagnostics — and they are natural calibration observables.
        total = qc + qi
        has_cond = total > _MIN_CONDENSATE
        phase3d = jnp.where(has_cond, qc / jnp.where(has_cond, total, 1.0), 0.0)

        cf = clouds.cloud_fraction
        top = cloud_top_sample(
            cod3d=cod3d, f3d=cf, t3d=temperature, phase3d=phase3d,
            cdr3d=r_liq_m, icr3d=r_ice_m, cdnc3d=cdnc_ic, overlap=self.overlap)

        lwp = _column_integral(qc, p_half)
        iwp = _column_integral(qi, p_half)
        return {
            "aerocom_clt": top["clt"],
            "aerocom_ttop": top["ttop"],
            "aerocom_cdr": top["cdr"],
            "aerocom_icr": top["icr"],
            "aerocom_cdnc": top["cdnc"],
            "aerocom_lcc": top["lcc"],
            "aerocom_icc": top["icc"],
            "aerocom_cod": jnp.sum(cod3d, axis=0),
            "aerocom_codliq": jnp.sum(tau_liq, axis=0),
            "aerocom_codice": jnp.sum(tau_ice, axis=0),
            "aerocom_lwp": lwp,
            "aerocom_iwp": iwp,
            # CMIP/AeroCom aliases for the same paths.
            "aerocom_cllvi": lwp,
            "aerocom_clivi": iwp,
        }

    def _column_group(self, state, diagnostics, p_half, qnc, qni) -> dict:
        """Water-vapour path, column number concentrations, albedo."""
        out = {
            # Post-physics humidity, so prw matches the hus saved at this
            # timestamp rather than preceding the step's diffusion,
            # convection and microphysics.
            "aerocom_prw": _column_integral(
                _post_physics(state, diagnostics, "specific_humidity"), p_half),
        }
        # Column number [m^-2]. The prognostic number tracers are per unit
        # MASS (kg^-1), so dp/g is the exact measure for them; a volumetric
        # (m^-3) number would instead need dz. Using dp/g on an m^-3 field
        # would silently fold in an extra density factor.
        #
        # Both keys are emitted unconditionally, with zeros where the active
        # microphysics carries no such tracer: the diagnostics dict is part
        # of the scan carry, so a key set that varies with configuration (or
        # between steps) changes the carry pytree and the scan rejects it.
        dm = _layer_mass(p_half)
        zero = jnp.zeros(p_half.shape[1:], dtype=p_half.dtype)
        out["aerocom_cdnum"] = jnp.sum(qnc * dm, axis=0) if qnc is not None else zero
        out["aerocom_icnum"] = jnp.sum(qni * dm, axis=0) if qni is not None else zero

        # SURFACE albedo, from the surface fluxes. The TOA ratio is the
        # PLANETARY albedo, which folds in cloud and aerosol reflection and
        # would therefore move with the very ACI signal these experiments
        # are trying to isolate.
        rad = diagnostics.get("radiation")
        sfc_down = getattr(rad, "surface_sw_down", None) if rad is not None else None
        sfc_up = getattr(rad, "surface_sw_up", None) if rad is not None else None
        if sfc_down is not None and sfc_up is not None:
            lit = sfc_down > 0.0
            albedo = jnp.where(lit, sfc_up / jnp.where(lit, sfc_down, 1.0), 0.0)
        else:
            albedo = zero
        out["aerocom_albedo"] = albedo
        return out

    def _plev_group(self, state, diagnostics, temperature, p_full) -> dict:
        """Winds on pressure surfaces and lower-tropospheric stability."""
        out = {}
        # Post-physics winds: gravity-wave drag and vertical diffusion move
        # u/v within the step and never touch thermo_run, so the step-start
        # fields disagree with the winds saved at this timestamp.
        u_post = _post_physics(state, diagnostics, "u_wind")
        v_post = _post_physics(state, diagnostics, "v_wind")
        # LTS likewise: the caller passes the thermo_run temperature, which
        # radiation and gravity-wave drag never advance even though they move
        # the saved ta. Use the accumulator, which captures every term.
        temperature = _post_physics(state, diagnostics, "temperature")
        for target in self.plev_pa:
            tag = f"{int(round(target / 100.0)):d}"
            out[f"aerocom_u{tag}"] = _interp_to_pressure(u_post, p_full, target)
            out[f"aerocom_v{tag}"] = _interp_to_pressure(v_post, p_full, target)

        # LTS = theta(700 hPa) - theta(surface), the standard Klein & Hartmann
        # inversion strength. Potential temperature uses the model's own
        # kappa so it stays consistent with the dynamics.
        p_sfc = p_full[-1]
        t700 = _interp_to_pressure(temperature, p_full, 70000.0)
        theta700 = t700 * (100000.0 / 70000.0) ** c.akap
        theta_sfc = temperature[-1] * (100000.0 / p_sfc) ** c.akap
        out["aerocom_lts"] = theta700 - theta_sfc

        # Pressure vertical velocity (the wap/w500/w700 request,
        # jax-gcm#409): supplied by the dycore's omega provider
        # (DinosaurDycore(compute_omega=True), config key
        # dycore.compute_omega) as a dycore field, since only the
        # dynamics knows the mass fluxes consistent with its own
        # continuity equation. Zero-filled when the provider is off,
        # which is static per config, so the emitted key set stays
        # scan-carry-stable. wap is the full model-level field
        # (leading level axis, the cdnc3d layout); w500/w700 are the
        # requested pressure-surface slices.
        omega = (diagnostics.get("_dycore_fields") or {}).get("omega")
        if omega is not None:
            out["aerocom_wap"] = omega
            out["aerocom_w500"] = _interp_to_pressure(omega, p_full, 50000.0)
            out["aerocom_w700"] = _interp_to_pressure(omega, p_full, 70000.0)
        else:
            out["aerocom_wap"] = jnp.zeros(p_full.shape, dtype=p_full.dtype)
            out["aerocom_w500"] = jnp.zeros(p_full.shape[1:],
                                            dtype=p_full.dtype)
            out["aerocom_w700"] = jnp.zeros(p_full.shape[1:],
                                            dtype=p_full.dtype)

        # WMO tropopause pressure (the ptp request): the wmo_tropopause
        # module's finder, on the post-physics temperature. Its default
        # search indices (13..35) encode the L47 grid; on other grids they
        # can miss the tropopause entirely (L95: indices 13-34 sit at
        # 31-501 Pa — Codex review on PR #604). Derive the window from the
        # column-mean pressures instead: the WMO search belongs between
        # ~40 hPa and ~550 hPa on any grid. jnp.searchsorted on a traced
        # mean profile would give traced (unusable) slice bounds, so the
        # bounds come from a Python-level reduction over the STATIC level
        # structure: mean pressure per level is computed with lax-free
        # numpy on the hybrid coefficients when available, else falls back
        # to the finder's defaults on 47-level grids only.
        z_full = diagnostics.get("height_full")
        nlev = p_full.shape[0]
        if z_full is not None:
            from jcm.physics.diagnostics.wmo_tropopause import (
                find_tropopause_level)
            ref_p = self._nominal_level_pressures(nlev)
            if ref_p is not None:
                ncctop = int(np.searchsorted(ref_p, 4000.0))
                nccbot = int(np.searchsorted(ref_p, 55000.0))
                nccbot = max(nccbot, ncctop + 2)
                out["aerocom_ptp"] = find_tropopause_level(
                    temperature.T, p_full.T, z_full.T,
                    ncctop=ncctop, nccbot=nccbot)
            elif nlev == 47:
                out["aerocom_ptp"] = find_tropopause_level(
                    temperature.T, p_full.T, z_full.T)
            else:
                # No nominal pressures and not the grid the defaults were
                # tuned for: a constant-fallback ptp would be misleading.
                out["aerocom_ptp"] = jnp.zeros(p_full.shape[1:],
                                               dtype=p_full.dtype)
        else:
            out["aerocom_ptp"] = jnp.zeros(p_full.shape[1:],
                                           dtype=p_full.dtype)
        return out

    def _nominal_level_pressures(self, nlev):
        """Return static nominal mid-level pressures [Pa], or None.

        Cached from ``cache_coords`` when the model provides a vertical
        coordinate with sigma centers; used to derive grid-independent
        tropopause search bounds as PYTHON ints (slice bounds must be
        static under jit).
        """
        ref = getattr(self, "_ref_level_pressures", None)
        if ref is not None and len(ref) == nlev:
            return ref
        return None

    def cache_coords(self, coords) -> None:
        """Record nominal level pressures for the tropopause window."""
        try:
            vertical = coords.vertical
            # Both forms are dimensionless sigma at level centres; nominal
            # pressure follows by scaling with the reference surface
            # pressure (hybrid coordinates fold their (a, b) into the
            # sigma centres at this reference).
            if hasattr(vertical, "centers"):
                sigma = np.asarray(vertical.centers)
            else:
                sigma = np.asarray(vertical.get_sigma_centers(101325.0))
            self._ref_level_pressures = sigma * 101325.0
        except Exception:  # pragma: no cover - defensive; coords vary
            self._ref_level_pressures = None

    def _nearsurface_group(self, state, diagnostics, terrain,
                           temperature, p_full) -> dict:
        """2 m / 10 m diagnostics, sea-level pressure, precipitation split.

        The 2 m temperature and 10 m winds interpolate between the surface
        and the lowest model level with the NEUTRAL logarithmic profile,
        using the tile-averaged momentum roughness the surface term
        publishes (heat roughness = 0.1 z0m, the model's own ratio).
        Stability corrections are deliberately omitted in this first
        version: they modify the 2 m values by O(1 K) in strongly
        stable/unstable layers, which matters for NWP verification but not
        for the AeroCom context fields — documented so nobody mistakes
        this for a Monin-Obukhov implementation. ``dew2`` converts the
        (well-mixed) lowest-level specific humidity to a dew point at
        surface pressure via the inverted Magnus formula. ``psl`` is the
        standard WMO reduction with the 6.5 K/km lapse. ``wbase`` is the
        SAME updraft the 2M activation uses (fact_tke sqrt(2 TKE),
        lohmann_2m fact_tke = 0.7), sampled at the diagnosed cloud base.
        Convective precipitation is split rain/snow by the lowest-level
        temperature (the melt criterion the COSP hook already uses).
        """
        clouds = diagnostics["clouds"]
        z_full = diagnostics.get("height_full")
        z_half = diagnostics.get("height_half")
        ncols_shape = state.temperature.shape[1:]
        out: dict[str, jnp.ndarray] = {}

        t_low = temperature[-1]
        q_low = _post_physics(state, diagnostics, "specific_humidity")[-1]
        u_low = _post_physics(state, diagnostics, "u_wind")[-1]
        v_low = _post_physics(state, diagnostics, "v_wind")[-1]
        # The ACTUAL surface pressure, not the lowest level-centre pressure:
        # at L47 the two differ by ~400 Pa even over ocean, which would bias
        # psl and the dew point directly (Codex review on PR #604).
        p_sfc = state.normalized_surface_pressure.reshape(ncols_shape) * c.p0

        sfc = diagnostics.get("surface")
        t_skin = getattr(sfc, "surface_temperature", None)
        z0m = getattr(sfc, "roughness_length", None)
        if t_skin is None or z0m is None or z_full is None or z_half is None:
            zero = jnp.zeros(ncols_shape, dtype=temperature.dtype)
            for k in ("tas", "uas", "vas", "dew2", "wbase"):
                out[f"aerocom_{k}"] = zero
        else:
            z_agl = jnp.maximum(z_full[-1] - z_half[-1], 10.0)
            z0m = jnp.clip(z0m.reshape(ncols_shape), 1e-5, 2.0)
            z0h = 0.1 * z0m
            # Neutral log-profile ratios; winds vanish at z0m, scalars
            # reach the skin value at z0h.
            r10 = jnp.log(10.0 / z0m) / jnp.log(z_agl / z0m)
            r2 = jnp.log(2.0 / z0h) / jnp.log(z_agl / z0h)
            t_skin = t_skin.reshape(ncols_shape)
            out["aerocom_tas"] = t_skin + (t_low - t_skin) * jnp.clip(r2, 0.0, 1.0)
            out["aerocom_uas"] = u_low * jnp.clip(r10, 0.0, 1.0)
            out["aerocom_vas"] = v_low * jnp.clip(r10, 0.0, 1.0)
            # Magnus inversion: e = q p / (eps + (1-eps) q); Td from
            # ln(e/611.2) = 17.62 Td / (Td + 243.12) (Td in Celsius).
            eps_rd = 0.622
            e_pa = jnp.maximum(
                q_low * p_sfc / (eps_rd + (1.0 - eps_rd) * q_low), 1e-3)
            ln_ratio = jnp.log(e_pa / 611.2)
            td_c = 243.12 * ln_ratio / (17.62 - ln_ratio)
            out["aerocom_dew2"] = jnp.minimum(
                td_c + 273.15, out["aerocom_tas"])

            # Cloud-base updraft from the vdiff TKE (see docstring).
            vdiff = diagnostics.get("vertical_diffusion")
            tke = getattr(vdiff, "tke", None)
            if tke is None:
                out["aerocom_wbase"] = jnp.zeros(ncols_shape,
                                                 dtype=temperature.dtype)
            else:
                nlev = temperature.shape[0]
                tke = tke.reshape(-1, nlev).T if tke.shape[0] != nlev                     else tke.reshape(nlev, -1)
                w_act = 0.7 * jnp.sqrt(jnp.maximum(2.0 * tke, 0.0))
                cf = clouds.cloud_fraction
                cloudy = cf > THRES_CLD
                # Lowest cloudy level, TOA-first: flip, argmax, unflip.
                rev = cloudy[::-1]
                k_base = nlev - 1 - jnp.argmax(rev, axis=0)
                take = jnp.take_along_axis(w_act, k_base[None, ...], axis=0)[0]
                has_cloud = jnp.any(cloudy, axis=0)
                out["aerocom_wbase"] = jnp.where(has_cloud, take, 0.0)

        # Sea-level pressure, WMO reduction (T_star at the surface from the
        # lowest level with the standard 6.5 K/km lapse; mean of T_star and
        # T at msl in the exponent denominator).
        orog = jnp.reshape(terrain.orog, ncols_shape) if terrain is not None             else jnp.zeros(ncols_shape)
        t_star = t_low + 0.0065 * (
            (z_full[-1] - z_half[-1]) if z_full is not None else 0.0)
        t_msl = t_star + 0.0065 * orog
        rd = 287.04
        out["aerocom_psl"] = p_sfc * jnp.exp(
            c.grav * orog / (rd * 0.5 * (t_star + t_msl)))

        # Precipitation split: large-scale rain/snow come straight from the
        # cloud scheme; convective is split by the lowest-level temperature.
        conv = diagnostics.get("convection")
        prc = getattr(conv, "precip_conv", None)
        prc = (prc.reshape(ncols_shape) if prc is not None
               else jnp.zeros(ncols_shape))
        frozen = t_low < c.tmelt
        prcs = jnp.where(frozen, prc, 0.0)
        out["aerocom_prcr"] = prc - prcs
        out["aerocom_prcs"] = prcs
        out["aerocom_prsn"] = clouds.precip_snow.reshape(ncols_shape) + prcs
        return out

    def _aerosol_group(self, state, diagnostics, p_half) -> dict:
        """Aerosol column burdens and number-concentration metrics."""
        out = {}
        dm = _layer_mass(p_half)
        # Per-tracer burdens: every aerosol mass tracer the state carries.
        tracers = getattr(state, "tracers", None) or {}
        # Carry-stored cloud-borne aerosol (#602 item 3): when the JAM
        # population keeps its mc_*/nc_* phase in the physics carry rather
        # than the dycore tracers, the burden must still count it. Carry
        # fields are already post-physics (updated sequentially within the
        # step), so they join AFTER the _post_physics_tracer adjustment
        # below rather than through it.
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        cb_carry = diagnostics.get(CARRY_KEY) or {}
        # MAM4-JAX splits aerosol mass between interstitial (``m_*``) and
        # cloud-borne (``mc_*``) populations; the AeroCom species burden is
        # the TOTAL, so both are summed here (omitting the cloud-borne half
        # undercounts wherever aerosol has been activated). Gas-phase
        # species use the ``g_<spec>`` tracer.
        zero_col = jnp.zeros(p_half.shape[1:], dtype=p_half.dtype)
        for spec in _BURDEN_SPECIES:
            total = None
            for tname, raw in tracers.items():
                if jnp.ndim(raw) < 1:
                    continue
                if (tname.startswith((f"m_{spec}_", f"mc_{spec}_"))
                        or tname == f"g_{spec}"):
                    # Post-physics mass, so the burden matches the aerosol
                    # tracers saved at this timestamp rather than missing the
                    # step's emissions, chemistry, sedimentation, deposition
                    # and wet scavenging.
                    field = _post_physics_tracer(state, diagnostics, tname)
                    contrib = jnp.sum(field * dm, axis=0)
                    total = contrib if total is None else total + contrib
            for tname, raw in cb_carry.items():
                if tname.startswith(f"mc_{spec}_"):
                    contrib = jnp.sum(raw * dm, axis=0)
                    total = contrib if total is None else total + contrib
            out[f"aerocom_burden_{spec}"] = zero_col if total is None else total

        # The number/PM keys are emitted unconditionally (zero-filled when the
        # modal state is unavailable) for the same reason as the column group:
        # the diagnostics dict is part of the scan carry, so a key set that
        # appears only on some steps changes the carry pytree and the scan
        # rejects it. ``_jam_state`` in particular is absent from the initial
        # carry probe but present once the JAM chain has run.
        jam = diagnostics.get("_jam_state")
        if jam is None:
            # 3-D like the real fields (they are per-level, mapped as
            # ModelLevel by the writer): a surface-shaped zero here would
            # change the carry pytree between the probe and the run, and
            # write 2-D data into a ModelLevel variable when the group is
            # enabled without JAM.
            nlev = p_half.shape[0] - 1
            zero = jnp.zeros((nlev,) + p_half.shape[1:], dtype=p_half.dtype)
            for label in (*_N_THRESHOLDS, *_PM_THRESHOLDS):
                out[f"aerocom_{label}"] = zero
            return out
        # Modal number metrics. jam_state is (n_aer, nlev, ncols); the
        # per-mode widths are static configuration, so a Python loop over
        # modes is fine and keeps the lognormal integral readable.
        n_modes = jam.number.shape[0]
        # ``_jam_state`` is diagnosed by the JAM core, BEFORE sedimentation,
        # dry deposition and the post-cloud aqueous / wet-scavenging block
        # have removed material. Rebuild the modal amounts from the
        # post-physics tracers so N70/N100/PM describe the aerosol saved at
        # this timestamp rather than the mid-chain population.
        #
        # r_dry / kappa keep the core's diagnosis: recomputing them means
        # re-running the modal size diagnosis, and they are ratios of mass to
        # number, so a step that removes both changes them very little —
        # unlike the amounts themselves, which is what these diagnostics
        # report. Documented rather than silently approximated.
        from jcm.physics.aerosol.jam import mass_name, number_name
        spec = getattr(self, "_jam_spec", None) or _default_jam_spec()
        num_post, mass_post = [], []
        for mode in spec.modes:
            n = _post_physics_tracer(state, diagnostics, number_name(mode.short))
            num_post.append(n if n is not None else None)
            parts = [
                _post_physics_tracer(state, diagnostics, mass_name(sp, mode.short))
                for sp in mode.species
            ]
            parts = [x for x in parts if x is not None]
            mass_post.append(sum(parts) if parts else None)
        number_pp = (jnp.stack(num_post) if all(x is not None for x in num_post)
                     else jam.number)
        mass_pp = (jnp.stack(mass_post) if all(x is not None for x in mass_post)
                   else jam.mass)
        # One width per live mode, positionally. The previous cycling idiom
        # ((sigma * n)[:n]) silently handed mode 5 mode 1's width if the
        # modal scheme ever grew; fail loudly instead.
        if len(self.mode_sigma_g) != n_modes:
            raise ValueError(
                f"mode_sigma_g has {len(self.mode_sigma_g)} entries but the "
                f"modal state carries {n_modes} modes; pass one width per mode")
        sigmas = self.mode_sigma_g
        rho_air = diagnostics.get("air_density")
        for label, d_thresh in _N_THRESHOLDS.items():
            total = None
            for m in range(n_modes):
                nm = _lognormal_number_above(
                    number_pp[m], jam.r_dry[m], jnp.asarray(sigmas[m]), d_thresh)
                total = nm if total is None else total + nm
            # kg^-1 -> m^-3 where air density is available.
            if rho_air is not None:
                total = total * rho_air
            out[f"aerocom_{label}"] = total
        for label, d_thresh in _PM_THRESHOLDS.items():
            total = None
            for m in range(n_modes):
                # Mass below the cut: complement of the number integral
                # applied to the mass distribution, whose median diameter
                # is shifted by exp(3 ln^2 sigma_g) (Hatch-Choate).
                sg = jnp.asarray(sigmas[m])
                d_g_mass = 2.0 * jam.r_dry[m] * jnp.exp(3.0 * jnp.log(sg) ** 2)
                # PM1/PM10 cutoffs are AERODYNAMIC diameters, and an impactor
                # sizes by settling velocity, not by geometry. Comparing the
                # geometric diameter against them over-includes dense
                # particles: d_ae = d_g * sqrt(rho_p / rho_0) with
                # rho_0 = 1000 kg/m3, so dust (~2650) and sea salt (~2200)
                # reach a given aerodynamic cut at ~0.6 of the geometric
                # diameter the cut would otherwise imply. Slip correction is
                # omitted — it matters below ~0.5 um, where both cuts pass
                # essentially all the mass anyway, so it cannot move PM1/PM10.
                rho_p = jnp.maximum(jam.rho[m], 1.0)
                d_g_mass = d_g_mass * jnp.sqrt(rho_p / 1000.0)
                valid = (d_g_mass > 0.0) & (mass_pp[m] > 0.0)
                d_safe = jnp.where(valid, d_g_mass, 1.0)
                z = jnp.log(d_thresh / d_safe) / (jnp.sqrt(2.0) * jnp.log(sg))
                frac_below = 0.5 * (1.0 + jax.scipy.special.erf(z))
                mm = jnp.where(valid, mass_pp[m] * frac_below, 0.0)
                total = mm if total is None else total + mm
            if rho_air is not None:
                total = total * rho_air
            out[f"aerocom_{label}"] = total
        return out
