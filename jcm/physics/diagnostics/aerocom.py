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

# Densities used for the geometric-optics optical depth.
_RHO_WATER = 1000.0  # kg/m^3
_RHO_ICE = 917.0     # kg/m^3

# Dry diameters for the aerosol number-concentration diagnostics [m].
_N_THRESHOLDS = {"N70": 70e-9, "N100": 100e-9}
# Aerodynamic diameters for particulate-matter diagnostics [m].
_PM_THRESHOLDS = {"PM1": 1e-6, "PM10": 10e-6}


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

    Everything is emitted as grid-box means, per the protocol Q/A —
    in-cloud values are formed in the analysis, after time-averaging.
    """

    name: ClassVar[str] = "aerocom_diagnostics"
    category: ClassVar[str] = "diagnostics"
    requires: ClassVar[tuple[str, ...]] = ("clouds", "pressure_full", "pressure_half")
    provides: ClassVar[tuple[str, ...]] = (
        "aerocom_clt", "aerocom_ttop", "aerocom_cdr", "aerocom_icr",
        "aerocom_cdnc", "aerocom_lcc", "aerocom_icc", "aerocom_cod",
        "aerocom_codliq", "aerocom_codice", "aerocom_lwp", "aerocom_iwp",
    )

    ALL_GROUPS: ClassVar[tuple[str, ...]] = ("cloud", "column", "plev", "aerosol")

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
        self.mode_sigma_g = (tuple(mode_sigma_g) if mode_sigma_g is not None
                             else (1.6, 1.8, 1.8, 1.6))

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
        thermo = diagnostics.get("thermo_run")
        temperature = thermo["temperature"] if thermo else state.temperature

        if "cloud" in self.groups:
            out.update(self._cloud_group(clouds, temperature, p_half))
        if "column" in self.groups:
            out.update(self._column_group(state, diagnostics, p_half))
        if "plev" in self.groups:
            out.update(self._plev_group(state, temperature, p_full))
        if "aerosol" in self.groups:
            out.update(self._aerosol_group(state, diagnostics, p_half))

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
        )
        return tendency, {**diagnostics, **out}

    def _cloud_group(self, clouds, temperature, p_half) -> dict:
        """Cloud-top sampling, optical depths and condensate paths."""
        # jcm carries effective radii in microns; the protocol wants metres.
        r_liq_m = clouds.r_eff_liq * 1e-6
        r_ice_m = clouds.r_eff_ice * 1e-6
        tau_liq, tau_ice = _cloud_optical_depth(
            clouds.qc, clouds.qi, r_liq_m, r_ice_m, p_half)
        cod3d = tau_liq + tau_ice

        # Liquid fraction of the condensate; a condensate-free layer is
        # assigned phase 0 but is masked out by the visibility test anyway.
        total = clouds.qc + clouds.qi
        has_cond = total > 0.0
        phase3d = jnp.where(has_cond, clouds.qc / jnp.where(has_cond, total, 1.0), 0.0)

        # In-cloud droplet number for the cloud-top weighting: the protocol
        # asks for in-cloud cdnc3d as input to the sampler (the grid-mean
        # output then follows from the area weighting inside it).
        cf = clouds.cloud_fraction
        in_cloud = cf > THRES_CLD
        cdnc3d = jnp.where(
            in_cloud, clouds.droplet_number / jnp.where(in_cloud, cf, 1.0), 0.0)

        top = cloud_top_sample(
            cod3d=cod3d, f3d=cf, t3d=temperature, phase3d=phase3d,
            cdr3d=r_liq_m, icr3d=r_ice_m, cdnc3d=cdnc3d, overlap=self.overlap)

        lwp = _column_integral(clouds.qc, p_half)
        iwp = _column_integral(clouds.qi, p_half)
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

    def _column_group(self, state, diagnostics, p_half) -> dict:
        """Water-vapour path, column number concentrations, albedo."""
        out = {
            "aerocom_prw": _column_integral(state.specific_humidity, p_half),
        }
        clouds = diagnostics["clouds"]
        dm = _layer_mass(p_half)
        # Droplet/ice number are per kg of air, so the same mass weighting
        # gives the column number per unit area.
        out["aerocom_cdnum"] = jnp.sum(clouds.droplet_number * dm, axis=0)
        qni = getattr(clouds, "ice_number", None)
        if qni is not None:
            out["aerocom_icnum"] = jnp.sum(qni * dm, axis=0)

        rad = diagnostics.get("radiation")
        if rad is not None:
            toa_down = getattr(rad, "toa_sw_down", None)
            toa_up = getattr(rad, "toa_sw_up", None)
            if toa_down is not None and toa_up is not None:
                lit = toa_down > 0.0
                out["aerocom_albedo"] = jnp.where(
                    lit, toa_up / jnp.where(lit, toa_down, 1.0), 0.0)
        return out

    def _plev_group(self, state, temperature, p_full) -> dict:
        """Winds on pressure surfaces and lower-tropospheric stability."""
        out = {}
        for target in self.plev_pa:
            tag = f"{int(round(target / 100.0)):d}"
            out[f"aerocom_u{tag}"] = _interp_to_pressure(state.u_wind, p_full, target)
            out[f"aerocom_v{tag}"] = _interp_to_pressure(state.v_wind, p_full, target)

        # LTS = theta(700 hPa) - theta(surface), the standard Klein & Hartmann
        # inversion strength. Potential temperature uses the model's own
        # kappa so it stays consistent with the dynamics.
        p_sfc = p_full[-1]
        t700 = _interp_to_pressure(temperature, p_full, 70000.0)
        theta700 = t700 * (100000.0 / 70000.0) ** c.akap
        theta_sfc = temperature[-1] * (100000.0 / p_sfc) ** c.akap
        out["aerocom_lts"] = theta700 - theta_sfc
        return out

    def _aerosol_group(self, state, diagnostics, p_half) -> dict:
        """Aerosol column burdens and number-concentration metrics."""
        out = {}
        dm = _layer_mass(p_half)
        # Per-tracer burdens: every aerosol mass tracer the state carries.
        tracers = getattr(state, "tracers", None) or {}
        for tname, field in tracers.items():
            if tname.startswith(("m_", "g_")) and jnp.ndim(field) >= 1:
                out[f"aerocom_burden_{tname}"] = jnp.sum(field * dm, axis=0)

        jam = diagnostics.get("_jam_state")
        if jam is None:
            return out
        # Modal number metrics. jam_state is (n_aer, nlev, ncols); the
        # per-mode widths are static configuration, so a Python loop over
        # modes is fine and keeps the lognormal integral readable.
        n_modes = jam.number.shape[0]
        sigmas = (self.mode_sigma_g * n_modes)[:n_modes]
        rho_air = diagnostics.get("air_density")
        for label, d_thresh in _N_THRESHOLDS.items():
            total = None
            for m in range(n_modes):
                nm = _lognormal_number_above(
                    jam.number[m], jam.r_dry[m], jnp.asarray(sigmas[m]), d_thresh)
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
                valid = (d_g_mass > 0.0) & (jam.mass[m] > 0.0)
                d_safe = jnp.where(valid, d_g_mass, 1.0)
                z = jnp.log(d_thresh / d_safe) / (jnp.sqrt(2.0) * jnp.log(sg))
                frac_below = 0.5 * (1.0 + jax.scipy.special.erf(z))
                mm = jnp.where(valid, jam.mass[m] * frac_below, 0.0)
                total = mm if total is None else total + mm
            if rho_air is not None:
                total = total * rho_air
            out[f"aerocom_{label}"] = total
        return out
