"""Below-cloud impaction scavenging coefficients (CAM MAM4 / Slinn).

The coefficient Λ₁ a falling rain spectrum applies to a lognormal aerosol
mode, per unit precipitation: for a grid-mean precipitation flux ``R``
[kg m⁻² s⁻¹ ≡ mm s⁻¹] the first-order removal rate is ``Λ = Λ₁·R`` [1/s],
so Λ₁ carries units of 1/mm (CAM's "scav rate in 1/h at a precip rate of
1 mm/h").

``impaction_scavenging_rates`` is a transcription of CAM's
``calc_1_impact_rate`` (ESCOMP/CAM ``cam_development``,
``src/chemistry/modal_aero/aero_model.F90``): a double sum over a
Marshall-Palmer-like raindrop spectrum (50 bins, 0.005–0.25 cm radius) and
the mode's lognormal size distribution (51 bins about the volume-median
radius), weighted by Slinn's collection efficiency

    E = E_brown + E_intercept + E_impact,   E ≤ 1

with Brownian diffusion ``4(1 + 0.4·Re^½·Sc^⅓)/(Re·Sc)``, interception
``4χ(χ + (1 + 2·60·χ)/(1 + 60/Re^½))``, and inertial impaction
``((St - S*)/(St - S* + 2/3))^{3/2}`` above the critical Stokes number.
Number- and volume-weighted sums give the coefficients for the number and
mass mixing ratios respectively.

Because ``E ≤ 1``, Λ₁ is bounded by the rain's geometric sweep-out rate, and
the Brownian/impaction competition reproduces the Greenfield gap — the
minimum near 0.1–1 µm that an ``r²`` extrapolation of a submicron
coefficient cannot represent.

Evaluating the 50×51 double sum in every cell of every mode each step is
~10¹¹ flop at T63L47, so — exactly as CAM does in
``modal_aero_bcscavcoef_init``/``_get`` — the integral is tabulated once per
mode at construction time against the wet/dry diameter growth ratio (NumPy,
float64) and looked up at runtime by differentiable log-linear
interpolation. CAM tabulates at a single reference state (273.16 K,
750 hPa, dry material density) because the coefficient's residual
dependence on air temperature, pressure and particle density is weak; this
port keeps that choice so the two agree node for node.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np

import jcm.constants as c

# --- Raindrop spectrum (CAM ``calc_1_impact_rate``) ---------------------------
#: Radii 0.005–0.250 cm in 0.005 cm steps.
_RAIN_R_LO_CM = 0.005
_RAIN_R_HI_CM = 0.250
_RAIN_DR_CM = 0.005
#: Exponential (Marshall-Palmer-like) size distribution ``N(r) ∝ exp(-r/r₀)``.
_RAIN_R0_CM = 2.7e-2
#: Terminal fall speed at STP, ``v = A·d^B`` per drop-diameter band [cm]
#: (Beard-type empirical fit as tabulated by CAM). Bands are (d_max, A, B).
_VFALL_BANDS = (
    (0.007, 2.88e5, 2.0),
    (0.025, 2.8008e4, 1.528),
    (0.100, 4104.9, 1.008),
    (0.250, 1812.1, 0.638),
)
_VFALL_TAIL = (1069.8, 0.235)
#: Air density the fit is referenced to [g/cm³]; ``v ∝ √(ρ₀/ρ)``.
_VFALL_RHO_STP_CGS = 1.204e-3

# --- Sutherland's law for the dynamic viscosity of air ------------------------
#: ``μ = μ₀·(T_S₁/(T + T_S₂))·(T/T_S₃)^{3/2}`` [poise], CAM's coefficients.
_VISC_MU0_CGS = 1.8325e-4
_VISC_TS1_K = 416.16
_VISC_TS2_K = 120.0
_VISC_TS3_K = 296.16

# --- Fuchs slip correction ----------------------------------------------------
#: ``C = 1 + 1.246·Kn + 0.42·Kn·exp(-0.87/Kn)`` (Fuchs 1964; the
#: Davies coefficients CAM and Seinfeld & Pandis both use).
_FUCHS_A = 1.246
_FUCHS_B = 0.42
_FUCHS_C = 0.87
#: Molecular mean free path as ``λ = _MFP_COEF/c_air`` with ``c_air`` the air
#: molar density [mol/cm³]; the coefficient is CAM's, in cm·mol/cm³.
_MFP_COEF_CGS = 2.8052e-10

# --- Slinn (1983) collection efficiency ---------------------------------------
#: Brownian: ``E = 4(1 + a·Re^½·Sc^{b})/(Re·Sc)``.
_BROWN_A = 0.4
_BROWN_SC_EXP = 0.3333333
#: Critical Stokes number ``S* = (c₁ + ln(1+Re)/c₂)/(1 + ln(1+Re))``.
_SSTAR_C1 = 1.2
_SSTAR_C2 = 12.0
#: Inertial impaction ``E = ((St - S*)/(St - S* + c))^{p}`` above ``S*``.
_IMPACT_C = 0.6666667
_IMPACT_P = 1.5
#: Default ratio of water to air dynamic viscosity in Slinn's interception
#: term. Uncertain, and exposed as a tunable knob on ``WetDepParameters``.
MU_WATER_AIR_DEFAULT = 60.0
#: Default multiplier on the inertial-impaction efficiency — the least
#: constrained of the three collection mechanisms. 1.0 is CAM as written.
IMPACT_SCALE_DEFAULT = 1.0

#: CAM's tabulation reference state (``modal_aero_bcscavcoef_init``).
_REF_TEMPERATURE_K = 273.16
_REF_PRESSURE_PA = 0.75e5

# cgs values of the shared constants, converted at use: 1 J = 1e7 erg,
# 1 kg/mol = 1e3 g/mol. CAM's own ``mo_constants`` differ from these in the
# 6th-7th significant digit (see ``impaction_test``).
_BOLTZ_CGS = c.ak * 1.0e7                    # erg/K
_RGAS_CGS = c.r_universal * 1.0e7            # erg/K/mol
_M_AIR_CGS = c.m_air * 1.0e3                 # g/mol

#: Growth-ratio table: ``nimptblgrow_mind``/``_maxd`` and ``log(1.25)`` from
#: CAM ``aero_model.F90``. Covers wet/dry diameter ratios 0.21–14.6.
GROW_MIN = -7
GROW_MAX = 12
DLN_DG = math.log(1.25)


def _rain_spectrum(rho_air_cgs, xp=np):
    """Drop radii [cm], bin number concentrations [#/cm³] and fall speeds [cm/s].

    Normalised so the spectrum carries exactly 1 mm/h of precipitation.
    """
    n = 1 + int(round((_RAIN_R_HI_CM - _RAIN_R_LO_CM) / _RAIN_DR_CM))
    r = _RAIN_R_LO_CM + np.arange(n) * _RAIN_DR_CM
    xnum = np.exp(-r / _RAIN_R0_CM)

    d = 2.0 * r
    vfall_stp = _VFALL_TAIL[0] * d ** _VFALL_TAIL[1]
    for d_max, coef, expo in reversed(_VFALL_BANDS):
        vfall_stp = np.where(d <= d_max, coef * d ** expo, vfall_stp)
    # v ∝ √(ρ₀/ρ): the drop spectrum is the only air-density dependence, and
    # rho_air_cgs may be traced, so keep this in ``xp``.
    vfall = xp.asarray(vfall_stp) * xp.sqrt(_VFALL_RHO_STP_CGS / rho_air_cgs)

    precip = 1.0 / 36000.0                      # 1 mm/h in cm/s
    precip_sum = xp.sum(vfall * r ** 3 * xnum) * math.pi * 1.333333
    return xp.asarray(r), xp.asarray(xnum) * (precip / precip_sum), vfall


def _aerosol_spectrum(dg_wet_cm: float, geom_std_dev: float):
    """Radii [cm] and normalised number/volume weights of the lognormal mode.

    Static in the knobs, so plain NumPy: the bin count depends only on the
    mode's geometric standard deviation.
    """
    sx = math.log(geom_std_dev)
    xg0 = math.log(0.5 * dg_wet_cm)
    xg3 = xg0 + 3.0 * sx * sx
    dx = max(0.2 * sx, 0.01)
    half = max(4.0 * sx, 2.0 * dx)
    xlo, xhi = xg3 - half, xg3 + half
    na = 1 + int(round((xhi - xlo) / dx))

    x = xlo + np.arange(na) * dx
    a = np.exp(x)
    ynum = np.exp(-0.5 * ((x - xg0) / sx) ** 2)
    yvol = ynum * 1.3333 * math.pi * a ** 3
    return a, ynum / ynum.sum(), yvol / yvol.sum()


def impaction_scavenging_rates(
    dg_wet: float,
    geom_std_dev: float,
    particle_density: float,
    temperature: float = _REF_TEMPERATURE_K,
    pressure: float = _REF_PRESSURE_PA,
    *,
    mu_water_air=MU_WATER_AIR_DEFAULT,
    impact_scale=IMPACT_SCALE_DEFAULT,
    boltz_cgs: float = _BOLTZ_CGS,
    rgas_cgs: float = _RGAS_CGS,
    m_air_cgs: float = _M_AIR_CGS,
    xp=np,
) -> tuple[float, float]:
    """Compute the number- and volume-weighted impaction coefficients [1/mm].

    Transcribes CAM ``aero_model.F90::calc_1_impact_rate``. The size grids and
    the drop spectrum's shape are static; ``mu_water_air`` and ``impact_scale``
    may be traced, so pass ``xp=jnp`` to build the table differentiably.

    Args:
        dg_wet: Wet number-median DIAMETER of the mode [m].
        geom_std_dev: Geometric standard deviation σ_g of the mode.
        particle_density: Particle material density [kg/m³].
        temperature: Air temperature [K].
        pressure: Air pressure [Pa].
        mu_water_air: Water/air dynamic-viscosity ratio (Slinn interception).
        impact_scale: Multiplier on the inertial-impaction efficiency.
        boltz_cgs: Boltzmann constant [erg/K].
        rgas_cgs: Universal gas constant [erg/K/mol].
        m_air_cgs: Molar mass of dry air [g/mol].
        xp: ``numpy`` (reference, float64) or ``jax.numpy`` (differentiable).

    Returns:
        ``(number, volume)`` coefficients; multiply by a precipitation flux
        in kg m⁻² s⁻¹ to get a first-order removal rate in 1/s.

    """
    dg_cgs = dg_wet * 1.0e2                       # m -> cm
    rho_p_cgs = particle_density * 1.0e-3         # kg/m³ -> g/cm³
    press_cgs = pressure * 10.0                   # Pa -> dyne/cm²

    c_air = press_cgs / (rgas_cgs * temperature)   # mol/cm³
    rho_air = m_air_cgs * c_air                    # g/cm³
    freepath = _MFP_COEF_CGS / c_air               # cm
    dyn_visc = (_VISC_MU0_CGS * (_VISC_TS1_K / (temperature + _VISC_TS2_K))
                * (temperature / _VISC_TS3_K) ** 1.5)
    kin_visc = dyn_visc / rho_air

    r, xnum, vfall = _rain_spectrum(rho_air, xp=xp)
    a_np, fnum_np, fvol_np = _aerosol_spectrum(dg_cgs, geom_std_dev)
    a = xp.asarray(a_np)
    fnum, fvol = xp.asarray(fnum_np), xp.asarray(fvol_np)

    reynolds = r * vfall / kin_visc                     # (nr,)
    sqrt_re = xp.sqrt(reynolds)

    chi = a[None, :] / r[:, None]                       # (nr, na)

    dum = freepath / a
    fuchs = 1.0 + _FUCHS_A * dum + _FUCHS_B * dum * xp.exp(-_FUCHS_C / dum)
    tau = 2.0 * rho_p_cgs * a ** 2 * fuchs / (9.0 * rho_air * kin_visc)
    aero_mass = 4.0 * math.pi * a ** 3 * rho_p_cgs / 3.0
    diffus = boltz_cgs * temperature * tau / aero_mass
    schmidt = kin_visc / diffus                          # (na,)
    stokes = vfall[:, None] * tau[None, :] / r[:, None]  # (nr, na)

    e_brown = (4.0 * (1.0 + _BROWN_A * sqrt_re[:, None]
                      * schmidt[None, :] ** _BROWN_SC_EXP)
               / (reynolds[:, None] * schmidt[None, :]))
    dum2 = ((1.0 + 2.0 * mu_water_air * chi)
            / (1.0 + mu_water_air / sqrt_re[:, None]))
    e_intercept = 4.0 * chi * (chi + dum2)
    log_re = xp.log1p(reynolds)[:, None]
    s_star = (_SSTAR_C1 + log_re / _SSTAR_C2) / (1.0 + log_re)
    excess = xp.maximum(stokes - s_star, 0.0)
    e_impact = impact_scale * (excess / (excess + _IMPACT_C)) ** _IMPACT_P
    e_total = xp.minimum(e_brown + e_intercept + e_impact, 1.0)

    sweep = xnum * 4.0 * math.pi * r ** 2 * vfall        # (nr,)
    weighted = sweep[:, None] * e_total
    # x3600: cgs rate per (1 mm/h) -> 1/h per (mm/h) == 1/mm.
    num = xp.sum(weighted * fnum[None, :]) * 3600.0
    vol = xp.sum(weighted * fvol[None, :]) * 3600.0
    return (num, vol) if xp is not np else (float(num), float(vol))


@dataclasses.dataclass(frozen=True)
class ImpactionTable:
    """Static per-mode collection kernel on CAM's growth-ratio grid.

    Everything here is independent of the tunable knobs, so it is built once
    in float64 NumPy: the drop/aerosol geometry (``chi``), the Brownian and
    inertial efficiencies at unit scale, and the per-node weights. Only the
    knob-dependent arithmetic — the interception term and the impaction
    scale — is evaluated in JAX, by :func:`table_log_coefficients`.

    Arrays are ``(ngrow, nr, na)`` unless noted; ``ngrow`` is CAM's 20-node
    growth grid, ``nr`` the raindrop bins and ``na`` the aerosol bins.
    """

    dgnum: float            # dry reference number-median diameter [m]
    chi: np.ndarray         # aerosol/drop radius ratio
    sqrt_re: np.ndarray     # (nr,) square root of the drop Reynolds number
    e_brown: np.ndarray     # Brownian collection efficiency
    e_impact: np.ndarray    # inertial impaction efficiency at scale 1
    sweep: np.ndarray       # (nr,) rain sweep-out rate x bin concentration
    fnum: np.ndarray        # (ngrow, na) number weights
    fvol: np.ndarray        # (ngrow, na) volume weights


def _kernel_at(dg_wet, geom_std_dev, particle_density,
               temperature=_REF_TEMPERATURE_K, pressure=_REF_PRESSURE_PA):
    """Build the knob-independent pieces of ``calc_1_impact_rate``."""
    dg_cgs = dg_wet * 1.0e2
    rho_p_cgs = particle_density * 1.0e-3
    press_cgs = pressure * 10.0

    c_air = press_cgs / (_RGAS_CGS * temperature)
    rho_air = _M_AIR_CGS * c_air
    freepath = _MFP_COEF_CGS / c_air
    dyn_visc = (_VISC_MU0_CGS * (_VISC_TS1_K / (temperature + _VISC_TS2_K))
                * (temperature / _VISC_TS3_K) ** 1.5)
    kin_visc = dyn_visc / rho_air

    r, xnum, vfall = _rain_spectrum(rho_air, xp=np)
    a, fnum, fvol = _aerosol_spectrum(dg_cgs, geom_std_dev)

    reynolds = r * vfall / kin_visc
    sqrt_re = np.sqrt(reynolds)
    chi = a[None, :] / r[:, None]

    dum = freepath / a
    fuchs = 1.0 + _FUCHS_A * dum + _FUCHS_B * dum * np.exp(-_FUCHS_C / dum)
    tau = 2.0 * rho_p_cgs * a ** 2 * fuchs / (9.0 * rho_air * kin_visc)
    aero_mass = 4.0 * math.pi * a ** 3 * rho_p_cgs / 3.0
    diffus = _BOLTZ_CGS * temperature * tau / aero_mass
    schmidt = kin_visc / diffus
    stokes = vfall[:, None] * tau[None, :] / r[:, None]

    e_brown = (4.0 * (1.0 + _BROWN_A * sqrt_re[:, None]
                      * schmidt[None, :] ** _BROWN_SC_EXP)
               / (reynolds[:, None] * schmidt[None, :]))
    log_re = np.log1p(reynolds)[:, None]
    s_star = (_SSTAR_C1 + log_re / _SSTAR_C2) / (1.0 + log_re)
    excess = np.maximum(stokes - s_star, 0.0)
    e_impact = (excess / (excess + _IMPACT_C)) ** _IMPACT_P
    sweep = xnum * 4.0 * math.pi * r ** 2 * vfall
    return chi, sqrt_re, e_brown, e_impact, sweep, fnum, fvol


def build_impaction_table(
    dgnum: float, geom_std_dev: float, particle_density: float,
) -> ImpactionTable:
    """Tabulate a mode's static collection kernel over CAM's growth grid.

    Mirrors ``modal_aero_bcscavcoef_init``: 20 nodes spaced ``log(1.25)``
    apart in the wet/dry diameter ratio, at CAM's reference thermodynamic
    state with the mode's dry material density. Float64 NumPy, once per
    mode at construction.
    """
    pieces = [
        _kernel_at(dgnum * math.exp(j * DLN_DG), geom_std_dev, particle_density)
        for j in range(GROW_MIN, GROW_MAX + 1)
    ]
    stack = lambda k: np.stack([p[k] for p in pieces])
    return ImpactionTable(
        dgnum=float(dgnum),
        chi=stack(0), sqrt_re=pieces[0][1], e_brown=stack(2),
        e_impact=stack(3), sweep=pieces[0][4], fnum=stack(5), fvol=stack(6),
    )


def table_log_coefficients(
    table: ImpactionTable, mu_water_air, impact_scale,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Log coefficients at the growth nodes, differentiable in the knobs.

    Completes Slinn's efficiency from the static kernel:
    ``E = min(E_brown + 4χ(χ + (1 + 2μχ)/(1 + μ/√Re)) + s·E_impact, 1)``,
    then the number- and volume-weighted sweep-out sums. Cheap enough to
    rebuild inside the traced step — 20 nodes rather than one per cell.
    """
    chi = jnp.asarray(table.chi)
    sqrt_re = jnp.asarray(table.sqrt_re)[None, :, None]
    e_int = 4.0 * chi * (chi + (1.0 + 2.0 * mu_water_air * chi)
                         / (1.0 + mu_water_air / sqrt_re))
    e_total = jnp.minimum(
        jnp.asarray(table.e_brown) + e_int
        + impact_scale * jnp.asarray(table.e_impact), 1.0)
    weighted = jnp.asarray(table.sweep)[None, :, None] * e_total
    num = jnp.sum(weighted * jnp.asarray(table.fnum)[:, None, :], axis=(1, 2))
    vol = jnp.sum(weighted * jnp.asarray(table.fvol)[:, None, :], axis=(1, 2))
    # x3600: cgs rate per (1 mm/h) -> 1/h per (mm/h) == 1/mm.
    return jnp.log(num * 3600.0), jnp.log(vol * 3600.0)


def _interp_log_table(ln_table: jnp.ndarray, x_grow: jnp.ndarray) -> jnp.ndarray:
    """CAM's table lookup: clamp below the grid, extrapolate linearly above."""
    xg = jnp.maximum(x_grow, float(GROW_MIN))
    j = jnp.clip(jnp.floor(xg), GROW_MIN, GROW_MAX - 1)
    idx = (j - GROW_MIN).astype(jnp.int32)
    lo = jnp.take(ln_table, idx)
    hi = jnp.take(ln_table, idx + 1)
    return lo + (xg - j) * (hi - lo)


def bcscavcoef(
    r_wet: jnp.ndarray, dgnum: float,
    ln_number: jnp.ndarray, ln_volume: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Look up ``(number, volume)`` impaction coefficients [1/mm].

    ``r_wet`` is the wet number-median RADIUS [m]; CAM indexes the table by
    the wet/dry DIAMETER ratio, so the factor of two matters.
    """
    ratio = jnp.maximum(2.0 * r_wet, 1.0e-12) / dgnum
    x_grow = jnp.log(ratio) / DLN_DG
    return (jnp.exp(_interp_log_table(ln_number, x_grow)),
            jnp.exp(_interp_log_table(ln_volume, x_grow)))
