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

# Raindrop spectrum: radii 0.005–0.250 cm in 0.005 cm steps, number
# concentration ∝ exp(-r/0.027 cm), renormalised to 1 mm/h.
_RAIN_R_LO_CM = 0.005
_RAIN_R_HI_CM = 0.250
_RAIN_DR_CM = 0.005
#: Ratio of water to air dynamic viscosity used by Slinn's interception term.
_MU_WATER_AIR = 60.0
#: CAM's tabulation reference state (``modal_aero_bcscavcoef_init``).
_REF_TEMPERATURE_K = 273.16
_REF_PRESSURE_PA = 0.75e5
#: cgs constants matching CAM ``mo_constants`` (boltz_cgs, rgas_cgs).
_BOLTZ_CGS = 1.38065e-16
_RGAS_CGS = 8.3144724e7

#: Growth-ratio table: ``nimptblgrow_mind``/``_maxd`` and ``log(1.25)`` from
#: CAM ``aero_model.F90``. Covers wet/dry diameter ratios 0.21–14.6.
GROW_MIN = -7
GROW_MAX = 12
DLN_DG = math.log(1.25)


def _rain_spectrum(rho_air_cgs: np.ndarray):
    """Drop radii [cm], bin number concentrations [#/cm³] and fall speeds [cm/s].

    Normalised so the spectrum carries exactly 1 mm/h of precipitation.
    """
    n = 1 + int(round((_RAIN_R_HI_CM - _RAIN_R_LO_CM) / _RAIN_DR_CM))
    r = _RAIN_R_LO_CM + np.arange(n) * _RAIN_DR_CM
    xnum = np.exp(-r / 2.7e-2)

    d = 2.0 * r
    vfall_stp = np.where(
        d <= 0.007, 2.88e5 * d ** 2,
        np.where(
            d <= 0.025, 2.8008e4 * d ** 1.528,
            np.where(
                d <= 0.1, 4104.9 * d ** 1.008,
                np.where(d <= 0.25, 1812.1 * d ** 0.638, 1069.8 * d ** 0.235),
            ),
        ),
    )
    vfall = vfall_stp * math.sqrt(1.204e-3 / rho_air_cgs)

    precip = 1.0 / 36000.0                      # 1 mm/h in cm/s
    precip_sum = np.sum(vfall * r ** 3 * xnum) * math.pi * 1.333333
    return r, xnum * (precip / precip_sum), vfall


def _aerosol_spectrum(dg_wet_cm: float, geom_std_dev: float):
    """Radii [cm] and normalised number/volume weights of the lognormal mode."""
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
) -> tuple[float, float]:
    """Compute the number- and volume-weighted impaction coefficients [1/mm].

    Args:
        dg_wet: Wet number-median DIAMETER of the mode [m].
        geom_std_dev: Geometric standard deviation σ_g of the mode.
        particle_density: Particle material density [kg/m³].
        temperature: Air temperature [K].
        pressure: Air pressure [Pa].

    Returns:
        ``(number, volume)`` coefficients; multiply by a precipitation flux
        in kg m⁻² s⁻¹ to get a first-order removal rate in 1/s.

    """
    dg_cgs = dg_wet * 1.0e2                       # m -> cm
    rho_p_cgs = particle_density * 1.0e-3         # kg/m³ -> g/cm³
    press_cgs = pressure * 10.0                   # Pa -> dyne/cm²

    c_air = press_cgs / (_RGAS_CGS * temperature)  # mol/cm³
    rho_air = 28.966 * c_air                       # g/cm³
    freepath = 2.8052e-10 / c_air                  # cm
    dyn_visc = (1.8325e-4 * (416.16 / (temperature + 120.0))
                * (temperature / 296.16) ** 1.5)
    kin_visc = dyn_visc / rho_air

    r, xnum, vfall = _rain_spectrum(rho_air)
    a, fnum, fvol = _aerosol_spectrum(dg_cgs, geom_std_dev)

    reynolds = r * vfall / kin_visc                     # (nr,)
    sqrt_re = np.sqrt(reynolds)

    chi = a[None, :] / r[:, None]                       # (nr, na)

    dum = freepath / a
    fuchs = 1.0 + 1.246 * dum + 0.42 * dum * np.exp(-0.87 / dum)
    tau = 2.0 * rho_p_cgs * a ** 2 * fuchs / (9.0 * rho_air * kin_visc)
    aero_mass = 4.0 * math.pi * a ** 3 * rho_p_cgs / 3.0
    diffus = _BOLTZ_CGS * temperature * tau / aero_mass
    schmidt = kin_visc / diffus                          # (na,)
    stokes = vfall[:, None] * tau[None, :] / r[:, None]  # (nr, na)

    e_brown = (4.0 * (1.0 + 0.4 * sqrt_re[:, None] * schmidt[None, :] ** 0.3333333)
               / (reynolds[:, None] * schmidt[None, :]))
    dum2 = ((1.0 + 2.0 * _MU_WATER_AIR * chi)
            / (1.0 + _MU_WATER_AIR / sqrt_re[:, None]))
    e_intercept = 4.0 * chi * (chi + dum2)
    log_re = np.log1p(reynolds)[:, None]
    s_star = (1.2 + log_re / 12.0) / (1.0 + log_re)
    excess = np.maximum(stokes - s_star, 0.0)
    e_impact = (excess / (excess + 0.6666667)) ** 1.5
    e_total = np.minimum(e_brown + e_intercept + e_impact, 1.0)

    sweep = xnum * 4.0 * math.pi * r ** 2 * vfall        # (nr,)
    weighted = sweep[:, None] * e_total
    # x3600: cgs rate per (1 mm/h) -> 1/h per (mm/h) == 1/mm.
    return (float(np.sum(weighted * fnum[None, :]) * 3600.0),
            float(np.sum(weighted * fvol[None, :]) * 3600.0))


@dataclasses.dataclass(frozen=True)
class ImpactionTable:
    """Per-mode log-scavenging-coefficient table vs wet/dry diameter ratio."""

    dgnum: float           # dry reference number-median diameter [m]
    ln_number: jnp.ndarray  # (GROW_MAX - GROW_MIN + 1,)
    ln_volume: jnp.ndarray


def build_impaction_table(
    dgnum: float, geom_std_dev: float, particle_density: float,
) -> ImpactionTable:
    """Tabulate a mode's coefficients over CAM's growth-ratio grid.

    Mirrors ``modal_aero_bcscavcoef_init``: 20 nodes spaced ``log(1.25)``
    apart in the wet/dry diameter ratio, evaluated at CAM's reference
    thermodynamic state with the mode's dry material density.
    """
    ln_num, ln_vol = [], []
    for jgrow in range(GROW_MIN, GROW_MAX + 1):
        dg_wet = dgnum * math.exp(jgrow * DLN_DG)
        num, vol = impaction_scavenging_rates(
            dg_wet, geom_std_dev, particle_density)
        ln_num.append(math.log(num))
        ln_vol.append(math.log(vol))
    return ImpactionTable(
        dgnum=float(dgnum),
        ln_number=jnp.asarray(np.array(ln_num)),
        ln_volume=jnp.asarray(np.array(ln_vol)),
    )


def _interp_log_table(ln_table: jnp.ndarray, x_grow: jnp.ndarray) -> jnp.ndarray:
    """CAM's table lookup: clamp below the grid, extrapolate linearly above."""
    xg = jnp.maximum(x_grow, float(GROW_MIN))
    j = jnp.clip(jnp.floor(xg), GROW_MIN, GROW_MAX - 1)
    idx = (j - GROW_MIN).astype(jnp.int32)
    lo = jnp.take(ln_table, idx)
    hi = jnp.take(ln_table, idx + 1)
    return lo + (xg - j) * (hi - lo)


def bcscavcoef(
    r_wet: jnp.ndarray, table: ImpactionTable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Look up ``(number, volume)`` impaction coefficients [1/mm].

    ``r_wet`` is the wet number-median RADIUS [m]; CAM indexes the table by
    the wet/dry DIAMETER ratio, so the factor of two matters.
    """
    ratio = jnp.maximum(2.0 * r_wet, 1.0e-12) / table.dgnum
    x_grow = jnp.log(ratio) / DLN_DG
    return (jnp.exp(_interp_log_table(table.ln_number, x_grow)),
            jnp.exp(_interp_log_table(table.ln_volume, x_grow)))
