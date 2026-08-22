"""ECHAM 1-moment cloud microphysics (flux-coupled column sweep).

This module implements the single-moment bulk microphysics as a top-down
column sweep (:func:`cloud_microphysics_column_sweep`, wrapped by the
composable :class:`Echam1MMicrophysics` term):

- Autoconversion of cloud water to rain (Beheng 1994 default, or
  Khairoutdinov and Kogan 2000 via ``autoconversion_scheme``)
- Accretion of cloud droplets by rain
- Autoconversion/aggregation of cloud ice to snow (Levkov et al., 1992)
- Riming of cloud water by falling snow
- Melting of snow and cloud ice above the freezing level
- Sedimentation of cloud ice
- Rotstayn (1997) rain evaporation below cloud

Based on the ECHAM6/ICON ``mo_cloud.f90`` single-moment branch
(Lohmann and Roeckner, 1996).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
import tree_math

import jcm.constants as c


@tree_math.struct
class MicrophysicsParameters:
    """Configuration parameters for cloud microphysics"""
    
    # Autoconversion parameters
    ccraut: float        # Critical cloud water for autoconversion (kg/kg)
    smooth_ccraut: float # Sigmoid width of the KK2000 qc threshold (kg/kg);
                         # only read in KK2000 mode — Beheng uses ccraut as
                         # a rate coefficient (already smooth)
    ccracl: float        # Accretion coefficient (cloud to rain)
    cauloc: float        # ECHAM ``zrac2`` local-rain accretion enhancement.
                         # 0.0 is the ECHAM6.3 default (zrac2 disabled); raise to
                         # let the in-step autoconverted rain ALSO collect qc
                         # from its source layer (ECHAM mo_cloud.f90:791).
    clmin: float         # Lower bound on ``zauloc = clip(cauloc·dz/5000, clmin, clmax)``
    clmax: float         # Upper bound on ``zauloc`` (ECHAM6.3: 0.0 / 0.5).
    ceffmin: float       # Minimum cloud droplet radius (microns)
    ceffmax: float       # Maximum cloud droplet radius (microns)
    
    # Ice microphysics parameters
    cn0s: float          # Snow particle number density (1/m^3)
    crhosno: float       # Snow density (kg/m^3)
    ccsaut: float        # Levkov ice→snow autoconversion coefficient
                         # (ECHAM mo_echam_cloud_params: 95.0)
    ccsacl: float        # Riming efficiency of snow collecting cloud
                         # water (ECHAM: 0.10)
    cvtfall: float       # Terminal velocity factor for ice
    cthomi: float        # Homogeneous ice nucleation temperature (K)
    csecfrl: float       # Critical ice fraction for Bergeron-Findeisen
    
    # Collection efficiencies
    ccollec: float       # Collection efficiency rain/cloud
    ccollei: float       # Collection efficiency snow/ice
    
    # Time scale parameters
    tau_melt: float      # Melting time scale (s)
    tau_freeze: float    # Freezing time scale (s)
    
    # Evaporation/sublimation parameters
    cevaprain: float     # Rain evaporation coefficient
    cevapsnow: float     # Snow sublimation coefficient
    
    # Sedimentation parameters
    vt_ice: float        # Ice crystal fall speed (m/s)
    vt_snow_a: float     # Snow fall speed coefficient a
    vt_snow_b: float     # Snow fall speed exponent b
    vt_rain_a: float     # Rain fall speed coefficient a
    vt_rain_b: float     # Rain fall speed exponent b
    
    # Cloud droplet number concentration
    base_cdnc: float     # Baseline CDNC in clean air (1/m³), modulated by aerosol cdnc_factor

    # Mixed-phase split for the saturation-adjustment step. Below
    # ``t_mix_min`` condensate becomes 100% ice; above ``t_mix_max`` it
    # becomes 100% liquid. In between, the partition weighs liquid by
    # ``(T - t_mix_min)/(t_mix_max - t_mix_min)``. These match the
    # defaults previously held on ``CloudParameters``; the values live on
    # MicrophysicsParameters now because cuadjtq-style condensation is
    # part of the merged column-sweep cloud routine (see
    # :func:`cloud_microphysics_column_sweep`).
    t_mix_min: float
    t_mix_max: float

    # Numerical parameters. Two distinct floors, do NOT mix them:
    #  - ``epsilon`` (~1e-12): a PHYSICAL/numerical floor. Bounds a denominator
    #    or a quantity away from a value it should never realistically reach
    #    (e.g. a cloud fraction that gates whether a cell is cloudy). It changes
    #    the forward result and is chosen to be physically negligible there.
    #  - ``d_epsilon`` (~1e-30): a DIFFERENTIABILITY floor. Used only to keep
    #    the *masked/dead* branch of a ``where`` strictly positive so a
    #    ``sqrt``/fractional-power there has a finite derivative (issue #558).
    #    It must be FAR below any real value so it never changes the forward —
    #    using ``epsilon`` here silently perturbs the physics (it inflated the
    #    ice fall speed and opened the water budget ~22%).
    epsilon: float       # Small number for numerical stability
    d_epsilon: float     # Absolute floor for differentiability guards only
    dt_sedi: float       # Sub-timestep for sedimentation (s)
    cqtmin: float        # ECHAM ``cqtmin`` (mo_echam_cloud_params): the
                         # cloud-fraction floor below which a cell counts as
                         # cloud-free and its condensate force-evaporates
                         # (the ``nloidx`` partition, #668)
    ccwmin: float        # ECHAM ``ccwmin`` (mo_echam_cloud_params): grid-mean
                         # condensate below which a cell no longer counts as
                         # cloudy — drives the post-microphysics ``paclc``
                         # write-back (mo_cloud.f90:1280, #687)

    # Autoconversion scheme selector (int flag — JAX won't trace strings).
    # 0 = Beheng (1994) implicit form (default; robust at large dt).
    # 1 = Khairoutdinov & Kogan (2000) explicit form (good fit for 2M
    #     microphysics with prognostic Nc).
    # ``ccraut`` is interpreted differently by each scheme: in Beheng
    # it's the rate prefactor (default 15.0); in KK2000 it's the qc
    # threshold above which autoconversion fires (a small g/kg-scale
    # value is appropriate, e.g. 1e-5).
    autoconversion_scheme: int

    SCHEME_BEHENG = 0
    SCHEME_KK2000 = 1

    @classmethod
    def default(cls, ccraut=15.0, smooth_ccraut=5e-5,
                ccracl=6.0, cauloc=0.0, clmin=0.0, clmax=0.5,
                 ceffmin=10.0, ceffmax=150.0, cn0s=3.0e6,
                 crhosno=100.0, ccsaut=95.0, ccsacl=0.1,
                 cvtfall=3.29, cthomi=233.15, csecfrl=0.1, ccollec=0.7,
                 ccollei=0.3, tau_melt=100.0, tau_freeze=100.0, cevaprain=1.0e-3,
                 cevapsnow=5.0e-4, vt_ice=0.1, vt_snow_a=8.8, vt_snow_b=0.15,
                 vt_rain_a=386.0, vt_rain_b=0.67, base_cdnc=100.0e6,
                 t_mix_min=238.15, t_mix_max=273.15,
                 epsilon=1.0e-12, d_epsilon=1.0e-30, dt_sedi=10.0,
                 cqtmin=1.0e-12, ccwmin=1.0e-7,
                 autoconversion_scheme=0) -> 'MicrophysicsParameters':
        """Return default microphysics parameters.

        ``autoconversion_scheme`` accepts either the int constant
        (``SCHEME_BEHENG`` / ``SCHEME_KK2000``) or the string aliases
        ``"beheng"`` / ``"kk2000"``.
        """
        if isinstance(autoconversion_scheme, str):
            scheme_map = {
                "beheng": cls.SCHEME_BEHENG,
                "kk2000": cls.SCHEME_KK2000,
            }
            autoconversion_scheme = scheme_map[autoconversion_scheme]

        return cls(
            ccraut=jnp.array(ccraut),
            smooth_ccraut=jnp.array(smooth_ccraut),
            ccracl=jnp.array(ccracl),
            cauloc=jnp.array(cauloc),
            clmin=jnp.array(clmin),
            clmax=jnp.array(clmax),
            ceffmin=jnp.array(ceffmin),
            ceffmax=jnp.array(ceffmax),
            cn0s=jnp.array(cn0s),
            crhosno=jnp.array(crhosno),
            ccsaut=jnp.array(ccsaut),
            ccsacl=jnp.array(ccsacl),
            cvtfall=jnp.array(cvtfall),
            cthomi=jnp.array(cthomi),
            csecfrl=jnp.array(csecfrl),
            ccollec=jnp.array(ccollec),
            ccollei=jnp.array(ccollei),
            tau_melt=jnp.array(tau_melt),
            tau_freeze=jnp.array(tau_freeze),
            cevaprain=jnp.array(cevaprain),
            cevapsnow=jnp.array(cevapsnow),
            vt_ice=jnp.array(vt_ice),
            vt_snow_a=jnp.array(vt_snow_a),
            vt_snow_b=jnp.array(vt_snow_b),
            vt_rain_a=jnp.array(vt_rain_a),
            vt_rain_b=jnp.array(vt_rain_b),
            base_cdnc=jnp.array(base_cdnc),
            t_mix_min=jnp.array(t_mix_min),
            t_mix_max=jnp.array(t_mix_max),
            epsilon=jnp.array(epsilon),
            d_epsilon=jnp.array(d_epsilon),
            cqtmin=jnp.array(cqtmin),
            ccwmin=jnp.array(ccwmin),
            dt_sedi=jnp.array(dt_sedi),
            autoconversion_scheme=int(autoconversion_scheme),
        )


class MicrophysicsState(NamedTuple):
    """Microphysics state variables and diagnostics"""
    
    # Precipitation fluxes (kg/m²/s). ``rain_flux`` / ``snow_flux`` are
    # the grid-mean fluxes LEAVING each layer (crossing its lower
    # boundary) as the column sweep propagates precipitation downward:
    # the bottom level equals the surface ``precip_rain`` /
    # ``precip_snow`` by construction. ``snow_flux`` is the total frozen
    # flux — snow plus the sedimenting cloud-ice flux (``zxiflux``) that
    # ECHAM folds into surface snow at the bottom level. The per-layer
    # PRODUCTION (before evaporation depletes the falling flux) is kept
    # separately in ``rain_source`` / ``snow_source``.
    rain_flux: jnp.ndarray      # Rain flux leaving each level
    snow_flux: jnp.ndarray      # Snow(+falling-ice) flux leaving each level
    rain_source: jnp.ndarray    # Per-layer rain production (kg/m²/s)
    snow_source: jnp.ndarray    # Per-layer snow production (kg/m²/s)
    rain_evap_flux: jnp.ndarray  # Per-layer rain evaporation (kg/m²/s, #499)

    # In-cloud values
    qc_in_cloud: jnp.ndarray    # In-cloud liquid water (kg/kg)
    qi_in_cloud: jnp.ndarray    # In-cloud ice (kg/kg)
    
    # Process rates (kg/kg/s)
    autoconv_rate: jnp.ndarray  # Autoconversion rate
    accretion_rate: jnp.ndarray # Accretion rate
    melting_rate: jnp.ndarray   # Melting rate
    freezing_rate: jnp.ndarray  # Freezing rate
    
    # Precipitation at surface
    precip_rain: jnp.ndarray    # Surface rain (kg/m²/s)
    precip_snow: jnp.ndarray    # Surface snow (kg/m²/s)


class MicrophysicsTendencies(NamedTuple):
    """Tendencies from microphysics processes"""
    
    dtedt: jnp.ndarray          # Temperature tendency (K/s)
    dqdt: jnp.ndarray           # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray          # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray          # Cloud ice tendency (kg/kg/s)
    dqrdt: jnp.ndarray          # Rain water tendency (kg/kg/s)
    dqsdt: jnp.ndarray          # Snow tendency (kg/kg/s)


def cloud_droplet_radius(
    cloud_water: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Calculate effective cloud droplet radius
    
    Args:
        cloud_water: Cloud liquid water content (kg/kg)
        air_density: Air density (kg/m³)
        droplet_number: Droplet number concentration (1/kg)
        config: Microphysics configuration
        
    Returns:
        Effective radius (m)

    """
    # Convert mixing ratio to mass concentration
    cloud_water_density = cloud_water * air_density  # kg/m³
    
    # Convert droplet number from per kg to per m³
    droplet_density = droplet_number * air_density  # 1/m³
    
    # Volume of single droplet
    volume_per_droplet = cloud_water_density / (droplet_density + config.epsilon) / c.rhow  # m³
    
    # Volume mean radius. Double-where guard: the cube root has an infinite
    # derivative at 0 and the clip below has zero slope there, so without a
    # safe base the backward pass multiplies 0 × ∞ = NaN at cloud-free
    # points. Forward values are unchanged (0**(1/3) was 0, then clipped).
    has_water = volume_per_droplet > 0.0
    volume_safe = jnp.where(has_water, volume_per_droplet, 1.0)
    radius = jnp.where(
        has_water, (3.0 * volume_safe / (4.0 * jnp.pi)) ** (1.0 / 3.0), 0.0,
    )

    # Apply limits
    radius = jnp.clip(radius, config.ceffmin * 1e-6, config.ceffmax * 1e-6)
    
    return radius


def autoconversion_beheng(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Autoconversion of cloud water to rain — Beheng (1994) implicit form.

    Mirrors ECHAM ``mo_cloud.f90`` lines 841-863. The implicit integration
    is what makes this scheme robust at realistic post-convection cloud
    water values: the depletion fraction stays in [0, 1] even when the
    instantaneous Beheng rate × dt would overshoot.

        zraut_rate = ccraut * 1.2e27 / rho * Nc^-3.3 * rho^4.7 * qc^3.7
        qc_remain  = (1 + zraut_rate * dt * 3.7 * qc^3.7) ^ (-1/3.7)
        autoconv   = qc * (1 - qc_remain) / dt

    Default in the 1M scheme. The KK2000 form
    (``autoconversion_kk2000``) is also available and may be a better
    pairing with explicit-Nc 2M microphysics; pick via
    ``MicrophysicsParameters(autoconversion_scheme="beheng" | "kk2000")``.

    Args:
        cloud_water: Grid-mean cloud water mixing ratio (kg/kg)
        cloud_fraction: Cloud fraction (0-1)
        air_density: Air density (kg/m³)
        droplet_number: Cloud droplet number concentration (1/m³)
        dt: Time step (s)
        config: Microphysics configuration (uses ccraut, epsilon)

    Returns:
        Grid-mean autoconversion rate (kg/kg/s)

    """
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / jnp.maximum(cloud_fraction, config.epsilon),
        0.0,
    )

    zexm1 = 3.7  # 4.7 - 1.0
    nc_per_cm3 = droplet_number * air_density * 1e-6  # 1/cm³
    rho_g_cm3 = air_density * 1e-3                    # g/cm³

    # Beheng's Nc^-3.3 dependence blows up for Nc → 0; floor at 1/cm³
    nc_safe = jnp.maximum(nc_per_cm3, 1.0)
    zraut_rate = (
        config.ccraut * 1.2e27 / air_density
        * nc_safe ** (-3.3)
        * rho_g_cm3 ** 4.7
    )

    # Implicit integration: protect against (qc^zexm1) underflow at
    # near-zero qc — the formula gives no autoconv there anyway.
    qc_pow = jnp.where(qc_in_cloud > 1e-12, qc_in_cloud ** zexm1, 0.0)
    denominator = 1.0 + zraut_rate * dt * zexm1 * qc_pow
    qc_remaining_frac = denominator ** (-1.0 / zexm1)
    autoconv_in_cloud = qc_in_cloud * (1.0 - qc_remaining_frac) / dt

    # Convert to grid-mean
    return autoconv_in_cloud * cloud_fraction


def autoconversion_kk2000(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """Autoconversion of cloud water to rain — Khairoutdinov & Kogan (2000).

    Explicit-rate form:

        P_aut = 1350 * qc^2.47 * Nc_cm3^(-1.79)   [kg/kg/s, qc in kg/kg]

    Activates above the ``ccraut`` threshold. KK2000 was the original
    1M default and remains a good fit for 2M microphysics where the
    droplet number ``Nc`` is a prognostic variable. In the 1M context
    with prescribed ``Nc`` and large dt, the explicit form can produce
    ``rate × dt > qc`` at high cloud water (~37500 % depletion at
    qc = 0.3 g/kg, dt = 1800 s); downstream code must clip to mass
    conservation. ``autoconversion_beheng`` is the more robust 1M
    default; KK2000 is preferred when paired with prognostic ``Nc``.

    Args:
        cloud_water: Grid-mean cloud water mixing ratio (kg/kg)
        cloud_fraction: Cloud fraction (0-1)
        air_density: Air density (kg/m³)
        droplet_number: Cloud droplet number concentration (1/m³)
        dt: Time step (s) — unused (explicit rate); kept in signature
            for parity with ``autoconversion_beheng``.
        config: Microphysics configuration (uses ccraut, epsilon)

    Returns:
        Grid-mean autoconversion rate (kg/kg/s)

    """
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / jnp.maximum(cloud_fraction, config.epsilon),
        0.0,
    )

    nc_cm3 = droplet_number * air_density * 1e-6     # 1/cm³

    # KK2000 eq. 29 is a MIXING-RATIO rate: dqr/dt = 1350·qc^2.47·Nc^−1.79
    # with qc in kg/kg and Nc in cm⁻³, yielding kg/kg/s directly. The
    # previous code fed qc in g/m³ (×ρ×1000 ≈ ×1200) into the 2.47 power
    # and then applied a spurious g/m³→kg/kg back-conversion — a net
    # ~2.6e4× overestimate (review finding 1.5; non-default branch).
    # Smooth threshold (maintainability review B.2.5): with the hard
    # ``where(qc > ccraut, rate, 0)`` the threshold appears only in the
    # inequality, so d(rate)/d(ccraut) is identically zero — ccraut was
    # calibratable only in Beheng mode. The sigmoid ramp puts it in the
    # value; width -> 0 recovers the hard gate. The qc power base is
    # double-where-guarded so the ramp's sub-threshold tail cannot
    # differentiate a negative/zero base (0**x cotangent class).
    has_qc = qc_in_cloud > 0.0
    qc_safe = jnp.where(has_qc, qc_in_cloud, 1.0)
    ramp = jax.nn.sigmoid(
        (qc_in_cloud - config.ccraut) / config.smooth_ccraut
    )
    autoconv_rate = jnp.where(
        has_qc,
        ramp * 1350.0 * qc_safe ** 2.47 * (nc_cm3 + config.epsilon) ** (-1.79),
        0.0,
    )

    return autoconv_rate * cloud_fraction


def autoconversion(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters,
) -> jnp.ndarray:
    """Dispatcher — picks Beheng or KK2000 by ``config.autoconversion_scheme``.

    Both schemes have the same signature so ``lax.cond`` can switch
    cleanly between them at runtime.
    """
    return jax.lax.cond(
        config.autoconversion_scheme == MicrophysicsParameters.SCHEME_KK2000,
        lambda: autoconversion_kk2000(
            cloud_water, cloud_fraction, air_density, droplet_number, dt, config,
        ),
        lambda: autoconversion_beheng(
            cloud_water, cloud_fraction, air_density, droplet_number, dt, config,
        ),
    )


def ice_autoconversion(
    cloud_ice: jnp.ndarray,
    temperature: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters,
    air_density: jnp.ndarray = jnp.array(1.0),
) -> jnp.ndarray:
    """Ice→snow autoconversion — ECHAM's Levkov aggregation (mo_cloud.f90:996-1052).

    The aggregation timescale comes from the Moss (1995) effective radius
    of the in-cloud ice (``zrieff = 83.8·(IWC g/m³)^0.216`` µm), converted
    to a volume-mean size (Schumann form) and fed into Levkov's ``zdt2``;
    the rate coefficient ``ccsaut/zdt2`` is integrated IMPLICITLY
    (``x·(1 − 1/(1 + rate·dt·x))``) so per-step depletion is bounded with
    no artificial qi threshold and no 1/dt in the physical rate. The
    previous placeholder (``0.001·(qi−0.3e-3)/dt`` with a Gaussian T
    factor) was ~3 orders of magnitude too weak, timestep-dependent, and
    never seeded snow from cirrus (review finding 2.10).

    Args:
        cloud_ice: IN-CLOUD ice mixing ratio when ``cloud_fraction`` is 1
            (as the column sweep calls it), else grid-mean (converted
            internally).
        temperature: Temperature (K).
        cloud_fraction: Cloud fraction (0-1).
        dt: Time step (s).
        config: Microphysics configuration (ccsaut).
        air_density: Air density (kg/m³) for the IWC and the 1.3/ρ
            correction.

    Returns:
        Grid-mean autoconversion rate (kg/kg/s).

    """
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / jnp.maximum(cloud_fraction, config.epsilon),
        0.0,
    )
    iwc_gm3 = qi_in_cloud * air_density * 1000.0
    zrieff = 83.8 * jnp.where(iwc_gm3 > 0.0, iwc_gm3, 1.0) ** 0.216
    zrieff = jnp.clip(zrieff, config.ceffmin, config.ceffmax)
    zrih = jnp.sqrt(5113188.0 + 2809.0 * zrieff ** 3) - 2261.0
    zqrho_p033 = (1.3 / jnp.maximum(air_density, config.epsilon)) ** 0.33
    crhoi = 500.0  # ECHAM cloud-ice bulk density [kg/m³]
    zc1 = 17.5 * air_density / crhoi * zqrho_p033
    zdt2 = -6.0 / jnp.maximum(zc1, config.epsilon) * (
        jnp.log10(jnp.maximum(zrih, 1.0)) / 3.0 - 2.0
    )
    rate_coeff = config.ccsaut / jnp.maximum(zdt2, config.epsilon)
    zsaut = qi_in_cloud * (
        1.0 - 1.0 / (1.0 + rate_coeff * dt * jnp.maximum(qi_in_cloud, 0.0))
    )
    zsaut = jnp.where(qi_in_cloud > 0.0, jnp.maximum(zsaut, 0.0), 0.0)
    return cloud_fraction * zsaut / dt


def sedimentation_flux(
    hydrometeor: jnp.ndarray,
    air_density: jnp.ndarray,
    dz: jnp.ndarray,
    terminal_velocity: jnp.ndarray,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate sedimentation flux and tendency for a hydrometeor
    
    Uses upwind differencing with flux limiter to maintain stability.
    JAX-compatible implementation without loops.
    
    Args:
        hydrometeor: Hydrometeor mixing ratio (kg/kg) [nlev]
        air_density: Air density (kg/m³) [nlev]
        dz: Layer thickness (m) [nlev]
        terminal_velocity: Fall velocity (m/s) [nlev]
        dt: Time step (s)
        
    Returns:
        Tuple of (flux [nlev+1], tendency [nlev])

    """
    # Mass content (kg/m³)
    mass_content = hydrometeor * air_density
    
    # Calculate fluxes at each interface (upwind)
    # Flux from level k to k+1
    flux_unlimited = mass_content * terminal_velocity
    
    # CFL limiter to prevent overshooting
    max_flux = mass_content * dz / dt
    flux_limited = jnp.minimum(flux_unlimited, max_flux)
    
    # Build interface fluxes
    # flux[0] = 0 (top), flux[k+1] = flux from level k
    flux = jnp.concatenate([jnp.zeros(1), flux_limited])
    
    # Tendency from flux divergence
    # (flux_in - flux_out) / (dz * rho)
    flux_in = flux[:-1]  # Flux from above
    flux_out = flux[1:]  # Flux to below
    tendency = (flux_in - flux_out) / (dz * air_density)
    
    return flux, tendency


def _saturation_adjustment_layer(
    T: jnp.ndarray,
    q: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    p: jnp.ndarray,
    config: MicrophysicsParameters,
    cf: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-layer cuadjtq-style saturation adjustment.

    Ports the two-pass Newton step from ``sundqvist.condensation_evaporation``
    so the column-sweep microphysics can do its own condensation
    locally — the way ECHAM's ``mo_cloud.f90`` lines 696-784 do. With
    condensation inside the sweep, the autoconv / accretion / rain-evap
    that follow at the same level operate on post-condensation
    ``(T', q', qc', qi')``, which closes the rain-evap ↔ re-condensation
    loop within a single ``dt`` and breaks the two-step feedback that
    forced PR #458 to revert the column-sweep variant.

    Args:
        T: Temperature [K] (per-layer scalar inside the scan).
        q: Specific humidity [kg/kg].
        qc, qi: Cloud water and ice mixing ratios [kg/kg].
        p: Pressure at the layer [Pa].
        config: :class:`MicrophysicsParameters` — only ``t_mix_min`` /
            ``t_mix_max`` are read here; everything else is unused.

    Returns:
        ``(dT, dq, dqc, dqi)`` — per-step absolute increments
        (kg/kg, K) over ``dt``. Add to the input fields to get the
        post-adjustment values:

            T_post = T + dT
            q_post = q + dq
            qc_post = max(qc + dqc, 0)
            qi_post = max(qi + dqi, 0)

        Positive ``dqc`` / ``dqi`` indicate condensation onto cloud
        condensate; negative values are evaporation of cloud water/ice.
        ``dq = -(dqc + dqi)`` by construction so the column-integrated
        vapour balance closes.

    """
    # Imported here rather than at module top to keep the dependency
    # explicit and to avoid pulling sundqvist into the module-load path
    # of every code path that touches echam_1m.
    from jcm.physics.clouds.sundqvist import _qs_and_dqs_dt

    weight_liquid = jnp.clip(
        (T - config.t_mix_min)
        / jnp.maximum(config.t_mix_max - config.t_mix_min, 1e-3),
        0.0, 1.0,
    )
    L_eff = weight_liquid * c.alhc + (1.0 - weight_liquid) * c.alhs
    L_cp = L_eff / c.cpd

    # ---- Pass 1: linearised Newton step, CLOUD-FRACTION weighted ----
    # ECHAM's condensational growth/dissipation ``zqcdif`` carries a
    # ``zclcaux`` factor (mo_cloud.f90:729): condensation happens in the
    # cloudy part of the cell, so a cf=0 cell generates NO condensate from
    # pass 1 at all — its supersaturation stays vapour up to the pass-2
    # grid-box allowance below. Without this factor the adjustment was
    # grid-mean-unconditional, which made the ``zxlevap`` clearing in the
    # sweep a measured NO-OP: everything the clearing released re-condensed
    # immediately, in every regime (#668). ``cf=None`` (legacy callers)
    # preserves the unweighted behaviour.
    qs, dqs_dt = _qs_and_dqs_dt(p, T)
    q_excess = q - qs
    cond1 = q_excess / jnp.maximum(1.0 + L_cp * dqs_dt, 1e-3)
    if cf is not None:
        cond1 = cond1 * cf
    total_cloud = qc + qi
    cond1 = jnp.maximum(cond1, -total_cloud)
    cond1 = jnp.minimum(cond1, jnp.maximum(q, 0.0))

    # ---- Pass 2: cleanup any residual super-saturation above 1% qs ----
    T_p1 = T + L_cp * cond1
    q_p1 = q - cond1
    qs_p1, _ = _qs_and_dqs_dt(p, T_p1)
    oversat_tol = 0.01 * qs_p1
    cond2 = jnp.maximum(
        (q_p1 - qs_p1 - oversat_tol) / jnp.maximum(1.0 + L_cp * dqs_dt, 1e-3),
        0.0,
    )
    cond2 = jnp.minimum(cond2, jnp.maximum(q_p1, 0.0))
    cond_total = cond1 + cond2

    # ---- Partition between liquid / ice ----
    # The guard threshold is ``d_epsilon``, NOT ``> 0``. The double-where
    # protects the unselected branch, but the division VJP on the SELECTED
    # branch computes ``-g * qc / (safe_total * safe_total)``, and for
    # 0 < total_cloud < ~1e-154 (spectral-ringing condensate tails reach
    # 1e-287 in real JW columns) the squared denominator underflows to 0,
    # giving 0/0 = NaN in the reverse pass while the forward is perfectly
    # finite. Any total below ``d_epsilon`` is physically no cloud at all,
    # and treating it as the cloud-free branch changes the increments by
    # at most O(total_cloud) ~ 1e-30 kg/kg.
    has_cloud = total_cloud > config.d_epsilon
    safe_total = jnp.where(has_cloud, total_cloud, 1.0)
    qc_frac = jnp.where(has_cloud, qc / safe_total, 0.0)
    qi_frac = jnp.where(has_cloud, qi / safe_total, 0.0)
    L_evap = jnp.where(
        has_cloud,
        (qc * c.alhc + qi * c.alhs) / safe_total,
        L_eff,
    )

    dq = -cond_total
    dqc = jnp.where(
        cond_total > 0,
        weight_liquid * cond_total,
        cond_total * qc_frac,
    )
    dqi = jnp.where(
        cond_total > 0,
        (1.0 - weight_liquid) * cond_total,
        cond_total * qi_frac,
    )
    L_for_dT = jnp.where(cond_total > 0, L_eff, L_evap)
    dT = L_for_dT * cond_total / c.cpd
    return dT, dq, dqc, dqi


def _qsat_water(pressure: jnp.ndarray, temperature: jnp.ndarray):
    """Saturation specific humidity over water + the vapor pressure es.

    Uses the same Tetens form as :func:`sundqvist.saturation_vapor_pressure_water`
    so the rain-evaporation step is consistent with the condensation step.
    The conversion from ``es`` to ``qs`` follows the standard mixing-ratio
    formula ``qs = ε·es/(p - (1-ε)·es)`` (equivalent to ICON's
    ``zqsw = uaw/(p - vtmpc1·uaw)`` after expanding ``uaw = ε·es``).
    Returns ``(qsw, esw_pa)``.
    """
    t_c = temperature - c.tmelt
    es = 610.78 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    es_safe = jnp.minimum(es, 0.5 * pressure)
    qsw = c.eps * es_safe / jnp.maximum(pressure - (1.0 - c.eps) * es_safe, 1.0)
    return qsw, es_safe


def cloud_microphysics_column_sweep(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: Optional[MicrophysicsParameters] = None,
) -> Tuple[MicrophysicsTendencies, MicrophysicsState]:
    """ECHAM ``mo_cloud.f90`` column-sweep cloud + microphysics routine.

    Faithful port of ICON/ECHAM ``mo_cloud.f90`` lines 260-1080. Treats
    rain (``zrfl``) and snow (``zsfl``) as **downward fluxes** that
    propagate top-to-bottom through the column within a single ``dt``
    and now also does the **per-layer saturation adjustment** (cuadjtq
    Newton step) inside the same column sweep — matching ECHAM's
    structure where condensation, autoconversion, rain evap, and flux
    propagation all live in one routine.

    Per-layer order (top → bottom):

    1. **Snow melt** for incoming flux at ``T > 273 K``: convert
       ``zsfl`` → ``zrfl`` (``mo_cloud.f90:319-323``).
    2. **Saturation adjustment** (``_saturation_adjustment_layer``):
       linearised Newton step on the layer ``(T, q, qc, qi)`` so the
       layer is non-supersaturated *and* any subsaturated cloud
       water/ice evaporates. Mirrors ECHAM ``mo_cloud.f90`` lines
       696-784.
    3. **Microphysics** from the *post-condensation* ``(T', q', qc', qi')``:
       Beheng/KK2000 autoconversion (``qc' → rain``), Lin-style ice
       autoconversion (``qi' → snow``), rain accretion of cloud water,
       snow riming of cloud water (T < ``tmelt``), snow aggregation of
       cloud ice. Accretion / riming / aggregation use ECHAM's
       implicit-Euler form
       ``zrac1 = zxlb·(1 - exp(-ccracl·zxrp1·dt))`` with the
       Marshall-Palmer in-precipitating-area concentration ``zxrp1``
       (mo_cloud.f90:800-877), so per-step depletion is bounded in
       ``[0, qc]`` and can't drive ``qc`` negative even at high
       incoming rain flux.
    4. **Rotstayn (1997) rain evaporation** below cloud, using the
       *post-condensation* ``q'`` so it can't push the layer above
       saturation (``zevp_max_subsat = 0.99·(qs - q')``).
    5. **Flux update** for ``zrfl`` / ``zsfl`` / ``zclcpre`` carry.

    Why no within-step cleanup pass: the 0.99·(qs - q') cap on rain
    evap means the layer cannot be pushed past saturation in step 4,
    so a second saturation-adjustment pass would always be a no-op.
    An earlier draft of this routine ran one anyway as a defensive
    measure; in practice it re-condensed the slight super-saturation
    that rain-evap-cooling produced (qs drops with T → small
    super-saturation appears → cleanup condenses → more autoconv →
    more rain), reigniting the rain-evap ↔ re-condensation feedback
    PR #458 originally caught. The cap alone is sufficient.

    Bottom-of-column ``zrfl`` / ``zsfl`` become the surface precipitation
    flux (``state.precip_rain`` / ``state.precip_snow``).

    The per-layer ``(dT, dq, dqc, dqi)`` returned to the caller pool
    every contribution from steps 1-5 into rate-form tendencies — the
    composable physics integrator applies ``state += dt * tend`` as
    usual.

    What's INTENTIONALLY MISSING from this port:

    * **Snow sublimation in subsaturated layers** (``mo_cloud.f90``
      332-393, Lin et al. 1983). Same structural shape as rain evap;
      tracked as a separate add when stability data justifies it.
    * **Rain freezing** below ``cthomi`` and the **Bergeron-Findeisen**
      ice-from-supercooled-water process (covered by the 2M scheme).

    """
    if config is None:
        config = MicrophysicsParameters.default()

    nlev = temperature.shape[0]
    pmref = air_density * layer_thickness     # kg/m² per layer

    # Phase weights for the latent-heat update from snow melting / riming.
    zlsdcp = c.alhs / c.cpd
    zlvdcp = c.alhc / c.cpd
    zlfdcp = zlsdcp - zlvdcp        # alhf / cp

    def step(carry, level_inputs):
        zrfl, zsfl, zclcpre, zxiflux = carry
        T0, q0, p, qc0, qi0, cf, rho, dz, ndrop, mref, is_bottom = level_inputs

        # ---------- (0a) instant melt of cloud ice above the melting point
        # (ECHAM zimlt): ice cannot persist at T > tmelt; it converts to
        # cloud water, consuming the latent heat of fusion (review 2.16).
        zimlt = jnp.where(T0 > c.tmelt, qi0, 0.0)
        qi0 = qi0 - zimlt
        qc0 = qc0 + zimlt
        dTdt_imlt = -zlfdcp * zimlt / dt

        # ---------- (0b) ice sedimentation (ECHAM mo_cloud.f90:580-615) ----
        # Analytic exponential integral: the grid-mean qi relaxes toward the
        # influx-fed equilibrium ``zal2 = zxiflux/(ρ·v_fall)`` with rate
        # ``v_fall·g·ρ·dt/Δp`` — a layer can GAIN ice from the flux above.
        # The flux out feeds the level below through the scan carry; the
        # residual at the bottom level joins the snow flux (ECHAM jk==klev).
        # ECHAM 6.3's 1M does NOT sublimate the falling ice on the way down.
        # This was entirely absent from the sweep — cirrus had no sink and
        # never precipitated (review finding 2.9).
        zdp = mref * c.grav  # layer Δp [Pa]
        zxip1 = jnp.maximum(qi0, 0.0)
        # Double-where guard: ``x ** 0.16`` at ``x == 0`` (an ice-free layer,
        # the common case) has an infinite derivative, so the reverse pass
        # NaNs even though the forward is 0. The ``where`` keeps the forward
        # exactly 0 where there is no ice; the inner floor only has to make the
        # base strictly positive for the differentiated branch — hence the
        # negligible ``d_epsilon`` (NOT ``epsilon``: a 1e-12 floor would
        # inflate the fall speed of tiny-but-nonzero ice by orders of
        # magnitude, opening the water budget; see the ``epsilon`` /
        # ``d_epsilon`` note on MicrophysicsParameters). Issue #558.
        zxifall = config.cvtfall * jnp.where(
            rho * zxip1 > 0.0,
            jnp.maximum(rho * zxip1, config.d_epsilon) ** 0.16,
            0.0,
        )
        zal1 = jnp.exp(-zxifall * c.grav * rho * dt / jnp.maximum(zdp, config.epsilon))
        # Influx contribution ``zal2 * (1 - zal1)`` with
        # ``zal2 = zxiflux / (rho * v)``: analytically this has a REMOVABLE
        # 0/0 limit as the fall speed v -> 0 (it tends to
        # ``zxiflux * k / rho`` with ``k = g * rho * dt / dp``), but the
        # factored form with an epsilon floor destroys the cancellation in
        # reverse mode: d(zal2)/d(zxiflux) = 1/max(rho*v, eps) is up to 1e12
        # per level, and ``zxiflux`` is the scan carry, so these factors
        # COMPOUND across levels and overflow the backward pass to inf (the
        # first saturated min/max VJP then turns the inf into NaN — the
        # convection-parameter NaN gradients). Rewrite via the stable
        # phi(x) = (1 - exp(-x))/x with its series limit at small x, so both
        # the value and every partial derivative stay O(1).
        sed_x = zxifall * c.grav * rho * dt / jnp.maximum(zdp, config.epsilon)
        sed_x_safe = jnp.maximum(sed_x, 1.0e-8)
        sed_phi = jnp.where(
            sed_x > 1.0e-8,
            -jnp.expm1(-sed_x_safe) / sed_x_safe,
            1.0 - 0.5 * sed_x,
        )
        influx_gain = (
            zxiflux * c.grav * dt / jnp.maximum(zdp, config.epsilon) * sed_phi
        )
        zxised = jnp.maximum(0.0, zxip1 * zal1 + influx_gain)
        zqsed = zxised - zxip1
        zcons2_lev = 1.0 / (dt * c.grav)
        zxibot = jnp.maximum(0.0, zxiflux - zqsed * zcons2_lev * zdp)
        zqsed = (zxiflux - zxibot) / jnp.maximum(zcons2_lev * zdp, config.epsilon)
        qi0 = zxip1 + zqsed
        dqidt_sed = zqsed / dt
        # Bottom level: the remaining ice flux exits as snow (folded into
        # the snow flux below, before this layer's melt runs on it).
        zsfl = zsfl + jnp.where(is_bottom, zxibot, 0.0)
        zxiflux_out = jnp.where(is_bottom, 0.0, zxibot)

        # ---------- (1) snow melt at T > tmelt ----------
        # ICON ``mo_cloud.f90:319-323``. Uses the input T (pre-condensation)
        # since snow falling INTO this layer melts based on whether the
        # ambient air is above freezing — condensation hasn't run yet.
        zcons = (mref / dt) / jnp.maximum(zlfdcp, 1e-6)
        ztdif = jnp.maximum(0.0, T0 - c.tmelt)
        zsnmlt = jnp.minimum(0.99 * zsfl, zcons * ztdif)
        zrfl = zrfl + zsnmlt
        zsfl = zsfl - zsnmlt
        # ``zsnmlt`` is a FLUX [kg/m²/s]; the mixing-ratio rate is
        # flux/mref [1/s] with no further /dt (the extra /dt made melting
        # cool ~1800× too little — snow melted without paying the latent
        # heat of fusion, review finding 2.11).
        zsmlt_rate = zsnmlt / jnp.maximum(mref, config.epsilon)
        dTdt_melt = -zlfdcp * zsmlt_rate

        # ---------- (1b) cloud-free cells: force-evaporate ALL condensate --
        # ECHAM ``zxlevap``/``zxievap`` (mo_cloud.f90:660-670): in a cell the
        # cloud scheme declares cloud-free (``zclcaux <= cqtmin``), every
        # kg of condensate returns to vapour UNCONDITIONALLY — regardless of
        # saturation — with the matching latent cooling (:706-708, :711).
        # Without it, a cf=0 cell that reaches saturation (detrainment into
        # a clear cell, within-step moistening past the step-start cf
        # diagnosis, or the hard-zeroed cf above ``cloud_top_pressure_pa``)
        # accumulates condensate that no microphysical process can touch —
        # every source/sink below is cf-weighted, so at cf=0 only ice
        # sedimentation removes anything. A radiatively-active, permanently
        # growing condensate reservoir with no sink (#668, #537).
        #
        # Runs BEFORE the saturation adjustment so the adjustment acts on
        # the cleared state and may legitimately re-condense what the
        # thermodynamics supports — as vapour, subject to the cloud scheme
        # next step, not as orphaned condensate. NOTE: ECHAM's *partial*
        # clear-fraction evaporation ``(1−zclcaux)·...`` is commented out in
        # 6.3 (mo_cloud.f90:683-684), so cf>0 cells keep their condensate
        # here too — only the fully cloud-free cells clear.
        is_cloud_free = cf <= config.cqtmin
        zxlevap = jnp.where(is_cloud_free, jnp.maximum(qc0, 0.0), 0.0)
        zxievap = jnp.where(is_cloud_free, jnp.maximum(qi0, 0.0), 0.0)
        q0 = q0 + zxlevap + zxievap
        qc0 = qc0 - zxlevap
        qi0 = qi0 - zxievap
        T0 = T0 - zlvdcp * zxlevap - zlsdcp * zxievap
        dTdt_clearevap = (-zlvdcp * zxlevap - zlsdcp * zxievap) / dt
        dq_clearevap = zxlevap + zxievap        # absolute increments over dt

        # ---------- (2) pre-microphysics saturation adjustment ----------
        # Two-pass Newton condensation / evaporation on this layer's
        # ``(T0, q0, qc0, qi0)`` — same logic as ECHAM ``mo_cloud.f90``
        # 696-784. Outputs are absolute increments over ``dt``.
        dT_cond_a, dq_cond_a, dqc_cond_a, dqi_cond_a = _saturation_adjustment_layer(
            T0, q0, qc0, qi0, p, config, cf=cf,
        )
        T1 = T0 + dT_cond_a
        q1 = q0 + dq_cond_a
        qc1 = jnp.maximum(qc0 + dqc_cond_a, 0.0)
        qi1 = jnp.maximum(qi0 + dqi_cond_a, 0.0)

        # ---------- (3) microphysics on POST-condensation (T1, q1, qc1, qi1) ----------
        # Mirrors ECHAM ``mo_cloud.f90:795-879``: sequential depletion of
        # in-cloud ``zxlb`` (= qc/cf) and ``zxib`` (= qi/cf) by
        # autoconversion (zraut), accretion of cloud water by falling
        # rain (zrac1), local-rain accretion by the in-step autoconverted
        # rain (zrac2 — only fires when cauloc > 0, ECHAM default 0),
        # snow riming of cloud water (zsacl), and snow aggregation of
        # cloud ice (zsaci). Each step reads the post-previous-depletion
        # zxlb, so accretion sees the qc that autoconv left behind, not
        # the original. The grid-mean rain/snow source going into the
        # falling flux is cf-weighted ECHAM-style:
        #
        #     zrpr = cf · (zraut + zrac2) + zclcstar · zrac1
        #     zspr = cf · (zsaut + zsaci2) + zclcstar · zsaci1
        #
        # where ``zclcstar = min(cf, zclcpre)`` is the precipitating /
        # cloud area intersection (rain can only accrete from the area
        # where it overlaps cloud), and the in-cloud "wind back" via
        # the implicit-Euler ``1 - exp(-rate·dt)`` form bounds per-step
        # depletion in ``[0, zxlb]`` by construction so neither qc nor
        # qi can be driven negative.

        # Density correction: ECHAM defines ``zqrho = 1.3/ρ`` and uses
        # ``sqrt(zqrho)`` — i.e. sqrt(1.3/ρ) — in BOTH the Marshall-Palmer
        # concentrations and the Rotstayn rain evaporation (mo_cloud.f90;
        # review finding 2.13). The previous code used the inverted
        # sqrt(ρ/1.3) for zxrp1 (its comment asserted the opposite of the
        # reference), overestimating accretion by (1.3/ρ)^(8/9) — ~1.85×
        # at 500 hPa.
        zclcpre_safe = jnp.maximum(zclcpre, config.epsilon)
        # Density correction: ECHAM defines ``zqrho = 1.3/ρ`` and uses
        # ``sqrt(zqrho)`` in BOTH the Marshall-Palmer concentrations and
        # the Rotstayn rain evaporation (review finding 2.13 — the
        # previous zxrp1 divisor used the inverted sqrt(ρ/1.3)).
        zqrho_sqrt = jnp.sqrt(jnp.maximum(1.3 / jnp.maximum(rho, config.epsilon), 0.0))
        zqrho_sqrt_inv = zqrho_sqrt
        rain_present = (zrfl > config.epsilon) & (zclcpre > config.epsilon)
        snow_present = (zsfl > config.epsilon) & (zclcpre > config.epsilon)
        # Double-where guard on the fractional powers: ``x**(8/9)`` (and
        # ``x**(1/1.16)``) has an infinite derivative at x == 0, and masking
        # only the *output* with ``where`` still yields NaN in reverse mode
        # (the masked branch's 0 cotangent multiplies the ∞ derivative).
        # Substituting a safe base of 1.0 where the flux is absent keeps the
        # forward values bit-identical (the outer ``where`` already returned
        # 0 there) while making d(precip)/d(params) finite.
        zxrp1_base = jnp.where(
            rain_present,
            jnp.maximum(zrfl / zclcpre_safe / (12.45 * zqrho_sqrt), 0.0),
            1.0,
        )
        zxrp1 = jnp.where(rain_present, jnp.power(zxrp1_base, 8.0 / 9.0), 0.0)
        zxsp1_base = jnp.where(
            snow_present,
            jnp.maximum(zsfl / zclcpre_safe / config.cvtfall, 0.0),
            1.0,
        )
        zxsp1 = jnp.where(snow_present, jnp.power(zxsp1_base, 1.0 / 1.16), 0.0)

        # In-cloud values for the cascade. ECHAM works on ``zxlb`` /
        # ``zxib`` which are in-cloud mixing ratios (qc/cf, qi/cf).
        cf_safe = jnp.maximum(cf, config.epsilon)
        cloud_mask = cf > config.epsilon
        zxlb = jnp.where(cloud_mask, qc1 / cf_safe, 0.0)
        zxib = jnp.where(cloud_mask, qi1 / cf_safe, 0.0)
        zclcstar = jnp.minimum(cf, zclcpre)

        # Numerical safety: clamp the exponent in ``1 - exp(-x)``;
        # float32 overflows to denormalised zero past ~50 and gradients
        # through ``exp`` of a huge negative value are unstable.
        def _impl_depletion(arg):
            return 1.0 - jnp.exp(-jnp.minimum(arg, 50.0))

        # (3a) Beheng autoconversion: in-cloud qc → rain. Reuses the
        # standalone helper at cf=1 so the existing implementation owns
        # the rate formula; the returned value is then ``rate * 1 = rate``
        # in kg/kg/s in-cloud. Per-dt depletion = rate * dt.
        qcaut_rate_in_cloud = autoconversion(
            zxlb, jnp.array(1.0), rho, ndrop, dt, config,
        )
        zraut = jnp.minimum(qcaut_rate_in_cloud * dt, zxlb)  # in-cloud kg/kg over dt
        zxlb = zxlb - zraut

        # (3b) Rain accretion of cloud water (zrac1). Reads post-autoconv zxlb.
        zrac1 = zxlb * _impl_depletion(config.ccracl * zxrp1 * dt)
        zxlb = zxlb - zrac1

        # (3c) Snow riming of cloud water (ECHAM zsacl1, mo_cloud.f90:
        # 1054-1100): geometric sweep-out rate from the Marshall-Palmer
        # snow content, K = π·cn0s·3.078·λ^0.8125·√(1.3/ρ) with
        # λ = (zxsp1/(π·crhosno·cn0s))^0.8125-argument, times the riming
        # efficiency ccsacl = 0.10, integrated implicitly. The previous
        # code reused the RAIN accretion coefficient (ccracl = 6.0) on the
        # raw snow content — ~3× too much riming per step — and gated on
        # T < tmelt, which the Fortran doesn't (above melting the incoming
        # snow has already melted to rain). (Review finding 2.14.)
        zlamsm_arg = jnp.maximum(
            zxsp1 / (jnp.pi * config.crhosno * config.cn0s), 0.0,
        )
        ksnow = jnp.where(
            zxsp1 > config.epsilon,
            jnp.pi * config.cn0s * 3.078
            * jnp.where(zxsp1 > config.epsilon, zlamsm_arg, 1.0) ** 0.8125
            * zqrho_sqrt,
            0.0,
        )
        zsacl = zxlb * _impl_depletion(ksnow * config.ccsacl * dt)
        zxlb = zxlb - zsacl

        # (3d) Local-rain accretion (zrac2). ECHAM ``mo_cloud.f90:860``:
        # ``ztmp2 = -ccracl · zauloc · rho · zraut · dt`` then
        # ``zrac2 = zxlb · (1 - exp(ztmp2))``. ``zauloc`` scales with
        # layer thickness; clipped to ``[clmin, clmax]``. With the
        # ECHAM6.3 default ``cauloc=0`` this is identically zero — the
        # branch is here for parameter completeness and ICON-style
        # tunings that enable it.
        zauloc = jnp.clip(config.cauloc * dz / 5000.0, config.clmin, config.clmax)
        zrac2 = zxlb * _impl_depletion(
            config.ccracl * zauloc * rho * zraut,
        )
        zxlb = zxlb - zrac2

        # (3e) Ice autoconversion (qi → snow) and snow aggregation
        # (qi by falling snow). Sequential like the warm-rain side.
        qiaut_rate_in_cloud = ice_autoconversion(
            zxib, T1, jnp.array(1.0), dt, config, air_density=rho,
        )
        zsaut = jnp.minimum(qiaut_rate_in_cloud * dt, zxib)
        zxib = zxib - zsaut
        # Snow aggregation of cloud ice (zsaci1): same geometric kernel as
        # riming but with the temperature-dependent collection efficiency
        # zcolleffi = exp(0.025·(T − tmelt)) instead of ccsacl (finding
        # 2.14 — cold aggregation lacked the exponential suppression).
        zcolleffi = jnp.exp(0.025 * (T1 - c.tmelt))
        zsaci = zxib * _impl_depletion(ksnow * zcolleffi * dt)
        zxib = zxib - zsaci

        # Convert in-cloud per-dt depletions to grid-mean tendencies
        # (kg/kg/s). ECHAM's area weighting (mo_cloud.f90:1054-1100):
        # processes driven by the FALLING flux (zrac1 rain accretion,
        # zsacl riming, zsaci aggregation — all zxsp1/zxrp1-based) act on
        # the cloud∩precip overlap zclcstar; in-cloud processes (zraut,
        # zrac2, zsaut) act on the full cloud fraction. The previous code
        # put riming under cf (finding 2.14f).
        dqcdt_micro = -(
            cf * (zraut + zrac2) + zclcstar * (zrac1 + zsacl)
        ) / dt
        dqidt_micro = -(cf * zsaut + zclcstar * zsaci) / dt
        # Riming latent heat with the SAME area weight as the mass
        # (Fortran heats with the already-weighted zsacl, line 1251).
        dTdt_rime = zlfdcp * zclcstar * zsacl / dt

        # ---------- (4) Rotstayn rain evaporation on POST-condensation q1 ----------
        # ICON ``mo_cloud.f90:397-435``. ``zsusatw`` is the (negative)
        # sub-saturation w.r.t. liquid; ``zast+zbst`` are Rotstayn's
        # thermodynamic + vapour-diffusion coefficients. Using ``q1``
        # (not ``q0``) means rain evap can't push the layer above
        # saturation — the 0.99·(qs - q1) cap is what enforces this and
        # makes the original PR #458 within-step re-condensation pass
        # unnecessary in this version.
        qsw, esw = _qsat_water(p, T1)
        zsusatw = jnp.minimum(q1 / jnp.maximum(qsw, config.epsilon) - 1.0, 0.0)
        zdv = 2.21 / jnp.maximum(p, config.epsilon)
        zast = (
            c.alhc * (c.alhc / (c.rv * jnp.maximum(T1, 1.0)) - 1.0)
            / jnp.maximum(T1, 1.0) / 0.024
        )
        # Rotstayn's vapour-diffusion term is R_v·T/(D_v·e_sw) — ECHAM's
        # ``zesat = esw/rv`` in the denominator (review finding 2.12; the
        # missing R_v made rain evap ~1.5× too strong at 280 K, ~5× at
        # 250 K).
        zbst = c.rv * T1 / jnp.maximum(zdv * esw, config.epsilon)
        zthermo = jnp.maximum(zast + zbst, config.epsilon)
        zrfl_in_cf = zrfl / zclcpre_safe
        # Rotstayn (1997) per-area rate. The density factor here is the
        # *inverse* of the one accretion uses: see ECHAM mo_cloud.f90:415
        # — ``870 * sub * (zrfl/zclcpre)**0.61 * zqrho/cqtmin / zthermo``
        # where ``zqrho = sqrt(1.3/rho)``. Earlier drafts of this routine
        # mistakenly reused the accretion-direction ``sqrt(rho/1.3)``
        # here, which inverted the density dependence (suppressing
        # rain-evap in low-density layers and amplifying it in dense
        # layers — the opposite of physical).
        # Same double-where guard as zxrp1 above: ``x**0.61`` at x == 0 has
        # an infinite derivative, and ``zevp`` is where-masked to 0 below
        # when no rain is present — the safe base of 1.0 in that masked
        # region leaves every forward value unchanged but keeps the
        # backward pass finite.
        zrfl_in_cf_base = jnp.where(
            rain_present, jnp.maximum(zrfl_in_cf, 0.0), 1.0,
        )
        zzepr_rate = (
            870.0 * zsusatw * jnp.power(zrfl_in_cf_base, 0.61)
            * zqrho_sqrt_inv / jnp.sqrt(1.3) / zthermo
        )
        zevp_unbounded = -zzepr_rate * dt * zclcpre
        zevp_max_rain = zrfl / jnp.maximum(mref, config.epsilon) * dt
        zevp_max_subsat = jnp.maximum(0.99 * (qsw - q1), 0.0)
        zevp = jnp.minimum(zevp_unbounded, zevp_max_subsat)
        zevp = jnp.maximum(zevp, 0.0)
        zevp = jnp.minimum(zevp, zevp_max_rain)
        zevp = jnp.where(rain_present, zevp, 0.0)
        dq_evap = zevp                                                # kg/kg over dt
        dTdt_evap = -zlvdcp * (dq_evap / dt)                          # K/s
        rain_evap_flux = zevp * mref / dt                             # kg/m²/s

        # ---------- (6) flux update ----------
        # ECHAM ``mo_cloud.f90:879`` rain source:
        #   zrpr = cf · (zraut + zrac2)  +  zclcstar · zrac1
        # and analogously for snow (line 950). Rain produced in-cloud
        # (autoconv, local-rain accretion) covers the full cloud area;
        # rain accretion of cloud water by *falling* rain covers only
        # the intersection of cloud area and the incoming precipitating
        # area (zclcstar). All the in-cloud zXXX values are per-dt
        # depletion amounts in kg/kg, so divide by dt to get rate, then
        # multiply by mref to get the kg/m²/s flux into ``zrfl``.
        rain_source = (
            cf * (zraut + zrac2) + zclcstar * zrac1
        ) * mref / dt
        snow_source = (
            cf * zsaut + zclcstar * (zsaci + zsacl)
        ) * mref / dt
        # Clamp to ≥ 0 against float round-off when rain evap consumes
        # essentially all of the incoming flux.
        zrfl_out = jnp.maximum(zrfl + rain_source - rain_evap_flux, 0.0)
        zsfl_out = jnp.maximum(zsfl + snow_source, 0.0)

        # ---------- (7) zclcpre carry update per ICON 1006-1013 ----------
        zpretot = zrfl + zsfl
        zpredel = rain_source + snow_source
        zpresum = zpretot + zpredel
        zclcpre1 = jnp.where(
            zpresum > config.epsilon,
            (cf * zpredel + zclcpre * zpretot) / jnp.maximum(zpresum, config.epsilon),
            0.0,
        )
        zclcpre1 = jnp.clip(jnp.maximum(zclcpre, zclcpre1), 0.0, 1.0)
        zclcpre_out = jnp.where(zpresum > config.epsilon, zclcpre1, 0.0)

        # Pool every contribution into per-step rates (kg/kg/s, K/s) that
        # the composable physics integrator multiplies by dt and adds to
        # the dynamics state. The single condensation pass returns
        # absolute increments over dt, so divide by dt to convert to a
        # rate.
        dTdt = (dTdt_melt + dTdt_rime + dTdt_evap + dTdt_imlt
                + dTdt_clearevap + dT_cond_a / dt)
        dqdt = (dq_evap / dt) + dq_clearevap / dt + dq_cond_a / dt
        dqcdt = dqcdt_micro + dqc_cond_a / dt + (zimlt - zxlevap) / dt
        dqidt = (dqidt_micro + dqi_cond_a / dt + dqidt_sed
                 - (zimlt + zxievap) / dt)

        # ``zraut`` is the in-cloud per-dt autoconversion depletion
        # (kg/kg over dt). Convert to a grid-mean rate (kg/kg/s) for
        # the public ``autoconv_rate`` diagnostic.
        autoconv_rate_diag = cf * zraut / dt
        # Accretion likewise (Codex on PR #604: the sweep computes zrac1
        # and, when cauloc > 0, zrac2 — reporting zero misstated an
        # active pathway). Weights follow the rain-source ledger:
        # zrpr = cf(zraut + zrac2) + zclcstar·zrac1.
        accretion_rate_diag = (cf * zrac2 + zclcstar * zrac1) / dt
        # Per-level flux profiles for downstream (COSP/CloudSat)
        # diagnostics: the rain / frozen fluxes LEAVING this layer. The
        # frozen flux adds the sedimenting cloud-ice carry ``zxiflux_out``
        # so the profile is the total falling frozen water; at the bottom
        # level ``zxiflux_out`` is 0 (the residual was already folded into
        # ``zsfl`` above), so the bottom row equals the surface snow flux
        # exactly.
        out = (
            dTdt, dqdt, dqcdt, dqidt, rain_source, snow_source,
            autoconv_rate_diag, accretion_rate_diag,
            zrfl_out, zsfl_out + zxiflux_out,
            # Per-layer rain evaporation flux (#499): the depletion the
            # flux ledger above already applied, exposed for the JAM
            # wet-scavenging re-injection budget. The 1M scheme has no
            # snow sublimation, so this is the whole stratiform
            # evaporation term.
            rain_evap_flux,
        )
        return (zrfl_out, zsfl_out, zclcpre_out, zxiflux_out), out

    is_bottom_level = jnp.arange(nlev) == (nlev - 1)
    level_inputs = (
        temperature, specific_humidity, pressure,
        cloud_water, cloud_ice, cloud_fraction,
        air_density, layer_thickness, droplet_number, pmref,
        is_bottom_level,
    )
    (zrfl_surface, zsfl_surface, _zclcpre_surface, _zxiflux_sfc), per_level_out = jax.lax.scan(
        step,
        (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
        level_inputs,
    )
    (dtedt, dqdt, dqcdt, dqidt, rain_source, snow_source, autoconv_rate,
     accretion_rate, rain_flux, snow_flux, rain_evap_flux) = per_level_out

    tendencies = MicrophysicsTendencies(
        dtedt=dtedt, dqdt=dqdt, dqcdt=dqcdt, dqidt=dqidt,
        dqrdt=jnp.zeros(nlev),  # rain/snow live in the falling flux, not state
        dqsdt=jnp.zeros(nlev),
    )
    # In-cloud qc / qi from the *input* state — preserves the public
    # diagnostic signature; the within-step post-condensation values are
    # local to the scan and not exposed.
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / jnp.maximum(cloud_fraction, config.epsilon), 0.0,
    )
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / jnp.maximum(cloud_fraction, config.epsilon), 0.0,
    )
    state = MicrophysicsState(
        rain_flux=rain_flux, snow_flux=snow_flux,
        rain_source=rain_source, snow_source=snow_source,
        rain_evap_flux=rain_evap_flux,
        qc_in_cloud=qc_in_cloud, qi_in_cloud=qi_in_cloud,
        autoconv_rate=autoconv_rate, accretion_rate=accretion_rate,
        melting_rate=jnp.zeros(nlev), freezing_rate=jnp.zeros(nlev),
        precip_rain=zrfl_surface, precip_snow=zsfl_surface,
    )
    return tendencies, state


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class Echam1MMicrophysics(PhysicsTerm):
    """ECHAM 1-moment cloud microphysics as a composable PhysicsTerm.

    Consumes the post-condensation ``cloud_fraction``, ``qc``, ``qi``
    written to the public ``"clouds"`` key by
    :class:`~jcm.physics.clouds.sundqvist.SundqvistCloudFraction` so it
    must be composed downstream of that term. Reads ``cdnc_factor`` from
    the public ``"aerosol"`` key (set by
    :class:`~jcm.physics.aerosol.Macv2SpAerosol`) to apply the Twomey
    indirect effect on droplet number — when the aerosol term is absent,
    falls back to the bare ``base_cdnc`` from the parameters.

    Reads ``pressure_full``, ``air_density``, ``layer_thickness`` from
    the moist-air diagnostics dict and the model timestep from
    ``diagnostics["_dt_seconds"]`` (injected by ``ComposablePhysics``).
    Writes ``precip_rain``, ``precip_snow``, ``droplet_number`` back
    into the public ``"clouds"`` key (preserving the upstream
    ``cloud_fraction`` / ``qc`` / ``qi`` fields).
    """

    name: ClassVar[str] = "echam_1m_microphysics"
    category: ClassVar[str] = "clouds"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "air_density", "layer_thickness",
        "clouds", "aerosol",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "autoconv", "accretn", "wbf", "clouds",
    )

    def __init__(self, params: MicrophysicsParameters | None = None):
        """Hold the scheme-native :class:`MicrophysicsParameters`."""
        self.params = nnx.Param(
            params or MicrophysicsParameters.default(),
        )

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """``qc`` / ``qi`` are read each step; declared so dynamics carries them."""
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
        )

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute microphysics tendencies + precip/droplet diagnostics."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        air_density = diagnostics["air_density"]
        layer_thickness = diagnostics["layer_thickness"]
        clouds = diagnostics["clouds"]

        # Post-(vdiff+convection) thermodynamic state (sequential
        # vdiff→convection→cloud coupling, ECHAM physc order; same pattern
        # as the 2M term / PR #539): the upstream vdiff and convection terms
        # have already advanced ``thermo_run`` with their tendencies and
        # convection forwarded its detrained condensate into ``clouds.qc/qi``. The
        # sweep's saturation balance and rain evaporation must see THAT
        # (T, q) — using the step-start state let the same supersaturation
        # be condensed by both convection and microphysics, and computed
        # evaporation against a stale qsat (review finding 2.15).
        thermo_run = diagnostics.get("thermo_run")
        if thermo_run is None:
            temperature_in = state.temperature
            specific_humidity_in = state.specific_humidity
        else:
            temperature_in = thermo_run["temperature"]
            specific_humidity_in = thermo_run["specific_humidity"]

        qc_interim = clouds.qc
        qi_interim = clouds.qi
        cloud_fraction = clouds.cloud_fraction

        # Twomey effect: aerosol term provides per-column cdnc_factor
        # (validated as a required upstream key at composition time).
        cdnc_factor = diagnostics["aerosol"].cdnc_factor
        cdnc_m3 = (
            jnp.ones_like(state.temperature)
            * params.base_cdnc
            * cdnc_factor[jnp.newaxis, :]
        )
        droplet_number_per_kg = cdnc_m3 / air_density

        # ECHAM ``mo_cloud.f90`` column-sweep: per-layer saturation
        # adjustment + autoconversion / accretion / riming / rain-evap +
        # rain/snow flux propagation, all top-to-bottom in one ``lax.scan``.
        # The condensation step lives inside the sweep (see
        # :func:`_saturation_adjustment_layer`) so the rain-evap that
        # follows can't push the layer past saturation, and the cleanup
        # pass at the end of each layer's step closes any residual
        # supersat within the same ``dt`` — breaking the rain-evap ↔
        # re-condensation feedback that drove PR #458 to revert to the
        # per-level scheme. With this in place, Sundqvist is a pure
        # cloud-fraction diagnostic upstream — see
        # :class:`~jcm.physics.clouds.sundqvist.SundqvistCloudFraction`.
        micro_tend, micro_state = jax.vmap(
            cloud_microphysics_column_sweep,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),
            out_axes=(0, 0),
        )(
            temperature_in, specific_humidity_in, pressure_full,
            qc_interim, qi_interim, cloud_fraction,
            air_density, layer_thickness,
            droplet_number_per_kg, dt, params,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=micro_tend.dtedt.T,
            specific_humidity=micro_tend.dqdt.T,
            tracers={
                "qc": micro_tend.dqcdt.T,
                "qi": micro_tend.dqidt.T,
            },
        )

        # AeroCom process rates (jax-gcm#585 acceptance: both schemes,
        # zero where a pathway is absent). Autoconversion and accretion
        # come from the sweep's own per-level rates (grid-mean kg/kg/s,
        # dp-weighted to kg/m^2/s); WBF stays zero — the 1M scheme has
        # no explicit Wegener-Bergeron-Findeisen transfer — and the key
        # is still published so the diagnostic set is scheme-independent.
        # air_density/layer_thickness are (nlev, ncols); the vmapped
        # micro_state fields are (ncols, nlev) — transpose the mass weight.
        dm_col = (air_density * layer_thickness).T
        autoconv_col = jnp.sum(micro_state.autoconv_rate * dm_col, axis=-1)
        accretn_col = jnp.sum(micro_state.accretion_rate * dm_col, axis=-1)
        zero_col = jnp.zeros_like(autoconv_col)
        diagnostics = {**diagnostics, "autoconv": autoconv_col,
                       "accretn": accretn_col, "wbf": zero_col}

        # Post-microphysics cloud-cover write-back (ECHAM mo_cloud.f90:1280
        # — ``paclc = FSEL(-(zxlp1_d*zxip1_d), paclc, 0)``): a cell whose
        # end-of-step condensate falls below ``ccwmin`` in BOTH phases is
        # no longer cloudy. This makes ``clouds.cloud_fraction`` mean the
        # same thing under cloud_scheme='1m' and '2m' (#687): the cover
        # the step actually leaves behind, which radiation, COSP, AeroCom
        # and the JAM cloud-borne/aqueous/wetdep terms all read. The
        # end-of-step condensate is interim + the scheme's own tendency
        # (upstream increments are already inside the interim values).
        qc_end = qc_interim + dt * micro_tend.dqcdt.T
        qi_end = qi_interim + dt * micro_tend.dqidt.T
        cloud_fraction_out = jnp.where(
            (qc_end < params.ccwmin) & (qi_end < params.ccwmin),
            0.0, cloud_fraction,
        )

        clouds = clouds.copy(
            cloud_fraction=cloud_fraction_out,
            precip_rain=micro_state.precip_rain,
            precip_snow=micro_state.precip_snow,
            # Per-level precipitation flux profiles for satellite-simulator
            # diagnostics (COSP/CloudSat). The vmap over columns puts the
            # column axis first — transpose back to the (nlev, ncols)
            # CloudData layout, same as the tendency fields above.
            rain_flux=micro_state.rain_flux.T,
            snow_flux=micro_state.snow_flux.T,
            # Per-level process rates for JAM wet scavenging (#499),
            # converted from the sweep's per-layer fluxes [kg/m²/s] to
            # grid-mean mixing-ratio rates [kg/kg/s] with the layer mass.
            # Formation is the full condensate→precip ledger (rain: autoconv
            # + accretion; snow: ice autoconv + aggregation + riming);
            # evaporation is Rotstayn rain evap (the 1M scheme has no snow
            # sublimation).
            precip_formation_rate=(
                micro_state.rain_source + micro_state.snow_source
            ).T / dm_col.T,
            precip_evaporation_rate=micro_state.rain_evap_flux.T / dm_col.T,
            droplet_number=cdnc_m3,
        )

        # Advance the running condensate view so terms downstream (the
        # satellite simulators and the AeroCom diagnostics) describe the
        # POST-microphysics atmosphere, matching the tracers saved at the
        # same timestamp. ``thermo_run`` is a parallel diagnostic view,
        # never the prognostic state, so this cannot alter the trajectory
        # (see ``advance_thermo_run``).
        from jcm.physics.diagnostics.moist_air_state import (
            advance_thermo_run)
        diagnostics = advance_thermo_run(
            diagnostics, dt,
            d_temperature=tendency.temperature,
            d_specific_humidity=tendency.specific_humidity,
            d_qc=tendency.tracers.get("qc"), d_qi=tendency.tracers.get("qi"))

        return tendency, {**diagnostics, "clouds": clouds}
