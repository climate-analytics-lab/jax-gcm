"""Data structures for the Tiedtke-Nordeng convection scheme.

Pure data containers (parameters, per-column state, tendencies and
diagnostics) split out of ``tiedtke_nordeng.py`` so that the scheme
submodules (``updraft``, ``downdraft``, ``flux_tendencies``) can import
them without pulling in the full orchestrator. This module must not
import any other module from this package (no cycles).
"""

import jax.numpy as jnp
from typing import NamedTuple
import tree_math


@tree_math.struct
class ConvectionParameters:
    """Configuration parameters for Tiedtke-Nordeng convection scheme"""
    
    # Time stepping
    dt_conv: float           # Convection timestep (s)
    
    # Entrainment/detrainment parameters
    entrpen: float           # Entrainment rate for penetrative convection (m⁻¹)
    entrscv: float           # Entrainment rate for shallow convection (m⁻¹) 
    entrmid: float           # Entrainment rate for mid-level convection (m⁻¹)
    
    # CAPE closure
    tau: float               # CAPE adjustment timescale (s)
    
    # Cloud base mass flux
    cmfcmax: float           # Maximum cloud base mass flux (kg/m²/s)
    cmfcmin: float           # Minimum cloud base mass flux (kg/m²/s)
    
    # Precipitation parameters
    cprcon: float            # Precip conversion coefficient [s²/m²] — ECHAM
                             # mo_echam_conv_constants.f90:127: 2.5e-4 at
                             # T31/T63, 1.5e-4 at T127+. Enters the ascent as
                             # ``plu/(1 + cprcon·Δgeopotential)``.
    cu_dnoprc_ocean: float   # Pressure thickness above cloud base before
                             # precip generation starts, over ocean (Pa)
                             # — ECHAM ``zdnoprc`` ocean default 1.5e4
    cu_dnoprc_land: float    # Same threshold over land (Pa) — ECHAM
                             # ``zdnoprc`` land default 3.0e4 (continental
                             # convection has thicker non-precipitating
                             # cloud-base layer)

    # Evaporation parameters
    cevapcu: float           # Coefficient for rain evaporation
    
    # Numerical parameters
    epsilon: float           # Small number for numerical stability
    
    # Convection type thresholds
    rlcrit: float            # Critical relative humidity for shallow convection
    rhcrit: float            # Critical relative humidity threshold
    
    # Momentum transport
    cmfctop: float           # Mass flux fraction at cloud top

    # Downdraft parameters
    cmfdeps: float           # Downdraft mass flux fraction for LFS threshold
    entrdd: float            # Downdraft fractional entrainment rate (m⁻¹)

    # Smooth-trigger parameters (maintainability review Part B). The
    # hard CAPE/type/termination gates gave every convection parameter an
    # exactly-zero gradient over most of state space; each gate is now a
    # sigmoid whose WIDTH is itself a differentiable, annealable
    # parameter — width → 0 recovers the hard behaviour exactly.
    trigger_cape: float      # CAPE activation threshold (J/kg; ex-hardcoded 100)
    smooth_trigger_j: float  # Sigmoid width of the CAPE trigger (J/kg)
    smooth_type_j: float     # Width of the deep/other blend at 1000 J/kg (J/kg)
    smooth_rh: float         # Width of the moist-free-troposphere RH gate
    smooth_term_buoy: float  # Updraft-termination buoyancy width (m/s²; ~3e-4 ≈ 0.01 K)
    smooth_term_mf: float    # Updraft-termination mass-flux-ratio width
    smooth_precip_pa: float  # zdnoprc precip-onset width (Pa)

    # Switches (ECHAM namelist lmfdudv; carried as a traced bool so the
    # struct stays a plain tree_math pytree)
    lmfdudv: jnp.ndarray

    @classmethod
    def default(cls, dt_conv=3600.0, entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4, # FIXME: validate dt_conv
                 tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10, cprcon=2.5e-4,
                 cu_dnoprc_ocean=1.5e4, cu_dnoprc_land=3.0e4,
                 cevapcu=2.0e-5, epsilon=1.0e-12, rlcrit=8.0e-4, rhcrit=0.9,
                 cmfctop=0.2, cmfdeps=0.3, entrdd=2.0e-4,
                 trigger_cape=100.0, smooth_trigger_j=25.0,
                 smooth_type_j=100.0, smooth_rh=0.02,
                 smooth_term_buoy=3.0e-4, smooth_term_mf=2.0e-3,
                 smooth_precip_pa=2.0e3,
                 lmfdudv=True) -> 'ConvectionParameters':
        """Return default convection parameters"""
        return cls(
            dt_conv=jnp.array(dt_conv),
            entrpen=jnp.array(entrpen),
            entrscv=jnp.array(entrscv),
            entrmid=jnp.array(entrmid),
            tau=jnp.array(tau),
            cmfcmax=jnp.array(cmfcmax),
            cmfcmin=jnp.array(cmfcmin),
            cprcon=jnp.array(cprcon),
            cu_dnoprc_ocean=jnp.array(cu_dnoprc_ocean),
            cu_dnoprc_land=jnp.array(cu_dnoprc_land),
            cevapcu=jnp.array(cevapcu),
            epsilon=jnp.array(epsilon),
            rlcrit=jnp.array(rlcrit),
            rhcrit=jnp.array(rhcrit),
            cmfctop=jnp.array(cmfctop),
            cmfdeps=jnp.array(cmfdeps),
            entrdd=jnp.array(entrdd),
            trigger_cape=jnp.array(trigger_cape),
            smooth_trigger_j=jnp.array(smooth_trigger_j),
            smooth_type_j=jnp.array(smooth_type_j),
            smooth_rh=jnp.array(smooth_rh),
            smooth_term_buoy=jnp.array(smooth_term_buoy),
            smooth_term_mf=jnp.array(smooth_term_mf),
            smooth_precip_pa=jnp.array(smooth_precip_pa),
            lmfdudv=jnp.array(lmfdudv),
        )


class ConvectionState(NamedTuple):
    """State variables for convection scheme"""
    
    # Updraft properties
    tu: jnp.ndarray          # Updraft temperature (K)
    qu: jnp.ndarray          # Updraft specific humidity (kg/kg)  
    lu: jnp.ndarray          # Updraft liquid water content (kg/kg)
    uu: jnp.ndarray          # Updraft zonal wind (m/s)
    vu: jnp.ndarray          # Updraft meridional wind (m/s)
    
    # Downdraft properties  
    td: jnp.ndarray          # Downdraft temperature (K)
    qd: jnp.ndarray          # Downdraft specific humidity (kg/kg)
    ud: jnp.ndarray          # Downdraft zonal wind (m/s)
    vd: jnp.ndarray          # Downdraft meridional wind (m/s)
    
    # Mass fluxes
    mfu: jnp.ndarray         # Updraft mass flux (kg/m²/s)
    mfd: jnp.ndarray         # Downdraft mass flux (kg/m²/s)
    
    # Convection diagnostics
    ktype: jnp.ndarray       # Convection type (0=none, 1=deep, 2=shallow, 3=mid)
    kbase: jnp.ndarray       # Cloud base level index
    ktop: jnp.ndarray        # Cloud top level index
    
    # Precipitation
    prate: jnp.ndarray       # Precipitation rate (kg/m²/s)


class ConvectionTendencies(NamedTuple):
    """Tendencies from convection scheme"""
    
    dtedt: jnp.ndarray       # Temperature tendency (K/s)
    dqdt: jnp.ndarray        # Specific humidity tendency (kg/kg/s)
    dudt: jnp.ndarray        # Zonal wind tendency (m/s²)
    dvdt: jnp.ndarray        # Meridional wind tendency (m/s²)
    
    # Convective fluxes
    qc_conv: jnp.ndarray     # Convective cloud water (kg/kg)
    qi_conv: jnp.ndarray     # Convective cloud ice (kg/kg)
    
    # Surface fluxes
    precip_conv: jnp.ndarray # Convective precipitation (kg/m²/s)
    
    # Fixed tracer tendencies (qc, qi only)
    dqc_dt: jnp.ndarray      # Cloud water tendency (kg/kg/s)
    dqi_dt: jnp.ndarray      # Cloud ice tendency (kg/kg/s)


@tree_math.struct
class ConvectionData:
    """Diagnostic outputs from the Tiedtke-Nordeng convection scheme.

    Stored in the diagnostics dict under the ``"convection"`` key (no
    leading underscore — flows to user-facing xarray output as
    ``convection.<field>``). The ``mass_flux_*`` / ``cloud_base`` /
    ``cloud_top`` / ``cape`` fields are reserved for the future port of
    the equivalent ECHAM diagnostics; they are zero-filled today.
    """

    mass_flux_up: jnp.ndarray        # Updraft mass flux [kg/m²/s] (nlev, ncols)
    mass_flux_down: jnp.ndarray      # Downdraft mass flux [kg/m²/s] (nlev, ncols)
    cloud_base: jnp.ndarray          # Cloud base level index (ncols,)
    cloud_top: jnp.ndarray           # Cloud top level index (ncols,)
    cape: jnp.ndarray                # CAPE [J/kg] (ncols,)
    ktype: jnp.ndarray               # Convection type per column (0=off,
                                     # 1=deep, 2=shallow, 3=mid) — consumed
                                     # by the Sundqvist stratocumulus guard
                                     # (ECHAM gates on ktype==0) (ncols,)
    precip_conv: jnp.ndarray         # Convective precipitation [kg/m²/s] (ncols,)
    qc_conv: jnp.ndarray             # Convective cloud water [kg/kg] (nlev, ncols)
    qi_conv: jnp.ndarray             # Convective cloud ice [kg/kg] (nlev, ncols)
    # Convective heating / moistening rates actually applied to the column
    # (post-cap; see the ``_DTDT_MAX`` limiter in ``TiedtkeConvection``). These
    # are the genuine per-level convective tendencies — the thing that balances
    # radiative cooling in an RCE column — exposed as first-class diagnostics so
    # they ride the saved trajectory exactly like ``RadiationData``'s
    # ``sw_heating_rate`` / ``lw_heating_rate``, rather than being recoverable
    # only by re-running the term. ``heating_rate`` mirrors the radiation naming
    # convention ([K/s]); ``moistening_rate`` is its specific-humidity analog.
    heating_rate: jnp.ndarray        # Convective heating rate [K/s] (nlev, ncols)
    moistening_rate: jnp.ndarray     # Convective moistening rate [kg/kg/s] (nlev, ncols)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        """Construct a zero-filled ``ConvectionData`` for the given grid."""
        return cls(
            mass_flux_up=jnp.zeros((nlev,) + nodal_shape),
            mass_flux_down=jnp.zeros((nlev,) + nodal_shape),
            cloud_base=jnp.zeros(nodal_shape, dtype=int),
            cloud_top=jnp.zeros(nodal_shape, dtype=int),
            cape=jnp.zeros(nodal_shape),
            ktype=jnp.zeros(nodal_shape, dtype=jnp.int32),
            precip_conv=jnp.zeros(nodal_shape),
            qc_conv=jnp.zeros((nlev,) + nodal_shape),
            qi_conv=jnp.zeros((nlev,) + nodal_shape),
            heating_rate=jnp.zeros((nlev,) + nodal_shape),
            moistening_rate=jnp.zeros((nlev,) + nodal_shape),
        )
