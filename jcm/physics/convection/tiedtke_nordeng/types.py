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
    cu_dqcv_width: float     # Width [kg/m2/s] of the deep/shallow moisture-
                             # convergence sigmoid. ECHAM's test is a hard
                             # switch ``zdqcv > MAX(0, -1.1*pqhfla*g)``
                             # (mo_cumastr.f90:571); the default keeps this
                             # hard at atmospheric flux scales (~1% of a
                             # typical tropical E) while staying
                             # differentiable. Replaced the non-ECHAM CAPE
                             # sigmoid at 1000 J/kg (#699).
    smooth_rh: float         # Width of the moist-free-troposphere RH gate
    smooth_term_buoy: float  # Updraft-termination buoyancy width (m/s²; ~3e-4 ≈ 0.01 K)
    smooth_term_mf: float    # Updraft-termination mass-flux-ratio width
    smooth_precip_pa: float  # zdnoprc precip-onset width (Pa)

    # Cloud-base sub-grid buoyancy excess — ECHAM ``cubase``
    # (mo_cuinitialize.f90:291) ``zlift = MAX(cminbuoy, MIN(cmaxbuoy,
    # pthvsig*cbfac))``, then ``MIN(zlift, 1.0)``. This is the thermal
    # excess of the warmest boundary-layer plumes over the grid mean; it is
    # what lets a parcel cross the thin negative-buoyancy layer between its
    # LCL and its LFC. Without it a grid-mean parcel is essentially never
    # buoyant at its own LCL and no column convects.
    cu_cminbuoy: float       # Floor on the excess (K) — ECHAM 0.2
    cu_cmaxbuoy: float       # Ceiling on the excess (K) — ECHAM 1.0
    cu_cbfac: float          # Multiplier on thvsig (-) — ECHAM 1.0
    cu_thvsig: float         # FALLBACK sub-grid σ(θ_v) at the lowest half
                             # level (K), used only when the caller supplies
                             # no vdiff diagnostic (column-mode tests,
                             # standalone drivers). The model path takes
                             # ``thv_sigma`` from vdiff's prognostic θ_v
                             # variance — see ``cloud_base_lift``.

    # Mid-level convection trigger — ECHAM ``cubasmc`` (mo_cuascent.f90:593).
    # The SECOND way ECHAM starts a plume: at a level with no surface
    # connection at all, when the environment there is nearly saturated and
    # resolved-scale ascent is lifting it. This is what covers elevated
    # convection above a stable layer, warm-conveyor and frontal ascent, and
    # nocturnal elevated convection over land — everything the deliberately
    # strict surface-parcel ``cubase`` walk is not meant to catch.
    cu_midlev_rh: float      # Environmental RH the candidate level must
                             # exceed (-) — ECHAM's hard-coded 0.90
    cu_midlev_zmin: float    # Minimum height of the candidate layer's TOP
                             # interface above the surface (m) — ECHAM's
                             # ``pgeoh(kk)/grav > 1500``, which keeps the
                             # trigger out of the boundary layer that
                             # ``cubase`` already owns
    cu_midlev_ptop: float    # Pressure floor for the base (Pa) — ECHAM's
                             # ``nmctop``, the 300 hPa level, above which a
                             # mid-level base is not allowed

    # Switches (ECHAM namelist lmfdudv / lmfmid; carried as traced bools so
    # the struct stays a plain tree_math pytree)
    lmfdudv: jnp.ndarray
    cu_lmfmid: jnp.ndarray   # ECHAM ``lmfmid`` (setphys.f90:71, default
                             # .TRUE.): enable the mid-level trigger. Turning
                             # it off is the escape hatch for a dycore that
                             # cannot supply omega — see ``TiedtkeConvection``.

    @classmethod
    def default(cls, entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4,
                 tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10, cprcon=2.5e-4,
                 cu_dnoprc_ocean=1.5e4, cu_dnoprc_land=3.0e4,
                 cevapcu=2.0e-5, cmfdeps=0.3, entrdd=2.0e-4,
                 trigger_cape=100.0, smooth_trigger_j=25.0,
                 cu_dqcv_width=2.0e-7, smooth_rh=0.02,
                 smooth_term_buoy=3.0e-4, smooth_term_mf=2.0e-3,
                 smooth_precip_pa=2.0e3,
                 cu_cminbuoy=0.2, cu_cmaxbuoy=1.0, cu_cbfac=1.0,
                 cu_thvsig=1.0,
                 cu_midlev_rh=0.90, cu_midlev_zmin=1500.0,
                 cu_midlev_ptop=30_000.0,
                 lmfdudv=True, cu_lmfmid=True) -> 'ConvectionParameters':
        """Return default convection parameters"""
        return cls(
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
            cmfdeps=jnp.array(cmfdeps),
            entrdd=jnp.array(entrdd),
            trigger_cape=jnp.array(trigger_cape),
            smooth_trigger_j=jnp.array(smooth_trigger_j),
            cu_dqcv_width=jnp.array(cu_dqcv_width),
            smooth_rh=jnp.array(smooth_rh),
            smooth_term_buoy=jnp.array(smooth_term_buoy),
            smooth_term_mf=jnp.array(smooth_term_mf),
            smooth_precip_pa=jnp.array(smooth_precip_pa),
            cu_cminbuoy=jnp.array(cu_cminbuoy),
            cu_cmaxbuoy=jnp.array(cu_cmaxbuoy),
            cu_cbfac=jnp.array(cu_cbfac),
            cu_thvsig=jnp.array(cu_thvsig),
            cu_midlev_rh=jnp.array(cu_midlev_rh),
            cu_midlev_zmin=jnp.array(cu_midlev_zmin),
            cu_midlev_ptop=jnp.array(cu_midlev_ptop),
            lmfdudv=jnp.array(lmfdudv),
            cu_lmfmid=jnp.array(cu_lmfmid),
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
    entr: jnp.ndarray        # Fractional updraft entrainment rate (1/m)

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
    precip_formation: jnp.ndarray  # Per-layer updraft precip generation
                                   # (ECHAM ``pdmfup``) [kg/m²/s] (nlev,)
    
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
    ``convection.<field>``). The ``cloud_base`` / ``cloud_top`` / ``cape``
    fields are reserved for the future port of the equivalent ECHAM
    diagnostics; they are zero-filled today. ``mass_flux_up``/``down`` and
    ``entrain_up``/``entrain_down`` are populated (post-rescale, post-cap —
    the same ledger scaling as the tendencies) for the convective tracer
    transport (#602, #622): the updraft flux at each layer's TOP
    interface, the downdraft flux at each layer's BOTTOM interface (the
    downdraft scan's convention), and the absolute per-layer entrainment
    fluxes; per-layer detrainment follows from plume continuity, so it is
    not stored separately.
    """

    mass_flux_up: jnp.ndarray        # Updraft mass flux [kg/m²/s] (nlev, ncols)
    mass_flux_down: jnp.ndarray      # Downdraft mass flux [kg/m²/s] (nlev, ncols)
    entrain_up: jnp.ndarray          # Updraft entrainment flux per layer
                                     # [kg/m²/s] (nlev, ncols)
    entrain_down: jnp.ndarray        # Downdraft entrainment flux per layer
                                     # [kg/m²/s] (nlev, ncols)
    cloud_base: jnp.ndarray          # Cloud base level index (ncols,)
    cloud_top: jnp.ndarray           # Cloud top level index (ncols,)
    cape: jnp.ndarray                # CAPE [J/kg] (ncols,)
    ktype: jnp.ndarray               # Convection type per column (0=off,
                                     # 1=deep, 2=shallow, 3=mid) — consumed
                                     # by the Sundqvist stratocumulus guard
                                     # (ECHAM gates on ktype==0) (ncols,)
    precip_conv: jnp.ndarray         # Convective precipitation [kg/m²/s] (ncols,)
    qc_conv: jnp.ndarray             # Convective cloud water [kg/kg] (nlev, ncols)
    precip_formation: jnp.ndarray    # Per-layer updraft precip generation
                                     # [kg/m²/s] (nlev, ncols)
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
            entrain_up=jnp.zeros((nlev,) + nodal_shape),
            entrain_down=jnp.zeros((nlev,) + nodal_shape),
            cloud_base=jnp.zeros(nodal_shape, dtype=int),
            cloud_top=jnp.zeros(nodal_shape, dtype=int),
            cape=jnp.zeros(nodal_shape),
            ktype=jnp.zeros(nodal_shape, dtype=jnp.int32),
            precip_conv=jnp.zeros(nodal_shape),
            precip_formation=jnp.zeros((nlev,) + nodal_shape),
            qc_conv=jnp.zeros((nlev,) + nodal_shape),
            qi_conv=jnp.zeros((nlev,) + nodal_shape),
            heating_rate=jnp.zeros((nlev,) + nodal_shape),
            moistening_rate=jnp.zeros((nlev,) + nodal_shape),
        )


#: CF/units metadata for the :class:`ConvectionData` fields as they appear in
#: the output Dataset — flattened to ``convection.<field>`` keys (#740). Set as
#: ``output_attrs`` on :class:`TiedtkeConvection`. Units are taken verbatim from
#: the field comments above; CF standard names are used only where exact.
CONVECTION_OUTPUT_ATTRS: dict[str, dict[str, str]] = {
    "convection.mass_flux_up": {
        "standard_name": "atmosphere_updraft_convective_mass_flux",
        "units": "kg m-2 s-1", "long_name": "updraft convective mass flux"},
    "convection.mass_flux_down": {
        "standard_name": "atmosphere_downdraft_convective_mass_flux",
        "units": "kg m-2 s-1", "long_name": "downdraft convective mass flux"},
    "convection.entrain_up": {
        "units": "kg m-2 s-1",
        "long_name": "updraft entrainment flux per layer"},
    "convection.entrain_down": {
        "units": "kg m-2 s-1",
        "long_name": "downdraft entrainment flux per layer"},
    "convection.cloud_base": {
        "units": "1", "long_name": "convective cloud base level index"},
    "convection.cloud_top": {
        "units": "1", "long_name": "convective cloud top level index"},
    "convection.cape": {
        "units": "J kg-1",
        "long_name": "convective available potential energy"},
    "convection.ktype": {
        "units": "1",
        "long_name": "convection type (0=off, 1=deep, 2=shallow, 3=mid)"},
    "convection.precip_conv": {
        "standard_name": "convective_precipitation_flux",
        "units": "kg m-2 s-1", "long_name": "convective precipitation flux"},
    "convection.qc_conv": {
        "units": "kg kg-1", "long_name": "convective cloud water mixing ratio"},
    "convection.precip_formation": {
        "units": "kg m-2 s-1",
        "long_name": "per-layer updraft precipitation generation"},
    "convection.qi_conv": {
        "units": "kg kg-1", "long_name": "convective cloud ice mixing ratio"},
    "convection.heating_rate": {
        "standard_name": "tendency_of_air_temperature_due_to_convection",
        "units": "K s-1", "long_name": "convective heating rate"},
    "convection.moistening_rate": {
        "standard_name": "tendency_of_specific_humidity_due_to_convection",
        "units": "kg kg-1 s-1", "long_name": "convective moistening rate"},
}
