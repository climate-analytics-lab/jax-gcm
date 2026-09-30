"""Cloud diagnostic sub-struct shared across cloud schemes.

The diagnostic dict key ``"clouds"`` carries a :class:`CloudData` typed
sub-struct that is read and written by the cloud-fraction scheme
(``SundqvistCloudFraction``) and the microphysics schemes
(``Echam1MMicrophysics``, ``Lohmann2MMicrophysics``). It lives next to
those schemes so a downstream user adding a new cloud scheme can extend
or replace it without reaching into the ECHAM tree.
"""

from __future__ import annotations

import jax.numpy as jnp
import tree_math


@tree_math.struct
class CloudData:
    """Cloud-fraction, condensate, and surface-precip diagnostics."""

    # Cloud fraction. SEMANTICS (#687): after the microphysics term has
    # run, this is the POST-microphysics cover — cells the scheme emptied
    # of both condensates (end-of-step qc AND qi below ccwmin) have
    # cloud_fraction = 0, under BOTH the 1M and 2M schemes (ECHAM's
    # ``paclc`` write-back). Between SundqvistCloudFraction and the
    # microphysics term it is the RH-diagnosed cover. The microphysics
    # may only ever REMOVE cover relative to the diagnosed value, never
    # add it. Consumers (radiation via the carry, COSP, AeroCom, the JAM
    # cloud-borne/aqueous/wetdep terms) therefore always see the cloud
    # the step actually left behind.
    cloud_fraction: jnp.ndarray      # Cloud fraction [1] (nlev, ncols)

    # Cloud condensate (updated by condensation within the cloud scheme)
    qc: jnp.ndarray                  # Cloud water [kg/kg] (nlev, ncols)
    qi: jnp.ndarray                  # Cloud ice [kg/kg] (nlev, ncols)

    # Convective detrainment of cloud condensate [kg/kg/s], grid-mean,
    # (nlev, ncols): ECHAM's ``pxtecl`` (liquid) / ``pxteci`` (ice). They
    # are the part of this step's ``clouds.qc`` / ``clouds.qi`` increment
    # that the convection term ADDED, exactly as applied (after its
    # tendency cap), written by ``TiedtkeConvection`` every step and zero
    # otherwise: ``SundqvistCloudFraction`` resets them when it seeds the
    # step's ``clouds`` from the carry, so a step without convection — or
    # a restart into a stack without it — can never re-apply a previous
    # step's detrainment.
    #
    # The split is Tiedtke's own: all liquid where the environment
    # temperature it received exceeds tmelt, all ice otherwise (``cudtdq``,
    # mo_cufluxdts.f90:646-666, keyed to ``pten``). Both cloud schemes read
    # these fields through ``cloud_inputs.cloud_scheme_inputs``, which takes
    # them out of the running condensate tendency and passes them on by
    # themselves, as ECHAM passes ``pxtecl``/``pxteci`` (physc.f90:1081).
    # The 1M scheme uses the split as-is: ECHAM's mo_cloud adds
    # ``pxtecl``/``pxteci`` to its condensate increments (mo_cloud.f90:
    # 666-680). ECHAM's 2M ignores it: it re-splits the TOTAL detrained
    # condensate by its own WBF criterion (mo_cloud_micro_2m.f90:1300-1316),
    # sediments only the pre-detrainment ice, gives the detrained ice a
    # crystal number, and undoes the ice latent heat convection booked for
    # condensate it reclassifies as liquid, all of which need the detrained
    # part distinguished from the rest of the increment.
    conv_detrainment_qc: jnp.ndarray  # Detrained cloud liquid [kg/kg/s] (nlev, ncols)
    conv_detrainment_qi: jnp.ndarray  # Detrained cloud ice    [kg/kg/s] (nlev, ncols)

    # Surface precipitation (from microphysics autoconversion)
    precip_rain: jnp.ndarray         # Rain precipitation [kg/m²/s] (ncols,)
    precip_snow: jnp.ndarray         # Snow precipitation [kg/m²/s] (ncols,)

    # Large-scale precipitation flux profiles for satellite-simulator
    # diagnostics (COSP/CloudSat): the grid-mean flux leaving each layer
    # (i.e. crossing its lower boundary) as the microphysics column sweep
    # propagates precipitation downward within the step. Level index 0 is
    # the physics-internal model top, so the bottom level equals the
    # ``precip_rain`` / ``precip_snow`` surface diagnostics by
    # construction. ``snow_flux`` is the total frozen flux: snow plus the
    # sedimenting-ice flux that the schemes fold into surface snow at the
    # bottom level.
    rain_flux: jnp.ndarray           # LS rain flux [kg/m²/s] (nlev, ncols)
    snow_flux: jnp.ndarray           # LS snow(+ice) flux [kg/m²/s] (nlev, ncols)
    # Per-level precipitation process rates (#499), grid-mean [kg/kg/s]:
    # the condensate→precip conversion (rain: autoconversion + accretion;
    # frozen: ice autoconversion + aggregation + riming) and the
    # evaporation/sublimation of the falling stratiform precip (rain
    # evaporation + snow sublimation; the sedimenting cloud-ice flux's own
    # sublimation is not included — it is not a scavenging carrier, which
    # also means the cirrus ice that reaches the surface as snow removes no
    # aerosol). These
    # are the rates the schemes actually integrate, exposed for JAM wet
    # scavenging (in-cloud nucleation removal at the true local formation
    # rate; re-injection of scavenged aerosol where precip re-evaporates)
    # instead of a column reconstruction from the surface fluxes.
    precip_formation_rate: jnp.ndarray    # (nlev, ncols)
    precip_evaporation_rate: jnp.ndarray  # (nlev, ncols)
    # Column-integrated rain-source split [kg/m²/s], written by the 2M
    # scheme (zero under 1M, which does not separate the pathways): rain
    # formed by the warm chain (autoconversion + accretion of liquid) and
    # rain formed by melting snow. warm / (warm + melt) is the model's
    # warm-rain fraction, the CloudSat-style observable used to constrain
    # the warm-rain and aerosol-activation parameters.
    rain_formation_warm: jnp.ndarray  # (ncols,)
    rain_from_melt: jnp.ndarray       # (ncols,)

    # Column-integrated latent heating of the 2M negative-mass repair
    # [W/m²] (#689): the ECHAM zdxlcor/zdxicor guard returns condensate
    # below ccwmin (including dycore-ringing negatives) to vapour with
    # the matching latent heat. Thermodynamically consistent but
    # sign-definite — undershoots always become warming + drying — so it
    # is exported for monitoring rather than folded silently into the
    # tendencies. Zero under the 1M scheme.
    negative_mass_repair: jnp.ndarray  # (ncols,)

    # The ECHAM-HAM wet-scavenging interface (#708): the process-time
    # ledger ``cloud_subm_2`` receives, written by the 2M scheme. The 1M
    # scheme leaves these zero, which is why the factory rejects
    # aerosol_module='jam' with cloud_scheme='1m' — the JAM terms that
    # read them would silently scavenge nothing.
    # ``incloud_liquid``/``incloud_ice`` are ECHAM's
    # zmlwc/zmiwc — IN-CLOUD condensate captured before precipitation
    # formation, zeroed (faithfully) where the post-write-back cover fell
    # below clc_min; a zeroed pool with a positive formation rate marks a
    # cell fully converted to precipitation, which consumers must read as
    # scavenged-fraction 1 (see ScavengingLedger in lohmann_2m/types.py).
    # The formation rates are IN-CLOUD [kg/kg/s]: rain formation
    # (zmratepr), snow formation (zmrateps, including the ice-
    # sedimentation carrier), and riming of droplets by snow (zmsnowacl,
    # a LIQUID sink into frozen precip). Snow formation can be NEGATIVE:
    # its sedimentation seed is signed, as ECHAM's, and negative where the
    # level absorbs more falling ice than it sheds; consumers floor it
    # apart from the liquid rates, as HAM does. ``process_cloud_fraction`` is
    # the cover the processes ran under (nonzero in cells the write-back
    # cleared). ``condensate_evaporation_rate`` is the grid-mean
    # cloud-condensate evaporation ledger (zxlevap+zxievap) — the
    # resuspension key: evaporated droplets release their aerosol,
    # rained-out ones do not.
    incloud_liquid: jnp.ndarray             # (nlev, ncols) [kg/kg]
    incloud_ice: jnp.ndarray                # (nlev, ncols) [kg/kg]
    incloud_rain_formation: jnp.ndarray     # (nlev, ncols) [kg/kg/s]
    incloud_snow_formation: jnp.ndarray     # (nlev, ncols) [kg/kg/s]
    incloud_riming: jnp.ndarray             # (nlev, ncols) [kg/kg/s]
    process_cloud_fraction: jnp.ndarray     # (nlev, ncols) [1]
    condensate_evaporation_rate: jnp.ndarray  # (nlev, ncols) [kg/kg/s]

    # Cloud properties
    droplet_number: jnp.ndarray  # Droplet number concentration [1/m³] (nlev, ncols)

    # Droplet / crystal effective radii [um] (nlev, ncols) the radiation used
    # (ECHAM ``re_droplets2d``/``re_crystals2d``): a DIAGNOSTIC written by the
    # radiation term (RRTMGP or the emulator) from the state it radiated,
    # 0 where the phase had no in-cloud condensate; on a cached radiation step
    # it holds the radii of the solve the heating came from. No term reads it
    # as an input to the radiation (see ``cloud_optics.radiation_effective_radii``),
    # and the post-physics diagnostics (COSP, AeroCom) do not read it either:
    # they describe the post-microphysics condensate, so they form its radii
    # with the same law (``cloud_optics.post_physics_effective_radii``).
    r_eff_liq: jnp.ndarray
    r_eff_ice: jnp.ndarray
    # Previous-timestep (t-dt) 2M number concentrations carried across
    # steps so the 2M ``update_tendencies_and_important_vars`` step has
    # the tm1 state it needs. Stored per kg of air (matching the
    # qnc/qni tracer convention).
    qnc_prev: jnp.ndarray            # Previous-step cloud droplet number [1/kg] (nlev, ncols)
    qni_prev: jnp.ndarray            # Previous-step ice crystal number    [1/kg] (nlev, ncols)

    # All-sky and clear-sky outgoing TOA fluxes from the radiation term's
    # cloudy + clear beam-split, written so users can compute the cloud
    # radiative effect (CRE) directly from a single output dataset:
    #
    #     CRE_SW = toa_sw_up_clear - toa_sw_up_all   (negative; cooling)
    #     CRE_LW = toa_lw_up_clear - toa_lw_up_all   (positive; warming)
    #     CRE    = CRE_SW + CRE_LW
    #
    # The all-sky values mirror ``RadiationData.toa_{sw,lw}_up`` — they
    # are duplicated here so the CRE consumer can read everything from
    # the ``"clouds"`` diagnostic key without having to cross-reference
    # the radiation key.
    toa_sw_up_all: jnp.ndarray       # All-sky outgoing SW at TOA [W/m²] (ncols,)
    toa_sw_up_clear: jnp.ndarray     # Clear-sky outgoing SW at TOA [W/m²] (ncols,)
    toa_lw_up_all: jnp.ndarray       # All-sky OLR at TOA [W/m²] (ncols,)
    toa_lw_up_clear: jnp.ndarray     # Clear-sky OLR at TOA [W/m²] (ncols,)

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        return cls(
            cloud_fraction=jnp.zeros((nlev,) + nodal_shape),
            qc=jnp.zeros((nlev,) + nodal_shape),
            qi=jnp.zeros((nlev,) + nodal_shape),
            conv_detrainment_qc=jnp.zeros((nlev,) + nodal_shape),
            conv_detrainment_qi=jnp.zeros((nlev,) + nodal_shape),
            precip_rain=jnp.zeros(nodal_shape),
            precip_snow=jnp.zeros(nodal_shape),
            rain_flux=jnp.zeros((nlev,) + nodal_shape),
            snow_flux=jnp.zeros((nlev,) + nodal_shape),
            precip_formation_rate=jnp.zeros((nlev,) + nodal_shape),
            precip_evaporation_rate=jnp.zeros((nlev,) + nodal_shape),
            rain_formation_warm=jnp.zeros(nodal_shape),
            rain_from_melt=jnp.zeros(nodal_shape),
            negative_mass_repair=jnp.zeros(nodal_shape),
            incloud_liquid=jnp.zeros((nlev,) + nodal_shape),
            incloud_ice=jnp.zeros((nlev,) + nodal_shape),
            incloud_rain_formation=jnp.zeros((nlev,) + nodal_shape),
            incloud_snow_formation=jnp.zeros((nlev,) + nodal_shape),
            incloud_riming=jnp.zeros((nlev,) + nodal_shape),
            process_cloud_fraction=jnp.zeros((nlev,) + nodal_shape),
            condensate_evaporation_rate=jnp.zeros((nlev,) + nodal_shape),
            droplet_number=jnp.zeros((nlev,) + nodal_shape),
            r_eff_liq=jnp.zeros((nlev,) + nodal_shape),
            r_eff_ice=jnp.zeros((nlev,) + nodal_shape),
            qnc_prev=jnp.zeros((nlev,) + nodal_shape),
            qni_prev=jnp.zeros((nlev,) + nodal_shape),
            toa_sw_up_all=jnp.zeros(nodal_shape),
            toa_sw_up_clear=jnp.zeros(nodal_shape),
            toa_lw_up_all=jnp.zeros(nodal_shape),
            toa_lw_up_clear=jnp.zeros(nodal_shape),
        )

    def copy(self, **kwargs):
        new_data = {
            'cloud_fraction': self.cloud_fraction,
            'qc': self.qc,
            'qi': self.qi,
            'conv_detrainment_qc': self.conv_detrainment_qc,
            'conv_detrainment_qi': self.conv_detrainment_qi,
            'precip_rain': self.precip_rain,
            'precip_snow': self.precip_snow,
            'rain_flux': self.rain_flux,
            'snow_flux': self.snow_flux,
            'precip_formation_rate': self.precip_formation_rate,
            'precip_evaporation_rate': self.precip_evaporation_rate,
            'rain_formation_warm': self.rain_formation_warm,
            'rain_from_melt': self.rain_from_melt,
            'negative_mass_repair': self.negative_mass_repair,
            'incloud_liquid': self.incloud_liquid,
            'incloud_ice': self.incloud_ice,
            'incloud_rain_formation': self.incloud_rain_formation,
            'incloud_snow_formation': self.incloud_snow_formation,
            'incloud_riming': self.incloud_riming,
            'process_cloud_fraction': self.process_cloud_fraction,
            'condensate_evaporation_rate': self.condensate_evaporation_rate,
            'droplet_number': self.droplet_number,
            'r_eff_liq': self.r_eff_liq,
            'r_eff_ice': self.r_eff_ice,
            'qnc_prev': self.qnc_prev,
            'qni_prev': self.qni_prev,
            'toa_sw_up_all': self.toa_sw_up_all,
            'toa_sw_up_clear': self.toa_sw_up_clear,
            'toa_lw_up_all': self.toa_lw_up_all,
            'toa_lw_up_clear': self.toa_lw_up_clear,
        }
        new_data.update(kwargs)
        return CloudData(**new_data)


#: CF/units metadata for the :class:`CloudData` fields as they appear in the
#: output Dataset — flattened to ``clouds.<field>`` keys (#740). Shared by every
#: term that writes the ``clouds`` sub-struct (Sundqvist cover, ECHAM 1M and
#: Lohmann 2M microphysics), which set ``output_attrs = CLOUD_OUTPUT_ATTRS`` on
#: their PhysicsTerm. Units are taken verbatim from the field comments above;
#: CF standard names are used only where the match is exact.
CLOUD_OUTPUT_ATTRS: dict[str, dict[str, str]] = {
    "clouds.cloud_fraction": {
        "standard_name": "cloud_area_fraction_in_atmosphere_layer",
        "units": "1", "long_name": "cloud fraction"},
    "clouds.qc": {
        "standard_name": "mass_fraction_of_cloud_liquid_water_in_air",
        "units": "kg kg-1", "long_name": "cloud liquid water mixing ratio"},
    "clouds.qi": {
        "standard_name": "mass_fraction_of_cloud_ice_in_air",
        "units": "kg kg-1", "long_name": "cloud ice mixing ratio"},
    "clouds.conv_detrainment_qc": {
        "units": "kg kg-1 s-1",
        "long_name": "convective detrainment of cloud liquid water"},
    "clouds.conv_detrainment_qi": {
        "units": "kg kg-1 s-1",
        "long_name": "convective detrainment of cloud ice"},
    "clouds.precip_rain": {
        "standard_name": "rainfall_flux",
        "units": "kg m-2 s-1", "long_name": "surface rainfall flux"},
    "clouds.precip_snow": {
        "standard_name": "snowfall_flux",
        "units": "kg m-2 s-1", "long_name": "surface snowfall flux"},
    "clouds.rain_flux": {
        "units": "kg m-2 s-1",
        "long_name": "large-scale rain flux leaving each layer"},
    "clouds.snow_flux": {
        "units": "kg m-2 s-1",
        "long_name": "large-scale snow (plus ice) flux leaving each layer"},
    "clouds.precip_formation_rate": {
        "units": "kg kg-1 s-1",
        "long_name": "grid-mean condensate-to-precipitation conversion rate"},
    "clouds.precip_evaporation_rate": {
        "units": "kg kg-1 s-1",
        "long_name": "grid-mean falling-precipitation evaporation rate"},
    "clouds.rain_formation_warm": {
        "units": "kg m-2 s-1",
        "long_name": "column rain formed by the warm chain"},
    "clouds.rain_from_melt": {
        "units": "kg m-2 s-1",
        "long_name": "column rain formed by melting snow"},
    "clouds.negative_mass_repair": {
        "units": "W m-2",
        "long_name": "latent heating of the negative-mass repair"},
    "clouds.incloud_liquid": {
        "units": "kg kg-1",
        "long_name": "in-cloud liquid water before precipitation formation"},
    "clouds.incloud_ice": {
        "units": "kg kg-1",
        "long_name": "in-cloud ice before precipitation formation"},
    "clouds.incloud_rain_formation": {
        "units": "kg kg-1 s-1", "long_name": "in-cloud rain formation rate"},
    "clouds.incloud_snow_formation": {
        "units": "kg kg-1 s-1",
        "long_name": ("in-cloud snow formation rate (signed: negative where "
                      "the level absorbs sedimenting ice)")},
    "clouds.incloud_riming": {
        "units": "kg kg-1 s-1",
        "long_name": "in-cloud riming of droplets by snow"},
    "clouds.process_cloud_fraction": {
        "units": "1", "long_name": "cloud cover the microphysics ran under"},
    "clouds.condensate_evaporation_rate": {
        "units": "kg kg-1 s-1",
        "long_name": "grid-mean cloud-condensate evaporation rate"},
    "clouds.droplet_number": {
        "standard_name":
            "number_concentration_of_cloud_liquid_water_particles_in_air",
        "units": "m-3", "long_name": "cloud droplet number concentration"},
    "clouds.r_eff_liq": {
        "units": "um",
        "long_name": "cloud droplet effective radius used by the radiation"},
    "clouds.r_eff_ice": {
        "units": "um",
        "long_name": "cloud ice effective radius used by the radiation"},
    "clouds.qnc_prev": {
        "units": "kg-1",
        "long_name": "previous-step cloud droplet number per kg air"},
    "clouds.qni_prev": {
        "units": "kg-1",
        "long_name": "previous-step ice crystal number per kg air"},
    "clouds.toa_sw_up_all": {
        "standard_name": "toa_outgoing_shortwave_flux",
        "units": "W m-2", "long_name": "all-sky TOA outgoing shortwave flux"},
    "clouds.toa_sw_up_clear": {
        "standard_name": "toa_outgoing_shortwave_flux_assuming_clear_sky",
        "units": "W m-2", "long_name": "clear-sky TOA outgoing shortwave flux"},
    "clouds.toa_lw_up_all": {
        "standard_name": "toa_outgoing_longwave_flux",
        "units": "W m-2", "long_name": "all-sky TOA outgoing longwave flux"},
    "clouds.toa_lw_up_clear": {
        "standard_name": "toa_outgoing_longwave_flux_assuming_clear_sky",
        "units": "W m-2", "long_name": "clear-sky TOA outgoing longwave flux"},
}


def radiation_cloud_fields(state, diagnostics):
    """Return the cloud fields radiation sees, as ECHAM hands them over.

    ECHAM ``physc`` calls ``cover`` before radiation and passes radiation
    the diagnosed cover with the step-start grid-mean condensate
    ``xlm1``/``xim1`` (``physc.f90`` l.566-573); the cloud microphysics runs
    later. Inside ``radiation`` (``mo_radiation.f90`` l.428-434) the
    condensate is clipped at zero, ``xq = MAX(qm, 0)``, and the cover is
    kept only where there is condensate,
    ``xc_frc = MERGE(cld_frc, 0, xq_liq > 0 .OR. xq_ice > 0)``. That masked
    cover is what the radiative transfer, its total-cover diagnostic and
    COSP (``cosp_f3d``, ``mo_psrad_interface.f90`` l.414) see; the cover
    field itself is not changed. This returns the same: the step-start
    ``qc``/``qi`` tracers clipped at zero, and the diagnosed
    ``clouds.cloud_fraction`` zeroed where both are zero.

    The mask is ECHAM's hard test, derivative included. A cell it clears has
    no condensate, so it has no optical depth either way and contributes to
    no flux through its own optics.
    """
    clouds = diagnostics["clouds"]
    zeros = jnp.zeros_like(state.temperature)
    cloud_water = jnp.maximum(state.tracers.get("qc", zeros), 0.0)
    cloud_ice = jnp.maximum(state.tracers.get("qi", zeros), 0.0)
    return (cloud_water, cloud_ice,
            condensate_masked_cover(clouds.cloud_fraction, cloud_water,
                                    cloud_ice))


def condensate_masked_cover(cloud_fraction, cloud_water, cloud_ice):
    """ECHAM's ``MERGE(cld_frc, 0, xq_liq > 0 .OR. xq_ice > 0)``.

    ``mo_radiation.f90`` l.433-434: the cover where either condensate is
    positive, 0 elsewhere.
    """
    has_condensate = (cloud_water > 0.0) | (cloud_ice > 0.0)
    return jnp.where(has_condensate, cloud_fraction, 0.0)
