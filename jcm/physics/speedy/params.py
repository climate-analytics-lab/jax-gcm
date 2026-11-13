"""For storing variables used by multiple physics schemes."""
import dataclasses

import tree_math
import jax.numpy as jnp
from jax import tree_util

@tree_math.struct
class ConvectionParameters:
    psmin: jnp.ndarray # Minimum (normalised) surface pressure for the occurrence of convection
    trcnv: jnp.ndarray # Time of relaxation (in hours) towards reference state
    rhil: jnp.ndarray # Relative humidity threshold in intermediate layers for secondary mass flux
    rhbl: jnp.ndarray # Relative humidity threshold in the boundary layer
    entmax: jnp.ndarray # Maximum entrainment as a fraction of cloud-base mass flux
    smf: jnp.ndarray # Ratio between secondary and primary mass flux at cloud-base
    # Surrogate-derivative half-widths (jcm.physics.speedy.smoothing): the
    # trigger, the onset and the surface-pressure switch keep their hard
    # values at any width, and the width gives them the derivative of a
    # smooth function; 0 = the reference derivative. trigger_smoothing is
    # an RH fraction for the boundary-layer humidity trigger;
    # precnv_smoothing is a cloud-top moisture excess per unit cloud-base
    # mass flux [g/kg] for the convective-precipitation onset hinge
    # (formed per unit so a switching-on plume's tangent rains at the
    # plume's own yield); psmin_smoothing is a normalised surface
    # pressure for the psa > psmin switch and the clipped ramp
    # psa * min(1, (psa - psmin) * 2/(1 - psmin)) it puts on the cloud-base
    # mass flux, both plateaux of d/d(psmin).
    trigger_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    precnv_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    psmin_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))

    @classmethod
    def default(cls, trigger_smoothing=0.0, precnv_smoothing=0.0, psmin_smoothing=0.0):
        return cls(
            psmin = jnp.array(0.8),
            trcnv = jnp.array(6.0),
            rhil = jnp.array(0.7),
            rhbl = jnp.array(0.9),
            entmax = jnp.array(0.5),
            smf = jnp.array(0.8),
            trigger_smoothing = jnp.array(trigger_smoothing),
            precnv_smoothing = jnp.array(precnv_smoothing),
            psmin_smoothing = jnp.array(psmin_smoothing)
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)


@tree_math.struct
class CondensationParameters:
    trlsc: jnp.ndarray   # Relaxation time (in hours) for specific humidity
    rhlsc: jnp.ndarray  # Maximum relative humidity threshold (at sigma=1)
    drhlsc: jnp.ndarray  # Vertical range of relative humidity threshold
    rhblsc: jnp.ndarray # Relative humidity threshold for boundary layer
    # Surrogate-derivative half-widths; the gated values stay the hard
    # operations at any width and 0 = the reference derivative (see
    # jcm.physics.speedy.smoothing). cap_smoothing (a fraction of the cap)
    # is for the heating-rate cap; bl_threshold_smoothing (an RH fraction)
    # is for the bottom-level threshold selection max(rhref[-1], rhblsc),
    # a plateau in which only one of rhblsc and rhlsc has a derivative at
    # that level.
    cap_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    bl_threshold_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))

    @classmethod
    def default(cls, cap_smoothing=0.0, bl_threshold_smoothing=0.0):
        return cls(
            trlsc = jnp.array(4.0),
            rhlsc = jnp.array(0.9),
            drhlsc = jnp.array(0.1),
            rhblsc = jnp.array(0.95),
            cap_smoothing = jnp.array(cap_smoothing),
            bl_threshold_smoothing = jnp.array(bl_threshold_smoothing)
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ShortwaveRadiationParameters:
    # parameters for `get_zonal_average_fields`

    albcl:  jnp.ndarray # Cloud albedo (for cloud cover = 1)
    albcls: jnp.ndarray # Stratiform cloud albedo (for st. cloud cover = 1)

    # Shortwave absorptivities (for dp = 10^5 Pa)
    absdry: jnp.ndarray # Absorptivity of dry air (visible band)
    absaer: jnp.ndarray # Absorptivity of aerosols (visible band)
    abswv1: jnp.ndarray # Absorptivity of water vapour
    abswv2: jnp.ndarray # Absorptivity of water vapour
    abscl1: jnp.ndarray # Absorptivity of clouds (visible band, maximum value)
    abscl2: jnp.ndarray # Absorptivity of clouds

    # Longwave absorptivities (for dp = 10^5 Pa)
    ablwin: jnp.ndarray # Absorptivity of air in "window" band
    ablwv1: jnp.ndarray # Absorptivity of water vapour in H2O band 1 (weak) (for dq = 1 g/kg)
    ablwv2: jnp.ndarray # Absorptivity of water vapour in H2O band 2 (strong) (for dq = 1 g/kg)
    ablcl1: jnp.ndarray # Absorptivity of "thick" clouds in window band (below cloud top)
    ablcl2: jnp.ndarray # Absorptivity of "thin" upper clouds in window and H2O bands

    # parameters for `clouds`
    rhcl1: jnp.ndarray  # Relative humidity threshold corresponding to cloud cover = 0
    rhcl2: jnp.ndarray  # Relative humidity correponding to cloud cover = 1
    qacl: jnp.ndarray  # Specific humidity threshold for cloud cover
    wpcl: jnp.ndarray   # Cloud cover weight for the square-root of precipitation (for p = 1 mm/day)
    pmaxcl: jnp.ndarray  # Maximum value of precipitation (mm/day) contributing to cloud cover
    clsmax: jnp.ndarray  # Maximum stratiform cloud cover
    clsminl: jnp.ndarray  # Minimum stratiform cloud cover over land (for RH = 1)
    gse_s0: jnp.ndarray # Gradient of dry static energy corresponding to stratiform cloud cover = 0
    gse_s1: jnp.ndarray  # Gradient of dry static energy corresponding to stratiform cloud cover = 1
    # Clear-sky reflectance of the atmosphere (molecular and aerosol backscatter) per unit
    # normalised surface pressure and zenith-path factor, applied to the visible band at the top
    # layer. Not in SPEEDY's Fortran, whose clear sky only absorbs (default 0 = the reference);
    # see docs/source/design/speedy_retune.md for why the speedy preset sets it.
    refclr: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    # Water-vapour self-continuum absorptivity of the "window" band (for dq = 1 g/kg at
    # p = 10^5 Pa), an addition to SPEEDY, whose window absorbs independently of humidity
    # (default 0 = the reference): the layer absorptivity gains ablwvc * q^2 * p/p0, the
    # scaling of the self continuum (vapour pressure times vapour amount). See
    # docs/source/science/radiation.md and docs/source/design/speedy_longwave_fit.md.
    ablwvc: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    # Surrogate-derivative half-width (in cover/RH-fraction units) for the
    # cloud-cover hinges and clips in the diagnosis (RH hinge at rhcl1,
    # cover saturation at 1, the fstab clip, the stratiform hinges, the
    # sqrt(precip) corner), which keep their hard values at any width;
    # 0 = the reference derivative (see jcm.physics.speedy.smoothing).
    cover_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))

    @classmethod
    def default(cls, cover_smoothing=0.0):
        return cls(
            albcl = jnp.array(0.43),
            albcls = jnp.array(0.50),
            absdry = jnp.array(0.033),
            absaer = jnp.array(0.033),
            abswv1 = jnp.array(0.022),
            abswv2 = jnp.array(15.000),
            abscl1 = jnp.array(0.015),
            abscl2 = jnp.array(0.15),
            ablwin = jnp.array(0.3),
            ablwv1 = jnp.array(0.7),
            ablwv2 = jnp.array(50.0),
            ablcl1 = jnp.array(12.0),
            ablcl2 = jnp.array(0.6),
            rhcl1 = jnp.array(0.30),
            rhcl2 = jnp.array(1.00),
            qacl = jnp.array(0.20),
            wpcl = jnp.array(0.2),
            pmaxcl = jnp.array(10.0),
            clsmax = jnp.array(0.60),
            clsminl = jnp.array(0.15),
            gse_s0 = jnp.array(0.25),
            gse_s1 = jnp.array(0.40),
            cover_smoothing = jnp.array(cover_smoothing)
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ModRadConParameters:
    # Every SPEEDY term that reads this record holds its own copy, and reads only its own. Of
    # the longwave fields, the forcing term reads ``ablco2_ref``; the shortwave term ``epslw``;
    # the downward and upward longwave terms ``epslw``, ``emisfc`` and the band fractions; the
    # surface-flux term ``emisfc``. A value therefore has to be set in every holder:
    # ``speedy_physics()`` passes one record to all of them, and a YAML sets it through one
    # anchor (``configuration/speedy-t31.yaml``). An override of one term's copy alone leaves
    # the others at their defaults.

    # Albedo values
    albsea: jnp.ndarray  # Albedo over sea
    albice: jnp.ndarray  # Albedo over sea ice (for ice fraction = 1)
    albsn: jnp.ndarray # Albedo over snow (for snow cover = 1)

    # Longwave parameters
    epslw: jnp.ndarray  # Fraction of blackbody spectrum absorbed/emitted by PBL only
    emisfc: jnp.ndarray  # Longwave surface emissivity

    # Absorptivity of air in the CO2 band (for dp = 10^5 Pa) at the reference CO2
    # concentration (jcm.physics.forcing.speedy_forcing.CO2_VMR_REF_PPMV); the forcing scales
    # it with the run's CO2. SPEEDY's value is physics_data.ablco2_ref.
    ablco2_ref: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(6.0))
    # Energy fraction of the blackbody spectrum in the CO2 band and in the weak and strong
    # water-vapour bands, each ``a + b * (T - t0)**2`` with T clipped to 200-320 K (SPEEDY's
    # radset table, whose values are the defaults); the window band takes the remainder.
    fbco2_a: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.148))
    fbco2_b: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(-3.0e-6))
    fbco2_t0: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(247.0))
    fbwv1_a: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.356))
    fbwv1_b: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(-5.2e-6))
    fbwv1_t0: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(282.0))
    fbwv2_a: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.314))
    fbwv2_b: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(1.0e-5))
    fbwv2_t0: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(315.0))

    @classmethod
    def default(cls):
        return cls(
            albsea = jnp.array(0.07),
            albice = jnp.array(0.60),
            albsn = jnp.array(0.60),
            epslw = jnp.array(0.05),
            emisfc = jnp.array(0.98)
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class SurfaceFluxParameters:
    fwind0: jnp.ndarray # Ratio of near-sfc wind to lowest-level wind

    # Weight for near-sfc temperature extrapolation (0-1) :
    # 1 : linear extrapolation from two lowest levels
    # 0 : constant potential temperature ( = lowest level)
    ftemp0: jnp.ndarray

    # Weight for near-sfc specific humidity extrapolation (0-1) :
    # 1 : extrap. with constant relative hum. ( = lowest level)
    # 0 : constant specific hum. ( = lowest level)
    fhum0: jnp.ndarray

    cdl: jnp.ndarray   # Drag coefficient for momentum over land
    cds: jnp.ndarray   # Drag coefficient for momentum over sea
    chl: jnp.ndarray  # Heat exchange coefficient over land
    chs: jnp.ndarray   # Heat exchange coefficient over sea
    vgust: jnp.ndarray   # Wind speed for sub-grid-scale gusts
    ctday: jnp.ndarray # Daily-cycle correction (dTskin/dSSRad)
    dtheta: jnp.ndarray   # Potential temp. gradient for stability correction
    fstab: jnp.ndarray   # Amplitude of stability correction (fraction)
    clambda: jnp.ndarray  # Heat conductivity in skin-to-root soil layer
    clambsn: jnp.ndarray # Heat conductivity in soil for snow cover = 1

    lscasym: jnp.bool   # true : use an asymmetric stability coefficient
    lskineb: jnp.bool   # true : redefine skin temp. from energy balance

    hdrag: jnp.ndarray # Height scale for orographic correction
    # Surrogate-derivative half-widths; the gated values stay the hard
    # operations at any width and 0 = the reference derivative (see
    # jcm.physics.speedy.smoothing). evap_smoothing [g/kg] is for the
    # soil-moisture-limited evaporation onset hinge; stability_smoothing
    # [K] is for the clip of the surface-to-air temperature excess to
    # +/- dtheta in the stability correction, a plateau of d/d(dtheta).
    evap_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    stability_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))

    @classmethod
    def default(cls, evap_smoothing=0.0, stability_smoothing=0.0):
        return cls(
            fwind0 = jnp.array(0.95),
            ftemp0 = jnp.array(1.0),
            fhum0 = jnp.array(0.0),
            cdl = jnp.array(2.4e-3),
            cds = jnp.array(1.0e-3),
            chl = jnp.array(1.2e-3),
            chs = jnp.array(0.9e-3),
            vgust = jnp.array(5.0),
            ctday = jnp.array(1.0e-2),
            dtheta = jnp.array(3.0),
            fstab = jnp.array(0.67),
            clambda = jnp.array(7.0),
            clambsn = jnp.array(7.0),
            lscasym = True,
            lskineb = True,
            hdrag = jnp.array(2000.0),
            evap_smoothing = jnp.array(evap_smoothing),
            stability_smoothing = jnp.array(stability_smoothing)
        )

    def isnan(self):
        self.lscasym = 0
        self.lskineb = 0
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class LandModelParameters:
    """SPEEDY's slab land (``land_model.f90::land_model_init``).

    The land surface is a single slab -- soil, or land ice where the
    background albedo marks an ice sheet -- whose temperature anomaly from the
    prescribed climatology ``stl_am`` is driven by the land tile's net heat
    flux ``hfluxn`` and relaxed back to the climatology on ``tdland``:

        C d(T')/dt = hfluxn_land - C T' / tdland,

    only where the land fraction is at least ``flandmin`` (elsewhere the land
    keeps the climatology, as in the reference). Soil moisture and snow stay
    prescribed, as in SPEEDY. Values are the reference's.
    """

    depth_soil: jnp.ndarray  # Soil layer depth [m]
    depth_lice: jnp.ndarray  # Land-ice layer depth [m]
    hcap_soil: jnp.ndarray   # Volumetric heat capacity of soil [J/m3/K]
    hcap_lice: jnp.ndarray   # Volumetric heat capacity of land ice [J/m3/K]
    # Dissipation (relaxation) time of the anomaly [days]. The relaxation is
    # an energy source/sink of C/tdland per kelvin of anomaly (0.72 W/m2/K for
    # the 1 m soil at 40 days); jnp.inf gives a free, energy-conserving slab.
    tdland: jnp.ndarray
    flandmin: jnp.ndarray    # Minimum land fraction for a prognostic anomaly
    albice_threshold: jnp.ndarray  # Background albedo at and above which the slab is land ice

    @classmethod
    def default(cls):
        return cls(
            depth_soil=jnp.array(1.0),
            depth_lice=jnp.array(5.0),
            hcap_soil=jnp.array(2.50e6),
            hcap_lice=jnp.array(1.93e6),
            tdland=jnp.array(40.0),
            flandmin=jnp.array(1.0 / 3.0),
            albice_threshold=jnp.array(0.4),
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class VerticalDiffusionParameters:
    trshc: jnp.ndarray  # Relaxation time (in hours) for shallow convection
    trvdi: jnp.ndarray  # Relaxation time (in hours) for moisture diffusion
    trvds: jnp.ndarray  # Relaxation time (in hours) for super-adiabatic conditions
    redshc: jnp.ndarray  # Reduction factor of shallow convection in areas of deep convection
    rhgrad: jnp.ndarray  # Maximum gradient of relative humidity (d_RH/d_sigma)
    segrad: jnp.ndarray  # Minimum gradient of dry static energy (d_DSE/d_phi)
    # Surrogate-derivative half-widths for the shallow-convection gates,
    # which keep their hard values at any width; 0 = the reference
    # derivative (see jcm.physics.speedy.smoothing). mse_gate_smoothing
    # [J/kg] is for the dmse >= 0 shallow-convection / moisture-diffusion
    # split (the moisture flux is proportional to drh, not to dmse, so the
    # hard split is a value jump) and the shallow-convection heat flux's
    # kink; rh_gate_smoothing [RH fraction] is for the drh tests: the
    # drh > drh0 onset of the moisture-diffusion fluxes, a jump for the same
    # reason, and shallow convection's drh >= 0, a kink.
    mse_gate_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))
    rh_gate_smoothing: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array(0.0))

    @classmethod
    def default(cls, mse_gate_smoothing=0.0, rh_gate_smoothing=0.0):
        return cls(
            trshc = jnp.array(6.0),
            trvdi = jnp.array(24.0),
            trvds = jnp.array(6.0),
            redshc = jnp.array(0.5),
            rhgrad = jnp.array(0.5),
            segrad = jnp.array(0.1),
            mse_gate_smoothing = jnp.array(mse_gate_smoothing),
            rh_gate_smoothing = jnp.array(rh_gate_smoothing)
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class Parameters:
    convection: ConvectionParameters
    condensation: CondensationParameters
    shortwave_radiation: ShortwaveRadiationParameters
    mod_radcon: ModRadConParameters
    surface_flux: SurfaceFluxParameters
    vertical_diffusion: VerticalDiffusionParameters

    @classmethod
    def default(cls):
        return cls(
            convection = ConvectionParameters.default(),
            condensation = CondensationParameters.default(),
            shortwave_radiation = ShortwaveRadiationParameters.default(),
            mod_radcon = ModRadConParameters.default(),
            surface_flux = SurfaceFluxParameters.default(),
            vertical_diffusion = VerticalDiffusionParameters.default(),
        )

    def isnan(self):
        return Parameters(
            convection=self.convection.isnan(),
            condensation=self.condensation.isnan(),
            shortwave_radiation=self.shortwave_radiation.isnan(),
            mod_radcon = self.mod_radcon.isnan(),
            surface_flux = self.surface_flux.isnan(),
            vertical_diffusion = self.vertical_diffusion.isnan(),
        )

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))

    @classmethod
    def float_zeros(cls):
        """Return a Parameters instance with all fields replaced by float zeros.
        This is useful for creating parameter co-tangents.
        """
        import jax

        def _float_zeros(x):
            if jnp.issubdtype(jnp.result_type(x), jnp.bool_):
                return jnp.zeros((), dtype=jax.dtypes.float0)
            elif jnp.issubdtype(jnp.result_type(x), jnp.integer):
                return jnp.zeros((), dtype=jax.dtypes.float0)
            else:
                return jnp.zeros_like(x)
        return tree_util.tree_map(lambda x: _float_zeros(x), cls.default())

    def __str__(self):
        """Return a human-readable string representation of the Parameters instance."""
        from pprint import pformat

        def to_readable_format(x):
            if isinstance(x, jnp.ndarray):
                return x.tolist()
            return x

        return pformat(tree_util.tree_map(to_readable_format, self))