"""``echam_physics()`` factory.

Every ECHAM parameterisation lives as a ``PhysicsTerm`` next to its
underlying numerical implementation (``TiedtkeConvection``,
``SundqvistCloudFraction``, ``Echam1MMicrophysics``,
``RRTMGPRadiation``, …) and owns its own scheme-native
``Parameters``. This module is the user-facing factory that wires the
scheme-named terms together in a validated default ordering and returns
a ready-to-run ``ComposablePhysics`` with column vectorisation enabled.

The factory accepts per-scheme ``Parameters`` objects directly — there
is no monolithic ECHAM ``Parameters`` aggregator. Each unspecified
sub-Parameters falls through to its scheme's ``.default()`` constructor,
so callers only have to pass the knobs they want to tune. A mapping of field
overrides in place of an object (what a Hydra ``physics.<scheme>.<field>=``
override delivers) is applied on top of the object the factory would
otherwise use.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Mapping

from jcm.physics.aerosol import Macv2SpAerosol
from jcm.physics.aerosol.macv2_sp_params import AerosolParameters
from jcm.physics.chemistry import SimpleChemistry
from jcm.physics.clouds.echam_1m import (
    Echam1MMicrophysics,
    MicrophysicsParameters,
)
from jcm.physics.clouds.lohmann_2m import Lohmann2MMicrophysics
from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
from jcm.physics.clouds.sundqvist import (
    SundqvistCloudFraction,
    CloudParameters,
)
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.convection.tiedtke_nordeng import (
    TiedtkeConvection,
    ConvectionParameters,
)
from jcm.physics.diagnostics.moist_air_state import MoistAirColumnState
from jcm.physics.forcing.echam_boundary_conditions import (
    EchamBoundaryConditions,
)
from jcm.physics.gravity_waves.hines import HinesGwd, HinesParameters
from jcm.physics.gravity_waves.sso import LottMillerSso, SSOParameters
from jcm.physics.physics_term import with_field_overrides
from jcm.physics.surface.echam.jsbach_land import JsbachLandParameters
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.resolution_defaults import (
    default_parameters,
    defaults_flag_kwargs,
    spectral_truncation,
)
from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
from jcm.physics.radiation.aerosol_free import (
    resolve_aerosol_free_interval,
)
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
from jcm.physics.surface.echam.surface_physics import EchamSurface
from jcm.physics.surface.echam.surface_exchange_publisher import (
    EchamSurfaceExchange,
)
from jcm.physics.surface.echam.surface_types import SurfaceParameters
from jcm.physics.surface.prescribed_flux import PrescribedSurfaceFlux
from jcm.physics.vertical_diffusion.tte_tke import TteTkeVerticalDiffusion
from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
    VDiffParameters,
)

if TYPE_CHECKING:
    # Only the annotations of ``echam_physics`` name these: the JAM package is
    # imported in the JAM branch of the factory, not for every composition.
    from jcm.physics.aerosol.jam.activation.arg_term import ArgParameters
    from jcm.physics.aerosol.jam.chemistry.aqueous import (
        AqueousSulfurParameters)
    from jcm.physics.aerosol.jam.chemistry.oxidants import OxidantParameters
    from jcm.physics.aerosol.jam.chemistry.sulfur_gas import (
        SulfurGasParameters)
    from jcm.physics.aerosol.jam.cloud_borne import (
        CloudBorneExchangeParameters)
    from jcm.physics.aerosol.jam.drydep.drydep_term import DryDepParameters
    from jcm.physics.aerosol.jam.emissions.anthropogenic import (
        EmissionParameters)
    from jcm.physics.aerosol.jam.emissions.dms import DmsParameters
    from jcm.physics.aerosol.jam.emissions.seasalt import SeaSaltParameters
    from jcm.physics.aerosol.jam.sedimentation.sedi_term import SedParameters
    from jcm.physics.aerosol.jam.wetdep.wetdep_term import WetDepParameters
    from jcm.physics.convection.tracer_transport import ConvTransportParameters
    from jcm.physics.vertical_diffusion.tracer_diffusion import (
        TracerDiffusionParameters)


#: Raised for ``echam_physics(radiation_scheme="grey")``. The grey two-stream
#: is an idealized scheme (like Betts-Miller convection): it has no ECHAM
#: reference formulation and was never validated in an ECHAM composition, so
#: the ECHAM factory does not offer it by name. It can still be composed
#: explicitly as a term, which makes the idealized choice visible at the call
#: site.
GREY_RADIATION_REJECTION = (
    "radiation_scheme='grey' is not an ECHAM radiation option: the grey "
    "two-stream is an idealized scheme, not ECHAM physics (no ECHAM reference, "
    "never validated in an ECHAM composition). echam_physics() offers "
    "'rrtmgp' (default) and 'emulated' (the fast option). To run an idealized "
    "composition with the grey scheme, build the term and pass the instance:\n"
    "    from jcm.physics.radiation.grey_two_stream import "
    "GreyTwoStreamRadiation\n"
    "    physics = echam_physics(radiation_scheme=GreyTwoStreamRadiation())\n"
    "(radiation parameters go to its constructor, params=...). The factory "
    "derives the band structure and the JAM optics cadence from the term it is "
    "given, so this is the one supported route."
)


def default_radiation_parameters(aerosol_module: str = "macv2sp"
                                 ) -> RadiationParameters:
    """``RadiationParameters`` the ECHAM factory uses when none are given.

    ECHAM-HAM re-tunes the ice-cloud inhomogeneity for its two-moment +
    Abdul-Razzak & Ghan configuration (``lcdnc_progn`` with ``ncd_activ = 2``:
    ``zinhomi = 0.7`` at T63, L31 and L47 alike;
    ``mo_cloud_optics.f90::setup_cloud_optics``). jcm's JAM stack is exactly
    that pairing — the 2M scheme fed by ARG activation — so it takes HAM's
    value; every other composition keeps ECHAM6's T63 ``zinhomi = 0.8``.

    Public so a caller that builds its own radiation term for
    ``echam_physics(radiation_scheme=<term>)`` can give it the same defaults
    the string route would.
    """
    return RadiationParameters.default(
        cloud_inhomogeneity_ice=0.7 if aerosol_module == "jam" else 0.8)


def _warn_if_shared_cloud_constants_differ(cover_params, cloud_params, cloud_scheme):
    """Warn when jcm's two copies of one ECHAM cloud constant differ.

    ECHAM has one ``csecfrl`` (``mo_echam_cloud_params.f90`` l.76) and one
    ``cthomi`` (l.54), read by both its cover and its cloud scheme. jcm holds
    one copy in the cover's ``CloudParameters`` (``csecfrl``, ``t_ice``) and
    one in the cloud scheme's parameters (the 1M's ``csecfrl`` and
    ``cthomi``; the 2M's ``cthomi``). Differing copies are allowed, and are a
    departure from ECHAM, so they are reported rather than refused. Values
    that are not concrete at construction (traced) are not compared.
    """
    import math

    import numpy as np

    pairs = [("t_ice", "cthomi")]
    if cloud_scheme == "1m":
        pairs.insert(0, ("csecfrl", "csecfrl"))
    for cover_name, cloud_name in pairs:
        try:
            a = float(np.asarray(getattr(cover_params, cover_name)))
            b = float(np.asarray(getattr(cloud_params, cloud_name)))
        except Exception:  # a traced leaf has no value to compare at build time
            continue
        if not math.isclose(a, b, rel_tol=1e-6):
            warnings.warn(
                f"CloudParameters.{cover_name} = {a:g} but the {cloud_scheme} "
                f"cloud scheme's {cloud_name} = {b:g}: ECHAM has one "
                f"{cloud_name} for its cover and its cloud scheme; the two jcm "
                "copies are used as given.", UserWarning, stacklevel=3)


def echam_physics(
    *,
    convection: ConvectionParameters | Mapping[str, Any] | None = None,
    clouds: CloudParameters | Mapping[str, Any] | None = None,
    microphysics: MicrophysicsParameters | Mapping[str, Any] | None = None,
    microphysics_2m: CloudParams2M | Mapping[str, Any] | None = None,
    radiation: RadiationParameters | Mapping[str, Any] | None = None,
    vertical_diffusion: VDiffParameters | Mapping[str, Any] | None = None,
    surface: SurfaceParameters | Mapping[str, Any] | None = None,
    land_surface: JsbachLandParameters | Mapping[str, Any] | None = None,
    aerosol: AerosolParameters | Mapping[str, Any] | None = None,
    hines: HinesParameters | Mapping[str, Any] | None = None,
    sso: SSOParameters | Mapping[str, Any] | None = None,
    seasalt: SeaSaltParameters | Mapping[str, Any] | None = None,
    dms: DmsParameters | Mapping[str, Any] | None = None,
    anthropogenic_params: EmissionParameters | Mapping[str, Any] | None = None,
    oxidants: OxidantParameters | Mapping[str, Any] | None = None,
    sulfur_gas: SulfurGasParameters | Mapping[str, Any] | None = None,
    aqueous: AqueousSulfurParameters | Mapping[str, Any] | None = None,
    activation: ArgParameters | Mapping[str, Any] | None = None,
    cloud_borne_exchange: (
        CloudBorneExchangeParameters | Mapping[str, Any] | None) = None,
    sedimentation: SedParameters | Mapping[str, Any] | None = None,
    drydep: DryDepParameters | Mapping[str, Any] | None = None,
    wetdep: WetDepParameters | Mapping[str, Any] | None = None,
    tracer_diffusion: (
        TracerDiffusionParameters | Mapping[str, Any] | None) = None,
    conv_transport: ConvTransportParameters | Mapping[str, Any] | None = None,
    gw_scheme: str = "hines",
    checkpoint_terms: bool = True,
    radiation_scheme: str | PhysicsTerm = "rrtmgp",
    emulator_weights_file: str | None = "auto",
    radiation_compute_cre: bool = True,
    cloud_scheme: str = "1m",
    aerosol_module: str = "macv2sp",
    jam_microphysics: str = "placeholder",
    jam_cloud_borne: bool = True,
    jam_optics: bool = True,
    jam_optics_backend: str = "jcm",
    jam_optics_tables_dir: str | os.PathLike | None = None,
    jam_seasalt_scheme: str = "gong",
    jam_arg_variant: str = "arg2000",
    jam_activation_scheme: str = "arg",
    jam_nactivpdf: int = 0,
    jam_aqueous_scheme: str = "full",
    jam_dust_preset: int = 4,
    jam_dust_nudged: bool = False,
    jam_dust_nduscale_scale: float | None = None,
    jam_anthropogenic: bool = False,
    jam_prescribed_speciated: bool = False,
    jam_convective_transport: bool = True,
    convective_updraft_precip_cover: bool | None = None,
    enable_cosp: bool = False,
    cosp_ncolumns: int = 40,
    cosp_calipso: bool = False,
    cosp_modis: bool = False,
    cosp_isccp: bool = False,
    aerosol_free_interval: int | None = None,
    enable_aerocom: bool = False,
    aerocom_groups: tuple[str, ...] = ("cloud", "column"),
    aerocom_overlap: str = "maximum-random",
    aerocom_optics: bool = False,
    diagnose_omega: bool = False,
    cu_lmfmid: bool | None = None,
    prescribed_surface_fluxes: bool = False,
    coords=None,
):
    """Create a ``ComposablePhysics`` with the standard ECHAM term ordering.

    Each per-scheme ``Parameters`` object is optional; ``None`` resolves
    to the scheme's ``.default()``. There is no monolithic aggregator —
    the composition assembled here is the only place where the ECHAM
    stack's per-scheme parameters meet.

    Each per-scheme argument also accepts a mapping of field overrides,
    e.g. ``convection={"entrpen": 4e-4, "tau": 3600.0}``. It is applied on
    top of the object this factory would otherwise use (its ``.default()``
    or the factory's own choice, such as the ``aerosol_module``-dependent
    radiation defaults or ``cu_lmfmid``), so the unspecified fields keep
    those values. This is how the factory-built Hydra presets take
    ``physics.convection.entrpen=...``; the conversion is the one the
    term-list presets use (:func:`~jcm.physics.physics_term.
    with_field_overrides`): an unknown field raises ``ValueError`` listing
    the valid ones, and numeric fields stay differentiable pytree leaves.
    A mapping for a scheme the composition does not include (``microphysics``
    with ``cloud_scheme="2m"``, ``hines`` with ``gw_scheme`` ``"frontal"`` or
    ``"none"``, ``aerosol`` with ``aerosol_module="jam"``, the JAM schemes
    below without it, ``anthropogenic_params`` without ``jam_anthropogenic``,
    ``cloud_borne_exchange`` without ``jam_cloud_borne``, ``conv_transport``
    without ``jam_convective_transport``, ``radiation`` with a radiation term
    instance) and a mapping that sets ``cu_lmfmid`` alongside the scalar
    ``cu_lmfmid`` flag are rejected rather than ignored. The JAM dust
    parameters have no argument here beyond the ``jam_dust_*`` flags yet
    (jax-gcm#995).

    Args:
        convection: Override for ``ConvectionParameters``.
        clouds: Override for the diagnostic cloud-fraction
            ``CloudParameters``.
        microphysics: Override for 1-moment microphysics
            ``MicrophysicsParameters`` (used when ``cloud_scheme="1m"``).
        microphysics_2m: Override for 2-moment microphysics
            ``CloudParams2M`` (used when ``cloud_scheme="2m"``).
        radiation: Override for ``RadiationParameters`` of the named
            radiation scheme. When omitted, :func:`default_radiation_parameters`
            supplies them (``aerosol_module="jam"`` takes ECHAM-HAM's 2M + ARG
            ice inhomogeneity ``cloud_inhomogeneity_ice = 0.7``). Rejected
            alongside a radiation ``PhysicsTerm`` instance, which carries its
            own parameters.
        vertical_diffusion: Override for TTE-TKE ``VDiffParameters``.
        surface: Override for ``SurfaceParameters``.
        land_surface: Override for the land tile's
            :class:`~jcm.physics.surface.echam.jsbach_land.JsbachLandParameters`
            (the JSBACH evaporation form and skin energy balance, #979).
        aerosol: Override for MACv2-SP ``AerosolParameters``. Also
            supplies the SPA activation knobs read by the 2M scheme
            when ``cloud_scheme="2m"``.
        hines: Override for non-orographic GW ``HinesParameters``.
        sso: Override for sub-grid-scale orography ``SSOParameters``.
        seasalt: Override for the JAM Gong (2003) sea-salt emission
            :class:`~jcm.physics.aerosol.jam.emissions.seasalt.SeaSaltParameters`
            (``scale``, the overall emission multiplier, and
            ``wind_exponent``). ``aerosol_module="jam"`` only: an argument
            given without it is rejected rather than ignored.
        dms: Override for the JAM Nightingale (2000) DMS sea-air emission
            :class:`~jcm.physics.aerosol.jam.emissions.dms.DmsParameters`
            (``flux_scale``, the multiplier on the emitted DMS flux).
            ``aerosol_module="jam"`` only, as ``seasalt``.
        anthropogenic_params: Override for the prescribed CEDS anthropogenic
            emission
            :class:`~jcm.physics.aerosol.jam.emissions.anthropogenic.EmissionParameters`
            (``scale``, the overall multiplier, and the per-super-sector
            injection profile). ``aerosol_module="jam"`` with
            ``jam_anthropogenic=True`` only.
        oxidants: Override for the prescribed-oxidant
            :class:`~jcm.physics.aerosol.jam.chemistry.oxidants.OxidantParameters`
            (``oh_ref``, ``h2o2_ref_vmr``, ``no3_ref_vmr``). JAM only, as
            ``seasalt``. These proxies are read only when the run supplies no
            oxidant climatology (``forcing.oxidants_file``, ``auto`` by
            default), which supersedes them; the ozone and solar-geometry
            fallbacks (``o3_fallback_vmr``, ``cos_zenith_fallback``) are never
            read in this factory's compositions, and a mapping that sets them
            warns.
        sulfur_gas: Override for the gas-phase sulfur
            :class:`~jcm.physics.aerosol.jam.chemistry.sulfur_gas.SulfurGasParameters`
            (``soag_production``). JAM only.
        aqueous: Override for the in-cloud aqueous sulfur
            :class:`~jcm.physics.aerosol.jam.chemistry.aqueous.AqueousSulfurParameters`
            (``rate_scale``). JAM only.
        activation: Override for the ARG activation
            :class:`~jcm.physics.aerosol.jam.activation.arg_term.ArgParameters`
            (the TKE updraft closure, ``tke_factor`` and ``w_min``; the
            ``updraft_default`` fallback is never read here and warns). JAM
            only.
        cloud_borne_exchange: Override for the interstitial/cloud-borne
            exchange
            :class:`~jcm.physics.aerosol.jam.cloud_borne.CloudBorneExchangeParameters`
            (the activation and resuspension timescales). JAM with
            ``jam_cloud_borne=True`` only: the exchange is not composed
            without the cloud-borne phase.
        sedimentation: Override for the Stokes sedimentation
            :class:`~jcm.physics.aerosol.jam.sedimentation.sedi_term.SedParameters`
            (``velocity_scale``). JAM only.
        drydep: Override for the Slinn surface dry deposition
            :class:`~jcm.physics.aerosol.jam.drydep.drydep_term.DryDepParameters`
            (``z_ref`` and ``z0``; the ``u_star_default`` fallback is never
            read here and warns). JAM only.
        wetdep: Override for the JAM wet-scavenging
            :class:`~jcm.physics.aerosol.jam.wetdep.wetdep_term.WetDepParameters`
            (``incloud_scale``, the multiplier on the stratiform in-cloud
            removal; ``impact_scale``, the multiplier on the inertial
            impaction efficiency of below-cloud removal, stratiform and
            convective; ``sol_factb``, ``mu_water_air``, and the convective
            ``conv_scav_ratio`` and ``conv_updraft_velocity``). JAM only.
            ``incloud_scale`` does not reach the convective in-plume removal,
            which the convective tracer transport owns
            (``jam_convective_transport=True``, the default): scale that with
            ``conv_transport``'s ``conv_scav_scale``. ``conv_scav_ratio`` is
            read only without convective transport (a mapping that sets it
            with convective transport on warns).
        tracer_diffusion: Override for the JAM tracers' turbulent vertical
            mixing
            :class:`~jcm.physics.vertical_diffusion.tracer_diffusion.TracerDiffusionParameters`
            (``diffusion_scale``), for the advected tracers and the
            cloud-borne carry alike. JAM only.
        conv_transport: Override for the JAM tracers' convective transport
            :class:`~jcm.physics.convection.tracer_transport.ConvTransportParameters`
            (``transport_scale``, the multiplier on the mass-flux ledger every
            tracer is moved by, and ``conv_scav_scale``, the multiplier on the
            in-plume scavenging fractions ``csr_conv``, clipped so a fraction
            stays at most one). The per-tracer ``csr_conv`` array itself is
            the mode layout's and is not overridable. JAM with
            ``jam_convective_transport=True`` only.
        gw_scheme: Non-orographic gravity-wave scheme: ``"hines"`` (ECHAM's
            Doppler-spread scheme, the default), ``"frontal"`` (CAM's
            frontogenesis-triggered spectral scheme — requires a
            frontogenesis provider, e.g.
            ``DinosaurDycore(compute_frontogenesis=True)``), ``"both"``
            (Hines background + frontal storm-track deposition; some
            double-counting near strong fronts — retune ``taubgnd`` and
            the Hines source strength jointly if it shows), or ``"none"``.
        checkpoint_terms: Whether to checkpoint each term's compute
            (memory-saving for long backward passes).
        radiation_scheme: ``"rrtmgp"`` (default; the RRTMGP correlated-k
            scheme), ``"emulated"`` (the neural-network emulator of RRTMGP,
            the fast option), or a radiation ``PhysicsTerm`` instance, which
            is composed as given; with a cadence-dependent sibling composed
            (the JAM optics), the instance must expose its
            ``RadiationParameters`` as ``.params`` (an ``nnx.Param``). ``"grey"`` raises ``ValueError``: the grey
            two-stream is an idealized scheme, not ECHAM physics, so it is
            composed only explicitly as
            ``radiation_scheme=GreyTwoStreamRadiation()``.
        emulator_weights_file: ``radiation_scheme="emulated"`` only — the NN
            checkpoint. Case-sensitive value set: ``"auto"`` (default, and what
            an omitted or ``null`` config key resolves to) loads the packaged
            trained weights (``jcm/data/emulator_weights_per_band_u64.nc``), so
            the emulated scheme runs out of the box; the literal string
            ``"random"`` builds RANDOM untrained weights (training-from-scratch
            / zero_tendency cost benchmarks only — they NaN within a step, so
            this must be asked for by name); any other value is a checkpoint
            path. ``None`` is treated as the ``"auto"`` default — the Hydra
            builder drops ``null`` kwargs, so ``physics.emulator_weights_file=
            null`` could not otherwise reach a random init and would silently
            mean ``auto``; use ``"random"`` to train from scratch. Rejected
            (not silently ignored) with any non-emulated scheme.
        radiation_compute_cre: RRTMGP only — run the extra clear-sky solve
            for the cloud-radiative-effect diagnostic (default True).
            ``False`` halves the RRTMGP cost on radiation-compute steps;
            use for production throughput runs that don't analyse CRE.
        cloud_scheme: ``"1m"`` (default, single-moment) or ``"2m"``
            (two-moment warm-rain).
        aerosol_module: ``"macv2sp"`` (default; prescribed simple plumes) or
            ``"jam"`` (online JAM harness — emissions, microphysics core,
            ARG activation, deposition, sedimentation; #461). JAM requires
            ``cloud_scheme="2m"``: its scavenging and resuspension terms
            read the process-time ledger only the 2M scheme publishes,
            and only the 2M scheme consumes JAM's activation and
            ice-nuclei products. The JAM path
            *augments* MACv2-SP rather than replacing it: MACv2-SP is kept for
            the aerosol radiative optics and Twomey factor that radiation and
            the cloud schemes require, while JAM adds the prognostic aerosol
            tracers and an ``activated_cdnc`` that the 2M scheme prefers over
            the SPA floor. The online aerosol *direct radiative* effect that
            would let JAM fully replace MACv2-SP optics is tracked in #495.
        jam_microphysics: JAM core when ``aerosol_module="jam"`` —
            ``"placeholder"`` (κ-Köhler equilibrium on the MAM4 population,
            default) today; ``"mam4_jax"`` is #490 (optional ``jcm[mam4]``
            extra). ``"m7_placeholder"`` is the same κ-Köhler core on the M7
            population instead (the ``echam-ham-m7`` preset's chain-test
            vehicle, #1017) — the real M7 core adapter is a later task.
        jam_cloud_borne: prognose the explicit cloud-borne aerosol phase
            (#602). ``True`` (default) cycles the ``mc_*``/``nc_*`` phase
            in the physics carry (activation transfer, resuspension,
            in-droplet wet removal, dry deposition); ``False`` drops the
            store and scavenges interstitial aerosol by its activated
            fraction instead (the implicit M7/TOMAS-style treatment).
        jam_optics: include the online JAM aerosol direct-effect optics
            (#495), which overwrite the MACv2-SP optics that radiation
            reads. ``False`` keeps MACv2-SP optics (cheaper; also makes
            the JAM aerosol radiatively passive, which controlled A/B
            experiments rely on).
        jam_optics_backend: ``jam_aerosol_physics``'s ``optics_backend`` --
            ``"jcm"`` (default) or ``"ham_lut"`` (ECHAM-HAM M7's own
            Mie-table lookup, #1017). Ignored when ``jam_optics=False``.
        jam_optics_tables_dir: directory holding HAM's authentic
            ``lut_optical_properties_M7.nc``/``lut_optical_properties_lw_
            M7.nc`` for ``jam_optics_backend="ham_lut"``. ``None`` (default)
            reads the ``HAM_INPUT_DIR`` environment variable instead;
            ignored for ``jam_optics_backend="jcm"``.
        jam_seasalt_scheme: ``"gong"`` (default, unchanged) or ``"long"``
            (Long et al. 2011 + the Sofiev et al. 2011 SST correction, HAM
            ``nseasalt=7``; #1017) — passed through to
            :func:`~jcm.physics.aerosol.jam.jam_terms.jam_aerosol_physics`'s
            ``seasalt_scheme``. ``"long"`` requires ``jam_microphysics``'s
            population to carry exactly two ``ss`` classes (HAM's own
            accumulation-then-coarse split); the default MAM4 population
            carries three (it also has an Aitken-mode sea salt tracer, which
            HAM's M7 configuration does not) and so is rejected with
            ``"long"``.
        jam_arg_variant: ``"arg2000"`` (default) or ``"ghosh2025"`` activation.
        jam_activation_scheme: ``jam_aerosol_physics``'s ``activation_scheme``
            -- ``"arg"`` (default), ``"ham_arg"`` or ``"ham_lin_leaitch"``
            (#1017). An ``activation`` mapping applies to the chosen scheme's
            Parameters class.
        jam_nactivpdf: ``jam_aerosol_physics``'s ``nactivpdf`` (HAM's updraft
            PDF switch, ``"ham_arg"`` only; default 0).
        jam_dust_preset: HAMMOZ ``ndust`` preset for the Tegen dust scheme —
            4 (default, HAM2: Stier 2005 + East-Asian soils), 3 (Stier 2005)
            or 2 (Cheng 2008). The resolution-dependent regional tuning vector
            is rebuilt at the model's own truncation.
        jam_dust_nudged: take HAM's *nudged* regional tuning vector
            (0.95/1.25 at T63) instead of the free-running one (1.05/1.45).
            The shipped config leaves this ``null``, which the runner fills
            from ``cfg.nudging.enabled``.
        jam_dust_nduscale_scale: global multiplier on that regional vector —
            jcm's single dust-emission calibration knob (#808). ``null``
            takes the calibrated default, which exists at T63 ``ndust = 4``
            only.
        jam_aqueous_scheme: ``"full"`` (default, HAM port) or ``"simple"``
            (H2O2-limited) in-cloud aqueous sulfur chemistry.
        jam_anthropogenic: include prescribed CEDS anthropogenic emissions
            (#498), the bulk in-model-speciated path; inert until CEDS forcing
            fluxes are supplied.
        jam_prescribed_speciated: include the CAM6/MAM4-faithful already-
            speciated emission path (#498); inert until per-tracer forcing
            fields are supplied.
        convective_updraft_precip_cover: choose the fractional precipitation
            cover ECHAM ``cuflx`` uses for the sub-cloud rain evaporation
            (``mo_cufluxdts.f90:414-420``, jax-gcm#812). ``None`` (default)
            follows ECHAM's ``lham`` submodel dependence: the updraft area
            ``pmfu/(zwu·ρ_u)`` when the JAM chain is composed
            (``aerosol_module='jam'``), the constant ``0.05`` otherwise.
            Set ``True``/``False`` to pin it — the escape hatch for an A/B
            against the constant cover, mirroring ``cu_lmfmid``.

    Returns:
        A ``ComposablePhysics`` instance with all ECHAM terms in the
        validated default order, configured for column vectorisation.

            enable_cosp: Attach the CloudSat satellite-simulator
            diagnostic (``CloudsatCosp``; requires the optional
            jax-cosp dependency, ``pip install jcm[cosp]``). Runs
            after the cloud microphysics and writes the
            ``cosp_*`` warm-rain / precip-cover diagnostics.
        cosp_ncolumns: Stochastic subcolumns per gridbox for the
            radar simulator (COSP canonical value is 100; fewer
            is cheaper and averages out in climatologies).
        diagnose_omega: Publish the dycore's pressure vertical velocity
            [Pa/s] as an ``omega`` output field (needs
            ``DinosaurDycore(compute_omega=True)``; the CLI enables the
            provider automatically). Model-agnostic, independent of the
            AeroCom wap/w500/w700 fields.
        cu_lmfmid: Enable ECHAM's mid-level convection trigger
            (``lmfmid``, default on). ``None`` leaves the trigger as the
            ``convection`` Parameters set it (on by default); ``False``
            disables it. The trigger needs the dycore's ``omega``; a
            backend that cannot supply it (pySES, #698) must set this
            ``False`` or Model construction raises — hence the ne30
            experiments turn it off (#715). Mutually exclusive with a
            ``ConvectionParameters`` object (set the field on that object
            instead) and with ``cu_lmfmid`` in a ``convection`` mapping;
            with a mapping of other fields it sets the base the mapping is
            applied to.
        coords: The model's coordinate system. The schemes whose tunable
            parameters have resolution-dependent defaults (the cloud cover's
            ``CloudParameters``, the 1M ``MicrophysicsParameters``) take the
            defaults for its spectral truncation
            (:mod:`jcm.physics.resolution_defaults`); ``None`` takes the T63
            defaults. An explicit ``Parameters`` object is used as given, and
            a field-override mapping replaces its fields on top of the grid's
            defaults.
        prescribed_surface_fluxes: Forced surface mode (jax-gcm#301):
            compose ``TteTkeVerticalDiffusion(couple_surface=False)``
            (interior-only mixing — the implicit solve's surface Robin BC
            is off) plus a :class:`~jcm.physics.surface.prescribed_flux.
            PrescribedSurfaceFlux` term that delivers the ``prescribed_*``
            fields of the run's ``ForcingData`` as explicit bottom-layer
            fluxes in place of the interactive surface exchange. Units and
            signs follow the surface-exchange coupling contract
            (``docs/source/design/surface_exchange.md``).
        enable_aerocom: Attach the AeroCom phase-4 derived
            diagnostics term (cloud-top sampling, column
            integrals, pressure-level fields, aerosol number
            metrics). Diagnostic-only; adds no tendency. See
            ``jcm.physics.diagnostics.aerocom``.
        aerocom_groups: Which diagnostic groups to compute
            (``cloud``/``column``/``plev``/``aerosol``); a run
            pays only for the groups it selects.
        aerocom_optics: Add the AeroCom per-species / per-mode /
            spectral aerosol optics diagnostics (jax-gcm#584). Requires
            ``aerosol_module="jam"``; a second Mie sweep at the
            observation wavelengths, riding the radiation gate.
        aerocom_overlap: Cloud-overlap hypothesis for the
            cloud-top scan; should match the radiation scheme's.
        cosp_calipso: Also run the CALIPSO lidar simulator on the
            SAME subcolumn realization, giving the CFMIP
            ``cltcalipso``/``cllcalipso``/``clmcalipso``/
            ``clhcalipso`` layered cloud cover.
        cosp_modis: Also run the MODIS imager simulator on that
            realization (``cltmodis``, ``clwmodis``, ``climodis``,
            ``tauwmodis``, ``tauimodis``, ``reffclwmodis``,
            ``reffclimodis``, ``lwpmodis``, ``iwpmodis``, and the
            joint histograms ``clmodis`` / ``jpdftaure*modis`` /
            ``lwpreffmodis`` / ``iwpreffmodis``).
        cosp_isccp: Also run the ISCCP (ICARUS) simulator on that
            realization (``clisccp`` tau/CTP histogram and
            ``cltisccp``); see jax-gcm#597.
        aerosol_free_interval: radiation steps between aerosol-free
            companion solves, which produce the AeroCom ``*noa`` TOA fluxes
            (rsutnoa/rlutnoa and clear-sky variants) that ERFari is
            diagnosed from (jax-gcm#583). RRTMGP only.

            ``None`` (default)
                No ``*noa`` fluxes and no extra cost.
            ``1``
                A SECOND RRTMGP solve per compute step with the aerosol
                optics zeroed. The exact reference. ~+64 % runtime,
                radiation being the dominant cost of a step.
            ``N > 1``
                That companion only every Nth step, holding the aerosol
                EFFECT (as a fraction of the all-sky flux) in between. A
                monotonic cost/fidelity dial: ~+17 % at N=4, for a
                measured ERFari error of ~12 % — though that figure
                predates three fixes to the hold and is a stale upper
                bound (jax-gcm#648).

            The simulation is bit-identical at every N; only the diagnostic
            is approximated. See ``docs/source/design/
            aerocom_erfari_sampling.md``.

    """
    # Checked first so the grey request gets its own explanation rather than
    # a message from whichever later validation it happens to trip.
    if isinstance(radiation_scheme, str) and radiation_scheme == "grey":
        raise ValueError(GREY_RADIATION_REJECTION)

    # Validate for EVERY radiation scheme, not just RRTMGP. The emulated
    # branch and a custom term never construct RRTMGPRadiation here, so
    # leaving this to the term's own constructor would let
    # `echam_physics(radiation_scheme="emulated", aerosol_free_interval=0)`
    # through in silence — exactly the class of silently-ignored argument
    # this knob is meant to abolish.
    resolve_aerosol_free_interval(aerosol_free_interval)
    if aerosol_free_interval is not None and radiation_scheme != "rrtmgp":
        raise ValueError(
            f"aerosol_free_interval={aerosol_free_interval!r} needs "
            "radiation_scheme='rrtmgp' — "
            "the emulated scheme carries no aerosol optics to zero and a "
            "radiation term instance is composed as given, so "
            f"radiation_scheme={radiation_scheme!r} would silently emit "
            "all-zero *noa fluxes.")

    # ``None`` is normalised to the "auto" default: the Hydra builder strips
    # ``null`` kwargs (so ``physics.emulator_weights_file=null`` never reaches
    # here and would fall back to the default anyway), and a direct Python
    # ``None`` must mean the same "unset → packaged weights" as an omitted key.
    # Train-from-scratch (random init) is reached only via the explicit
    # ``"random"`` sentinel below, never by an absent/null value.
    if emulator_weights_file is None:
        emulator_weights_file = "auto"

    # ``emulator_weights_file`` only means anything for the emulator; a value
    # other than the "auto" default paired with another scheme is a silently-
    # ignored argument (the class of bug this factory abolishes — same
    # precedent as aerosol_free_interval above).
    if emulator_weights_file != "auto" and radiation_scheme != "emulated":
        raise ValueError(
            f"emulator_weights_file={emulator_weights_file!r} needs "
            "radiation_scheme='emulated' — any other radiation scheme loads "
            f"no NN checkpoint, so radiation_scheme={radiation_scheme!r} "
            "would ignore it.")

    # Field-override mappings (see the docstring) are set aside here, so the
    # resolution below builds exactly the object this factory would use
    # without them — including its own non-default choices (``cu_lmfmid``,
    # the aerosol-dependent radiation defaults) — and are applied on top of
    # that object once it exists. Applying them to the bare class default
    # instead would let a one-field override silently reset those choices.
    _scheme_args = dict(
        convection=convection, clouds=clouds, microphysics=microphysics,
        microphysics_2m=microphysics_2m, radiation=radiation,
        vertical_diffusion=vertical_diffusion, surface=surface,
        land_surface=land_surface, aerosol=aerosol, hines=hines, sso=sso)
    field_overrides = {name: value for name, value in _scheme_args.items()
                       if isinstance(value, Mapping)}
    (convection, clouds, microphysics, microphysics_2m, radiation,
     vertical_diffusion, surface, land_surface, aerosol, hines, sso) = (
        None if name in field_overrides else value
        for name, value in _scheme_args.items())
    # An override of a scheme that is not composed would be silently
    # ignored, so it is rejected (the factory's rule for every argument).
    _inactive = {
        "microphysics": cloud_scheme != "1m",
        "microphysics_2m": cloud_scheme != "2m",
        "hines": gw_scheme not in ("hines", "both"),
        "radiation": isinstance(radiation_scheme, PhysicsTerm),
        # JAM composes no MACv2-SP and its 2M activation reads ARG, not the
        # SPA knobs, so AerosolParameters is unused there.
        "aerosol": aerosol_module == "jam",
    }
    _ignored = sorted(n for n in field_overrides if _inactive.get(n, False))
    if _ignored:
        raise ValueError(
            f"Field overrides for {_ignored} would be ignored: that scheme is "
            f"not composed (cloud_scheme={cloud_scheme!r}, "
            f"gw_scheme={gw_scheme!r}, aerosol_module={aerosol_module!r}; "
            "radiation overrides need a named "
            "radiation_scheme, a term instance carries its own parameters).")
    # The JAM schemes' parameters (``JAM_PARAMETER_CLASSES``) belong to the JAM
    # chain, which a MACv2-SP composition does not include; the anthropogenic
    # emission and the cloud-borne exchange are composed only with their own
    # flag. Unlike the arguments above they have no earlier behaviour to keep,
    # so a Parameters object is refused as well as a mapping: either would
    # otherwise be dropped without a trace.
    _jam_args = dict(
        seasalt=seasalt, dms=dms, anthropogenic_params=anthropogenic_params,
        oxidants=oxidants, sulfur_gas=sulfur_gas, aqueous=aqueous,
        activation=activation, cloud_borne_exchange=cloud_borne_exchange,
        sedimentation=sedimentation, drydep=drydep, wetdep=wetdep,
        tracer_diffusion=tracer_diffusion, conv_transport=conv_transport)
    _jam_not_composed = {
        "anthropogenic_params": not jam_anthropogenic,
        "cloud_borne_exchange": not jam_cloud_borne,
        "conv_transport": not jam_convective_transport,
    }
    _jam_unused = [
        name for name, value in _jam_args.items() if value is not None
        and (aerosol_module != "jam" or _jam_not_composed.get(name, False))]
    if _jam_unused:
        raise ValueError(
            f"{_jam_unused} would be ignored: that scheme is not composed. "
            "The JAM scheme parameters need aerosol_module='jam' (got "
            f"{aerosol_module!r}); anthropogenic_params also needs "
            f"jam_anthropogenic=True (got {jam_anthropogenic}), "
            "cloud_borne_exchange jam_cloud_borne=True (got "
            f"{jam_cloud_borne}) and conv_transport "
            f"jam_convective_transport=True (got {jam_convective_transport}).")
    if cu_lmfmid is not None and "cu_lmfmid" in field_overrides.get(
            "convection", {}):
        raise ValueError(
            "cu_lmfmid is set both as the scalar flag and in the convection "
            "field overrides — set it in one place only.")

    # ``cu_lmfmid`` is the scalar escape hatch for the ECHAM mid-level
    # convection trigger (ECHAM ``lmfmid``, default on). With it on,
    # TiedtkeConvection declares an ``omega`` dycore requirement; a backend
    # that cannot supply omega — today pySES (#698) — then fails at Model
    # construction, so the ne30 experiments turn it off. It is a scalar flag
    # so a preset can pin it in one line (``physics.cu_lmfmid=false``).
    # Passing it with a ConvectionParameters object is a silently-ignored-
    # argument bug — the class this factory abolishes — so it is rejected;
    # set cu_lmfmid on the object you pass instead. With a field-override
    # mapping the flag chooses the base the mapping is applied to.
    if cu_lmfmid is not None:
        if convection is not None:
            raise ValueError(
                "cu_lmfmid and an explicit convection override are mutually "
                "exclusive — set cu_lmfmid on the ConvectionParameters you "
                "pass as convection=."
            )
        convection_p = ConvectionParameters.default(cu_lmfmid=cu_lmfmid)
    else:
        convection_p = convection or ConvectionParameters.default()
    # Resolution defaults are built here, at construction, for the run's
    # truncation (T63 without a grid), so the parameter pytree the caller
    # gets back is final. An explicit Parameters object is used as given.
    truncation = 63 if coords is None else spectral_truncation(coords)
    clouds_are_defaults = clouds is None
    microphysics_are_defaults = microphysics is None
    clouds_p = clouds or default_parameters(CloudParameters, truncation)
    microphysics_p = microphysics or default_parameters(
        MicrophysicsParameters, truncation)
    microphysics_2m_are_defaults = microphysics_2m is None
    microphysics_2m_p = microphysics_2m or default_parameters(
        CloudParams2M, truncation)
    if isinstance(radiation_scheme, PhysicsTerm):
        # A radiation term instance carries its own parameters; the factory
        # reads them back (below) rather than composing a second, possibly
        # different, set. ``radiation=`` would be silently ignored, so it is
        # rejected — pass the parameters to the term's constructor instead.
        if radiation is not None:
            raise ValueError(
                "radiation= and a radiation_scheme term instance are mutually "
                "exclusive — the instance carries its own RadiationParameters; "
                "pass them to its constructor (params=...).")
        radiation_p = None
    else:
        # An explicit ``radiation=`` wins over the factory default.
        radiation_p = radiation or default_radiation_parameters(aerosol_module)
    vertical_diffusion_p = vertical_diffusion or VDiffParameters.default()
    surface_p = surface or SurfaceParameters.default()
    land_surface_p = land_surface or JsbachLandParameters.default()
    aerosol_p = aerosol or AerosolParameters.default()
    hines_p = hines or HinesParameters.default()
    sso_p = sso or SSOParameters.default()
    jam_p = {}
    if aerosol_module == "jam":
        # Imported here, not at module scope: the JAM package is only worth
        # loading for a JAM composition (as ``jam_aerosol_physics`` below).
        from jcm.physics.aerosol.jam.jam_terms import (
            JAM_PARAMETER_CLASSES, activation_parameter_class)
        _jam_classes = {**JAM_PARAMETER_CLASSES,
                        "activation": activation_parameter_class(
                            jam_activation_scheme)}
        # A mapping is applied on the class default (the JAM schemes have no
        # factory choice of their own to keep); an object is used as given and
        # ``None`` leaves the scheme to build its default.
        jam_p = {
            name: (with_field_overrides(
                       _jam_classes[name].default(), value,
                       scheme=name)
                   if isinstance(value, Mapping) else value)
            for name, value in _jam_args.items()}
    # Fields a valid mapping can set but no run of this composition reads. A
    # sweep over one would run identical arms, so each is flagged, not
    # rejected (the scheme still takes the object, and a configuration that
    # composes differently may read it).
    _jam_inert_fields = {
        ("wetdep", "conv_scav_ratio"): (
            jam_convective_transport,
            "the convective in-cloud scavenging is then the transport term's "
            "per-mode csr_conv (scaled by conv_transport.conv_scav_scale); "
            "this ratio is read only with jam_convective_transport=False"),
        ("oxidants", "o3_fallback_vmr"): (
            True,
            "EchamBoundaryConditions supplies the ozone before the oxidants "
            "in every echam_physics composition, so this fallback for an "
            "absent ozone is never read"),
        ("oxidants", "cos_zenith_fallback"): (
            True,
            "EchamBoundaryConditions supplies the solar geometry before the "
            "oxidants in every echam_physics composition, so this fallback "
            "for an absent one is never read"),
        ("activation", "updraft_default"): (
            True,
            "the TKE-based updraft reads the vertical-diffusion carry, which "
            "every echam_physics composition seeds at initialisation, so "
            "this fallback for its absence is never read"),
        ("drydep", "u_star_default"): (
            True,
            "the surface friction velocity comes from the vertical-diffusion "
            "carry, which every echam_physics composition seeds at "
            "initialisation, so this fallback for its absence is never read"),
    }
    for (_scheme, _field), (_inert, _why) in _jam_inert_fields.items():
        if (_inert and isinstance(_jam_args[_scheme], Mapping)
                and _field in _jam_args[_scheme]):
            warnings.warn(
                f"{_scheme}.{_field} has no effect: {_why}.",
                UserWarning, stacklevel=2)
    if field_overrides:
        _resolved = dict(
            convection=convection_p, clouds=clouds_p,
            microphysics=microphysics_p, microphysics_2m=microphysics_2m_p,
            radiation=radiation_p, vertical_diffusion=vertical_diffusion_p,
            surface=surface_p, land_surface=land_surface_p, aerosol=aerosol_p,
            hines=hines_p, sso=sso_p)
        _resolved.update({
            name: with_field_overrides(_resolved[name], fields, scheme=name)
            for name, fields in field_overrides.items()})
        (convection_p, clouds_p, microphysics_p, microphysics_2m_p,
         radiation_p, vertical_diffusion_p, surface_p, land_surface_p,
         aerosol_p, hines_p, sso_p) = _resolved.values()

    if cloud_scheme in ("1m", "2m"):
        _warn_if_shared_cloud_constants_differ(
            clouds_p, microphysics_p if cloud_scheme == "1m" else microphysics_2m_p,
            cloud_scheme)

    if isinstance(radiation_scheme, PhysicsTerm):
        if radiation_scheme.category != "radiation":
            raise ValueError(
                "Custom radiation_scheme terms must have category "
                "'radiation'."
            )
        rad_term = radiation_scheme
        # Siblings that follow the radiation cadence (the JAM optics gate)
        # must read it from the instance itself. Every in-tree radiation term
        # holds its RadiationParameters as ``self.params``; a term that does
        # not leaves ``radiation_p`` None, which is an error only where a
        # sibling needs it (below), never a silent default.
        _held = getattr(rad_term, "params", None)
        _held = _held.get_value() if hasattr(_held, "get_value") else _held
        radiation_p = (_held if isinstance(_held, RadiationParameters)
                       else None)
    elif radiation_scheme == "rrtmgp":
        # compute_cre doubles the RRTMGP work on radiation steps (a second
        # full clear-sky solve) purely for the CRE diagnostic — production
        # throughput runs can turn it off.
        rad_term = RRTMGPRadiation(
            params=radiation_p,
            compute_cre=radiation_compute_cre,
            aerosol_free_interval=aerosol_free_interval)
    elif radiation_scheme == "emulated":
        # "auto" (default) resolves the packaged trained checkpoint
        # (jcm/data/emulator_weights_per_band_u64.nc). The explicit "random"
        # sentinel maps to weights_file=None, which builds RANDOM untrained
        # weights — the term's own docs note these drive the model to NaN
        # within a step, so this is training-from-scratch / a zero_tendency
        # cost benchmark only, never a valid simulation, and must be requested
        # by name (an absent/null config value means "auto", see above). Any
        # other value is a checkpoint path validated by the term at load.
        weights_file = (None if emulator_weights_file == "random"
                        else emulator_weights_file)
        rad_term = NNEmulatorRadiation(
            params=radiation_p, weights_file=weights_file)
    else:
        raise ValueError(
            f"Unknown radiation_scheme={radiation_scheme!r}. "
            "Choose 'rrtmgp' (default), 'emulated', or pass a radiation "
            "PhysicsTerm instance."
        )
    # Aerosol and cloud optics need the same band metadata as the selected
    # radiation term, so Python-created RRTMGP compositions must carry the
    # multi-band config just like the Hydra runner path — both resolve it
    # through RadiationBandConfig.for_terms. The emulator is included: its
    # per-band features expect the RRTMGP band structure, and a broadband
    # 1-SW/0-LW aerosol layout fails its band-count check at first compute.
    band_config = RadiationBandConfig.for_terms([rad_term])

    if cloud_scheme == "1m":
        micro_term = Echam1MMicrophysics(
            params=microphysics_p,
            **defaults_flag_kwargs(
                Echam1MMicrophysics, microphysics_are_defaults))
    elif cloud_scheme == "2m":
        micro_term = Lohmann2MMicrophysics(
            params=microphysics_2m_p,
            **defaults_flag_kwargs(
                Lohmann2MMicrophysics, microphysics_2m_are_defaults))
        # SPA activation knobs live on AerosolParameters — wire them into
        # the 2M term so it stays self-contained at compose time. Pass the
        # values through untouched (no float() cast) so the gradient path
        # from AerosolParameters to the 2M activation stays intact.
        micro_term.configure_spa(
            aerosol_p.spa_prefactor,
            aerosol_p.spa_exponent,
            aerosol_p.spa_cap_smoothing,
        )
    else:
        raise ValueError(
            f"Unknown cloud_scheme={cloud_scheme!r}. Choose '1m' or '2m'."
        )

    # MACv2-SP and JAM are mutually exclusive aerosol sources (#640), so the
    # JAM composition does not include MACv2-SP. JAM owns the ``aerosol`` slot
    # through its own
    # ``AerosolCarrySeeder`` (radiatively passive when ``jam_optics=False``,
    # overwritten by ``JamOpticsTerm`` when on) and supplies the cloud
    # microphysics activation through ``ArgActivation``; the 2M scheme falls
    # back to its ECHAM-HAM ``cdnc_min`` floor where ARG is empty rather than to
    # the (now absent) MACv2-SP SPA floor.
    #
    # Wet deposition is split out and placed *after* the cloud microphysics
    # term so it scavenges against the current step's precip/condensate (the
    # rest of the JAM chain runs in the pre-cloud aerosol block — activation
    # must precede the cloud term that consumes ``activated_cdnc``).
    jam_post_cloud_terms: list[PhysicsTerm] = []
    # The per-species/per-mode split only exists for a modal scheme with
    # explicit species tracers. Asking for it with MACv2-SP (prescribed
    # plumes, no species) or with the aerosol module off would otherwise
    # produce an output file silently missing the very fields the run was
    # configured to get.
    if aerocom_optics and aerosol_module != "jam":
        raise ValueError(
            "aerocom_optics=True needs aerosol_module='jam' — the per-species "
            "and per-mode optics come from the JAM modal population, which "
            f"aerosol_module={aerosol_module!r} does not carry."
        )
    if aerosol_module == "macv2sp":
        aerosol_terms = [Macv2SpAerosol(params=aerosol_p)]
    elif aerosol_module == "jam":
        if cloud_scheme != "2m":
            # The JAM wet-deposition and cloud-borne exchange terms key to
            # the process-time scavenging ledger only the 2M scheme
            # publishes on CloudData (#708), and prognostic aerosol without
            # aerosol-aware microphysics is not a configuration we support:
            # the 1M scheme would read none of JAM's activation/ice-nuclei
            # products while the ledger fields stayed all-zero, silently
            # producing no stratiform in-cloud scavenging.
            raise ValueError(
                "aerosol_module='jam' requires cloud_scheme='2m' — the JAM "
                "scavenging and resuspension terms read the 2M scheme's "
                f"process-time ledger, and cloud_scheme={cloud_scheme!r} "
                "does not publish it."
            )
        from jcm.physics.aerosol.jam.jam_terms import jam_aerosol_physics
        jam_terms = jam_aerosol_physics(
            microphysics=jam_microphysics, cloud_borne=jam_cloud_borne,
            optics=jam_optics, optics_backend=jam_optics_backend,
            ham_optics_tables_dir=jam_optics_tables_dir,
            seasalt_scheme=jam_seasalt_scheme,
            arg_variant=jam_arg_variant,
            activation_scheme=jam_activation_scheme,
            nactivpdf=jam_nactivpdf,
            aqueous_scheme=jam_aqueous_scheme,
            dust_preset=jam_dust_preset,
            dust_nudged=jam_dust_nudged,
            dust_nduscale_scale=jam_dust_nduscale_scale,
            **jam_p,
            anthropogenic=jam_anthropogenic,
            prescribed_speciated=jam_prescribed_speciated,
            convective_transport=jam_convective_transport,
            optics_diagnostics=aerocom_optics,
        )
        # The per-band Mie optics are only consumed by the interval-gated
        # radiation term, so the optics term skips recomputing them on the
        # steps where radiation replays its cache (see
        # ``JamOpticsTerm.configure_radiation_gate``) — same post-compose
        # configuration pattern as ``configure_spa`` below.
        # Only a composed gated term needs the cadence: with
        # ``jam_optics=False`` there is none, and a radiation term that
        # exposes no parameters is as valid as in any other composition.
        _gated = [_t for _t in jam_terms
                  if hasattr(_t, "configure_radiation_gate")]
        if _gated and radiation_p is None:
            raise ValueError(
                f"The JAM optics follow the radiation cadence, but the "
                f"radiation term {type(rad_term).__name__} "
                "exposes no RadiationParameters: the factory reads "
                "`<term>.params` (an nnx.Param holding RadiationParameters, "
                "whose radiation_interval it uses).")
        for _t in _gated:
            _t.configure_radiation_gate(radiation_p.radiation_interval)
        # Aqueous chemistry + wet deposition need the current step's clouds, so
        # they run after the cloud microphysics term; the rest of the JAM chain
        # is the pre-cloud aerosol block.
        # Cloud-borne exchange sits with the other cloud-consuming terms:
        # it needs the current step's cloud fraction, and must precede the
        # aqueous split and wet scavenging (order preserved from jam_terms).
        _post_cloud = (
            "aerosol_cloud_borne", "aerosol_aqueous_chemistry",
            "aerosol_wetdep",
        )
        jam_post_cloud_terms = [
            t for t in jam_terms if t.category in _post_cloud
        ]
        jam_pre_cloud_terms = [
            t for t in jam_terms if t.category not in _post_cloud
        ]
        # No MACv2-SP in the JAM path (#640): ``jam_pre_cloud_terms`` already
        # begins with the ``AerosolCarrySeeder`` that owns the ``aerosol`` slot.
        aerosol_terms = [*jam_pre_cloud_terms]
    else:
        raise ValueError(
            f"Unknown aerosol_module={aerosol_module!r}. Choose 'macv2sp' "
            "or 'jam'."
        )

    # Term ordering follows ECHAM's ``physc`` process sequence:
    # radheat -> vdiff -> (gwdrag) -> cucall -> cloud. In particular the
    # TTE-TKE vertical diffusion (which carries the surface exchange as
    # the bottom row of its implicit solve) runs BEFORE Tiedtke
    # convection, so convection sees the SAME-STEP vdiff moisture
    # tendency (ECHAM's ``pqte`` at cucall time) for the zdqpbl closure
    # supply and the same-step delivered surface evaporation. Running
    # convection first forced a one-step-lagged supply, which let the
    # convergence->convection->convergence feedback compound (heating
    # pinned at the stability cap, then NaN — onset7 analysis); ECHAM's
    # same-step pqte self-limits because convection consumes exactly
    # what converged this step. ``EchamSurface`` only republishes the
    # vdiff-delivered fluxes, so it sits immediately after vdiff.
    # Deviation from ECHAM: gravity-wave drag (Hines + SSO) stays after
    # the moist physics rather than between vdiff and cucall — it feeds
    # nothing that convection/cloud read same-step, and moving it is an
    # independent change we keep out of this reordering.
    if gw_scheme == "hines":
        nonoro_gw_terms: list[PhysicsTerm] = [HinesGwd(params=hines_p)]
    elif gw_scheme == "frontal":
        from jcm.physics.gravity_waves.spectral.term import (
            FrontalGravityWaveDrag,
        )
        nonoro_gw_terms = [FrontalGravityWaveDrag()]
    elif gw_scheme == "both":
        # Hines carries the broad-spectrum background (the role CAM fills
        # with its convective + ridge sources, which are not ported);
        # frontal adds the storm-track/vortex-edge deposition. Replacing
        # Hines with frontal-only under-drags the subtropical jet (+15-25
        # m/s by day 60 in the v4 ne30 year, blowing up at the NH spring
        # transition) — the frontal source only launches where fronts
        # exceed frontgfc. Some overlap double-counting near strong fronts
        # is accepted; retune taubgnd/rms_launch_wind jointly if it shows.
        from jcm.physics.gravity_waves.spectral.term import (
            FrontalGravityWaveDrag,
        )
        nonoro_gw_terms = [HinesGwd(params=hines_p),
                           FrontalGravityWaveDrag()]
    elif gw_scheme == "none":
        nonoro_gw_terms = []
    else:
        raise ValueError(
            f"gw_scheme={gw_scheme!r} not in "
            "('hines', 'frontal', 'both', 'none')")

    cosp_terms: list[PhysicsTerm] = []
    if enable_cosp:
        from jcm.physics.diagnostics.cosp_cloudsat import CloudsatCosp
        cosp_terms = [CloudsatCosp(ncolumns=cosp_ncolumns,
                                   enable_calipso=cosp_calipso,
                                   enable_modis=cosp_modis,
                                   enable_isccp=cosp_isccp)]

    omega_terms: list[PhysicsTerm] = []
    if diagnose_omega:
        from jcm.physics.diagnostics.omega import OmegaDiagnostic
        omega_terms = [OmegaDiagnostic()]

    aerocom_terms: list[PhysicsTerm] = []
    if enable_aerocom:
        from jcm.physics.diagnostics.aerocom import AerocomDiagnostics
        aerocom_terms = [AerocomDiagnostics(
            groups=tuple(aerocom_groups), overlap=aerocom_overlap)]

    # Forced surface mode (#301): the vdiff implicit solve runs
    # interior-only (its surface Robin BC off) and the prescribed fluxes
    # are delivered explicitly by the PrescribedSurfaceFlux term sitting
    # exactly where the interactive delivery happened — between vdiff and
    # EchamSurface, so the surface term (and Tiedtke's moisture-budget
    # closure behind it) republishes the prescribed values same-step.
    prescribed_terms: list[PhysicsTerm] = (
        [PrescribedSurfaceFlux()] if prescribed_surface_fluxes else []
    )

    return ComposablePhysics(
        terms=[
            MoistAirColumnState(),
            EchamBoundaryConditions(),
            *aerosol_terms,
            SimpleChemistry(),
            SundqvistCloudFraction(
                params=clouds_p, params_are_defaults=clouds_are_defaults),
            rad_term,
            TteTkeVerticalDiffusion(
                params=vertical_diffusion_p,
                land_params=land_surface_p,
                couple_surface=not prescribed_surface_fluxes,
            ),
            *prescribed_terms,
            EchamSurface(params=surface_p),
            TiedtkeConvection(
                params=convection_p,
                # ECHAM keys the sub-cloud rain-evaporation footprint on the
                # HAM submodel (mo_cufluxdts.f90:414-420): the updraft area
                # under ``lham``, the constant 0.05 otherwise. jcm's ``lham``
                # is "the JAM aerosol chain is composed" (jax-gcm#812); the
                # explicit override pins it for an A/B.
                updraft_precip_cover=(
                    (aerosol_module == "jam")
                    if convective_updraft_precip_cover is None
                    else bool(convective_updraft_precip_cover)
                ),
            ),
            micro_term,
            # Publishes the package-independent surface-exchange coupling
            # struct (#754). After the microphysics so the stratiform
            # precipitation it reads is the SAME step's.
            EchamSurfaceExchange(),
            *cosp_terms,
            *jam_post_cloud_terms,
            *nonoro_gw_terms,
            LottMillerSso(params=sso_p),
            *omega_terms,
            # Terminal: summarises the completed step, so it runs after
            # every term that can still modify the state.
            *aerocom_terms,
        ],
        checkpoint_terms=checkpoint_terms,
        vectorize_columns=True,
        band_config=band_config,
    )
