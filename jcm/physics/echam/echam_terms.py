"""``echam_physics()`` factory.

Every ECHAM parameterisation lives as a ``PhysicsTerm`` next to its
underlying numerical implementation (``TiedtkeConvection``,
``SundqvistCloudFraction``, ``Echam1MMicrophysics``,
``GreyTwoStreamRadiation``, …) and owns its own scheme-native
``Parameters``. This module is the user-facing factory that wires the
scheme-named terms together in a validated default ordering and returns
a ready-to-run ``ComposablePhysics`` with column vectorisation enabled.

The factory accepts per-scheme ``Parameters`` objects directly — there
is no monolithic ECHAM ``Parameters`` aggregator. Each unspecified
sub-Parameters falls through to its scheme's ``.default()`` constructor,
so callers only have to pass the knobs they want to tune.
"""

from __future__ import annotations

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
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.radiation.rrtmgp import RRTMGPRadiation, _ensure_rrtmgp
from jcm.physics.surface.echam.surface_physics import EchamSurface
from jcm.physics.surface.echam.surface_types import SurfaceParameters
from jcm.physics.vertical_diffusion.tte_tke import TteTkeVerticalDiffusion
from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
    VDiffParameters,
)


def echam_physics(
    *,
    convection: ConvectionParameters | None = None,
    clouds: CloudParameters | None = None,
    microphysics: MicrophysicsParameters | None = None,
    microphysics_2m: CloudParams2M | None = None,
    radiation: RadiationParameters | None = None,
    vertical_diffusion: VDiffParameters | None = None,
    surface: SurfaceParameters | None = None,
    aerosol: AerosolParameters | None = None,
    hines: HinesParameters | None = None,
    sso: SSOParameters | None = None,
    checkpoint_terms: bool = True,
    radiation_scheme: str | PhysicsTerm = "grey",
    cloud_scheme: str = "1m",
    aerosol_module: str = "macv2sp",
    jam_microphysics: str = "placeholder",
    jam_arg_variant: str = "arg2000",
    jam_aqueous_scheme: str = "full",
    jam_ice_scheme: str = "niemand",
    jam_anthropogenic: bool = False,
    jam_prescribed_speciated: bool = False,
):
    """Create a ``ComposablePhysics`` with the standard ECHAM term ordering.

    Each per-scheme ``Parameters`` object is optional; ``None`` resolves
    to the scheme's ``.default()``. There is no monolithic aggregator —
    the composition assembled here is the only place where the ECHAM
    stack's per-scheme parameters meet.

    Args:
        convection: Override for ``ConvectionParameters``.
        clouds: Override for the diagnostic cloud-fraction
            ``CloudParameters``.
        microphysics: Override for 1-moment microphysics
            ``MicrophysicsParameters`` (used when ``cloud_scheme="1m"``).
        microphysics_2m: Override for 2-moment microphysics
            ``CloudParams2M`` (used when ``cloud_scheme="2m"``).
        radiation: Override for ``RadiationParameters`` (shared by all
            three radiation backends).
        vertical_diffusion: Override for TTE-TKE ``VDiffParameters``.
        surface: Override for ``SurfaceParameters``.
        aerosol: Override for MACv2-SP ``AerosolParameters``. Also
            supplies the SPA activation knobs read by the 2M scheme
            when ``cloud_scheme="2m"``.
        hines: Override for non-orographic GW ``HinesParameters``.
        sso: Override for sub-grid-scale orography ``SSOParameters``.
        checkpoint_terms: Whether to checkpoint each term's compute
            (memory-saving for long backward passes).
        radiation_scheme: ``"grey"`` (default), ``"rrtmgp"``,
            ``"emulated"``, or a custom radiation ``PhysicsTerm``.
        cloud_scheme: ``"1m"`` (default, single-moment) or ``"2m"``
            (two-moment warm-rain).
        aerosol_module: ``"macv2sp"`` (default; prescribed simple plumes) or
            ``"jam"`` (online JAM harness — emissions, microphysics core,
            ARG activation, deposition, sedimentation; #461). The JAM path
            *augments* MACv2-SP rather than replacing it: MACv2-SP is kept for
            the aerosol radiative optics and Twomey factor that radiation and
            the cloud schemes require, while JAM adds the prognostic aerosol
            tracers and an ``activated_cdnc`` that the 2M scheme prefers over
            the SPA floor. The online aerosol *direct radiative* effect that
            would let JAM fully replace MACv2-SP optics is tracked in #495.
        jam_microphysics: JAM core when ``aerosol_module="jam"`` —
            ``"placeholder"`` (κ-Köhler equilibrium) today; MAM4-JAX is #490.
        jam_arg_variant: ``"arg2000"`` (default) or ``"ghosh2025"`` activation.
        jam_ice_scheme: heterogeneous ice nucleation scheme — ``"niemand"``
            (default) or ``"lohmann_diehl"`` (drives the 2M ICNC).
        jam_aqueous_scheme: ``"full"`` (default, HAM port) or ``"simple"``
            (H2O2-limited) in-cloud aqueous sulfur chemistry.
        jam_anthropogenic: include prescribed CEDS anthropogenic emissions
            (#498), the bulk in-model-speciated path; inert until CEDS forcing
            fluxes are supplied.
        jam_prescribed_speciated: include the CAM6/MAM4-faithful already-
            speciated emission path (#498); inert until per-tracer forcing
            fields are supplied.

    Returns:
        A ``ComposablePhysics`` instance with all ECHAM terms in the
        validated default order, configured for column vectorisation.

    """
    convection_p = convection or ConvectionParameters.default()
    clouds_p = clouds or CloudParameters.default()
    microphysics_p = microphysics or MicrophysicsParameters.default()
    microphysics_2m_p = microphysics_2m or CloudParams2M.default()
    radiation_p = radiation or RadiationParameters.default()
    vertical_diffusion_p = vertical_diffusion or VDiffParameters.default()
    surface_p = surface or SurfaceParameters.default()
    aerosol_p = aerosol or AerosolParameters.default()
    hines_p = hines or HinesParameters.default()
    sso_p = sso or SSOParameters.default()

    if isinstance(radiation_scheme, PhysicsTerm):
        if radiation_scheme.category != "radiation":
            raise ValueError(
                "Custom radiation_scheme terms must have category "
                "'radiation'."
            )
        rad_term = radiation_scheme
    elif radiation_scheme == "rrtmgp":
        rad_term = RRTMGPRadiation(params=radiation_p)
    elif radiation_scheme == "grey":
        rad_term = GreyTwoStreamRadiation(params=radiation_p)
    elif radiation_scheme == "emulated":
        rad_term = NNEmulatorRadiation(params=radiation_p)
    else:
        raise ValueError(
            f"Unknown radiation_scheme={radiation_scheme!r}. "
            "Choose 'grey', 'rrtmgp', 'emulated', or pass a radiation "
            "PhysicsTerm."
        )
    # Aerosol and cloud optics need the same band metadata as the selected
    # radiation term, so Python-created RRTMGP compositions must carry the
    # multi-band config just like the Hydra runner path.
    band_config = (
        RadiationBandConfig.from_rrtmgp(_ensure_rrtmgp())
        if isinstance(rad_term, RRTMGPRadiation)
        else RadiationBandConfig.broadband()
    )

    if cloud_scheme == "1m":
        micro_term = Echam1MMicrophysics(params=microphysics_p)
    elif cloud_scheme == "2m":
        micro_term = Lohmann2MMicrophysics(params=microphysics_2m_p)
        # SPA activation knobs live on AerosolParameters — wire them into
        # the 2M term so it stays self-contained at compose time. Pass the
        # values through untouched (no float() cast) so the gradient path
        # from AerosolParameters to the 2M activation stays intact.
        micro_term.configure_spa(
            aerosol_p.spa_prefactor,
            aerosol_p.spa_exponent,
        )
    else:
        raise ValueError(
            f"Unknown cloud_scheme={cloud_scheme!r}. Choose '1m' or '2m'."
        )

    # For now MACv2-SP is kept alongside JAM to provide the ``aerosol``
    # optics/Twomey diagnostic that radiation and the cloud schemes read — a
    # temporary fudge in lieu of proper JAM aerosol↔radiation and
    # aerosol↔microphysics coupling (#495). Once JAM supplies those, MACv2-SP
    # need not be included in the JAM path.
    #
    # Wet deposition is split out and placed *after* the cloud microphysics
    # term so it scavenges against the current step's precip/condensate (the
    # rest of the JAM chain runs in the pre-cloud aerosol block — activation
    # must precede the cloud term that consumes ``activated_cdnc``).
    jam_post_cloud_terms: list[PhysicsTerm] = []
    if aerosol_module == "macv2sp":
        aerosol_terms = [Macv2SpAerosol(params=aerosol_p)]
    elif aerosol_module == "jam":
        from jcm.physics.aerosol.jam.jam_terms import jam_aerosol_physics
        jam_terms = jam_aerosol_physics(
            microphysics=jam_microphysics, arg_variant=jam_arg_variant,
            aqueous_scheme=jam_aqueous_scheme,
            ice_scheme=jam_ice_scheme,
            anthropogenic=jam_anthropogenic,
            prescribed_speciated=jam_prescribed_speciated,
        )
        # Aqueous chemistry + wet deposition need the current step's clouds, so
        # they run after the cloud microphysics term; the rest of the JAM chain
        # is the pre-cloud aerosol block.
        _post_cloud = ("aerosol_wetdep", "aerosol_aqueous_chemistry")
        jam_post_cloud_terms = [
            t for t in jam_terms if t.category in _post_cloud
        ]
        jam_pre_cloud_terms = [
            t for t in jam_terms if t.category not in _post_cloud
        ]
        aerosol_terms = [Macv2SpAerosol(params=aerosol_p), *jam_pre_cloud_terms]
    else:
        raise ValueError(
            f"Unknown aerosol_module={aerosol_module!r}. Choose 'macv2sp' "
            "or 'jam'."
        )

    return ComposablePhysics(
        terms=[
            MoistAirColumnState(),
            EchamBoundaryConditions(),
            *aerosol_terms,
            SimpleChemistry(),
            SundqvistCloudFraction(params=clouds_p),
            rad_term,
            TiedtkeConvection(params=convection_p),
            micro_term,
            *jam_post_cloud_terms,
            TteTkeVerticalDiffusion(params=vertical_diffusion_p),
            EchamSurface(params=surface_p),
            HinesGwd(params=hines_p),
            LottMillerSso(params=sso_p),
        ],
        checkpoint_terms=checkpoint_terms,
        vectorize_columns=True,
        band_config=band_config,
    )
