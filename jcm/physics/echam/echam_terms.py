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
from jcm.physics.radiation.aerosol_free import (
    resolve_aerosol_free_interval,
)
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
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
    gw_scheme: str = "hines",
    checkpoint_terms: bool = True,
    radiation_scheme: str | PhysicsTerm = "grey",
    emulator_weights_file: str | None = "auto",
    radiation_compute_cre: bool = True,
    cloud_scheme: str = "1m",
    aerosol_module: str = "macv2sp",
    jam_microphysics: str = "placeholder",
    jam_cloud_borne: bool = True,
    jam_optics: bool = True,
    jam_arg_variant: str = "arg2000",
    jam_aqueous_scheme: str = "full",
    jam_ice_scheme: str = "niemand",
    jam_anthropogenic: bool = False,
    jam_prescribed_speciated: bool = False,
    jam_convective_transport: bool = True,
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
        radiation_scheme: ``"grey"`` (default), ``"rrtmgp"``,
            ``"emulated"``, or a custom radiation ``PhysicsTerm``.
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
            ``"placeholder"`` (κ-Köhler equilibrium) today; MAM4-JAX is #490.
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
    # Validate for EVERY radiation scheme, not just RRTMGP. The grey and
    # emulated branches never construct RRTMGPRadiation, so leaving this to
    # the term's own constructor let
    # `echam_physics(radiation_scheme="grey", aerosol_free_interval=0)`
    # through in silence — exactly the class of silently-ignored argument
    # this knob is meant to abolish.
    resolve_aerosol_free_interval(aerosol_free_interval)
    if aerosol_free_interval is not None and radiation_scheme != "rrtmgp":
        raise ValueError(
            f"aerosol_free_interval={aerosol_free_interval!r} needs "
            "radiation_scheme='rrtmgp' — "
            "the grey and emulated schemes carry no aerosol optics to zero, "
            f"so radiation_scheme={radiation_scheme!r} would silently emit "
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
            "radiation_scheme='emulated' — the grey and rrtmgp schemes load no "
            f"NN checkpoint, so radiation_scheme={radiation_scheme!r} would "
            "ignore it.")

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
        # compute_cre doubles the RRTMGP work on radiation steps (a second
        # full clear-sky solve) purely for the CRE diagnostic — production
        # throughput runs can turn it off.
        rad_term = RRTMGPRadiation(
            params=radiation_p,
            compute_cre=radiation_compute_cre,
            aerosol_free_interval=aerosol_free_interval)
    elif radiation_scheme == "grey":
        rad_term = GreyTwoStreamRadiation(params=radiation_p)
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
            "Choose 'grey', 'rrtmgp', 'emulated', or pass a radiation "
            "PhysicsTerm."
        )
    # Aerosol and cloud optics need the same band metadata as the selected
    # radiation term, so Python-created RRTMGP compositions must carry the
    # multi-band config just like the Hydra runner path — both resolve it
    # through RadiationBandConfig.for_terms. The emulator is included: its
    # per-band features expect the RRTMGP band structure, and a broadband
    # 1-SW/0-LW aerosol layout fails its band-count check at first compute.
    band_config = RadiationBandConfig.for_terms([rad_term])

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
            aerosol_p.spa_cap_smoothing,
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
            optics=jam_optics,
            arg_variant=jam_arg_variant,
            aqueous_scheme=jam_aqueous_scheme,
            ice_scheme=jam_ice_scheme,
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
        for _t in jam_terms:
            if hasattr(_t, "configure_radiation_gate"):
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
        aerosol_terms = [Macv2SpAerosol(params=aerosol_p), *jam_pre_cloud_terms]
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

    return ComposablePhysics(
        terms=[
            MoistAirColumnState(),
            EchamBoundaryConditions(),
            *aerosol_terms,
            SimpleChemistry(),
            SundqvistCloudFraction(params=clouds_p),
            rad_term,
            TteTkeVerticalDiffusion(params=vertical_diffusion_p),
            EchamSurface(params=surface_p),
            TiedtkeConvection(params=convection_p),
            micro_term,
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
