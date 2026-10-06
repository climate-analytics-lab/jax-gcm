"""``jam_aerosol_physics()`` factory — the ordered JAM harness term list.

Returns the HAMMOZ-style process chain (emissions → microphysics core →
activation → sedimentation → dry deposition → cloud-borne exchange →
aqueous chemistry → wet deposition) as a list of
``PhysicsTerm``s, ready to splice into ``echam_physics``. The microphysics
core is the swap point: pass ``"placeholder"`` (default κ-Köhler equilibrium)
or any ``ModalMicrophysicsTerm`` instance (e.g. a future MAM4-JAX wrapper,
#490). Every harness term is handed the core's population so they all agree
on mode/species layout.
"""

from __future__ import annotations

import dataclasses

from jcm.physics.aerosol.carry_seeder import AerosolCarrySeeder
from jcm.physics.aerosol.jam.activation.arg_term import (
    ArgActivation,
    ArgParameters,
)
from jcm.physics.aerosol.jam.cloud_borne import (
    CloudBorneExchange,
    CloudBorneExchangeParameters,
)
from jcm.physics.aerosol.jam.cloud_borne_store import (
    CloudBorneCarryStore,
    carry_mode,
)
from jcm.physics.aerosol.jam.chemistry.aqueous import (
    AqueousSulfur,
    AqueousSulfurParameters,
)
from jcm.physics.aerosol.jam.chemistry.oxidants import (
    OxidantParameters,
    PrescribedOxidants,
)
from jcm.physics.aerosol.jam.chemistry.sulfur_gas import (
    SulfurGasChemistry,
    SulfurGasParameters,
)
from jcm.physics.aerosol.jam.drydep.drydep_term import (
    DryDepParameters,
    SlinnDryDeposition,
)
from jcm.physics.aerosol.jam.emissions.anthropogenic import (
    AnthropogenicEmissions,
    EmissionParameters,
)
from jcm.physics.aerosol.jam.emissions.prescribed import PreSpeciatedEmissions
from jcm.physics.aerosol.jam.emissions.dms import DmsEmissions, DmsParameters
from jcm.physics.aerosol.jam.emissions.dust import DustEmissions, DustParameters
from jcm.physics.aerosol.jam.emissions.seasalt import (
    SeaSaltEmissions,
    SeaSaltParameters,
)
from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import (
    HamFreezingClasses,
)
from jcm.physics.aerosol.jam.ice_nucleation.ice_term import IceNucleation
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.optics.ham_lut_optics_term import HamLutOpticsTerm
from jcm.physics.aerosol.jam.sedimentation.sedi_term import (
    StokesSedimentation,
    SedParameters,
)
from jcm.physics.aerosol.jam.wetdep.convective_fractions import convective_csr
from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
)
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.convection.tracer_transport import (
    ConvectiveTracerTransport,
    ConvTransportParameters,
)
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.vertical_diffusion.tracer_diffusion import (
    TracerDiffusionParameters,
    TracerVerticalDiffusion,
)

#: ``jam_aerosol_physics`` keyword -> the ``Parameters`` class of that scheme,
#: for every scheme a field-override mapping can be applied to. It is the one
#: list the factory door of ``echam_physics`` builds on: a mapping
#: (``wetdep={"incloud_scale": 0.5}``, Hydra's
#: ``+physics.wetdep.incloud_scale=0.5``) is applied on top of the class's
#: ``.default()``, and an object is forwarded as given.
#: ``ConvTransportParameters.default()`` leaves the per-tracer ``csr_conv``
#: empty, which the transport term fills from the mode layout below.
#:
#: ``dust`` is not here because its default depends on the composition rather
#: than on the class: the dust preset is rebuilt at the model's own truncation
#: in ``DustEmissions.cache_coords`` (an explicit object is never rebuilt). A
#: mapping for it has to be applied there, so ``echam_physics`` has no mapping
#: door for it beyond the ``jam_dust_*`` flags yet (jax-gcm#995).
JAM_PARAMETER_CLASSES = {
    "seasalt": SeaSaltParameters,
    "dms": DmsParameters,
    "anthropogenic_params": EmissionParameters,
    "oxidants": OxidantParameters,
    "sulfur_gas": SulfurGasParameters,
    "aqueous": AqueousSulfurParameters,
    "activation": ArgParameters,
    "cloud_borne_exchange": CloudBorneExchangeParameters,
    "sedimentation": SedParameters,
    "drydep": DryDepParameters,
    "wetdep": WetDepParameters,
    "tracer_diffusion": TracerDiffusionParameters,
    "conv_transport": ConvTransportParameters,
}


def _load_mam4_jax() -> type[ModalMicrophysicsTerm]:
    """Import the MAM4-JAX core lazily (optional GPL-3.0 dependency)."""
    from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
        Mam4JaxMicrophysics,
    )

    return Mam4JaxMicrophysics


def _load_m7_jax() -> type[ModalMicrophysicsTerm]:
    """Import the M7-JAX core lazily (optional ``jcm[m7]`` dependency)."""
    from jcm.physics.aerosol.jam.microphysics.m7_jax import M7JaxMicrophysics

    return M7JaxMicrophysics


# Core resolvers (each takes a spec override, ``None`` for the core default).
# ``placeholder``/``m7_placeholder`` are built-in; ``mam4_jax`` is loaded
# lazily so the optional GPL-3.0 ``mam4-jax`` dependency is only imported
# when selected.
_MICROPHYSICS = {
    "placeholder": lambda spec: PlaceholderMicrophysics(spec=spec),
    "mam4_jax": lambda spec: _load_mam4_jax()(spec=spec),
    # The κ-Köhler zero-tendency core on the M7 population (jax-gcm#1017) —
    # the chain-test vehicle for the echam-ham-m7 preset until the real M7
    # core adapter lands. ``spec`` defaults to M7_SPEC rather than
    # PlaceholderMicrophysics's own MAM4_SPEC default.
    "m7_placeholder": lambda spec: PlaceholderMicrophysics(
        spec=spec or M7_SPEC),
    # The ECHAM-HAM M7 core over m7-jax (jax-gcm#1017), the optional
    # ``jcm[m7]`` extra; loaded lazily like mam4_jax.
    "m7_jax": lambda spec: _load_m7_jax()(spec=spec),
}


def _resolve_microphysics(
    microphysics: ModalMicrophysicsTerm | str,
    cloud_borne: bool | None = None,
) -> ModalMicrophysicsTerm:
    if isinstance(microphysics, ModalMicrophysicsTerm):
        if (
            cloud_borne is not None
            and microphysics.spec.cloud_borne != cloud_borne
        ):
            raise ValueError(
                f"cloud_borne={cloud_borne} conflicts with the supplied "
                "core's population "
                f"(spec.cloud_borne={microphysics.spec.cloud_borne}); "
                "construct the core with a spec carrying the intended flag "
                "instead."
            )
        return microphysics
    try:
        factory = _MICROPHYSICS[microphysics]
    except KeyError:
        raise ValueError(
            f"Unknown aer microphysics {microphysics!r}. "
            f"Choose one of {sorted(_MICROPHYSICS)} or pass a "
            "ModalMicrophysicsTerm instance."
        ) from None
    core = factory(None)
    if cloud_borne is not None and core.spec.cloud_borne != cloud_borne:
        # Rebuild on the same population with the flag flipped; construction
        # is compose-time only, so the double build costs nothing at run time.
        core = factory(
            dataclasses.replace(core.spec, cloud_borne=cloud_borne)
        )
    return core


def jam_aerosol_physics(
    *,
    microphysics: ModalMicrophysicsTerm | str = "placeholder",
    cloud_borne: bool | None = None,
    arg_variant: str = "arg2000",
    optics: bool = True,
    optics_backend: str = "jcm",
    optics_diagnostics: bool = False,
    seasalt: SeaSaltParameters | None = None,
    seasalt_scheme: str = "gong",
    dms: DmsParameters | None = None,
    dust: DustParameters | None = None,
    dust_preset: int = 4,
    dust_nudged: bool = False,
    dust_nduscale_scale: float | None = None,
    anthropogenic: bool = False,
    anthropogenic_params: EmissionParameters | None = None,
    prescribed_speciated: bool = False,
    oxidants: OxidantParameters | None = None,
    sulfur_gas: SulfurGasParameters | None = None,
    aqueous: AqueousSulfurParameters | None = None,
    aqueous_scheme: str = "full",
    freezing_classes: HamFreezingClasses | None = None,
    activation: ArgParameters | None = None,
    cloud_borne_exchange: CloudBorneExchangeParameters | None = None,
    sedimentation: SedParameters | None = None,
    drydep: DryDepParameters | None = None,
    wetdep: WetDepParameters | None = None,
    vertical_mixing: bool = True,
    tracer_diffusion: TracerDiffusionParameters | None = None,
    convective_transport: bool = True,
    conv_transport: ConvTransportParameters | None = None,
) -> list[PhysicsTerm]:
    """Build the ordered JAM harness term list.

    Args:
        microphysics: the swappable core — ``"placeholder"`` or a
            ``ModalMicrophysicsTerm`` instance.
        optics_backend: which ``_mode_optics`` implementation ``optics=True``
            attaches (``docs/source/design/jam_optics_mode_seam.md``):
            ``"jcm"`` (default) is the on-the-fly Gauss-Hermite quadrature
            over jcm's own Mie LUT; ``"ham_lut"`` is ``HamLutOpticsTerm``,
            ECHAM-HAM M7's own nearest-neighbour Mie-table lookup (#1017).
            This is the one in-tree exception to the seam's "no registry"
            design — the implementation lives in this repository, so it
            gets a selector here rather than requiring an out-of-tree
            subclass and a manual ``physics.replace(...)``.
        cloud_borne: prognose an explicit cloud-borne aerosol phase (#602).
            ``None`` (default) follows the core population's own
            ``spec.cloud_borne``; ``True``/``False`` override it for a
            string-named core (and merely validate an instance core). On:
            the ``mc_*``/``nc_*`` phase lives in the physics carry and is
            cycled — activation transfer + resuspension
            (``CloudBorneExchange``), in-droplet wet removal, surface dry
            deposition, and the aqueous sulfate split. Off: no cloud-borne
            store at all and the harness scavenges interstitial aerosol by
            its activated fraction, the implicit M7/TOMAS-style treatment.
            Both settings are complete physics, one flag apart.
        arg_variant: ``"arg2000"`` (default) or ``"ghosh2025"`` activation.
        seasalt/dms/dust: optional ``Parameters`` overrides for the natural
            emission schemes (Gong/Long sea salt, Nightingale DMS, Tegen dust).
        seasalt_scheme: ``"gong"`` (default, unchanged) or ``"long"`` (Long
            et al. 2011 + the Sofiev et al. 2011 SST correction, HAM
            ``nseasalt=7``; #1017). ``"long"`` requires the microphysics
            core's population to carry exactly two ``ss`` classes, in HAM's
            own accumulation-then-coarse order — see
            :class:`~jcm.physics.aerosol.jam.emissions.seasalt.SeaSaltEmissions`.
        dust_preset: HAMMOZ ``ndust`` preset — 4 (default, Stier 2005 +
            East-Asian soils = HAM2), 3 (Stier 2005) or 2 (Cheng 2008).
            Ignored when an explicit ``dust`` parameter object is given.
        dust_nudged: use HAM's nudged regional tuning vector (0.95/1.25 at
            T63) rather than the free-running one (1.05/1.45).
        dust_nduscale_scale: global multiplier on that regional vector, jcm's
            single dust calibration knob (#808). ``None`` takes the
            calibrated default (T63 ndust=4 only; HAM's value elsewhere).
        anthropogenic: include prescribed CEDS anthropogenic emissions (#498),
            the *bulk* path (in-model differentiable speciation + smooth
            injection); ``anthropogenic_params`` overrides the defaults.
        prescribed_speciated: include the CAM6/MAM4-faithful *already-speciated*
            emission path (#498) — per-tracer fields injected directly, no
            in-model speciation. Independent of ``anthropogenic``; both, either,
            or neither may be enabled.
        oxidants/sulfur_gas/aqueous: optional ``Parameters`` for the
            prescribed-oxidant + gas-phase + aqueous sulfur chemistry (#496).
        aqueous_scheme: ``"full"`` (default, HAM ``ham_wet_chemistry`` port) or
            ``"simple"`` (H2O2-limited stoichiometric oxidation).
        freezing_classes: which classes of the population play HAM's roles in
            the aerosol inputs to mixed-phase freezing (``HamFreezingClasses``);
            ``None`` takes the MAM4 mapping, ``MAM4_FREEZING_CLASSES``.
        activation/cloud_borne_exchange/sedimentation/drydep/wetdep/
            tracer_diffusion/conv_transport: optional per-process
            ``Parameters`` overrides (each ``None`` resolves to its
            default).
        vertical_mixing: turbulent vertical diffusion of every JAM tracer
            with the TTE-TKE exchange coefficients (#602 item 2 — ECHAM
            diffuses all tracers in vdiff; without this the dycore is the
            sole aerosol transporter). On by default.
        convective_transport: bulk mass-flux transport of the interstitial
            aerosol and gas tracers through Tiedtke updrafts and
            downdrafts with compensating subsidence (ECHAM ``cuxtte``
            analogue; #602 item 2, #622), including in-plume scavenging
            of every aerosol mode at HAMMOZ's per-mode ``csr_conv`` and
            the plume's precipitation efficiency (#621) — which moves the
            convective in-cloud wet-removal pathway out of
            ``WetScavenging`` (``in_plume_convective``). Cloud-borne
            mirrors are deliberately excluded — their updraft processing
            is entangled with convective scavenging and neither reference
            model transports a stratiform cloud-borne phase convectively.
            On by default.

    Stratiform in-cloud wet removal and cloud-borne resuspension key to
    the cloud scheme's process-time scavenging ledger (#708 — the
    ECHAM-HAM ``cloud_subm`` interface), so the harness requires a cloud
    scheme that publishes the ``CloudData`` ledger fields (the 2M
    scheme); ``echam_physics`` enforces this at compose time.

    Returns:
        The ordered term list: natural emissions, prescribed oxidants and
        gas-phase sulfur chemistry, the microphysics core (optionally followed
        by online optics), activation, sedimentation, dry deposition,
        cloud-borne exchange (when the population prognoses one), in-cloud
        aqueous sulfur chemistry, and wet deposition.

    """
    core = _resolve_microphysics(microphysics, cloud_borne)
    spec = core.spec
    emissions = [
        SeaSaltEmissions(params=seasalt, spec=spec, scheme=seasalt_scheme),
        DmsEmissions(params=dms, spec=spec),
        DustEmissions(params=dust, ndust=dust_preset, nudged=dust_nudged,
                      nduscale_scale=dust_nduscale_scale, spec=spec),
    ]
    if anthropogenic:
        # Prescribed CEDS anthropogenic SO2/BC/OC (#498); inert until forcing
        # fluxes are supplied.
        emissions.append(
            AnthropogenicEmissions(params=anthropogenic_params, spec=spec)
        )
    if prescribed_speciated:
        # CAM6/MAM4-faithful already-speciated emissions (#498); inert until
        # per-tracer forcing fields are supplied.
        emissions.append(PreSpeciatedEmissions())
    if emissions:
        # Must precede every emitter: the emi_* accumulators are additive
        # across terms, and the diagnostics dict is threaded back in from the
        # previous step, so they have to be zeroed once per step.
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            ResetEmissionFluxes)
        emissions = [ResetEmissionFluxes(), *emissions]
    chemistry = [
        # Sulfur chemistry: oxidants → gas-phase DMS/SO2 oxidation, producing
        # the H2SO4/SOAG gas the core condenses + nucleates this same step.
        PrescribedOxidants(params=oxidants),
        SulfurGasChemistry(params=sulfur_gas),
    ]
    # Physics-side vertical transport (#602 item 2). The tracer set is
    # everything the composed JAM terms declare (aerosol in both phases +
    # gas precursors); ECHAM's vdiff diffuses all of them. Convection
    # moves the interstitial + gas tracers only (see the docstring).
    # Placed right after the emitters so the narrative order matches
    # ECHAM's physc (vdiff -> convection -> chemistry); under operator
    # splitting the tendencies sum regardless.
    transport_names: list[str] = []
    for _t in [core, *emissions, *chemistry]:
        for _s in _t.required_tracers():
            if _s.name not in transport_names:
                transport_names.append(_s.name)
    transport_terms: list[PhysicsTerm] = []
    if vertical_mixing:
        transport_terms.append(
            TracerVerticalDiffusion(
                tuple(transport_names), params=tracer_diffusion,
            )
        )
    if convective_transport:
        interstitial_names = tuple(
            n for n in transport_names if not n.startswith(("mc_", "nc_"))
        )
        # In-plume scavenging (jax-gcm#621): every aerosol tracer of a mode
        # takes HAMMOZ's convective in-droplet fraction ``csr_conv`` of the
        # M7 class the mode corresponds to (number and mass alike;
        # ``wetdep.convective_fractions.HAM_CSR_CONV`` holds the mapping and
        # its reasoning); the gas precursors ride the plume unscavenged.
        # WetScavenging retires its own environment-profile convective
        # pathway in turn (``in_plume_convective`` below).
        csr_of: dict[str, float] = {}
        for mode in spec.modes:
            csr = convective_csr(mode)
            csr_of[number_name(mode.short)] = csr
            for sp in mode.species:
                csr_of[mass_name(sp, mode.short)] = csr
        transport_terms.append(
            ConvectiveTracerTransport(
                interstitial_names, params=conv_transport,
                csr_conv=tuple(csr_of.get(n, 0.0) for n in interstitial_names),
            )
        )
    # Carry-stored cloud-borne phase (#602 item 3, the measured decision
    # — see cloud_borne_store, which also records why the advected-tracer
    # alternative was removed): the store term runs first so every
    # consumer this step sees a well-formed store, and applies the
    # carry's turbulent vertical mixing (its fields are not in
    # state.tracers, so TracerVerticalDiffusion never sees them).
    # The carry's mixing uses the same exchange-coefficient scale as the
    # advected tracers' (``TracerDiffusionParameters``), so one setting must
    # reach both phases; each term holds its own copy of the object.
    store_terms = (
        [CloudBorneCarryStore(params=tracer_diffusion, spec=spec,
                              vertical_mixing=vertical_mixing)]
        if carry_mode(spec) else []
    )
    pre_core = [
        # Own the shared ``aerosol`` carry slot (#640). Radiation hard-requires
        # it and the 2M microphysics reads ``aerosol.Nccn``; with MACv2-SP gone
        # from the JAM path this seeder resets it to an all-zero base each step
        # (JamOpticsTerm overwrites the optics when ``optics=True``; left zero =
        # radiatively passive when ``optics=False``). Runs first so every
        # downstream aerosol/radiation consumer sees a well-formed slot.
        AerosolCarrySeeder(),
        *store_terms,
        *emissions,
        *transport_terms,
        *chemistry,
    ]
    # Online aerosol direct radiative effect (#495): placed right after the core
    # (needs ``_jam_state``); overwrites the MACv2-SP ``aerosol`` optics.
    # ``optics_diagnostics`` adds the AeroCom per-species / per-mode /
    # spectral optics pass (jax-gcm#584) — a second Mie sweep at the
    # observation wavelengths, off unless a run asks for it.
    if optics_backend == "jcm":
        optics_cls = JamOpticsTerm
    elif optics_backend == "ham_lut":
        optics_cls = HamLutOpticsTerm
    else:
        raise ValueError(
            f"Unknown optics_backend={optics_backend!r}. Choose 'jcm' or "
            "'ham_lut'."
        )
    optics_terms = [
        optics_cls(spec=spec, optics_diagnostics=optics_diagnostics)
    ] if optics else []
    post_core = [
        ArgActivation(params=activation, spec=spec, variant=arg_variant),
        # ECHAM-HAM's aerosol inputs to mixed-phase freezing (mo_ham_freezing
        # ham_IN_setup) -> ``freezing_aerosol``, which the 2M scheme turns into
        # contact + immersion freezing rates (#953). After ARG: HAM's
        # activated number per class is ARG's.
        IceNucleation(spec=spec, classes=freezing_classes),
        StokesSedimentation(params=sedimentation, spec=spec),
        SlinnDryDeposition(params=drydep, spec=spec),
        # Cloud-borne cycling (#602): activation transfer + resuspension
        # against the current step's cloud field, so it runs in the
        # post-cloud block, before the aqueous chemistry that splits its
        # product by cloud-borne number and the scavenging that drains the
        # reservoir. Only composed when the population prognoses the phase.
        *([CloudBorneExchange(params=cloud_borne_exchange, spec=spec)]
          if spec.cloud_borne else []),
        # In-cloud aqueous SO2 oxidation → cloud-borne sulfate; runs in the
        # post-cloud block (needs current clouds), just before wet scavenging.
        AqueousSulfur(params=aqueous, spec=spec, scheme=aqueous_scheme),
        WetScavenging(params=wetdep, spec=spec,
                      in_plume_convective=convective_transport),
    ]
    terms = [*pre_core, core, *optics_terms, *post_core]
    return terms
