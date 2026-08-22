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
from jcm.physics.aerosol.jam.ice_nucleation.ice_term import IceNucleation
from jcm.physics.aerosol.jam.ice_nucleation.params import (
    IceNucleationParameters,
)
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.sedimentation.sedi_term import (
    StokesSedimentation,
    SedParameters,
)
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

def _load_mam4_jax() -> type[ModalMicrophysicsTerm]:
    """Import the MAM4-JAX core lazily (optional GPL-3.0 dependency)."""
    from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
        Mam4JaxMicrophysics,
    )

    return Mam4JaxMicrophysics


# Core resolvers (each takes a spec override, ``None`` for the core default).
# ``placeholder`` is built-in; ``mam4_jax`` is loaded lazily so the optional
# GPL-3.0 ``mam4-jax`` dependency is only imported when selected.
_MICROPHYSICS = {
    "placeholder": lambda spec: PlaceholderMicrophysics(spec=spec),
    "mam4_jax": lambda spec: _load_mam4_jax()(spec=spec),
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
    optics_diagnostics: bool = False,
    seasalt: SeaSaltParameters | None = None,
    dms: DmsParameters | None = None,
    dust: DustParameters | None = None,
    anthropogenic: bool = False,
    anthropogenic_params: EmissionParameters | None = None,
    prescribed_speciated: bool = False,
    oxidants: OxidantParameters | None = None,
    sulfur_gas: SulfurGasParameters | None = None,
    aqueous: AqueousSulfurParameters | None = None,
    aqueous_scheme: str = "full",
    ice_scheme: str = "niemand",
    ice_nucleation_params: IceNucleationParameters | None = None,
    activation: ArgParameters | None = None,
    cloud_borne_exchange: CloudBorneExchangeParameters | None = None,
    sedimentation: SedParameters | None = None,
    drydep: DryDepParameters | None = None,
    wetdep: WetDepParameters | None = None,
    vertical_mixing: bool = True,
    tracer_diffusion: TracerDiffusionParameters | None = None,
    convective_transport: bool = True,
    conv_transport: ConvTransportParameters | None = None,
    scavenging_ledger: bool = True,
) -> list[PhysicsTerm]:
    """Build the ordered JAM harness term list.

    Args:
        microphysics: the swappable core — ``"placeholder"`` or a
            ``ModalMicrophysicsTerm`` instance.
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
            emission schemes (Gong sea salt, Nightingale DMS, Tegen dust).
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
        ice_scheme: heterogeneous freezing scheme — ``"niemand"`` (default,
            singular/active-site) or ``"lohmann_diehl"`` (ECHAM-HAM number-based);
            ``ice_nucleation_params`` overrides the differentiable defaults.
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
            analogue; #602 item 2, #622), including CAM
            ``aero_convproc``-style in-plume scavenging of the soluble
            (activatable-mode) tracers (#621) — which moves the
            convective in-cloud wet-removal pathway out of
            ``WetScavenging`` (``in_plume_convective``). Cloud-borne
            mirrors are deliberately excluded — their updraft processing
            is entangled with convective scavenging and neither reference
            model transports a stratiform cloud-borne phase convectively.
            On by default.
        scavenging_ledger: key stratiform in-cloud wet removal and
            cloud-borne resuspension to the cloud scheme's process-time
            scavenging ledger (#708 — the ECHAM-HAM ``cloud_subm``
            interface). Requires a cloud scheme that publishes the
            ``CloudData`` ledger fields (the 2M scheme); set False for the
            1M scheme, which does not yet (#712), so both terms fall back to the
            legacy cover-keyed reconstruction. Static by design — the
            fallback must never be a silent per-cell branch.

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
        SeaSaltEmissions(params=seasalt, spec=spec),
        DmsEmissions(params=dms, spec=spec),
        DustEmissions(params=dust, spec=spec),
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
        # In-plume scavenging weights (jax-gcm#621): soluble = the
        # activatable modes' interstitial tracers; insoluble aerosol and
        # the gas precursors ride the plume unscavenged. WetScavenging
        # retires its own environment-profile convective pathway in turn
        # (``in_plume_convective`` below).
        soluble = set()
        for mode in spec.modes:
            if mode.can_activate:
                soluble.add(number_name(mode.short))
                soluble.update(
                    mass_name(sp, mode.short) for sp in mode.species
                )
        transport_terms.append(
            ConvectiveTracerTransport(
                interstitial_names, params=conv_transport,
                scav_weights=tuple(
                    1.0 if n in soluble else 0.0 for n in interstitial_names
                ),
            )
        )
    # Carry-stored cloud-borne phase (#602 item 3, the measured decision
    # — see cloud_borne_store, which also records why the advected-tracer
    # alternative was removed): the store term runs first so every
    # consumer this step sees a well-formed store, and applies the
    # carry's turbulent vertical mixing (its fields are not in
    # state.tracers, so TracerVerticalDiffusion never sees them).
    store_terms = (
        [CloudBorneCarryStore(spec=spec, vertical_mixing=vertical_mixing)]
        if carry_mode(spec) else []
    )
    pre_core = [
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
    optics_terms = [
        JamOpticsTerm(spec=spec, optics_diagnostics=optics_diagnostics)
    ] if optics else []
    post_core = [
        ArgActivation(params=activation, spec=spec, variant=arg_variant),
        # Heterogeneous ice nucleation on dust/BC → ``ice_nuclei`` for the 2M
        # cloud scheme (#494). Harmless with the 1M scheme (diagnostic unused).
        IceNucleation(
            params=ice_nucleation_params, spec=spec, scheme=ice_scheme,
        ),
        StokesSedimentation(params=sedimentation, spec=spec),
        SlinnDryDeposition(params=drydep, spec=spec),
        # Cloud-borne cycling (#602): activation transfer + resuspension
        # against the current step's cloud field, so it runs in the
        # post-cloud block, before the aqueous chemistry that splits its
        # product by cloud-borne number and the scavenging that drains the
        # reservoir. Only composed when the population prognoses the phase.
        *([CloudBorneExchange(params=cloud_borne_exchange, spec=spec,
                              evaporation_ledger=scavenging_ledger)]
          if spec.cloud_borne else []),
        # In-cloud aqueous SO2 oxidation → cloud-borne sulfate; runs in the
        # post-cloud block (needs current clouds), just before wet scavenging.
        AqueousSulfur(params=aqueous, spec=spec, scheme=aqueous_scheme),
        WetScavenging(params=wetdep, spec=spec,
                      in_plume_convective=convective_transport,
                      formation_ledger=scavenging_ledger),
    ]
    terms = [*pre_core, core, *optics_terms, *post_core]
    return terms
