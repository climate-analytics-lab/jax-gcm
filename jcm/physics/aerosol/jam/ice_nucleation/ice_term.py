"""``IceNucleation``: ECHAM-HAM's aerosol inputs to mixed-phase freezing (#953).

Computes, from the prognostic JAM population, the eight fields
``mo_ham_freezing.f90::ham_IN_setup`` hands to the two-moment cloud scheme's
heterogeneous mixed-phase freezing (``het_mxphase_freezing``): the dust and
black-carbon fractions of the activated droplets (immersion) and of the
insoluble aerosol (contact), and the insoluble-mode wet radii. They are
published as ``freezing_aerosol``; the 2M scheme then freezes supercooled
cloud water at ECHAM-HAM's contact and immersion rates instead of its
aerosol-free DeMott (2010) closure. The partition and the MAM4 -> HAM class
mapping are in :mod:`jcm.physics.aerosol.jam.ice_nucleation.ham_freezing`.

Runs after ``ArgActivation`` (its per-mode activated fraction gives HAM's
``nact_strat``) in the JAM block ahead of the cloud microphysics.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from jcm.physics.aerosol.jam.removal_split import split_view
from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import (
    MAM4_FREEZING_CLASSES,
    HamFreezingClasses,
    ham_freezing_aerosol,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

_FRACTION = {"units": "1"}
FREEZING_AEROSOL_OUTPUT_ATTRS: dict[str, dict[str, str]] = {
    "freezing_aerosol.dust_soluble": {
        **_FRACTION,
        "long_name": "dust fraction of activated droplets (immersion freezing, HAM pfracdusol)"},
    "freezing_aerosol.dust_insoluble_accumulation": {
        **_FRACTION,
        "long_name": "insoluble accumulation dust fraction of insoluble aerosol (HAM pfracduai)"},
    "freezing_aerosol.dust_insoluble_coarse": {
        **_FRACTION,
        "long_name": "insoluble coarse dust fraction of insoluble aerosol (HAM pfracduci)"},
    "freezing_aerosol.bc_soluble": {
        **_FRACTION,
        "long_name": "black-carbon fraction of activated droplets (immersion freezing, HAM pfracbcsol)"},
    "freezing_aerosol.bc_insoluble": {
        **_FRACTION,
        "long_name": "black-carbon fraction of insoluble aerosol (HAM pfracbcinsol)"},
    "freezing_aerosol.wet_radius_insoluble_aitken": {
        "units": "m", "long_name": "wet radius of the insoluble Aitken class (HAM prwetki)"},
    "freezing_aerosol.wet_radius_insoluble_accumulation": {
        "units": "m", "long_name": "wet radius of the insoluble accumulation class (HAM prwetai)"},
    "freezing_aerosol.wet_radius_insoluble_coarse": {
        "units": "m", "long_name": "wet radius of the insoluble coarse class (HAM prwetci)"},
}


class IceNucleation(PhysicsTerm):
    """HAM's mixed-phase freezing inputs from the JAM population."""

    name: ClassVar[str] = "jam_ice_nucleation"
    category: ClassVar[str] = "aerosol_ice_nucleation"
    requires: ClassVar[tuple[str, ...]] = (
        "air_density", "_jam_state", "_jam_activation", "activated_cdnc",
    )
    provides: ClassVar[tuple[str, ...]] = ("freezing_aerosol",)
    output_attrs: ClassVar[dict[str, dict[str, str]]] = FREEZING_AEROSOL_OUTPUT_ATTRS

    def __init__(
        self,
        *,
        spec: ModalAerosolSpec | None = None,
        classes: HamFreezingClasses | None = None,
    ):
        """Hold the population and its HAM freezing-class mapping.

        ``classes`` defaults to the MAM4 mapping; a population with other
        class names must say which of its classes play HAM's roles.
        """
        self._spec = spec or MAM4_SPEC
        self._classes = classes or MAM4_FREEZING_CLASSES
        known = set(self._spec.mode_shorts)
        named = set(self._classes.soluble) | {
            s for s in (self._classes.insoluble_aitken,
                        self._classes.insoluble_accumulation,
                        self._classes.insoluble_coarse) if s is not None}
        if not named <= known:
            raise ValueError(
                f"HAM freezing classes {sorted(named - known)} are not classes of "
                f"the population {sorted(known)}; pass classes= to name the "
                "population's own classes.")
        self._classes.validate(self._spec)

    def __call__(self, state, diagnostics, forcing, terrain):
        spec = self._spec
        rho = diagnostics["air_density"]
        aer = diagnostics["_jam_state"]
        act = diagnostics["_jam_activation"]
        view = split_view(spec, state, diagnostics)
        zeros = jnp.zeros_like(rho)

        def tracer(name):
            return view.get(name, zeros)

        classes = self._classes
        named = set(classes.soluble) | {
            s for s in (classes.insoluble_aitken, classes.insoluble_accumulation,
                        classes.insoluble_coarse) if s is not None}
        # Composition of a class: its interstitial plus cloud-borne mass,
        # floored at zero. HAM takes the composition, the activation and the
        # insoluble number from one time level (pxtm1). Here the composition
        # and the insoluble number are the step-start tracers, while the
        # activated number and the wet radii are the core's post-step state
        # that ARG activates from (``_jam_state``), so the activated classes
        # sum to the ``activated_cdnc`` the 2M scheme receives. The core's
        # aging and condensation within the step shift the ratios by that
        # step's increment only.
        masses = {}
        for short in named:
            for sp in spec.mode(short).species:
                masses[(sp, short)] = jnp.maximum(
                    tracer(mass_name(sp, short))
                    + tracer(mass_name(sp, short, cloud_borne=True)), 0.0)
        number, wet_radius, activated = {}, {}, {}
        for i, mode in enumerate(spec.modes):
            if not mode.soluble:
                number[mode.short] = jnp.maximum(
                    tracer(number_name(mode.short))
                    + tracer(number_name(mode.short, cloud_borne=True)), 0.0)
            wet_radius[mode.short] = aer.r_wet[i]
            # HAM's nact_strat: ARG's activated fraction of the class times the
            # number ARG activated it from (floored at 0 as ARG floors it), so
            # the classes sum to ``activated_cdnc``.
            activated[mode.short] = (
                act.number_frac[i] * jnp.maximum(aer.number[i], 0.0) * rho)

        freezing = ham_freezing_aerosol(
            spec, classes, masses, number, activated, wet_radius, rho,
            diagnostics["activated_cdnc"],
        )
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "freezing_aerosol": freezing}
