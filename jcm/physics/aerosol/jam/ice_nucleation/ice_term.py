"""``IceNucleation`` — heterogeneous freezing term writing ``ice_nuclei`` (#494).

Computes the dust/BC IN populations from the prognostic aerosol, the ambient
temperature and ice saturation ratio, and (for the rate-based scheme) a
characteristic cooling rate from the TTE-TKE updraft; applies the selected
freezing parameterization (:mod:`niemand` or :mod:`lohmann_diehl`); and writes
the heterogeneous ice-crystal number ``ice_nuclei`` [m⁻³]. The 2-moment cloud
scheme reads it (like ARG's ``activated_cdnc``) to set the het ICNC.

Runs in the JAM pre-cloud block, before the cloud microphysics.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.constants import cpd as _CPD
from jcm.constants import grav as _G
from jcm.physics.aerosol.jam.ice_nucleation.in_populations import in_populations
from jcm.physics.aerosol.jam.ice_nucleation.lohmann_diehl import (
    lohmann_diehl_inp,
)
from jcm.physics.aerosol.jam.ice_nucleation.niemand import niemand_inp
from jcm.physics.aerosol.jam.ice_nucleation.params import IceNucleationParameters
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.convection.saturation import saturation_specific_humidity
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

_W_MIN = 0.01      # m/s — floor on the characteristic updraft
_W_DEFAULT = 0.1   # m/s — fallback updraft when TKE is unavailable (step 1)


class IceNucleation(PhysicsTerm):
    """Heterogeneous (immersion+deposition) freezing on dust + BC."""

    name: ClassVar[str] = "jam_ice_nucleation"
    category: ClassVar[str] = "aerosol_ice_nucleation"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full", "air_density")
    provides: ClassVar[tuple[str, ...]] = ("ice_nuclei", "ice_nuclei_deposition")

    def __init__(
        self,
        params: IceNucleationParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
        scheme: str = "niemand",
    ):
        """Hold params, the population, and the freezing scheme."""
        self.params = nnx.Param(params or IceNucleationParameters.default())
        self._spec = spec or MAM4_SPEC
        if scheme not in ("niemand", "lohmann_diehl"):
            raise ValueError(
                f"Unknown ice scheme {scheme!r}; choose 'niemand' or "
                "'lohmann_diehl'."
            )
        self._scheme = scheme

    def _cooling_rate(self, diagnostics, temperature):
        """Characteristic cooling rate [K/s] from the TKE updraft (ascent)."""
        vd = diagnostics.get("vertical_diffusion")
        if vd is not None:
            w = jnp.sqrt(jnp.maximum(2.0 / 3.0 * vd.tke, 0.0))
        else:
            w = jnp.full_like(temperature, _W_DEFAULT)
        return jnp.maximum(w, _W_MIN) * _G / _CPD

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        rho = diagnostics["air_density"]
        pressure = diagnostics["pressure_full"]
        t = state.temperature

        from jcm.physics.aerosol.jam.cloud_borne_store import tracer_view
        pops = in_populations(
            self._spec, tracer_view(self._spec, state, diagnostics),
            rho, p.frac_du_soluble,
        )
        qsat_ice = saturation_specific_humidity(t, pressure, phase="ice")
        s_ice = state.specific_humidity / jnp.maximum(qsat_ice, 1.0e-30)

        if self._scheme == "niemand":
            inp_imm, inp_dep = niemand_inp(pops, t, s_ice, p)
        else:
            dt = jnp.asarray(diagnostics["_dt_seconds"], t.dtype)
            cooling = self._cooling_rate(diagnostics, t)
            inp_imm, inp_dep = lohmann_diehl_inp(pops, t, s_ice, cooling, dt, p)

        # Immersion feeds the 2M mixed-phase het freezing; deposition feeds the
        # cirrus nucleation hook (newly_formed_ice). Splitting them avoids
        # double-counting where the regimes overlap.
        tendency = PhysicsTendency.zeros(t.shape)
        return tendency, {
            **diagnostics,
            "ice_nuclei": jnp.maximum(inp_imm, 0.0),
            "ice_nuclei_deposition": jnp.maximum(inp_dep, 0.0),
        }
