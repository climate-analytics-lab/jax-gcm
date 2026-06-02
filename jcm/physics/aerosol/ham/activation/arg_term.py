"""``ArgActivation`` — ARG aerosol activation as a composable PhysicsTerm.

Reads the per-mode dry radius / κ / number from the ``_ham_state`` diagnostic
(written by the microphysics core) and writes ``activated_cdnc`` (m^-3) and
``activated_fraction`` for the downstream cloud microphysics. ``activated_cdnc``
is the same diagnostics key the 2M scheme's SPA floor produces, so ARG and SPA
are interchangeable activation sources (one factory arg) — see #342.

Updraft velocity: the characteristic activation updraft is read from
``updraft_velocity`` if some upstream term provides it; otherwise it is
derived from the previous step's TTE-TKE (w = √(2·TKE/3)), falling back to a
constant. It is therefore *not* a hard ``requires`` (no term provides it yet).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.ham.activation.arg import arg_activation
from jcm.physics.aerosol.ham.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.ham.population import ModalAerosolSpec
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency


@tree_math.struct
class ArgParameters:
    """Tunable knobs for ARG activation (differentiable)."""

    updraft_default: jnp.ndarray   # fallback characteristic updraft [m/s]
    tke_factor: jnp.ndarray        # w = sqrt(tke_factor * TKE)
    w_min: jnp.ndarray             # minimum updraft [m/s]

    @classmethod
    def default(cls) -> "ArgParameters":
        return cls(
            updraft_default=jnp.asarray(0.3),
            tke_factor=jnp.asarray(2.0 / 3.0),
            w_min=jnp.asarray(0.1),
        )


class ArgActivation(PhysicsTerm):
    """Abdul-Razzak & Ghan modal activation term."""

    name: ClassVar[str] = "arg_activation"
    category: ClassVar[str] = "aerosol_activation"
    requires: ClassVar[tuple[str, ...]] = (
        "_ham_state", "pressure_full", "air_density",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "activated_cdnc", "activated_fraction",
    )

    def __init__(
        self,
        params: ArgParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
        variant: str = "arg2000",
    ):
        """Hold params, the population, and the ARG variant string."""
        self.params = nnx.Param(params or ArgParameters.default())
        self._spec = spec or MAM4_SPEC
        if variant not in ("arg2000", "ghosh2025"):
            raise ValueError(f"Unknown ARG variant {variant!r}.")
        self._variant = variant
        # Static per-mode metadata (compose-time constants).
        self._sigma_g = tuple(m.geom_std_dev for m in self._spec.modes)
        self._can_activate = tuple(
            float(m.can_activate) for m in self._spec.modes
        )
        self._sigma_acc = self._spec.mode("accum").geom_std_dev

    def _updraft(self, diagnostics, shape, params):
        """Characteristic activation updraft [m/s]."""
        if "updraft_velocity" in diagnostics:
            w = diagnostics["updraft_velocity"]
        elif "vertical_diffusion" in diagnostics:
            tke = diagnostics["vertical_diffusion"].tke
            w = jnp.sqrt(jnp.maximum(params.tke_factor * tke, 0.0))
        else:
            w = jnp.full(shape, params.updraft_default)
        return jnp.maximum(w, params.w_min)

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        ham = diagnostics["_ham_state"]
        air_density = diagnostics["air_density"]  # (nlev, ncols)

        n_modes = ham.r_dry.shape[0]
        sigma_g = jnp.asarray(self._sigma_g).reshape(n_modes, 1, 1)
        can_activate = jnp.asarray(self._can_activate).reshape(n_modes, 1, 1)

        # number_per_mode is in kg^-1; convert to m^-3.
        number_vol = ham.number_per_mode * air_density[jnp.newaxis, :, :]

        updraft = self._updraft(diagnostics, state.temperature.shape, params)

        activated_cdnc, activated_fraction, _ = arg_activation(
            r_dry=ham.r_dry,
            kappa=ham.kappa,
            number_vol=number_vol,
            sigma_g=sigma_g,
            can_activate=can_activate,
            updraft=updraft,
            temperature=state.temperature,
            pressure=diagnostics["pressure_full"],
            sigma_acc=self._sigma_acc,
            variant=self._variant,
        )

        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {
            **diagnostics,
            "activated_cdnc": activated_cdnc,
            "activated_fraction": activated_fraction,
        }
