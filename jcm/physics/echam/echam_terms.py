"""``echam_physics()`` factory + ``ComposableEchamPhysics`` parameter wrapper.

After Phase 3 of the scheme-named-terms refactor, every ECHAM
parameterisation lives as a ``PhysicsTerm`` next to its underlying
numerical implementation (``TiedtkeConvection``, ``SundqvistCloudFraction``,
``Echam1MMicrophysics``, ``GreyTwoStreamRadiation``, …). The legacy
``apply_*`` wrappers in ``echam_physics.py`` are gone, the
``EchamTermBase`` / ``_data_from_diagnostics`` / ``_diagnostics_from_data``
bridge helpers that translated between the diagnostics dict and the
typed ``PhysicsData`` struct are gone, and this module shrinks to the
two pieces that still matter:

- :class:`ComposableEchamPhysics` — a ``ComposablePhysics`` subclass
  that still owns the shared ``Parameters`` struct so ``Model`` can
  call ``apply_timestep(dt_seconds)`` to keep ``parameters.convection.dt_conv``
  in sync. Will go away in Phase 4 once each scheme reads dt from
  ``diagnostics["_date"].dt_seconds`` directly.
- :func:`echam_physics` — the user-facing factory that wires the
  scheme-named terms together in a validated default ordering.

Date: 2026-04-13
"""

from __future__ import annotations

from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.echam.parameters import Parameters
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.aerosol import Macv2SpAerosol
from jcm.physics.chemistry import SimpleChemistry
from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
from jcm.physics.clouds.lohmann_2m import Lohmann2MMicrophysics
from jcm.physics.clouds.sundqvist import SundqvistCloudFraction
from jcm.physics.convection.tiedtke_nordeng import TiedtkeConvection
from jcm.physics.diagnostics.moist_air_state import MoistAirColumnState
from jcm.physics.forcing.echam_boundary_conditions import (
    EchamBoundaryConditions,
)
from jcm.physics.gravity_waves.hines import HinesGwd
from jcm.physics.gravity_waves.sso import LottMillerSso
from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
from jcm.physics.surface.echam.surface_physics import EchamSurface
from jcm.physics.vertical_diffusion.tte_tke import TteTkeVerticalDiffusion


# ------------------------------------------------------------------
# ComposableEchamPhysics — ECHAM parameter management
# ------------------------------------------------------------------

class ComposableEchamPhysics(ComposablePhysics):
    """ComposablePhysics with ECHAM shared parameter management.

    Column vectorization is handled by the parent class via
    ``vectorize_columns=True``. This subclass holds a single
    :class:`~jcm.physics.echam.parameters.Parameters` struct and
    implements ``apply_timestep(dt_seconds)`` so ``Model.__init__`` can
    sync ``parameters.convection.dt_conv`` to the model dt.

    Phase 4 of the refactor will remove this subclass and the
    isinstance gate in ``Model``: each scheme will read dt directly
    from ``diagnostics["_date"].dt_seconds``.
    """

    def __init__(self, terms, checkpoint_terms=True, parameters=None):
        """Initialize with ECHAM-specific parameter storage."""
        super().__init__(
            terms, checkpoint_terms, vectorize_columns=True,
        )
        self._echam_parameters = nnx.Variable(
            parameters or Parameters.default(),
        )

    @property
    def parameters(self) -> Parameters:
        """Read access to the shared ECHAM parameters struct."""
        return self._echam_parameters.get_value()

    def replace(self, category, new_term):
        """Replace a term, preserving ComposableEchamPhysics type."""
        new_terms = []
        inserted = False
        for t in self.terms:
            if t.category == category:
                if not inserted:
                    new_terms.append(new_term)
                    inserted = True
            else:
                new_terms.append(t)
        if not inserted:
            raise ValueError(
                f"No term with category {category!r} found.",
            )
        return ComposableEchamPhysics(
            terms=new_terms,
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._echam_parameters.get_value(),
        )

    def remove(self, category):
        """Remove terms, preserving ComposableEchamPhysics type."""
        return ComposableEchamPhysics(
            terms=[t for t in self.terms if t.category != category],
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._echam_parameters.get_value(),
        )

    def __add__(self, other):
        """Append term(s), preserving ComposableEchamPhysics type.

        Without this override the parent ``ComposablePhysics.__add__``
        returns a plain ``ComposablePhysics``; ``Model.__init__`` would
        then skip the ``apply_timestep`` call (it's gated on
        ``isinstance(..., ComposableEchamPhysics)``) and ECHAM terms
        would silently keep the default ``dt_conv = 3600 s`` regardless
        of the actual model timestep — corrupting any flux that uses
        ``dt_conv`` (e.g. surface implicit damping factor).
        """
        if hasattr(other, "terms"):
            other_terms = list(other.terms)
        elif hasattr(other, "category") and callable(other):
            other_terms = [other]
        else:
            return NotImplemented
        return ComposableEchamPhysics(
            terms=list(self.terms) + other_terms,
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._echam_parameters.get_value(),
        )

    def apply_timestep(self, dt_seconds: float):
        """Update timestep on the shared ECHAM parameters."""
        p = self._echam_parameters.get_value()
        self._echam_parameters = nnx.Variable(
            p.with_timestep(dt_seconds),
        )


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def echam_physics(
    parameters: Parameters | None = None,
    checkpoint_terms: bool = True,
    radiation_scheme: str | PhysicsTerm = "grey",
    cloud_scheme: str = "1m",
):
    """Create a ComposableEchamPhysics with standard ECHAM ordering.

    Args:
        parameters: Optional ECHAM Parameters. Uses defaults if None.
        checkpoint_terms: Whether to checkpoint terms.
        radiation_scheme: "grey" (default), "rrtmgp", "emulated", or a
            custom ``PhysicsTerm`` with category "radiation".
        cloud_scheme: "1m" (default, single-moment) or "2m" (two-moment
            warm-rain; see issue #341 for ongoing scheme completion).

    Returns:
        A ComposableEchamPhysics instance with all ECHAM terms.

    """
    p = parameters or Parameters.default()

    if isinstance(radiation_scheme, PhysicsTerm):
        if radiation_scheme.category != "radiation":
            raise ValueError(
                "Custom radiation_scheme terms must have category "
                "'radiation'."
            )
        rad_term = radiation_scheme
    elif radiation_scheme == "rrtmgp":
        rad_term = RRTMGPRadiation(params=p.radiation)
    elif radiation_scheme == "grey":
        rad_term = GreyTwoStreamRadiation(params=p.radiation)
    elif radiation_scheme == "emulated":
        rad_term = NNEmulatorRadiation(params=p.radiation)
    else:
        raise ValueError(
            f"Unknown radiation_scheme={radiation_scheme!r}. "
            "Choose 'grey', 'rrtmgp', 'emulated', or pass a radiation "
            "PhysicsTerm."
        )

    if cloud_scheme == "1m":
        micro_term = Echam1MMicrophysics(params=p.microphysics)
    elif cloud_scheme == "2m":
        micro_term = Lohmann2MMicrophysics(params=p.microphysics_2m)
        # SPA activation knobs live on AerosolParameters — wire them into
        # the 2M term so it stays self-contained at compose time.
        micro_term.configure_spa(
            float(p.aerosol.spa_prefactor),
            float(p.aerosol.spa_exponent),
        )
    else:
        raise ValueError(
            f"Unknown cloud_scheme={cloud_scheme!r}. Choose '1m' or '2m'."
        )

    return ComposableEchamPhysics(
        terms=[
            MoistAirColumnState(),
            EchamBoundaryConditions(),
            Macv2SpAerosol(params=p.aerosol),
            SimpleChemistry(),
            rad_term,
            TiedtkeConvection(params=p.convection),
            SundqvistCloudFraction(params=p.clouds),
            micro_term,
            TteTkeVerticalDiffusion(params=p.vertical_diffusion),
            EchamSurface(params=p.surface),
            HinesGwd(params=p.hines),
            LottMillerSso(params=p.sso),
        ],
        checkpoint_terms=checkpoint_terms,
        parameters=p,
    )
