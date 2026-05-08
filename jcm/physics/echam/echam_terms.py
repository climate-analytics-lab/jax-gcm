"""``echam_physics()`` factory.

Every ECHAM parameterisation lives as a ``PhysicsTerm`` next to its
underlying numerical implementation (``TiedtkeConvection``,
``SundqvistCloudFraction``, ``Echam1MMicrophysics``,
``GreyTwoStreamRadiation``, …). This module is the user-facing factory
that wires the scheme-named terms together in a validated default
ordering and returns a ready-to-run ``ComposablePhysics`` with column
vectorisation enabled.

Date: 2026-04-13
"""

from __future__ import annotations

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


def echam_physics(
    parameters: Parameters | None = None,
    checkpoint_terms: bool = True,
    radiation_scheme: str | PhysicsTerm = "grey",
    cloud_scheme: str = "1m",
):
    """Create a ``ComposablePhysics`` with the standard ECHAM term ordering.

    Args:
        parameters: Optional ECHAM Parameters. Uses defaults if None.
        checkpoint_terms: Whether to checkpoint terms.
        radiation_scheme: "grey" (default), "rrtmgp", "emulated", or a
            custom ``PhysicsTerm`` with category "radiation".
        cloud_scheme: "1m" (default, single-moment) or "2m" (two-moment
            warm-rain; see issue #341 for ongoing scheme completion).

    Returns:
        A ``ComposablePhysics`` instance with all ECHAM terms in the
        validated default order, configured for column vectorisation.

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

    return ComposablePhysics(
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
        vectorize_columns=True,
    )
