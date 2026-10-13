"""Idealized ECHAM-stack composition for cheap tests.

**This is not ECHAM physics.** :func:`idealized_echam_physics` returns the
``echam_physics()`` term stack with the radiation slot filled by the grey
two-stream scheme
(:class:`~jcm.physics.radiation.grey_two_stream.GreyTwoStreamRadiation`), an
idealized scheme with no ECHAM reference that has
never been validated in an ECHAM composition. It exists because many tests
exercise *machinery* — composition and ordering, the physics carry,
checkpointing, output and CF metadata, dycore coupling, tracer plumbing,
budget-closure bookkeeping — and need a full, realistic term chain to do it,
but assert nothing about radiation or ECHAM science. The grey scheme gives
them that chain at a fraction of RRTMGP's compile and run cost (no gas-optics
tables, one broadband solve).

A test that asserts something about ECHAM physics behaviour must use the real
``echam_physics()`` (RRTMGP) instead; a test of the grey scheme itself builds
the grey term directly.
"""

from __future__ import annotations

from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.echam.echam_terms import (
    default_radiation_parameters,
    echam_physics,
)
from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
from jcm.physics.radiation.radiation_types import RadiationParameters


def idealized_echam_physics(
    *, radiation: RadiationParameters | None = None, **kwargs,
) -> ComposablePhysics:
    """ECHAM term stack with idealized grey radiation — for cheap tests only.

    Not ECHAM physics: see the module docstring. The grey term is composed
    explicitly through ``echam_physics(radiation_scheme=<term>)``, so every
    other factory argument behaves exactly as it does for the real stack.

    Args:
        radiation: ``RadiationParameters`` for the grey term. ``None`` takes
            the parameters ``echam_physics()`` would give its own radiation
            term
            (:func:`~jcm.physics.echam.echam_terms.default_radiation_parameters`
            for the requested ``aerosol_module``).
        **kwargs: Forwarded to ``echam_physics()``; ``radiation_scheme`` is
            fixed by this helper and may not be passed.

    Returns:
        A ``ComposablePhysics`` with the ECHAM term ordering and grey radiation.

    """
    if "radiation_scheme" in kwargs:
        raise TypeError(
            "idealized_echam_physics() fixes the radiation scheme to the grey "
            "two-stream; call echam_physics() directly to choose another.")
    params = radiation or default_radiation_parameters(
        kwargs.get("aerosol_module", "macv2sp"))
    return echam_physics(
        radiation_scheme=GreyTwoStreamRadiation(params=params), **kwargs)
