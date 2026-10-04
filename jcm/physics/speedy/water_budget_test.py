"""SPEEDY water budgets use the model's layer masses, including CLI grids."""

import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

import jcm.constants as c
from jcm.forcing import default_forcing
from jcm.physics.clouds.speedy_humidity import get_qsat
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics.surface.surface_exchange import SURFACE_EXCHANGE_KEY
from jcm.physics_interface import PhysicsState
from jcm.runners import build_coords
from jcm.terrain import TerrainData
from jcm.utils import get_coords


@pytest.mark.parametrize("grid", ["native", "cli", "custom"])
@pytest.mark.parametrize("nlev", [8, 16])
@pytest.mark.parametrize("prescribed", [False, True])
def test_water_budget_per_column(grid, nlev, prescribed):
    """Delivered E-P and redistribution close with actual pressure thicknesses."""
    if grid == "native":
        coords = get_speedy_coords(layers=nlev, spectral_truncation=21)
    elif grid == "cli":
        coords = build_coords(OmegaConf.create({"grid": {
            "vertical": "sigma", "layers": nlev, "spectral_truncation": 21,
        }}))
    else:
        coords = get_coords(np.linspace(0, 1, nlev + 1) ** 1.5,
                            spectral_truncation=21)
    physics = speedy_physics(checkpoint_terms=False)
    surface = next(t for t in physics.terms if t.category == "surface")
    surface.prescribed_fluxes = prescribed
    physics.cache_coords(coords)
    shape = coords.nodal_shape
    xy = coords.horizontal.nodal_shape
    sigma = jnp.asarray(coords.vertical.centers)[:, None, None]
    # Live moisture redistribution with spatially varying surface pressure.
    psa = jnp.broadcast_to(jnp.linspace(0.7, 1.05, xy[-1]), xy)
    temperature = jnp.broadcast_to(220.0 + 75.0 * sigma, shape)
    qsat = get_qsat(temperature, psa, sigma)
    state = PhysicsState.zeros(shape).copy(
        temperature=temperature,
        specific_humidity=qsat * (0.1 + 0.85 * sigma) / 1000.0,
        geopotential=jnp.broadcast_to(70000.0 * (1.0 - sigma), shape),
        u_wind=jnp.full(shape, 5.0), normalized_surface_pressure=psa,
    )
    forcing = default_forcing(coords.horizontal)
    if prescribed:
        forcing = forcing.copy(
            prescribed_evaporation=jnp.full(xy, 3e-5),
            prescribed_sensible_heat_flux=jnp.full(xy, 10.0),
            prescribed_stress_u=jnp.zeros(xy),
            prescribed_stress_v=jnp.zeros(xy),
        )
    terrain = TerrainData.aquaplanet(coords)
    mass = (jnp.diff(jnp.asarray(coords.vertical.boundaries))[:, None, None]
            * c.p0 * psa / c.grav)
    diagnostics = {}
    total = jnp.zeros(shape)
    for term in physics.terms:
        tendency, diagnostics = term(state, diagnostics, forcing, terrain)
        qdot = tendency.specific_humidity
        total = total + qdot
        column = jnp.sum(mass * qdot, axis=0)
        if term.category == "surface":
            evaporation = diagnostics[SURFACE_EXCHANGE_KEY].evaporation
            assert float(jnp.max(jnp.abs(evaporation))) > 1e-6
            np.testing.assert_allclose(column, evaporation, rtol=2e-6, atol=1e-11)
        elif term.category == "vertical_diffusion":
            assert float(jnp.max(jnp.abs(qdot))) > 1e-10
            np.testing.assert_allclose(column, 0.0, atol=1e-10)
    exchange = diagnostics[SURFACE_EXCHANGE_KEY]
    np.testing.assert_allclose(
        jnp.sum(mass * total, axis=0),
        exchange.evaporation - exchange.precipitation,
        rtol=2e-5, atol=1e-10,
    )
