"""Test the per-scheme Parameters wiring in ``echam_physics()``.

After the scheme-named-terms refactor, each ECHAM term owns its own
scheme-native ``Parameters`` struct (``ConvectionParameters``,
``CloudParameters``, …). The ``echam_physics()`` factory accepts each
sub-Parameters as a keyword argument; ``None`` resolves to the scheme's
``.default()``. There is no monolithic aggregator.
"""

import jax.numpy as jnp

from jcm.physics.clouds.echam_1m import MicrophysicsParameters
from jcm.physics.clouds.sundqvist import CloudParameters
from jcm.physics.convection.tiedtke_nordeng import ConvectionParameters
from jcm.physics.echam.echam_terms import echam_physics


def test_per_scheme_defaults():
    """Each scheme's ``.default()`` reproduces the documented values."""
    convection = ConvectionParameters.default()
    clouds = CloudParameters.default()
    microphysics = MicrophysicsParameters.default()

    # ECHAM-matching convention: crt at surface (0.9), crs aloft (0.7);
    # ccraut = 15.0 (ECHAM default — Beheng-1994 coefficient, not the
    # KK2000 threshold the previous JAX port used).
    assert abs(float(convection.entrpen) - 1.0e-4) < 1e-7
    assert abs(float(clouds.crt) - 0.9) < 1e-7
    assert abs(float(clouds.crs) - 0.7) < 1e-7
    assert abs(float(microphysics.ccraut) - 15.0) < 1e-5


def test_echam_physics_default_kwargs():
    """``echam_physics()`` with no kwargs builds the scheme defaults."""
    physics = echam_physics()
    convection_term = next(
        t for t in physics.terms if t.category == "convection"
    )
    assert abs(float(convection_term.params.value.entrpen) - 1.0e-4) < 1e-7


def test_echam_physics_per_scheme_kwargs():
    """Per-scheme ``Parameters`` overrides flow through to the right term."""
    custom_convection = ConvectionParameters.default().__class__(
        **{**ConvectionParameters.default().__dict__, "entrpen": 5.0e-4}
    )
    physics = echam_physics(convection=custom_convection)
    convection_term = next(
        t for t in physics.terms if t.category == "convection"
    )
    assert abs(float(convection_term.params.value.entrpen) - 5.0e-4) < 1e-7

    # Untouched terms still see their defaults.
    cloud_fraction_term = next(
        t for t in physics.terms if t.category == "cloud_fraction"
    )
    assert abs(float(cloud_fraction_term.params.value.crt) - 0.9) < 1e-7


def test_physics_terms_compute_tendencies():
    """Smoke test: ``echam_physics`` with custom params computes tendencies."""
    from datetime import datetime

    import jax_datetime as jdt
    import numpy as np

    from jcm.date import DateData
    from jcm.forcing import ForcingData
    from jcm.physics_interface import PhysicsState
    from jcm.terrain import TerrainData
    from jcm.utils import get_coords

    nlev, nlat, nlon = 8, 64, 32
    state = PhysicsState(
        u_wind=jnp.zeros((nlev, nlat, nlon)),
        v_wind=jnp.zeros((nlev, nlat, nlon)),
        temperature=jnp.ones((nlev, nlat, nlon)) * 280.0,
        specific_humidity=jnp.ones((nlev, nlat, nlon)) * 0.005,
        geopotential=jnp.ones((nlev, nlat, nlon)) * 1000.0,
        normalized_surface_pressure=jnp.ones((nlat, nlon)),
        tracers={
            "qc": jnp.zeros((nlev, nlat, nlon)),
            "qi": jnp.zeros((nlev, nlat, nlon)),
        },
    )

    custom_clouds = CloudParameters.default().__class__(
        **{**CloudParameters.default().__dict__, "crt": 0.8}
    )
    physics = echam_physics(clouds=custom_clouds)

    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(nlat, nlon))
    terrain = TerrainData.aquaplanet(coords)
    physics.cache_coords(coords)
    forcing = ForcingData.zeros(
        (nlat, nlon),
        sea_surface_temperature=jnp.ones((nlat, nlon)) * 288.0,
        sice_am=jnp.zeros((nlat, nlon)),
    )
    date = DateData.set_date(
        jdt.Datetime.from_pydatetime(datetime(2020, 6, 21))
    )
    forcing = forcing.select(date)

    tendencies, _ = physics.compute_tendencies(
        state, forcing=forcing, terrain=terrain, date=date,
    )

    assert tendencies.temperature.shape == (nlev, nlat, nlon)
    assert "qc" in tendencies.tracers
    assert "qi" in tendencies.tracers


if __name__ == "__main__":
    test_per_scheme_defaults()
    test_echam_physics_default_kwargs()
    test_echam_physics_per_scheme_kwargs()
    test_physics_terms_compute_tendencies()
    print("\nAll parameter tests passed!")
