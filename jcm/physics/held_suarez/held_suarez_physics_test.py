"""Layout-agnostic Held-Suarez forcing: one code path for column, block and grid."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.held_suarez.held_suarez_physics import (
    HeldSuarez,
    held_suarez_physics,
)
from jcm.physics.held_suarez.utils import get_held_suarez_coords
from jcm.physics_interface import PhysicsState
from jcm.testing import check_gradients

_COORDS = get_held_suarez_coords(spectral_truncation=21)
_NLEV = 8
_NLON, _NLAT = _COORDS.horizontal.nodal_shape
_NCOLS = _NLON * _NLAT


def _grid_fields():
    """Deterministic, non-degenerate fields on the ``(nlev, nlon, nlat)`` grid."""
    shape = (_NLEV, _NLON, _NLAT)
    ii = np.arange(np.prod(shape)).reshape(shape)
    temperature = 250 + 40 * np.sin(0.37 * ii)
    u_wind = 15 * np.cos(0.11 * ii)
    v_wind = 8 * np.sin(0.23 * ii)
    ps = 1 + 0.03 * np.sin(0.5 * np.arange(_NCOLS)).reshape(_NLON, _NLAT)
    return temperature, u_wind, v_wind, ps


def _state(temperature, u_wind, v_wind, ps, tracers=None):
    return PhysicsState.zeros(
        temperature.shape,
        u_wind=jnp.asarray(u_wind),
        v_wind=jnp.asarray(v_wind),
        temperature=jnp.asarray(temperature),
        normalized_surface_pressure=jnp.asarray(ps),
        tracers=tracers,
    )


def _cached_term():
    term = HeldSuarez()
    term.cache_coords(_COORDS)
    return term


def _call(term, state):
    tendency, _ = term(state, {}, None, None)
    return tendency


def _grid_state():
    return _state(*_grid_fields())


def _flatten(arr):
    """``(nlev, nlon, nlat) -> (nlev, ncols)`` / ``(nlon, nlat) -> (ncols,)`` lon-major."""
    arr = np.asarray(arr)
    return arr.reshape(arr.shape[:-2] + (_NCOLS,))


def test_whole_grid_matches_values_from_the_previous_implementation():
    """Frozen outputs of the grid-only implementation, bit-level up to f32 round-off."""
    tend = _call(_cached_term(), _grid_state())
    reference = {
        # (level, lon, lat): (dT, du, dv)
        (0, 3, 5): (-1.0726822438300587e-05, 0.0, 0.0),
        (2, 10, 20): (-2.558368578320369e-05, 0.0, 0.0),
        (5, 40, 2): (-4.914906185149448e-06, 0.0, 0.0),
        (7, 63, 31): (1.1659534720820375e-05, -5.98654369241558e-05,
                      7.475434540538117e-05),
        (6, 17, 16): (0.00010204371938016266, -7.079333590809256e-05,
                      -3.881510201608762e-05),
    }
    for idx, (d_t, d_u, d_v) in reference.items():
        np.testing.assert_allclose(float(tend.temperature[idx]), d_t, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(float(tend.u_wind[idx]), d_u, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(float(tend.v_wind[idx]), d_v, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(float(tend.temperature.sum()), 0.04542200267314911, rtol=1e-4)
    np.testing.assert_allclose(float(tend.u_wind.sum()), 0.001540337223559618, rtol=1e-4)
    assert not np.any(np.asarray(tend.specific_humidity))


def test_matches_independent_held_suarez_1994_formula():
    """Cross-check against the closed form of Held & Suarez (1994), in float64 numpy."""
    term = _cached_term()
    temperature, u_wind, v_wind, ps = _grid_fields()
    tend = _call(term, _state(temperature, u_wind, v_wind, ps))

    v = lambda name: float(getattr(term, name).get_value())  # noqa: E731
    sigma = np.asarray(_COORDS.vertical.centers, dtype=np.float64)[:, None, None]
    lat = np.asarray(_COORDS.horizontal.latitudes, dtype=np.float64)[None, None, :]
    p_ratio = sigma * ps[None]
    t_eq = np.maximum(
        v("minT"),
        p_ratio ** term._kappa * (
            v("maxT") - v("dTy") * np.sin(lat) ** 2
            - v("dThz") * np.log(p_ratio) * np.cos(lat) ** 2),
    )
    cutoff = np.maximum(0.0, (sigma - v("sigma_b")) / (1.0 - v("sigma_b")))
    k_t = v("ka") + (v("ks") - v("ka")) * cutoff * np.cos(lat) ** 4
    k_v = v("kf") * cutoff
    np.testing.assert_allclose(tend.temperature, -k_t * (temperature - t_eq), rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(tend.u_wind, -k_v * u_wind, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(tend.v_wind, -k_v * v_wind, rtol=1e-5, atol=1e-12)


def test_grid_block_and_single_column_agree_per_column():
    """The same term, un-modified, on ``(kx, ix, il)``, ``(kx, ncols)`` and ``(kx,)``."""
    temperature, u_wind, v_wind, ps = _grid_fields()
    grid = _call(_cached_term(), _state(temperature, u_wind, v_wind, ps))

    block = _call(_cached_term(), _state(*map(_flatten, (temperature, u_wind, v_wind, ps))))
    for name in ("temperature", "u_wind", "v_wind", "specific_humidity"):
        assert getattr(block, name).shape == (_NLEV, _NCOLS)
        np.testing.assert_array_equal(getattr(block, name), _flatten(getattr(grid, name)))

    # A one-column host: latitude of flat column ``icol`` supplied through the
    # same ``column_latitudes`` hook a scattered-column grid uses.
    lat_cols = np.asarray(jnp.meshgrid(_COORDS.horizontal.latitudes,
                                       _COORDS.horizontal.longitudes)[0]).reshape(-1)
    for icol in (0, 37, _NCOLS - 1):
        one = types.SimpleNamespace(
            vertical=_COORDS.vertical,
            horizontal=types.SimpleNamespace(
                column_latitudes=lat_cols[icol:icol + 1],
                column_longitudes=np.zeros(1)),
        )
        term = HeldSuarez()
        term.cache_coords(one)
        col = _call(term, _state(*(_flatten(a)[..., icol] for a in
                                   (temperature, u_wind, v_wind, ps))))
        for name in ("temperature", "u_wind", "v_wind", "specific_humidity"):
            assert getattr(col, name).shape == (_NLEV,)
            np.testing.assert_array_equal(
                getattr(col, name), _flatten(getattr(grid, name))[:, icol])


@pytest.mark.parametrize("vectorize", [False, True])
def test_composable_physics_hosts_agree(vectorize):
    """``ComposablePhysics`` gives the same grid tendencies with the flag on and off."""
    from jcm.forcing import ForcingData
    from jcm.terrain import TerrainData

    def run(vectorize_columns):
        physics = held_suarez_physics(vectorize_columns=vectorize_columns)
        physics.cache_coords(_COORDS)
        tend, _ = physics.compute_tendencies(
            _grid_state(), ForcingData.zeros((_NLON, _NLAT)),
            TerrainData.aquaplanet(_COORDS))
        return tend

    reference = run(False)
    tend = run(vectorize)
    for name in ("temperature", "u_wind", "v_wind", "specific_humidity"):
        assert getattr(tend, name).shape == (_NLEV, _NLON, _NLAT)
        np.testing.assert_allclose(
            getattr(tend, name), getattr(reference, name), rtol=1e-6, atol=1e-12)
    assert np.any(np.asarray(tend.temperature))


def test_column_path_carries_tracers_through_composition():
    """On the column host a tracer in the state composes with the term (no tendency)."""
    from jcm.forcing import ForcingData
    from jcm.terrain import TerrainData

    physics = held_suarez_physics(vectorize_columns=True)
    physics.cache_coords(_COORDS)
    temperature, u_wind, v_wind, ps = _grid_fields()
    state = _state(temperature, u_wind, v_wind, ps,
                   tracers={"soot": jnp.ones_like(jnp.asarray(temperature))})
    tend, _ = physics.compute_tendencies(
        state, ForcingData.zeros((_NLON, _NLAT)), TerrainData.aquaplanet(_COORDS))
    assert set(tend.tracers) == {"soot"}
    assert not np.any(np.asarray(tend.tracers["soot"]))


def test_gradients_through_parameters_and_state():
    """AD agrees with a central difference through the Held-Suarez tunables."""
    temperature, u_wind, v_wind, ps = _grid_fields()
    block = [jnp.asarray(_flatten(a), dtype=jnp.float64 if jax.config.jax_enable_x64
                         else jnp.float32) for a in (temperature, u_wind, v_wind, ps)]
    reference_term = _cached_term()

    def f(kf, ka, ks, maxT, dTy, dThz, t, u, v):
        # Built inside the traced function: an nnx Variable cannot be mutated
        # from a different trace level.
        term = _cached_term()
        for name, val in dict(kf=kf, ka=ka, ks=ks, maxT=maxT, dTy=dTy, dThz=dThz).items():
            getattr(term, name).set_value(val)
        tend = _call(term, _state(t, u, v, block[3]))
        return tend.temperature, tend.u_wind, tend.v_wind

    args = tuple(jnp.asarray(getattr(reference_term, n).get_value())
                 for n in ("kf", "ka", "ks", "maxT", "dTy", "dThz")) + tuple(block[:3])
    check_gradients(f, args, rtol=5e-2)
