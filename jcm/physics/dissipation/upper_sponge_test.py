"""Tests for the upper-level sponge dissipation term."""

import jax.numpy as jnp
import numpy as np
import pytest

from jcm.forcing import ForcingData
from jcm.physics.dissipation.upper_sponge import UpperSponge
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import PhysicsState
from jcm.testing import check_gradients

_COORDS = get_speedy_coords(layers=8, spectral_truncation=21)
_NLEV = _COORDS.nodal_shape[0]
_NLON, _NLAT = _COORDS.horizontal.nodal_shape


def _fields(shape, seed=0):
    """Return (u, v, T) with real horizontal structure for the zonal mean."""
    rng = np.random.default_rng(seed)
    return (jnp.asarray(20.0 + 5.0 * rng.standard_normal(shape), jnp.float32),
            jnp.asarray(-8.0 + 4.0 * rng.standard_normal(shape), jnp.float32),
            jnp.asarray(230.0 + 10.0 * rng.standard_normal(shape),
                        jnp.float32))


def _term(**kwargs):
    term = UpperSponge(**kwargs)
    term.cache_coords(_COORDS)
    return term


class TestUpperSpongeGradients:
    """AD against a central difference for the sponge (#820).

    Green, and expected to be: every branch of this term is linear in the
    state — Rayleigh drag on u and v, a zonal-mean anomaly relaxation and an
    absolute-target relaxation on T, all with a precomputed ``1/tau(k)``. It
    is here as a fence rather than a hunt: a later guard or reshape added
    upstream would show up as a failing comparison instead of as a drifting
    run.

    The temperature branch reshapes a flattened column axis back to
    ``(nlon, nlat)`` to take the zonal mean, so the two host layouts are
    genuinely different code paths and both are checked. The fields carry
    real horizontal structure; a uniform field would make the zonal-mean
    anomaly identically zero and the relaxation branch would be tested
    against nothing.
    """

    @staticmethod
    def _tendencies(term, shape, horizontal_shape):
        """Return ``f(u, v, T) -> (du, dv, dT)`` for one host layout."""
        forcing = ForcingData.zeros((_NLON, _NLAT))

        def f(u_wind, v_wind, temperature):
            state = PhysicsState.zeros(
                shape, temperature=temperature, u_wind=u_wind, v_wind=v_wind,
                normalized_surface_pressure=jnp.ones(horizontal_shape))
            tendency, _ = term(state, {"_dt_seconds": 1800.0}, forcing, None)
            return (tendency.u_wind, tendency.v_wind, tendency.temperature)

        return f

    @pytest.mark.parametrize("seed", [0, 4])
    def test_column_block_gradients(self, seed):
        """The ``vectorize_columns`` layout: ``(nlev, nlon*nlat)``."""
        shape = (_NLEV, _NLON * _NLAT)
        term = _term(n_sponge_levels=3, damp_temperature=True,
                     target_T_K=220.0)
        check_gradients(self._tendencies(term, shape, (_NLON * _NLAT,)),
                        _fields(shape), rtol=1e-3, seed=seed)

    def test_whole_grid_gradients(self):
        """The un-vectorised layout: ``(nlev, nlon, nlat)``."""
        shape = (_NLEV, _NLON, _NLAT)
        term = _term(n_sponge_levels=3, damp_temperature=True,
                     target_T_K=220.0)
        check_gradients(self._tendencies(term, shape, (_NLON, _NLAT)),
                        _fields(shape), rtol=1e-3)

    def test_wind_only_gradients(self):
        """``damp_temperature=False``: the momentum-only default."""
        shape = (_NLEV, _NLON * _NLAT)
        term = _term(n_sponge_levels=3, damp_temperature=False)
        check_gradients(self._tendencies(term, shape, (_NLON * _NLAT,)),
                        _fields(shape), rtol=1e-3)
