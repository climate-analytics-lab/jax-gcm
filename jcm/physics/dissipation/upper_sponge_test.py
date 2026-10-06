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


def _anomaly_fields(shape, seed=0):
    """``_fields`` without their large offsets, for the gradient checks.

    The term is linear and returns zonal anomalies only, so a float32 central
    difference on T ~ 230 K would measure the cancellation in ``T - [T]``
    rather than the term; removing the offsets leaves the term unchanged.
    """
    return tuple(x - jnp.mean(x) for x in _fields(shape, seed))


def _term(**kwargs):
    term = UpperSponge(**kwargs)
    term.cache_coords(_COORDS)
    return term


class TestUpperSpongeGradients:
    """AD against a central difference for the sponge (#820).

    Green, and expected to be: the term is linear in the state — an implicit
    relaxation of the zonal anomalies of u, v and T with a precomputed
    coefficient profile. It
    is here as a fence rather than a hunt: a later guard or reshape added
    upstream would show up as a failing comparison instead of as a drifting
    run.

    The zonal anomaly reshapes a flattened column axis back to
    ``(nlon, nlat)``, so both host layouts are checked. The fields carry
    real horizontal structure; a uniform field would make the zonal-mean
    anomaly identically zero and the relaxation would be tested against
    nothing.
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
        term = _term(n_sponge_levels=3, damp_temperature=True, enspodi=2.0)
        check_gradients(self._tendencies(term, shape, (_NLON * _NLAT,)),
                        _anomaly_fields(shape), rtol=1e-3, seed=seed)

    def test_whole_grid_gradients(self):
        """The un-vectorised layout: ``(nlev, nlon, nlat)``."""
        shape = (_NLEV, _NLON, _NLAT)
        term = _term(n_sponge_levels=3, damp_temperature=True, enspodi=2.0)
        check_gradients(self._tendencies(term, shape, (_NLON, _NLAT)),
                        _anomaly_fields(shape), rtol=1e-3)

    def test_wind_only_gradients(self):
        """``damp_temperature=False``: wind only."""
        shape = (_NLEV, _NLON * _NLAT)
        term = _term(n_sponge_levels=3, damp_temperature=False)
        check_gradients(self._tendencies(term, shape, (_NLON * _NLAT,)),
                        _anomaly_fields(shape), rtol=1e-3)


class TestUpperSpongeIsEchamUspnge:
    """The term is ECHAM's ``uspnge`` (``mo_upper_sponge.f90``)."""

    DT = 720.0

    def _apply(self, term, shape, horizontal_shape, seed=1):
        u, v, T = _fields(shape, seed)
        state = PhysicsState.zeros(
            shape, temperature=T, u_wind=u, v_wind=v,
            normalized_surface_pressure=jnp.ones(horizontal_shape))
        tend, _ = term(state, {"_dt_seconds": self.DT},
                       ForcingData.zeros((_NLON, _NLAT)), None)
        return (u, v, T), tend

    def test_defaults_are_echam_setdyn(self):
        term = UpperSponge()
        assert term.n_sponge_levels == 1
        np.testing.assert_allclose(1.0 / term.sponge_timescale_s, 0.926e-4)
        assert term.enspodi == 1.0
        assert term.damp_temperature

    def test_zonal_mean_untouched_and_anomaly_damped_implicitly(self):
        """M = 0 is never damped (``IF (mymsp(is) /= 0)``); the m != 0 part is
        multiplied by ``1 / (1 + zlf*dt)`` over one step, exactly.
        """
        term = _term()
        shape = (_NLEV, _NLON, _NLAT)
        fields, tend = self._apply(term, shape, (_NLON, _NLAT))
        zlf = 1.0 / term.sponge_timescale_s
        factor = 1.0 / (1.0 + zlf * self.DT)
        for x, dx in zip(fields, (tend.u_wind, tend.v_wind, tend.temperature)):
            x = np.asarray(x, np.float64)
            dx = np.asarray(dx, np.float64)
            # zonal mean of the tendency is zero at every level
            np.testing.assert_allclose(dx.mean(axis=1), 0.0, atol=1e-6)
            stepped = x + self.DT * dx
            anomaly = x - x.mean(axis=1, keepdims=True)
            np.testing.assert_allclose(
                stepped[0] - x[0].mean(axis=0, keepdims=True),
                factor * anomaly[0], rtol=1e-5, atol=1e-4)
            # below the sponge nothing changes
            np.testing.assert_array_equal(dx[1:], 0.0)

    def test_level_profile_enspodi(self):
        """``enspodi`` multiplies the coefficient per level going up."""
        term = _term(n_sponge_levels=3, sponge_timescale_s=3600.0,
                     enspodi=2.0)
        zlf = np.asarray(term._zlf.get_value())
        np.testing.assert_allclose(
            zlf[:3], [1 / 3600.0, 1 / 7200.0, 1 / 14400.0], rtol=1e-6)
        np.testing.assert_array_equal(zlf[3:], 0.0)

    def test_column_and_grid_layouts_agree(self):
        term = _term(n_sponge_levels=2)
        _, grid = self._apply(term, (_NLEV, _NLON, _NLAT), (_NLON, _NLAT))
        _, cols = self._apply(term, (_NLEV, _NLON * _NLAT), (_NLON * _NLAT,))
        for a, b in ((grid.u_wind, cols.u_wind), (grid.v_wind, cols.v_wind),
                     (grid.temperature, cols.temperature)):
            np.testing.assert_allclose(
                np.asarray(a).reshape(np.asarray(b).shape), np.asarray(b),
                rtol=1e-6, atol=1e-9)

    def test_rejects_a_grid_without_a_longitude_axis(self):
        class _Flat:
            nodal_shape = (_NLEV, 100)

            class horizontal:  # noqa: N801
                nodal_shape = (100,)

        with pytest.raises(ValueError, match="lid_sponge"):
            UpperSponge().cache_coords(_Flat)

    @pytest.mark.parametrize("kwargs", [
        {"n_sponge_levels": 0}, {"sponge_timescale_s": 0.0},
        {"enspodi": -1.0}])
    def test_rejects_invalid_profiles(self, kwargs):
        with pytest.raises(ValueError):
            UpperSponge(**kwargs)
