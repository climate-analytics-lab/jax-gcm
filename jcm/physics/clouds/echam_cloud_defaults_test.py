"""ECHAM's grid-dependent cloud defaults and how they reach the cover.

Pins the table against ``mo_echam_cloud_params.f90::sucloud`` (r7492
l.198-237), the interpolation between ECHAM's truncations, ECHAM's inversion
levels, and the precedence of the values a run ends up with: an explicit
``CloudParameters`` object, then a field override, then the grid's default.
"""

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics import resolution_defaults
from jcm.physics.clouds.echam_cloud_defaults import (
    ECHAM_CLOUD_DEFAULTS,
    echam_cloud_defaults,
    inversion_levels,
)
from jcm.physics.clouds.sundqvist import (
    CloudParameters,
    SundqvistCloudFraction,
    calculate_cloud_fraction,
)
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.physics_term import with_field_overrides

FIELDS = ("crs", "crt", "nex", "nadd", "csatsc", "cinv", "cvtfall",
          "csecfrl", "clwprat")

#: ``mo_echam_cloud_params.f90`` l.198-237, transcribed independently of the
#: module under test, in the Fortran's order.
FORTRAN = {
    31: (0.95, 0.85, 1, 1, 0.1, 0.5, 3.0, 5.0e-7, 0.0),
    63: (0.975, 0.75, 2, 0, 0.7, 0.25, 2.5, 5.0e-6, 4.0),
    127: (0.994, 0.75, 2, 0, 0.7, 0.25, 3.0, 1.0e-5, 4.0),
    255: (0.994, 0.75, 2, 0, 0.7, 0.25, 3.0, 1.0e-5, 4.0),
}


def _grid(truncation, nlev=47, nodal_shape=(192, 96)):
    """Build a stand-in coordinate system: what the cover reads from one."""
    horizontal = SimpleNamespace(nodal_shape=nodal_shape)
    if truncation is not None:
        horizontal.longitude_wavenumbers = truncation + 1
    return SimpleNamespace(horizontal=horizontal,
                           vertical=get_echam_levels(nlev))


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """Each test sees the once-per-process warnings afresh."""
    resolution_defaults._WARNED.clear()
    yield
    resolution_defaults._WARNED.clear()


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("truncation", sorted(FORTRAN))
def test_echam_truncations_return_the_fortran_values(truncation):
    got = echam_cloud_defaults(truncation)
    assert tuple(got[f] for f in FIELDS) == FORTRAN[truncation]
    assert got == ECHAM_CLOUD_DEFAULTS[truncation]


def test_t106_is_interpolated_between_t63_and_t127():
    """Linear in the truncation number: T106 is 43/64 of the way."""
    w = (106 - 63) / (127 - 63)
    got = echam_cloud_defaults(106)
    for f, lo, hi in zip(FIELDS, FORTRAN[63], FORTRAN[127]):
        if f in ("nex", "nadd"):
            assert got[f] == lo == hi           # equal at both neighbours
        else:
            assert got[f] == pytest.approx(lo + w * (hi - lo), rel=1e-15)
    assert got["crs"] == pytest.approx(0.987765625)
    assert got["cvtfall"] == pytest.approx(2.8359375)
    assert got["csecfrl"] == pytest.approx(8.359375e-6)


def test_integer_fields_take_the_nearer_truncation():
    """``nex``/``nadd`` are integers in ECHAM: never interpolated."""
    assert (echam_cloud_defaults(42)["nex"], echam_cloud_defaults(42)["nadd"]) == (1, 1)
    assert (echam_cloud_defaults(47)["nex"], echam_cloud_defaults(47)["nadd"]) == (2, 0)
    assert (echam_cloud_defaults(46)["nex"], echam_cloud_defaults(46)["nadd"]) == (1, 1)
    # ... while the real-valued ones are interpolated at the same T42
    assert echam_cloud_defaults(42)["crt"] == pytest.approx(
        0.85 + (42 - 31) / 32 * (0.75 - 0.85))


@pytest.mark.parametrize("truncation, end", [(21, 31), (511, 255)])
def test_outside_the_range_holds_the_end_and_warns_once(truncation, end):
    with pytest.warns(UserWarning, match=f"T{truncation}"):
        got = echam_cloud_defaults(truncation)
    assert got == ECHAM_CLOUD_DEFAULTS[end]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        echam_cloud_defaults(truncation)        # second time: silent


def test_non_spectral_grid_gets_t63_with_a_warning():
    with pytest.warns(UserWarning, match="no spectral truncation"):
        got = echam_cloud_defaults(None)
    assert got == ECHAM_CLOUD_DEFAULTS[63]


def test_inversion_levels_are_echams():
    """``sucloud`` l.152-162: ECHAM's L47 gives jbmin/jbmax = 40/45."""
    assert inversion_levels(get_echam_levels(47)) == (39, 44)
    assert inversion_levels(get_echam_levels(95)) == (87, 92)
    assert inversion_levels(_grid(63)) == (39, 44)


# ---------------------------------------------------------------------------
# Precedence: explicit object > field override > grid default
# ---------------------------------------------------------------------------

def _cover_term(physics):
    return next(t for t in physics.terms
                if isinstance(t, SundqvistCloudFraction))


def _params(physics):
    return _cover_term(physics).params.get_value()


class TestPrecedence:

    def test_grid_default(self):
        from jcm.physics.echam.echam_terms import echam_physics
        p = _params(echam_physics(coords=_grid(127)))
        assert float(p.crs) == pytest.approx(0.994)
        assert float(p.csecfrl) == pytest.approx(1e-5)
        assert p.defaults_truncation == 127

    def test_no_grid_is_t63(self):
        from jcm.physics.echam.echam_terms import echam_physics
        p = _params(echam_physics())
        assert float(p.crs) == pytest.approx(0.975)
        assert p.defaults_truncation == 63

    def test_field_override_wins_over_the_grid_default(self):
        from jcm.physics.echam.echam_terms import echam_physics
        p = _params(echam_physics(coords=_grid(127), clouds={"crs": 0.99}))
        assert float(p.crs) == pytest.approx(0.99)          # override
        assert float(p.csecfrl) == pytest.approx(1e-5)      # T127 default

    def test_explicit_object_is_used_as_given(self):
        from jcm.physics.echam.echam_terms import echam_physics
        mine = CloudParameters.default(truncation=63, crt=0.8)
        physics = echam_physics(coords=_grid(127), clouds=mine)
        p = _params(physics)
        assert p is mine or all(
            float(getattr(p, f)) == float(getattr(mine, f))
            for f in ("crt", "crs", "csecfrl"))
        # and no grid warning, even though it carries T63 defaults
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _cover_term(physics).cache_coords(_grid(127))

    def test_runner_term_list_door(self):
        from jcm.runners import _build_term
        entry = {"_target_": "jcm.physics.clouds.sundqvist.SundqvistCloudFraction",
                 "params": {"cinv": 0.3}}
        term = _build_term("sundqvist_cloud_fraction", entry, 127)
        p = term.params.get_value()
        assert float(p.cinv) == pytest.approx(0.3)
        assert float(p.crs) == pytest.approx(0.994)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(_grid(127))

    def test_runner_passes_the_grid_to_the_factory(self):
        from jcm.runners import _build_physics_from_factory
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"builder": "echam_physics",
                                "clouds": {"crt": 0.7}})
        p = _params(_build_physics_from_factory(cfg, _grid(255)))
        assert float(p.crt) == pytest.approx(0.7)
        assert float(p.crs) == pytest.approx(0.994)

    def test_gradients_of_overridden_and_defaulted_fields_are_live(self):
        """A calibration can differentiate every field, however it was set."""
        base = with_field_overrides(CloudParameters.default(truncation=127),
                                    {"crt": 0.72}, scheme="clouds")
        pf = jnp.array([20000.0, 50000.0, 80000.0, 95000.0])
        t = jnp.array([225.0, 255.0, 280.0, 287.0])
        geo = jnp.array([1.2e5, 5.6e4, 1.9e4, 4.3e3])
        q = jnp.array([6e-5, 8e-4, 5e-3, 1.0e-2])

        def total(params):
            return calculate_cloud_fraction(
                t, q, jnp.zeros(4), pf, jnp.asarray(100000.0), geo, params,
                (1, 2))[0].sum()

        g = jax.grad(total)(base)
        assert np.isfinite(float(g.crt)) and float(g.crt) != 0.0   # overridden
        assert np.isfinite(float(g.crs)) and float(g.crs) != 0.0   # defaulted


class TestGridCheck:

    def test_defaults_for_another_grid_warn_naming_both(self):
        from jcm.physics.echam.echam_terms import echam_physics
        term = _cover_term(echam_physics())              # T63 defaults
        with pytest.warns(UserWarning, match=r"T63.*T127"):
            term.cache_coords(_grid(127))

    def test_matching_grid_is_silent(self):
        from jcm.physics.echam.echam_terms import echam_physics
        term = _cover_term(echam_physics(coords=_grid(106)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(_grid(106))

    def test_single_column_grid_is_not_checked(self):
        term = SundqvistCloudFraction()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(_grid(None, nodal_shape=(1, 1)))

    def test_non_spectral_grid_warns(self):
        term = SundqvistCloudFraction()
        with pytest.warns(UserWarning, match="non-spectral"):
            term.cache_coords(_grid(None, nodal_shape=(21600,)))

    def test_physics_built_with_the_non_spectral_grid_passes_its_own_check(self):
        """The pySES runner builds the physics with the dycore's grid.

        That grid has no spectral truncation, so the defaults are ECHAM's T63
        row (with the one warning that says so at construction) and record
        that they were built for a non-spectral grid; the check at
        ``cache_coords`` then finds nothing to report, where physics built
        without the grid is flagged.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        grid = _grid(None, nodal_shape=(1, 21600))
        with pytest.warns(UserWarning, match="no spectral truncation"):
            term = _cover_term(echam_physics(coords=grid))
        assert term.params.get_value().crs == ECHAM_CLOUD_DEFAULTS[63]["crs"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(grid)


# ---------------------------------------------------------------------------
# The 2M's cvtfall: ECHAM's 2M reads sucloud's per-truncation value
# ---------------------------------------------------------------------------

def _two_moment_term(physics):
    from jcm.physics.clouds.lohmann_2m import Lohmann2MMicrophysics
    return next(t for t in physics.terms
                if isinstance(t, Lohmann2MMicrophysics))


class TestTwoMomentCvtfall:
    """``mo_cloud_micro_2m.f90`` l.97, 536 read ``cvtfall`` from ``sucloud``."""

    @pytest.mark.parametrize("truncation", sorted(FORTRAN))
    def test_echam_truncations_return_the_fortran_value(self, truncation):
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        p = CloudParams2M.default(truncation=truncation)
        assert float(p.cvtfall) == pytest.approx(FORTRAN[truncation][6],
                                                 rel=1e-6)
        assert p.defaults_truncation == truncation

    def test_t106_is_interpolated_like_the_1m(self):
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        assert float(CloudParams2M.default(truncation=106).cvtfall) == \
            pytest.approx(2.8359375, rel=1e-6)

    def test_t63_is_bit_identical_to_the_value_before_the_table(self):
        """Every leaf at T63 is what ``default()`` has always returned."""
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        explicit = CloudParams2M.default(cvtfall=2.5)
        for built in (CloudParams2M.default(),
                      CloudParams2M.default(truncation=63)):
            for a, b in zip(jax.tree_util.tree_leaves(built),
                            jax.tree_util.tree_leaves(explicit)):
                assert np.array_equal(np.asarray(a), np.asarray(b))
            assert np.asarray(built.cvtfall).dtype == \
                np.asarray(explicit.cvtfall).dtype

    def test_non_spectral_grid_takes_the_t63_row_with_a_warning(self):
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        with pytest.warns(UserWarning, match="no spectral truncation"):
            p = CloudParams2M.default(truncation=None)
        assert float(p.cvtfall) == 2.5 and p.defaults_truncation is None

    def test_explicit_cvtfall_wins(self):
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        assert float(CloudParams2M.default(truncation=127,
                                           cvtfall=2.0).cvtfall) == 2.0

    def test_factory_builds_the_grid_default(self):
        from jcm.physics.echam.echam_terms import echam_physics
        term = _two_moment_term(echam_physics(cloud_scheme="2m",
                                              coords=_grid(127)))
        p = term.params.get_value()
        assert float(p.cvtfall) == pytest.approx(3.0)
        assert p.defaults_truncation == 127
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(_grid(127))

    def test_factory_without_a_grid_is_t63_and_warns_on_another_grid(self):
        from jcm.physics.echam.echam_terms import echam_physics
        term = _two_moment_term(echam_physics(cloud_scheme="2m"))
        assert float(term.params.get_value().cvtfall) == 2.5
        with pytest.warns(UserWarning, match=r"T63.*T127"):
            term.cache_coords(_grid(127))

    def test_field_override_wins_and_keeps_the_grid_check(self):
        from jcm.physics.echam.echam_terms import echam_physics
        term = _two_moment_term(echam_physics(
            cloud_scheme="2m", coords=_grid(127),
            microphysics_2m={"ccraut": 9.0}))
        p = term.params.get_value()
        assert float(p.ccraut) == pytest.approx(9.0)
        assert float(p.cvtfall) == pytest.approx(3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(_grid(127))

    def test_explicit_object_is_used_as_given_and_not_checked(self):
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        from jcm.physics.echam.echam_terms import echam_physics
        mine = CloudParams2M.default(truncation=63, ccraut=9.0)
        term = _two_moment_term(echam_physics(
            cloud_scheme="2m", coords=_grid(127), microphysics_2m=mine))
        assert float(term.params.get_value().cvtfall) == 2.5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(_grid(127))

    def test_runner_term_list_door(self):
        from jcm.runners import _build_term
        entry = {"_target_": "jcm.physics.clouds.lohmann_2m.Lohmann2MMicrophysics",
                 "params": {"ccraut": 9.0}}
        term = _build_term("lohmann_2m_microphysics", entry, 127)
        p = term.params.get_value()
        assert float(p.ccraut) == pytest.approx(9.0)
        assert float(p.cvtfall) == pytest.approx(3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.cache_coords(_grid(127))
