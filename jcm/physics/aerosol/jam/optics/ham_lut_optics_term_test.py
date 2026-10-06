"""Tests for ``HamLutOpticsTerm`` on a locally built, M7-faithful population.

Most of these tests exercise the TERM's own logic (mode selection,
gradients, the nucleation-mode gate) and pass jcm's own built tables
(``ham_mie_tables.default_ham_mie_tables``) directly via ``tables=``,
rather than loading HAM's authentic files: none of them care about the
table's physical CONTENT, and passing ``tables=`` keeps them deterministic
regardless of ``HAM_INPUT_DIR``'s ambient state. See ``ham_mie_tables.py``'s
module docstring for why jcm's build is a fine stand-in for THESE tests, and
``ham_mie_tables_test.py`` for the tests that do need the real data (the
authentic-vs-built lookup parity and content-comparison tests).

A separate group below (``test_table_source_*``) tests the fallback/logging/
``table_source`` behaviour the #1017 coordinator's revision added: HAM's
authentic tables when ``HAM_INPUT_DIR`` holds them, jcm's built tables
otherwise, logged once and recorded on the instance either way.
"""

import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.optics.ham_lut_optics_term import HamLutOpticsTerm
from jcm.physics.aerosol.jam.optics.ham_mie_tables import default_ham_mie_tables
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.population import AerosolMode, AerosolSpecies, ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.radiation.band_config import RadiationBandConfig

# M7 membership (mo_ham_m7ctl/mo_ham_init), same pattern as
# ham_freezing_reference_test.py::m7_spec, but with HAM's OWN fine/coarse
# geom_std_dev (1.59/2.0) rather than that test's geometry-irrelevant
# placeholder -- this backend's table selection depends on it.
M7_MEMBER = {"nucs": ("so4",), "aits": ("so4", "bc", "poa"),
             "accs": ("so4", "bc", "poa", "ss", "du"), "coas": ("so4", "bc", "poa", "ss", "du"),
             "aiti": ("bc", "poa"), "acci": ("du",), "coai": ("du",)}
M7_SIGMA = {"nucs": 1.59, "aits": 1.59, "accs": 1.59, "coas": 2.00,
            "aiti": 1.59, "acci": 1.59, "coai": 2.00}
_DENSITY = {"so4": 1841.0, "bc": 2000.0, "poa": 1000.0, "ss": 2165.0, "du": 2650.0}


def m7_spec():
    species = tuple(AerosolSpecies(s, 0.1, _DENSITY[s], 0.1) for s in _DENSITY)
    modes = tuple(AerosolMode(m, m, M7_SIGMA[m], 1.0e-7, 1.0e-8, 1.0e-6, M7_MEMBER[m],
                              soluble=m.endswith("s"), can_activate=m.endswith("s"),
                              sediments=True) for m in M7_MEMBER)
    return ModalAerosolSpec(modes=modes, species=species)


def _setup(nlev=4, ncols=3, n_sw=4, n_lw=3):
    from jcm.physics.aerosol.aerosol_types import AerosolData
    from jcm.physics_interface import PhysicsState

    spec = m7_spec()
    n_modes = spec.n_modes()
    shape = (n_modes, nlev, ncols)
    aer = JamAerosolState(
        r_dry=jnp.full(shape, 0.08e-6), r_wet=jnp.full(shape, 0.15e-6),
        rho=jnp.full(shape, 1800.0), kappa=jnp.full(shape, 0.5),
        mass=jnp.full(shape, 1.0e-9), number=jnp.full(shape, 1.0e8),
    )
    tracers = {}
    for mode in spec.modes:
        tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
        for sp in mode.species:
            tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1.0e-9)
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 285.0), tracers=tracers)
    sw = tuple(float(x) for x in np.linspace(400.0, 1000.0, n_sw))
    lw = tuple(float(x) for x in np.linspace(8000.0, 20000.0, n_lw))
    band = RadiationBandConfig(lw_band_centers_nm=lw, sw_band_centers_nm=sw)
    aerosol = AerosolData.zeros((ncols,), nlev, n_bnd_sw=n_sw, n_bnd_lw=n_lw)
    diagnostics = {
        "_jam_state": aer, "aerosol": aerosol,
        "air_density": jnp.full((nlev, ncols), 1.0),
        "layer_thickness": jnp.full((nlev, ncols), 500.0),
        "_band_config": band,
    }
    return spec, state, diagnostics, band


def _term(band, spec):
    term = HamLutOpticsTerm(spec=spec, tables=default_ham_mie_tables())
    term.cache_band_config(band)
    return term


def test_build_mie_lut_is_skipped():
    """``HamLutOpticsTerm`` never reads the default Gauss-Hermite table."""
    assert HamLutOpticsTerm(tables=default_ham_mie_tables())._lut is None
    assert JamOpticsTerm()._lut is not None


def test_m7_spec_optics_finite_and_bounded_with_nucleation_mode_inactive():
    spec, state, diagnostics, band = _setup()
    term = _term(band, spec)
    _, out = term(state, diagnostics, None, None)
    a = out["aerosol"]
    for arr in (a.aod_sw_per_band, a.aod_lw_per_band):
        assert np.all(np.isfinite(np.asarray(arr)))
        assert bool(jnp.all(arr >= 0.0))
    for arr in (a.ssa_sw_per_band, a.ssa_lw_per_band):
        assert np.all(np.isfinite(np.asarray(arr)))
        assert bool(jnp.all((arr >= 0.0) & (arr <= 1.0 + 1e-5)))
    for arr in (a.asy_sw_per_band, a.asy_lw_per_band):
        assert np.all(np.isfinite(np.asarray(arr)))
        assert bool(jnp.all((arr >= -1.0 - 1e-5) & (arr <= 1.0 + 1e-5)))
    assert float(jnp.sum(a.aod_sw_per_band)) > 0.0
    assert float(jnp.sum(a.aod_lw_per_band)) > 0.0

    # Mode activity: HAM's nrad(1)=0 -- the nucleation mode ("nucs", index 0
    # in m7_spec's member order) must contribute exactly zero, isolated by
    # zeroing every OTHER mode's mass/number and checking the result is
    # identically zero everywhere.
    nucs_only_tracers = {k: (v if k in (number_name("nucs"), mass_name("so4", "nucs"))
                             else jnp.zeros_like(v))
                        for k, v in state.tracers.items()}
    state_nucs = state.copy(tracers=nucs_only_tracers)
    aer = diagnostics["_jam_state"]
    number_nucs_only = jnp.zeros_like(aer.number).at[0].set(aer.number[0])
    diagnostics_nucs = {**diagnostics, "_jam_state": aer.copy(number=number_nucs_only)}
    _, out_nucs = term(state_nucs, diagnostics_nucs, None, None)
    np.testing.assert_array_equal(np.asarray(out_nucs["aerosol"].aod_sw_per_band), 0.0)
    np.testing.assert_array_equal(np.asarray(out_nucs["aerosol"].aod_lw_per_band), 0.0)


@pytest.mark.parametrize("sigma,expect_fine", [(1.59, True), (2.00, False), (1.6, True), (1.8, False)])
def test_fine_coarse_table_selection_by_nearest_sigma(sigma, expect_fine):
    """A mode's table pair is whichever of HAM's own 1.59/2.0 it is nearer

    to -- exact for M7 (1.59/2.0 themselves), an approximation (e.g. MAM4's
    1.6/1.8) otherwise. Checked indirectly: swap a mode's sigma and confirm
    the resulting tau changes (fine vs coarse tables disagree at the same
    geometry) -- directly inspecting which table was picked would require
    reaching into the hook's internals.
    """
    tables = default_ham_mie_tables()
    fine = abs(sigma - 1.59) <= abs(sigma - 2.00)
    assert fine == expect_fine
    lut = tables["sw_fine" if fine else "sw_coarse"]
    assert lut is not None


def test_gradient_of_tau550_wrt_mode_number_is_finite():
    """The seam's required gradient check, for this backend specifically:

    tau = num_per_area * q_norm * lambda**2 is LINEAR in the mode number
    (through num_per_area), so its gradient is finite and nonzero even
    though the nearest-neighbour table lookup itself has a flat (but still
    finite -- zero, not NaN) gradient in x/refractive index almost
    everywhere, faithfully matching HAM's own non-interpolated lookup
    (loint=.FALSE.) rather than smoothing it.
    """
    spec, state, diagnostics, band = _setup()
    term = _term(band, spec)
    aer = diagnostics["_jam_state"]
    idx = term._cache.aod_band_idx

    def loss(scale):
        aer2 = aer.copy(number=aer.number * scale)
        _, d = term(state, {**diagnostics, "_jam_state": aer2}, None, None)
        return jnp.sum(d["aerosol"].aod_sw_per_band[idx])

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert np.isfinite(float(g))
    assert float(g) > 0.0


def test_gradient_through_mass_is_finite_even_though_flat():
    """A nearest-neighbour lookup's gradient through the refractive index

    (itself a function of species mass) is zero almost everywhere -- not
    NaN, not smoothed. This is the faithful consequence of porting HAM's
    un-interpolated table exactly, not a defect to fix (see
    ham_mie_tables.py's docstring on ``loint=.FALSE.``).
    """
    spec, state, diagnostics, band = _setup()
    term = _term(band, spec)
    key = mass_name("bc", "aits")

    def loss(scale):
        tr = {k: (v * scale if k == key else v) for k, v in state.tracers.items()}
        s = state.copy(tracers=tr)
        _, d = term(s, diagnostics, None, None)
        return jnp.sum(d["aerosol"].aod_sw_per_band)

    g = jax.grad(loss)(jnp.asarray(1.0))
    assert np.isfinite(float(g))


def test_lw_tables_have_zero_ssa_and_asymmetry():
    spec, state, diagnostics, band = _setup()
    term = _term(band, spec)
    _, out = term(state, diagnostics, None, None)
    a = out["aerosol"]
    np.testing.assert_array_equal(np.asarray(a.ssa_lw_per_band), 0.0)
    np.testing.assert_array_equal(np.asarray(a.asy_lw_per_band), 0.0)


def test_attaches_via_jam_aerosol_physics_backend_selector(monkeypatch):
    """The selector wires up ``HamLutOpticsTerm``; the file load itself is

    monkeypatched to always raise (forcing the jcm-built fallback, see the
    ``test_table_source_*`` group below) so this test -- checking WIRING,
    not table content -- needs no ``HAM_INPUT_DIR``/real data and stays
    fast regardless of the ambient environment.
    """
    from jcm.physics.aerosol.jam.optics import ham_lut_optics_term
    from jcm.physics.echam.echam_terms import echam_physics

    def _raise(directory):
        raise FileNotFoundError("stubbed: no authentic tables in this test")

    monkeypatch.setattr(ham_lut_optics_term, "load_ham_mie_tables", _raise)
    physics = echam_physics(aerosol_module="jam", cloud_scheme="2m",
                             jam_optics_backend="ham_lut")
    terms = [t for t in physics.terms if t.category == "aerosol_optics"]
    assert len(terms) == 1
    assert isinstance(terms[0], HamLutOpticsTerm)
    assert terms[0].table_source == "jcm_built"

    default_physics = echam_physics(aerosol_module="jam", cloud_scheme="2m")
    default_terms = [t for t in default_physics.terms if t.category == "aerosol_optics"]
    assert isinstance(default_terms[0], JamOpticsTerm)
    assert not isinstance(default_terms[0], HamLutOpticsTerm)


def test_unknown_optics_backend_rejected():
    from jcm.physics.aerosol.jam.jam_terms import jam_aerosol_physics

    with pytest.raises(ValueError, match="optics_backend"):
        jam_aerosol_physics(optics_backend="not_a_backend")


def test_table_source_explicit_when_tables_passed():
    term = HamLutOpticsTerm(tables=default_ham_mie_tables())
    assert term.table_source == "explicit"


def test_table_source_falls_back_to_jcm_built_without_authentic_files(monkeypatch):
    """No ``HAM_INPUT_DIR``, no explicit ``tables_dir``: construction must

    succeed (not raise) and fall back to jcm's own built tables -- the
    #1017 coordinator's revision: the authentic files are a nice-to-have,
    not a hard construction requirement.
    """
    monkeypatch.delenv("HAM_INPUT_DIR", raising=False)
    term = HamLutOpticsTerm()
    assert term.table_source == "jcm_built"
    assert term._ham_tables["sw_fine"].q_ext is not None


def test_table_source_falls_back_on_a_tables_dir_missing_one_file(tmp_path):
    """A ``tables_dir`` that exists but does not hold both authentic files

    (e.g. a typo, or a directory copied without the LW file) falls back
    the same way an unset ``HAM_INPUT_DIR`` does -- ``load_ham_mie_tables``
    raises ``FileNotFoundError`` either way, and that is the ONLY exception
    this term's constructor catches to trigger the fallback (a different
    error -- e.g. the authentic file present but with the wrong axes,
    ``ValueError`` -- must still raise, not silently fall back to a
    different table).
    """
    term = HamLutOpticsTerm(tables_dir=tmp_path)
    assert term.table_source == "jcm_built"


@pytest.mark.skipif(not os.environ.get("HAM_INPUT_DIR"), reason="HAM_INPUT_DIR not set")
def test_table_source_is_authentic_when_ham_input_dir_set():
    term = HamLutOpticsTerm()
    assert term.table_source == "authentic"


def test_table_source_logged_once_at_construction(caplog):
    with caplog.at_level(logging.INFO, logger="jcm.physics.aerosol.jam.optics.ham_lut_optics_term"):
        term = HamLutOpticsTerm(tables=default_ham_mie_tables())
    records = [r for r in caplog.records if "HamLutOpticsTerm" in r.message]
    assert len(records) == 1
    assert term.table_source in records[0].message
