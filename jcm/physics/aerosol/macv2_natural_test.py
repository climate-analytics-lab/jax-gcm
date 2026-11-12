"""Tests for the MACv2 natural aerosol background term."""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jcm.data.mirror.macv2 import (
    LW_WAVENUMBER_BOUNDS,
    SW_WAVELENGTH_NM,
    SW_WAVENUMBER_BOUNDS,
)
from jcm.physics.aerosol.aerosol_types import AerosolData
from jcm.physics.aerosol.macv2_natural import (
    Macv2NaturalAerosol,
    add_to_aerosol,
    band_index,
    distribute_layer_fractions,
    natural_aerosol_optics,
)
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.testing import check_gradients

_BOTTOM = jnp.asarray([0.0, 500.0, 1000.0, 2000.0, 5000.0])
_TOP = jnp.asarray([500.0, 1000.0, 2000.0, 5000.0, 20000.0])
_SW_LO = jnp.asarray([b[0] for b in SW_WAVENUMBER_BOUNDS], dtype=jnp.float32)
_SW_HI = jnp.asarray([b[1] for b in SW_WAVENUMBER_BOUNDS], dtype=jnp.float32)
_LW_LO = jnp.asarray([b[0] for b in LW_WAVENUMBER_BOUNDS], dtype=jnp.float32)
_LW_HI = jnp.asarray([b[1] for b in LW_WAVENUMBER_BOUNDS], dtype=jnp.float32)

# RRTMGP's bands, in its own (ascending wavenumber) order. They differ from
# RRTMG's at a few edges, which is the case the centre-in-band mapping serves.
_RRTMGP_SW = np.array([
    [820, 2680], [2680, 3250], [3250, 4000], [4000, 4650], [4650, 5150],
    [5150, 6150], [6150, 7700], [7700, 8050], [8050, 12850], [12850, 16000],
    [16000, 22650], [22650, 29000], [29000, 38000], [38000, 50000]], float)
_RRTMGP_LW = np.array([
    [10, 250], [250, 500], [500, 630], [630, 700], [700, 820], [820, 980],
    [980, 1080], [1080, 1180], [1180, 1390], [1390, 1480], [1480, 1800],
    [1800, 2080], [2080, 2250], [2250, 2390], [2390, 2680], [2680, 3250]],
    float)


def _climatology(horiz=(), seed=0, n_sw=14, n_lw=16):
    """Build a time-selected climatology mapping with trailing ``horiz`` axes."""
    rng = np.random.default_rng(seed)

    def field(n, lo, hi):
        return jnp.asarray(rng.uniform(lo, hi, (n,) + horiz), dtype=jnp.float32)

    def fractions():
        raw = rng.uniform(0.1, 1.0, (_TOP.shape[0],) + horiz)
        return jnp.asarray(raw / raw.sum(axis=0), dtype=jnp.float32)

    return {
        "aod_sw_coarse": field(n_sw, 0.02, 0.3),
        "ssa_sw_coarse": field(n_sw, 0.7, 0.99),
        "asy_sw_coarse": field(n_sw, 0.6, 0.9),
        "aod_sw_fine": field(n_sw, 0.01, 0.1),
        "ssa_sw_fine": field(n_sw, 0.8, 0.99),
        "asy_sw_fine": field(n_sw, 0.2, 0.7),
        "aod_lw_coarse": field(n_lw, 0.01, 0.2),
        "ssa_lw_coarse": field(n_lw, 0.1, 0.6),
        "layer_fraction_coarse": fractions(),
        "layer_fraction_fine": fractions(),
        "wavenumber_lower_sw": _SW_LO, "wavenumber_upper_sw": _SW_HI,
        "wavenumber_lower_lw": _LW_LO, "wavenumber_upper_lw": _LW_HI,
        "layer_bottom": _BOTTOM, "layer_top": _TOP,
    }


def _heights(nlev=12, surface=0.0, lid=30000.0, horiz=()):
    """Interface heights, top first, broadcast over ``horiz``."""
    z = jnp.linspace(lid, surface, nlev + 1)
    return jnp.broadcast_to(z.reshape((-1,) + (1,) * len(horiz)),
                            (nlev + 1,) + horiz)


def _centres(limits):
    return jnp.asarray(limits.mean(axis=1), dtype=jnp.float32)


class TestBandIndex(unittest.TestCase):
    def test_rrtmg_bands_map_to_themselves(self):
        centres = 0.5 * (_SW_LO + _SW_HI)
        np.testing.assert_array_equal(
            band_index(centres, _SW_LO, _SW_HI), np.arange(14))

    def test_rrtmgp_bands_are_a_one_to_one_reordering(self):
        sw = np.asarray(band_index(_centres(_RRTMGP_SW), _SW_LO, _SW_HI))
        # RRTMGP's first band is the far near-infrared one the climatology
        # stores last; the rest run in the opposite direction.
        np.testing.assert_array_equal(sw, [13, *range(12, -1, -1)])
        lw = np.asarray(band_index(_centres(_RRTMGP_LW), _LW_LO, _LW_HI))
        np.testing.assert_array_equal(lw, np.arange(15, -1, -1))

    def test_broadband_takes_the_550nm_band(self):
        index = band_index(jnp.asarray([1.0e7 / 550.0]), _SW_LO, _SW_HI)
        self.assertEqual(int(index[0]), SW_WAVELENGTH_NM.index(550))

    def test_centre_outside_every_band_takes_the_nearest(self):
        index = band_index(jnp.asarray([60000.0, 100.0]), _SW_LO, _SW_HI)
        np.testing.assert_array_equal(index, [0, 13])


class TestDistributeLayerFractions(unittest.TestCase):
    def test_model_layers_on_the_climatology_layers_return_the_fractions(self):
        fraction = jnp.asarray([0.4, 0.3, 0.15, 0.1, 0.05])
        half = jnp.concatenate([_TOP[::-1], _BOTTOM[:1]])      # top first
        profile = distribute_layer_fractions(fraction, _BOTTOM, _TOP, half)
        np.testing.assert_allclose(profile, fraction[::-1], rtol=1e-6)

    def test_a_layer_split_by_the_model_is_shared_by_depth(self):
        fraction = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
        half = jnp.asarray([20000.0, 500.0, 125.0, 0.0])
        profile = distribute_layer_fractions(fraction, _BOTTOM, _TOP, half)
        np.testing.assert_allclose(profile, [0.0, 0.75, 0.25], atol=1e-6)

    def test_column_is_kept_over_orography_and_under_a_low_lid(self):
        fraction = jnp.asarray([0.4, 0.3, 0.15, 0.1, 0.05])
        for surface, lid in ((1500.0, 30000.0), (0.0, 4000.0),
                             (3000.0, 9000.0)):
            profile = distribute_layer_fractions(
                fraction, _BOTTOM, _TOP, _heights(20, surface, lid))
            self.assertAlmostEqual(float(profile.sum()), 1.0, places=5)
            self.assertTrue(bool((profile >= 0.0).all()))
        # Above the surface the shape is the climatology's own: over 1500 m
        # terrain the two layers below it vanish and the rest is rescaled.
        half = jnp.asarray([20000.0, 5000.0, 2000.0, 1500.0])
        profile = distribute_layer_fractions(fraction, _BOTTOM, _TOP, half)
        above = np.array([0.05, 0.1, 0.075])
        np.testing.assert_allclose(profile, above / above.sum(), rtol=1e-5)

    def test_column_outside_the_climatology_is_empty_not_nan(self):
        fraction = jnp.asarray([0.4, 0.3, 0.15, 0.1, 0.05])
        half = jnp.asarray([40000.0, 30000.0, 25000.0])
        profile = distribute_layer_fractions(fraction, _BOTTOM, _TOP, half)
        np.testing.assert_array_equal(profile, 0.0)

    def test_trailing_axes_broadcast_per_column(self):
        horiz = (3, 4)
        clim = _climatology(horiz, seed=1)
        surface = jnp.asarray(np.random.default_rng(2).uniform(
            0.0, 3000.0, horiz), dtype=jnp.float32)
        z = jnp.linspace(1.0, 0.0, 11).reshape(-1, 1, 1)
        half = surface[None] + z * (25000.0 - surface[None])
        grid = distribute_layer_fractions(
            clim["layer_fraction_coarse"], _BOTTOM, _TOP, half)
        self.assertEqual(grid.shape, (10,) + horiz)
        columns = distribute_layer_fractions(
            clim["layer_fraction_coarse"].reshape(5, -1), _BOTTOM, _TOP,
            half.reshape(11, -1))
        np.testing.assert_allclose(grid.reshape(10, -1), columns, rtol=1e-6)
        single = distribute_layer_fractions(
            clim["layer_fraction_coarse"][:, 1, 2], _BOTTOM, _TOP,
            half[:, 1, 2])
        np.testing.assert_allclose(grid[:, 1, 2], single, rtol=1e-6)


class TestNaturalAerosolOptics(unittest.TestCase):
    def setUp(self):
        self.clim = _climatology((6,), seed=3)
        self.half = _heights(15, horiz=(6,))
        self.sw = _centres(_RRTMGP_SW)
        self.lw = _centres(_RRTMGP_LW)
        self.out = natural_aerosol_optics(self.clim, self.half, self.sw, self.lw)
        self.sw_index = np.asarray(band_index(self.sw, _SW_LO, _SW_HI))
        self.lw_index = np.asarray(band_index(self.lw, _LW_LO, _LW_HI))

    def test_shapes(self):
        self.assertEqual(self.out["tau_sw"].shape, (14, 15, 6))
        self.assertEqual(self.out["absorption_lw"].shape, (16, 15, 6))
        self.assertEqual(self.out["od550_coarse"].shape, (6,))

    def test_columns_are_the_climatologys_per_radiation_band(self):
        c, i = self.clim, self.sw_index
        np.testing.assert_allclose(
            self.out["tau_sw"].sum(axis=1),
            c["aod_sw_coarse"][i] + c["aod_sw_fine"][i], rtol=2e-5)
        np.testing.assert_allclose(
            self.out["scat_sw"].sum(axis=1),
            c["aod_sw_coarse"][i] * c["ssa_sw_coarse"][i]
            + c["aod_sw_fine"][i] * c["ssa_sw_fine"][i], rtol=2e-5)
        np.testing.assert_allclose(
            self.out["forward_sw"].sum(axis=1),
            c["aod_sw_coarse"][i] * c["ssa_sw_coarse"][i] * c["asy_sw_coarse"][i]
            + c["aod_sw_fine"][i] * c["ssa_sw_fine"][i] * c["asy_sw_fine"][i],
            rtol=2e-5)

    def test_longwave_is_coarse_mode_absorption_on_the_coarse_profile(self):
        c, j = self.clim, self.lw_index
        np.testing.assert_allclose(
            self.out["absorption_lw"].sum(axis=1),
            c["aod_lw_coarse"][j] * (1.0 - c["ssa_lw_coarse"][j]), rtol=2e-5)
        coarse = distribute_layer_fractions(
            c["layer_fraction_coarse"], _BOTTOM, _TOP, self.half)
        shape = self.out["absorption_lw"] / self.out["absorption_lw"].sum(
            axis=1, keepdims=True)
        np.testing.assert_allclose(shape[0], coarse, rtol=1e-4, atol=1e-7)

    def test_modes_keep_their_own_vertical_profiles(self):
        clim = dict(self.clim)
        clim["aod_sw_fine"] = jnp.zeros_like(clim["aod_sw_fine"])
        coarse_only = natural_aerosol_optics(clim, self.half, self.sw, self.lw)
        clim = dict(self.clim)
        clim["aod_sw_coarse"] = jnp.zeros_like(clim["aod_sw_coarse"])
        fine_only = natural_aerosol_optics(clim, self.half, self.sw, self.lw)
        np.testing.assert_allclose(
            coarse_only["tau_sw"] + fine_only["tau_sw"], self.out["tau_sw"],
            rtol=1e-5)
        self.assertFalse(np.allclose(
            coarse_only["tau_sw"][0] / coarse_only["tau_sw"][0].sum(axis=0),
            fine_only["tau_sw"][0] / fine_only["tau_sw"][0].sum(axis=0),
            rtol=1e-2))

    def test_diagnostics_are_the_550nm_columns(self):
        i550 = SW_WAVELENGTH_NM.index(550)
        np.testing.assert_array_equal(
            self.out["od550_coarse"], self.clim["aod_sw_coarse"][i550])
        np.testing.assert_array_equal(
            self.out["od550_fine"], self.clim["aod_sw_fine"][i550])

    def test_scheme_without_longwave_bands(self):
        out = natural_aerosol_optics(
            self.clim, self.half, jnp.asarray([1.0e7 / 550.0]), jnp.zeros((0,)))
        self.assertEqual(out["tau_sw"].shape, (1, 15, 6))
        self.assertEqual(out["absorption_lw"].shape, (0, 15, 6))

    def test_gradients_match_a_central_difference(self):
        clim = _climatology((2,), seed=4)
        static = {k: clim[k] for k in (
            "wavenumber_lower_sw", "wavenumber_upper_sw",
            "wavenumber_lower_lw", "wavenumber_upper_lw",
            "layer_bottom", "layer_top")}
        live = {k: v for k, v in clim.items() if k not in static}
        # Interfaces off the climatology's layer bounds, where the overlap
        # is differentiable in the heights.
        half = _heights(9, surface=137.0, lid=23110.0, horiz=(2,))

        def f(live, half):
            out = natural_aerosol_optics(
                {**live, **static}, half, _centres(_RRTMGP_SW),
                _centres(_RRTMGP_LW))
            return (out["tau_sw"], out["scat_sw"], out["forward_sw"],
                    out["absorption_lw"])

        check_gradients(f, (live, half), rtol=2e-2)


class TestAddToAerosol(unittest.TestCase):
    def setUp(self):
        self.clim = _climatology((4,), seed=5)
        self.optics = natural_aerosol_optics(
            self.clim, _heights(8, horiz=(4,)), _centres(_RRTMGP_SW),
            _centres(_RRTMGP_LW))

    def test_on_a_clean_slot_the_result_is_the_background(self):
        out = add_to_aerosol(AerosolData.zeros((4,), 8), self.optics)
        o = self.optics
        np.testing.assert_allclose(out.aod_sw_per_band, o["tau_sw"], rtol=1e-6)
        np.testing.assert_allclose(
            out.aod_sw_per_band * out.ssa_sw_per_band, o["scat_sw"], rtol=1e-5)
        np.testing.assert_allclose(
            out.aod_sw_per_band * out.ssa_sw_per_band * out.asy_sw_per_band,
            o["forward_sw"], rtol=1e-5)
        np.testing.assert_allclose(
            out.aod_lw_per_band, o["absorption_lw"], rtol=1e-6)
        np.testing.assert_array_equal(out.ssa_lw_per_band, 0.0)

    def test_plume_optics_combine_as_weighted_sums(self):
        rng = np.random.default_rng(6)
        shape = (14, 8, 4)
        plumes = AerosolData.zeros((4,), 8).copy(
            aod_sw_per_band=jnp.asarray(rng.uniform(0.0, 0.05, shape)),
            ssa_sw_per_band=jnp.asarray(rng.uniform(0.85, 0.97, shape)),
            asy_sw_per_band=jnp.asarray(rng.uniform(0.5, 0.7, shape)))
        out = add_to_aerosol(plumes, self.optics)
        tau = plumes.aod_sw_per_band + self.optics["tau_sw"]
        scat = (plumes.aod_sw_per_band * plumes.ssa_sw_per_band
                + self.optics["scat_sw"])
        forward = (plumes.aod_sw_per_band * plumes.ssa_sw_per_band
                   * plumes.asy_sw_per_band + self.optics["forward_sw"])
        np.testing.assert_allclose(out.aod_sw_per_band, tau, rtol=1e-6)
        np.testing.assert_allclose(out.ssa_sw_per_band, scat / tau, rtol=1e-5)
        np.testing.assert_allclose(out.asy_sw_per_band, forward / scat,
                                   rtol=1e-5)

    def test_gradients_match_a_central_difference(self):
        # A slot with optical depth in every layer and band, so no entry sits
        # on the empty-layer limits (where the completing divisions have a
        # corner at zero optical depth and only a one-sided derivative).
        rng = np.random.default_rng(7)
        slot = {
            "aod_sw_per_band": rng.uniform(0.01, 0.05, (14, 8, 4)),
            "ssa_sw_per_band": rng.uniform(0.85, 0.97, (14, 8, 4)),
            "asy_sw_per_band": rng.uniform(0.5, 0.7, (14, 8, 4)),
            "aod_lw_per_band": rng.uniform(0.005, 0.02, (16, 8, 4)),
            "ssa_lw_per_band": rng.uniform(0.2, 0.6, (16, 8, 4)),
        }
        slot = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in slot.items()}

        def f(slot, optics):
            out = add_to_aerosol(AerosolData.zeros((4,), 8).copy(**slot),
                                 optics)
            return (out.aod_sw_per_band, out.ssa_sw_per_band,
                    out.asy_sw_per_band, out.aod_lw_per_band,
                    out.ssa_lw_per_band)

        check_gradients(f, (slot, self.optics), rtol=2e-2)

    def test_twomey_and_550nm_plume_fields_are_untouched(self):
        plumes = AerosolData.zeros((4,), 8).copy(
            cdnc_factor=jnp.full((4,), 1.3), Nccn=jnp.full((4,), 80.0),
            aod_total=jnp.full((4,), 0.04),
            aod_anthropogenic=jnp.full((4,), 0.04),
            aod_background=jnp.full((4,), 0.02),
            aod_profile=jnp.full((8, 4), 0.005))
        out = add_to_aerosol(plumes, self.optics)
        for name in ("cdnc_factor", "Nccn", "aod_total", "aod_anthropogenic",
                     "aod_background", "aod_profile", "ssa_profile",
                     "asy_profile", "angstrom"):
            np.testing.assert_array_equal(getattr(out, name),
                                          getattr(plumes, name))

    def test_longwave_absorber_dilutes_existing_scattering(self):
        shape = (16, 8, 4)
        before = AerosolData.zeros((4,), 8).copy(
            aod_lw_per_band=jnp.full(shape, 0.01),
            ssa_lw_per_band=jnp.full(shape, 0.5),
            asy_lw_per_band=jnp.full(shape, 0.3))
        out = add_to_aerosol(before, self.optics)
        tau = 0.01 + self.optics["absorption_lw"]
        np.testing.assert_allclose(out.aod_lw_per_band, tau, rtol=1e-6)
        np.testing.assert_allclose(out.aod_lw_per_band * out.ssa_lw_per_band,
                                   0.005, rtol=1e-5)
        np.testing.assert_array_equal(out.asy_lw_per_band,
                                      before.asy_lw_per_band)

    def test_empty_layers_take_the_reference_limits(self):
        optics = {k: jnp.zeros_like(v) for k, v in self.optics.items()}
        out = add_to_aerosol(AerosolData.zeros((4,), 8), optics)
        np.testing.assert_array_equal(out.ssa_sw_per_band, 1.0)
        np.testing.assert_array_equal(out.asy_sw_per_band, 0.0)
        grads = jax.grad(lambda t: add_to_aerosol(
            AerosolData.zeros((4,), 8), {**optics, "tau_sw": t}
        ).ssa_sw_per_band.sum())(optics["tau_sw"])
        self.assertTrue(bool(jnp.isfinite(grads).all()))

    def test_slot_without_longwave_bands_is_left_alone(self):
        slot = AerosolData.zeros((4,), 8, n_bnd_sw=1, n_bnd_lw=0)
        optics = natural_aerosol_optics(
            self.clim, _heights(8, horiz=(4,)), jnp.asarray([1.0e7 / 550.0]),
            jnp.zeros((0,)))
        out = add_to_aerosol(slot, optics)
        self.assertEqual(out.aod_sw_per_band.shape, (1, 8, 4))
        self.assertEqual(out.aod_lw_per_band.shape, (0, 8, 4))


class TestTerm(unittest.TestCase):
    """The term in both hosts: grid ``(kx, ix, il)`` and column ``(kx, ncols)``."""

    nlev, nlon, nlat = 8, 4, 3

    def _forcing(self):
        # Forcing fields are grid-shaped (..., nlon, nlat) in either host.
        return SimpleNamespace(
            natural_aerosol=_climatology((self.nlon, self.nlat), seed=7))

    def _run(self, horiz, forcing, band_config=None):
        state = SimpleNamespace(
            temperature=jnp.zeros((self.nlev,) + horiz, dtype=jnp.float32))
        surface = jnp.asarray(np.random.default_rng(8).uniform(
            0.0, 2000.0, (self.nlon, self.nlat)), dtype=jnp.float32)
        z = jnp.linspace(1.0, 0.0, self.nlev + 1).reshape(-1, 1, 1)
        half = (surface[None] + z * (26000.0 - surface[None])).reshape(
            (self.nlev + 1,) + horiz)
        band_config = band_config or RadiationBandConfig(
            lw_band_centers_nm=tuple(1.0e7 / _RRTMGP_LW.mean(axis=1)),
            sw_band_centers_nm=tuple(1.0e7 / _RRTMGP_SW.mean(axis=1)))
        n_sw = len(band_config.sw_band_centers_nm)
        n_lw = len(band_config.lw_band_centers_nm)
        diagnostics = {
            "aerosol": AerosolData.zeros(horiz, self.nlev, n_bnd_sw=n_sw,
                                         n_bnd_lw=n_lw),
            "height_half": half, "_band_config": band_config,
        }
        return Macv2NaturalAerosol()(state, diagnostics, forcing, None)

    def test_declares_what_it_reads_and_writes(self):
        term = Macv2NaturalAerosol()
        self.assertEqual(term.consumed_forcing_fields(), ("natural_aerosol",))
        self.assertIn("aerosol", term.requires)
        self.assertIn("height_half", term.requires)
        self.assertIn("aerosol", term.provides)

    def test_no_climatology_passes_the_slot_through(self):
        horiz = (self.nlon * self.nlat,)
        tendency, out = self._run(horiz, SimpleNamespace(natural_aerosol=None))
        np.testing.assert_array_equal(out["aerosol"].aod_sw_per_band, 0.0)
        np.testing.assert_array_equal(out["natural_od550_coarse"],
                                      np.zeros(horiz))
        np.testing.assert_array_equal(out["natural_od550_fine"],
                                      np.zeros(horiz))
        np.testing.assert_array_equal(tendency.temperature, 0.0)

    def test_grid_and_column_hosts_agree_per_column(self):
        forcing = self._forcing()
        _, grid = self._run((self.nlon, self.nlat), forcing)
        _, cols = self._run((self.nlon * self.nlat,), forcing)
        for name in ("aod_sw_per_band", "ssa_sw_per_band", "asy_sw_per_band",
                     "aod_lw_per_band", "ssa_lw_per_band"):
            a = np.asarray(getattr(grid["aerosol"], name))
            b = np.asarray(getattr(cols["aerosol"], name))
            # A model interface a few metres from a climatology bound makes
            # a thin overlap that float32 heights resolve to ~1e-4 relative.
            np.testing.assert_allclose(a.reshape(b.shape), b, rtol=1e-4,
                                       atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(grid["natural_od550_coarse"]).reshape(-1),
            cols["natural_od550_coarse"], rtol=1e-6)
        self.assertGreater(float(cols["aerosol"].aod_sw_per_band.sum(axis=1).min()),
                           0.0)

    def test_columns_carry_the_climatology_of_their_own_grid_cell(self):
        forcing = self._forcing()
        _, cols = self._run((self.nlon * self.nlat,), forcing)
        i550 = SW_WAVELENGTH_NM.index(550)
        expected = forcing.natural_aerosol["aod_sw_coarse"][i550].reshape(-1)
        np.testing.assert_array_equal(cols["natural_od550_coarse"], expected)
        # RRTMGP's 550 nm band is its eleventh; its column sum is the cell's.
        column = cols["aerosol"].aod_sw_per_band[10].sum(axis=0)
        total = (forcing.natural_aerosol["aod_sw_coarse"][i550]
                 + forcing.natural_aerosol["aod_sw_fine"][i550]).reshape(-1)
        np.testing.assert_allclose(column, total, rtol=2e-5)

    def test_broadband_host_gets_one_shortwave_band_and_no_longwave(self):
        forcing = self._forcing()
        _, out = self._run((self.nlon * self.nlat,), forcing,
                           band_config=RadiationBandConfig.broadband())
        self.assertEqual(out["aerosol"].aod_sw_per_band.shape,
                         (1, self.nlev, self.nlon * self.nlat))
        self.assertEqual(out["aerosol"].aod_lw_per_band.shape[0], 0)

    def test_jit(self):
        forcing = self._forcing()
        horiz = (self.nlon * self.nlat,)

        @jax.jit
        def run(climatology):
            _, out = self._run(horiz, SimpleNamespace(natural_aerosol=climatology))
            return out["aerosol"].aod_sw_per_band

        np.testing.assert_allclose(
            run(forcing.natural_aerosol),
            self._run(horiz, forcing)[1]["aerosol"].aod_sw_per_band, rtol=1e-4,
            atol=1e-7)


class TestWithThePlumes(unittest.TestCase):
    """The pair the ECHAM presets compose: plumes, then the background."""

    def test_repeated_steps_do_not_accumulate(self):
        # The ``aerosol`` slot is carried from step to step. The plumes
        # rewrite it and the background adds to it, so feeding one step's
        # slot into the next must reproduce the same optics.
        from jcm.forcing import ForcingData
        from jcm.physics.aerosol import Macv2SpAerosol
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nlon, nlat = coords.horizontal.nodal_shape
        ncols, nlev = nlon * nlat, 8
        plumes, background = Macv2SpAerosol(), Macv2NaturalAerosol()
        plumes.cache_coords(coords)
        band_config = RadiationBandConfig(
            lw_band_centers_nm=tuple(1.0e7 / _RRTMGP_LW.mean(axis=1)),
            sw_band_centers_nm=tuple(1.0e7 / _RRTMGP_SW.mean(axis=1)))
        plumes.cache_band_config(band_config)
        half = jnp.broadcast_to(
            jnp.linspace(28000.0, 0.0, nlev + 1)[:, None], (nlev + 1, ncols))
        state = SimpleNamespace(temperature=jnp.zeros((nlev, ncols)))
        forcing = ForcingData.ones((nlon, nlat)).copy(
            natural_aerosol=_climatology((nlon, nlat), seed=12))
        terrain = SimpleNamespace(orog=jnp.zeros((nlon, nlat)))
        diagnostics = {
            "height_half": half, "height_full": 0.5 * (half[1:] + half[:-1]),
            "layer_thickness": half[:-1] - half[1:],
            "_band_config": band_config,
            **plumes.initial_carry_state(coords),
        }

        def step(diagnostics):
            _, diagnostics = plumes(state, diagnostics, forcing, terrain)
            _, diagnostics = background(state, diagnostics, forcing, terrain)
            return diagnostics

        first = step(diagnostics)
        second = step(first)
        third = step(second)
        for name in ("aod_sw_per_band", "ssa_sw_per_band", "asy_sw_per_band",
                     "aod_lw_per_band", "ssa_lw_per_band", "asy_lw_per_band"):
            np.testing.assert_array_equal(getattr(third["aerosol"], name),
                                          getattr(first["aerosol"], name))
        self.assertGreater(float(first["aerosol"].aod_lw_per_band.max()), 0.0)
        # The plume diagnostics and Twomey factor are the plumes' own.
        alone = plumes(state, diagnostics, forcing, terrain)[1]["aerosol"]
        np.testing.assert_array_equal(first["aerosol"].cdnc_factor,
                                      alone.cdnc_factor)
        np.testing.assert_array_equal(first["aerosol"].aod_total,
                                      alone.aod_total)
        self.assertTrue(bool(jnp.all(
            first["aerosol"].aod_sw_per_band >= alone.aod_sw_per_band)))


class TestComposition(unittest.TestCase):
    """Where the term is composed by default."""

    def test_term_list_presets(self):
        from pathlib import Path

        from omegaconf import OmegaConf

        import jcm

        presets = Path(jcm.__file__).parent / "config" / "physics"
        for name, composed in (("echam", True), ("echam-rrtmgp-2m", True),
                               ("echam-rrtmgp-2m-cosp", True),
                               ("echam-emulated-2m", False)):
            with self.subTest(preset=name):
                terms = list(OmegaConf.load(presets / f"{name}.yaml").terms)
                self.assertEqual("macv2_natural_aerosol" in terms, composed)
                if composed:
                    # Directly after the term that owns the slot, and before
                    # the radiation that reads it.
                    at = terms.index("macv2_natural_aerosol")
                    self.assertEqual(terms[at - 1], "macv2_sp_aerosol")
                    self.assertLess(at, terms.index("rrtmgp_radiation"))

    def test_factory_default_follows_aerosol_module_and_radiation(self):
        from jcm.physics.echam.echam_terms import echam_physics

        names = [t.name for t in echam_physics().terms]
        at = names.index("macv2_natural_aerosol")
        self.assertEqual(names[at - 1], "macv2_sp_aerosol")
        self.assertLess(at, names.index("rrtmgp_radiation"))
        names = [t.name for t in echam_physics(natural_aerosol=False).terms]
        self.assertNotIn("macv2_natural_aerosol", names)

    def test_factory_refuses_the_background_with_prognostic_aerosol(self):
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, "natural_aerosol=True needs"):
            echam_physics(aerosol_module="jam", cloud_scheme="2m",
                          natural_aerosol=True)


if __name__ == "__main__":
    unittest.main()
