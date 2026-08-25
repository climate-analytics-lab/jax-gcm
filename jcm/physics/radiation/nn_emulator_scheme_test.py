"""Tests for the NN emulator radiation scheme.

Every test here drives ``radiation_scheme_emulated`` for real rather than
asserting shapes on weights in isolation: jax-gcm#702 went unnoticed
because the existing suite did the latter, so ``AerosolData`` grew six
per-band fields underneath a vmap that never ran.
"""

import os
import tempfile
import unittest
from datetime import datetime

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from jax_solar import OrbitalTime

from jcm.forcing import SolarGeometry
from jcm.physics.aerosol.aerosol_types import AerosolData
from jcm.physics.radiation.nn_emulator import (
    InputScaling,
    init_emulator_weights,
    n_input_features,
    save_emulator_weights,
)
from jcm.physics.radiation.nn_emulator_scheme import radiation_scheme_emulated
from jcm.physics.radiation.radiation_types import RadiationParameters

NLEV, N_BND_SW, N_BND_LW = 20, 14, 16


def _solar():
    """Equinox noon at lon 0 — a lit column, so the SW path really runs."""
    ot = OrbitalTime.from_datetime(
        jdt.Datetime.from_pydatetime(datetime(2024, 3, 20, 12, 0)))
    return SolarGeometry(
        tyear=jnp.asarray(ot.orbital_phase / (2.0 * jnp.pi), dtype=jnp.float32),
        orbital_phase=jnp.asarray(ot.orbital_phase, dtype=jnp.float32),
        synodic_phase=jnp.asarray(ot.synodic_phase, dtype=jnp.float32),
    )


def _aerosol(ncols=None, nlev=NLEV):
    """Aerosol with non-zero per-band optics, per-column or single."""
    trailing = () if ncols is None else (ncols,)
    shape = (nlev,) + trailing

    def band(n_bnd, value):
        return jnp.full((n_bnd,) + shape, value)

    return AerosolData.zeros(
        trailing or (1,), nlev, n_bnd_sw=N_BND_SW, n_bnd_lw=N_BND_LW,
    ).copy(
        aod_sw_per_band=band(N_BND_SW, 0.02),
        ssa_sw_per_band=band(N_BND_SW, 0.92),
        asy_sw_per_band=band(N_BND_SW, 0.68),
        aod_lw_per_band=band(N_BND_LW, 0.005),
        ssa_lw_per_band=band(N_BND_LW, 0.40),
        asy_lw_per_band=band(N_BND_LW, 0.55),
        aod_profile=jnp.full(shape, 0.02),
        ssa_profile=jnp.full(shape, 0.92),
        asy_profile=jnp.full(shape, 0.68),
        cdnc_factor=jnp.ones(trailing or ()),
        aod_total=jnp.full(trailing or (), 0.2),
        aod_anthropogenic=jnp.full(trailing or (), 0.1),
        aod_background=jnp.full(trailing or (), 0.1),
        Nccn=jnp.full(trailing or (), 100.0),
        angstrom=jnp.full(trailing or (), 1.5),
    )


def _column(ncols=None):
    """Profiles and surface values for one column, or ncols of them."""
    trailing = () if ncols is None else (ncols,)
    p_half = jnp.linspace(1.0, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[1:] + p_half[:-1])

    def prof(x, n=NLEV):
        col = jnp.full((n,), x)
        return col if ncols is None else jnp.broadcast_to(
            col[:, None], (n,) + trailing)

    def scalar(x):
        return jnp.asarray(x) if ncols is None else jnp.full(trailing, x)

    temperature = jnp.linspace(220.0, 295.0, NLEV)
    return dict(
        temperature=(temperature if ncols is None else jnp.broadcast_to(
            temperature[:, None], (NLEV,) + trailing)),
        specific_humidity=prof(5e-3),
        pressure_levels=(p_full if ncols is None else jnp.broadcast_to(
            p_full[:, None], (NLEV,) + trailing)),
        pressure_interfaces=(p_half if ncols is None else jnp.broadcast_to(
            p_half[:, None], (NLEV + 1,) + trailing)),
        layer_thickness=prof(500.0), air_density=prof(0.8),
        cloud_water=prof(1e-4), cloud_ice=prof(5e-5),
        cloud_fraction=prof(0.5),
        surface_temperature=scalar(288.0),
        surface_albedo_vis=scalar(0.1), surface_albedo_nir=scalar(0.25),
        surface_emissivity=scalar(0.98),
        ozone_vmr=prof(5e-6),
    )


def _run(band_mode="per_band", specific_humidity=None, weights=None):
    """Drive the scheme on a single column."""
    col = _column()
    if specific_humidity is not None:
        col["specific_humidity"] = specific_humidity
    n_sw = n_input_features(band_mode, N_BND_SW)
    n_lw = n_input_features(band_mode, N_BND_LW)
    if weights is None:
        weights = init_emulator_weights(sw_features=n_sw, lw_features=n_lw)
    return radiation_scheme_emulated(
        col["temperature"], col["specific_humidity"],
        col["pressure_levels"], col["pressure_interfaces"],
        col["layer_thickness"], col["air_density"],
        col["cloud_water"], col["cloud_ice"], col["cloud_fraction"],
        col["surface_temperature"], col["surface_albedo_vis"],
        col["surface_albedo_nir"], col["surface_emissivity"],
        _solar(), 0.0, 0.0, RadiationParameters.default(), _aerosol(),
        col["ozone_vmr"], 400e-6, weights, None, None, band_mode,
    )


class BandModeTest(unittest.TestCase):
    """Each aerosol band handling must trace and stay finite."""

    def test_every_band_mode_runs_and_is_finite(self):
        # 10 base features (T, log p, h2o, o3, lwp, iwp, cloud fraction,
        # mu0, r_eff_liq, r_eff_ice) plus 3 per aerosol band.
        for mode, n_sw in (("none", 10), ("broadband", 13), ("per_band", 52)):
            with self.subTest(mode=mode):
                self.assertEqual(n_input_features(mode, N_BND_SW), n_sw)
                tend, diag = _run(mode)
                self.assertTrue(
                    np.isfinite(np.asarray(tend.temperature_tendency)).all())
                self.assertTrue(
                    np.isfinite(np.asarray(diag.sw_flux_up_clear)).all())

    def test_unknown_band_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _run("per-band")


class SolarBoundaryTest(unittest.TestCase):
    """The SW path must actually be lit, and its boundary exact."""

    def test_toa_downward_flux_is_the_incoming_flux(self):
        _, diag = _run()
        toa = float(diag.toa_sw_down)
        self.assertGreater(toa, 1000.0)
        # Both sky states are pinned to the incoming flux by construction,
        # so this cannot drift with the weights.
        self.assertAlmostEqual(float(diag.sw_flux_down[0]), toa, places=2)
        self.assertAlmostEqual(float(diag.sw_flux_down_clear[0]), toa, places=2)

    def test_clear_sky_toa_matches_the_profile_endpoint(self):
        """Pins the interface ordering: index 0 is TOA, not the surface."""
        _, diag = _run()
        self.assertAlmostEqual(
            float(diag.toa_sw_up_clear), float(diag.sw_flux_up_clear[0]),
            places=4)
        self.assertAlmostEqual(
            float(diag.toa_lw_up_clear), float(diag.lw_flux_up_clear[0]),
            places=4)


class DiagnosticsContractTest(unittest.TestCase):
    """The published RadiationData must honour the cross-scheme contracts."""

    def test_stored_zenith_is_the_actual_cosine_not_the_feature_clip(self):
        # The cached-step rescale divides by the stored cos_zenith, so it
        # must be the ACTUAL solve-time cosine (matching RRTMGP and
        # current_cos_zenith), not the min_cos_zenith clip the network
        # feature uses — the clip made twilight columns rescale by
        # mu_now/0.035 (PR #730 review). Longitude 180 at the fixture's
        # noon is midnight: the actual cosine is negative, which the
        # clipped value never is.
        from jax_solar import get_solar_sin_altitude

        col = _column()
        weights = init_emulator_weights(
            sw_features=n_input_features("per_band", N_BND_SW),
            lw_features=n_input_features("per_band", N_BND_LW),
        )
        solar = _solar()
        ot = OrbitalTime(
            orbital_phase=solar.orbital_phase,
            synodic_phase=solar.synodic_phase,
        )
        for lon in (0.0, 180.0):
            with self.subTest(lon=lon):
                _, diag = radiation_scheme_emulated(
                    col["temperature"], col["specific_humidity"],
                    col["pressure_levels"], col["pressure_interfaces"],
                    col["layer_thickness"], col["air_density"],
                    col["cloud_water"], col["cloud_ice"],
                    col["cloud_fraction"], col["surface_temperature"],
                    col["surface_albedo_vis"], col["surface_albedo_nir"],
                    col["surface_emissivity"], solar, lon, 0.0,
                    RadiationParameters.default(), _aerosol(),
                    col["ozone_vmr"], 400e-6, weights, None, None,
                    "per_band",
                )
                expected = get_solar_sin_altitude(ot, lon, 0.0)
                np.testing.assert_allclose(
                    np.asarray(diag.cos_zenith), np.asarray(expected),
                    rtol=1e-6,
                )
        self.assertLess(float(get_solar_sin_altitude(ot, 180.0, 0.0)), 0.0)

    def test_aerosol_free_slots_are_withheld(self):
        # The emulator never runs an aerosol-free solve; publishing the
        # zero defaults turns a downstream ERFari into the whole all-sky
        # flux (jax-gcm#647). Clear-sky keys are real network outputs and
        # must NOT be withheld.
        from jcm.physics.radiation.aerosol_free import NOA_KEYS
        from jcm.physics.radiation.nn_emulator_scheme import (
            NNEmulatorRadiation,
        )

        keys = set(NNEmulatorRadiation().withheld_output_keys())
        for k in NOA_KEYS:
            self.assertIn(f"radiation.{k}_noa", keys)
            self.assertIn(f"radiation.noa_frac_{k}", keys)
        self.assertNotIn("radiation.toa_sw_up_clear", keys)
        self.assertNotIn("radiation.sw_flux_up_clear", keys)


class NegativeHumidityTest(unittest.TestCase):
    """Regression for jax-gcm#702 defect 3.

    A spectral dycore delivers small negative specific humidity from
    Gibbs ringing, and the unguarded quartic root turned that into NaN
    across the whole column.
    """

    def test_negative_humidity_does_not_produce_nan(self):
        q = jnp.full((NLEV,), 5e-3).at[0].set(-1e-12)
        tend, _ = _run(specific_humidity=q)
        self.assertTrue(
            np.isfinite(np.asarray(tend.temperature_tendency)).all())


class PerBandVmapTest(unittest.TestCase):
    """Regression for jax-gcm#702 defect 1.

    The term maps ``aerosol_data`` with ``in_axes=0``, which applies to
    every leaf. The six per-band fields kept their band axis leading and
    vmap rejected the trace, blocking every ECHAM composition.
    """

    def test_vmaps_over_columns_with_per_band_aerosol(self):
        ncols = 4
        col = _column(ncols)
        aer = _aerosol(ncols)

        def to_col(arr, n_bnd):
            return arr.reshape(n_bnd, NLEV, ncols).transpose(2, 0, 1)

        aer = aer.copy(
            aod_profile=aer.aod_profile.T, ssa_profile=aer.ssa_profile.T,
            asy_profile=aer.asy_profile.T,
            aod_sw_per_band=to_col(aer.aod_sw_per_band, N_BND_SW),
            ssa_sw_per_band=to_col(aer.ssa_sw_per_band, N_BND_SW),
            asy_sw_per_band=to_col(aer.asy_sw_per_band, N_BND_SW),
            aod_lw_per_band=to_col(aer.aod_lw_per_band, N_BND_LW),
            ssa_lw_per_band=to_col(aer.ssa_lw_per_band, N_BND_LW),
            asy_lw_per_band=to_col(aer.asy_lw_per_band, N_BND_LW),
        )
        n_sw = n_input_features("per_band", N_BND_SW)
        n_lw = n_input_features("per_band", N_BND_LW)
        weights = init_emulator_weights(sw_features=n_sw, lw_features=n_lw)

        tend, diag = jax.vmap(
            radiation_scheme_emulated,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                     None, 0, 0, None, 0, 1, None, None, None, None, None),
            out_axes=(0, 0), axis_size=ncols,
        )(
            col["temperature"], col["specific_humidity"],
            col["pressure_levels"], col["pressure_interfaces"],
            col["layer_thickness"], col["air_density"],
            col["cloud_water"], col["cloud_ice"], col["cloud_fraction"],
            col["surface_temperature"], col["surface_albedo_vis"],
            col["surface_albedo_nir"], col["surface_emissivity"],
            _solar(), jnp.zeros(ncols), jnp.zeros(ncols),
            RadiationParameters.default(), aer, col["ozone_vmr"], 400e-6,
            weights, None, None, "per_band",
        )
        self.assertEqual(
            tend.temperature_tendency.shape, (ncols, NLEV))
        self.assertEqual(diag.sw_flux_up_clear.shape, (ncols, NLEV + 1))
        self.assertTrue(
            np.isfinite(np.asarray(tend.temperature_tendency)).all())


class TermComputeFullTest(unittest.TestCase):
    """Drive ``NNEmulatorRadiation._compute_full`` itself.

    :class:`PerBandVmapTest` transposes the per-band arrays in the test,
    so it cannot catch a regression in the term's own reshape block —
    which is where jax-gcm#702 defect 1 actually lived. This feeds the
    term the production layout ``(n_bnd, nlev, ncols)`` and lets it do
    the transposing.
    """

    def _run_term(self):
        from types import SimpleNamespace

        from dinosaur.sigma_coordinates import SigmaCoordinates

        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics.chemistry.simple_chemistry import ChemistryData
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
        from jcm.physics.radiation.radiation_types import RadiationData
        from jcm.physics_interface import PhysicsState
        from jcm.utils import get_coords

        nlev = 8
        coords = get_coords(
            SigmaCoordinates.equidistant(nlev), spectral_truncation=21)
        nlon, nlat = coords.horizontal.nodal_shape
        ncols = nlon * nlat
        shape = (nlev, ncols)

        n_sw = n_input_features("per_band", N_BND_SW)
        n_lw = n_input_features("per_band", N_BND_LW)
        params = RadiationParameters.default(
            emulator_weights=init_emulator_weights(
                sw_features=n_sw, lw_features=n_lw),
        )
        term = NNEmulatorRadiation(params, band_mode="per_band")
        term.cache_coords(coords)

        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 260.0),
            specific_humidity=jnp.full(shape, 3e-3),
            tracers={"qc": jnp.full(shape, 1e-5),
                     "qi": jnp.full(shape, 5e-6)},
        )
        p_half = jnp.broadcast_to(
            jnp.linspace(1.0, 1.0e5, nlev + 1)[:, None], (nlev + 1, ncols))
        # Production layout: band axis leads, column axis trails.
        aerosol = _aerosol(ncols, nlev)
        diagnostics = {
            "pressure_full": jnp.broadcast_to(
                jnp.linspace(500.0, 9.9e4, nlev)[:, None], shape),
            "pressure_half": p_half,
            "layer_thickness": jnp.full(shape, 500.0),
            "air_density": jnp.full(shape, 0.8),
            "chemistry": ChemistryData.zeros((ncols,), nlev).copy(
                ozone_vmr=jnp.full(shape, 5.0)),
            "aerosol": aerosol,
            "radiation": RadiationData.zeros((ncols,), nlev).copy(
                surface_albedo_vis=jnp.full((ncols,), 0.1),
                surface_albedo_nir=jnp.full((ncols,), 0.25),
                surface_emissivity=jnp.full((ncols,), 0.98),
            ),
            "surface": SimpleNamespace(
                surface_temperature=jnp.full((ncols,), 288.0)),
            "clouds": CloudData.zeros((ncols,), nlev).copy(
                cloud_fraction=jnp.full(shape, 0.4)),
        }
        forcing = SimpleNamespace(solar=_solar(), co2_vmr=jnp.asarray(400.0))
        return term._compute_full(
            state, diagnostics, forcing, params), shape, ncols, nlev

    def test_zero_tendency_holds_on_the_cached_substep(self):
        """The cost-only mode must cover the cached branch too.

        Radiation sub-steps, so most calls rebuild the tendency from the
        heating rates on the carry rather than running the network.
        Zeroing only inside the compute branch left those steps applying
        untrained heating, and a T63L47 run reached 100% NaN by day 5.
        """
        from jcm.physics.radiation import cached_radiation_tendency
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
        from jcm.physics.radiation.radiation_types import RadiationData

        nlev, ncols = 8, 12
        shape = (nlev, ncols)
        # A carry holding heating rates as hostile as untrained weights give.
        radiation = RadiationData.zeros((ncols,), nlev).copy(
            sw_heating_rate=jnp.full(shape, 5e-3),
            lw_heating_rate=jnp.full(shape, -3e-3),
        )
        cached = cached_radiation_tendency(radiation, shape)
        self.assertGreater(float(jnp.abs(cached.temperature).max()), 0.0)

        term = NNEmulatorRadiation(zero_tendency=True)
        zeroed = term._zero_if_requested(cached)
        np.testing.assert_array_equal(
            np.asarray(zeroed.temperature), np.zeros(shape))

    def test_term_handles_production_per_band_layout(self):
        (tendency, rad_out), shape, ncols, nlev = self._run_term()
        self.assertEqual(tendency.temperature.shape, shape)
        self.assertEqual(rad_out.sw_flux_up_clear.shape, (nlev + 1, ncols))
        self.assertTrue(
            np.isfinite(np.asarray(tendency.temperature)).all())
        self.assertTrue(
            np.isfinite(np.asarray(rad_out.lw_flux_down_clear)).all())


class BandConfigTest(unittest.TestCase):
    """The emulator must run under RRTMGP's band structure.

    The band config follows the active radiation backend. With the
    emulator unrecognised it fell back to a single 550 nm SW band and no
    LW bands, so the aerosol term fed the network 10 features against
    weights sized for 49 — and, had the shapes happened to agree, would
    silently have supplied aerosol input unlike anything the training
    labels were generated under.
    """

    def test_emulator_term_selects_the_rrtmgp_bands(self):
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation
        from jcm.runners import _band_config_for_terms

        cfg = _band_config_for_terms([NNEmulatorRadiation()])
        self.assertEqual(len(cfg.sw_band_centers_nm), N_BND_SW)
        self.assertEqual(len(cfg.lw_band_centers_nm), N_BND_LW)

    def test_band_count_mismatch_raises_a_useful_error(self):
        """A mismatch must name the cause, not fail inside a GRU matmul."""
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        term = NNEmulatorRadiation(band_mode="per_band")
        with self.assertRaises(ValueError) as ctx:
            term._check_band_counts(1, 0)
        self.assertIn("input features", str(ctx.exception))
        self.assertIn("_band_config_for_terms", str(ctx.exception))


class WeightsFileTest(unittest.TestCase):
    """Loading a trained checkpoint onto the term.

    A checkpoint carries no record of the band structure other than its
    metadata, so the term must refuse anything that does not match its
    own configuration — the emulator has already been run once under the
    wrong band structure and only crashed by luck.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = os.path.join(self.tmpdir.name, "emulator.nc")

    def _write(self, band_mode="per_band", n_bnd_sw=N_BND_SW,
               n_bnd_lw=N_BND_LW, metadata=...):
        n_sw = n_input_features(band_mode, n_bnd_sw)
        n_lw = n_input_features(band_mode, n_bnd_lw)
        weights = init_emulator_weights(
            sw_features=n_sw, lw_features=n_lw, units=8,
            key=jax.random.key(5))
        if metadata is ...:
            metadata = {"band_mode": band_mode}
        save_emulator_weights(
            self.path, weights,
            InputScaling(x_max=jnp.ones(n_sw)),
            InputScaling(x_max=jnp.ones(n_lw)),
            metadata=metadata,
        )
        return weights

    def test_rejects_broadband_weights_trained_on_other_band_counts(self):
        # Under band_mode='broadband' the input width is band-count
        # independent, so only the stored metadata can catch a checkpoint
        # trained on a different band partition — which silently changes
        # every aerosol feature value (PR #730 review).
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        n_sw = n_input_features("broadband", 10)
        n_lw = n_input_features("broadband", 12)
        weights = init_emulator_weights(
            sw_features=n_sw, lw_features=n_lw, units=8,
            key=jax.random.key(9))
        save_emulator_weights(
            self.path, weights,
            InputScaling(x_max=jnp.ones(n_sw)),
            InputScaling(x_max=jnp.ones(n_lw)),
            metadata={"band_mode": "broadband",
                      "n_bnd_sw": 10, "n_bnd_lw": 12},
        )
        with self.assertRaises(ValueError) as ctx:
            NNEmulatorRadiation(
                band_mode="broadband", weights_file=self.path,
                n_bnd_sw=N_BND_SW, n_bnd_lw=N_BND_LW,
            )
        self.assertIn("n_bnd_sw", str(ctx.exception))

    def test_rejects_weights_without_clear_sky_channels(self):
        # A pre-clear-sky checkpoint has 2 output channels; JAX clamps the
        # out-of-bounds gather for channels 2/3, so without this check it
        # would load and silently publish the all-sky up-flux again as
        # "clear sky" (PR #730 review).
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        n_sw = n_input_features("per_band", N_BND_SW)
        n_lw = n_input_features("per_band", N_BND_LW)
        weights = init_emulator_weights(
            sw_features=n_sw, lw_features=n_lw, units=8, n_outputs=2,
            key=jax.random.key(7))
        save_emulator_weights(
            self.path, weights,
            InputScaling(x_max=jnp.ones(n_sw)),
            InputScaling(x_max=jnp.ones(n_lw)),
            metadata={"band_mode": "per_band"},
        )
        with self.assertRaises(ValueError) as ctx:
            NNEmulatorRadiation(band_mode="per_band", weights_file=self.path)
        self.assertIn("output", str(ctx.exception))
        self.assertIn("retrain", str(ctx.exception))

    def test_term_uses_the_weights_and_scalings_from_the_file(self):
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        saved = self._write()
        term = NNEmulatorRadiation(band_mode="per_band", weights_file=self.path)
        params = term.params.get_value()
        for a, b in zip(jax.tree.leaves(saved),
                        jax.tree.leaves(params.emulator_weights)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        # Both scalings come from the file, not left at None for the
        # scheme to substitute ones for.
        self.assertEqual(
            params.sw_scaling.x_max.shape,
            (n_input_features("per_band", N_BND_SW),))
        self.assertEqual(
            params.lw_scaling.x_max.shape,
            (n_input_features("per_band", N_BND_LW),))

    def test_no_weights_file_still_randomly_initialises(self):
        """The cost-only benchmark path must keep working untouched."""
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        term = NNEmulatorRadiation(band_mode="per_band")
        # Weights live in their own Param, separate from RadiationParameters,
        # so an optimizer can address the differentiable subtree alone.
        weights = term.weights.get_value()
        self.assertIsNotNone(weights)
        self.assertEqual(
            weights.sw.gru_fwd.kernel.shape[0],
            n_input_features("per_band", N_BND_SW))
        self.assertIsNone(term.params.get_value().sw_scaling)

    def test_band_mode_mismatch_raises(self):
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        self._write(band_mode="broadband")
        with self.assertRaises(ValueError) as ctx:
            NNEmulatorRadiation(band_mode="per_band", weights_file=self.path)
        message = str(ctx.exception)
        self.assertIn("'broadband'", message)
        self.assertIn("'per_band'", message)
        self.assertIn("band_mode", message)
        self.assertIn("echam-emulated-2m.yaml", message)

    def test_input_width_mismatch_raises(self):
        """Same band_mode, wrong band count — the silent-failure case."""
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        # Trained against 10 SW bands, run under the RRTMGP 14.
        self._write(band_mode="per_band", n_bnd_sw=10)
        with self.assertRaises(ValueError) as ctx:
            NNEmulatorRadiation(band_mode="per_band", weights_file=self.path)
        message = str(ctx.exception)
        self.assertIn("input features", message)
        self.assertIn("n_bnd_sw", message)
        self.assertIn(str(n_input_features("per_band", 10)), message)
        self.assertIn(str(n_input_features("per_band", N_BND_SW)), message)
        self.assertIn("echam-emulated-2m.yaml", message)

    def test_missing_band_mode_metadata_raises(self):
        from jcm.physics.radiation.nn_emulator_scheme import NNEmulatorRadiation

        self._write(metadata={})
        with self.assertRaises(ValueError) as ctx:
            NNEmulatorRadiation(band_mode="per_band", weights_file=self.path)
        self.assertIn("band_mode", str(ctx.exception))
        self.assertIn("save_emulator_weights", str(ctx.exception))


class GradientTest(unittest.TestCase):
    """A differentiable scheme is the whole point; gradients must reach it."""

    def test_gradient_flows_to_the_weights(self):
        n_sw = n_input_features("per_band", N_BND_SW)
        n_lw = n_input_features("per_band", N_BND_LW)
        weights = init_emulator_weights(sw_features=n_sw, lw_features=n_lw)

        def loss(w):
            tend, _ = _run("per_band", weights=w)
            return jnp.sum(tend.temperature_tendency ** 2)

        grads = jax.grad(loss)(weights)
        norm = float(jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))))
        self.assertTrue(np.isfinite(norm))
        self.assertGreater(norm, 0.0)


if __name__ == "__main__":
    unittest.main()
