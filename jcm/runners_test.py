"""Unit tests for ``jcm.runners`` and the Hydra config groups.

Verifies that each config-group combination resolves to a sensible model and
that a short integration step runs without raising. Kept deliberately cheap
so it can run in the regular pytest sweep — we do not test the full ECHAM
T85x47 grid here.
"""

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from jcm.runners import (
    build_coords,
    build_diffusion,
    build_model,
    build_physics,
    build_terrain,
    build_tracer_filter,
    configure_host_device_count,
    run,
)


CONFIG_DIR = str(Path(__file__).parent / "config")


def _compose(overrides=None):
    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


class TestTracerPositivityResolution(unittest.TestCase):
    """``diffusion.tracer_positivity`` resolution in build_tracer_filter.

    ``auto`` (the default) turns the mass-conserving positivity filter on only
    when the physics advects prognostic aerosol tracers (aerosol_module==jam),
    so aerosol-emission runs get the ringing fix by default while non-aerosol
    runs stay bit-identical; explicit true/false always wins.
    """

    def test_auto_on_for_prognostic_aerosol(self):
        self.assertIsNotNone(build_tracer_filter(_compose(["physics=echam-jam"])))

    def test_auto_off_for_diagnostic_aerosol(self):
        # echam term-list preset uses diagnostic MACv2-SP → no advected aerosol.
        self.assertIsNone(build_tracer_filter(_compose(["physics=echam"])))

    def test_auto_off_without_aerosol(self):
        self.assertIsNone(build_tracer_filter(_compose(["physics=held_suarez"])))

    def test_explicit_false_overrides_auto(self):
        self.assertIsNone(build_tracer_filter(
            _compose(["physics=echam-jam", "diffusion.tracer_positivity=false"])))

    def test_explicit_true_overrides_auto(self):
        self.assertIsNotNone(build_tracer_filter(
            _compose(["physics=echam", "diffusion.tracer_positivity=true"])))


class TestConfigComposition(unittest.TestCase):
    def test_default_compose(self):
        cfg = _compose()
        self.assertIn("speedy_convection", cfg.physics.terms)
        self.assertEqual(cfg.grid.vertical, "sigma")
        self.assertEqual(cfg.grid.layers, 8)
        self.assertEqual(cfg.run.time_step, 10)
        self.assertEqual(cfg.init.kind, "isothermal")
        self.assertEqual(cfg.terrain.kind, "aquaplanet")
        self.assertEqual(cfg.forcing.kind, "default")
        self.assertEqual(float(cfg.diffusion.scale), 1.0)

    def test_echam_compose(self):
        cfg = _compose([
            "physics=echam",
            "grid=echam_t42_l8_sigma",
        ])
        # The echam preset composes the supported ECHAM radiation (RRTMGP);
        # grey is a SPEEDY/debug scheme and must not be an ECHAM default.
        self.assertIn("rrtmgp_radiation", cfg.physics.terms)
        self.assertNotIn("grey_two_stream_radiation", cfg.physics.terms)
        self.assertIn("tiedtke_convection", cfg.physics.terms)
        self.assertEqual(cfg.grid.vertical, "sigma")

    def test_held_suarez_compose(self):
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
        ])
        self.assertIn("held_suarez", cfg.physics.terms)

    def test_run_smoke_overrides(self):
        cfg = _compose(["run=smoke"])
        self.assertEqual(cfg.run.total_time, 1)
        self.assertEqual(cfg.run.save_interval, 1)

    def test_init_jw_compose(self):
        cfg = _compose(["init=jw"])
        self.assertEqual(cfg.init.kind, "jw")


class TestBuilders(unittest.TestCase):
    def test_build_coords_speedy(self):
        cfg = _compose()
        coords = build_coords(cfg)
        self.assertEqual(coords.horizontal.nodal_shape, (96, 48))

    def test_build_coords_echam_sigma(self):
        cfg = _compose(["grid=echam_t42_l8_sigma"])
        coords = build_coords(cfg)
        self.assertEqual(coords.horizontal.nodal_shape, (128, 64))

    def test_build_physics_speedy(self):
        cfg = _compose()
        physics = build_physics(cfg)
        self.assertIsNotNone(physics)

    def test_build_physics_held_suarez(self):
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        physics = build_physics(cfg)
        self.assertIsNotNone(physics)

    def test_build_physics_param_overrides(self):
        # Override a per-term parameter via the new
        # ``physics.terms.<term>.params.<field>=...`` CLI path.
        cfg = _compose([
            "physics=echam",
            "grid=echam_t42_l8_sigma",
            "++physics.terms.tiedtke_convection.params.entrpen=4e-4",
        ])
        physics = build_physics(cfg)
        convection_term = next(
            t for t in physics.terms if t.category == "convection"
        )
        self.assertAlmostEqual(
            float(convection_term.params.value.entrpen), 4e-4,
        )

    def test_build_physics_curated_preset(self):
        # The echam-strong-conv preset bumps entrpen via the same
        # term-list pipeline.
        cfg = _compose([
            "physics=echam-strong-conv",
            "grid=echam_t42_l8_sigma",
        ])
        physics = build_physics(cfg)
        convection_term = next(
            t for t in physics.terms if t.category == "convection"
        )
        self.assertAlmostEqual(
            float(convection_term.params.value.entrpen), 4e-4,
        )

    def test_build_physics_swap_radiation_via_preset(self):
        # The echam-rrtmgp-2m preset composes rrtmgp_radiation in the
        # same logical slot as the base echam preset (the standalone
        # echam-rrtmgp preset was removed once physics=echam became
        # identical to it).
        cfg = _compose([
            "physics=echam-rrtmgp-2m",
            "grid=echam_t42_l8_sigma",
        ])
        physics = build_physics(cfg)
        rad_term = next(
            t for t in physics.terms if t.category == "radiation"
        )
        self.assertEqual(rad_term.name, "rrtmgp_radiation")

    def test_build_terrain_aquaplanet(self):
        cfg = _compose()
        coords = build_coords(cfg)
        terrain = build_terrain(cfg, coords)
        self.assertIsNotNone(terrain.orog)

    def test_build_diffusion_scaled(self):
        cfg = _compose(["diffusion=strong"])
        diffusion = build_diffusion(cfg)
        from jcm.diffusion import DiffusionFilter
        base = DiffusionFilter.default()
        self.assertAlmostEqual(
            float(diffusion.div_timescale),
            float(base.div_timescale) * 0.5,
        )

    def test_build_model_held_suarez(self):
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
            "run.time_step=180",
        ])
        model = build_model(cfg)
        self.assertEqual(model.coords.horizontal.nodal_shape, (96, 48))


class TestAttachOzonePreservesAquaplanetSST(unittest.TestCase):
    """Regression test for #484 codex P1.

    With ``forcing.kind == default`` and ``ozone_file`` set, the
    attach-ozone helper must keep the aquaplanet cos²-latitude SST
    profile from ``default_forcing(...)`` rather than swap it for the
    uniform 288.15 K placeholder that ``ForcingData.zeros`` would yield.
    """

    def test_default_forcing_with_ozone_keeps_cos2_sst(self):
        import tempfile
        import xarray as xr
        from jcm.forcing import default_forcing
        from jcm.runners import build_coords, build_forcing

        cfg = _compose(["physics=echam", "grid=echam_t42_l8_sigma"])
        coords = build_coords(cfg)
        nlon, nlat = coords.horizontal.nodal_shape
        nlev = coords.nodal_shape[0]

        # Synthetic 12-month ozone file in the (time, level, lat, lon)
        # layout that ``OzoneClimatology.from_file`` expects. Lat/lon
        # coords match the model grid (degrees from radians) so the
        # loader's coordinate-value check passes.
        model_lat_deg = np.asarray(coords.horizontal.latitudes) * 180.0 / np.pi
        model_lon_deg = np.asarray(coords.horizontal.longitudes) * 180.0 / np.pi
        with tempfile.TemporaryDirectory() as tmp:
            ozone_path = Path(tmp) / "ozone.nc"
            xr.Dataset(
                {"O3": (
                    ("time", "level", "lat", "lon"),
                    np.full((12, nlev, nlat, nlon), 1e-6, dtype=np.float32),
                )},
                coords={
                    "time": np.arange(12),
                    "level": np.arange(nlev, dtype=np.int32),
                    "lat": model_lat_deg,
                    "lon": model_lon_deg,
                },
            ).to_netcdf(ozone_path)
            cfg.forcing.kind = "default"
            cfg.forcing.ozone_file = str(ozone_path)

            forcing_with_ozone = build_forcing(cfg, coords)

        baseline = default_forcing(coords.horizontal)
        np.testing.assert_array_equal(
            np.asarray(forcing_with_ozone.sea_surface_temperature),
            np.asarray(baseline.sea_surface_temperature),
        )


class TestAutoOzoneDefault(unittest.TestCase):
    """``forcing.ozone_file: auto`` — packaged per-grid climatology default.

    The analytic ozone fallback carries ~7.6× the climatological ozone
    column (linear tropospheric ramp) and biases clear-sky OLR ~12 W/m²
    low; ``auto`` makes the packaged climatology the default wherever one
    matching the grid ships, and degrades loudly (warning) elsewhere.
    """

    def test_auto_resolves_t63_and_loads(self):
        from jcm.runners import _resolve_auto_ozone, build_forcing

        cfg = _compose(["physics=echam", "grid=echam_t63_l47_hybrid"])
        coords = build_coords(cfg)
        path = _resolve_auto_ozone(coords)
        self.assertIsNotNone(path)
        self.assertTrue(path.endswith(os.path.join("t63", "ozone.nc")), path)

        cfg.forcing.kind = "default"
        cfg.forcing.ozone_file = "auto"
        forcing = build_forcing(cfg, coords)
        clim = forcing.ozone_climatology
        self.assertTrue(bool(clim.is_loaded()))
        o3 = np.asarray(clim.o3_ppmv.values)
        # Climatological, not the analytic ramp: physical ppmv bounds with a
        # genuine stratospheric peak.
        self.assertGreater(float(o3.max()), 1.0)
        self.assertLess(float(o3.max()), 20.0)
        self.assertGreater(float(o3.min()), 0.0)

    def test_auto_miss_warns_and_falls_back(self):
        from jcm.runners import build_forcing

        cfg = _compose(["physics=echam", "grid=echam_t42_l8_sigma"])
        coords = build_coords(cfg)
        cfg.forcing.kind = "default"
        cfg.forcing.ozone_file = "auto"
        with self.assertLogs(level="WARNING") as logs:
            forcing = build_forcing(cfg, coords)
        self.assertTrue(any("ANALYTIC ozone" in m for m in logs.output))
        # ``kind: default`` with no attachments returns None (aquaplanet
        # built later); either way no ozone climatology must be loaded.
        if forcing is not None:
            self.assertFalse(bool(forcing.ozone_climatology.is_loaded()))


class TestEmissionsConfig(unittest.TestCase):
    """CLI/config plumbing for prescribed aerosol emissions (#498)."""

    def _coords(self):
        from jcm.runners import build_coords
        return build_coords(_compose(["physics=echam", "grid=echam_t42_l8_sigma"]))

    def _write(self, tmp, data_vars, nlon, nlat, lev=False):
        import xarray as xr
        coords = {"lon": np.linspace(0, 360, nlon, endpoint=False),
                  "lat": np.linspace(-87, 87, nlat), "time": np.arange(12)}
        if lev:
            coords["lev"] = np.arange(4)
        path = Path(tmp) / "emis.nc"
        xr.Dataset(data_vars, coords=coords).to_netcdf(path)
        return str(path)

    def test_echam_jam_factory_includes_emission_terms(self):
        # Exercise the factory-build path and emission-term wiring with
        # lightweight overrides — the preset's real defaults (mam4_jax core,
        # rrtmgp) need the optional ``jcm[mam4]`` extra / radiation data that the
        # base CI image doesn't carry; the wiring under test is independent of
        # both.
        from jcm.runners import build_physics
        cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                        "physics.jam_microphysics=placeholder",
                        "physics.radiation_scheme=grey"])
        names = [t.name for t in build_physics(cfg).terms]
        self.assertIn("jam_anthropogenic_emissions", names)
        self.assertIn("jam_prescribed_aerosol_emissions", names)

    def test_unknown_physics_key_raises(self):
        # A typo'd factory kwarg must fail loudly, not silently fall back
        # to the default it was trying to override (Codex on #624).
        from omegaconf import open_dict
        from jcm.runners import build_physics
        cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma"])
        with open_dict(cfg):
            cfg.physics.cloud_sheme = "2m"      # sic
        with self.assertRaisesRegex(ValueError, "cloud_sheme"):
            build_physics(cfg)

    def test_unknown_builder_raises(self):
        from jcm.runners import build_physics
        cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma"])
        cfg.physics.builder = "not_a_factory"
        with self.assertRaisesRegex(ValueError, "Unknown physics.builder"):
            build_physics(cfg)

    def test_bulk_file_autoroutes_to_anthropogenic(self):
        import tempfile
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, {"emis_surface_combustion_bc":
                                  (("lon", "lat", "time"),
                                   np.full((nlon, nlat, 12), 1e-11))}, nlon, nlat)
            cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file={p}"])
            f = build_forcing(cfg, coords)
        self.assertIn("emis_surface_combustion_bc", f.anthropogenic_emissions)
        self.assertIsNone(f.prescribed_aerosol_emissions)

    def test_speciated_file_autoroutes_to_prescribed(self):
        import tempfile
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, {"aero_emis_m_so4_acc":
                                  (("lon", "lat", "time"),
                                   np.full((nlon, nlat, 12), 1e-11))}, nlon, nlat)
            cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file={p}"])
            f = build_forcing(cfg, coords)
        self.assertIn("m_so4_acc", f.prescribed_aerosol_emissions)
        self.assertIsNone(f.anthropogenic_emissions)

    def test_multiple_files_merge_disjoint_channels(self):
        # A list of files (e.g. anthropogenic + biomass burning) is merged by
        # coords; channels from both end up on the forcing.
        import tempfile
        import xarray as xr
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        with tempfile.TemporaryDirectory() as tmp:
            base = {"lon": np.linspace(0, 360, nlon, endpoint=False),
                    "lat": np.linspace(-87, 87, nlat), "time": np.arange(12)}
            p1 = Path(tmp) / "anthro.nc"
            p2 = Path(tmp) / "bb.nc"
            xr.Dataset({"emis_surface_combustion_bc":
                        (("lon", "lat", "time"),
                         np.full((nlon, nlat, 12), 1e-11))},
                       coords=base).to_netcdf(p1)
            xr.Dataset({"emis_biomass_burning_bc":
                        (("lon", "lat", "time"),
                         np.full((nlon, nlat, 12), 2e-11))},
                       coords=base).to_netcdf(p2)
            cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file=[{p1},{p2}]"])
            f = build_forcing(cfg, coords)
        self.assertIn("emis_surface_combustion_bc", f.anthropogenic_emissions)
        self.assertIn("emis_biomass_burning_bc", f.anthropogenic_emissions)

    def test_grid_mismatch_raises(self):
        import tempfile
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        with tempfile.TemporaryDirectory() as tmp:
            # Wrong horizontal shape — must raise, not silently zero.
            p = self._write(tmp, {"emis_surface_combustion_bc":
                                  (("lon", "lat", "time"),
                                   np.full((nlon + 2, nlat, 12), 1e-11))},
                            nlon + 2, nlat)
            cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file={p}"])
            with self.assertRaisesRegex(ValueError, "model grid"):
                build_forcing(cfg, coords)

    def test_file_without_emission_vars_raises(self):
        import tempfile
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, {"sst": (("lon", "lat", "time"),
                                          np.zeros((nlon, nlat, 12)))}, nlon, nlat)
            cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file={p}"])
            with self.assertRaisesRegex(ValueError, "no emissions variables"):
                build_forcing(cfg, coords)


class TestNaturalForcingFilesConfig(unittest.TestCase):
    """CLI/config plumbing for the DMS / dust / oxidant climatology hooks.

    Synthetic HAMMOZ-layout files (``(time[, mlev], lat, lon)``, descending
    latitude) on the T42 grid verify that a Hydra cfg with ``dms_file`` /
    ``dust_file`` / ``oxidants_file`` populates the corresponding forcing
    fields nonzero, and that grid mismatches raise instead of silently
    zeroing the emissions.
    """

    def _coords(self):
        from jcm.runners import build_coords
        return build_coords(
            _compose(["physics=echam", "grid=echam_t42_l8_sigma"])
        )

    def _model_latlon(self, coords):
        lat = np.asarray(coords.horizontal.latitudes) * 180.0 / np.pi
        lon = np.asarray(coords.horizontal.longitudes) * 180.0 / np.pi
        return lat, lon

    def _write_files(self, tmp, coords, lat_offset=0.0, nlev=None):
        """Write tiny synthetic DMS/dust/oxidant files in the HAMMOZ layout."""
        import xarray as xr
        lat, lon = self._model_latlon(coords)
        lat = lat[::-1] + lat_offset          # descending (N→S), file style
        nlat, nlon = lat.size, lon.size
        nlev = nlev if nlev is not None else coords.nodal_shape[0]
        time = np.array([f"2000-{m:02d}-15" for m in range(1, 13)],
                        dtype="datetime64[ns]")
        base = {"time": time, "lat": lat, "lon": lon}

        dms = Path(tmp) / "dms.nc"
        xr.Dataset(
            {"DMS_sea": (("time", "lat", "lon"),
                         np.full((12, nlat, nlon), 2.0),
                         {"units": "nanomol l-1"})},
            coords=base,
        ).to_netcdf(dms)

        dust = Path(tmp) / "dust.nc"
        xr.Dataset(
            {"pot_source": (("time", "lat", "lon"),
                            np.full((12, nlat, nlon), 1.5),
                            {"units": "1."})},
            coords=base,
        ).to_netcdf(dust)

        ox = Path(tmp) / "oxidants.nc"
        ox_vars = {
            v: (("time", "mlev", "lat", "lon"),
                np.full((12, nlev, nlat, nlon), 1.0e-9),
                {"units": "mole mole-1"})
            for v in ("OH_VMR_avrg", "NO3_VMR_avrg", "O3_VMR_avrg",
                      "H2O2_VMR_avrg")
        }
        ox_vars["hybm"] = (("mlev",), np.linspace(0.0, 1.0, nlev))
        xr.Dataset(
            ox_vars, coords={**base, "mlev": np.arange(nlev)},
        ).to_netcdf(ox)
        return str(dms), str(dust), str(ox)

    def test_cfg_populates_all_three_fields_nonzero(self):
        import tempfile
        from jcm.forcing import WRAP_YEAR, TimeSeries
        from jcm.runners import build_forcing
        coords = self._coords()
        with tempfile.TemporaryDirectory() as tmp:
            dms, dust, ox = self._write_files(tmp, coords)
            cfg = _compose([
                "physics=echam-jam", "grid=echam_t42_l8_sigma",
                f"forcing.dms_file={dms}",
                f"forcing.dust_file={dust}",
                f"forcing.oxidants_file={ox}",
            ])
            f = build_forcing(cfg, coords)
        nlon, nlat = coords.horizontal.nodal_shape
        nlev = coords.nodal_shape[0]
        for leaf, shape in [
            (f.dms_seawater, (12, nlon, nlat)),
            (f.dust_source, (12, nlon, nlat)),
            (f.oxidant_vmr["oh"], (12, nlev, nlon, nlat)),
            (f.oxidant_vmr["o3"], (12, nlev, nlon, nlat)),
        ]:
            self.assertIsInstance(leaf, TimeSeries)
            self.assertEqual(int(leaf.align_mode), WRAP_YEAR)
            self.assertEqual(leaf.values.shape, shape)
            self.assertGreater(float(np.abs(np.asarray(leaf.values)).min()),
                               0.0)
        # DMS converted nmol/L → kg/m³; dust clipped to 1.
        self.assertAlmostEqual(float(f.dms_seawater.values[0, 0, 0]),
                               2.0 * 6.21324e-8, places=12)
        self.assertEqual(float(f.dust_source.values.max()), 1.0)
        # kind=default parent keeps the aquaplanet cos²-lat SST profile.
        from jcm.forcing import default_forcing
        np.testing.assert_array_equal(
            np.asarray(f.sea_surface_temperature),
            np.asarray(default_forcing(coords.horizontal)
                       .sea_surface_temperature),
        )

    def test_null_paths_are_noops(self):
        from jcm.runners import build_forcing
        coords = self._coords()
        cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma"])
        f = build_forcing(cfg, coords)
        # No files → no forcing at all (kind: default returns None).
        self.assertIsNone(f)

    def test_lat_mismatch_raises(self):
        import tempfile
        from jcm.runners import build_forcing
        coords = self._coords()
        with tempfile.TemporaryDirectory() as tmp:
            dms, _, _ = self._write_files(tmp, coords, lat_offset=3.0)
            cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.dms_file={dms}"])
            with self.assertRaisesRegex(ValueError, "latitudes"):
                build_forcing(cfg, coords)

    def test_oxidant_level_mismatch_raises(self):
        import tempfile
        from jcm.runners import build_forcing
        coords = self._coords()
        with tempfile.TemporaryDirectory() as tmp:
            _, _, ox = self._write_files(tmp, coords,
                                         nlev=coords.nodal_shape[0] + 3)
            cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.oxidants_file={ox}"])
            with self.assertRaisesRegex(ValueError, "levels"):
                build_forcing(cfg, coords)


class TestEndToEnd(unittest.TestCase):
    """Tiny end-to-end runs at T31/L8.

    Kept fast so the push CI exercises the full ``runners.run`` +
    ``Model.run`` path.
    """

    def test_run_held_suarez_smoke(self):
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
            "run=smoke",
            "run.time_step=180",
            "run.total_time=2",
            "run.save_interval=1",
        ])
        predictions = run(cfg)
        self.assertEqual(predictions.dynamics.u_wind.shape[0], 2)

    def test_run_speedy_default_smoke(self):
        cfg = _compose([
            "run.time_step=720",
            "run.total_time=2",
            "run.save_interval=1",
        ])
        predictions = run(cfg)
        self.assertEqual(predictions.dynamics.u_wind.shape[0], 2)

    def test_run_held_suarez_jw_and_balanced_inits(self):
        # The non-chunked ``init.kind`` dispatch: jw and balanced_isothermal
        # must go through inject + ``Model.resume`` (not ``Model.run``).
        for init in ("jw", "balanced_isothermal"):
            cfg = _compose([
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                f"init={init}" if init == "jw" else "init=balanced_isothermal",
                "run.time_step=180",
                "run.total_time=1",
                "run.save_interval=1",
            ])
            predictions = run(cfg)
            T = np.asarray(predictions.dynamics.temperature)
            self.assertTrue(np.isfinite(T).all(), f"init={init} produced NaNs")


class TestModeDispatch(unittest.TestCase):
    """Cover the ``run.mode = chunked / prescribed / scm`` dispatch paths."""

    def test_chunked_run_writes_per_chunk_netcdfs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _compose([
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.total_time=2",
                "run.save_interval=1",
                "run.chunk_days=1",
                f"run.output_prefix={tmpdir}/chunk",
            ])
            preds = run(cfg)
            # ``run_chunked`` returns a list of per-chunk health reports.
            self.assertIsInstance(preds, list)
            self.assertGreaterEqual(len(preds), 1)
            self.assertTrue(any(Path(tmpdir).glob("chunk_day*.nc")))

    def test_chunked_run_resumes_from_checkpoint(self):
        """``cfg.run.checkpoint_path`` makes a chunked run resumable.

        Drives ``run_chunked`` once for 1 of 2 chunks, then re-invokes
        with the same ``checkpoint_path`` and ``total_time=2`` and
        verifies the second invocation only steps the remaining chunk.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = f"{tmpdir}/run.ckpt"
            base_overrides = [
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.save_interval=1",
                "run.chunk_days=1",
                f"run.output_prefix={tmpdir}/chunk",
                f"run.checkpoint_path={ckpt_path}",
            ]

            # First invocation: run only the first chunk (1 day total).
            cfg1 = _compose(base_overrides + ["run.total_time=1"])
            reports1 = run(cfg1)
            self.assertEqual(len(reports1), 1)
            self.assertTrue(Path(ckpt_path).exists())

            # Second invocation: total 2 days, but the first chunk
            # should be skipped because the checkpoint records day=1.
            cfg2 = _compose(base_overrides + ["run.total_time=2"])
            reports2 = run(cfg2)
            self.assertEqual(len(reports2), 1, "should run only the remaining chunk")
            self.assertAlmostEqual(reports2[0]["elapsed_days"], 2.0, places=5)

    def test_chunked_resume_with_balanced_isothermal_init(self):
        """Resume path bootstraps the physics carry for inject-based inits.

        ``inject_balanced_isothermal_profile`` populates
        ``_final_dycore_state`` but leaves ``_final_physics_state`` for
        ``Model.resume`` to lazy-build. The resume-from-checkpoint code
        path must materialise the carry itself before calling
        ``load_checkpoint``, otherwise the load raises on the
        uninitialised template (codex review on PR #479). Held-Suarez is
        the cheapest physics that supports ``init=balanced_isothermal``.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = f"{tmpdir}/run.ckpt"
            base_overrides = [
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "init=balanced_isothermal",
                "run.time_step=180",
                "run.save_interval=1",
                "run.chunk_days=1",
                f"run.output_prefix={tmpdir}/chunk",
                f"run.checkpoint_path={ckpt_path}",
            ]
            run(_compose(base_overrides + ["run.total_time=1"]))
            self.assertTrue(Path(ckpt_path).exists())
            reports2 = run(_compose(base_overrides + ["run.total_time=2"]))
            self.assertEqual(len(reports2), 1)
            self.assertAlmostEqual(reports2[0]["elapsed_days"], 2.0, places=5)

    def test_from_state_warm_start(self):
        """init=from_state loads a donor checkpoint with the clock reset.

        Donor: 2 sim-days of chunked Held-Suarez (checkpoint carries
        elapsed_days=2). A from_state run then integrates 1 day: it must
        run exactly one fresh chunk (clock reset — a checkpoint RESUME
        with total_time=1 would run nothing, since 2 > 1) and start from
        the donor's fields, not the cold-start profile.
        """
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            donor_ckpt = f"{tmpdir}/donor.ckpt"
            common = [
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.save_interval=1",
                "run.chunk_days=1",
            ]
            run(_compose(common + [
                "init=balanced_isothermal",
                "run.total_time=2",
                f"run.output_prefix={tmpdir}/donor",
                f"run.checkpoint_path={donor_ckpt}",
            ]))
            self.assertTrue(Path(donor_ckpt).exists())

            reports = run(_compose(common + [
                "init=from_state",
                f"init.file={donor_ckpt}",
                "run.total_time=1",
                f"run.output_prefix={tmpdir}/warm",
            ]))
            # Clock reset: one 1-day chunk ran, ending at elapsed day 1.
            self.assertEqual(len(reports), 1)
            self.assertAlmostEqual(reports[0]["elapsed_days"], 1.0, places=5)

            import xarray as xr
            donor_end = xr.open_dataset(f"{tmpdir}/donor_day2.nc")
            warm = xr.open_dataset(f"{tmpdir}/warm_day1.nc")
            cold = xr.open_dataset(f"{tmpdir}/donor_day1.nc")
            # The warm run's day-1 mean is far closer to the donor's
            # evolved day-2 state than to the cold start's day-1 (the
            # donor fields, one further day evolved).
            d_warm = float(np.abs(warm.temperature.values
                                  - donor_end.temperature.values).mean())
            d_cold = float(np.abs(cold.temperature.values
                                  - donor_end.temperature.values).mean())
            self.assertLess(d_warm, d_cold)
            # The dycore sim_time must reset too: dates/forcing/output
            # timestamps derive from it, not from the chunk counter. The
            # warm run's output must be stamped ~day 1, EARLIER than the
            # donor's day-2 file — without the with_sim_time reset it
            # inherits the donor clock and stamps ~day 3.
            self.assertLess(float(np.asarray(warm.time.max())),
                            float(np.asarray(donor_end.time.max())))

    def _write_state_file(self, path):
        # Run a tiny full simulation and dump it so the prescribed/scm modes
        # have a JCM-shaped state to load.
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
            "run.time_step=180",
            "run.total_time=2",
            "run.save_interval=1",
        ])
        preds = run(cfg)
        preds.to_xarray().to_netcdf(path)

    def test_prescribed_mode_runs_from_state_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.nc"
            self._write_state_file(str(state_file))
            cfg = _compose([
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.mode=prescribed",
                f"run.state_file={state_file}",
            ])
            preds = run(cfg)
            self.assertEqual(preds.tendencies.temperature.shape[0], 2)

    def test_scm_mode_picks_column_from_state_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.nc"
            self._write_state_file(str(state_file))
            cfg = _compose([
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.mode=scm",
                f"run.state_file={state_file}",
                "run.column.lat_deg=0.0",
                "run.column.lon_deg=0.0",
            ])
            preds = run(cfg)
            # SCM output is 1-D in level with a leading time axis.
            self.assertEqual(preds.tendencies.temperature.shape, (2, 8))


class TestMainCLI(unittest.TestCase):
    """Smoke-test the Hydra CLI entry point at ``jcm.main``."""

    def test_main_writes_netcdf(self):
        # Hydra's testing helpers compose the same config the CLI would and
        # invoke the entry point; this covers ``main`` + ``resolve_output_path``
        # + ``save_predictions`` without spawning a subprocess.
        import tempfile
        from hydra.experimental.callback import Callback  # noqa: F401  (Hydra check)
        from jcm import main as main_module

        with tempfile.TemporaryDirectory() as tmpdir:
            with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
                cfg = compose(
                    config_name="config",
                    overrides=[
                        "physics=held_suarez",
                        "grid=held_suarez_t31_l8",
                        "run.time_step=180",
                        "run.total_time=2",
                        "run.save_interval=1",
                        f"run.output={tmpdir}/cli_test.nc",
                    ],
                    return_hydra_config=True,
                )
                # Hydra's runtime config isn't normally available outside the
                # ``@hydra.main`` decorator; resolve it manually for the test.
                from hydra.core.hydra_config import HydraConfig
                HydraConfig.instance().set_config(cfg)
                main_module.main.__wrapped__(cfg)
            self.assertTrue(Path(tmpdir, "cli_test.nc").exists())


class TestConfigureHostDeviceCount(unittest.TestCase):
    """In-process control flow of the CPU device-count helper.

    The real multi-device branches are exercised by the subprocess-based
    sharding tests (coverage can't see those). Here we mock the jax calls so
    the test is deterministic and never actually changes the process-wide
    device count, while still covering the no-op, idempotent ``XLA_FLAGS``
    append, and "backend already live" warning paths.
    """

    _FLAG = "--xla_cpu_enable_concurrency_optimized_scheduler=false"

    def setUp(self):
        self._saved_flags = os.environ.get("XLA_FLAGS")

    def tearDown(self):
        if self._saved_flags is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = self._saved_flags

    def test_noop_for_single_or_none(self):
        os.environ.pop("XLA_FLAGS", None)
        for n in (None, 0, 1):
            configure_host_device_count(n)
            # A no-op must not touch the device count or the env.
            self.assertNotIn("XLA_FLAGS", os.environ)

    def test_appends_serialized_collective_flag(self):
        os.environ.pop("XLA_FLAGS", None)
        with mock.patch("jax.config.update"), \
                mock.patch("jax.device_count", return_value=4):
            configure_host_device_count(4)
        self.assertIn(self._FLAG, os.environ["XLA_FLAGS"])

    def test_flag_append_is_idempotent(self):
        os.environ["XLA_FLAGS"] = f"--foo {self._FLAG}"
        with mock.patch("jax.config.update"), \
                mock.patch("jax.device_count", return_value=4):
            configure_host_device_count(4)
        self.assertEqual(os.environ["XLA_FLAGS"].count(self._FLAG), 1)

    def test_warns_when_backend_already_live(self):
        # config.update raising RuntimeError mimics a backend that is already
        # initialised (the CLI case); device_count then falls short of the
        # request, so the helper must warn rather than silently proceed.
        os.environ.pop("XLA_FLAGS", None)
        with mock.patch("jax.config.update", side_effect=RuntimeError), \
                mock.patch("jax.device_count", return_value=1):
            with self.assertLogs("jcm.runners", level="WARNING") as cm:
                configure_host_device_count(8)
        self.assertTrue(any("already initialised" in m for m in cm.output))


class TestBuilderErrorAndSelectorPaths(unittest.TestCase):
    """Config-validation and selector branches of the ``build_*`` helpers.

    All of these are config-only paths (no dycore construction, no
    integration), so they stay cheap in the fast sweep.
    """

    def test_build_coords_hybrid_unsupported_layers_raises(self):
        cfg = _compose(["grid=echam_t63_l47_hybrid"])
        cfg.grid.layers = 13  # no pre-tuned hybrid table for 13 levels
        with self.assertRaisesRegex(ValueError, "not pre-configured"):
            build_coords(cfg)

    def test_build_coords_hybrid_l47(self):
        cfg = _compose(["grid=echam_t63_l47_hybrid"])
        # Shrink the horizontal so the test stays cheap; the point is the
        # hybrid-vertical branch.
        cfg.grid.spectral_truncation = 21
        coords = build_coords(cfg)
        from dinosaur.hybrid_coordinates import HybridCoordinates
        self.assertIsInstance(coords.vertical, HybridCoordinates)
        self.assertEqual(coords.nodal_shape[0], 47)

    def test_build_coords_unknown_vertical_raises(self):
        cfg = _compose()
        cfg.grid.vertical = "isentropic"
        with self.assertRaisesRegex(ValueError, "Unknown grid.vertical"):
            build_coords(cfg)

    def test_build_terrain_unknown_kind_raises(self):
        cfg = _compose()
        cfg.terrain.kind = "flat_earth"
        with self.assertRaisesRegex(ValueError, "Unknown terrain.kind"):
            build_terrain(cfg, coords=None)

    def test_build_forcing_unknown_kind_raises(self):
        from jcm.runners import build_forcing
        cfg = _compose()
        cfg.forcing.kind = "bogus"
        with self.assertRaisesRegex(ValueError, "Unknown forcing.kind"):
            build_forcing(cfg, coords=None)

    def test_build_forcing_without_forcing_block_is_none(self):
        # A config with no ``forcing`` group at all must fall through every
        # attach helper untouched and return None (Model then defaults to
        # the aquaplanet forcing).
        from omegaconf import OmegaConf
        from jcm.runners import build_forcing
        self.assertIsNone(build_forcing(OmegaConf.create({}), coords=None))

    def test_build_physics_requires_terms_or_builder(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"physics": {"terms": None}})
        with self.assertRaisesRegex(ValueError, "physics.terms is required"):
            build_physics(cfg)

    def test_build_physics_null_term_entry_is_skipped(self):
        # Hydra's `~`-removal / explicit null idiom disables a term.
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        baseline_n = len(build_physics(cfg).terms)
        cfg.physics.terms["held_suarez"] = None
        self.assertEqual(len(build_physics(cfg).terms), baseline_n - 1)

    def test_build_term_without_target_raises(self):
        from jcm.runners import _build_term
        with self.assertRaisesRegex(ValueError, "_target_"):
            _build_term("broken_term", {"params": {}})

    def test_build_diffusion_pinned_echam_kinds(self):
        from omegaconf import OmegaConf
        from jcm.diffusion import DiffusionFilter

        for kind, factory in (
            ("echam_t63_l47", DiffusionFilter.echam_t63_l47),
            ("echam_t85_l47", DiffusionFilter.echam_t85_l47),
        ):
            cfg = OmegaConf.create({"diffusion": {"kind": kind}})
            diffusion = build_diffusion(cfg)
            expected = factory()
            self.assertEqual(
                float(diffusion.temp_timescale), float(expected.temp_timescale),
                f"kind={kind} did not pin the matching factory",
            )
            # The lmidatm profiles are level-dependent (del2 at top),
            # unlike the uniform SPEEDY default.
            self.assertIsNotNone(diffusion.level_orders_temp)

    @staticmethod
    def _grid_cfg(layers, truncation, kind="auto", vertical="hybrid"):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "diffusion": {"kind": kind},
            "grid": {"vertical": vertical, "layers": layers,
                     "spectral_truncation": truncation},
        })

    def test_build_diffusion_auto_picks_lmidatm_for_every_echam_grid(self):
        """Every shipped hybrid grid must get its ECHAM lmidatm profile.

        Regression for #579: ``auto`` matched on ``layers == 47``, so the L95
        middle-atmosphere grids — which exist precisely to resolve the
        stratosphere — silently received SPEEDY's uniform profile instead.
        """
        from jcm.diffusion import DiffusionFilter

        for layers, truncation in ((47, 63), (47, 85), (47, 106), (47, 119),
                                   (95, 63), (95, 106), (95, 119)):
            with self.subTest(layers=layers, truncation=truncation):
                diffusion = build_diffusion(self._grid_cfg(layers, truncation))
                expected = DiffusionFilter.echam_lmidatm(truncation, layers)
                self.assertEqual(float(diffusion.temp_timescale),
                                 float(expected.temp_timescale))
                self.assertEqual(len(diffusion.level_orders_temp), layers)

    def test_build_diffusion_auto_warns_when_falling_back_on_a_hybrid_grid(self):
        """The SPEEDY fallback on an ECHAM-family grid must not be silent."""
        from jcm.diffusion import DiffusionFilter

        with self.assertLogs("jcm.runners", level="WARNING") as captured:
            diffusion = build_diffusion(self._grid_cfg(layers=31, truncation=63))
        self.assertEqual(float(diffusion.temp_timescale),
                         float(DiffusionFilter.default().temp_timescale))
        self.assertIn("no ECHAM lmidatm profile", "\n".join(captured.output))

    def test_build_diffusion_auto_stays_silent_for_non_hybrid_grids(self):
        """SPEEDY/Held-Suarez sigma grids are tuned for the uniform profile."""
        import logging

        from jcm.diffusion import DiffusionFilter

        with self.assertNoLogs("jcm.runners", level=logging.WARNING):
            diffusion = build_diffusion(
                self._grid_cfg(layers=8, truncation=31, vertical="sigma"))
        self.assertEqual(float(diffusion.temp_timescale),
                         float(DiffusionFilter.default().temp_timescale))

    def test_build_diffusion_rejects_a_profile_of_the_wrong_length(self):
        """Pinning an L47 profile on an L95 grid must fail with a clear error.

        Previously this surfaced deep inside the spectral filter as
        "objects cannot be broadcast to a single shape ... (95, 213, 108)
        and (47,)", which named neither the config key nor the grid (#579).
        """
        cfg = self._grid_cfg(layers=95, truncation=106, kind="echam_t85_l47")
        with self.assertRaisesRegex(ValueError, r"47-level .* grid has 95 levels"):
            build_diffusion(cfg)

    def test_build_diffusion_unknown_kind_raises(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"diffusion": {"kind": "del99"}})
        with self.assertRaisesRegex(ValueError, "Unknown diffusion.kind"):
            build_diffusion(cfg)

    def test_maybe_add_sponge_appends_upper_sponge_term(self):
        from jcm.physics.dissipation import UpperSponge
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.runners import maybe_add_sponge

        # run=longrun carries the production sponge block (10 levels).
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8",
                        "run=longrun"])
        physics = held_suarez_physics()
        n_before = len(physics.terms)
        with_sponge = maybe_add_sponge(physics, cfg)
        self.assertEqual(len(with_sponge.terms), n_before + 1)
        sponge_term = with_sponge.terms[-1]
        self.assertIsInstance(sponge_term, UpperSponge)

        # Default config has sponge disabled -> pass-through, same object.
        cfg_off = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        self.assertIs(maybe_add_sponge(physics, cfg_off), physics)


class TestRunDispatchErrorPaths(unittest.TestCase):
    def test_run_unknown_mode_raises(self):
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        cfg.run.mode = "teleport"
        with self.assertRaisesRegex(ValueError, "Unknown run.mode"):
            run(cfg)

    def test_run_applies_constants_overrides_before_dispatch(self):
        # The constants block must be applied (set_constants path) before
        # the mode dispatch — use the current value so the process-global
        # singleton is unchanged, and the unknown mode aborts before any
        # model construction.
        import jcm.constants as c
        grav_before = float(c.grav)
        cfg = _compose([
            "physics=held_suarez", "grid=held_suarez_t31_l8",
            f"+constants.grav={grav_before}",
        ])
        cfg.run.mode = "teleport"
        with self.assertRaisesRegex(ValueError, "Unknown run.mode"):
            run(cfg)
        self.assertEqual(float(c.grav), grav_before)

    def test_run_full_unknown_init_kind_raises(self):
        import types as _types
        from jcm.runners import _run_full

        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        cfg.init.kind = "from_mars"
        # A stub model shortcuts build_model: _run_full only touches
        # ``model.coords`` (for the forcing) before the init dispatch.
        stub = _types.SimpleNamespace(coords=build_coords(cfg))
        with self.assertRaisesRegex(ValueError, "Unknown init.kind"):
            _run_full(cfg, model=stub)

    def test_prescribed_mode_requires_state_file(self):
        from jcm.runners import _load_states_from_cfg
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        cfg.run.mode = "prescribed"
        with self.assertRaisesRegex(ValueError, "state_file"):
            _load_states_from_cfg(cfg)

    def test_scm_mode_requires_column(self):
        from jcm.runners import _run_scm
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        cfg.run.mode = "scm"
        # The default config carries a (nulled-out) column block; drop it
        # to exercise the guard for configs that never define one.
        cfg.run.column = None
        with self.assertRaisesRegex(ValueError, "run.column"):
            _run_scm(cfg)


class TestOutputPathAndSave(unittest.TestCase):
    def test_resolve_output_path_relative_run_and_multirun(self):
        import tempfile
        import types as _types
        from jcm.runners import resolve_output_path

        cfg = _compose(["run.output=state.nc"])
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                hydra_single = _types.SimpleNamespace(
                    run=_types.SimpleNamespace(dir="outputs/2026-01-01/x"),
                    mode="RunMode.RUN",
                    job=_types.SimpleNamespace(num=0),
                )
                p = resolve_output_path(cfg, hydra_single)
                self.assertEqual(
                    p, Path("outputs/2026-01-01/x/state.nc"),
                )
                self.assertTrue(p.parent.is_dir())

                hydra_multi = _types.SimpleNamespace(
                    run=_types.SimpleNamespace(dir="outputs/2026-01-01/x"),
                    mode="RunMode.MULTIRUN",
                    job=_types.SimpleNamespace(num=3),
                )
                p_multi = resolve_output_path(cfg, hydra_multi)
                self.assertEqual(
                    p_multi,
                    Path("outputs/2026-01-01/x/multirun/3/state.nc"),
                )
            finally:
                os.chdir(cwd)

    def test_save_predictions_skips_chunked_report_lists(self):
        import tempfile
        from jcm.runners import save_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.nc"
            # run_chunked returns a list of health reports — nothing to dump.
            save_predictions([{"ok": True}], out)
            self.assertFalse(out.exists())


class TestInjectProfilesOrographyAndHybrid(unittest.TestCase):
    """Orography-rebalance and hybrid-vertical branches of the inits.

    The aquaplanet-based tests elsewhere never hit the ``orog > 1`` ps
    rebalance; real T30 terrain exercises it. The JW init additionally
    has a hybrid-coordinate branch for the sigma-center lookup.
    """

    def _real_terrain_model(self):
        from importlib import resources

        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        from jcm.terrain import TerrainData

        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        data_dir = resources.files('jcm.data.bc.t30.clim')
        terrain = TerrainData.from_file(
            data_dir / 'terrain.nc', coords=coords,  # real orography
        )
        model = Model(coords=coords, terrain=terrain,
                      physics=held_suarez_physics(), time_step=180)
        model.bootstrap_state()
        return model, terrain

    def _nodal_ps(self, model):
        log_ps_nodal = model.coords.horizontal.to_nodal(
            model._final_dycore_state.log_surface_pressure
        )[0]
        from dinosaur.scales import units
        scale = float(
            model.dycore.physics_specs.dimensionalize(1.0, units.pascal).m
        )
        return np.exp(np.asarray(log_ps_nodal)) * scale

    def test_balanced_isothermal_rebalances_ps_over_orography(self):
        from jcm.runners import inject_balanced_isothermal_profile

        model, terrain = self._real_terrain_model()
        inject_balanced_isothermal_profile(model)
        ps = self._nodal_ps(model)
        orog = np.asarray(terrain.orog)

        # Surface pressure must drop hydrostatically over high terrain.
        # (Pointwise bounds are softened by the spectral round-trip of
        # log_ps, so assert on the >2 km population mean + correlation.)
        high = orog > 2000.0
        self.assertTrue(high.any(), "terrain data has no >2 km orography")
        sea_level = ps[orog < 1.0].mean()
        self.assertAlmostEqual(sea_level / 101325.0, 1.0, places=1)
        self.assertLess(ps[high].mean(), 0.8 * sea_level)
        self.assertLess(np.corrcoef(orog.ravel(), ps.ravel())[0, 1], -0.95)
        # And the p(z) relation should be monotone: the highest point has
        # the lowest surface pressure.
        self.assertEqual(np.argmin(ps), np.argmax(orog))

    def test_jw_profile_rebalances_ps_and_injects_humidity(self):
        from jcm.runners import inject_jw_profile

        model, terrain = self._real_terrain_model()
        inject_jw_profile(model, rh=0.6)
        ps = self._nodal_ps(model)
        orog = np.asarray(terrain.orog)
        self.assertLess(
            ps[orog > 2000.0].mean(), 0.8 * ps[orog < 1.0].mean(),
        )

        physics_state = model.dycore.to_physics_state(
            model._final_dycore_state
        )
        q = np.asarray(physics_state.specific_humidity)
        # Moist near the surface, dry above the 200 hPa cap (level 0 is
        # the model top in the physics state layout).
        self.assertGreater(q[-1].mean(), 1e-4)
        self.assertLess(q[0].max(), 1e-6)

    def test_jw_profile_on_hybrid_l47_grid(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.runners import inject_jw_profile
        from jcm.utils import get_coords

        coords = get_coords(
            vertical_coords=get_echam_levels(47), spectral_truncation=21,
        )
        model = Model(coords=coords, physics=held_suarez_physics(),
                      time_step=180)
        model.bootstrap_state()
        inject_jw_profile(model, rh=0.5)

        physics_state = model.dycore.to_physics_state(
            model._final_dycore_state
        )
        T = np.asarray(physics_state.temperature)
        # Lapse-rate profile bounded by the JW floor and surface values.
        self.assertGreaterEqual(T.min(), 240.0)
        self.assertLessEqual(T.max(), 290.0)
        # Warm at the bottom, at the 250 K floor near the top.
        self.assertGreater(T[-1].mean(), 280.0)
        self.assertLess(T[0].mean(), 255.0)


# ---------------------------------------------------------------------------
# Slow-marked companions
#
# The PR CI runs ``pytest -m "slow" --cov-fail-under=80``. The push CI runs
# ``-m "not slow" --cov-fail-under=90``. We need the same end-to-end paths
# exercised in *both* passes so neither coverage threshold drops below the
# bar after we add new code. Subclassing inherits every test method and the
# class-level ``slow`` marker decides which CI pass picks them up.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEndToEndSlow(TestEndToEnd):
    pass


@pytest.mark.slow
class TestModeDispatchSlow(TestModeDispatch):
    pass


@pytest.mark.slow
class TestMainCLISlow(TestMainCLI):
    pass


class TestInjectJwHumidityMagnitude(unittest.TestCase):
    """``inject_jw_profile`` must hand the gridpoint physics a physical
    humidity magnitude (a few g/kg, i.e. O(1e-2) kg/kg), not 1000x larger.

    Regression for the moist-init blow-up: storing the raw kg/kg ``q_profile``
    into the dynamics ``State.tracers`` skipped the
    ``nondimensionalize(q * gram/kilogram)`` that the canonical
    physics->dynamics bridge applies. The forward bridge then re-dimensionalized
    (~x1000), so the physics saw q ~ 5 kg/kg; the cloud saturation adjustment
    read that as hugely supersaturated and dumped ~7000 K of latent heat in a
    single step, NaNing every moist init at step 1.
    """

    def test_jw_physics_q_is_physical_magnitude(self):
        from jcm.runners import inject_jw_profile
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        physics = echam_physics(cloud_scheme="2m", checkpoint_terms=False)
        model = Model(coords=get_speedy_coords(), physics=physics, time_step=180)
        model.bootstrap_state()
        inject_jw_profile(model, rh=0.6)

        ps = model.dycore.to_physics_state(model._final_dycore_state)
        qmax = float(np.max(np.asarray(ps.specific_humidity)))
        # rh*q_sat at the warm surface is a few g/kg -> O(1e-2) kg/kg. The units
        # bug produced ~5 (>1 kg/kg); bound it tightly on both sides.
        self.assertLess(
            qmax, 0.05,
            f"gridpoint q={qmax:.4g} kg/kg is unphysical (1000x units bug)",
        )
        self.assertGreater(
            qmax, 1e-4,
            f"gridpoint q={qmax:.4g} kg/kg too dry; humidity not injected",
        )


class TestInjectJwPreservesCloudTracers(unittest.TestCase):
    """``inject_jw_profile`` must inject only the analytic humidity profile and
    keep the other prognostic tracers (qc/qi/qnc/qni/qr/qs) the dycore seeded.

    Regression for the CRE ≡ 0 bug: overwriting ``state.tracers`` wholesale
    dropped the cloud tracers, so radiation saw zero cloud water for the entire
    JW-initialised run.
    """

    def test_jw_keeps_2m_cloud_tracers(self):
        from jcm.runners import inject_jw_profile
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        physics = echam_physics(cloud_scheme="2m", checkpoint_terms=False)
        model = Model(coords=get_speedy_coords(), physics=physics, time_step=180)
        model.bootstrap_state()
        inject_jw_profile(model, rh=0.5)

        keys = set(model._final_dycore_state.tracers.keys())
        self.assertIn("specific_humidity", keys)
        # qr/qs are no longer prognostic (2M precipitation is flux-form,
        # review finding 2.18) — the guard covers the four cloud tracers.
        self.assertTrue(
            {"qc", "qi", "qnc", "qni"}.issubset(keys),
            f"inject_jw_profile dropped cloud tracers; tracers present: {keys}",
        )


class CompilationCacheTest(unittest.TestCase):
    def test_default_on_override_and_off(self):
        # On by default with a machine-appropriate location (#592);
        # JCM_CACHE_DIR relocates it, and "off" disables.
        import jax

        from jcm.runners import maybe_enable_compilation_cache
        old = jax.config.jax_compilation_cache_dir
        old_scratch = os.environ.get("SCRATCH")
        try:
            os.environ.pop("JCM_CACHE_DIR", None)
            os.environ["SCRATCH"] = "/nonexistent/scratch"
            maybe_enable_compilation_cache()
            self.assertEqual(jax.config.jax_compilation_cache_dir,
                             "/nonexistent/scratch/jcm-jax-cache")

            os.environ["JCM_CACHE_DIR"] = "/nonexistent/jcm-cache-test"
            maybe_enable_compilation_cache()
            self.assertEqual(jax.config.jax_compilation_cache_dir,
                             "/nonexistent/jcm-cache-test")

            jax.config.update("jax_compilation_cache_dir", old)
            os.environ["JCM_CACHE_DIR"] = "off"
            maybe_enable_compilation_cache()
            self.assertEqual(jax.config.jax_compilation_cache_dir, old)
        finally:
            os.environ.pop("JCM_CACHE_DIR", None)
            if old_scratch is None:
                os.environ.pop("SCRATCH", None)
            else:
                os.environ["SCRATCH"] = old_scratch
            jax.config.update("jax_compilation_cache_dir", old)


class ResolveDataPathTest(unittest.TestCase):
    """hf:// path resolution for boundary-file config values."""

    def test_hf_prefix_fetches_and_plain_passes_through(self):
        from jcm import runners

        with mock.patch("jcm.data.remote.fetch",
                        side_effect=lambda p: f"/cache/{p}") as m:
            self.assertEqual(
                runners._resolve_data_path("hf://bundles/t63/terrain.nc"),
                "/cache/bundles/t63/terrain.nc")
            m.assert_called_once_with("bundles/t63/terrain.nc")
        self.assertEqual(runners._resolve_data_path("/local/x.nc"),
                         "/local/x.nc")
        self.assertIsNone(runners._resolve_data_path(None))

    def test_list_paths_resolve_elementwise_and_mappings_pass_through(self):
        from omegaconf import OmegaConf

        from jcm import runners

        with mock.patch("jcm.data.remote.fetch",
                        side_effect=lambda p: f"/cache/{p}"):
            out = runners._resolve_data_path(["hf://a.nc", "/local/b.nc"])
        self.assertEqual(out, ["/cache/a.nc", "/local/b.nc"])
        # a mis-typed mapping must NOT be flattened to its keys
        dc = OmegaConf.create({"so2": "/a.nc"})
        self.assertIs(runners._resolve_data_path(dc), dc)


class TestAutoInputResolution(unittest.TestCase):
    """Per-grid auto-resolution of ozone/terrain (#638 matrix rot)."""

    def _coords(self, truncation=63, layers=47):
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.utils import get_coords
        return get_coords(get_echam_levels(layers),
                          spectral_truncation=truncation)

    def test_packaged_t63_wins_without_fetch(self):
        from unittest import mock

        from jcm.runners import _resolve_auto_ozone, _resolve_auto_terrain
        coords = self._coords(63)
        with mock.patch("jcm.data.remote.fetch",
                        side_effect=AssertionError("fetch must not be hit")):
            self.assertIn("t63", _resolve_auto_ozone(coords))
            self.assertIn("t63", _resolve_auto_terrain(coords))

    def test_mirror_fallback_paths(self):
        from unittest import mock

        from jcm.runners import _resolve_auto_ozone, _resolve_auto_terrain
        coords = self._coords(21, layers=47)   # t21 not packaged
        with mock.patch("jcm.data.remote.fetch",
                        return_value="/cached/file.nc") as f:
            self.assertEqual(_resolve_auto_ozone(coords), "/cached/file.nc")
            self.assertEqual(f.call_args.args[0],
                             "bundles/t21_l47/ozone_pd.nc")
            self.assertEqual(_resolve_auto_terrain(coords), "/cached/file.nc")
            self.assertEqual(f.call_args.args[0], "bundles/t21/terrain.nc")

    def test_terrain_auto_raises_loudly_when_unresolvable(self):
        from unittest import mock

        from jcm.runners import _resolve_auto_terrain
        coords = self._coords(21)
        with mock.patch("jcm.data.remote.fetch",
                        side_effect=OSError("offline")):
            with self.assertRaises(FileNotFoundError):
                _resolve_auto_terrain(coords)

    def test_terrain_auto_composes(self):
        cfg = _compose(["terrain=auto", "physics=speedy"])
        self.assertEqual(cfg.terrain.kind, "auto")


class TestYearExpansionAndStartDate(unittest.TestCase):
    """{year} pattern expansion + run.start_date threading (#610)."""

    def test_pattern_expands_inclusive_range(self):
        from jcm import runners
        out = runners._expand_years("hf://bundles/t63/forcing_amip/{year}.nc",
                                    [1979, 1981])
        self.assertEqual(out, [
            "hf://bundles/t63/forcing_amip/1979.nc",
            "hf://bundles/t63/forcing_amip/1980.nc",
            "hf://bundles/t63/forcing_amip/1981.nc",
        ])

    def test_available_years_pads_one_each_side(self):
        # Mid-month samples need a bracketing December/January from the
        # neighbouring years, else by_date_interp clamps at the run
        # boundaries (Codex P1 on #611).
        from jcm import runners
        out = runners._expand_years("/x/{year}.nc", [1979, 1980],
                                    available=[1870, 2022])
        self.assertEqual(out, ["/x/1978.nc", "/x/1979.nc",
                               "/x/1980.nc", "/x/1981.nc"])

    def test_available_years_clips_at_coverage_edges(self):
        from jcm import runners
        self.assertEqual(
            runners._expand_years("/x/{year}.nc", [1870, 1871],
                                  available=[1870, 2022])[0],
            "/x/1870.nc")
        self.assertEqual(
            runners._expand_years("/x/{year}.nc", [2021, 2022],
                                  available=[1870, 2022])[-1],
            "/x/2022.nc")

    def test_plain_paths_and_none_pass_through(self):
        from jcm import runners
        self.assertEqual(runners._expand_years("/x/forcing.nc", [1979, 1981]),
                         "/x/forcing.nc")
        self.assertIsNone(runners._expand_years(None, [1979, 1981]))
        self.assertEqual(runners._expand_years("/x/forcing.nc", None),
                         "/x/forcing.nc")

    def test_pattern_without_years_raises(self):
        from jcm import runners
        with self.assertRaisesRegex(ValueError, "year range"):
            runners._expand_years("/x/forcing_{year}.nc", None)

    def test_reversed_range_raises(self):
        from jcm import runners
        with self.assertRaisesRegex(ValueError, "reversed"):
            runners._expand_years("/x/forcing_{year}.nc", [1981, 1979])

    def test_amip_preset_composes(self):
        cfg = _compose(["forcing=amip", "forcing.years=[1979,1980]",
                        "run.start_date=1979-01-01"])
        self.assertEqual(cfg.forcing.kind, "from_file")
        self.assertEqual(cfg.forcing.align, "by_date_interp")
        self.assertIn("{year}", cfg.forcing.file)
        self.assertEqual(list(cfg.forcing.years), [1979, 1980])
        self.assertEqual(cfg.run.start_date, "1979-01-01")

    def test_ozone_coverage_falls_back_and_overrides(self):
        # Per-product coverage (Codex P1 on #633): the era5 preset's
        # forcing runs to 2024 but its FZJ ozone product ends in 2022 —
        # without the override, forcing.years=[2022,2022] would expand
        # the ozone pattern to a 2023 file that was never built.
        from omegaconf import OmegaConf

        from jcm import runners
        cfg = OmegaConf.create({"available_years": [1979, 2024],
                                "ozone_available_years": [1850, 2022]})
        self.assertEqual(
            runners._expand_years(
                "/o3/{year}.nc", [2022, 2022],
                runners._product_available_years(
                    cfg, "ozone_available_years")),
            ["/o3/2021.nc", "/o3/2022.nc"])
        self.assertEqual(
            runners._expand_years(
                "/o3/{year}.nc", [2023, 2024],
                runners._product_available_years(
                    cfg, "ozone_available_years")),
            ["/o3/2022.nc"])
        # A range entirely past coverage clamps to the last edge file
        # rather than inverting into an empty expansion.
        self.assertEqual(
            runners._expand_years("/o3/{year}.nc", [2024, 2024],
                                  available=[1850, 2022]),
            ["/o3/2022.nc"])
        fallback = OmegaConf.create({"available_years": [1979, 2024]})
        self.assertEqual(
            runners._product_available_years(
                fallback, "ozone_available_years"),
            fallback.available_years)

    def test_era5_preset_composes(self):
        cfg = _compose(["forcing=era5", "forcing.years=[2023,2024]",
                        "run.start_date=2023-01-01"])
        self.assertEqual(cfg.forcing.align, "by_date_interp")
        self.assertIn("forcing_era5", cfg.forcing.file)
        self.assertEqual(list(cfg.forcing.ozone_available_years)[-1], 2022)

    def test_start_date_resolves_to_datetime(self):
        from omegaconf import OmegaConf

        from jcm import runners
        cfg = OmegaConf.create({"run": {"start_date": "1979-01-01"}})
        dt = runners._resolve_start_date(cfg)
        self.assertIsNotNone(dt)
        import jax_datetime as jdt
        self.assertEqual(
            int((dt - jdt.to_datetime("1979-01-01")).days), 0)

    def test_start_date_null_keeps_model_default(self):
        from omegaconf import OmegaConf

        from jcm import runners
        self.assertIsNone(runners._resolve_start_date(
            OmegaConf.create({"run": {}})))
        self.assertIsNone(runners._resolve_start_date(
            OmegaConf.create({"run": {"start_date": None}})))

    def test_start_date_threads_into_model(self):
        from jcm import runners
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8",
                        "run.time_step=180", "run.start_date=1979-01-01"])
        model = runners.build_model(cfg)
        import jax_datetime as jdt
        self.assertEqual(
            int((model.start_date - jdt.to_datetime("1979-01-01")).days), 0)
