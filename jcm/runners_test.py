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
    guard_emulator_ghg_forcing,
    run,
)


CONFIG_DIR = str(Path(__file__).parent / "config")


def _compose(overrides=None):
    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


# The four prescribed-emission keys default to ``auto`` (the per-grid HF bundle
# when a JAM package is active — issue #640). Emission-path unit tests run
# offline on grids the mirror does not carry, so they null the keys they don't
# provide to stay hermetic; a specific ``forcing.<key>=<path>`` listed *after*
# these wins. Prepend with ``[*_NULL_EMISSIONS, ...]``.
_NULL_EMISSIONS = (
    "forcing.emissions_file=null", "forcing.dms_file=null",
    "forcing.dust_file=null", "forcing.oxidants_file=null",
)


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

    def test_run_groups_share_one_schema(self):
        # ``run/default.yaml`` is the complete base schema and the other run
        # groups inherit it (``defaults: [default, _self_]``), so every run
        # group must expose exactly the same set of keys (#640 smell 3).
        keysets = {name: set(_compose([f"run={name}"]).run.keys())
                   for name in ("default", "longrun", "smoke", "pyses_year")}
        base = keysets["default"]
        for name, keys in keysets.items():
            self.assertEqual(keys, base, f"run={name} key set diverged")

    def test_run_key_override_needs_no_plus(self):
        # Because the schema is complete, formerly-per-group keys can be set on
        # any run group without a ``+`` (previously ``run=longrun`` had no
        # ``checkpoint_path`` and required ``+run.checkpoint_path=...``).
        cfg = _compose(["run=longrun", "run.checkpoint_path=/tmp/x.ckpt"])
        self.assertEqual(cfg.run.checkpoint_path, "/tmp/x.ckpt")
        # ``bail_on_unhealthy`` is now universal and defaults to the runner's
        # historical implicit default (True) on every group.
        self.assertTrue(cfg.run.bail_on_unhealthy)
        self.assertTrue(_compose(["run=default"]).run.bail_on_unhealthy)

    def test_init_jw_compose(self):
        cfg = _compose(["init=jw"])
        self.assertEqual(cfg.init.kind, "jw")


class TestExperimentGroup(unittest.TestCase):
    """The ``experiment`` group promotes each validated benchmark preset to a
    first-class ``python -m jcm.main +experiment=<name>`` composition (#640
    smell 2), so users get a validated configuration instead of a cold start.
    """

    EXPERIMENT_DIR = Path(__file__).parent / "config" / "experiment"

    def _compose_hydra(self, name):
        # ``return_hydra_config`` exposes ``cfg.hydra.runtime.choices`` so we
        # can assert which group option each experiment selected.
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            return compose(config_name="config",
                           overrides=[f"+experiment={name}"],
                           return_hydra_config=True)

    def test_every_experiment_composes(self):
        names = sorted(p.stem for p in self.EXPERIMENT_DIR.glob("*.yaml"))
        self.assertTrue(names, "experiment group is empty")
        for name in names:
            with self.subTest(experiment=name):
                cfg = self._compose_hydra(name)
                self.assertIn("physics", cfg.hydra.runtime.choices)
                # The run schema is complete, so ``time_step`` always exists.
                self.assertIn("time_step", cfg.run)

    def test_load_bearing_values(self):
        cases = {
            "speedy-t31": dict(physics="speedy", grid="speedy_t31_l8",
                               init="isothermal", run="default", time_step=15),
            "t63-echam-jam": dict(physics="echam-jam",
                                  grid="echam_t63_l47_hybrid", init="jw",
                                  run="longrun", time_step=12),
            "ma-t63-l95": dict(physics="echam-jam",
                               grid="echam_t63_l95_hybrid", init="jw",
                               run="longrun", time_step=12),
        }
        for name, want in cases.items():
            with self.subTest(experiment=name):
                cfg = self._compose_hydra(name)
                ch = cfg.hydra.runtime.choices
                self.assertEqual(ch["physics"], want["physics"])
                self.assertEqual(ch["grid"], want["grid"])
                self.assertEqual(cfg.init.kind, want["init"])
                self.assertEqual(ch["run"], want["run"])
                self.assertEqual(cfg.run.time_step, want["time_step"])

    def test_echam_experiments_use_dry_jw_and_sl_offcentering(self):
        # The L47 stability recipe: fully-dry JW init + SL off-centering.
        cfg = self._compose_hydra("t63-echam-rrtmgp")
        self.assertEqual(cfg.init.kind, "jw")
        self.assertEqual(cfg.init.rh, 0.0)
        self.assertEqual(cfg.sl_off_centering, 0.2)

    def test_pyses_experiment_keeps_isothermal_and_no_timestep(self):
        # pySES rejects the dinosaur JW init and adopts the dycore dt_seconds,
        # so its experiments override neither init nor run.time_step.
        cfg = self._compose_hydra("ma-ne30-l47")
        self.assertEqual(cfg.hydra.runtime.choices["dycore"], "pyses_ne30l47")
        self.assertEqual(cfg.init.kind, "isothermal")
        self.assertIsNone(cfg.run.time_step)

    def test_speedy_experiment_builds_model(self):
        # Cheap model-construction smoke on the smallest experiment. The big
        # JAM/pySES experiments are deliberately NOT built here (too expensive
        # and network-dependent); terrain/forcing are overridden to aquaplanet
        # so the build needs no boundary-file fetch.
        cfg = _compose(["+experiment=speedy-t31",
                        "terrain=aquaplanet", "forcing=default"])
        model = build_model(cfg)
        self.assertIsNotNone(model)



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
            float(convection_term.params.get_value().entrpen), 4e-4,
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
            float(convection_term.params.get_value().entrpen), 4e-4,
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
        cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
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
        cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma"])
        with open_dict(cfg):
            cfg.physics.cloud_sheme = "2m"      # sic
        with self.assertRaisesRegex(ValueError, "cloud_sheme"):
            build_physics(cfg)

    def test_unknown_builder_raises(self):
        from jcm.runners import build_physics
        cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma"])
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
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
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
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
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
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file=[{p1},{p2}]"])
            f = build_forcing(cfg, coords)
        self.assertIn("emis_surface_combustion_bc", f.anthropogenic_emissions)
        self.assertIn("emis_biomass_burning_bc", f.anthropogenic_emissions)

    def test_duplicate_variable_across_products_raises(self):
        # F1: two products claiming the SAME emission variable is ambiguous —
        # `dict.update` would silently keep the last (double-counting the
        # moment someone lists overlapping bundles). Reject with a build-time
        # error naming the colliding variable and BOTH products.
        import tempfile
        import xarray as xr
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        with tempfile.TemporaryDirectory() as tmp:
            base = {"lon": np.linspace(0, 360, nlon, endpoint=False),
                    "lat": np.linspace(-87, 87, nlat), "time": np.arange(12)}
            p1 = Path(tmp) / "prod_a.nc"
            p2 = Path(tmp) / "prod_b.nc"
            xr.Dataset({"emis_surface_combustion_bc":
                        (("lon", "lat", "time"),
                         np.full((nlon, nlat, 12), 1e-11))},
                       coords=base).to_netcdf(p1)
            xr.Dataset({"emis_surface_combustion_bc":
                        (("lon", "lat", "time"),
                         np.full((nlon, nlat, 12), 2e-11))},
                       coords=base).to_netcdf(p2)
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam",
                            "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file=[{p1},{p2}]"])
            with self.assertRaises(ValueError) as ctx:
                build_forcing(cfg, coords)
        msg = str(ctx.exception)
        self.assertIn("emis_surface_combustion_bc", msg)
        self.assertIn(str(p1), msg)
        self.assertIn(str(p2), msg)

    def test_duplicate_speciated_variable_across_products_raises(self):
        # Same disjoint-merge guard on the pre-speciated (aero_emis_*) channel.
        import tempfile
        import xarray as xr
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        with tempfile.TemporaryDirectory() as tmp:
            base = {"lon": np.linspace(0, 360, nlon, endpoint=False),
                    "lat": np.linspace(-87, 87, nlat), "time": np.arange(12)}
            p1 = Path(tmp) / "spec_a.nc"
            p2 = Path(tmp) / "spec_b.nc"
            for p in (p1, p2):
                xr.Dataset({"aero_emis_m_so4_acc":
                            (("lon", "lat", "time"),
                             np.full((nlon, nlat, 12), 1e-11))},
                           coords=base).to_netcdf(p)
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam",
                            "grid=echam_t42_l8_sigma",
                            f"forcing.emissions_file=[{p1},{p2}]"])
            with self.assertRaises(ValueError) as ctx:
                build_forcing(cfg, coords)
        msg = str(ctx.exception)
        self.assertIn("m_so4_acc", msg)
        self.assertIn(str(p1), msg)
        self.assertIn(str(p2), msg)

    def test_mixed_transient_and_climatology_products_align_independently(self):
        # Codex P1 (round 3): a list mixing a {year} transient product with a
        # non-pattern time-bearing climatology must NOT feed one by-coords
        # merge — disjoint/incompatible time axes would either NaN-fill the
        # non-overlapping steps or clash (integer month vs datetime). Each
        # element is now its own product, opened and time-aligned on its own,
        # and their per-variable TimeSeries leaves merge into one ForcingData
        # keeping distinct alignments.
        import tempfile

        import jax_datetime as jdt
        import xarray as xr
        from omegaconf import open_dict

        from jcm.date import DateData
        from jcm.forcing import BY_DATE, WRAP_YEAR, TimeSeries
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        base = {"lon": np.linspace(0, 360, nlon, endpoint=False),
                "lat": np.linspace(-87, 87, nlat)}
        with tempfile.TemporaryDirectory() as tmp:
            # Transient biomass-burning product: one file per year, 12 monthly
            # datetime steps each; value encodes year*100 + month so the
            # sampled slice is identifiable.
            for yr in (2000, 2001):
                times = np.array([np.datetime64(f"{yr}-{m:02d}-15")
                                  for m in range(1, 13)])
                vals = np.stack([np.full((nlon, nlat), yr * 100 + m)
                                 for m in range(12)])
                xr.Dataset(
                    {"emis_biomass_burning_bc": (("time", "lon", "lat"), vals)},
                    coords={**base, "time": times},
                ).to_netcdf(Path(tmp) / f"bb_{yr}.nc")
            # Climatology product: 12-month, INTEGER month axis — the exact
            # integer-month-vs-datetime mix a single by-coords merge clashes on.
            cvals = np.stack([np.full((nlon, nlat), 10.0 + m)
                              for m in range(12)])
            anthro_path = Path(tmp) / "anthro.nc"
            xr.Dataset(
                {"emis_surface_combustion_bc": (("time", "lon", "lat"), cvals)},
                coords={**base, "time": np.arange(12)},
            ).to_netcdf(anthro_path)

            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam",
                            "grid=echam_t42_l8_sigma"])
            with open_dict(cfg):
                cfg.forcing.emissions_file = [str(Path(tmp) / "bb_{year}.nc"),
                                              str(anthro_path)]
                cfg.forcing.years = [2000, 2001]
            f = build_forcing(cfg, coords)

        em = f.anthropogenic_emissions
        self.assertIn("emis_biomass_burning_bc", em)
        self.assertIn("emis_surface_combustion_bc", em)
        bb, an = em["emis_biomass_burning_bc"], em["emis_surface_combustion_bc"]
        self.assertIsInstance(bb, TimeSeries)
        self.assertIsInstance(an, TimeSeries)
        # Independent alignment: transient BY_DATE over 24 months, climatology
        # WRAP_YEAR over 12 — not one shared axis.
        self.assertEqual(int(bb.align_mode), BY_DATE)
        self.assertEqual(bb.values.shape[0], 24)
        self.assertEqual(int(an.align_mode), WRAP_YEAR)
        self.assertEqual(an.values.shape[0], 12)
        # No NaN fill (the outer-join corruption mode).
        self.assertFalse(bool(np.isnan(np.asarray(bb.values)).any()))
        self.assertFalse(bool(np.isnan(np.asarray(an.values)).any()))
        # Both sampled correctly at a mid-year date: 2001-07 → bb month index 6
        # of 2001 (200106) and climatology month index 6 (16).
        date = DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(
                jdt.to_datetime("2001-07-15")),
            calendar="gregorian")
        sel = f.select(date, calendar="gregorian").anthropogenic_emissions
        self.assertAlmostEqual(
            float(np.asarray(sel["emis_biomass_burning_bc"])[0, 0]), 200106.0)
        self.assertAlmostEqual(
            float(np.asarray(sel["emis_surface_combustion_bc"])[0, 0]), 16.0)

    def test_all_transient_list_products_load_by_date(self):
        # A list of *only* transient {year} products keeps working: each is its
        # own multi-year BY_DATE product and all variables merge.
        import tempfile

        import xarray as xr
        from omegaconf import open_dict

        from jcm.forcing import BY_DATE, TimeSeries
        from jcm.runners import build_forcing
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        base = {"lon": np.linspace(0, 360, nlon, endpoint=False),
                "lat": np.linspace(-87, 87, nlat)}
        with tempfile.TemporaryDirectory() as tmp:
            for stem, var in (("bb", "emis_biomass_burning_bc"),
                              ("an", "emis_surface_combustion_bc")):
                for yr in (2000, 2001):
                    times = np.array([np.datetime64(f"{yr}-{m:02d}-15")
                                      for m in range(1, 13)])
                    vals = np.full((12, nlon, nlat), 1e-11)
                    xr.Dataset({var: (("time", "lon", "lat"), vals)},
                               coords={**base, "time": times}).to_netcdf(
                        Path(tmp) / f"{stem}_{yr}.nc")
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam",
                            "grid=echam_t42_l8_sigma"])
            with open_dict(cfg):
                cfg.forcing.emissions_file = [str(Path(tmp) / "bb_{year}.nc"),
                                              str(Path(tmp) / "an_{year}.nc")]
                cfg.forcing.years = [2000, 2001]
            f = build_forcing(cfg, coords)
        em = f.anthropogenic_emissions
        for var in ("emis_biomass_burning_bc", "emis_surface_combustion_bc"):
            self.assertIsInstance(em[var], TimeSeries)
            self.assertEqual(int(em[var].align_mode), BY_DATE)
            self.assertEqual(em[var].values.shape[0], 24)

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
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
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
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
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
                *_NULL_EMISSIONS,
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
        cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma"])
        f = build_forcing(cfg, coords)
        # No files → no forcing at all (kind: default returns None).
        self.assertIsNone(f)

    def test_lat_mismatch_raises(self):
        import tempfile
        from jcm.runners import build_forcing
        coords = self._coords()
        with tempfile.TemporaryDirectory() as tmp:
            dms, _, _ = self._write_files(tmp, coords, lat_offset=3.0)
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
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
            cfg = _compose([*_NULL_EMISSIONS, "physics=echam-jam", "grid=echam_t42_l8_sigma",
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
        """Resume-from-checkpoint bootstraps a template for state-based inits.

        The resume-from-checkpoint path must build both state pytrees (via
        ``bootstrap_state(balanced_isothermal_state(model))``) as
        deserialization templates before calling ``load_checkpoint``,
        otherwise the load raises on the uninitialised template (codex review
        on PR #479). Held-Suarez is the cheapest physics that supports
        ``init=balanced_isothermal``.
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

        with self.assertLogs("jcm.diffusion", level="WARNING") as captured:
            diffusion = build_diffusion(self._grid_cfg(layers=31, truncation=63))
        self.assertEqual(float(diffusion.temp_timescale),
                         float(DiffusionFilter.default().temp_timescale))
        self.assertIn("no ECHAM lmidatm profile", "\n".join(captured.output))

    def test_build_diffusion_auto_stays_silent_for_non_hybrid_grids(self):
        """SPEEDY/Held-Suarez sigma grids are tuned for the uniform profile."""
        import logging

        from jcm.diffusion import DiffusionFilter

        with self.assertNoLogs("jcm.diffusion", level=logging.WARNING):
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
        with self.assertRaisesRegex(ValueError, r"47 levels .* grid has 95 levels"):
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
        # A stub model shortcuts build_model: before the init dispatch
        # _run_full only touches ``model.coords`` (for the forcing) and
        # ``model.physics`` (for the emulator GHG guard and the config-trap
        # check, both of which no-op on an empty term list).
        stub = _types.SimpleNamespace(
            coords=build_coords(cfg),
            physics=_types.SimpleNamespace(terms=[]))
        with self.assertRaisesRegex(ValueError, "Unknown init.kind"):
            _run_full(cfg, model=stub)

    def test_prescribed_mode_requires_state_file(self):
        from jcm.runners import _load_states_from_cfg
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        cfg.run.mode = "prescribed"
        with self.assertRaisesRegex(ValueError, "state_file"):
            _load_states_from_cfg(cfg, None)

    def test_scm_mode_requires_column(self):
        from jcm.runners import _run_scm
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        cfg.run.mode = "scm"
        # The default config carries a (nulled-out) column block; drop it
        # to exercise the guard for configs that never define one.
        cfg.run.column = None
        with self.assertRaisesRegex(ValueError, "run.column"):
            _run_scm(cfg)


class TestPrescribedStateTracerLoading(unittest.TestCase):
    """``prescribed`` / ``scm`` load the tracers the physics declares (#718).

    With ``run.tracer_vars`` unset the condensate a saved state carries used
    to be dropped, so cloud-aware physics ran against a clear sky and
    returned plausible, wrong fluxes. The tracer list now comes from the
    configured physics rather than from the user remembering to write it out.
    """

    NLEV = 4

    class _StubPhysics:
        """Stands in for a physics package that declares condensate tracers.

        ``_load_states_from_cfg`` only ever calls ``required_tracers()``, so
        building a real ECHAM package here would cost seconds to test one
        line of plumbing.
        """

        def required_tracers(self):
            from jcm.physics.physics_term import TracerSpec
            return (TracerSpec(name="qc"), TracerSpec(name="qi"))

    def _write_state_file(self, path):
        """Write a minimal JCM-convention state file: surface-first, with qc/qi."""
        import xarray as xr

        nlev, nlon, nlat = self.NLEV, 3, 2
        shape = (nlev, nlon, nlat)
        dims = ("level", "lon", "lat")

        def column(profile):
            return np.broadcast_to(
                np.asarray(profile, dtype=float)[:, None, None], shape).copy()

        # Surface-first, as every JCM output file is written: level index 0 is
        # the surface (sigma ~1, ~1e5 Pa).
        ds = xr.Dataset(
            data_vars={
                "u_wind": (dims, np.zeros(shape)),
                "v_wind": (dims, np.zeros(shape)),
                "temperature": (dims, column([288.0, 260.0, 230.0, 200.0])),
                "specific_humidity": (dims, column([1e-2, 5e-3, 1e-3, 1e-6])),
                "geopotential": (dims, np.zeros(shape)),
                "pressure_full": (dims, column([9.9e4, 7e4, 4e4, 1e4])),
                "qc": (dims, column([2e-4, 1e-4, 0.0, 0.0])),
                "qi": (dims, column([0.0, 0.0, 3e-5, 1e-5])),
                "normalized_surface_pressure": (
                    ("lon", "lat"), np.ones((nlon, nlat))),
            },
            coords={"level": np.linspace(0.99, 0.01, nlev)},
        )
        ds.to_netcdf(path)

    def _cfg_with_state_file(self, tmpdir):
        state_file = Path(tmpdir) / "state.nc"
        self._write_state_file(str(state_file))
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        cfg.run.mode = "prescribed"
        cfg.run.state_file = str(state_file)
        return cfg

    def test_declared_tracers_are_loaded_when_tracer_vars_is_unset(self):
        import tempfile
        from jcm.runners import _load_states_from_cfg

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._cfg_with_state_file(tmpdir)
            self.assertIsNone(cfg.run.get("tracer_vars", None))
            _, states = _load_states_from_cfg(cfg, self._StubPhysics())

        self.assertEqual(set(states.tracers), {"qc", "qi"})
        # Loaded in the top-first physics frame like the rest of the state:
        # the file's surface value (index 0) ends up last.
        self.assertAlmostEqual(
            float(np.asarray(states.tracers["qc"])[-1, 0, 0]), 2e-4)

    def test_explicit_empty_mapping_still_loads_nothing(self):
        """``tracer_vars: {}`` is the documented opt-out.

        It must stay distinct from ``null`` -- otherwise, now that ``null``
        means "infer", there is no way to ask for no tracers at all.
        """
        import tempfile
        from jcm.runners import _load_states_from_cfg

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._cfg_with_state_file(tmpdir)
            cfg.run.tracer_vars = {}
            _, states = _load_states_from_cfg(cfg, self._StubPhysics())

        self.assertEqual(dict(states.tracers), {})

    def test_physics_without_tracers_loads_none(self):
        import tempfile
        from jcm.runners import _load_states_from_cfg

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._cfg_with_state_file(tmpdir)
            _, states = _load_states_from_cfg(cfg, build_physics(cfg))

        # held_suarez declares no tracers, so nothing is picked up even
        # though the file carries qc/qi.
        self.assertEqual(dict(states.tracers), {})


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

    def _nodal_ps(self, state, model):
        log_ps_nodal = model.coords.horizontal.to_nodal(
            state.log_surface_pressure
        )[0]
        from dinosaur.scales import units
        scale = float(
            model.dycore.physics_specs.dimensionalize(1.0, units.pascal).m
        )
        return np.exp(np.asarray(log_ps_nodal)) * scale

    def test_balanced_isothermal_rebalances_ps_over_orography(self):
        from jcm.initial_states import balanced_isothermal_state

        model, terrain = self._real_terrain_model()
        state = balanced_isothermal_state(model)
        ps = self._nodal_ps(state, model)
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
        from jcm.initial_states import jw_state

        model, terrain = self._real_terrain_model()
        state = jw_state(model, rh=0.6)
        ps = self._nodal_ps(state, model)
        orog = np.asarray(terrain.orog)
        self.assertLess(
            ps[orog > 2000.0].mean(), 0.8 * ps[orog < 1.0].mean(),
        )

        physics_state = model.dycore.to_physics_state(state)
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
        from jcm.initial_states import jw_state
        from jcm.utils import get_coords

        coords = get_coords(
            vertical_coords=get_echam_levels(47), spectral_truncation=21,
        )
        model = Model(coords=coords, physics=held_suarez_physics(),
                      time_step=180)
        state = jw_state(model, rh=0.5)

        physics_state = model.dycore.to_physics_state(state)
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
    """``jw_state`` must hand the gridpoint physics a physical
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
        from jcm.initial_states import jw_state
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        physics = echam_physics(cloud_scheme="2m", checkpoint_terms=False)
        model = Model(coords=get_speedy_coords(), physics=physics, time_step=180)
        state = jw_state(model, rh=0.6)

        ps = model.dycore.to_physics_state(state)
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
    """``jw_state`` must inject only the analytic humidity profile and
    keep the other prognostic tracers (qc/qi/qnc/qni/qr/qs) the dycore seeded.

    Regression for the CRE ≡ 0 bug: overwriting ``state.tracers`` wholesale
    dropped the cloud tracers, so radiation saw zero cloud water for the entire
    JW-initialised run.
    """

    def test_jw_keeps_2m_cloud_tracers(self):
        from jcm.initial_states import jw_state
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        physics = echam_physics(cloud_scheme="2m", checkpoint_terms=False)
        model = Model(coords=get_speedy_coords(), physics=physics, time_step=180)
        state = jw_state(model, rh=0.5)

        keys = set(state.tracers.keys())
        self.assertIn("specific_humidity", keys)
        # qr/qs are no longer prognostic (2M precipitation is flux-form,
        # review finding 2.18) — the guard covers the four cloud tracers.
        self.assertTrue(
            {"qc", "qi", "qnc", "qni"}.issubset(keys),
            f"jw_state dropped cloud tracers; tracers present: {keys}",
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

    def test_emissions_coverage_clamps_while_surface_keeps_range(self):
        # Codex P2 (round 8): an era5-style cfg runs its SURFACE forcing to
        # 2024 but the mirror's emissions_amip bundle ends 2022. The
        # emissions_available_years override must clamp the emissions
        # expansion to 2022 while the shared available_years keeps the full
        # surface range — following the transient warning's advice otherwise
        # fetches never-built 2023/2024 emission files.
        from omegaconf import OmegaConf

        from jcm import runners
        cfg = OmegaConf.create({
            "years": [2023, 2024],
            "available_years": [1979, 2024],
            "emissions_available_years": [1850, 2022],
        })
        # Emissions clamp to the last built (2022) file...
        self.assertEqual(
            runners._forcing_products(
                "/emis/{year}.nc", cfg.years,
                runners._product_available_years(
                    cfg, "emissions_available_years")),
            [["/emis/2022.nc"]])
        # ...while the surface product (shared available_years, coverage to
        # 2024) still reaches the requested 2023-2024 (plus the one-year
        # by_date_interp bracket on the low side).
        self.assertEqual(
            runners._expand_years(
                "/sst/{year}.nc", cfg.years,
                runners._product_available_years(cfg, "available_years")),
            ["/sst/2022.nc", "/sst/2023.nc", "/sst/2024.nc"])

    def test_emissions_oxidants_coverage_fall_back_to_shared(self):
        # Per-product override beats the shared fallback; absent it, each
        # transient-capable input inherits available_years (so it can never
        # silently drift from the surface coverage).
        from omegaconf import OmegaConf

        from jcm import runners
        cfg = OmegaConf.create({
            "available_years": [1979, 2024],
            "emissions_available_years": [1850, 2022],
        })
        self.assertEqual(
            list(runners._product_available_years(
                cfg, "emissions_available_years")),
            [1850, 2022])
        # oxidants has no override here → shared available_years.
        self.assertEqual(
            runners._product_available_years(
                cfg, "oxidants_available_years"),
            cfg.available_years)

    def test_era5_preset_composes(self):
        cfg = _compose(["forcing=era5", "forcing.years=[2023,2024]",
                        "run.start_date=2023-01-01"])
        self.assertEqual(cfg.forcing.align, "by_date_interp")
        self.assertIn("forcing_era5", cfg.forcing.file)
        self.assertEqual(list(cfg.forcing.ozone_available_years)[-1], 2022)
        # Surface files run to 2024 but the emissions_amip bundle ends 2022,
        # so the preset ships a per-product emissions clamp (Codex round 8).
        self.assertEqual(list(cfg.forcing.available_years)[-1], 2024)
        self.assertEqual(list(cfg.forcing.emissions_available_years)[-1], 2022)

    def test_list_spec_splits_into_per_element_products(self):
        # emissions_file may be a list (e.g. biomass-burning + anthropogenic).
        # Each element is its OWN product: a {year} element expands to its
        # yearly file list (one product concatenated along one time axis),
        # while a static element stays scalar. The list is NOT flattened, so
        # each product is opened and time-aligned on its own downstream (a
        # transient product and a climatology must not share one time axis).
        from jcm import runners
        self.assertEqual(
            runners._forcing_products(
                ["/bb_{year}.nc", "/anthro.nc"], [2000, 2001], None),
            [["/bb_2000.nc", "/bb_2001.nc"], "/anthro.nc"])
        # A list of static paths keeps one product per element, unchanged.
        self.assertEqual(
            runners._forcing_products(["/a.nc", "/b.nc"], [2000, 2001], None),
            ["/a.nc", "/b.nc"])
        # A scalar spec is a single product; a {year} scalar becomes that
        # product's yearly file list.
        self.assertEqual(
            runners._forcing_products("/anthro.nc", [2000, 2001], None),
            ["/anthro.nc"])
        self.assertEqual(
            runners._forcing_products("/bb_{year}.nc", [2000, 2001], None),
            [["/bb_2000.nc", "/bb_2001.nc"]])

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


class TestEmulatorWeightsBuilderPath(unittest.TestCase):
    """``emulator_weights_file`` via the echam_physics builder (Codex P2).

    ``_build_physics_from_factory`` drops ``null`` kwargs, so
    ``physics.emulator_weights_file=null`` cannot mean "random init" — it
    resolves to the ``auto`` default. Train-from-scratch is the explicit
    ``"random"`` sentinel instead.
    """

    def _rad_term(self, physics):
        return next(t for t in physics.terms
                    if getattr(t, "name", "") == "nn_emulator_radiation")

    def _build(self, **extra):
        from omegaconf import OmegaConf

        from jcm.runners import _build_physics_from_factory
        cfg = OmegaConf.create({
            "builder": "echam_physics",
            "radiation_scheme": "emulated",
            **extra,
        })
        return _build_physics_from_factory(cfg)

    def test_random_reaches_random_init(self):
        term = self._rad_term(self._build(emulator_weights_file="random"))
        self.assertIsNone(term._weights_file)

    def test_null_falls_back_to_auto(self):
        # null is dropped by the builder → the "auto" default (packaged).
        term = self._rad_term(self._build(emulator_weights_file=None))
        self.assertIsNotNone(term._weights_file)
        self.assertTrue(str(term._weights_file).endswith(
            "emulator_weights_per_band_u64.nc"))

    def test_absent_defaults_to_auto(self):
        term = self._rad_term(self._build())
        self.assertIsNotNone(term._weights_file)
        self.assertTrue(str(term._weights_file).endswith(
            "emulator_weights_per_band_u64.nc"))


class TestEmulatorGhgGuard(unittest.TestCase):
    """The emulator must refuse CH4/N2O forcing it cannot represent.

    Its features carry only ozone and CO2 and its labels are generated at
    RRTMGP's defaults, so a scenario varying either gas would get fluxes
    with no trace of the forcing (jax-gcm#738).
    """

    def _physics(self, emulated):
        from jcm.physics.echam.echam_terms import echam_physics
        return echam_physics(
            radiation_scheme="emulated" if emulated else "rrtmgp")

    def _forcing(self, **overrides):
        from jcm.forcing import (
            DEFAULT_CH4_VMR_PPMV, DEFAULT_N2O_VMR_PPMV,
        )
        import types
        return types.SimpleNamespace(
            ch4_vmr=np.asarray(overrides.get("ch4", DEFAULT_CH4_VMR_PPMV)),
            n2o_vmr=np.asarray(overrides.get("n2o", DEFAULT_N2O_VMR_PPMV)),
        )

    def test_default_greenhouse_gases_pass(self):
        guard_emulator_ghg_forcing(self._physics(True), self._forcing())

    def test_non_default_ch4_is_rejected_with_an_actionable_message(self):
        with self.assertRaises(ValueError) as ctx:
            guard_emulator_ghg_forcing(self._physics(True),
                                       self._forcing(ch4=3.0))
        msg = str(ctx.exception)
        self.assertIn("ch4_vmr", msg)
        self.assertIn("echam-rrtmgp-2m", msg)

    def test_non_default_n2o_is_rejected(self):
        with self.assertRaises(ValueError):
            guard_emulator_ghg_forcing(self._physics(True),
                                       self._forcing(n2o=0.40))

    def test_rrtmgp_consumes_both_gases_so_it_is_never_guarded(self):
        guard_emulator_ghg_forcing(self._physics(False),
                                   self._forcing(ch4=3.0, n2o=0.40))

    def test_absent_forcing_is_not_an_error(self):
        guard_emulator_ghg_forcing(self._physics(True), None)

    def test_transient_timeseries_forcing_is_inspected(self):
        # A file-based scenario stores the gas as a TimeSeries, which is
        # exactly the case the guard exists for; np.asarray raises on it,
        # so an unwrapped check would silently pass the run through.
        import types

        from jcm.forcing import DEFAULT_CH4_VMR_PPMV, make_time_series

        rising = make_time_series(
            np.array([DEFAULT_CH4_VMR_PPMV, 2.4, 3.0]),
            np.array([0.0, 1.0, 2.0]),
        )
        forcing = types.SimpleNamespace(
            ch4_vmr=rising, n2o_vmr=self._forcing().n2o_vmr)
        with self.assertRaises(ValueError) as ctx:
            guard_emulator_ghg_forcing(self._physics(True), forcing)
        self.assertIn("ch4_vmr", str(ctx.exception))

    def test_constant_default_timeseries_still_passes(self):
        import types

        from jcm.forcing import DEFAULT_CH4_VMR_PPMV, make_time_series

        flat = make_time_series(
            np.full(3, DEFAULT_CH4_VMR_PPMV), np.array([0.0, 1.0, 2.0]))
        forcing = types.SimpleNamespace(
            ch4_vmr=flat, n2o_vmr=self._forcing().n2o_vmr)
        guard_emulator_ghg_forcing(self._physics(True), forcing)


class TestWarnOnConfigTraps:
    """The config cross-validation warnings (invalid-but-runnable combos).

    Uses lightweight stand-ins — a physics object is just something with a
    ``.terms`` list of named terms, and a forcing is a namespace carrying the
    two MACv2-SP weight fields — so the checks are exercised without building a
    real (expensive) model. Every finding is a WARNING; the tests assert both
    that it fires on its trap combo and that it stays silent on the sane one.
    """

    @staticmethod
    def _physics(*names):
        import types
        return types.SimpleNamespace(
            terms=[types.SimpleNamespace(name=n) for n in names])

    @staticmethod
    def _cfg(terrain="aquaplanet", forcing_kind="default", **forcing_keys):
        from omegaconf import OmegaConf
        return OmegaConf.create(
            {"terrain": {"kind": terrain},
             "forcing": {"kind": forcing_kind, **forcing_keys}})

    @staticmethod
    def _coords(truncation=63):
        # ``_grid_token`` reads ``coords.horizontal.total_wavenumbers`` and maps
        # ``total_wavenumbers - 2`` to ``t{trunc}``; a lightweight stub avoids
        # building a real (expensive) coordinate system just to pick the grid
        # token. t63 is published; t42 (truncation=42) is not.
        import types
        return types.SimpleNamespace(
            horizontal=types.SimpleNamespace(
                total_wavenumbers=int(truncation) + 2))

    @staticmethod
    def _pyses_dycore():
        # ``is_pyses`` is decided by ``hasattr(dycore, "colmap")``.
        import types
        return types.SimpleNamespace(colmap=object())

    @staticmethod
    def _macv2_forcing(loaded=False):
        import types

        import jax.numpy as jnp

        from jcm.forcing import make_time_series
        if loaded:
            yw = make_time_series(jnp.full((2, 9), 0.3), jnp.arange(2.0))
            ac = make_time_series(jnp.full((2, 2, 9), 0.7), jnp.arange(2.0))
        else:
            yw = jnp.ones(9)
            ac = jnp.ones((2, 9))
        return types.SimpleNamespace(
            aerosol_year_weight=yw, aerosol_ann_cycle=ac)

    # 1. JAM + aquaplanet terrain
    def test_jam_aquaplanet_terrain_warns(self, caplog):
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="aquaplanet"),
                self._physics("jam_seasalt_emissions"), None)
        assert "terrain=aquaplanet" in caplog.text
        assert "Gong sea-salt" in caplog.text

    def test_jam_realistic_terrain_silent(self, caplog):
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="from_file"),
                self._physics("jam_seasalt_emissions"), None)
        assert "terrain=aquaplanet" not in caplog.text

    # 2. aquaplanet terrain + from_file forcing
    def test_aquaplanet_from_file_forcing_warns(self, caplog):
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="aquaplanet", forcing_kind="from_file"),
                self._physics("macv2_sp_aerosol"), None)
        assert "from_file over terrain=aquaplanet" in caplog.text

    def test_from_file_forcing_with_real_terrain_silent(self, caplog):
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="from_file", forcing_kind="from_file"),
                self._physics("macv2_sp_aerosol"),
                self._macv2_forcing(loaded=True))
        assert "from_file over terrain=aquaplanet" not in caplog.text

    # 3. Prognostic aerosol with every emission input nulled
    def test_jam_zero_emissions_warns_and_names_keys(self, caplog):
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(terrain="from_file", emissions_file=None,
                        dms_file=None, dust_file=None, oxidants_file=None)
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"), None)
        assert "zero-emission JAM baseline" in caplog.text
        for key in ("emissions_file", "dms_file", "dust_file",
                    "oxidants_file"):
            assert key in caplog.text

    def test_jam_with_emissions_silent(self, caplog):
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(terrain="from_file",
                        emissions_file="hf://bundles/t63/emissions_pd.nc",
                        dms_file="hf://bundles/t63/dms.nc",
                        dust_file="hf://bundles/t63/dust.nc",
                        oxidants_file="hf://bundles/t63_l47/oxidants_pd.nc")
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"), None)
        assert "zero-emission JAM baseline" not in caplog.text

    # 4. MACv2-SP all-ones default weights
    def test_macv2_default_weights_warn(self, caplog):
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="from_file"),
                self._physics("macv2_sp_aerosol"),
                self._macv2_forcing(loaded=False))
        assert "perpetual year-2005" in caplog.text

    def test_macv2_default_weights_warn_when_forcing_none(self, caplog):
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="from_file"),
                self._physics("macv2_sp_aerosol"), None)
        assert "perpetual year-2005" in caplog.text

    def test_macv2_loaded_weights_silent(self, caplog):
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="from_file"),
                self._physics("macv2_sp_aerosol"),
                self._macv2_forcing(loaded=True))
        assert "perpetual year-2005" not in caplog.text

    def test_macv2_on_jam_path_silent(self, caplog):
        # JAM keeps MACv2-SP as a passive optics fudge; warning 4 must not
        # fire there (the all-ones weights are not the concern on that path).
        from jcm.runners import warn_on_config_traps
        with caplog.at_level("WARNING"):
            warn_on_config_traps(
                self._cfg(terrain="from_file"),
                self._physics("macv2_sp_aerosol", "jam_seasalt_emissions"),
                self._macv2_forcing(loaded=False))
        assert "perpetual year-2005" not in caplog.text

    # 5. Transient (amip/era5) forcing + present-day JAM emissions
    def test_transient_forcing_present_day_jam_emissions_warns(self, caplog):
        # amip/era5 encode transience as a `years` range + by_date_interp
        # align on top of `kind: from_file`; emissions_file/oxidants_file: auto
        # then resolve the present-day *_pd bundles. Read on a PUBLISHED grid
        # (t63) so `auto` resolves to a real bundle — the mismatch this warns
        # about (F2). On an unpublished grid `auto` resolves to None instead,
        # which warning 3 covers (see the pySES/t42 tests below).
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(terrain="from_file", forcing_kind="from_file",
                        years=[1979, 1983], align="by_date_interp",
                        emissions_file="auto", dms_file="auto",
                        dust_file="auto", oxidants_file="auto")
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"),
                                 None, coords=self._coords(63))
        assert "present-day JAM emissions" in caplog.text
        assert "emissions_file" in caplog.text
        assert "oxidants_file" in caplog.text

    def test_transient_forcing_explicit_emission_paths_silent(self, caplog):
        # Explicit year-matched paths (not `auto`) opt out of the warning.
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(
            terrain="from_file", forcing_kind="from_file",
            years=[1979, 1983], align="by_date_interp",
            emissions_file="hf://bundles/t63/emissions_1980.nc",
            oxidants_file="hf://bundles/t63_l47/oxidants_1980.nc")
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"), None)
        assert "present-day JAM emissions" not in caplog.text

    def test_default_forcing_auto_emissions_silent(self, caplog):
        # Non-transient forcing (no `years`, default align) + auto on a
        # PUBLISHED grid is the canonical present-day run — auto resolves to
        # real bundles, so neither the transient mismatch (5) nor the
        # zero-emission baseline (3) fires.
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(terrain="from_file", forcing_kind="default",
                        emissions_file="auto", dms_file="auto",
                        dust_file="auto", oxidants_file="auto")
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"),
                                 None, coords=self._coords(63))
        assert "present-day JAM emissions" not in caplog.text
        assert "zero-emission JAM baseline" not in caplog.text

    # 3 (F2). RESOLVED-value awareness: `auto` that nulls on the pySES path or
    # an unpublished grid is the silently-degraded run the zero-emission
    # warning must catch — the raw cfg still reads "auto" and would miss it.
    def test_jam_auto_null_on_pyses_warns(self, caplog):
        # ne30/pySES JAM: the four emission keys are `auto` (default) but the
        # pySES backend publishes no per-grid bundles, so every one resolves to
        # None — a sea-salt-only run that used to be silent.
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(terrain="from_file", forcing_kind="from_file",
                        emissions_file="auto", dms_file="auto",
                        dust_file="auto", oxidants_file="auto")
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"),
                                 None, coords=self._coords(63),
                                 dycore=self._pyses_dycore())
        assert "zero-emission JAM baseline" in caplog.text
        assert "pySES" in caplog.text
        # No spurious present-day-emissions warning: those keys resolved to
        # None, not to a *_pd bundle.
        assert "present-day JAM emissions" not in caplog.text

    def test_jam_auto_null_non_mirrored_grid_warns(self, caplog):
        # A JAM spectral run on an unpublished grid (t42): `auto` resolves to
        # None (no bundle to fetch) and the ONE combined warning names the grid
        # — replacing the invisible info-level log the resolver used to emit.
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(terrain="from_file", forcing_kind="from_file",
                        emissions_file="auto", dms_file="auto",
                        dust_file="auto", oxidants_file="auto")
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"),
                                 None, coords=self._coords(42))
        assert "zero-emission JAM baseline" in caplog.text
        assert "'t42'" in caplog.text
        for key in ("emissions_file", "dms_file", "dust_file",
                    "oxidants_file"):
            assert key in caplog.text

    def test_jam_auto_on_published_grid_silent(self, caplog):
        # t63 JAM with the `auto` bundles: every key resolves to a real
        # present-day bundle, so the zero-emission warning stays silent.
        from jcm.runners import warn_on_config_traps
        cfg = self._cfg(terrain="from_file", forcing_kind="from_file",
                        emissions_file="auto", dms_file="auto",
                        dust_file="auto", oxidants_file="auto")
        with caplog.at_level("WARNING"):
            warn_on_config_traps(cfg, self._physics("jam_dust_emissions"),
                                 None, coords=self._coords(63))
        assert "zero-emission JAM baseline" not in caplog.text


class TestAttachMacv2Weights(unittest.TestCase):
    """``forcing.macv2_file`` loads real MACv2-SP weights onto ForcingData."""

    @staticmethod
    def _write_macv2(path):
        import numpy as np
        import xarray as xr
        nplume, years, nweek, nfeat = 9, np.array([2004, 2005, 2006]), 52, 2
        yw = np.arange(nplume * years.size,
                       dtype=float).reshape(nplume, years.size)
        ac = np.arange(nplume * nweek * nfeat,
                       dtype=float).reshape(nplume, nweek, nfeat)
        xr.Dataset(
            {"year_weight": (("plume", "years"), yw),
             "ann_cycle": (("plume", "week", "feature"), ac)},
            coords={"years": years},
        ).to_netcdf(path)

    def test_macv2_file_attaches_timeseries(self):
        import tempfile

        from omegaconf import OmegaConf

        from jcm.forcing import TimeSeries
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.runners import _attach_macv2_weights

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MACv2.0-SP_v1.nc"
            self._write_macv2(p)
            cfg = OmegaConf.create({"kind": "default", "macv2_file": str(p)})
            forcing = _attach_macv2_weights(None, cfg, coords)
        self.assertIsInstance(forcing.aerosol_year_weight, TimeSeries)
        self.assertIsInstance(forcing.aerosol_ann_cycle, TimeSeries)
        self.assertEqual(forcing.aerosol_year_weight.values.shape, (3, 9))

    def test_macv2_file_unset_is_noop(self):
        from omegaconf import OmegaConf

        from jcm.runners import _attach_macv2_weights
        cfg = OmegaConf.create({"kind": "default", "macv2_file": None})
        self.assertIsNone(_attach_macv2_weights(None, cfg, None))

    def test_pyses_path_attaches_macv2_weights(self):
        """``forcing=macv2_sp`` on pySES must still load ``macv2_file`` (F1).

        ``build_forcing``'s pySES branch returns before the spectral tail, so
        the ONE dycore-agnostic attachment the tail performs that
        ``pyses_build_forcing`` does not — the plume-indexed MACv2-SP weights
        (no horizontal field, so no column sampling needed) — must be applied on
        the pySES path via the shared ``_attach_macv2_weights``. Otherwise
        ``forcing=macv2_sp`` loads the surface file but silently drops its
        mandatory ``macv2_file``, the exact silent-ignore trap warning 4
        recommends this very config to escape.
        """
        import tempfile
        import types
        from unittest import mock

        from omegaconf import OmegaConf

        from jcm.forcing import ForcingData, TimeSeries
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.runners import build_forcing

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        # ``is_pyses`` is decided by ``hasattr(dycore, "colmap")``; a stub with
        # that attribute takes the pySES branch without a real CAM-SE build.
        fake_dycore = types.SimpleNamespace(colmap=object())
        base = ForcingData.zeros(nodal_shape=(1, 4))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MACv2.0-SP_v1.nc"
            self._write_macv2(p)
            cfg = OmegaConf.create({"forcing": {
                "kind": "from_file", "file": "dummy.nc", "ozone_file": "null",
                "emissions_file": "null", "dms_file": "null",
                "dust_file": "null", "oxidants_file": "null",
                "macv2_file": str(p)}})
            with mock.patch("jcm.dycore.pyses.forcing.build_forcing",
                            return_value=base) as pfb:
                forcing = build_forcing(cfg, coords, dycore=fake_dycore)
        pfb.assert_called_once()
        # The weights rode through the pySES early return.
        self.assertIsInstance(forcing.aerosol_year_weight, TimeSeries)
        self.assertIsInstance(forcing.aerosol_ann_cycle, TimeSeries)


class TestEmissionAutoResolution(unittest.TestCase):
    """The ``auto`` prescribed-emission resolution (issue #640).

    ``auto`` is the only grid-portable mechanism: it composes the concrete
    per-grid bundle path from :mod:`jcm.data.bundle_names` + the grid token, so
    one config follows the grid. There is no user-facing ``{grid}``/``{nlev}``
    path template (removed as a redundant simplification, #640) — an explicit
    path is taken verbatim; only ``{year}`` is expanded downstream.
    """

    def _coords(self):
        from jcm.runners import build_coords
        return build_coords(
            _compose(["physics=echam", "grid=echam_t63_l47_hybrid"]))

    def test_explicit_path_taken_verbatim(self):
        # No {grid}/{nlev} substitution: an explicit forcing path passes through
        # _resolve_one_emission_input unchanged (auto is the grid-portable path).
        from jcm.runners import _resolve_one_emission_input
        coords = self._coords()
        self.assertEqual(
            _resolve_one_emission_input(
                "hf://bundles/t63_l47/oxidants_pd.nc", "oxidants_file",
                coords, jam=True, is_pyses=False),
            "hf://bundles/t63_l47/oxidants_pd.nc")
        # A {year} pattern is left intact for the later yearly expansion.
        self.assertEqual(
            _resolve_one_emission_input(
                "hf://bundles/t63/emissions/{year}.nc", "emissions_file",
                coords, jam=True, is_pyses=False),
            "hf://bundles/t63/emissions/{year}.nc")
        # Explicit null opts out.
        self.assertIsNone(
            _resolve_one_emission_input(
                "null", "emissions_file", coords, jam=True, is_pyses=False))

    def test_auto_builds_per_grid_bundle_for_jam(self):
        from unittest import mock

        from jcm import runners
        coords = self._coords()
        # Echo the hf:// path instead of fetching, so we can assert what the
        # auto default resolved to.
        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=lambda p: p):
            emis = runners._resolve_one_emission_input(
                "auto", "emissions_file", coords, jam=True, is_pyses=False)
            ox = runners._resolve_one_emission_input(
                "auto", "oxidants_file", coords, jam=True, is_pyses=False)
        self.assertEqual(emis, "hf://bundles/t63/emissions_pd.nc")
        self.assertEqual(ox, "hf://bundles/t63_l47/oxidants_pd.nc")

    def test_auto_is_none_for_non_jam_and_pyses(self):
        from jcm.runners import _resolve_one_emission_input
        coords = self._coords()
        self.assertIsNone(_resolve_one_emission_input(
            "auto", "dms_file", coords, jam=False, is_pyses=False))
        self.assertIsNone(_resolve_one_emission_input(
            "auto", "dms_file", coords, jam=True, is_pyses=True))

    def test_explicit_null_opts_out(self):
        from jcm.runners import _resolve_one_emission_input
        coords = self._coords()
        for val in (None, "null", ""):
            self.assertIsNone(_resolve_one_emission_input(
                val, "dust_file", coords, jam=True, is_pyses=False))

    def test_auto_fetch_failure_is_loud_and_names_the_fix(self):
        from unittest import mock

        from jcm import runners
        coords = self._coords()

        def _raise(_p):
            raise FileNotFoundError("cold cache")

        with mock.patch.object(runners, "_resolve_data_path", side_effect=_raise):
            with self.assertRaises(FileNotFoundError) as ctx:
                runners._resolve_one_emission_input(
                    "auto", "emissions_file", coords, jam=True, is_pyses=False)
        msg = str(ctx.exception)
        self.assertIn("hf://bundles/t63/emissions_pd.nc", msg)
        self.assertIn("forcing.emissions_file=null", msg)
        self.assertIn("prefetch", msg.lower())


class TestBuildForcingAutoEmissionsWiring(unittest.TestCase):
    """End-to-end: a JAM ``auto`` default reaches the bundle fetch.

    And fails loudly when the bundle cannot be resolved.
    """

    def test_jam_default_forcing_fetches_bundles_and_fails_loud(self):
        from unittest import mock

        from jcm.runners import build_coords, build_forcing
        # A published grid (t63): the auto default resolves the per-grid bundle
        # and eagerly fetches it, so a cold cache must fail loudly here.
        cfg = _compose(["physics=echam-jam", "grid=echam_t63_l47_hybrid",
                        "physics.jam_microphysics=placeholder",
                        "physics.radiation_scheme=grey"])
        coords = build_coords(cfg)

        def _raise(_path, **_kw):
            raise FileNotFoundError("no network in test")

        # The auto default resolves the first emission bundle, whose fetch we
        # force to fail — build_forcing must surface it, not silently continue.
        with mock.patch("jcm.data.remote.fetch", side_effect=_raise):
            with self.assertRaises(FileNotFoundError) as ctx:
                build_forcing(cfg, coords)
        self.assertIn("hf://bundles/t63/", str(ctx.exception))

    def test_jam_auto_nulls_on_non_mirrored_grid_without_fetch(self):
        """A non-mirrored grid (t42) auto-nulls emissions, no fetch (Codex P1).

        The mirror publishes bundles only for ``PUBLISHED_GRIDS``; a JAM run on
        any other grid (e.g. echam_t42_l8_sigma) must restore the null,
        emission-free baseline automatically — resolving ``auto`` to None with
        NO fetch (so the documented ``physics=echam-jam grid=echam_t42_l8_sigma
        forcing.emissions_file=<explicit>`` workflow no longer aborts). The
        silent-degrade WARNING is emitted by ``warn_on_config_traps`` from the
        resolved values (F2), not an info log in the resolver — so the resolver
        stays a pure, side-effect-light resolution
        (see ``TestConfigTraps.test_jam_auto_null_non_mirrored_grid_warns``).
        """
        from unittest import mock

        from jcm import runners
        from jcm.runners import build_coords
        cfg = _compose(["physics=echam-jam", "grid=echam_t42_l8_sigma",
                        "physics.jam_microphysics=placeholder",
                        "physics.radiation_scheme=grey"])
        coords = build_coords(cfg)

        def _no_fetch(path):
            raise AssertionError(f"no fetch expected, got {path!r}")

        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=_no_fetch):
            out = runners._resolve_emission_inputs(
                cfg.forcing, cfg, coords, is_pyses=False)
        for key in ("emissions_file", "dms_file", "dust_file",
                    "oxidants_file"):
            self.assertIsNone(out.get(key))

    def test_t119_jam_experiment_resolves_emission_free(self):
        """The ma-t119 experiments run emission-free (Codex P1).

        T119 has no mirror bundle, so the JAM ``auto`` default cannot resolve
        the four emission keys (bundles/t119/*.nc do not exist and the fetch
        would abort build_forcing). The experiment yamls null the keys
        explicitly (round-1 documentation — no longer load-bearing now that the
        published-grid whitelist auto-nulls any non-mirrored grid, see
        ``test_jam_auto_nulls_on_non_mirrored_grid_without_fetch``); assert the
        resolver requests NO fetch and yields None either way, so
        ``python -m jcm.main +experiment=ma-t119-l47`` is one command.
        """
        from unittest import mock

        from jcm import runners
        from jcm.runners import build_coords
        for name in ("ma-t119-l47", "ma-t119-l95"):
            with self.subTest(name):
                cfg = _compose([f"+experiment={name}"])
                coords = build_coords(cfg)
                calls = []

                def _record(path):
                    calls.append(path)
                    raise AssertionError("no emission fetch expected")

                with mock.patch.object(runners, "_resolve_data_path",
                                       side_effect=_record):
                    out = runners._resolve_emission_inputs(
                        cfg.forcing, cfg, coords, is_pyses=False)
                self.assertEqual(calls, [])
                for key in ("emissions_file", "dms_file", "dust_file",
                            "oxidants_file"):
                    self.assertIsNone(out.get(key))

    def test_year_pattern_resolves_end_to_end(self):
        """A ``{year}`` emissions pattern expands to one file per year.

        ``{year}`` is the only remaining path template; the yearly expansion
        (issue #610) produces one file per year, opened together as one product.
        """
        from unittest import mock

        import xarray as xr
        from omegaconf import OmegaConf

        from jcm import runners
        from jcm.runners import build_coords, build_forcing
        cfg = _compose([
            "physics=echam-jam", "grid=echam_t42_l8_sigma",
            "physics.jam_microphysics=placeholder",
            "physics.radiation_scheme=grey",
            *_NULL_EMISSIONS,
        ])
        # Set the pattern path and year range directly: the Hydra override
        # grammar treats the ``{`` in ``{year}`` as syntax, so it cannot be
        # passed on the command line — but a preset yaml carries it verbatim.
        # ``years`` is likewise not in the base forcing struct.
        OmegaConf.set_struct(cfg, False)
        cfg.forcing.emissions_file = "hf://bundles/t42/emis/{year}.nc"
        cfg.forcing.years = [2000, 2001]
        coords = build_coords(cfg)

        seen = {}

        def _capture_mfdataset(paths, **_kw):
            seen["paths"] = list(paths)
            return xr.Dataset()

        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=lambda p: p), \
                mock.patch("xarray.open_mfdataset",
                           side_effect=_capture_mfdataset), \
                mock.patch("jcm.forcing.read_anthropogenic_emissions",
                           return_value={"sector": object()}), \
                mock.patch("jcm.forcing.read_prescribed_aerosol_emissions",
                           return_value=None), \
                mock.patch("jcm.forcing.validate_emissions_grid"):
            build_forcing(cfg, coords)

        # {year} expanded to the inclusive range; opened as one product.
        self.assertEqual(
            seen["paths"],
            ["hf://bundles/t42/emis/2000.nc", "hf://bundles/t42/emis/2001.nc"])

    def test_oxidants_year_matched_pattern_expands_and_concatenates(self):
        """A year-matched oxidants pattern expands + concatenates (Codex P1).

        Warning 5 recommends ``oxidants_file=.../{year}.nc`` for a transient
        run; ``_attach_oxidants`` must therefore honour the same yearly
        expansion + by-coords merge the emissions path does, or that remedy
        fails at startup. Assert the per-year files reach ``open_mfdataset``.
        """
        from unittest import mock

        import xarray as xr
        from omegaconf import OmegaConf

        from jcm import runners
        from jcm.runners import build_coords, build_forcing
        cfg = _compose([
            "physics=echam-jam", "grid=echam_t42_l8_sigma",
            "physics.jam_microphysics=placeholder",
            "physics.radiation_scheme=grey",
            *_NULL_EMISSIONS,
        ])
        OmegaConf.set_struct(cfg, False)
        cfg.forcing.oxidants_file = \
            "hf://bundles/t42_l8/oxidants_{year}.nc"
        cfg.forcing.years = [2000, 2001]
        coords = build_coords(cfg)

        seen = {}

        def _capture_mfdataset(paths, **_kw):
            seen["paths"] = list(paths)
            return xr.Dataset()

        # The set carries >1 file, so _attach_oxidants opens each to classify
        # its time axis (the incompatible-mixture guard). Both yearly files are
        # transient (datetime), so the set is uniform and the read proceeds.
        def _open_datetime_stub(path, **_kw):
            return xr.Dataset(
                coords={"time": np.array(["2000-06-15"], dtype="datetime64[ns]")})

        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=lambda p: p), \
                mock.patch("xarray.open_mfdataset",
                           side_effect=_capture_mfdataset), \
                mock.patch("xarray.open_dataset",
                           side_effect=_open_datetime_stub), \
                mock.patch("jcm.forcing.read_oxidant_vmr",
                           return_value={"oh": object()}) as read_mock, \
                mock.patch("jcm.forcing.validate_oxidant_levels"):
            build_forcing(cfg, coords)

        # {year} expanded, files concatenated by coords as one product.
        self.assertEqual(
            seen["paths"],
            ["hf://bundles/t42_l8/oxidants_2000.nc",
             "hf://bundles/t42_l8/oxidants_2001.nc"])
        # Multi-year axis → "auto" alignment (BY_DATE for the transient run).
        self.assertEqual(read_mock.call_args.kwargs["align_mode"], "auto")

    def test_oxidants_explicit_list_is_one_product(self):
        """An explicit-list oxidants_file is ONE product, opened together (F2).

        Unlike emissions (per-product merge over disjoint variables), every
        oxidant file must carry all four gases, so a list is the yearly files of
        a single product: the whole set goes to one ``open_mfdataset`` along one
        time axis and is read once. (A per-product ``dict.update`` would have
        been pure last-one-wins for the fully-overlapping oxidant maps.)
        """
        from unittest import mock

        import xarray as xr
        from omegaconf import OmegaConf

        from jcm import runners
        from jcm.runners import build_coords, build_forcing
        cfg = _compose([
            "physics=echam-jam", "grid=echam_t42_l8_sigma",
            "physics.jam_microphysics=placeholder",
            "physics.radiation_scheme=grey", *_NULL_EMISSIONS,
        ])
        OmegaConf.set_struct(cfg, False)
        cfg.forcing.oxidants_file = ["/ox_a.nc", "/ox_b.nc"]
        coords = build_coords(cfg)

        seen = {}

        def _capture_mfdataset(paths, **_kw):
            seen["paths"] = list(paths)
            return xr.Dataset()

        # Both members transient (datetime) → uniform, read proceeds.
        def _open_datetime_stub(path, **_kw):
            return xr.Dataset(coords={
                "time": np.array(["2000-06-15"], dtype="datetime64[ns]")})

        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=lambda p: p), \
                mock.patch("xarray.open_mfdataset",
                           side_effect=_capture_mfdataset), \
                mock.patch("xarray.open_dataset",
                           side_effect=_open_datetime_stub), \
                mock.patch("jcm.forcing.read_oxidant_vmr",
                           return_value={"oh": object(), "no3": object()}), \
                mock.patch("jcm.forcing.validate_oxidant_levels"):
            build_forcing(cfg, coords)

        # The whole list reached a single open_mfdataset (one product, one axis).
        self.assertEqual(seen["paths"], ["/ox_a.nc", "/ox_b.nc"])

    def test_oxidants_mixed_time_axes_raise(self):
        """A mixed integer-month + datetime oxidant set is rejected loudly (F2).

        The genuinely-incompatible case ``_assert_uniform_oxidant_time_axis``
        exists to catch: one member on an integer-month climatology axis, one on
        a datetime transient axis, in a single product's file set — silent
        NaN-fill / cryptic dtype clash under ``open_mfdataset`` is exactly the
        silent-ignore class this hardening abolishes.
        """
        from unittest import mock

        import xarray as xr
        from omegaconf import OmegaConf

        from jcm import runners
        from jcm.runners import build_coords, build_forcing
        cfg = _compose([
            "physics=echam-jam", "grid=echam_t42_l8_sigma",
            "physics.jam_microphysics=placeholder",
            "physics.radiation_scheme=grey", *_NULL_EMISSIONS,
        ])
        OmegaConf.set_struct(cfg, False)
        cfg.forcing.oxidants_file = ["/clim.nc", "/transient.nc"]
        coords = build_coords(cfg)

        def _open_mixed(path, **_kw):
            if "clim" in str(path):
                return xr.Dataset(coords={"time": np.arange(12)})  # integer month
            return xr.Dataset(coords={
                "time": np.array(["2000-06-15"], dtype="datetime64[ns]")})

        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=lambda p: p), \
                mock.patch("xarray.open_dataset", side_effect=_open_mixed):
            with self.assertRaisesRegex(ValueError, "incompatible time axes"):
                build_forcing(cfg, coords)

    def _pyses_cfg(self, **forcing):
        """Build a minimal pySES-path forcing cfg (nulls unless overridden)."""
        from omegaconf import OmegaConf
        base = {"kind": "from_file", "file": "dummy.nc", "ozone_file": "null",
                "emissions_file": "null", "dms_file": "null",
                "dust_file": "null", "oxidants_file": "null"}
        base.update(forcing)
        return OmegaConf.create({"forcing": base})

    def _pyses_dycore_and_coords(self):
        import types

        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        # ``is_pyses`` is decided by ``hasattr(dycore, "colmap")``; the stub
        # takes the pySES branch without a real CAM-SE build (the pySES forcing
        # loader is mocked, so its column internals are never read).
        return types.SimpleNamespace(colmap=object()), get_speedy_coords(
            layers=8, spectral_truncation=31)

    def test_pyses_oxidants_year_pattern_expands_before_loader(self):
        """Expand + open pySES oxidants ``{year}`` together before the loader (F1).

        The pySES branch previously handed ``oxidants_file`` straight through
        ``_resolve_data_path``, bypassing ``_expand_years`` and the uniform
        time-axis check — a documented transient run
        (``oxidants_file=.../{year}.nc`` + ``forcing.years``) would fetch a
        literal-brace path. Assert the expanded yearly list (via the SAME shared
        ``_resolve_oxidant_paths`` the spectral path uses) reaches the pySES
        forcing loader.
        """
        import xarray as xr

        from jcm import runners
        from jcm.forcing import ForcingData
        from jcm.runners import build_forcing
        dycore, coords = self._pyses_dycore_and_coords()
        cfg = self._pyses_cfg(
            oxidants_file="hf://bundles/t42_l8/oxidants_{year}.nc",
            years=[2000, 2001])
        base = ForcingData.zeros(nodal_shape=(1, 4))
        seen = {}

        def _capture(_forcing_file, _dycore, **kw):
            seen.update(kw)
            return base

        # The >1-file set is opened per-member to classify its time axis
        # (uniform-mixture guard); both yearly files are transient (datetime).
        def _open_dt(_path, **_kw):
            return xr.Dataset(coords={
                "time": np.array(["2000-06-15"], dtype="datetime64[ns]")})

        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=lambda p: p), \
                mock.patch("xarray.open_dataset", side_effect=_open_dt), \
                mock.patch("jcm.dycore.pyses.forcing.build_forcing",
                           side_effect=_capture):
            build_forcing(cfg, coords, dycore=dycore)
        self.assertEqual(
            seen["oxidants_file"],
            ["hf://bundles/t42_l8/oxidants_2000.nc",
             "hf://bundles/t42_l8/oxidants_2001.nc"])

    def test_pyses_emissions_year_pattern_expands_before_loader(self):
        """Expand pySES emissions ``{year}`` before the loader (same class as F1).

        The emissions input had the same bypass; a scalar ``{year}`` pattern must
        expand to that one product's yearly files (which ``attach_jam_forcing``
        opens by coords) instead of a literal-brace path.
        """
        from jcm import runners
        from jcm.forcing import ForcingData
        from jcm.runners import build_forcing
        dycore, coords = self._pyses_dycore_and_coords()
        cfg = self._pyses_cfg(
            emissions_file="hf://bundles/t42/emis_{year}.nc",
            years=[2000, 2001])
        base = ForcingData.zeros(nodal_shape=(1, 4))
        seen = {}

        def _capture(_forcing_file, _dycore, **kw):
            seen.update(kw)
            return base

        with mock.patch.object(runners, "_resolve_data_path",
                               side_effect=lambda p: p), \
                mock.patch("jcm.dycore.pyses.forcing.build_forcing",
                           side_effect=_capture):
            build_forcing(cfg, coords, dycore=dycore)
        self.assertEqual(
            seen["emissions_file"],
            ["hf://bundles/t42/emis_2000.nc", "hf://bundles/t42/emis_2001.nc"])

    def test_pyses_transient_ozone_rejected_clearly(self):
        """Transient ``{year}`` ozone is rejected with a clear message on pySES.

        Unlike oxidants/emissions, transient ozone is genuinely unsupported on
        the pySES backend (the column ozone climatology is a 12-month WRAP_YEAR
        field). A ``{year}`` pattern must raise the documented limitation up
        front, not reach ``xr.open_dataset`` as a literal-brace file-not-found.
        """
        from jcm.runners import build_forcing
        dycore, coords = self._pyses_dycore_and_coords()
        cfg = self._pyses_cfg(
            ozone_file="hf://bundles/t42_l8/ozone_{year}.nc", years=[2000])
        with self.assertRaisesRegex(
                ValueError, "transient ozone is not supported"):
            build_forcing(cfg, coords, dycore=dycore)

    def test_pyses_transient_surface_file_rejected_clearly(self):
        """A ``{year}`` surface ``forcing.file`` is rejected clearly on pySES.

        Matrix audit (round 8): the SPECTRAL path year-expands the surface
        ``file`` (that IS the transient input for forcing=amip/era5), so a user
        reasonably expects ``forcing=era5`` to work. On pySES the column reader
        opens a single 12-month climatology, so a ``{year}`` pattern must raise
        the documented limitation up front — mirroring the ozone guard — rather
        than reach ``_resolve_data_path`` as a confusing hf:// 404.
        """
        from jcm.runners import build_forcing
        dycore, coords = self._pyses_dycore_and_coords()
        cfg = self._pyses_cfg(
            file="hf://bundles/t63/forcing_era5/{year}.nc", years=[2000, 2001])
        with self.assertRaisesRegex(
                ValueError, "transient surface forcing is not supported"):
            build_forcing(cfg, coords, dycore=dycore)
