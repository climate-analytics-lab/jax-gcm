"""Tests for ``run.output_keep`` / ``run.output_drop`` (:mod:`jcm.output_selection`)."""

import inspect
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir

from jcm import output_selection as osel
from jcm.output_selection import OutputSelection

CONFIG_DIR = str(Path(__file__).parent / "config")


def _compose(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=list(overrides))


def _speedy_model():
    from jcm.model import Model
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    return Model(coords=get_speedy_coords(), time_step=30)


_RUN = dict(save_interval=1 / 24., total_time=2 / 24.)


class TestOutputSelection(unittest.TestCase):
    def test_default_selects_everything(self):
        sel = OutputSelection()
        self.assertTrue(sel.is_default)
        self.assertTrue(sel.selected("anything"))
        ds = xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])})
        self.assertIs(sel.filter_dataset(ds), ds)

    def test_drop_is_applied_after_keep(self):
        sel = OutputSelection.from_args(keep=["a", "b"], drop=["b"])
        self.assertTrue(sel.selected("a"))
        self.assertFalse(sel.selected("b"))
        self.assertFalse(sel.selected("c"))

    def test_a_single_name_is_one_name_not_its_characters(self):
        self.assertEqual(OutputSelection.from_args(keep="temperature").keep,
                         ("temperature",))

    def test_empty_keep_is_refused(self):
        with self.assertRaisesRegex(ValueError, "empty list"):
            OutputSelection.from_args(keep=[])

    def test_time_bounds_is_always_written(self):
        sel = OutputSelection.from_args(keep=["a"], drop=["time_bounds"])
        self.assertTrue(sel.selected("time_bounds"))

    def test_filter_dataset_keeps_coordinates(self):
        ds = xr.Dataset(
            {"a": ("x", [1.0]), "b": ("x", [2.0]),
             "time_bounds": ("x", [3.0])},
            coords={"x": [0]})
        out = OutputSelection.from_args(keep=["a"]).filter_dataset(ds)
        self.assertEqual(set(out.data_vars), {"a", "time_bounds"})
        self.assertIn("x", out.coords)

    def test_unknown_name_lists_the_available_ones(self):
        sel = OutputSelection.from_args(keep=["temprature"], drop=["nope"])
        with self.assertRaises(ValueError) as ctx:
            sel.validate({"temperature", "u_wind"})
        msg = str(ctx.exception)
        self.assertIn("output_keep: 'temprature' (did you mean temperature?)", msg)
        self.assertIn("output_drop: 'nope'", msg)
        self.assertIn("Available: temperature, u_wind", msg)

    def test_requiring_extends_keep_and_cancels_drop(self):
        sel = OutputSelection.from_args(keep=["a"], drop=["g"]).requiring(["g", "h"])
        self.assertEqual(sel.keep, ("a", "g", "h"))
        self.assertEqual(sel.drop, ())
        everything = OutputSelection.from_args(drop=["g"]).requiring(["g"])
        self.assertIsNone(everything.keep)
        self.assertTrue(everything.is_default)

    def test_selection_is_hashable_for_the_jit_static_argument(self):
        self.assertEqual(hash(OutputSelection.from_args(keep=["a"])),
                         hash(OutputSelection.from_args(keep=["a"])))


class TestSelectDynamics(unittest.TestCase):
    def _state(self):
        from jcm.physics_interface import PhysicsState

        shape = (3, 4, 5)
        return PhysicsState.ones(
            shape, tracers={"qc": jnp.ones(shape), "so2": jnp.ones(shape)})

    def test_dropped_fields_become_none_and_tracers_are_removed(self):
        sel = OutputSelection.from_args(keep=["temperature", "qc"])
        out = osel.select_dynamics(self._state(), sel)
        self.assertIsNotNone(out.temperature)
        self.assertIsNone(out.u_wind)
        self.assertIsNone(out.normalized_surface_pressure)
        self.assertEqual(set(out.tracers), {"qc"})
        # None carries no leaves: nothing is accumulated or stacked for them.
        self.assertEqual(len(jax.tree_util.tree_leaves(out)), 2)

    def test_one_three_d_prognostic_stays_as_the_shape_reference(self):
        sel = OutputSelection.from_args(keep=["normalized_surface_pressure"])
        out = osel.select_dynamics(self._state(), sel)
        self.assertEqual(osel.reference_field(out).shape, (3, 4, 5))
        self.assertIsNone(out.u_wind)

    def test_default_is_the_same_object(self):
        state = self._state()
        self.assertIs(osel.select_dynamics(state, OutputSelection()), state)


class TestPhysicsPruner(unittest.TestCase):
    """The plan is made from names the physics itself produces."""

    @staticmethod
    def _physics_and_template():
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.speedy.physics_data import SpeedyRadiationFluxes  # noqa: F401

        model = _speedy_model()
        template = model.physics.get_empty_data(model.coords)
        return model.physics, template, tuple(model.coords.nodal_shape), ComposablePhysics

    def test_struct_fields_dict_entries_and_arrays_are_pruned(self):
        physics, template, nodal, _ = self._physics_and_template()
        keep = ["radiation.toa_sw_up", "pressure_full",
                "surface_exchange.tile_fraction.0"]
        sel = OutputSelection.from_args(keep=keep)
        pruned = osel.make_physics_pruner(physics, sel, template, nodal)(template)
        # A typed struct keeps only its selected field (the others are None).
        rad = pruned["radiation"]
        self.assertIsNotNone(rad.toa_sw_up)
        self.assertIsNone(rad.toa_lw_up)
        # A wholly unselected entry is gone, plumbing included.
        self.assertNotIn("_surface_flux", pruned)
        self.assertNotIn("_prev_step", pruned)
        self.assertIn("pressure_full", pruned)
        self.assertNotIn("relative_humidity", pruned)
        # What the Dataset would carry is the selection, plus the sibling
        # channel of the one array: channels of an array are expanded after
        # it is stacked, so ``to_xarray`` removes the unselected one.
        stacked = jax.tree_util.tree_map(lambda x: x[None], pruned)
        out = physics.data_struct_to_dict(stacked, nodal)
        self.assertEqual(
            set(out), set(keep) | {"surface_exchange.tile_fraction.1"})

    def test_pruning_leaves_far_fewer_leaves_to_accumulate(self):
        physics, template, nodal, _ = self._physics_and_template()
        sel = OutputSelection.from_args(keep=["radiation.toa_sw_up"])
        pruned = osel.make_physics_pruner(physics, sel, template, nodal)(template)
        self.assertEqual(len(jax.tree_util.tree_leaves(pruned)), 1)
        self.assertGreater(len(jax.tree_util.tree_leaves(template)), 50)

    def test_default_returns_the_input_unchanged(self):
        physics, template, nodal, _ = self._physics_and_template()
        prune = osel.make_physics_pruner(physics, OutputSelection(), template, nodal)
        self.assertIs(prune(template), template)

    def test_the_available_names_are_the_dataset_names(self):
        model = _speedy_model()
        names = model.output_variable_names()
        ds = model.run(**_RUN, output_averages=True).to_xarray()
        # Integer/boolean diagnostics are omitted from interval means (see
        # ``ModelPredictions.to_xarray``), so they are named but not written.
        self.assertLessEqual(set(ds.data_vars) - {"time_bounds"}, names)
        self.assertGreater(len(set(ds.data_vars)), 0.9 * len(names))


class TestModelRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full_model = _speedy_model()
        cls.full = cls.full_model.run(**_RUN, output_averages=True)
        cls.full_ds = cls.full.to_xarray()

    def test_default_is_deterministic_and_untouched(self):
        again = _speedy_model().run(**_RUN, output_averages=True).to_xarray()
        xr.testing.assert_identical(again, self.full_ds)
        # Nothing is pruned: every prognostic and the carry plumbing ride the
        # saved trajectory exactly as without the feature.
        self.assertIsNotNone(self.full.dynamics.u_wind)
        self.assertIn("_prev_step", self.full.physics)
        self.assertTrue(self.full._output_selection.is_default)

    def test_keep_writes_only_the_listed_variables_with_the_same_values(self):
        keep = ["temperature", "radiation.toa_sw_up",
                "surface_exchange.tile_fraction.0", "pressure_full"]
        preds = _speedy_model().run(
            **_RUN, output_averages=True, output_keep=keep)
        ds = preds.to_xarray()
        self.assertEqual(set(ds.data_vars), set(keep) | {"time_bounds"})
        for name in keep:
            np.testing.assert_allclose(
                ds[name].values, self.full_ds[name].values, rtol=1e-5, atol=1e-6)
        # Pruned before accumulation: the saved trajectory itself is small.
        self.assertIsNone(preds.dynamics.u_wind)
        self.assertEqual(set(preds.physics), {"pressure_full", "radiation",
                                              "surface_exchange"})
        self.assertIsNone(preds.physics["radiation"].toa_lw_up)
        full_leaves = sum(
            x.size for x in jax.tree_util.tree_leaves(self.full._predictions))
        kept_leaves = sum(
            x.size for x in jax.tree_util.tree_leaves(preds._predictions))
        self.assertLess(kept_leaves, full_leaves / 10)

    def test_drop_removes_prognostic_and_diagnostic(self):
        ds = _speedy_model().run(
            **_RUN, output_averages=True,
            output_drop=["u_wind", "relative_humidity"]).to_xarray()
        self.assertEqual(set(self.full_ds.data_vars) - set(ds.data_vars),
                         {"u_wind", "relative_humidity"})

    def test_keep_then_drop(self):
        ds = _speedy_model().run(
            **_RUN, output_averages=True,
            output_keep=["temperature", "u_wind"],
            output_drop=["u_wind"]).to_xarray()
        self.assertEqual(set(ds.data_vars), {"temperature", "time_bounds"})

    def test_dropping_every_3d_prognostic_still_gives_a_dataset(self):
        ds = _speedy_model().run(
            **_RUN, output_averages=True,
            output_keep=["radiation.toa_sw_up"]).to_xarray()
        self.assertEqual(set(ds.data_vars), {"radiation.toa_sw_up", "time_bounds"})
        self.assertEqual(ds["radiation.toa_sw_up"].sizes["time"], 2)

    def test_instantaneous_output_is_selected_too(self):
        preds = _speedy_model().run(
            **_RUN, output_averages=False,
            output_keep=["temperature", "radiation.toa_sw_up"])
        ds = preds.to_xarray()
        self.assertEqual(set(ds.data_vars) - {"time_bounds"},
                         {"temperature", "radiation.toa_sw_up"})
        self.assertNotIn("_prev_step", preds.physics)

    def test_unknown_name_fails_before_any_integration(self):
        model = _speedy_model()
        with mock.patch.object(type(model), "_run_from_state") as integrate:
            with self.assertRaisesRegex(ValueError, "temprature"):
                model.run(**_RUN, output_averages=True,
                          output_keep=["temprature"])
            with self.assertRaisesRegex(ValueError, "no_such_field"):
                model.run(**_RUN, output_averages=True,
                          output_drop=["no_such_field"])
        integrate.assert_not_called()

    def test_resume_and_run_from_state_take_the_selection(self):
        model = _speedy_model()
        model.run(**_RUN, output_averages=True, output_keep=["temperature"])
        ds = model.resume(**_RUN, output_averages=True,
                          output_keep=["temperature"]).to_xarray()
        self.assertEqual(set(ds.data_vars), {"temperature", "time_bounds"})
        state, _ = _speedy_model().run_from_state(
            self.full_model.initial_state(), None or _default_forcing(),
            **_RUN, output_averages=True, output_keep=["specific_humidity"])
        del state

    def test_the_simulation_and_its_carry_do_not_depend_on_the_selection(self):
        """Outputs are a saved copy: the state and carry a restart uses agree.

        The selection changes the compiled program (dead outputs go), so the
        two runs agree to rounding rather than bit for bit; what must hold is
        that the carry has the same structure and values to that rounding.
        """
        sel_model = _speedy_model()
        sel_model.run(**_RUN, output_averages=True, output_keep=["temperature"])
        a = jax.tree_util.tree_leaves(
            (self.full_model.dycore_state, self.full_model.physics_carry))
        b = jax.tree_util.tree_leaves(
            (sel_model.dycore_state, sel_model.physics_carry))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(np.shape(x), np.shape(y))
            np.testing.assert_allclose(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                rtol=1e-3, atol=1e-3)

    def test_a_selected_run_resumes_bit_identically_to_itself(self):
        keep = ["temperature", "radiation.toa_sw_up"]
        whole = _speedy_model()
        w = whole.run(save_interval=1 / 24., total_time=2 / 24.,
                      output_averages=True, output_keep=keep).to_xarray()
        split = _speedy_model()
        first = split.run(save_interval=1 / 24., total_time=1 / 24.,
                          output_averages=True, output_keep=keep).to_xarray()
        second = split.resume(save_interval=1 / 24., total_time=1 / 24.,
                              output_averages=True, output_keep=keep).to_xarray()
        # Two interval means from one 2 h run, or one each from a split run.
        np.testing.assert_allclose(
            np.concatenate([first.temperature.values,
                            second.temperature.values]),
            w.temperature.values, rtol=1e-5)
        for x, y in zip(
                jax.tree_util.tree_leaves(
                    (whole.dycore_state, whole.physics_carry)),
                jax.tree_util.tree_leaves(
                    (split.dycore_state, split.physics_carry))):
            np.testing.assert_allclose(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                rtol=1e-4, atol=1e-4)

    def test_snapshot_stream_is_independent_of_output_drop(self):
        preds = _speedy_model().run(
            save_interval=1 / 24., total_time=1 / 24., output_averages=True,
            snapshot_interval=0.5 / 24.,
            snapshot_variables=["radiation.toa_sw_up"],
            output_drop=["radiation.toa_sw_up"])
        snap = preds.snapshot_dataset()
        self.assertIn("radiation_toa_sw_up", snap.data_vars)
        self.assertGreater(float(np.abs(snap["radiation_toa_sw_up"]).max()), 0.0)
        self.assertNotIn("radiation.toa_sw_up", preds.to_xarray().data_vars)


def _default_forcing():
    from jcm.forcing import default_forcing
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    return default_forcing(get_speedy_coords().horizontal)


@pytest.mark.slow
class TestEchamSelection(unittest.TestCase):
    """The ECHAM package: typed sub-structs, dict entries, tracers, renames."""

    def test_selection_matches_the_full_output(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)

        def make():
            return Model(
                coords=coords,
                physics=idealized_echam_physics(checkpoint_terms=False),
                time_step=12.0)

        kw = dict(save_interval=1 / 24., total_time=1 / 24., output_averages=True)
        model = make()
        names = model.output_variable_names()
        full = model.run(**kw).to_xarray()
        self.assertLessEqual(set(full.data_vars) - {"time_bounds"}, names)
        keep = [n for n in ("temperature", "qc", "radiation.toa_sw_up") if n in names]
        sel = make().run(output_keep=keep, **kw).to_xarray()
        self.assertEqual(set(sel.data_vars), set(keep) | {"time_bounds"})
        for n in keep:
            np.testing.assert_allclose(
                sel[n].values, full[n].values, rtol=1e-4, atol=1e-6)


class TestRunnerReadFields(unittest.TestCase):
    """The names the chunk loop reads back, kept in step with its readers."""

    def test_covers_every_variable_check_health_reads_by_name(self):
        from jcm import diagnostics

        source = inspect.getsource(diagnostics.check_health)
        body = source.split('"""', 2)[2]
        read = set(re.findall(r'ds\["([^"]+)"\]', body))
        read |= set(re.findall(r'"([a-z_0-9.]+)" (?:not )?in ds', body))
        read |= set(re.findall(r'^\s+"(radiation\.[a-z_]+)"', body, re.M))
        read |= set(re.findall(r'"(convection\.[a-z_]+)"', body))
        self.assertTrue(read, "the scan found nothing; the pattern is stale")
        covered = (set(diagnostics._HEALTH_VARIABLES)
                   | set(diagnostics._HEALTH_PROGNOSTICS))
        self.assertEqual(read - covered, set())

    def test_picks_gate_surface_temperature_and_budget_names(self):
        from jcm.diagnostics import runner_read_fields

        names = {"temperature", "wind_10m_model_level", "radiation.toa_sw_up",
                 "surface_flux.surface_temperature", "surface_temperature_tendency",
                 "budget_dyn_so4", "emi_so4", "ordinary", "wet_so4"}
        self.assertEqual(
            runner_read_fields(names),
            {"wind_10m_model_level", "radiation.toa_sw_up",
             "surface_flux.surface_temperature", "budget_dyn_so4", "emi_so4",
             "wet_so4"})
        self.assertIn("temperature",
                      runner_read_fields(names, dynamics_needed=True))


class TestChunkedRunner(unittest.TestCase):
    """The chunk loop: gate visibility, written files, restarts."""

    _BASE = ["physics=speedy", "grid=speedy_t31_l8", "run.time_step=30",
             "run.total_time=1", "run.save_interval=0.5", "run.chunk_days=0.5",
             "run.output_averages=true", "terrain=aquaplanet"]

    def _run(self, tmpdir, extra=()):
        from jcm.runners import run

        return run(_compose(
            self._BASE + [f"run.output_prefix={tmpdir}/c"] + list(extra)))

    def test_gate_still_sees_what_it_reads_and_files_hold_only_the_selection(self):
        seen = {}

        def spy(ds, *args, **kwargs):
            seen["names"] = set(ds.data_vars)
            seen["boundary"] = kwargs.get("boundary_state") is not None
            return True, {"chunk": 0, "elapsed_days": 0.5, "ok": True,
                          "reasons": [], "n_nan_vars": 0, "n_total_vars": 0}

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("jcm.diagnostics.check_health", side_effect=spy), \
                mock.patch("jcm.diagnostics.print_report"):
            self._run(tmpdir, ["run.output_keep=[temperature]"])
            files = sorted(Path(tmpdir).glob("c_day*.nc"))
            self.assertTrue(files)
            written = xr.open_dataset(files[0])
            self.assertEqual(set(written.data_vars), {"temperature", "time_bounds"})
        # The gate got the fatal-criterion inputs and the boundary state even
        # though neither was selected.
        self.assertTrue(seen["boundary"])
        self.assertTrue({"radiation.toa_lw_up", "radiation.toa_sw_up"}
                        <= seen["names"])
        self.assertGreater(len(seen["names"]), 2)

    def test_drop_only_keeps_the_gate_fields_computed_not_written(self):
        seen = {}

        def spy(ds, *args, **kwargs):
            seen["names"] = set(ds.data_vars)
            return True, {"chunk": 0, "elapsed_days": 0.5, "ok": True,
                          "reasons": [], "n_nan_vars": 0, "n_total_vars": 0}

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("jcm.diagnostics.check_health", side_effect=spy), \
                mock.patch("jcm.diagnostics.print_report"):
            self._run(tmpdir, ["run.output_drop=[radiation.toa_sw_up,u_wind]"])
            written = xr.open_dataset(sorted(Path(tmpdir).glob("c_day*.nc"))[0])
        self.assertIn("radiation.toa_sw_up", seen["names"])
        self.assertNotIn("radiation.toa_sw_up", written.data_vars)
        self.assertNotIn("u_wind", written.data_vars)
        self.assertIn("temperature", written.data_vars)

    def test_unknown_name_fails_before_the_first_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("jcm.model.Model.run") as run_chunk:
                with self.assertRaisesRegex(ValueError, "tmperature"):
                    self._run(tmpdir, ["run.output_keep=[tmperature]"])
            run_chunk.assert_not_called()

    def test_monthly_stream_carries_only_the_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._run(tmpdir, ["run.output_keep=[temperature]",
                               "run.monthly_means=true",
                               "run.start_time=2000-01-01",
                               "run.total_time=1", "run.save_chunks=false"])
            months = sorted(Path(tmpdir).glob("c_monthly_*.nc"))
            self.assertTrue(months)
            ds = xr.open_dataset(months[0])
            self.assertEqual(
                {v for v in ds.data_vars
                 if "bounds" not in v and "coverage" not in v},
                {"temperature"})

    def test_checkpoint_holds_state_not_outputs(self):
        import flax.serialization

        def checkpoint_keys(extra):
            with tempfile.TemporaryDirectory() as tmpdir:
                self._run(tmpdir, [f"run.checkpoint_path={tmpdir}/r.ckpt",
                                   *extra])
                raw = flax.serialization.msgpack_restore(
                    Path(f"{tmpdir}/r.ckpt").read_bytes())
                return _tree_keys(raw)

        self.assertEqual(checkpoint_keys([]),
                         checkpoint_keys(["run.output_keep=[temperature]"]))

    def test_a_selected_run_resumes_from_its_checkpoint(self):
        """A restart under a selection continues exactly like an unbroken run."""
        with tempfile.TemporaryDirectory() as split_dir, \
                tempfile.TemporaryDirectory() as whole_dir:
            sel = ["run.output_keep=[temperature]"]
            from jcm.runners import run

            def go(tmpdir, total, ckpt):
                return run(_compose(
                    self._BASE + sel + [f"run.output_prefix={tmpdir}/c",
                                        f"run.total_time={total}",
                                        f"run.checkpoint_path={ckpt}"]))

            go(split_dir, 0.5, f"{split_dir}/r.ckpt")
            go(split_dir, 1, f"{split_dir}/r.ckpt")
            go(whole_dir, 1, f"{whole_dir}/r.ckpt")
            a = xr.open_dataset(sorted(Path(split_dir).glob("c_day1*.nc"))[0])
            b = xr.open_dataset(sorted(Path(whole_dir).glob("c_day1*.nc"))[0])
            np.testing.assert_array_equal(a.temperature.values,
                                          b.temperature.values)

    def test_pressure_levels_keeps_pressure_full_implicitly(self):
        from jcm.runners import _chunk_output_selections

        cfg = _compose(["run.output_keep=[temperature]",
                        "run.pressure_levels=[500, 850]"])
        model = types.SimpleNamespace(
            dycore=object(),
            output_variable_names=lambda: {"temperature", "pressure_full"})
        written, computed = _chunk_output_selections(cfg, model, [500, 850])
        self.assertTrue(written.selected("pressure_full"))
        self.assertTrue(computed.selected("pressure_full"))
        bad = _compose(["run.output_drop=[pressure_full]",
                        "run.pressure_levels=[500]"])
        with self.assertRaisesRegex(ValueError, "pressure_levels"):
            _chunk_output_selections(bad, model, [500])

    def test_default_never_asks_the_model_for_names(self):
        from jcm.runners import _chunk_output_selections

        model = mock.Mock(side_effect=AssertionError("no model access"))
        written, computed = _chunk_output_selections(_compose(), model, None)
        self.assertTrue(written.is_default and computed.is_default)

    def test_diagnostic_modes_refuse_the_selection(self):
        from jcm.runners import _reject_full_mode_only_knobs

        cfg = _compose(["run.mode=prescribed", "run.output_keep=[temperature]"])
        with self.assertRaisesRegex(ValueError, "run.mode=full"):
            _reject_full_mode_only_knobs(cfg)


class TestDoors(unittest.TestCase):
    def test_hydra_keys_reach_model_run_kwargs(self):
        from jcm import configurations

        cfg = _compose(["run.output_keep=[temperature]",
                        "run.output_drop=[u_wind]"])
        kwargs = configurations._run_kwargs(cfg, mock.Mock())
        self.assertEqual(kwargs["output_keep"], ("temperature",))
        self.assertEqual(kwargs["output_drop"], ("u_wind",))
        default = configurations._run_kwargs(_compose(), mock.Mock())
        self.assertIsNone(default["output_keep"])
        self.assertEqual(default["output_drop"], ())

    def test_cli_plain_run_writes_the_selection(self):
        from jcm.runners import run, save_predictions

        cfg = _compose(["physics=speedy", "grid=speedy_t31_l8",
                        "run.time_step=30", "run.total_time=0.5",
                        "run.save_interval=0.25", "run.output_averages=true",
                        "terrain=aquaplanet",
                        "run.output_keep=[temperature]"])
        preds = run(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_predictions(preds, Path(tmpdir) / "out.nc")
            ds = xr.open_dataset(Path(tmpdir) / "out.nc")
        self.assertEqual(set(ds.data_vars), {"temperature", "time_bounds"})


def _tree_keys(tree, prefix=""):
    if isinstance(tree, dict):
        out = set()
        for key, value in tree.items():
            out |= _tree_keys(value, f"{prefix}/{key}")
        return out
    return {prefix}
