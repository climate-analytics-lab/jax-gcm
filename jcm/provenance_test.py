"""Tests for run provenance capture (#591, #732)."""
import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

from jcm import provenance


class ProbeCodeTest(unittest.TestCase):
    def test_jcm_probed_with_git_state(self):
        code = provenance.probe_code()
        self.assertIn("jcm", code)
        entry = code["jcm"]
        self.assertTrue(Path(entry["path"]).is_dir())
        # This test runs from a git worktree, so the git triple must be
        # present and well-formed.
        self.assertRegex(entry["sha"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(entry["dirty"], bool)
        if entry["dirty"]:
            self.assertRegex(entry["dirty_diff_sha"], r"^[0-9a-f]{12}$")

    def test_version_libs_recorded(self):
        code = provenance.probe_code()
        self.assertIn("jax", code)
        self.assertIn("version", code["jax"])


class ProbeEnvironmentTest(unittest.TestCase):
    def test_environment_fields(self):
        env = provenance.probe_environment()
        self.assertIn(env["jax_enable_x64"], (True, False))
        self.assertGreaterEqual(env["device_count"], 1)
        self.assertTrue(env["hostname"])
        self.assertIn("platform", env)


class DescribeInputTest(unittest.TestCase):
    def test_size_mtime_default_and_hash_opt_in(self):
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"hello provenance")
            f.flush()
            d = provenance.describe_input(f.name)
            self.assertEqual(d["size"], 16)
            self.assertIn("mtime", d)
            self.assertNotIn("sha256", d)
            try:
                os.environ["JCM_HASH_INPUTS"] = "1"
                d = provenance.describe_input(f.name)
            finally:
                os.environ.pop("JCM_HASH_INPUTS", None)
            self.assertEqual(
                d["sha256"],
                hashlib.sha256(b"hello provenance").hexdigest())

    def test_missing_file_flagged(self):
        d = provenance.describe_input("/nonexistent/never.nc")
        self.assertTrue(d["missing"])


class RegistryAndHashTest(unittest.TestCase):
    def setUp(self):
        provenance.start_run()

    def test_record_input_dedup_and_missing_skipped(self):
        provenance.record_input("/nonexistent/never.nc")
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"x")
            f.flush()
            provenance.record_input(f.name)
            provenance.record_input("hf://bundles/x.nc", f.name)
            prov = provenance.collect()
        self.assertEqual(len(prov["inputs"]), 1)
        entry = next(iter(prov["inputs"].values()))
        self.assertEqual(entry["requested"], "hf://bundles/x.nc")

    def test_run_hash_stable_and_input_sensitive(self):
        h0 = provenance.collect()["run_hash"]
        self.assertEqual(provenance.collect()["run_hash"], h0)
        self.assertRegex(h0, r"^[0-9a-f]{12}$")
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"data")
            f.flush()
            provenance.record_input(f.name)
            self.assertNotEqual(provenance.collect()["run_hash"], h0)

    def test_facts_reach_attrs_and_summary(self):
        provenance.record_fact("ozone_source", "prescribed:/x/ozone.nc")
        attrs = provenance.attrs()
        self.assertEqual(attrs["jcm_prov_ozone_source"],
                         "prescribed:/x/ozone.nc")
        self.assertIn("ozone=prescribed:/x/ozone.nc", provenance.summary())

    def test_attrs_are_netcdf_safe_strings(self):
        attrs = provenance.attrs()
        for key, value in attrs.items():
            self.assertTrue(key.startswith("jcm_prov_"), key)
            self.assertIsInstance(value, str)
        # created (run start) and written (output time) bracket the run.
        self.assertIn("jcm_prov_created", attrs)
        self.assertIn("jcm_prov_written", attrs)
        # nested parts round-trip through JSON
        self.assertIn("jcm", json.loads(attrs["jcm_prov_code"]))

    def test_config_hashed_into_attrs_and_sidecar(self):
        from omegaconf import OmegaConf
        provenance.start_run(OmegaConf.create({"run": {"total_time": 10}}))
        attrs = provenance.attrs()
        self.assertRegex(attrs["jcm_prov_config_sha"], r"^[0-9a-f]{12}$")
        with tempfile.TemporaryDirectory() as d:
            sidecar = provenance.write_sidecar(Path(d) / "out.nc")
            self.assertEqual(sidecar.name, "out.nc.provenance.json")
            full = json.loads(sidecar.read_text())
            self.assertIn("total_time: 10", full["config_yaml"])
            self.assertEqual(full["run_hash"], attrs["jcm_prov_run_hash"])


class SavePredictionsAttachesTest(unittest.TestCase):
    def test_netcdf_attrs_and_sidecar_written(self):
        import xarray as xr

        from jcm.runners import save_predictions

        class _Preds:
            def to_xarray(self):
                return xr.Dataset({"t": ("x", [1.0])})

        provenance.start_run()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run.nc"
            save_predictions(_Preds(), out)
            with xr.open_dataset(out) as ds:
                self.assertIn("jcm_prov_run_hash", ds.attrs)
                self.assertIn("jcm_prov_code", ds.attrs)
            sidecar = Path(d) / "run.nc.provenance.json"
            self.assertTrue(sidecar.exists())
            self.assertEqual(json.loads(sidecar.read_text())["run_hash"],
                             ds.attrs["jcm_prov_run_hash"])


class ResolveDataPathRecordsTest(unittest.TestCase):
    def test_resolver_feeds_the_registry(self):
        from jcm.runners import _resolve_data_path
        provenance.start_run()
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            f.write(b"terrain")
            f.flush()
            out = _resolve_data_path(f.name)
            self.assertEqual(out, f.name)
            self.assertIn(f.name, provenance.collect()["inputs"])
        # non-file strings pass through without registering
        provenance.start_run()
        _resolve_data_path("auto")
        self.assertEqual(provenance.collect()["inputs"], {})


@dataclasses.dataclass
class _FakeParamBlock:
    """Stands in for a scheme's parameter struct: named fields."""

    order: int = 2


class DescribeLeafTest(unittest.TestCase):
    def test_scalars_pass_through(self):
        self.assertIsNone(provenance._describe_leaf(None))
        self.assertIs(provenance._describe_leaf(True), True)
        self.assertEqual(provenance._describe_leaf("cubic"), "cubic")

    def test_array_by_value_under_the_cap_and_hashed_over_it(self):
        import jax.numpy as jnp

        small = provenance._describe_leaf(jnp.arange(5.0))
        self.assertEqual(small["values"], [0.0, 1.0, 2.0, 3.0, 4.0])
        # Shape and dtype ride along with the values, so a (2, 3) is not
        # confusable with a (3, 2), nor float32 with float64.
        self.assertEqual(small["shape"], [5])
        self.assertEqual(small["dtype"], "float32")
        n = provenance._PARAM_ARRAY_MAX_ELEMS + 1
        big = provenance._describe_leaf(jnp.arange(float(n)))
        self.assertEqual(big["shape"], [n])
        self.assertEqual(big["dtype"], "float32")
        self.assertNotIn("values", big)
        # The hash has to separate one weight set from another, or a
        # summarized parameter is no record at all.
        other = provenance._describe_leaf(jnp.arange(float(n)) + 1.0)
        self.assertNotEqual(big["sha256"], other["sha256"])

    def test_scalar_keeps_the_value_the_model_held(self):
        import jax.numpy as jnp

        # float32 0.1 is not 0.1; record what ran, not a prettier rounding.
        self.assertEqual(provenance._describe_leaf(jnp.float32(0.1)),
                         0.10000000149011612)

    def test_unrepresentable_values_are_described_not_raised(self):
        # A parameter block may hold a callable (a schedule, an
        # activation) or something numpy cannot convert. Provenance
        # describes it and moves on rather than failing the record.
        def relu(x):
            return x

        self.assertEqual(provenance._describe_leaf(relu), "<callable relu>")
        self.assertIn("object", provenance._describe_leaf(object()))

    def test_tracer_is_marked_not_forced(self):
        import jax
        import jax.numpy as jnp

        seen = []

        def f(x):
            seen.append(provenance._describe_leaf(x))
            return x * 2

        jax.jit(f)(jnp.float32(1.0))
        self.assertEqual(seen, ["<traced>"])


class DescribeValueTest(unittest.TestCase):
    def _walk(self, value):
        out = {}
        provenance._describe_value(value, "p", out)
        return out

    def test_short_scalar_sequence_is_one_value(self):
        # A per-level profile or a tracer-name tuple reads better as one
        # entry than as p.0, p.1, p.2 ...
        self.assertEqual(self._walk(["qc", "qi"]), {"p": ["qc", "qi"]})
        self.assertEqual(self._walk((1.0, 2.0)), {"p": [1.0, 2.0]})

    def test_long_scalar_sequence_is_hashed_not_just_counted(self):
        # A bare length made two AerocomDiagnostics terms with different
        # 100-element plev_pa tuples record identically, though those
        # pressures are interpolated to.
        n = provenance._PARAM_ARRAY_MAX_ELEMS + 1
        one = self._walk([float(i) for i in range(n)])["p"]
        two = self._walk([float(i) + 1 for i in range(n)])["p"]
        self.assertEqual(one["length"], n)
        self.assertRegex(one["sha256"], r"^[0-9a-f]{12}$")
        self.assertNotEqual(one["sha256"], two["sha256"])

    def test_container_of_structures_is_walked_by_index(self):
        walked = self._walk([_FakeParamBlock(), {"k": 2.0}])
        self.assertEqual(walked["p.0.order"], 2)
        self.assertEqual(walked["p.1.k"], 2.0)

    def test_namedtuple_fields_are_named(self):
        import collections

        pair = collections.namedtuple("pair", "lo hi")
        self.assertEqual(self._walk(pair(1.0, 2.0)),
                         {"p.lo": 1.0, "p.hi": 2.0})

    def test_runaway_nesting_is_truncated_not_followed(self):
        deep = value = {}
        for _ in range(provenance._PARAM_MAX_DEPTH + 3):
            nxt = {}
            value["down"] = nxt
            value = nxt
        walked = self._walk(deep)
        self.assertEqual(len(walked), 1)
        self.assertIn("truncated", next(iter(walked.values())))


class DescribePhysicsParamsTest(unittest.TestCase):
    """The gap #732 closes: values that live in code, not in the config."""

    def setUp(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        self.params = provenance.describe_params(speedy_physics())

    def test_scheme_defaults_are_recorded_not_just_overrides(self):
        from jcm.physics.speedy.speedy_terms import ConvectionParameters

        # Nothing overrode this, so it is exactly the case the composed
        # Hydra config cannot describe: the value is Parameters.default().
        key = "speedy_convection.params.entmax"
        self.assertIn(key, self.params)
        self.assertAlmostEqual(
            self.params[key],
            float(ConvectionParameters.default().entmax), places=12)

    def test_keys_locate_the_value_owner_first(self):
        # A recorded key must say where the value lives, unambiguously
        # and owner first: <term>.<variable>.<field>. Note this is NOT
        # always the Hydra override path — Hydra addresses the term's
        # constructor keyword, and SpeedyConvection takes
        # convection_params= but stores it as `params`.
        for key in self.params:
            self.assertRegex(key, r"^[A-Za-z0-9_#]+\.[A-Za-z0-9_]+")

    def test_variables_not_named_params_are_captured(self):
        # Keying on nnx.Param rather than a `.params` attribute is the
        # point: these two hang off differently-named variables and a
        # `.params`-only walk would silently drop them.
        self.assertIn("speedy_shortwave_radiation.sw_params.albcl",
                      self.params)
        self.assertIn("speedy_upward_longwave.mod_radcon_params.emisfc",
                      self.params)

    def test_non_differentiable_parameter_variables_are_captured(self):
        # A parameter block containing a bool cannot be an nnx.Param, so
        # the schemes hold those as plain Variables. Filtering on Param
        # dropped SPEEDY's whole surface-flux set even though changing
        # e.g. cdl changes the simulation.
        self.assertIn("speedy_surface_flux.surface_params.cdl", self.params)
        self.assertIn("speedy_surface_flux.surface_params.fwind0",
                      self.params)

    def test_held_suarez_parameters_are_recorded(self):
        # Every Held-Suarez tuning constant is an nnx.Variable, so under a
        # Param-only filter this physics recorded NOTHING at all.
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        params = provenance.describe_params(held_suarez_physics())
        for knob in ("kf", "ka", "ks", "dTy", "dThz", "sigma_b"):
            self.assertIn(f"held_suarez.{knob}", params)
        # ...while its grid caches, held as plain Variables right beside
        # those knobs with no naming convention to tell them apart, stay
        # out. This is why the plain-Variable filter is by shape.
        self.assertNotIn("held_suarez.sigma", params)
        self.assertNotIn("held_suarez.latitudes", params)

    def test_coordinate_caches_do_not_enter_the_record(self):
        # All eleven SPEEDY terms cache the SAME _speedy_coords. Taking
        # plain Variables wholesale put eleven identical copies of the
        # vertical grid in the record, 85% of a T31L8 run's bytes.
        for key in self.params:
            self.assertNotIn("_speedy_coords", key)

    def test_declared_params_keep_their_arrays(self):
        # The knob-shape filter applies to plain Variables only: an
        # nnx.Param is a declared parameter, so a tuned array on one must
        # survive in full. MACv2-SP's plume shapes are the shipped case.
        from jcm.physics.aerosol import Macv2SpAerosol

        class _One:
            terms = [Macv2SpAerosol()]

        params = provenance.describe_params(_One())
        theta = params["macv2_sp_aerosol.params.theta"]
        self.assertIn("values", theta)
        self.assertEqual(theta["shape"], [2, 9])

    def test_physics_without_variables_yields_no_record(self):
        # No composition and no state at all: nothing to say, and saying
        # nothing must not be an error.
        class _Bare:
            pass

        self.assertEqual(provenance.describe_params(_Bare()), {})

    def test_no_physics_is_not_an_error(self):
        self.assertEqual(provenance.describe_params(None), {})

    def test_two_instances_of_one_term_both_survive(self):
        # A composition may call one scheme twice (the double-radiation
        # A/B). Keying purely on term name would clobber the first.
        from jcm.physics.speedy.speedy_terms import SpeedyConvection

        class _Pair:
            terms = [SpeedyConvection(), SpeedyConvection()]

        params = provenance.describe_params(_Pair())
        self.assertIn("speedy_convection.params.entmax", params)
        self.assertIn("speedy_convection#1.params.entmax", params)


class ParamsAttrsTest(unittest.TestCase):
    def test_empty_record_stamps_nothing(self):
        self.assertEqual(provenance.params_attrs({}), {})
        self.assertEqual(provenance.params_attrs(None), {})

    def test_values_and_hash_are_netcdf_safe_strings(self):
        attrs = provenance.params_attrs({"a.b.c": 1.0})
        for key, value in attrs.items():
            self.assertTrue(key.startswith("jcm_prov_"), key)
            self.assertIsInstance(value, str)
        self.assertRegex(attrs["jcm_prov_params_sha"], r"^[0-9a-f]{12}$")
        self.assertEqual(
            json.loads(attrs["jcm_prov_params"])["a.b.c"], 1.0)

    def test_read_params_round_trips_and_tolerates_absence(self):
        small = {"t.params.entrpen": 1e-4}
        self.assertEqual(
            provenance.read_params(provenance.params_attrs(small)), small)
        self.assertEqual(provenance.read_params({}), {})

    def test_run_hash_separates_parameter_sweep_members(self):
        # Same code, same config, same inputs: without the parameters in
        # the hash every member of a sweep would share one run_hash.
        provenance.start_run()
        a = provenance.attrs({"t.params.entrpen": 1e-4})
        b = provenance.attrs({"t.params.entrpen": 4e-4})
        self.assertNotEqual(a["jcm_prov_run_hash"], b["jcm_prov_run_hash"])

    def test_sidecar_and_attrs_agree_on_the_run_hash(self):
        params = {"t.params.entrpen": 1e-4}
        provenance.start_run()
        attrs = provenance.attrs(params)
        with tempfile.TemporaryDirectory() as d:
            sidecar = provenance.write_sidecar(Path(d) / "out.nc", params)
            full = json.loads(sidecar.read_text())
        self.assertEqual(full["run_hash"], attrs["jcm_prov_run_hash"])
        self.assertEqual(full["params"], params)


class ModelPredictionsCaptureTest(unittest.TestCase):
    """Capture happens at the model-to-user handoff, once, eagerly."""

    def _physics(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        return speedy_physics()

    def _preds(self, physics):
        from jcm.predictions import ModelPredictions
        return ModelPredictions(None, None, physics)

    def test_parameters_recorded_on_the_predictions(self):
        preds = self._preds(self._physics())
        self.assertIn("speedy_convection.params.entmax", preds.params)

    def test_snapshot_does_not_follow_later_mutation(self):
        # Each record belongs to the predictions it was built with: a
        # later mutation must not reach back into an earlier record.
        # (Whether a mutation reaches the *computation* is a separate
        # question — see test_traced_record_wins_over_the_live_module.)
        import jax.numpy as jnp

        physics = self._physics()
        preds = self._preds(physics)
        before = preds.params["speedy_convection.params.entmax"]

        term = next(t for t in physics.terms if t.name == "speedy_convection")
        current = term.params.get_value()
        term.params.set_value(current.replace(entmax=jnp.array(0.25)))

        after = self._preds(physics).params[
            "speedy_convection.params.entmax"]
        self.assertNotAlmostEqual(before, after, places=6)
        self.assertEqual(
            preds.params["speedy_convection.params.entmax"],
            before)

    def test_traced_record_wins_over_the_live_module(self):
        # `self` is a static argument to
        # Model._run_from_state, so the parameters are constants inside
        # the compiled executable and an in-place change afterwards does
        # not reach the computation. Reading the live module at the
        # handoff therefore stamped a trajectory with values that never
        # ran — a confident, wrong record, which is worse than no record.
        import jax.numpy as jnp

        from jcm.predictions import ModelPredictions

        physics = self._physics()
        traced = provenance.describe_params(physics)
        term = next(t for t in physics.terms if t.name == "speedy_convection")
        term.params.set_value(
            term.params.get_value().replace(entmax=jnp.array(0.25)))

        with self.assertLogs("jcm.predictions", level="WARNING") as logged:
            preds = ModelPredictions(None, None, physics, params=traced)
        self.assertEqual(
            preds.params["speedy_convection.params.entmax"],
            traced["speedy_convection.params.entmax"])
        # ...and the divergence is surfaced rather than papered over: the
        # user's parameter change did nothing to the run, which is a
        # scientific error they need told about.
        self.assertIn("live_parameters_differ_from_compiled", preds.params)
        self.assertIn("does NOT affect the computation",
                      "".join(logged.output))

    def test_no_false_alarm_when_live_matches_compiled(self):
        from jcm.predictions import ModelPredictions

        physics = self._physics()
        traced = provenance.describe_params(physics)
        preds = ModelPredictions(None, None, physics, params=traced)
        self.assertEqual(preds.params, traced)
        self.assertNotIn("live_parameters_differ_from_compiled", preds.params)

    @pytest.mark.slow
    def test_each_executable_keeps_its_own_traced_record(self):
        """One static signature can own several compiled executables.

        Keying the store on the static arguments let a later trace
        overwrite an earlier executable's record, so re-running the first
        one reported the second's parameters. The record is
        keyed by a trace id the executable itself carries back, so a cache
        hit returns the id of the executable that actually ran.
        """
        import dataclasses

        import jax.numpy as jnp

        from jcm.forcing import default_forcing, make_time_series
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        key = "speedy_vertical_diffusion.params.trvdi"
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        model = Model(coords=coords, physics=self._physics(), time_step=30.0)

        # The static arguments are held IDENTICAL throughout. The second
        # run differs only in a dynamic aval: co2_vmr as a scalar
        # (fixed-CO2) versus as a TimeSeries (historical forcing), which
        # is a documented pair of real configurations. Varying a static
        # argument instead would not test anything, since keying the store
        # on the static arguments — the implementation this regresses —
        # would separate those two on its own.
        fixed_co2 = default_forcing(coords.horizontal)
        co2 = float(fixed_co2.co2_vmr)
        historical_co2 = dataclasses.replace(
            fixed_co2,
            co2_vmr=make_time_series(jnp.array([co2, co2]),
                                     jnp.array([0.0, 1e12])))

        first = model.run(forcing=fixed_co2, save_interval=0.5,
                          total_time=0.5)
        original = first.params[key]

        term = next(t for t in model.physics.terms
                    if t.name == "speedy_vertical_diffusion")
        term.params.set_value(
            term.params.get_value().replace(trvdi=jnp.array(2.0)))
        second = model.run(forcing=historical_co2, save_interval=0.5,
                           total_time=0.5)
        self.assertEqual(second.params[key], 2.0)

        # Re-running the first forcing reuses the FIRST executable, which
        # still holds the original value.
        again = model.run(forcing=fixed_co2, save_interval=0.5,
                          total_time=0.5)
        self.assertEqual(again.params[key], original)

    def test_traced_parameter_records_are_bounded(self):
        # A model that keeps meeting new input shapes retraces
        # indefinitely, and jax.clear_caches() cannot reach this dict, so
        # an unbounded store would grow for the model's lifetime.
        from jcm import model as model_module
        from jcm.model import Model

        cap = model_module._MAX_TRACED_PARAM_RECORDS
        model = Model.__new__(Model)
        model._traced_params = {}
        for trace_id in range(cap + 5):
            model._remember_traced_params(trace_id, {"a.b.c": trace_id})

        self.assertEqual(len(model._traced_params), cap)
        # The oldest go, the newest stay: a record is never replaced by a
        # different executable's values, only by nothing.
        self.assertNotIn(0, model._traced_params)
        self.assertEqual(model._traced_params[cap + 4], {"a.b.c": cap + 4})

    def test_capture_failure_never_breaks_a_completed_run(self):
        class _Exploding:
            @property
            def terms(self):
                raise RuntimeError("boom")

        with self.assertLogs("jcm.predictions", level="WARNING"):
            preds = self._preds(_Exploding())
        self.assertEqual(preds.params, {})

    def test_to_xarray_stamps_without_the_hydra_runners(self):
        # model.run(...).to_xarray().to_netcdf(...) never goes near the
        # runners, and is exactly how an interactive user writes a file.
        import xarray as xr

        from jcm.predictions import ModelPredictions

        class _Preds(ModelPredictions):
            def _trajectory_dataset(self):
                return xr.Dataset({"t": ("x", [1.0])})

        ds = _Preds(None, None, self._physics()).to_xarray()
        recorded = json.loads(ds.attrs["jcm_prov_params"])
        self.assertIn("speedy_convection.params.entmax", recorded)

    def test_observation_datasets_are_stamped(self):
        # An observer stream is often persisted on its own, and would
        # otherwise be the one output unable to say what produced it.
        import numpy as np
        import xarray as xr

        from jcm.predictions import ModelPredictions

        class _Observer:
            name = "stations"

            def to_dataset(self, samples, t0, dt):
                return xr.Dataset({"t": ("time", np.asarray(samples))})

        preds = ModelPredictions(
            None, None, self._physics(),
            observations=([1.0, 2.0],), observers=(_Observer(),),
            obs_t0_days=0.0, obs_dt_seconds=1800.0)
        ds = preds.observation_datasets()["stations"]
        recorded = json.loads(ds.attrs["jcm_prov_params"])
        self.assertIn("speedy_convection.params.entmax", recorded)

    def test_pytree_roundtrip_carries_no_model_and_says_so(self):
        import jax

        from jcm.predictions import ModelPredictions

        preds = ModelPredictions(None, None, self._physics())
        self.assertTrue(preds.params)
        # Unflattening rebuilds without coords/physics by design, so there
        # is no model to read parameters off.
        rebuilt = jax.tree_util.tree_unflatten(
            *reversed(jax.tree_util.tree_flatten(preds)))
        self.assertEqual(rebuilt.params, {})


if __name__ == "__main__":
    unittest.main()
