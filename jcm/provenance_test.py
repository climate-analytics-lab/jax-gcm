"""Tests for run provenance capture (#591)."""
import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

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
class _FakeDiffusion:
    """Stands in for a DiffusionFilter: all-scalar, so it is a knob."""

    timescale: int = 43200
    order: int = 2
    level_orders: None = None


@dataclasses.dataclass
class _FakeCoords:
    """Stands in for a CoordinateSystem: carries grid arrays, not knobs."""

    latitudes: object = None
    layers: int = 8


class _FakeDycore:
    def __init__(self, constants=None):
        import numpy as np
        self.dt_seconds = 900.0
        self.compute_omega = True
        self.diffusion = _FakeDiffusion()
        self._sl_options = {"interpolation_order": "cubic",
                            "off_centering": 0.5}
        self.coords = _FakeCoords(latitudes=np.linspace(-90, 90, 32))
        self.orography = np.zeros((32, 16))
        self.step_fn = lambda s: s
        if constants is not None:
            self.constants = constants


class DescribeLeafTest(unittest.TestCase):
    def test_scalars_pass_through(self):
        self.assertIsNone(provenance._describe_leaf(None))
        self.assertIs(provenance._describe_leaf(True), True)
        self.assertEqual(provenance._describe_leaf("cubic"), "cubic")

    def test_array_by_value_under_the_cap_and_hashed_over_it(self):
        import jax.numpy as jnp

        small = provenance._describe_leaf(jnp.arange(5.0))
        self.assertEqual(small, [0.0, 1.0, 2.0, 3.0, 4.0])
        n = provenance._PARAM_ARRAY_MAX_ELEMS + 1
        big = provenance._describe_leaf(jnp.arange(float(n)))
        self.assertEqual(big["shape"], [n])
        self.assertEqual(big["dtype"], "float32")
        # The hash has to separate one weight set from another, or a
        # summarized parameter is no record at all.
        other = provenance._describe_leaf(jnp.arange(float(n)) + 1.0)
        self.assertNotEqual(big["sha256"], other["sha256"])

    def test_scalar_keeps_the_value_the_model_held(self):
        import jax.numpy as jnp

        # float32 0.1 is not 0.1; record what ran, not a prettier rounding.
        self.assertEqual(provenance._describe_leaf(jnp.float32(0.1)),
                         0.10000000149011612)

    def test_tracer_is_marked_not_forced(self):
        import jax
        import jax.numpy as jnp

        seen = []

        def f(x):
            seen.append(provenance._describe_leaf(x))
            return x * 2

        jax.jit(f)(jnp.float32(1.0))
        self.assertEqual(seen, ["<traced>"])


class DescribePhysicsParamsTest(unittest.TestCase):
    """The gap: values that live in code, not in the config."""

    def setUp(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        self.params = provenance.describe_params(speedy_physics())["physics"]

    def test_scheme_defaults_are_recorded_not_just_overrides(self):
        from jcm.physics.speedy.speedy_terms import ConvectionParameters

        # Nothing overrode this, so it is exactly the case the composed
        # Hydra config cannot describe: the value is Parameters.default().
        key = "speedy_convection.params.entmax"
        self.assertIn(key, self.params)
        self.assertAlmostEqual(
            self.params[key],
            float(ConvectionParameters.default().entmax), places=12)

    def test_keys_locate_the_value_term_variable_field(self):
        # A recorded key must say where the value lives, unambiguously:
        # <term name>.<nnx variable>.<field>. Note this is NOT always the
        # Hydra override path — Hydra addresses the constructor keyword,
        # and SpeedyConvection takes convection_params= but stores it as
        # `params`. The record names the variable, not the keyword.
        for key in self.params:
            self.assertRegex(key, r"^[a-z0-9_#]+\.[a-z0-9_]+\.")

    def test_variables_not_named_params_are_captured(self):
        # Keying on nnx.Param rather than a `.params` attribute is the
        # point: these two hang off differently-named variables and a
        # `.params`-only walk would silently drop them.
        self.assertIn("speedy_shortwave_radiation.sw_params.albcl",
                      self.params)
        self.assertIn("speedy_upward_longwave.mod_radcon_params.emisfc",
                      self.params)

    def test_physics_without_params_yields_no_block(self):
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        self.assertNotIn("physics",
                         provenance.describe_params(held_suarez_physics()))

    def test_no_physics_is_not_an_error(self):
        self.assertNotIn("physics", provenance.describe_params(None))


class DescribeDycoreParamsTest(unittest.TestCase):
    def setUp(self):
        self.params = provenance.describe_params(
            dycore=_FakeDycore())["dycore"]

    def test_scalar_knobs_and_all_scalar_containers_kept(self):
        self.assertEqual(self.params["dt_seconds"], 900.0)
        self.assertIs(self.params["compute_omega"], True)
        self.assertEqual(self.params["diffusion.timescale"], 43200)
        self.assertIsNone(self.params["diffusion.level_orders"])
        # Private, but as much a knob as dt_seconds.
        self.assertEqual(self.params["_sl_options.interpolation_order"],
                         "cubic")

    def test_grid_data_and_callables_dropped(self):
        # coords/terrain carry arrays: grid data, not knobs. They must be
        # rejected on the array *without* being pulled off the device.
        for key in self.params:
            self.assertFalse(key.startswith("coords"), key)
            self.assertNotEqual(key, "orography")
            self.assertNotEqual(key, "step_fn")


class DescribeConstantsTest(unittest.TestCase):
    def test_live_constants_recorded(self):
        import jcm.constants as c

        params = provenance.describe_params()["constants"]
        self.assertEqual(params["constants.grav"], c.physical_constants.grav)

    def test_dycore_divergence_is_surfaced(self):
        # jcm.constants is live for attribute-access physics but captured
        # at construction by the dycore, so an override applied after the
        # model was built leaves the two genuinely disagreeing. Recording
        # only the live values would hide exactly that.
        import jcm.constants as c

        stale = c.physical_constants._replace(grav=1.62)
        params = provenance.describe_params(
            dycore=_FakeDycore(constants=stale))["constants"]
        self.assertEqual(params["constants_dycore.grav"], 1.62)
        self.assertEqual(params["constants.grav"],
                         c.physical_constants.grav)
        # Only the differing fields are repeated.
        self.assertNotIn("constants_dycore.cpd", params)

    def test_agreeing_constants_are_not_duplicated(self):
        import jcm.constants as c

        params = provenance.describe_params(
            dycore=_FakeDycore(constants=c.physical_constants))["constants"]
        self.assertFalse([k for k in params
                          if k.startswith("constants_dycore")])


class ParamsAttrsTest(unittest.TestCase):
    def test_empty_record_stamps_nothing(self):
        self.assertEqual(provenance.params_attrs({}), {})
        self.assertEqual(provenance.params_attrs(None), {})

    def test_values_and_hash_are_netcdf_safe_strings(self):
        attrs = provenance.params_attrs({"physics": {"a.b.c": 1.0}})
        for key, value in attrs.items():
            self.assertTrue(key.startswith("jcm_prov_"), key)
            self.assertIsInstance(value, str)
        self.assertRegex(attrs["jcm_prov_params_sha"], r"^[0-9a-f]{12}$")
        self.assertEqual(
            json.loads(attrs["jcm_prov_params"])["physics"]["a.b.c"], 1.0)

    def test_oversized_record_keeps_its_hash_and_points_at_the_sidecar(self):
        huge = {"physics": {f"term.params.p{i}": float(i)
                            for i in range(20000)}}
        attrs = provenance.params_attrs(huge)
        self.assertRegex(attrs["jcm_prov_params_sha"], r"^[0-9a-f]{12}$")
        self.assertIn("sidecar", attrs["jcm_prov_params"])

    def test_run_hash_separates_parameter_sweep_members(self):
        # Same code, same config, same inputs: without the parameters in
        # the hash every member of a sweep would share one run_hash.
        provenance.start_run()
        a = provenance.attrs({"physics": {"t.params.entrpen": 1e-4}})
        b = provenance.attrs({"physics": {"t.params.entrpen": 4e-4}})
        self.assertNotEqual(a["jcm_prov_run_hash"], b["jcm_prov_run_hash"])

    def test_sidecar_and_attrs_agree_on_the_run_hash(self):
        params = {"physics": {"t.params.entrpen": 1e-4}}
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

    def _preds(self, physics, dycore=None):
        from jcm.predictions import ModelPredictions
        return ModelPredictions(None, None, physics, dycore=dycore)

    def test_parameters_recorded_on_the_predictions(self):
        preds = self._preds(self._physics(), _FakeDycore())
        self.assertIn("speedy_convection.params.entmax",
                      preds.params["physics"])
        self.assertEqual(preds.params["dycore"]["dt_seconds"], 900.0)

    def test_snapshot_does_not_follow_later_mutation(self):
        # A calibration loop updates the term parameters in place between
        # iterations. A lazily-read record would report the *next*
        # iteration's values against this iteration's trajectory, which is
        # the failure this capture point exists to prevent.
        import jax.numpy as jnp

        physics = self._physics()
        preds = self._preds(physics)
        before = preds.params["physics"]["speedy_convection.params.entmax"]

        term = next(t for t in physics.terms if t.name == "speedy_convection")
        current = term.params.get_value()
        term.params.set_value(current.replace(entmax=jnp.array(0.25)))

        after = self._preds(physics).params[
            "physics"]["speedy_convection.params.entmax"]
        self.assertNotAlmostEqual(before, after, places=6)
        self.assertEqual(
            preds.params["physics"]["speedy_convection.params.entmax"],
            before)

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
        self.assertIn("speedy_convection.params.entmax",
                      recorded["physics"])

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
