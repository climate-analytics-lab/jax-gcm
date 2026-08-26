"""Tests for run provenance capture (#591, #732)."""
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
    """Stands in for a CoordinateSystem: grid arrays beside scalar identity."""

    latitudes: object = None
    layers: int = 8


class _OpaqueBackendObject:
    """No fields worth recording, and a default repr carrying an address."""


class _FakeDycore:
    """Mirrors the shapes a real backend presents.

    In particular ``hypervis`` mixes 0-d array coefficients with a bulk
    profile in one container, the way pySES's ``diffusion_config`` does.
    """

    def __init__(self, constants=None):
        import jax.numpy as jnp
        import numpy as np
        self.dt_seconds = 900.0
        self.compute_omega = True
        self.diffusion = _FakeDiffusion()
        self._sl_options = {"interpolation_order": "cubic",
                            "off_centering": 0.5}
        self.hypervis = {"nu": jnp.asarray(2.5e-9),
                         "nu_top": jnp.asarray(250000.0),
                         "nu_ramp": jnp.arange(8.0)}
        self.coords = _FakeCoords(latitudes=np.linspace(-90, 90, 32))
        self.orography = np.zeros((32, 16))
        self.step_fn = lambda s: s
        self.colmap = _OpaqueBackendObject()
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

    def test_long_scalar_sequence_is_summarized(self):
        n = provenance._PARAM_ARRAY_MAX_ELEMS + 1
        self.assertEqual(self._walk([0.0] * n), {"p": f"<{n} scalars>"})

    def test_container_of_structures_is_walked_by_index(self):
        walked = self._walk([_FakeDiffusion(), {"k": 2.0}])
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

    def test_non_differentiable_parameter_variables_are_captured(self):
        # A parameter block containing a bool cannot be an nnx.Param, so
        # the schemes hold those as plain Variables. Filtering on Param
        # dropped SPEEDY's whole surface-flux set even though changing
        # e.g. cdl changes the simulation (#733 review).
        self.assertIn("speedy_surface_flux.surface_params.cdl", self.params)
        self.assertIn("speedy_surface_flux.surface_params.fwind0",
                      self.params)

    def test_held_suarez_parameters_are_recorded(self):
        # Every Held-Suarez tuning constant is an nnx.Variable, so under a
        # Param-only filter this physics recorded NOTHING at all.
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        params = provenance.describe_params(held_suarez_physics())["physics"]
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

        params = provenance.describe_params(_One())["physics"]
        self.assertIsInstance(params["macv2_sp_aerosol.params.theta"], list)

    def test_physics_without_variables_yields_no_block(self):
        class _Bare:
            terms = []

        self.assertNotIn("physics", provenance.describe_params(_Bare()))

    def test_no_physics_is_not_an_error(self):
        self.assertNotIn("physics", provenance.describe_params(None))

    def test_two_instances_of_one_term_both_survive(self):
        # A composition may call one scheme twice (the double-radiation
        # A/B). Keying purely on term name would clobber the first.
        from jcm.physics.speedy.speedy_terms import SpeedyConvection

        class _Pair:
            terms = [SpeedyConvection(), SpeedyConvection()]

        params = provenance.describe_params(_Pair())["physics"]
        self.assertIn("speedy_convection.params.entmax", params)
        self.assertIn("speedy_convection#1.params.entmax", params)


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

    def test_mixed_container_keeps_its_knobs_and_drops_its_arrays(self):
        # The #733 review: a backend may hold tuning coefficients and a
        # grid profile in ONE container (pySES's diffusion_config), so an
        # all-or-nothing rule on the container silently dropped the whole
        # hyperviscosity setting.
        # Approximate because a float32 0-d array widens to the exact
        # float64 it represents, which is the value the model held.
        self.assertAlmostEqual(self.params["hypervis.nu"], 2.5e-9, places=16)
        self.assertEqual(self.params["hypervis.nu_top"], 250000.0)
        self.assertNotIn("hypervis.nu_ramp", self.params)

    def test_zero_dim_arrays_are_knobs(self):
        # pySES stores nu/nu_top as 0-d arrays; an isinstance-scalar test
        # would drop exactly the coefficients worth recording.
        self.assertIsInstance(self.params["hypervis.nu"], float)

    def test_scalar_grid_identity_kept_but_not_the_grid(self):
        self.assertEqual(self.params["coords.layers"], 8)
        self.assertNotIn("coords.latitudes", self.params)
        self.assertNotIn("orography", self.params)
        self.assertNotIn("step_fn", self.params)

    def test_opaque_backend_objects_are_skipped(self):
        # Their repr carries no setting, and it embeds an address that
        # would make the record differ between two identical runs.
        self.assertNotIn("colmap", self.params)
        self.assertNotIn("0x", json.dumps(self.params))

    def test_record_is_reproducible(self):
        # params_sha feeds run_hash, so a second identical model must
        # produce a byte-identical record.
        again = provenance.describe_params(dycore=_FakeDycore())["dycore"]
        self.assertEqual(json.dumps(self.params, sort_keys=True),
                         json.dumps(again, sort_keys=True))

    def test_frozendict_style_mappings_are_walked(self):
        # pySES's timestep_config is a frozendict, not a dict subclass;
        # an isinstance(value, dict) test fell through to the leaf and
        # recorded the whole mapping's repr instead of its keys.
        import collections.abc

        class _Frozen(collections.abc.Mapping):
            def __init__(self, d):
                self._d = dict(d)

            def __getitem__(self, k):
                return self._d[k]

            def __iter__(self):
                return iter(self._d)

            def __len__(self):
                return len(self._d)

        out = {}
        provenance._describe_value(_Frozen({"tracer_subcycle": 3}), "ts", out)
        self.assertEqual(out, {"ts.tracer_subcycle": 3})


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

    def test_oversized_record_stays_recoverable_from_the_file_itself(self):
        # The #733 review: an over-cap record used to be replaced by a
        # pointer to the .provenance.json sidecar, but to_xarray and the
        # runners' snapshot files stamp these attributes WITHOUT writing
        # one, so the values became unrecoverable from the only artifact
        # that held them. Compress in place instead.
        huge = {"physics": {f"term.params.p{i}": float(i)
                            for i in range(20000)}}
        attrs = provenance.params_attrs(huge)
        self.assertRegex(attrs["jcm_prov_params_sha"], r"^[0-9a-f]{12}$")
        self.assertNotIn("sidecar", attrs["jcm_prov_params"])
        self.assertEqual(provenance.read_params(attrs), huge)

    def test_read_params_handles_both_forms_and_absence(self):
        small = {"physics": {"t.params.entrpen": 1e-4}}
        self.assertEqual(
            provenance.read_params(provenance.params_attrs(small)), small)
        self.assertEqual(provenance.read_params({}), {})

    def test_oversized_record_survives_a_real_netcdf_round_trip(self):
        import xarray as xr

        huge = {"physics": {f"term.params.p{i}": float(i)
                            for i in range(20000)}}
        ds = xr.Dataset({"t": ("x", [1.0])},
                        attrs=provenance.params_attrs(huge))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "big.nc"
            ds.to_netcdf(path)
            with xr.open_dataset(path) as back:
                self.assertEqual(provenance.read_params(back.attrs), huge)

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
        self.assertIn("speedy_convection.params.entmax", recorded["physics"])

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
