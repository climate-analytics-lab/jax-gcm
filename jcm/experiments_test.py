"""Tests for the recipe door ``jcm.experiments`` (issue #751).

Deliberately imports neither ``hydra`` nor ``omegaconf`` at module top level: the
door hides them, so a caller (and this test) needs only ``jcm.experiments``. A
meta-test below enforces that on this file's own AST.
"""

import ast
import contextlib
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from jcm import experiments, runners
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData


def _t63l47_coords():
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.utils import get_coords
    return get_coords(vertical_coords=get_echam_levels(47),
                      spectral_truncation=63)


def _term_names(model):
    return [type(t).__name__ for t in model.physics.terms]


class TestExperimentsDoor(unittest.TestCase):
    def test_available_lists_names_and_summaries(self):
        av = experiments.available()
        self.assertIn("speedy-t31", av)
        self.assertIn("t63-echam-jam", av)
        # The one-line summary is the yaml's first human comment.
        self.assertIn("SPEEDY", av["speedy-t31"])
        self.assertTrue(all(isinstance(v, str) for v in av.values()))

    def test_unknown_name_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown experiment"):
            experiments.load("does-not-exist")

    def test_load_speedy_builds_and_hides_hydra(self):
        # Cheap real build (SPEEDY needs no network); aquaplanet/default keep it
        # offline. Exercises load()'s whole build path + the isothermal init
        # branch, and asserts no DictConfig leaks out.
        exp = experiments.load(
            "speedy-t31", **{"terrain": "aquaplanet", "forcing": "default",
                             "run.total_time": 2.0, "run.save_interval": 1.0})
        from jcm.model import Model
        self.assertIsInstance(exp.model, Model)
        self.assertIsInstance(exp.config, dict)
        self.assertEqual(exp.run_kwargs["total_time"], 2.0)
        # isothermal init supplies no initial_state.
        self.assertNotIn("initial_state", exp.run_kwargs)
        # No omegaconf container survives on the returned surface.
        self.assertNotIn("DictConfig", type(exp.config).__name__)
        self.assertIs(exp.run_kwargs["forcing"], exp.forcing)

    def test_module_imports_no_hydra_or_omegaconf_at_top_level(self):
        # The door's whole point: a caller (this file) never imports hydra.
        tree = ast.parse(Path(__file__).read_text())
        banned = {"hydra", "omegaconf"}
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.append(node.module.split(".")[0])
        self.assertFalse(banned & set(top_level),
                         f"top-level imports leak hydra/omegaconf: {top_level}")


def _patched_engine(shape):
    """Network-free stand-ins so both doors traverse the same patched engine.

    Terrain is forced to aquaplanet (identical for both builds, so model
    equivalence is unaffected) because the global ``open_dataset`` stub the
    forcing readers need would otherwise starve the terrain load.
    """
    base = ForcingData.zeros(shape)
    return [
        mock.patch.object(runners, "_resolve_data_path", side_effect=lambda p: p),
        mock.patch.object(runners, "_resolve_auto_ozone", return_value=None),
        mock.patch.object(runners, "build_terrain",
                          side_effect=lambda cfg, c: TerrainData.aquaplanet(c)),
        mock.patch.object(ForcingData, "from_file", return_value=base),
        mock.patch("xarray.open_dataset", return_value=xr.Dataset()),
        mock.patch("jcm.forcing.read_anthropogenic_emissions",
                   return_value={"emis_so2_ant": jnp.ones(shape)}),
        mock.patch("jcm.forcing.read_prescribed_aerosol_emissions",
                   return_value=None),
        mock.patch("jcm.forcing.validate_emissions_grid"),
        mock.patch("jcm.forcing.read_dms_seawater", return_value=jnp.ones(shape)),
        mock.patch("jcm.forcing.read_dust_source", return_value=jnp.ones(shape)),
        mock.patch("jcm.forcing.read_oxidant_vmr",
                   return_value={"oh": jnp.ones((1, *shape))}),
        mock.patch("jcm.forcing.validate_oxidant_levels"),
    ]


@pytest.mark.slow
class TestExperimentsAcceptance(unittest.TestCase):
    """#751 acceptance: the Python door reproduces the CLI composition."""

    def test_jam_load_equivalent_to_cli_composition(self):
        coords = _t63l47_coords()
        shape = tuple(int(x) for x in coords.horizontal.nodal_shape)
        with contextlib.ExitStack() as stack:
            for p in _patched_engine(shape):
                stack.enter_context(p)
            exp = experiments.load("t63-echam-jam")
            # The CLI composition, built through the same runners the door uses.
            cfg = experiments._compose("t63-echam-jam", [])
            ref_model = runners.build_model(cfg)
            ref_forcing = runners.build_forcing(
                cfg, ref_model.coords,
                dycore=getattr(ref_model, "dycore", None))

        # Model equivalence: same coords, physics term names, and timestep.
        self.assertEqual(exp.model.coords.nodal_shape, ref_model.coords.nodal_shape)
        self.assertEqual(_term_names(exp.model), _term_names(ref_model))
        self.assertEqual(float(exp.model.dt_si.m), float(ref_model.dt_si.m))
        # Forcing equivalence: pytree-equal leaf for leaf.
        la = jax.tree_util.tree_leaves(exp.forcing)
        lb = jax.tree_util.tree_leaves(ref_forcing)
        self.assertEqual(len(la), len(lb))
        for x, y in zip(la, lb):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        # The jw recipe's init is applied, ready for model.run(**run_kwargs).
        self.assertIn("initial_state", exp.run_kwargs)


@pytest.mark.slow
class TestExperimentsSmoke(unittest.TestCase):
    def test_speedy_run_kwargs_produce_finite_output(self):
        # No mocks: SPEEDY needs no network. Run one save interval and confirm
        # model.run accepts **run_kwargs and yields finite output.
        exp = experiments.load(
            "speedy-t31", **{"terrain": "aquaplanet", "forcing": "default",
                             "run.total_time": 1.0, "run.save_interval": 1.0})
        preds = exp.model.run(**exp.run_kwargs)
        ds = preds.to_xarray()
        self.assertTrue(bool(np.isfinite(ds.temperature.values).all()))


if __name__ == "__main__":
    unittest.main()
