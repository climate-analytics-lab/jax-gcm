"""Tests for the MAM4-JAX microphysics core wrapper (issue #490).

Skipped unless the optional GPL-3.0 ``mam4-jax`` dependency is installed
(``pip install jcm[mam4]``); CI without the extra simply skips them. The
wrapper enables ``jax_enable_x64`` on construction, so each test restores the
prior flag in ``tearDown`` to keep sibling tests on jcm's default float32.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Importing mam4_jax flips jax_enable_x64 on globally (it needs float64).
# Capture and restore the flag around the import so *collection* of this module
# doesn't leave x64 on and corrupt sibling tests' float32 dtype assertions when
# the optional dependency is installed. Each test below re-enables x64 (via the
# term's lazy import) and restores it in tearDown.
_x64_at_import = jax.config.read("jax_enable_x64")
pytest.importorskip("mam4_jax")
jax.config.update("jax_enable_x64", _x64_at_import)


def _column_state(nlev=4, ncols=2):
    """Build a PhysicsState + diagnostics with seeded aerosol tracers."""
    from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
    from jcm.physics_interface import PhysicsState

    tracers = {}
    for mode in MAM4_SPEC.modes:
        tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
        tracers[number_name(mode.short, cloud_borne=True)] = jnp.full(
            (nlev, ncols), 1.0e6
        )
        for sp in mode.species:
            tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1.0e-10)
            tracers[mass_name(sp, mode.short, cloud_borne=True)] = jnp.full(
                (nlev, ncols), 1.0e-12
            )
    # Gas-phase precursors fed to the core.
    tracers["g_h2so4"] = jnp.full((nlev, ncols), 1.0e-12)
    tracers["g_soag"] = jnp.full((nlev, ncols), 1.0e-12)
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 280.0),
        specific_humidity=jnp.full((nlev, ncols), 5.0e-3),
        tracers=tracers,
    )
    diagnostics = {
        "pressure_full": jnp.full((nlev, ncols), 9.0e4),
        "height_full": jnp.full((nlev, ncols), 3.0e3),
        "_dt_seconds": 1800.0,
    }
    return state, diagnostics


class Mam4JaxAdapterTest(unittest.TestCase):
    def setUp(self):
        self._x64 = jax.config.read("jax_enable_x64")

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._x64)

    def test_factory_resolves_mam4_jax(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics(microphysics="mam4_jax")
        core = next(t for t in terms if t.category == "aerosol_microphysics")
        self.assertEqual(core.name, "jam_mam4_jax_microphysics")

    def test_gas_tracers_map_to_their_pcnst_slots(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        # Gases share the single q_pack with the aerosol tracers.
        packed = dict(Mam4JaxMicrophysics()._q_pack)
        self.assertEqual(packed["g_h2so4"], 6)
        self.assertEqual(packed["g_soag"], 9)

    def test_packing_covers_every_interstitial_tracer(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        term = Mam4JaxMicrophysics()
        packed = {name for name, _ in term._q_pack}
        expected = {"g_h2so4", "g_soag"}
        for mode in MAM4_SPEC.modes:
            expected.add(number_name(mode.short))
            for sp in mode.species:
                expected.add(mass_name(sp, mode.short))
        self.assertEqual(packed, expected)
        # pcnst indices are unique and inside the aerosol/gas band [6, 34].
        idxs = [idx for _, idx in term._q_pack]
        self.assertEqual(len(idxs), len(set(idxs)))
        self.assertTrue(all(6 <= i <= 34 for i in idxs))

    def test_forward_finite_and_physical(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        state, diagnostics = _column_state()
        term = Mam4JaxMicrophysics()
        tend, diags = term(state, diagnostics, None, None)

        key = mass_name(MAM4_SPEC.modes[0].species[0], MAM4_SPEC.modes[0].short)
        self.assertIn(key, tend.tracers)
        for v in tend.tracers.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))))
        # Tendencies are returned at the model dtype, not float64.
        self.assertEqual(tend.tracers[key].dtype, state.temperature.dtype)

        aer = diags["_jam_state"]
        self.assertEqual(aer.r_dry.shape, (MAM4_SPEC.n_modes(), 4, 2))
        for f in (aer.r_dry, aer.r_wet, aer.rho, aer.kappa):
            self.assertTrue(np.all(np.isfinite(np.asarray(f))))
        self.assertTrue(np.all(np.asarray(aer.r_wet) >= np.asarray(aer.r_dry)))
        self.assertTrue(np.all(np.asarray(aer.r_dry) > 0.0))

    def test_input_sanitisation_keeps_finite(self):
        from jcm.physics.aerosol.jam import mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        # A non-finite / negative input tracer must be sanitised before the
        # solve so it can never produce a NaN tendency.
        state, diagnostics = _column_state()
        bad = dict(state.tracers)
        bad[mass_name("so4", "acc")] = bad[mass_name("so4", "acc")].at[0, 0].set(
            jnp.nan
        )
        bad[mass_name("bc", "acc")] = bad[mass_name("bc", "acc")].at[1, 0].set(
            -1.0
        )
        state = state.copy(tracers=bad)
        tend, _ = Mam4JaxMicrophysics()(state, diagnostics, None, None)
        for v in tend.tracers.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))))

    def test_nonconvergence_gate_keeps_finite_and_logs(self):
        from unittest import mock

        from jcm.physics.aerosol.jam import mass_name
        from jcm.physics.aerosol.jam.microphysics import mam4_jax as _m

        # Force the core to emit non-finite output (a diverged / non-converged
        # solve). The gate must (a) keep the whole output finite (fall back to a
        # zero tendency) and (b) log the count rather than silently hiding it.
        state, diagnostics = _column_state()
        calcsize, wateruptake, amicphys, data = _m._core()

        def poisoned_amicphys(s):
            out = dict(amicphys(s))
            out["q"] = out["q"] * jnp.inf  # every cell non-finite
            return out

        term = _m.Mam4JaxMicrophysics()
        key = mass_name("so4", "acc")
        with mock.patch.object(
            _m, "_core",
            return_value=(calcsize, wateruptake, poisoned_amicphys, data),
        ):
            with self.assertLogs(_m.logger, level="WARNING") as cm:
                tend, _ = term(state, diagnostics, None, None)
                jax.block_until_ready(tend.tracers[key])

        for v in tend.tracers.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))))
        self.assertTrue(any("did not converge" in m for m in cm.output))

    def test_grad_through_a_tracer_is_finite(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        state, diagnostics = _column_state()
        term = Mam4JaxMicrophysics()
        key = mass_name(MAM4_SPEC.modes[0].species[0], MAM4_SPEC.modes[0].short)
        base = state.tracers[key]

        def loss(scale):
            tr = {**state.tracers, key: base * scale}
            s = state.copy(tracers=tr)
            tend, _ = term(s, diagnostics, None, None)
            return sum(jnp.sum(v.astype(jnp.float64) ** 2)
                       for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0, jnp.float64))
        self.assertTrue(np.isfinite(float(g)))


@pytest.mark.slow
class Mam4JaxModelTest(unittest.TestCase):
    """End-to-end: the MAM4-JAX core runs inside the full ECHAM GCM.

    Guards the per-cell ``jax.vmap`` of the box-model core. amicphys's
    gas-exchange uses an implicit diffrax solver; handing it the whole grid as
    one batched state couples its Jacobian across every cell and the T21
    compile exceeds 80 GB. With the per-cell vmap the same run compiles in
    ~1 GB — so a regression here surfaces as an out-of-memory blow-up, not a
    silent slowdown.
    """

    def setUp(self):
        self._x64 = jax.config.read("jax_enable_x64")

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._x64)

    def test_t21_runs_finite_with_mam4_core(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, 21), spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        model = Model(
            coords=coords, time_step=30, terrain=terrain,
            physics=echam_physics(
                aerosol_module="jam", cloud_scheme="2m",
                jam_microphysics="mam4_jax",
            ),
        )
        pred = model.run(save_interval=0.0625, total_time=0.0625)
        dyn = pred.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        self.assertTrue(bool(jnp.all(dyn.temperature > 150.0)))
        self.assertTrue(bool(jnp.all(dyn.temperature < 360.0)))


if __name__ == "__main__":
    unittest.main()
