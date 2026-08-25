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
# The dotted path also skips (rather than errors) when an OLD pre-0.3
# mam4-jax layout is installed — the adapter needs the core/coupling/physics
# package structure plus pcarbon aging (jax-gcm#721); the pinned version has
# both. The restore MUST be in a finally: importorskip imports the parent
# ``mam4_jax`` (flipping x64 on) and then raises Skipped when ``coupling``
# is missing — without the finally, that abandons the whole xdist worker
# in float64 and unrelated tests' dtype assertions fail.
try:
    pytest.importorskip("mam4_jax.coupling")
finally:
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

    def _assert_backend_runs_finite(self, backend):
        from mam4_jax.coupling import amicphys as _amicphys

        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        if not hasattr(_amicphys, "configure_condensation"):
            self.skipTest("mam4_jax lacks configure_condensation (PR #59)")

        # A condensation backend must produce finite, physical output through
        # the full wrapper. Restore the process-global default afterwards so it
        # can't leak into other tests sharing this process.
        state, diagnostics = _column_state()
        try:
            term = Mam4JaxMicrophysics(
                condensation_backend=backend, n_substeps=4,
            )
            self.assertEqual(_amicphys._COND["backend"], backend)
            self.assertEqual(term._condensation_backend, backend)
            tend, diags = term(state, diagnostics, None, None)
            for v in tend.tracers.values():
                self.assertTrue(np.all(np.isfinite(np.asarray(v))))
            aer = diags["_jam_state"]
            for f in (aer.r_dry, aer.r_wet, aer.rho, aer.kappa):
                self.assertTrue(np.all(np.isfinite(np.asarray(f))))
            self.assertTrue(np.all(np.asarray(aer.r_dry) > 0.0))
            key = mass_name(
                MAM4_SPEC.modes[0].species[0], MAM4_SPEC.modes[0].short
            )
            self.assertIn(key, tend.tracers)
        finally:
            _amicphys.configure_condensation(backend="substep")

    def test_substep_backend_runs_finite_and_physical(self):
        self._assert_backend_runs_finite("substep")

    def test_astem_backend_runs_finite_and_physical(self):
        self._assert_backend_runs_finite("astem")

    def test_default_backend_is_substep(self):
        from mam4_jax.coupling import amicphys as _amicphys

        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        if not hasattr(_amicphys, "configure_condensation"):
            self.skipTest("mam4_jax lacks configure_condensation (PR #59)")
        try:
            term = Mam4JaxMicrophysics()
            self.assertEqual(term._condensation_backend, "substep")
        finally:
            _amicphys.configure_condensation(backend="substep")

    def test_enable_x64_control(self):
        from mam4_jax.coupling import amicphys as _amicphys

        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        if not hasattr(_amicphys, "configure_condensation"):
            self.skipTest("mam4_jax lacks configure_condensation (PR #59)")
        # setUp/tearDown restore the process-global x64 flag around this test.
        try:
            # Default (None) keeps float64 (current behaviour).
            self.assertTrue(Mam4JaxMicrophysics()._enable_x64)
            self.assertTrue(jax.config.read("jax_enable_x64"))
            # Opt into float32 with a float32-safe backend.
            term = Mam4JaxMicrophysics(
                condensation_backend="substep", enable_x64=False,
            )
            self.assertFalse(term._enable_x64)
            self.assertFalse(jax.config.read("jax_enable_x64"))
            # Backend validation is delegated to mam4_jax: an unknown backend
            # raises the library's own ValueError (jcm adds no guard of its own).
            with self.assertRaises(ValueError):
                Mam4JaxMicrophysics(condensation_backend="not-a-backend")
        finally:
            _amicphys.configure_condensation(backend="substep")

    def test_carbon_aging_moves_bc_from_pcm_to_accum(self):
        """Ageing (jax-gcm#721): condensed H2SO4 coats the pcm mode and the
        core transfers the coated fraction of BC (and pcm number) to accum.
        Attribution is by monolayer-threshold sensitivity: an absurdly thick
        required coating (1e9 monolayers) makes ageing inert, leaving only
        the pcm→acc coagulation pathway, so the default-vs-inert difference
        isolates the ageing transfer.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        state, diagnostics = _column_state()
        # A healthy H2SO4 reservoir so within-step condensation builds a
        # real shell on pcm.
        state = state.copy(tracers={**state.tracers,
                                    "g_h2so4": jnp.full((4, 2), 1.0e-9)})
        # No try/finally needed: the constructor holds the threshold as
        # an nnx.Param and passes it per call — it never mutates the
        # core's process-global config, so instances cannot interfere.
        aged, _ = Mam4JaxMicrophysics()(state, diagnostics, None, None)
        inert_term = Mam4JaxMicrophysics(n_so4_monolayers=1.0e9)
        inert, _ = inert_term(state, diagnostics, None, None)

        d_bc_pcm = np.asarray(aged.tracers["m_bc_pcm"]
                              - inert.tracers["m_bc_pcm"], np.float64)
        d_bc_acc = np.asarray(aged.tracers["m_bc_acc"]
                              - inert.tracers["m_bc_acc"], np.float64)
        d_n_pcm = np.asarray(aged.tracers["n_pcm"]
                             - inert.tracers["n_pcm"], np.float64)
        self.assertTrue(np.all(d_bc_pcm < 0.0),
                        "ageing must remove BC from the pcm mode")
        self.assertTrue(np.all(d_bc_acc > 0.0),
                        "ageing must deliver BC to the accum mode")
        self.assertTrue(np.all(d_n_pcm < 0.0),
                        "ageing must move pcm number out")
        # BC has no other source/sink in the core: the ageing difference
        # must conserve BC between the two modes.
        np.testing.assert_allclose(
            d_bc_pcm + d_bc_acc, np.zeros_like(d_bc_pcm),
            atol=1e-6 * float(np.abs(d_bc_pcm).max()),
            err_msg="ageing must conserve BC across pcm+acc")

    def test_aging_threshold_is_a_differentiable_param_leaf(self):
        """The threshold must be an nnx.Param LEAF (visible to jax.grad /
        optimizers per the repo's differentiable-parameters rule), and
        the gradient of a tendency through the term w.r.t. it must be
        finite and non-zero.
        """
        from flax import nnx

        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        term = Mam4JaxMicrophysics()
        leaves = nnx.state(term, nnx.Param)
        flat = jax.tree.leaves(leaves)
        assert any(
            np.asarray(v).shape == () and float(np.asarray(v)) == 3.0
            for v in flat
        ), "n_so4_monolayers must appear as a Param leaf (default 3.0)"

        # Moderate H2SO4 so the pcm shell stays SUB-saturated at n=3:
        # in the saturated regime the criterion clamps and d/dn is
        # (correctly) exactly zero, which would make this test vacuous.
        state, diagnostics = _column_state()
        state = state.copy(tracers={**state.tracers,
                                    "g_h2so4": jnp.full((4, 2), 1.0e-11)})

        def bc_acc_tendency(n):
            t = Mam4JaxMicrophysics(n_so4_monolayers=1.0)
            t.n_so4_monolayers.set_value(n)
            tend, _ = t(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_bc_acc"])

        g = jax.grad(bc_acc_tendency)(jnp.asarray(3.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertLess(float(g), 0.0,
                        "thicker required coating must age less BC")

    def test_core_dtype_scoped_float32(self):
        # The float32 core runs under a SCOPED x64-off context: the global
        # flag (float64 host, e.g. pySES dynamics) must stay untouched, and
        # the float32-core forward tendencies must closely track the float64
        # core (forward is float32-safe per MAM4-JAX #60; the DEFAULT stays
        # float64 because the float32 REVERSE pass is non-finite — see the
        # constructor docstring).
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        t64 = Mam4JaxMicrophysics()          # default: float64 core
        self.assertFalse(t64._core_f32)
        state, diagnostics = _column_state()
        tend64, _ = t64(state, diagnostics, None, None)

        t32 = Mam4JaxMicrophysics(core_dtype="float32")
        self.assertTrue(t32._core_f32)
        tend32, _ = t32(state, diagnostics, None, None)
        self.assertTrue(jax.config.read("jax_enable_x64"),
                        "scoped f32 core must not clear the global x64 flag")

        for k in tend64.tracers:
            a = np.asarray(tend32.tracers[k], np.float64)
            b = np.asarray(tend64.tracers[k], np.float64)
            scale = max(float(np.abs(b).max()), 1e-30)
            # The 1e-16 floor absorbs tendencies that are numerically zero in
            # float64 (O(1e-17) round-off) and flush to exact zero in float32.
            np.testing.assert_allclose(
                a, b, atol=1e-3 * scale + 1e-16, err_msg=f"tracer {k}",
            )

        with self.assertRaises(ValueError):
            Mam4JaxMicrophysics(core_dtype="bf16")

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

    Guards the per-cell ``jax.vmap`` of the box-model core: the upstream MAM4
    box model runs a single cell, so each (level, column) point is integrated
    independently. A regression that batches the whole grid through the core at
    once would surface here as a blow-up rather than a silent slowdown.
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
