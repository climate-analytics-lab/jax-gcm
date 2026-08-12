"""Tests for the cloud-borne aerosol switch and exchange term (#602)."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import (
    MAM4_SPEC,
    CloudBorneExchange,
    CloudBorneExchangeParameters,
    mass_name,
    number_name,
    tracer_specs,
)
from jcm.physics.aerosol.jam.activation.arg_term import JamActivationData
from jcm.physics_interface import PhysicsState

_IMPLICIT_SPEC = dataclasses.replace(MAM4_SPEC, cloud_borne=False)


class TracerLayoutSwitchTest(unittest.TestCase):
    def test_explicit_population_declares_both_phases(self):
        names = {s.name for s in tracer_specs(MAM4_SPEC)}
        self.assertIn(number_name("acc"), names)
        self.assertIn(number_name("acc", cloud_borne=True), names)
        # One number per mode and one mass per (mode, species), doubled.
        n_interstitial = MAM4_SPEC.n_modes() + sum(
            len(m.species) for m in MAM4_SPEC.modes
        )
        self.assertEqual(len(names), 2 * n_interstitial)

    def test_implicit_population_declares_interstitial_only(self):
        names = {s.name for s in tracer_specs(_IMPLICIT_SPEC)}
        self.assertIn(number_name("acc"), names)
        self.assertFalse(
            any(n.startswith(("mc_", "nc_")) for n in names),
            "implicit population must not declare cloud-borne mirrors",
        )
        n_interstitial = MAM4_SPEC.n_modes() + sum(
            len(m.species) for m in MAM4_SPEC.modes
        )
        self.assertEqual(len(names), n_interstitial)


class _Clouds:
    def __init__(self, cloud_fraction):
        self.cloud_fraction = cloud_fraction


class CloudBorneExchangeTest(unittest.TestCase):
    def _setup(
        self,
        nlev=3,
        ncols=2,
        cloud_fraction=0.5,
        number_frac=0.3,
        mass_frac=0.9,
        q_int=1.0e-9,
        n_int=1.0e8,
        q_cb=0.0,
        n_cb=0.0,
    ):
        shape = (nlev, ncols)
        tracers = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full(shape, n_int)
            tracers[number_name(mode.short, cloud_borne=True)] = jnp.full(
                shape, n_cb
            )
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full(shape, q_int)
                tracers[mass_name(sp, mode.short, cloud_borne=True)] = (
                    jnp.full(shape, q_cb)
                )
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers=tracers,
        )
        # Per-mode fractions with the non-activatable pcm mode masked to
        # zero, as ARG publishes them. Modes get DISTINCT fractions
        # (base / (1 + i), so acc keeps the base value) so a per-mode
        # misindexing in the exchange term cannot hide behind uniformity.
        n_modes = MAM4_SPEC.n_modes()
        can = jnp.asarray(
            [float(m.can_activate) for m in MAM4_SPEC.modes]
        ).reshape(-1, 1, 1)
        per_mode = can / (1.0 + jnp.arange(n_modes).reshape(-1, 1, 1))
        act = JamActivationData(
            number_frac=per_mode * jnp.full((n_modes,) + shape, number_frac),
            mass_frac=per_mode * jnp.full((n_modes,) + shape, mass_frac),
        )
        diagnostics = {
            "_jam_activation": act,
            "clouds": _Clouds(jnp.full(shape, cloud_fraction)),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    def test_transfer_conserves_each_pair_exactly(self):
        state, diagnostics = self._setup(q_cb=2.0e-10, n_cb=1.0e7)
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        for mode in MAM4_SPEC.modes:
            pairs = [(number_name(mode.short),
                      number_name(mode.short, cloud_borne=True))]
            pairs += [
                (mass_name(sp, mode.short),
                 mass_name(sp, mode.short, cloud_borne=True))
                for sp in mode.species
            ]
            for int_nm, cb_nm in pairs:
                np.testing.assert_array_equal(
                    np.asarray(tend.tracers[int_nm] + tend.tracers[cb_nm]),
                    0.0,
                    err_msg=f"{int_nm}/{cb_nm} exchange must conserve",
                )

    def test_activation_transfer_fills_cloud_borne(self):
        state, diagnostics = self._setup()
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        cb = tend.tracers[mass_name("so4", "acc", cloud_borne=True)]
        self.assertTrue(bool(jnp.all(cb > 0.0)))
        # The move is the relaxation fraction of the equilibrium target
        # cf * f_mass * (q_int + q_cb).
        dt = 1800.0
        target = 0.5 * 0.9 * 1.0e-9
        phi = -np.expm1(-dt / 900.0)
        np.testing.assert_allclose(
            np.asarray(cb) * dt, target * phi, rtol=1e-6,
        )
        # And each mode uses ITS OWN fraction (the fixture halves it for
        # aitken), so a mode-axis misindexing shows up here.
        cb_ait = tend.tracers[mass_name("so4", "ait", cloud_borne=True)]
        np.testing.assert_allclose(
            np.asarray(cb_ait) * dt, 0.5 * target * phi, rtol=1e-6,
        )

    def test_mass_and_number_use_their_own_fractions(self):
        # Large particles activate preferentially: the mass fraction (0.9)
        # must drive 3x the relative transfer of the number fraction (0.3).
        state, diagnostics = self._setup(q_int=1.0, n_int=1.0)
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        m = float(tend.tracers[mass_name("so4", "acc", cloud_borne=True)][0, 0])
        n = float(tend.tracers[number_name("acc", cloud_borne=True)][0, 0])
        self.assertAlmostEqual(m / n, 3.0, places=5)

    def test_clear_sky_resuspends_to_interstitial(self):
        state, diagnostics = self._setup(
            cloud_fraction=0.0, q_cb=1.0e-9, n_cb=1.0e8,
        )
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        cb_key = mass_name("so4", "acc", cloud_borne=True)
        self.assertTrue(bool(jnp.all(tend.tracers[cb_key] < 0.0)))
        self.assertTrue(
            bool(jnp.all(tend.tracers[mass_name("so4", "acc")] > 0.0))
        )
        # Bounded: the reservoir cannot go negative in one step.
        q_new = 1.0e-9 + np.asarray(tend.tracers[cb_key]) * 1800.0
        self.assertGreaterEqual(float(q_new.min()), 0.0)

    def test_non_activatable_mode_only_resuspends(self):
        # pcm cannot activate (fraction masked to zero), so its cloud-borne
        # reservoir drains even under full cloud cover.
        state, diagnostics = self._setup(
            cloud_fraction=1.0, q_cb=1.0e-10, n_cb=1.0e7,
        )
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        cb = tend.tracers[mass_name("poa", "pcm", cloud_borne=True)]
        self.assertTrue(bool(jnp.all(cb < 0.0)))

    def test_equilibrium_is_a_fixed_point(self):
        # cf = 1 and q_cb == f * (q_int + q_cb) → target == q_cb → no flux.
        # With f = 0.5 that is q_cb == q_int.
        state, diagnostics = self._setup(
            cloud_fraction=1.0, number_frac=0.5, mass_frac=0.5,
            q_int=1.0e-9, q_cb=1.0e-9, n_int=1.0e8, n_cb=1.0e8,
        )
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        for key in (mass_name("so4", "acc", cloud_borne=True),
                    number_name("acc", cloud_borne=True)):
            np.testing.assert_allclose(
                np.asarray(tend.tracers[key]), 0.0, atol=1e-25,
            )

    def test_positivity_preserved_both_phases(self):
        state, diagnostics = self._setup(q_cb=5.0e-10, n_cb=5.0e7)
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        dt = 1800.0
        for nm, dq in tend.tracers.items():
            q_new = np.asarray(state.tracers[nm]) + np.asarray(dq) * dt
            self.assertGreaterEqual(float(q_new.min()), 0.0, nm)

    def test_empty_probe_state_is_safe(self):
        # ``Model.get_empty_data`` runs terms with no tracers seeded.
        state, diagnostics = self._setup()
        state = state.copy(tracers={})
        tend, _ = CloudBorneExchange()(state, diagnostics, None, None)
        for dq in tend.tracers.values():
            np.testing.assert_array_equal(np.asarray(dq), 0.0)

    def test_implicit_population_rejected_at_compose_time(self):
        # Composed against a population without the mirrors, the cloud-borne
        # tendencies would be silently dropped while the interstitial side
        # still fires — venting mass. Must fail loudly instead.
        with self.assertRaisesRegex(ValueError, "cloud_borne"):
            CloudBorneExchange(spec=_IMPLICIT_SPEC)

    def test_grad_through_timescales(self):
        state, diagnostics = self._setup(q_cb=2.0e-10, n_cb=1.0e7)

        def loss(tau):
            params = CloudBorneExchangeParameters(
                activation_timescale=tau,
                resuspension_timescale=jnp.asarray(900.0),
            )
            term = CloudBorneExchange(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(900.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)


class FactorySwitchTest(unittest.TestCase):
    def test_default_composes_exchange_and_mirrors(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics()
        cats = [t.category for t in terms]
        self.assertIn("aerosol_cloud_borne", cats)
        # Exchange sits after drydep and before the aqueous split that
        # distributes by cloud-borne number.
        self.assertLess(
            cats.index("aerosol_drydep"), cats.index("aerosol_cloud_borne"),
        )
        self.assertLess(
            cats.index("aerosol_cloud_borne"),
            cats.index("aerosol_aqueous_chemistry"),
        )
        names = set()
        for t in terms:
            names |= {s.name for s in t.required_tracers()}
        self.assertIn(number_name("acc", cloud_borne=True), names)

    def test_cloud_borne_off_drops_term_and_mirrors(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics(cloud_borne=False)
        self.assertNotIn(
            "aerosol_cloud_borne", [t.category for t in terms],
        )
        names = set()
        for t in terms:
            names |= {s.name for s in t.required_tracers()}
        self.assertFalse(
            any(n.startswith(("mc_", "nc_")) for n in names),
            "cloud_borne=False must not declare mirror tracers",
        )
        # The A/B cost claim: half the aerosol tracers are gone.
        on = {
            s.name
            for t in jam_aerosol_physics()
            for s in t.required_tracers()
        }
        self.assertEqual(
            len([n for n in on if n.startswith(("mc_", "nc_"))]),
            len([n for n in on if n.startswith(("m_", "n_"))]),
        )

    def test_instance_core_with_conflicting_flag_raises(self):
        from jcm.physics.aerosol.jam import (
            PlaceholderMicrophysics,
            jam_aerosol_physics,
        )

        core = PlaceholderMicrophysics()
        with self.assertRaisesRegex(ValueError, "cloud_borne"):
            jam_aerosol_physics(microphysics=core, cloud_borne=False)
        # A matching flag merely validates.
        terms = jam_aerosol_physics(microphysics=core, cloud_borne=True)
        self.assertIn("aerosol_cloud_borne", [t.category for t in terms])


if __name__ == "__main__":
    unittest.main()
