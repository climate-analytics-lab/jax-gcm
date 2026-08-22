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
from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
from jcm.physics_interface import PhysicsState

_IMPLICIT_SPEC = dataclasses.replace(MAM4_SPEC, cloud_borne=False)


class TracerLayoutSwitchTest(unittest.TestCase):
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
    """CloudData stub: cover plus the #708 scavenging-ledger fields.

    All-zero ledger fields mean "no cloud process this step", which routes
    the exchange term's downward direction to the slow timescale drain —
    the pre-#708 behaviour the legacy tests were written against.
    """

    def __init__(self, cloud_fraction, **ledger):
        self.cloud_fraction = cloud_fraction
        zeros = jnp.zeros_like(cloud_fraction)
        for f in ("incloud_liquid", "incloud_ice", "incloud_rain_formation",
                  "incloud_snow_formation", "incloud_riming",
                  "process_cloud_fraction", "condensate_evaporation_rate"):
            setattr(self, f, ledger.get(f, zeros))


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
        carry = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full(shape, n_int)
            carry[number_name(mode.short, cloud_borne=True)] = jnp.full(
                shape, n_cb
            )
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full(shape, q_int)
                carry[mass_name(sp, mode.short, cloud_borne=True)] = (
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
            CARRY_KEY: carry,
            "_jam_activation": act,
            "clouds": _Clouds(jnp.full(shape, cloud_fraction)),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    @staticmethod
    def _cb_rate(diagnostics_in, diagnostics_out, nm, dt=1800.0):
        """Effective cloud-borne rate from the sequential carry update."""
        return (
            np.asarray(diagnostics_out[CARRY_KEY][nm])
            - np.asarray(diagnostics_in[CARRY_KEY][nm])
        ) / dt

    def test_transfer_conserves_each_pair_exactly(self):
        state, diagnostics = self._setup(q_cb=2.0e-10, n_cb=1.0e7)
        tend, out = CloudBorneExchange()(state, diagnostics, None, None)
        for mode in MAM4_SPEC.modes:
            pairs = [(number_name(mode.short),
                      number_name(mode.short, cloud_borne=True))]
            pairs += [
                (mass_name(sp, mode.short),
                 mass_name(sp, mode.short, cloud_borne=True))
                for sp in mode.species
            ]
            for int_nm, cb_nm in pairs:
                cb_rate = self._cb_rate(diagnostics, out, cb_nm)
                dq_int = np.asarray(tend.tracers[int_nm])
                # Tolerance covers the f32 loss in the test's own
                # (new - old)/dt reconstruction of the carry rate; the
                # update itself is a single fused add.
                scale = float(np.abs(dq_int).max())
                np.testing.assert_allclose(
                    dq_int + cb_rate, 0.0,
                    atol=max(1e-4 * scale, 1e-22),
                    err_msg=f"{int_nm}/{cb_nm} exchange must conserve",
                )

    def test_activation_transfer_fills_cloud_borne(self):
        state, diagnostics = self._setup()
        tend, out = CloudBorneExchange()(state, diagnostics, None, None)
        cb = self._cb_rate(
            diagnostics, out, mass_name("so4", "acc", cloud_borne=True),
        )
        self.assertTrue(bool((cb > 0.0).all()))
        # The move is the relaxation fraction of the equilibrium target
        # f_mass * (q_int + q_cb) — cloud fraction sets the RATE (it stretches
        # the activation timescale by 1/cf), not the target.
        dt = 1800.0
        target = 0.9 * 1.0e-9
        phi = -np.expm1(-dt / (900.0 / 0.5))
        np.testing.assert_allclose(
            np.asarray(cb) * dt, target * phi, rtol=1e-6,
        )
        # And each mode uses ITS OWN fraction (the fixture halves it for
        # aitken), so a mode-axis misindexing shows up here.
        cb_ait = self._cb_rate(
            diagnostics, out, mass_name("so4", "ait", cloud_borne=True),
        )
        np.testing.assert_allclose(
            np.asarray(cb_ait) * dt, 0.5 * target * phi, rtol=1e-6,
        )

    def test_reservoir_is_not_capped_by_cloud_fraction(self):
        """A thin cloud fills the reservoir slowly, not partially.

        Capping the target at ``cf * f_act * q_total`` makes the grid-mean
        removal ``cf * f_act * rate_cb``, which is algebraically the implicit
        (no cloud-borne phase) treatment — the explicit reservoir then buys
        only a delay, and accumulation-mode sulfate, whose only real sink is
        in-cloud scavenging, is left removing far too slowly (#658). Cloud
        fraction belongs in the rate: thin cloud processes the box slowly but
        still processes all of it.
        """
        thin, thick = self._setup(cloud_fraction=0.1), self._setup(cloud_fraction=0.9)
        nm = mass_name("so4", "acc", cloud_borne=True)
        rate_thin = np.asarray(self._cb_rate(thin[1], CloudBorneExchange()(*thin, None, None)[1], nm))
        rate_thick = np.asarray(self._cb_rate(thick[1], CloudBorneExchange()(*thick, None, None)[1], nm))
        # Thicker cloud still activates faster...
        self.assertTrue(bool((rate_thick > rate_thin).all()))
        # ...but the thin-cloud transfer is far more than the 1/9 a
        # cf-proportional target would give.
        self.assertGreater(float(rate_thin.mean() / rate_thick.mean()), 0.2)

    def test_reservoir_drains_when_the_cloud_is_gone(self):
        """With no cloud the target is zero, so the phase resuspends."""
        state, diagnostics = self._setup(cloud_fraction=0.0)
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        cb = self._cb_rate(diagnostics, out, mass_name("so4", "acc", cloud_borne=True))
        self.assertTrue(bool((np.asarray(cb) <= 0.0).all()))

    def test_mass_and_number_use_their_own_fractions(self):
        # Large particles activate preferentially: the mass fraction (0.9)
        # must drive 3x the relative transfer of the number fraction (0.3).
        state, diagnostics = self._setup(q_int=1.0, n_int=1.0)
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        m = float(self._cb_rate(
            diagnostics, out, mass_name("so4", "acc", cloud_borne=True),
        )[0, 0])
        n = float(self._cb_rate(
            diagnostics, out, number_name("acc", cloud_borne=True),
        )[0, 0])
        self.assertAlmostEqual(m / n, 3.0, places=5)

    def test_clear_sky_resuspends_to_interstitial(self):
        state, diagnostics = self._setup(
            cloud_fraction=0.0, q_cb=1.0e-9, n_cb=1.0e8,
        )
        tend, out = CloudBorneExchange()(state, diagnostics, None, None)
        cb_key = mass_name("so4", "acc", cloud_borne=True)
        self.assertTrue(
            bool((self._cb_rate(diagnostics, out, cb_key) < 0.0).all())
        )
        self.assertTrue(
            bool(jnp.all(tend.tracers[mass_name("so4", "acc")] > 0.0))
        )
        # Bounded: the (sequentially updated) reservoir stays non-negative.
        self.assertGreaterEqual(
            float(np.asarray(out[CARRY_KEY][cb_key]).min()), 0.0,
        )

    def test_non_activatable_mode_only_resuspends(self):
        # pcm cannot activate (fraction masked to zero), so its cloud-borne
        # reservoir drains even under full cloud cover.
        state, diagnostics = self._setup(
            cloud_fraction=1.0, q_cb=1.0e-10, n_cb=1.0e7,
        )
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        cb = self._cb_rate(
            diagnostics, out, mass_name("poa", "pcm", cloud_borne=True),
        )
        self.assertTrue(bool((cb < 0.0).all()))

    def test_equilibrium_is_a_fixed_point(self):
        # cf = 1 and q_cb == f * (q_int + q_cb) → target == q_cb → no flux.
        # With f = 0.5 that is q_cb == q_int.
        state, diagnostics = self._setup(
            cloud_fraction=1.0, number_frac=0.5, mass_frac=0.5,
            q_int=1.0e-9, q_cb=1.0e-9, n_int=1.0e8, n_cb=1.0e8,
        )
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        for key in (mass_name("so4", "acc", cloud_borne=True),
                    number_name("acc", cloud_borne=True)):
            np.testing.assert_allclose(
                self._cb_rate(diagnostics, out, key), 0.0, atol=1e-25,
            )

    def test_positivity_preserved_both_phases(self):
        state, diagnostics = self._setup(q_cb=5.0e-10, n_cb=5.0e7)
        tend, out = CloudBorneExchange()(state, diagnostics, None, None)
        dt = 1800.0
        for nm, dq in tend.tracers.items():
            q_new = np.asarray(state.tracers[nm]) + np.asarray(dq) * dt
            self.assertGreaterEqual(float(q_new.min()), 0.0, nm)
        for nm, v in out[CARRY_KEY].items():
            self.assertGreaterEqual(float(np.asarray(v).min()), 0.0, nm)

    def test_empty_probe_state_is_safe(self):
        # ``Model.get_empty_data`` runs terms with no tracers seeded (the
        # store term guarantees the carry exists, so it stays in the
        # fixture).
        state, diagnostics = self._setup(q_cb=0.0, n_cb=0.0)
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


    # ----- The evaporation-ledger keying (#708) ------------------------

    def _ledger_setup(self, *, e_gm=0.0, pool_ic=0.0, form_ic=0.0,
                      cf_proc=0.5, q_cb=1.0e-10, n_cb=1.0e7):
        """Clear post-microphysics sky (cf=0) with a chosen process ledger."""
        state, diagnostics = self._setup(
            cloud_fraction=0.0, q_cb=q_cb, n_cb=n_cb,
        )
        shape = state.temperature.shape
        dt = 1800.0
        clouds = diagnostics["clouds"]
        clouds.condensate_evaporation_rate = jnp.full(shape, e_gm / dt)
        clouds.incloud_liquid = jnp.full(shape, pool_ic)
        clouds.incloud_rain_formation = jnp.full(shape, form_ic / dt)
        clouds.process_cloud_fraction = jnp.full(shape, cf_proc)
        return state, diagnostics

    def test_rained_out_cell_does_not_resuspend(self):
        # A cell whose condensate fully converted to precipitation ends
        # with cover 0 AND a zeroed pool — indistinguishable from an
        # evaporated cell by cover alone. The formation ledger marks it
        # live with f_evap = 0: the exchange term must leave q_cb for the
        # wetdep term (running just after) to rain out, instead of
        # resuspending ~86% of it into the interstitial phase in the same
        # step (the #708 race).
        state, diagnostics = self._ledger_setup(form_ic=1.0e-4)
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        nm = mass_name("so4", "acc", cloud_borne=True)
        np.testing.assert_allclose(
            np.asarray(out[CARRY_KEY][nm]),
            np.asarray(diagnostics[CARRY_KEY][nm]), rtol=1e-7,
        )

    def test_evaporated_cell_resuspends_fully(self):
        # A cell cleared by evaporation (positive evaporation ledger,
        # nothing formed, pool gone) releases the WHOLE reservoir in one
        # step — CAM's instant resuspension, not a 900 s relaxation.
        state, diagnostics = self._ledger_setup(e_gm=5.0e-4)
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        nm = mass_name("so4", "acc", cloud_borne=True)
        q1 = np.asarray(out[CARRY_KEY][nm])
        self.assertLess(q1.max(), 1e-6 * 1.0e-10)

    def test_partial_evaporation_releases_that_share(self):
        # E = pool: half the droplet population evaporated, half persists
        # -> release half the reservoir (target 0 under cf=0).
        state, diagnostics = self._ledger_setup(
            e_gm=1.0e-4, pool_ic=2.0e-4, cf_proc=0.5,
        )
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        nm = mass_name("so4", "acc", cloud_borne=True)
        q1 = np.asarray(out[CARRY_KEY][nm])
        np.testing.assert_allclose(q1, 0.5e-10, rtol=1e-5)

    def test_rainout_claim_caps_resuspension(self):
        # Evaporation and rainout both claimed condensate this step; the
        # rainout fraction (which wetdep will remove right after) caps the
        # released share so the two sinks cannot jointly overdraw:
        # f_evap = 0.5 but f_form = 1 -> nothing resuspends.
        state, diagnostics = self._ledger_setup(
            e_gm=1.0e-4, pool_ic=2.0e-4, form_ic=4.0e-4, cf_proc=0.5,
        )
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        nm = mass_name("so4", "acc", cloud_borne=True)
        np.testing.assert_allclose(
            np.asarray(out[CARRY_KEY][nm]),
            np.asarray(diagnostics[CARRY_KEY][nm]), rtol=1e-7,
        )

    def test_phase_transfer_is_not_evaporation(self):
        # WBF / freezing move condensate between phases WITHIN the pool
        # (liquid pool empty, ice pool full, no evaporation ledger): the
        # aerosol rides into the ice — no resuspension, even at cover 0.
        state, diagnostics = self._ledger_setup()
        shape = state.temperature.shape
        diagnostics["clouds"].incloud_ice = jnp.full(shape, 2.0e-4)
        _, out = CloudBorneExchange()(state, diagnostics, None, None)
        nm = mass_name("so4", "acc", cloud_borne=True)
        np.testing.assert_allclose(
            np.asarray(out[CARRY_KEY][nm]),
            np.asarray(diagnostics[CARRY_KEY][nm]), rtol=1e-7,
        )


class FactorySwitchTest(unittest.TestCase):
    def test_default_composes_exchange_and_store(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics()
        cats = [t.category for t in terms]
        self.assertIn("aerosol_cloud_borne", cats)
        self.assertIn("aerosol_cloud_borne_store", cats)
        # Exchange sits after drydep and before the aqueous split that
        # distributes by cloud-borne number.
        self.assertLess(
            cats.index("aerosol_drydep"), cats.index("aerosol_cloud_borne"),
        )
        self.assertLess(
            cats.index("aerosol_cloud_borne"),
            cats.index("aerosol_aqueous_chemistry"),
        )
        # Cloud-borne names are never dycore tracers.
        names = set()
        for t in terms:
            names |= {s.name for s in t.required_tracers()}
        self.assertNotIn(number_name("acc", cloud_borne=True), names)

    def test_cloud_borne_off_drops_terms(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics(cloud_borne=False)
        cats = [t.category for t in terms]
        self.assertNotIn("aerosol_cloud_borne", cats)
        self.assertNotIn("aerosol_cloud_borne_store", cats)

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
        self.assertIn(
            "jam_cloud_borne_store", [t.name for t in terms],
        )

if __name__ == "__main__":
    unittest.main()
