"""Tests for the carry-stored cloud-borne phase (#602 item 3)."""

import dataclasses
import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import (
    MAM4_SPEC,
    CloudBorneExchange,
    mass_name,
    number_name,
    tracer_specs,
)
from jcm.physics.aerosol.jam.activation.arg_term import JamActivationData
from jcm.physics.aerosol.jam.cloud_borne_store import (
    CARRY_KEY,
    CloudBorneCarryStore,
    apply_updates,
    carry_mode,
    mirror_names,
    tracer_view,
)
from jcm.physics_interface import PhysicsState

_IMPLICIT_SPEC = dataclasses.replace(MAM4_SPEC, cloud_borne=False)


class _VDiff:
    def __init__(self, kh):
        self.kh = kh


class StoreBasicsTest(unittest.TestCase):
    def test_no_mirror_tracers_declared_ever(self):
        # The cloud-borne phase is carry-only: tracer_specs is the
        # interstitial set regardless of cloud_borne (the #602 A/B verdict
        # plus the parked pySES check — CAM-SE keeps qqcw in pbuf too).
        for spec in (MAM4_SPEC, _IMPLICIT_SPEC):
            names = {s.name for s in tracer_specs(spec)}
            self.assertFalse(
                any(n.startswith(("mc_", "nc_")) for n in names)
            )
        n_interstitial = MAM4_SPEC.n_modes() + sum(
            len(m.species) for m in MAM4_SPEC.modes
        )
        self.assertEqual(len(list(tracer_specs(MAM4_SPEC))), n_interstitial)

    def test_store_term_rejects_implicit_spec(self):
        with self.assertRaisesRegex(ValueError, "cloud_borne"):
            CloudBorneCarryStore(spec=_IMPLICIT_SPEC)

    def test_view_and_apply_roundtrip(self):
        shape = (3, 2)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0),
            tracers={"m_so4_acc": jnp.full(shape, 1.0e-9)},
        )
        nm = mass_name("so4", "acc", cloud_borne=True)
        diagnostics = {CARRY_KEY: {nm: jnp.full(shape, 2.0e-10)}}
        view = tracer_view(MAM4_SPEC, state, diagnostics)
        np.testing.assert_allclose(float(view[nm][0, 0]), 2.0e-10)
        np.testing.assert_allclose(float(view["m_so4_acc"][0, 0]), 1.0e-9)

        # Carry mode integrates rate·dt now; tracers mode passes through.
        diagnostics2, passthrough = apply_updates(
            MAM4_SPEC, diagnostics, {nm: jnp.full(shape, 1.0e-13)}, 1000.0,
        )
        self.assertEqual(passthrough, {})
        np.testing.assert_allclose(
            float(diagnostics2[CARRY_KEY][nm][0, 0]), 3.0e-10, rtol=1e-6,
        )
        # An implicit population has no store: updates echo back.
        _, passthrough = apply_updates(
            _IMPLICIT_SPEC, diagnostics,
            {nm: jnp.full(shape, 1.0e-13)}, 1000.0,
        )
        self.assertIn(nm, passthrough)

    def test_store_term_seeds_and_fixes_structure(self):
        shape = (4, 2)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers={},
        )
        diagnostics = {
            "air_density": jnp.full(shape, 1.0),
            "layer_thickness": jnp.full(shape, 300.0),
        }
        term = CloudBorneCarryStore(spec=MAM4_SPEC)
        _, out = term(state, diagnostics, None, None)
        carry = out[CARRY_KEY]
        self.assertEqual(set(carry), set(mirror_names(MAM4_SPEC)))
        for v in carry.values():
            np.testing.assert_array_equal(np.asarray(v), 0.0)

    def test_store_mixes_carry_with_kh_and_conserves(self):
        shape = (6, 2)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers={},
        )
        nm = mass_name("so4", "acc", cloud_borne=True)
        carry = {
            n: jnp.zeros(shape) for n in mirror_names(MAM4_SPEC)
        }
        carry[nm] = jnp.zeros(shape).at[2].set(1.0e-9)
        diagnostics = {
            CARRY_KEY: carry,
            "air_density": jnp.full(shape, 1.0),
            "layer_thickness": jnp.full(shape, 300.0),
            "vertical_diffusion": _VDiff(jnp.full(shape, 30.0)),
            "_dt_seconds": 1800.0,
        }
        term = CloudBorneCarryStore(spec=MAM4_SPEC)
        _, out = term(state, diagnostics, None, None)
        mixed = np.asarray(out[CARRY_KEY][nm])
        self.assertGreater(float(mixed[1, 0]), 0.0)   # spread upward
        self.assertGreater(float(mixed[3, 0]), 0.0)   # and downward
        self.assertLess(float(mixed[2, 0]), 1.0e-9)
        np.testing.assert_allclose(float(mixed.sum(axis=0)[0]), 1.0e-9,
                                   rtol=1e-6)


class CarryPersistenceTest(unittest.TestCase):
    def test_carry_survives_across_steps_through_compute_tendencies(self):
        # A store that rebuilt zeros each step would pass every single-call
        # test and the cold-start integration run; this pins persistence:
        # a seeded reservoir threaded through repeated compute_tendencies
        # calls of a real ComposablePhysics (the store fed by the same
        # column-state provider the ECHAM chain uses) must survive with
        # its column mass intact — the store only mixes, never resets.
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.diagnostics.moist_air_state import (
            MoistAirColumnState,
        )
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords
        from jcm.forcing import ForcingData

        coords = get_coords(np.linspace(0, 1, 9), spectral_truncation=21)
        physics = ComposablePhysics(
            terms=[
                MoistAirColumnState(),
                CloudBorneCarryStore(spec=MAM4_SPEC),
            ],
            vectorize_columns=True,
        )
        physics.cache_coords(coords)
        prev = {
            **physics.get_empty_data(coords),
            **physics.initial_carry_state(coords),
        }
        nm = mass_name("so4", "acc", cloud_borne=True)
        seeded = dict(prev[CARRY_KEY])
        seeded[nm] = jnp.full_like(seeded[nm], 1.0e-10)
        prev = {**prev, CARRY_KEY: seeded}
        total0 = float(jnp.sum(seeded[nm]))

        nodal = tuple(coords.horizontal.nodal_shape)
        nlev = coords.nodal_shape[0]
        state = PhysicsState.zeros((nlev,) + nodal).copy(
            temperature=jnp.full((nlev,) + nodal, 288.0),
        )
        forcing = ForcingData.zeros(nodal)
        terrain = TerrainData.aquaplanet(coords)

        for step in range(3):
            _, prev = physics.compute_tendencies(
                state, forcing, terrain, prev_physics_data=prev,
            )
            field = np.asarray(prev[CARRY_KEY][nm])
            self.assertTrue(np.all(np.isfinite(field)), step)
            self.assertGreater(
                float(field.max()), 1.0e-12,
                f"carry lost after step {step + 1}",
            )
            # Mixing-only steps preserve the column mass exactly (no
            # removal terms composed here).
            np.testing.assert_allclose(
                float(np.sum(field)), total0, rtol=1e-5,
                err_msg=f"carry mass not conserved at step {step + 1}",
            )


class CarryModeExchangeTest(unittest.TestCase):
    def _setup(self, nlev=3, ncols=2, cf=0.5):
        shape = (nlev, ncols)
        tracers = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full(shape, 1.0e8)
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full(shape, 1.0e-9)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers=tracers,
        )
        n_modes = MAM4_SPEC.n_modes()
        can = jnp.asarray(
            [float(m.can_activate) for m in MAM4_SPEC.modes]
        ).reshape(-1, 1, 1)
        act = JamActivationData(
            number_frac=can * jnp.full((n_modes,) + shape, 0.3),
            mass_frac=can * jnp.full((n_modes,) + shape, 0.9),
        )

        class _Clouds:
            # Cover plus all-zero #708 ledger fields: "no cloud process
            # this step" routes resuspension to the timescale drain these
            # tests were written against.
            cloud_fraction = jnp.full(shape, cf)
            incloud_liquid = jnp.zeros(shape)
            incloud_ice = jnp.zeros(shape)
            incloud_rain_formation = jnp.zeros(shape)
            incloud_snow_formation = jnp.zeros(shape)
            incloud_riming = jnp.zeros(shape)
            process_cloud_fraction = jnp.zeros(shape)
            condensate_evaporation_rate = jnp.zeros(shape)

        diagnostics = {
            CARRY_KEY: {
                n: jnp.zeros(shape) for n in mirror_names(MAM4_SPEC)
            },
            "_jam_activation": act,
            "clouds": _Clouds(),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    def test_transfer_fills_carry_and_conserves_with_tendency(self):
        state, diagnostics = self._setup()
        term = CloudBorneExchange(spec=MAM4_SPEC)
        tend, out = term(state, diagnostics, None, None)
        nm_int = mass_name("so4", "acc")
        nm_cb = mass_name("so4", "acc", cloud_borne=True)
        # No cloud-borne tendencies through the accumulator in carry mode.
        self.assertNotIn(nm_cb, tend.tracers)
        dt = 1800.0
        carry_gain = np.asarray(out[CARRY_KEY][nm_cb])
        int_loss = np.asarray(tend.tracers[nm_int]) * dt
        self.assertGreater(float(carry_gain.min()), 0.0)
        # Pair conservation: carry gained exactly what the interstitial
        # tendency will remove over the step.
        np.testing.assert_allclose(carry_gain, -int_loss, rtol=1e-6)

    def test_resuspension_drains_the_carry(self):
        state, diagnostics = self._setup(cf=0.0)
        nm_cb = mass_name("so4", "acc", cloud_borne=True)
        diagnostics[CARRY_KEY][nm_cb] = jnp.full(
            state.temperature.shape, 1.0e-9,
        )
        term = CloudBorneExchange(spec=MAM4_SPEC)
        tend, out = term(state, diagnostics, None, None)
        self.assertLess(
            float(np.asarray(out[CARRY_KEY][nm_cb]).max()), 1.0e-9,
        )
        self.assertTrue(
            bool(jnp.all(tend.tracers[mass_name("so4", "acc")] > 0.0))
        )

    def test_mode_flag_helpers(self):
        self.assertTrue(carry_mode(MAM4_SPEC))
        self.assertFalse(carry_mode(_IMPLICIT_SPEC))


class CarryModeFactoryTest(unittest.TestCase):
    def test_factory_composes_store_first_and_drops_mirrors(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics()
        self.assertEqual(terms[0].name, "jam_cloud_borne_store")
        names = set()
        for t in terms:
            names |= {s.name for s in t.required_tracers()}
        self.assertFalse(any(n.startswith(("mc_", "nc_")) for n in names))
        # The exchange term is still composed (explicit phase, new home).
        self.assertIn(
            "jam_cloud_borne_exchange", [t.name for t in terms],
        )

    def test_implicit_composes_no_store(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        self.assertNotIn(
            "jam_cloud_borne_store",
            [t.name for t in jam_aerosol_physics(cloud_borne=False)],
        )

    def test_default_composes_store_first(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics()
        self.assertEqual(terms[0].name, "jam_cloud_borne_store")


if __name__ == "__main__":
    unittest.main()
