"""Round-trip tests for :mod:`jcm.dycore.dinosaur.state_bridge`.

The fast-test :mod:`jcm.physics_interface_test` already exercises both
directions of the conversion at T31L8 sigma. This file marks an end-to-end
round-trip as ``@pytest.mark.slow`` so the PR-time slow-coverage run also
walks ``physics_state_to_dynamics_state`` — without it the inverse direction
shows up as uncovered against the .coveragerc-pr threshold even though every
fast test that takes a ``PhysicsState`` initial state goes through it.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.dycore.dinosaur.state_bridge import (
    dynamics_state_to_physics_state,
    physics_state_to_dynamics_state,
    physics_tendency_to_dynamics_tendency,
)
from jcm.model import Model
from jcm.physics.physics_term import TracerSpec
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import PhysicsTendency


class TestSpecificHumidityContract(unittest.TestCase):
    """Humidity crosses the Dinosaur boundary as dimensionless kg/kg."""

    def setUp(self):
        self._previous_x64 = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", False)
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.model = Model(coords=self.coords, time_step=720)
        self.primitive = self.model.dycore.primitive

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._previous_x64)

    def test_float32_state_and_tendency_round_trip(self):
        """Every mass mixing ratio crosses the boundary unscaled, in kg/kg."""
        base = self.model.dycore.to_physics_state(self.model.initial_state())
        shape = base.specific_humidity.shape
        q = jnp.full(shape, 0.0125, dtype=jnp.float32)
        # A generic mass mixing ratio (condensate, aerosol or gas mass all
        # take this path) and a number concentration, which opts out.
        cloud_mass = jnp.full(shape, 2.5e-4, dtype=jnp.float32)
        number = jnp.full(shape, 1.0e8, dtype=jnp.float32)
        specs = {
            "qc": TracerSpec("qc", units="kg/kg"),
            "qnc": TracerSpec("qnc", units="kg^-1", nondimensionalize=False),
        }
        seeded = base.copy(
            specific_humidity=q,
            tracers={"qc": cloud_mass, "qnc": number},
        )

        modal = physics_state_to_dynamics_state(
            seeded, self.primitive, tracer_specs=specs,
        )
        q_in_dynamics = self.coords.horizontal.to_nodal(
            modal.tracers["specific_humidity"]
        )
        mass_in_dynamics = self.coords.horizontal.to_nodal(
            modal.tracers["qc"]
        )
        recovered = dynamics_state_to_physics_state(
            modal, self.primitive, tracer_specs=specs,
        )

        self.assertEqual(recovered.specific_humidity.dtype, jnp.float32)
        np.testing.assert_allclose(recovered.specific_humidity, q, rtol=1e-6)
        np.testing.assert_allclose(q_in_dynamics, q, rtol=1e-6)
        # Condensate is stored unscaled: Dinosaur reads it directly for the
        # virtual-temperature loading term, so a g/kg store would weaken that
        # coupling by 1000x — the same defect #666 fixed for humidity.
        np.testing.assert_allclose(mass_in_dynamics, cloud_mass, rtol=2e-6)
        np.testing.assert_allclose(
            recovered.tracers["qc"], cloud_mass, rtol=2e-6,
        )
        # ``nondimensionalize=False`` still passes straight through.
        np.testing.assert_allclose(
            recovered.tracers["qnc"], number, rtol=2e-6,
        )

        dqdt = jnp.full(shape, 1.25e-8, dtype=jnp.float32)
        cloud_mass_tend = jnp.full(shape, 2.5e-9, dtype=jnp.float32)
        tendency = PhysicsTendency.zeros(
            shape,
            specific_humidity=dqdt,
            tracers={"qc": cloud_mass_tend},
        )
        modal_tendency = physics_tendency_to_dynamics_tendency(
            tendency, self.primitive, tracer_specs=specs,
        )
        q_tend_in_dynamics = self.coords.horizontal.to_nodal(
            modal_tendency.tracers["specific_humidity"]
        )
        mass_tend_in_dynamics = self.coords.horizontal.to_nodal(
            modal_tendency.tracers["qc"]
        )
        self.assertEqual(q_tend_in_dynamics.dtype, jnp.float32)
        np.testing.assert_allclose(q_tend_in_dynamics, dqdt, rtol=1e-6)
        np.testing.assert_allclose(
            mass_tend_in_dynamics, cloud_mass_tend, rtol=2e-6,
        )

    def test_moist_geopotential_uses_specific_humidity(self):
        base = self.model.dycore.to_physics_state(self.model.initial_state())
        moist = base.copy(
            specific_humidity=jnp.full_like(base.specific_humidity, 0.02),
        )
        dry = base.copy(specific_humidity=jnp.zeros_like(base.specific_humidity))

        moist_phi = dynamics_state_to_physics_state(
            physics_state_to_dynamics_state(moist, self.primitive),
            self.primitive,
        ).geopotential
        dry_phi = dynamics_state_to_physics_state(
            physics_state_to_dynamics_state(dry, self.primitive),
            self.primitive,
        ).geopotential

        # Moist virtual temperature increases the layer thickness above the
        # same surface geopotential; the bottom full level is above the surface
        # too, so every layer responds.
        self.assertTrue(jnp.all(moist_phi > dry_phi))


class TestHybridVirtualTemperatureContract(unittest.TestCase):
    """The hybrid primitive equations see physical q, not q/1000."""

    def test_virtual_temperature_adjustment_uses_kg_per_kg(self):
        from dinosaur.primitive_equations import compute_diagnostic_state_hybrid

        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        model = Model(
            coords=coords,
            physics=held_suarez_physics(),
            time_step=180.0,
        )
        physical = model.dycore.to_physics_state(model.initial_state())
        q = jnp.full_like(physical.specific_humidity, 0.01)
        modal = physics_state_to_dynamics_state(
            physical.copy(specific_humidity=q), model.dycore.primitive,
        )
        diagnostic = compute_diagnostic_state_hybrid(modal, coords)

        adjustment = model.dycore.primitive._virtual_temperature_adjustment(
            diagnostic
        )
        expected = 1.0 + (
            model.dycore.physics_specs.R_vapor / model.dycore.physics_specs.R
            - 1.0
        ) * q
        np.testing.assert_allclose(adjustment, expected, rtol=2e-6)

    def test_condensate_loads_the_virtual_temperature(self):
        """Tv = T(1 + (Rv/Rd-1)q - (qc+qi+qr+qs)), ECHAM6 dyn.f90::ztv."""
        from dinosaur.primitive_equations import compute_diagnostic_state_hybrid

        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        specs = {
            name: TracerSpec(name, units="kg/kg")
            for name in ("qc", "qi", "qr", "qs")
        }
        model = Model(
            coords=coords, physics=held_suarez_physics(), time_step=180.0,
        )
        model.dycore.tracer_specs = specs

        physical = model.dycore.to_physics_state(
            model.initial_state()
        )
        q = jnp.full_like(physical.specific_humidity, 0.01)
        condensate = {
            "qc": jnp.full_like(q, 3.0e-4),
            "qi": jnp.full_like(q, 1.0e-4),
            "qr": jnp.full_like(q, 5.0e-5),
            "qs": jnp.full_like(q, 2.0e-5),
        }
        modal = physics_state_to_dynamics_state(
            physical.copy(specific_humidity=q, tracers=condensate),
            model.dycore.primitive,
            tracer_specs=specs,
        )
        diagnostic = compute_diagnostic_state_hybrid(modal, coords)

        adjustment = model.dycore.primitive._virtual_temperature_adjustment(
            diagnostic
        )
        ratio = (
            model.dycore.physics_specs.R_vapor
            / model.dycore.physics_specs.R - 1.0
        )
        expected = 1.0 + ratio * q - sum(condensate.values())
        np.testing.assert_allclose(adjustment, expected, rtol=2e-6)
        # Every species must be represented: dropping rain/snow would leave a
        # 7e-5 gap, far outside this tolerance.
        self.assertEqual(
            model.dycore._cloud_keys, ("qc", "qi", "qr", "qs"),
        )

    def test_cloud_keys_follow_the_composition(self):
        """Only condensate the composition declares enters the coupling."""
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        model = Model(
            coords=coords, physics=held_suarez_physics(), time_step=180.0,
        )
        # Held-Suarez alone carries no condensate.
        self.assertIsNone(model.dycore._cloud_keys)

        # A rebuild that introduces condensate re-derives the keys, and a
        # non-condensate tracer is not swept in.
        model.dycore.tracer_specs = {
            "qc": TracerSpec("qc", units="kg/kg"),
            "dust": TracerSpec("dust", units="kg/kg"),
        }
        self.assertEqual(model.dycore._cloud_keys, ("qc",))


@pytest.mark.slow
class TestStateBridgeRoundTripSlow(unittest.TestCase):
    """End-to-end gridpoint↔modal round-trip on the default SPEEDY coords."""

    def test_round_trip_preserves_scalars_and_tracer_specs(self):
        # A real Model gives us a fully-wired ``primitive`` operator we can
        # hand to the standalone conversion helpers; the same code paths run
        # inside ``Model.run`` every step. Tolerance is set against spectral
        # round-trip noise (T31 nodes ≠ modal truncation; the inverse loses
        # the high-wavenumber tail). Winds are NOT round-trip-stable on the
        # sphere — a uniform u≠0 has zero vorticity and non-zero divergence,
        # but the spectral path goes through vor/div decomposition and loses
        # the constant component, so we don't assert on them here.
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        model = Model(coords=coords, time_step=720)
        primitive = model.dycore.primitive

        nodal_shape = coords.horizontal.nodal_shape
        kx = coords.vertical.layers
        state = model.dycore.to_physics_state(model._prepare_initial_dycore_state())
        seeded = state.copy(
            temperature=state.temperature + 1.0,
            tracers={
                "co2_vmr": jnp.full((kx,) + nodal_shape, 4.2e-4),
            },
        )
        tracer_specs = {"co2_vmr": TracerSpec(
            "co2_vmr", initial_value=4.2e-4, nondimensionalize=False,
        )}
        modal = physics_state_to_dynamics_state(seeded, primitive, tracer_specs=tracer_specs)
        recovered = dynamics_state_to_physics_state(modal, primitive, tracer_specs=tracer_specs)

        self.assertTrue(jnp.allclose(recovered.temperature, seeded.temperature, rtol=1e-4))
        # ``nondimensionalize=False`` tracers must pass through the round-trip
        # without the gram/kg scaling — this is the load-bearing branch for
        # GHG / number-concentration tracers under the v2 dycore protocol.
        self.assertTrue(jnp.allclose(
            recovered.tracers["co2_vmr"], seeded.tracers["co2_vmr"], rtol=1e-4,
        ))


@pytest.mark.slow
class TestHybridSurfacePressureRoundTrip(unittest.TestCase):
    """Hybrid-coordinate surface pressure must survive a PhysicsState round-trip.

    Regression for the asymmetry where ``dynamics_state_to_physics_state``
    divides hybrid ``sp`` by ``p0`` (exposing ``P_s/p0``) but the inverse logged
    that normalized value directly. For hybrid coords dinosaur stores
    ``log(P_s)`` (nondim Pa), so the inverse must multiply by ``p0`` first;
    without it surface pressure collapses by a factor of ~p0.
    """

    def test_hybrid_round_trip_preserves_surface_pressure(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        model = Model(
            coords=coords,
            physics=echam_physics(radiation_scheme="grey", checkpoint_terms=False),
            time_step=180.0,
        )
        primitive = model.dycore.primitive
        tracer_specs = {spec.name: spec for spec in model.physics.required_tracers()}

        state = model.dycore.to_physics_state(model._prepare_initial_dycore_state())
        # A clearly non-trivial normalized surface pressure (P_s/p0 ≈ 0.97).
        seeded = state.copy(
            normalized_surface_pressure=state.normalized_surface_pressure * 0.97,
        )

        modal = physics_state_to_dynamics_state(seeded, primitive, tracer_specs=tracer_specs)
        recovered = dynamics_state_to_physics_state(modal, primitive, tracer_specs=tracer_specs)

        # The load-bearing assertion: surface pressure survives the round-trip.
        # Pre-fix, ``recovered`` is smaller than ``seeded`` by ~p0 (~1e5).
        self.assertTrue(jnp.allclose(
            recovered.normalized_surface_pressure,
            seeded.normalized_surface_pressure,
            rtol=1e-4,
        ))


if __name__ == "__main__":
    unittest.main()
