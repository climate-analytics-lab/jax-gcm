"""Cloud-radiative-effect guard: seeded clouds must reach the SW solver.

Two separate production regressions silently zeroed the cloud radiative
effect while every finiteness/equilibration test stayed green:

* the JW initial-condition helper overwrote ``state.tracers`` and dropped
  the cloud condensate tracers, so radiation saw a cloud-free atmosphere
  (planetary albedo pinned at the clear-sky ~0.115), and
* a Sundqvist refactor turned the scheme into a pure cloud-fraction
  diagnostic, leaving condensate ≈ 0 in the radiation path.

Nothing in the suite asserted that cloud condensate present in the model
state makes the planet brighter, so both shipped. This test pins the whole
chain: a saturated, condensate-carrying blob seeded through the public
``PhysicsState`` initialization path (the same bridge the JW regression
broke) must survive the round-trip into the dycore state, produce a cloud
fraction, and reflect measurably more TOA shortwave than the clear-sky
twin within two hours of model time.

(A cold-start T21 aquaplanet produces exactly zero condensate for at
least a day — measured — so the seed is what makes this test cheap.)
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.slow
class TestCloudRadiativeEffectActive(unittest.TestCase):
    def test_seeded_cloud_brightens_the_planet(self):
        from jcm.model import Model
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        nlev = 20
        coords = get_coords(np.linspace(0, 1, nlev + 1), spectral_truncation=21)
        model = Model(
            coords=coords, time_step=30,
            terrain=TerrainData.aquaplanet(coords),
            physics=echam_physics(radiation_scheme="rrtmgp"),
        )

        # A horizontally uniform atmosphere with a saturated, cloudy slab at
        # ~600-800 hPa: RH ≈ 0.95 (above the Sundqvist critical RH, so the
        # scheme diagnoses a cloud fraction there) carrying 3e-5 kg/kg of
        # liquid (LWP ~ tens of g/m² — a solidly reflective cloud).
        nlon, nlat = coords.horizontal.nodal_shape
        shape = (nlev, nlon, nlat)
        T_col = jnp.linspace(220.0, 295.0, nlev)
        p_col = jnp.linspace(5e3, 1.0e5, nlev)  # approx sigma-level pressures
        qsat_col = jnp.asarray([
            float(saturation_specific_humidity(p_col[k], T_col[k]))
            for k in range(nlev)
        ])
        cloud_levels = (jnp.arange(nlev) >= 12) & (jnp.arange(nlev) <= 15)
        q_col = jnp.where(cloud_levels, 0.95 * qsat_col, 0.2 * qsat_col)
        qc_col = jnp.where(cloud_levels, 3e-5, 0.0)

        broadcast = lambda col: jnp.broadcast_to(  # noqa: E731
            col[:, None, None], shape,
        )
        initial = PhysicsState(
            u_wind=jnp.zeros(shape),
            v_wind=jnp.zeros(shape),
            temperature=broadcast(T_col),
            specific_humidity=broadcast(q_col),
            geopotential=broadcast(jnp.linspace(30000.0, 0.0, nlev)),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={"qc": broadcast(qc_col), "qi": jnp.zeros(shape)},
        )

        model._final_dycore_state = model._prepare_initial_dycore_state(
            physics_state=initial,
        )
        # One 30-min step: radiation runs on the seeded state, and the
        # unsupported blob hasn't yet rained/evaporated away (measured: it
        # decays on a ~30-60 min timescale with no dynamical resupply, which
        # is fine — the guard is about the coupling, not cloud maintenance).
        preds = model.resume(save_interval=1 / 48.0, total_time=1 / 48.0)

        # (1) The seeded condensate survived the PhysicsState → dycore →
        # physics round-trip (the JW-regression failure mode dropped the
        # tracer dict at initialization, which reads back as exactly zero).
        # Note the saved dynamics tracers are in the dycore's stored
        # convention (~1e-3 × the physics kg/kg value); the measured healthy
        # value here is ~1e-7, so the threshold is two decades below it.
        qc = np.asarray(preds.dynamics.tracers["qc"])
        self.assertGreater(
            float(np.max(qc)), 1e-9,
            "seeded cloud water vanished — tracers dropped at init",
        )

        # (2) The cloud is radiatively active: on the dayside the all-sky
        # column reflects meaningfully more SW than its clear-sky twin
        # (SW CRE < 0). With condensate present but decoupled from the
        # optics (the Sundqvist-diagnostic failure mode), the two TOA
        # fields are identical and this fails.
        rad = preds.physics["radiation"]
        sw_up = np.asarray(rad.toa_sw_up)
        sw_up_clear = np.asarray(rad.toa_sw_up_clear)
        self.assertTrue(np.all(np.isfinite(sw_up)))
        max_brightening = float(np.max(sw_up - sw_up_clear))
        self.assertGreater(
            max_brightening, 1.0,  # W/m²
            "all-sky and clear-sky TOA SW are indistinguishable — clouds "
            "exist but never reach the radiation solver",
        )


if __name__ == "__main__":
    unittest.main()
