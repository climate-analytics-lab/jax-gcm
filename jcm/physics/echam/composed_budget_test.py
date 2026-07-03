"""Composed-physics column water closure (deferred from the #549 review).

Per-scheme conservation tests exist for convection (PR #550) and the 1M/2M
microphysics (this PR), but nothing previously closed the budget across the
COMPOSED `echam_physics()` step — exactly the loop where past regressions
leaked water (the imp_moist/1M chain in the project's moisture-loop
diagnosis, convection creating water at the precipitation rate, the 2M
qr/qs clip destroying mass).

The single-column model applies the summed physics tendencies, so the
budget is checked on the TENDENCIES of one step (the SCM's prescribed
tracer base makes state differences unusable for tracers):

    Σ (dq/dt + dqc/dt + dqi/dt)·Δp/g  +  P_conv + P_rain + P_snow − E ≈ 0
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c


@pytest.mark.slow
class TestComposedColumnWaterClosure(unittest.TestCase):
    @pytest.mark.xfail(
        strict=False,
        reason="Composed budget opens by ~P at every precipitating step "
               "(closes to ~1% when P=0): the convection→cloud detrained-"
               "condensate forwarding double-counts against the tracer "
               "state — issue #553. The per-scheme ledgers each close "
               "(#550, PR5); this test pins the composed target and flips "
               "to passing with the coupling fix.",
    )
    def test_full_echam_step_water_budget(self):
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.radiation_types import RadiationParameters
        from jcm.rce import rce_column, rce_initial_state, run_rce

        nlev = 40
        physics = echam_physics(
            radiation_scheme="grey",
            radiation=RadiationParameters.default(solar_constant=420.0),
        )
        scm = rce_column(
            sst=300.0, relative_humidity=0.7, lat_deg=0.0, nlev=nlev,
            dt_seconds=900.0, physics=physics, interactive_humidity=True,
        )
        ic = rce_initial_state(
            scm.vertical, sst=300.0, relative_humidity=0.7,
        ).copy(u_wind=jnp.full(nlev, 5.0))
        # Long enough that convection and large-scale rain have both fired;
        # the budget is evaluated at every saved step.
        preds = run_rce(scm, ic, n_days=2.0)

        a = np.asarray(scm.vertical.a_boundaries)
        b = np.asarray(scm.vertical.b_boundaries)
        mass = np.diff(a + b * 101325.0) / c.grav

        t = preds.tendencies
        nsteps = np.asarray(t.specific_humidity).shape[0]
        dq = np.asarray(t.specific_humidity).reshape(nsteps, nlev)
        dqc = np.asarray(t.tracers["qc"]).reshape(nsteps, nlev)
        dqi = np.asarray(t.tracers["qi"]).reshape(nsteps, nlev)

        ph = preds.physics_data
        P = (
            np.asarray(ph["clouds"].precip_rain).reshape(nsteps, -1)[:, 0]
            + np.asarray(ph["clouds"].precip_snow).reshape(nsteps, -1)[:, 0]
            + np.asarray(ph["convection"].precip_conv).reshape(nsteps, -1)[:, 0]
        )
        E = np.asarray(ph["surface"].evaporation).reshape(nsteps, -1)[:, 0]

        col = (dq + dqc + dqi) @ mass  # (nsteps,)
        residual = col + P - E
        scale = np.maximum.reduce([np.abs(E), np.abs(P), np.abs(col), np.full_like(E, 1e-9)])
        rel = np.abs(residual) / scale

        # Sanity: the column is actually doing moist physics.
        self.assertGreater(float(E.max()), 0.0)
        self.assertGreater(float(P.max()) * 86400.0, 0.01,
                           "no precipitation over two days — vacuous budget")
        # Closure on the SPUN-UP second day: 5 % of the dominant flux at
        # every step (measured ~1 % on the fixed physics; the pre-fix
        # schemes fail at the several-hundred-percent level — water
        # created at the precipitation rate). Day 1 is excluded because
        # spin-up convection bursts trip the _DTDT_MAX stability cap,
        # which scales the tendencies but cannot scale the scalar precip
        # diagnostic — the cap's documented, inherent conservation break
        # (it does not fire in equilibrated columns).
        spd = int(round(86400.0 / scm.dt_seconds))
        rel_eq = rel[spd:]
        residual_eq = residual[spd:]
        worst = int(np.argmax(rel_eq))
        self.assertLess(
            float(rel_eq.max()), 0.05,
            f"composed water budget open by {residual_eq[worst]:.3e} "
            f"kg/m2/s at equilibrated step {worst + spd} "
            f"(E={E[worst + spd]:.3e}, P={P[worst + spd]:.3e})",
        )


if __name__ == "__main__":
    unittest.main()
