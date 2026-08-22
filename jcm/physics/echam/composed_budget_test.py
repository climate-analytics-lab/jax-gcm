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

        # Layer masses at the column's ACTUAL surface pressure.
        # ``rce_initial_state`` builds the column at ``c.p0`` (1e5 Pa), and
        # the SCM never changes it. An earlier version of this test used
        # 101325 Pa here — a 1.3 % error in the mass measure that mimicked a
        # water leak proportional to whichever flux dominated the column
        # tendency. That artifact was chased across four commits as a real
        # conservation defect (the #696 "large-scale precipitation leak")
        # before a per-term ledger showed every term closing to round-off
        # and the composed residual collapsing 55x under the correct mass.
        ps = float(np.asarray(ic.normalized_surface_pressure) * c.p0)
        a = np.asarray(scm.vertical.a_boundaries)
        b = np.asarray(scm.vertical.b_boundaries)
        mass = np.diff(a + b * ps) / c.grav

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

        # Since the surface exchange became the bottom boundary row of the
        # vdiff implicit solve, the published evaporation IS the delivered
        # flux (the ECHAM ``pev_vdiff`` identity), so the raw-vs-damped
        # distinction is gone: evaporation == effective_evaporation.
        E_eff = np.asarray(
            ph["surface"].effective_evaporation,
        ).reshape(nsteps, -1)[:, 0]
        np.testing.assert_allclose(E_eff, E)

        col = (dq + dqc + dqi) @ mass  # (nsteps,)
        residual = col + P - E
        scale = np.maximum.reduce([np.abs(E), np.abs(P), np.abs(col), np.full_like(E, 1e-9)])
        rel = np.abs(residual) / scale

        # Sanity: the column is actually doing moist physics.
        self.assertGreater(float(E.max()), 0.0)
        self.assertGreater(float(P.max()) * 86400.0, 0.01,
                           "no precipitation over two days — vacuous budget")
        # Closure is judged on the SPUN-UP second day. Day 1 is excluded
        # because spin-up convection bursts trip the _DTDT_MAX stability cap,
        # which scales the tendencies but cannot scale the scalar precip
        # diagnostic — the cap's documented, inherent conservation break
        # (it does not fire in equilibrated columns).
        spd = int(round(86400.0 / scm.dt_seconds))
        rel_eq = rel[spd:]
        residual_eq = residual[spd:]

        # PRIMARY pin: the day-mean residual against the day-mean dominant
        # flux. This is the statement that actually means "the composed step
        # conserves water".
        #
        # History worth keeping: with the 101325-Pa mass error above, this
        # metric wandered 0.71 % - 2.20 % across the Tiedtke cloud-base work
        # and correlated 0.99 with large-scale precipitation, which was filed
        # as a stratiform conservation defect (#696). The correlation was the
        # artifact's signature, not a leak's: a mis-scaled mass measure makes
        # the residual a fixed fraction of the net column tendency, which
        # tracks whichever flux is bursty in the current regime. With the
        # correct mass the per-term ledger closes vdiff to its delivered
        # evaporation, convection to its precipitation, and the 1M to
        # round-off (1e-13); the composed day-2 mean residual measured
        # 6.1e-9 kg/m2/s = 0.03 % of E. The bound is 0.5 %: an order of
        # magnitude of margin for regime shifts, and an order of magnitude
        # tighter than anything a genuine flux-proportional leak would pass.
        mean_flux = float(np.maximum(np.abs(E[spd:]).mean(),
                                     np.abs(P[spd:]).mean()))
        mean_rel = float(np.abs(residual_eq).mean()) / mean_flux
        self.assertLess(
            mean_rel, 0.005,
            f"composed water budget leaks {mean_rel:.2%} of the day-mean "
            f"flux over the equilibrated day "
            f"(mean |residual| = {np.abs(residual_eq).mean():.3e} kg/m2/s)",
        )

        # SECONDARY pin: no single step may go grossly open. Its denominator
        # is instantaneous, so it stays looser than the day-mean pin — a
        # convective lull collapses the scale and inflates a small absolute
        # residual into a large ratio. The pre-fix schemes this test was
        # written against fail here at the several-hundred-percent level
        # (water created at the precipitation rate), which is what it
        # catches. Sanity that it is not vacuous: dropping P_conv from the
        # sink side of the ledger scores 174 %, so convective precip really
        # is debited.
        worst = int(np.argmax(rel_eq))
        self.assertLess(
            float(rel_eq.max()), 0.02,
            f"composed water budget open by {residual_eq[worst]:.3e} "
            f"kg/m2/s at equilibrated step {worst + spd} "
            f"(E={E[worst + spd]:.3e}, P={P[worst + spd]:.3e})",
        )


if __name__ == "__main__":
    unittest.main()
