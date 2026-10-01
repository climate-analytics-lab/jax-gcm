"""The BC-lifetime claim, measured in one column (§5 item 5).

The design record for the modal core says what the missing process costs:
without ageing, primary carbon never leaves the ``can_activate=False``
primary-carbon mode, so it is removed by dry deposition, settling and
below-cloud scavenging alone, and the modal core's ageing is what opens the
in-cloud path. This measures the two halves of that claim separately,
because one column cannot measure a global lifetime: no transport brings BC
back, and a tropical convective column rains far more than the global mean.

The column is the release validation's (``jcm.rce.jam_scavenging_column``
under ``convergent_initial_physics_data``, the stack
``tools/release_validation/scm_check.py`` runs): a prescribed warm tropical
sounding re-imposed every step so ECHAM's deep plume fires repeatedly, the
JAM tracers evolving through vertical diffusion, convective transport and
scavenging, the microphysics core and wet deposition. Three runs:

1. black carbon seeded in the **primary-carbon** mode, placeholder core — the
   no-ageing lifetime, removal by the pathways open to an insoluble mode;
2. the same mass seeded in the **accumulation** mode, placeholder core — the
   lifetime once BC is where it can activate and be scavenged in cloud;
3. BC in primary carbon, **modal** core, under a sustained sulfuric-acid
   vapour (held at a polluted-daytime 1e-11 kg/kg through the column each
   step — the coating supply a global run gets from its chemistry, which
   this prescribed column has no oxidants for) — ageing in action.

Asserted: the accumulation-mode lifetime is shorter than the primary-carbon
one (the in-cloud path removes BC the insoluble mode escapes); the modal
core puts BC into accumulation where the placeholder puts none; and the aged
column's lifetime is shorter than the no-ageing one. The accumulation
fraction at the end is a small steady balance — aged in, scavenged out at
the faster rate — so it is checked for existence, not size; the lifetime is
the measurement. The measured numbers are recorded in the design doc, not
pinned here.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c


def _run(core, days, *, seed_mode="pcm", sustained_h2so4=None, dt=900.0, nlev=47):
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.physics.echam.testing import idealized_echam_physics
    from jcm.rce import convergent_initial_physics_data, jam_scavenging_column
    from jcm.single_column_model import SingleColumnModel

    vertical = get_echam_levels(nlev)
    physics = idealized_echam_physics(cloud_scheme="2m", aerosol_module="jam",
                                      jam_microphysics=core)
    state, seed, pressure = jam_scavenging_column(vertical, physics)
    nlev = state.temperature.shape[0]
    boundary_layer = jnp.zeros(nlev).at[-5:].set(1.0)
    seed = dict(seed)
    seed[f"m_bc_{seed_mode}"] = boundary_layer * 1.0e-9 + 1e-30
    if seed_mode == "acc":
        seed["n_acc"] = seed["n_acc"] + boundary_layer * 2.0e8
    free = ()
    if sustained_h2so4 is not None:
        # Held at the prescribed profile every step rather than depleted:
        # every other tracer evolves (``free_evolve``), this one is the
        # column's stand-in for a chemistry source.
        # Through the whole column, not the boundary layer alone: this
        # column lofts its boundary-layer aerosol within hours, and BC that
        # has left the layer with the vapour in it would never age (in the
        # full physics the chemistry supplies acid aloft as well).
        state = state.copy(tracers={**state.tracers,
                                    "g_h2so4": jnp.full(nlev, sustained_h2so4)})
        seed["g_h2so4"] = state.tracers["g_h2so4"]
        free = tuple(name for name in seed if name != "g_h2so4")
    scm = SingleColumnModel(physics=physics, vertical=vertical, lat_deg=0.0,
                            lon_deg=150.0, dt_seconds=dt,
                            apply_tracer_tendencies=sustained_h2so4 is None,
                            free_evolve=free)
    n_steps = int(days * 86400 / dt)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_steps,) + jnp.shape(x)), state)
    preds = scm.run(states, initial_tracers=seed,
                    initial_physics_data=convergent_initial_physics_data(scm, state),
                    times=jnp.arange(n_steps) * dt / 86400.0)
    # Column burdens [kg/m²]: Σ q Δp / g, interstitial plus the cloud-borne carry.
    a_half = jnp.asarray(vertical.a_boundaries)
    b_half = jnp.asarray(vertical.b_boundaries)
    dp = np.abs(np.diff(np.asarray(a_half + b_half * c.p0)))

    def burden(array):
        # Tracer histories are (n_times, nlev); the carry keeps the SCM's
        # (nlev, 1) column, so it comes back (n_times, nlev, 1).
        return (np.asarray(array).reshape(n_steps, -1) * dp[None, :]).sum(axis=1) / c.grav

    carry = preds.physics_data.get("_jam_cloud_borne", {})
    pcm = burden(preds.tracer_states["m_bc_pcm"])
    acc = burden(preds.tracer_states["m_bc_acc"])
    if "mc_bc_acc" in carry:
        acc = acc + burden(carry["mc_bc_acc"])
    return np.asarray(preds.times), pcm, acc


def _lifetime_days(times, burden, start_day):
    mask = times >= start_day
    slope, _ = np.polyfit(times[mask], np.log(np.maximum(burden[mask], 1e-300)), 1)
    return -1.0 / slope


@pytest.mark.slow
class BlackCarbonLifetimeTest(unittest.TestCase):

    def test_ageing_opens_the_in_cloud_removal_path(self):
        days = 6.0
        t, pcm_only, acc_leak = _run("placeholder", days, seed_mode="pcm")
        _, pcm_seeded_in_acc, acc_only = _run("placeholder", days, seed_mode="acc")
        _, pcm_aged, acc_aged = _run("modal", days, seed_mode="pcm", sustained_h2so4=1.0e-11)
        insoluble = pcm_only + acc_leak
        soluble = pcm_seeded_in_acc + acc_only
        aged = pcm_aged + acc_aged
        for name, total in (("pcm/placeholder", insoluble), ("acc/placeholder", soluble),
                            ("pcm/modal", aged)):
            self.assertTrue(np.all(np.isfinite(total)), name)
            self.assertGreater(total[0], 0.0, name)
        tau_insoluble = _lifetime_days(t, insoluble, start_day=1.0)
        tau_soluble = _lifetime_days(t, soluble, start_day=1.0)
        tau_aged = _lifetime_days(t, aged, start_day=1.0)
        print(f"\nBC e-folding lifetime, days 1-{days:.0f}: in primary carbon {tau_insoluble:.1f} d, "
              f"in accumulation {tau_soluble:.1f} d, primary carbon + ageing {tau_aged:.1f} d; "
              f"accumulation fraction at day {days:.0f}: placeholder "
              f"{acc_leak[-1] / insoluble[-1]:.3f}, modal {acc_aged[-1] / aged[-1]:.2f}")
        # The placeholder moves nothing out of primary carbon.
        self.assertLess(acc_leak[-1] / insoluble[-1], 0.01)
        # Where BC can activate it is removed faster: the in-cloud path.
        self.assertLess(tau_soluble, 0.8 * tau_insoluble)
        # Ageing puts BC on that path — the end fraction is the steady
        # balance of ageing in and the faster removal out, so small ...
        self.assertGreater(acc_aged[-1] / aged[-1], 0.03)
        # ... and the column's BC lifetime is shorter for it (measured
        # 2026-10-02: 1.5 d against 2.0 d, the in-cloud limit being 0.9 d).
        self.assertLess(tau_aged, 0.9 * tau_insoluble)


if __name__ == "__main__":
    unittest.main()
