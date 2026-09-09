"""The removal chain composed: sequential, bounded, and ledger-consistent.

Sedimentation, dry deposition and wet scavenging each cap their own removal
at 100 %, so before operator splitting their SUM could exceed the available
mass. These tests drive the three terms exactly as ``ComposablePhysics``
does — step-start state plus a running tendency published as
``_tendency_run`` — on a raining marine surface cell holding coarse sea
salt, the case that lost 164 % of its mass.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
from jcm.physics.aerosol.jam.drydep.drydep_term import SlinnDryDeposition
from jcm.physics.aerosol.jam.sedimentation.sedi_term import StokesSedimentation
from jcm.physics.aerosol.jam.wetdep.wetdep_term import WetScavenging

DT = 1800.0


def _column(nlev=4, ncols=2, precip=1.0e-3, r_wet=1.78e-6, dz=200.0):
    """Build a raining, near-saturated marine column of coarse aerosol."""
    from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
    from jcm.physics.aerosol.jam.jam_state import JamAerosolState
    from jcm.physics.clouds.cloud_data import CloudData
    from jcm.physics_interface import PhysicsState

    n_modes = MAM4_SPEC.n_modes()
    shape = (n_modes, nlev, ncols)
    aer = JamAerosolState(
        r_dry=jnp.full(shape, 1.0e-6),
        r_wet=jnp.full(shape, r_wet),
        rho=jnp.full(shape, 1160.0),      # wet sea salt at ~80 % RH
        kappa=jnp.full(shape, 1.16),
        mass=jnp.full(shape, 1.0e-9),
        number=jnp.full(shape, 1.0e8),
    )
    tracers, carry = {}, {}
    for mode in MAM4_SPEC.modes:
        tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
        carry[number_name(mode.short, cloud_borne=True)] = jnp.full(
            (nlev, ncols), 1.0e8)
        for sp in mode.species:
            tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1.0e-9)
            carry[mass_name(sp, mode.short, cloud_borne=True)] = jnp.full(
                (nlev, ncols), 1.0e-9)
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 285.0),
        tracers=tracers,
    )
    dm = 1.0 * dz
    form = jnp.full((nlev, ncols), precip / (nlev * dm))
    clouds = CloudData.zeros((ncols,), nlev).copy(
        cloud_fraction=jnp.full((nlev, ncols), 0.3),
        qc=jnp.full((nlev, ncols), 1.0e-3),
        precip_rain=jnp.full((ncols,), precip),
        precip_formation_rate=form,
        rain_flux=jnp.cumsum(form * dm, axis=0),
        incloud_liquid=jnp.full((nlev, ncols), 1.0e-3 / 0.3),
        incloud_rain_formation=form / 0.3,
        process_cloud_fraction=jnp.full((nlev, ncols), 0.3),
    )
    diagnostics = {
        CARRY_KEY: carry,
        "_jam_state": aer,
        "activated_fraction": jnp.full((nlev, ncols), 0.7),
        "air_density": jnp.full((nlev, ncols), 1.0),
        "layer_thickness": jnp.full((nlev, ncols), dz),
        "pressure_full": jnp.full((nlev, ncols), 1.0e5),
        "clouds": clouds,
        "_dt_seconds": DT,
    }
    return state, diagnostics, MAM4_SPEC, mass_name, number_name


def _run_chain(state, diagnostics, sequential=True):
    """Run the removal terms as ``ComposablePhysics`` does.

    Returns ``(summed tendencies, final diagnostics)``. With
    ``sequential=False`` the running tendency is withheld, reproducing the
    pre-fix behaviour where every term saw the step-start state.
    """
    acc = {name: jnp.zeros_like(q) for name, q in state.tracers.items()}
    diagnostics = dict(diagnostics)
    for term in (StokesSedimentation(), SlinnDryDeposition(), WetScavenging()):
        if sequential:
            diagnostics["_tendency_run"] = {"tracers": acc}
        tend, diagnostics = term(state, diagnostics, None, None)
        acc = {
            name: value + tend.tracers.get(name, 0.0)
            for name, value in acc.items()
        }
    return acc, diagnostics


class SequentialRemovalTest(unittest.TestCase):

    def test_total_removal_never_exceeds_available_mass(self):
        state, diagnostics, spec, mass_name, number_name = _column()
        acc, _ = _run_chain(state, diagnostics)
        for name, q0 in state.tracers.items():
            q1 = np.asarray(q0 + DT * acc[name])
            self.assertGreaterEqual(
                float(q1.min()), 0.0,
                f"{name} went negative: the removal sum exceeded the mass")

    def test_step_start_reads_would_overdraw(self):
        # The defect this replaces: with every term reading the step-start
        # state the three independently-capped sinks sum past 100 % — here
        # to 124 % — and drive the tracer negative. Sequential reads keep
        # the same cell non-negative.
        kwargs = dict(precip=5.0e-2, r_wet=5.0e-6, dz=60.0)
        state, diagnostics, spec, mass_name, _ = _column(**kwargs)
        key = mass_name("ss", "cor")
        fractions = sum(
            -float((DT * term(state, diagnostics, None, None)[0]
                    .tracers[key] / state.tracers[key])[-1, 0])
            for term in (StokesSedimentation(), SlinnDryDeposition(),
                         WetScavenging())
        )
        self.assertGreater(fractions, 1.0)

        acc, _ = _run_chain(state, diagnostics, sequential=False)
        self.assertLess(
            float(np.asarray(state.tracers[key] + DT * acc[key]).min()), 0.0)
        acc, _ = _run_chain(state, diagnostics, sequential=True)
        self.assertGreaterEqual(
            float(np.asarray(state.tracers[key] + DT * acc[key]).min()), 0.0)

    def test_pathway_diagnostics_sum_to_the_mass_change(self):
        # dry_* (settling + surface deposition) and wet_* (scavenging net of
        # re-evaporation) must together account for exactly the interstitial
        # + cloud-borne mass the chain removed.
        state, diagnostics, spec, mass_name, _ = _column()
        acc, out = _run_chain(state, diagnostics)
        dm = np.asarray(diagnostics["air_density"]
                        * diagnostics["layer_thickness"])
        for species in ("ss", "du", "so4", "bc"):
            removed = 0.0
            for mode in spec.modes:
                if species not in mode.species:
                    continue
                name = mass_name(species, mode.short)
                removed -= float(np.sum(np.asarray(acc[name]) * dm))
                cb = mass_name(species, mode.short, cloud_borne=True)
                delta = (np.asarray(out[CARRY_KEY][cb])
                         - np.asarray(diagnostics[CARRY_KEY][cb]))
                removed -= float(np.sum(delta * dm)) / DT
            ledger = (float(np.sum(np.asarray(out[f"dry_{species}"])))
                      + float(np.sum(np.asarray(out[f"wet_{species}"]))))
            self.assertGreater(ledger, 0.0, species)
            self.assertAlmostEqual(ledger / removed, 1.0, places=5, msg=species)

    def test_dry_ledger_includes_surface_deposition(self):
        # Regression: ``dry_*`` used to carry sedimentation alone.
        state, diagnostics, _, _, _ = _column()
        _, sedi_only = StokesSedimentation()(state, diagnostics, None, None)
        acc, both = _run_chain(state, diagnostics)
        self.assertGreater(
            float(np.sum(np.asarray(both["dry_ss"]))),
            float(np.sum(np.asarray(sedi_only["dry_ss"]))) * 1.05,
        )

    def test_single_column_and_block_agree(self):
        # Broadcasting-native: one column and a block of identical columns
        # must give identical per-column answers.
        state1, diag1, _, mass_name, _ = _column(ncols=1)
        state4, diag4, _, _, _ = _column(ncols=4)
        acc1, _ = _run_chain(state1, diag1)
        acc4, _ = _run_chain(state4, diag4)
        key = mass_name("ss", "cor")
        np.testing.assert_allclose(
            np.asarray(acc4[key]),
            np.broadcast_to(np.asarray(acc1[key]), acc4[key].shape),
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
