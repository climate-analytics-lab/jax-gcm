"""Phase 5 tests: in-cloud + below-cloud scavenging and the term."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.wetdep.impaction import (
    bcscavcoef,
    build_impaction_table,
    table_log_coefficients,
)
from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
    below_cloud_rate,
    conv_below_cloud_rate,
    conv_in_cloud_rate,
    conv_precip_cover,
    reinjection_budget,
)


class ScavengingFunctionTest(unittest.TestCase):
    def test_reinjection_budget_conserves_the_column(self):
        # Aerosol scavenged at the top rides the precip down; a 50%-evap
        # layer releases half, a full-evap layer releases the rest, and
        # nothing reaches the surface. sum(scavenged - reinjected) must
        # equal the surface flux EXACTLY at every column.
        none = jnp.zeros((1, 3, 1))
        scavenged = none.at[0, 0, 0].set(2.0)
        evap_frac = jnp.array([[0.0], [0.5], [1.0]])
        reinjected, surface = reinjection_budget(none, scavenged, evap_frac)
        np.testing.assert_allclose(np.asarray(reinjected[0, :, 0]),
                                   [0.0, 1.0, 1.0])
        np.testing.assert_allclose(np.asarray(surface), 0.0)
        # Partial evap: the un-released remainder deposits.
        evap_frac = jnp.array([[0.0], [0.25], [0.0]])
        reinjected, surface = reinjection_budget(none, scavenged, evap_frac)
        np.testing.assert_allclose(np.asarray(surface[0, 0]), 1.5)
        np.testing.assert_allclose(
            float(jnp.sum(scavenged - reinjected)), float(surface[0, 0]),
        )

    def test_virga_releases_same_layer_impaction(self):
        # Aerosol impacted out of the INCOMING precip within a fully-
        # evaporating layer must be released there too, not ride a
        # terminated carrier to the surface through dry air (impaction
        # joins the ledger before the release, as in HAMMOZ).
        none = jnp.zeros((1, 3, 1))
        impacted = none.at[0, 2, 0].set(1.0)
        evap_frac = jnp.zeros((3, 1)).at[2].set(1.0)
        reinjected, surface = reinjection_budget(impacted, none, evap_frac)
        np.testing.assert_allclose(np.asarray(reinjected[0, 2, 0]), 1.0)
        np.testing.assert_allclose(np.asarray(surface), 0.0)

    def test_formation_in_full_evap_layer_still_deposits(self):
        # Codex P1 on #612: the incoming carrier's evaporation cannot touch
        # precip NEWLY FORMED in the same layer — both cloud schemes cap
        # evaporation by the incoming flux and add formation after. Aerosol
        # scavenged into that new precip must continue downward, not be
        # released by an evap fraction that belongs to the old carrier.
        none = jnp.zeros((1, 3, 1))
        formed = none.at[0, 1, 0].set(1.0)
        evap_frac = jnp.zeros((3, 1)).at[1].set(1.0)
        reinjected, surface = reinjection_budget(none, formed, evap_frac)
        np.testing.assert_allclose(np.asarray(reinjected), 0.0)
        np.testing.assert_allclose(np.asarray(surface[0, 0]), 1.0)

    def test_below_cloud_size_dependence(self):
        # Λ = sol_factb·Λ₁·R: coarse-mode aerosol sits above the
        # Greenfield gap and is collected far more efficiently.
        precip = jnp.full((1, 1), 1.0e-4)
        params = WetDepParameters.default()
        table = build_impaction_table(0.11e-6, 1.8, 1770.0)
        coarse_table = build_impaction_table(2.0e-6, 1.8, 2600.0)
        _, accum_coef = bcscavcoef(
            jnp.full((1, 1), 0.055e-6), table.dgnum,
            *table_log_coefficients(table, 60.0, 1.0))
        _, coarse_coef = bcscavcoef(
            jnp.full((1, 1), 1.0e-6), coarse_table.dgnum,
            *table_log_coefficients(coarse_table, 60.0, 1.0))
        accum = below_cloud_rate(precip, accum_coef, params)
        coarse = below_cloud_rate(precip, coarse_coef, params)
        self.assertGreater(float(coarse[0, 0]), float(accum[0, 0]))

    def test_no_precip_no_below_cloud(self):
        params = WetDepParameters.default()
        table = build_impaction_table(2.0e-6, 1.8, 2600.0)
        _, coef = bcscavcoef(
            jnp.full((1, 1), 1.0e-6), table.dgnum,
            *table_log_coefficients(table, 60.0, 1.0))
        rate = below_cloud_rate(jnp.zeros((1, 1)), coef, params)
        self.assertAlmostEqual(float(rate[0, 0]), 0.0)

    def test_conv_in_cloud_hammoz_form(self):
        # rate = ratio * (formation/(rho*dz)) / condensate on cloudy
        # layers; exactly zero where the updraft carries no condensate.
        params = WetDepParameters.default()
        form = jnp.array([[0.0], [1.0e-4], [0.0]])        # kg/m²/s
        qcond = jnp.array([[0.0], [1.0e-3], [1.0e-3]])    # kg/kg
        rho = jnp.ones((3, 1))
        dz = jnp.full((3, 1), 500.0)
        rate = conv_in_cloud_rate(form, qcond, rho, dz, params)
        self.assertAlmostEqual(float(rate[0, 0]), 0.0)    # no condensate
        expected = 0.99 * (1.0e-4 / 500.0) / 1.0e-3
        self.assertAlmostEqual(float(rate[1, 0]), expected, places=8)
        self.assertAlmostEqual(float(rate[2, 0]), 0.0)    # no formation
        none = conv_in_cloud_rate(jnp.zeros((3, 1)), qcond, rho, dz, params)
        self.assertAlmostEqual(float(jnp.abs(none).max()), 0.0)

    def test_conv_precip_cover_is_the_updraft_area(self):
        # HAMMOZ prep_wetdep_hydro: the precipitating fraction is the
        # updraft area mfu/(rho*w_u), with ECHAM cuflx's sub-cloud taper —
        # linear in the air mass below the interface, squared for
        # mid-level convection — and nothing at all without a plume.
        nlev, ncols = 4, 3
        mfu = jnp.array([0.0, 0.2, 0.2, 0.0])[:, None] * jnp.array([1.0, 1.0, 0.0])
        ktype = jnp.array([1, 3, 0], dtype=jnp.int32)
        layer_mass = jnp.full((nlev, ncols), 200.0)
        rho = jnp.ones((nlev, ncols))
        cover = np.asarray(conv_precip_cover(mfu, ktype, layer_mass, rho, 2.0))
        # In-cloud levels: 0.2 / (1 * 2) = 0.1. Sub-cloud level 3 sits under
        # base level 2 with half the air mass below its top interface
        # (200 of 400), so zzp = 0.5 — squared to 0.25 for ktype 3.
        np.testing.assert_allclose(cover[:, 0], [0.0, 0.1, 0.1, 0.05], rtol=1e-6)
        np.testing.assert_allclose(cover[:, 1], [0.0, 0.1, 0.1, 0.025], rtol=1e-6)
        np.testing.assert_array_equal(cover[:, 2], 0.0)
        # An updraft area above the whole box is clipped to the box.
        huge = jnp.array([0.0, 10.0, 10.0, 10.0])[:, None] * jnp.ones((1, ncols))
        np.testing.assert_array_equal(
            np.asarray(conv_precip_cover(huge, ktype, layer_mass, rho, 2.0))[1:],
            1.0,
        )
        # Broadcasting-native: a single column agrees with the block.
        single = conv_precip_cover(mfu[:, 0], ktype[0], layer_mass[:, 0],
                                   rho[:, 0], 2.0)
        np.testing.assert_allclose(np.asarray(single), cover[:, 0], rtol=1e-6)

    def test_conv_below_cloud_rate_removes_the_covered_fraction(self):
        # HAMMOZ ham_wetdep: the layer loses cover * (1 - exp(-Lambda*dt)),
        # so however hard it rains the step can take at most the covered
        # fraction; for small Lambda*dt the rate is simply cover * Lambda.
        params = WetDepParameters.default()
        dt = 1800.0
        cover = jnp.full((2, 1), 0.3)
        coef = jnp.ones((2, 1))
        saturating = conv_below_cloud_rate(
            jnp.full((2, 1), 1.0), cover, coef, params, dt)   # Lambda*dt = 180
        removed = -np.expm1(-np.asarray(saturating) * dt)
        np.testing.assert_allclose(removed, 0.3, rtol=1e-6)
        weak_flux = jnp.full((2, 1), 1.0e-6)
        weak = conv_below_cloud_rate(weak_flux, cover, coef, params, dt)
        expected = 0.3 * np.asarray(below_cloud_rate(weak_flux, coef, params))
        np.testing.assert_allclose(np.asarray(weak), expected, rtol=1e-3)
        none = conv_below_cloud_rate(
            jnp.full((2, 1), 1.0), jnp.zeros((2, 1)), coef, params, dt)
        np.testing.assert_array_equal(np.asarray(none), 0.0)


class WetDepTermTest(unittest.TestCase):
    def _setup(self, nlev=4, ncols=2, precip=1.0e-4):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
        from jcm.physics.aerosol.jam.jam_state import JamAerosolState
        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics_interface import PhysicsState

        n_modes = MAM4_SPEC.n_modes()
        shape = (n_modes, nlev, ncols)
        aer = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6),
            r_wet=jnp.full(shape, 0.2e-6),
            rho=jnp.full(shape, 1800.0),
            kappa=jnp.full(shape, 0.5),
            mass=jnp.full(shape, 1e-9),
            number=jnp.full(shape, 1.0e8),
        )
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY

        tracers = {}
        carry = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
            carry[number_name(mode.short, cloud_borne=True)] = jnp.full(
                (nlev, ncols), 1.0e8
            )
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full(
                    (nlev, ncols), 1e-9
                )
                carry[mass_name(sp, mode.short, cloud_borne=True)] = (
                    jnp.full((nlev, ncols), 1e-9)
                )
        tracers["qni"] = jnp.zeros((nlev, ncols))   # the 2M crystal number
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 275.0),
            tracers=tracers,
        )
        # Uniform formation through the column whose integral equals the
        # surface precip (dm = rho * dz = 200 kg/m² per layer here), with
        # the matching cumulative rain-flux profile — the per-level fields
        # the cloud schemes now expose (#499).
        dm = 1.0 * 200.0
        form = jnp.full((nlev, ncols), precip / (nlev * dm))
        rain_flux = jnp.cumsum(form * dm, axis=0)
        clouds = CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=jnp.full((nlev, ncols), 0.6),
            qc=jnp.full((nlev, ncols), 1.0e-3),
            precip_rain=jnp.full((ncols,), precip),
            precip_formation_rate=form,
            rain_flux=rain_flux,
            # The process-time scavenging ledger (#708), consistent with
            # the grid-mean fields above: in-cloud pool = qc/cf, in-cloud
            # formation = grid-mean formation/cf, and the process cover
            # equal to the published cover (nothing emptied here).
            incloud_liquid=jnp.full((nlev, ncols), 1.0e-3 / 0.6),
            incloud_rain_formation=form / 0.6,
            process_cloud_fraction=jnp.full((nlev, ncols), 0.6),
        )
        diagnostics = {
            CARRY_KEY: carry,
            "_jam_state": aer,
            "activated_fraction": jnp.full((nlev, ncols), 0.7),
            "air_density": jnp.full((nlev, ncols), 1.0),
            "layer_thickness": jnp.full((nlev, ncols), 200.0),
            "clouds": clouds,
            # The 2M scheme's in-cloud effective radii [um]. Zero here, so the
            # in-cloud impaction collects nothing and these tests isolate the
            # other pathways; ImpactionInWetScavengingTest sets them.
            "reffl": jnp.zeros((nlev, ncols)),
            "reffi": jnp.zeros((nlev, ncols)),
        }
        return state, diagnostics, MAM4_SPEC, mass_name

    @staticmethod
    def _cb_rate(diag_in, diag_out, nm, dt=1800.0):
        """Effective cloud-borne removal rate from the carry update."""
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        return (
            np.asarray(diag_out[CARRY_KEY][nm])
            - np.asarray(diag_in[CARRY_KEY][nm])
        ) / dt

    def test_scavenging_is_a_sink(self):
        state, diagnostics, spec, mass_name = self._setup()
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(bool(jnp.all(tend.tracers[key] <= 0.0)))
        self.assertTrue(np.all(np.isfinite(np.asarray(tend.tracers[key]))))

    def test_extreme_rate_stays_bounded(self):
        # A scavenging rate with rate·dt ≫ 1 (heavy precip + near-clear low qc +
        # large coarse wet radius) must NOT remove more than the available mass
        # in one step. The implicit q·exp(-rate·dt) update keeps a forward step
        # in [0, q]; the old explicit -rate·q overshot into a sign-flipped
        # runaway (the natural-emission blow-up). Regression guard.
        state, diagnostics, spec, mass_name = self._setup(precip=1.0e-2)
        dt = 1800.0
        diagnostics = dict(diagnostics)
        diagnostics["_dt_seconds"] = dt
        aer = diagnostics["_jam_state"]
        diagnostics["_jam_state"] = aer.copy(
            r_wet=jnp.full_like(aer.r_wet, 5.0e-6)        # huge below-cloud rate
        )
        diagnostics["clouds"] = diagnostics["clouds"].copy(
            qc=jnp.full_like(diagnostics["clouds"].qc, 1.0e-9)  # huge in-cloud rate
        )
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        for nm, dq in tend.tracers.items():
            q0 = np.asarray(state.tracers[nm])
            q_new = q0 + np.asarray(dq) * dt
            self.assertTrue(np.all(np.isfinite(q_new)), nm)
            # Bounds hold up to floating-point roundoff; assert relative to the
            # field scale so f32 roundoff on the ~1e8 number tracers (n_acc →
            # -8 in the full-suite build) isn't mistaken for a real overshoot.
            scale = float(np.abs(q0).max())
            self.assertGreaterEqual(float(q_new.min()), -1e-5 * scale, nm)
            self.assertLessEqual(float(q_new.max()), float(q0.max()) + 1e-5 * scale, nm)

    def test_cloud_fraction_gt_one_stays_finite(self):
        # The cloud scheme can hand back cloud_fraction > 1 (e.g. where RH > 1).
        # The below-cloud clear-sky fraction (1 - cf) then goes negative, which
        # made the scavenging rate negative and the implicit 1-exp(-rate·dt)
        # removed fraction overflow to +inf, NaN-ing every aerosol tracer.
        # The clear fraction (and the rate) are clamped to ≥0, so the tendency
        # must stay finite for cf > 1. Regression guard.
        state, diagnostics, spec, mass_name = self._setup(precip=1.0e-2)
        diagnostics = dict(diagnostics)
        diagnostics["clouds"] = diagnostics["clouds"].copy(
            cloud_fraction=jnp.full_like(diagnostics["clouds"].cloud_fraction, 1.3)
        )
        # coarse wet radius makes the below-cloud rate large in magnitude
        aer = diagnostics["_jam_state"]
        diagnostics["_jam_state"] = aer.copy(r_wet=jnp.full_like(aer.r_wet, 5.0e-6))
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        for nm, dq in tend.tracers.items():
            self.assertTrue(np.all(np.isfinite(np.asarray(dq))), nm)
            self.assertTrue(bool(jnp.all(dq <= 0.0)), nm)  # still a sink, not a source

    def test_no_precip_no_removal(self):
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(bool(jnp.allclose(tend.tracers[key], 0.0)))

    def _attach_convection(self, diagnostics, nlev, ncols, conv_precip=1.0e-4,
                           precip_flux=None, mass_flux_up=None):
        from jcm.physics.convection.tiedtke_nordeng.types import ConvectionData

        import dataclasses
        # Convective cloud on levels 1..nlev-2: condensate + formation
        # there, none at the top level or the sub-cloud bottom level.
        prof = jnp.ones((nlev, ncols)).at[0].set(0.0).at[-1].set(0.0)
        form = prof * 1.0e-4
        if precip_flux is None:
            # Flux ENTERING each layer = everything formed above it, which
            # is what the scheme's cuflx budget publishes.
            precip_flux = jnp.concatenate(
                [jnp.zeros((1, ncols)), jnp.cumsum(form, axis=0)[:-1]], axis=0,
            )
        if mass_flux_up is None:
            # A vigorous plume through the cloud levels: with rho = 1 and
            # w_u = 2 m/s this is a 10 % updraft area, the footprint the
            # convective washout acts in (tapering to 5 % in the sub-cloud
            # level below base level nlev-2).
            mass_flux_up = prof * 0.2
        conv = dataclasses.replace(
            ConvectionData.zeros((ncols,), nlev),
            mass_flux_up=mass_flux_up,
            ktype=jnp.ones((ncols,), dtype=jnp.int32),
            precip_conv=jnp.full((ncols,), conv_precip),
            precip_formation=form,
            precip_flux=precip_flux,
            qc_conv=prof * 1.0e-3,
        )
        diagnostics = dict(diagnostics)
        diagnostics["convection"] = conv
        return diagnostics

    def test_convective_precip_scavenges(self):
        # The convective pathway must strengthen removal vs the same state
        # without it: soluble modes via in-cloud + washout, the insoluble
        # pcm mode via impaction only (which sees the local flux).
        state, diagnostics, spec, mass_name = self._setup()
        term = WetScavenging()
        tend_ref, _ = term(state, diagnostics, None, None)
        tend_conv, _ = term(
            state, self._attach_convection(diagnostics, 4, 2), None, None,
        )
        for i, mode in enumerate(spec.modes):
            key = mass_name(mode.species[0], mode.short)
            self.assertLess(
                float(tend_conv.tracers[key].sum()),
                float(tend_ref.tracers[key].sum()),
                f"convective precip must add removal for mode {mode.short}",
            )
        # (Layer confinement of the convective in-cloud rate is asserted at
        # the function level in ``test_conv_in_cloud_confined_to_heated_layers``;
        # here the stratiform in-cloud term already near-saturates the implicit
        # exponential update in cloudy layers, so only the sign/monotonicity of
        # the total increment is meaningful.)

    def test_conv_washout_confined_below_cloud_top(self):
        # With ONLY convective precip, levels above where convective precip
        # first forms carry zero flux and must see EXACTLY zero removal —
        # rain cannot collect aerosol above where it forms. The confinement
        # is the flux profile itself, not a separately diagnosed cloud top.
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        diagnostics = self._attach_convection(diagnostics, 4, 2)
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        dq = np.asarray(tend.tracers[key])
        np.testing.assert_array_equal(dq[0], 0.0)      # nothing formed above
        self.assertTrue(np.all(dq[3] < 0.0))           # below cloud

    def test_conv_washout_scales_with_the_local_flux(self):
        # Impaction scales with the precipitation flux falling into THAT
        # level. The surface flux applied from the convective cloud top
        # down over-stated the carrier through the depth of the cloud,
        # where the flux is still accumulating.
        nlev, ncols = 4, 2
        # Small enough that 1 - exp(-rate*dt) is linear in the rate to
        # well under the tolerance below.
        flux = jnp.array([0.0, 1.0e-6, 2.0e-6, 4.0e-6])[:, None] * jnp.ones(
            (1, ncols))
        state, diagnostics, spec, mass_name = self._setup(
            nlev=nlev, ncols=ncols, precip=0.0)
        # A plume reaching the surface layer keeps the updraft footprint
        # uniform over levels 1..3, so only the flux varies between them.
        mfu = jnp.array([0.0, 0.2, 0.2, 0.2])[:, None] * jnp.ones((1, ncols))
        diagnostics = self._attach_convection(
            diagnostics, nlev, ncols, precip_flux=flux, mass_flux_up=mfu)
        # in_plume_convective retires the environment-profile convective
        # in-cloud rate, so washout is the only convective sink left.
        tend, _ = WetScavenging(in_plume_convective=True)(
            state, diagnostics, None, None,
        )
        # An insoluble (non-activatable) mode sees washout only.
        mode = next(m for m in spec.modes if not m.can_activate)
        dq = np.asarray(tend.tracers[mass_name(mode.species[0], mode.short)])
        np.testing.assert_array_equal(dq[0], 0.0)
        # Removal is ~linear in the rate at these magnitudes, so the
        # per-level removal follows the flux ratios 1 : 2 : 4.
        np.testing.assert_allclose(dq[2] / dq[1], 2.0, rtol=2e-3)
        np.testing.assert_allclose(dq[3] / dq[1], 4.0, rtol=2e-3)

    def test_conv_washout_acts_in_the_updraft_footprint(self):
        # jax-gcm#781: convective impaction removes aerosol only from the
        # fraction of the box the convective rain falls through — HAMMOZ's
        # updraft area — so it scales with the updraft mass flux and
        # vanishes without a plume, however much flux the profile carries.
        nlev, ncols = 4, 2
        flux = jnp.array([0.0, 1.0e-6, 1.0e-6, 1.0e-6])[:, None] * jnp.ones(
            (1, ncols))
        state, diagnostics, spec, mass_name = self._setup(
            nlev=nlev, ncols=ncols, precip=0.0)
        mode = next(m for m in spec.modes if not m.can_activate)
        key = mass_name(mode.species[0], mode.short)

        def washout(mfu_base):
            mfu = jnp.array([0.0, 1.0, 1.0, 1.0])[:, None] * jnp.full(
                (1, ncols), mfu_base)
            diag = self._attach_convection(
                diagnostics, nlev, ncols, precip_flux=flux, mass_flux_up=mfu)
            tend, _ = WetScavenging(in_plume_convective=True)(
                state, diag, None, None)
            return np.asarray(tend.tracers[key])

        weak, strong, none = washout(0.1), washout(0.2), washout(0.0)
        self.assertTrue(np.all(weak[1:] < 0.0))
        np.testing.assert_allclose(strong[1:] / weak[1:], 2.0, rtol=1e-3)
        np.testing.assert_array_equal(none, 0.0)

    def test_conv_scavenging_no_convection_key_is_noop(self):
        # Without a "convection" diagnostic the term must fall back to the
        # stratiform-only behaviour (composability without a convection scheme).
        state, diagnostics, spec, mass_name = self._setup()
        self.assertNotIn("convection", diagnostics)
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(np.all(np.isfinite(np.asarray(tend.tracers[key]))))

    def test_in_plume_flag_retires_env_conv_incloud(self):
        # With in-plume scavenging in the transport term (jax-gcm#621),
        # this term must drop its environment-profile convective in-cloud
        # removal (weaker sink than the default under convective precip)
        # while keeping the below-cloud convective washout (stronger sink
        # than with no convection at all).
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        diag_conv = self._attach_convection(diagnostics, 4, 2)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        both = WetScavenging()(state, diag_conv, None, None)[0]
        plume = WetScavenging(in_plume_convective=True)(
            state, diag_conv, None, None,
        )[0]
        none = WetScavenging(in_plume_convective=True)(
            state, diagnostics, None, None,
        )[0]
        self.assertGreater(                       # sums are ≤ 0: less removal
            float(plume.tracers[key].sum()), float(both.tracers[key].sum()),
        )
        self.assertLess(                          # washout pathway retained
            float(plume.tracers[key].sum()), float(none.tracers[key].sum()),
        )

    def test_conv_scav_flux_folds_into_wet_ledger(self):
        # The transport term's published surface fluxes must appear in the
        # AeroCom ``wet_*`` keys — but only when the in-plume pathway owns
        # convective scavenging.
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            _species_of)
        state, diagnostics, spec, mass_name = self._setup()
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        diagnostics = dict(diagnostics)
        diagnostics["_conv_scav_flux"] = {key: jnp.full((2,), 3.0e-12)}
        _, d_on = WetScavenging(in_plume_convective=True)(
            state, diagnostics, None, None,
        )
        _, d_off = WetScavenging()(state, diagnostics, None, None)
        wet_key = f"wet_{_species_of(key)}"
        np.testing.assert_allclose(
            np.asarray(d_on[wet_key]) - np.asarray(d_off[wet_key]),
            3.0e-12, rtol=1e-4,     # float32 accumulation roundoff
        )

    def test_incloud_driven_by_formation_not_surface_precip(self):
        # The in-cloud pathway must key off the cloud scheme's per-level
        # formation rate, not a reconstruction from the surface precip: a
        # stale surface value with zero formation (and no flux profile)
        # scavenges NOTHING. This fails on the pre-#499 reconstruction.
        from jcm.physics.clouds.cloud_data import CloudData

        state, diagnostics, spec, mass_name = self._setup()
        nlev, ncols = state.temperature.shape
        diagnostics = dict(diagnostics)
        diagnostics["clouds"] = CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=jnp.full((nlev, ncols), 0.6),
            qc=jnp.full((nlev, ncols), 1.0e-3),
            precip_rain=jnp.full((ncols,), 1.0e-3),   # stale, no formation
        )
        tend, _ = WetScavenging()(state, diagnostics, None, None)
        for dq in tend.tracers.values():
            np.testing.assert_array_equal(np.asarray(dq), 0.0)

    def test_below_cloud_confined_below_formation(self):
        # Impaction uses the per-level flux entering each layer: levels at
        # and above the formation level see no falling precip and must not
        # scavenge; levels below must. Probed with the non-activatable pcm
        # mode (below-cloud is its only stratiform pathway).
        from jcm.physics.clouds.cloud_data import CloudData

        state, diagnostics, spec, mass_name = self._setup()
        nlev, ncols = state.temperature.shape
        dm = 200.0
        form = jnp.zeros((nlev, ncols)).at[1].set(1.0e-7)
        rain_flux = jnp.cumsum(form * dm, axis=0)
        diagnostics = dict(diagnostics)
        diagnostics["clouds"] = CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=jnp.full((nlev, ncols), 0.6),
            qc=jnp.full((nlev, ncols), 1.0e-3),
            precip_rain=rain_flux[-1],
            precip_formation_rate=form,
            rain_flux=rain_flux,
        )
        tend, _ = WetScavenging()(state, diagnostics, None, None)
        pcm = spec.mode("pcm")
        dq = np.asarray(tend.tracers[mass_name(pcm.species[0], "pcm")])
        np.testing.assert_array_equal(dq[0], 0.0)   # above formation
        np.testing.assert_array_equal(dq[1], 0.0)   # the forming layer itself
        self.assertTrue(np.all(dq[2:] < 0.0))       # washed out below

    def test_reinjection_returns_scavenged_aerosol_where_precip_evaporates(self):
        # Cloud-borne aerosol scavenged in the cloudy upper levels rides
        # the rain down; a fully-evaporating layer below must re-inject it
        # into the INTERSTITIAL phase there, and the term's column budget
        # (removal + re-injection integrated over dm) must equal the
        # surviving surface flux exactly.
        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics.aerosol.jam import number_name

        state, diagnostics, spec, mass_name = self._setup()
        nlev, ncols = state.temperature.shape
        # LEVEL-DEPENDENT layer mass: the budget's dm weightings cancel on
        # a uniform grid (budget(x*dm)/dm == budget(x)), so a misplaced dm
        # would be invisible there — vary it so it isn't.
        dz = jnp.array([300.0, 250.0, 200.0, 150.0])[:, None] * jnp.ones(
            (1, ncols)
        )
        dm = 1.0 * dz
        diagnostics = dict(diagnostics)
        diagnostics["layer_thickness"] = dz
        # Rain forms at levels 0-1; level 2's evaporation consumes the
        # whole accumulated carrier (evap*dm2 == form*(dm0+dm1)); none
        # below.
        form = jnp.zeros((nlev, ncols)).at[0].set(1.0e-7).at[1].set(1.0e-7)
        evap_rate = 1.0e-7 * (300.0 + 250.0) / 200.0
        evap = jnp.zeros((nlev, ncols)).at[2].set(evap_rate)
        cf = jnp.zeros((nlev, ncols)).at[0:2].set(0.6)
        diagnostics["clouds"] = CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=cf,
            qc=jnp.zeros((nlev, ncols)).at[0:2].set(1.0e-3),
            precip_formation_rate=form,
            precip_evaporation_rate=evap,
            # Ledger fields (#708) consistent with the grid-mean ones.
            incloud_liquid=jnp.zeros((nlev, ncols)).at[0:2].set(
                1.0e-3 / 0.6),
            incloud_rain_formation=form / 0.6,
            process_cloud_fraction=cf,
        )
        # Isolate the in-cloud pathway: no impaction.
        params = WetDepParameters(
            incloud_scale=jnp.asarray(1.0),
            sol_factb=jnp.asarray(0.0),
            mu_water_air=jnp.asarray(60.0),
            impact_scale=jnp.asarray(1.0),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(2.0),
        )
        term = WetScavenging(params=params)
        tend, out = term(state, diagnostics, None, None)

        mode = spec.modes[0]
        cb = self._cb_rate(
            diagnostics, out,
            mass_name(mode.species[0], mode.short, cloud_borne=True),
        )
        it = np.asarray(tend.tracers[mass_name(mode.species[0], mode.short)])
        # Removal only where condensate converts (levels 0-1), from the
        # cloud-borne tracer.
        self.assertTrue(np.all(cb[0:2] < 0.0))
        np.testing.assert_array_equal(cb[2:], 0.0)
        # Re-injection lands in the INTERSTITIAL tracer in the evap layer.
        self.assertTrue(np.all(it[2] > 0.0))
        np.testing.assert_array_equal(it[[0, 1, 3]], 0.0)
        # Column budget: everything scavenged was re-released (full evap),
        # for mass and number alike.
        dm_np = np.asarray(dm)
        for pair in (
            (mass_name(mode.species[0], mode.short, cloud_borne=True),
             mass_name(mode.species[0], mode.short)),
            (number_name(mode.short, cloud_borne=True),
             number_name(mode.short)),
        ):
            cb_nm, int_nm = pair
            net = (
                self._cb_rate(diagnostics, out, cb_nm) * dm_np
                + np.asarray(tend.tracers[int_nm]) * dm_np
            )
            # Tolerance relative to the gross removal: the evap fraction
            # carries f32 round-off from the cumsum/divide, so "everything
            # re-released" holds to ~1e-6 of what was scavenged, not to
            # absolute zero on 1e8-scale number tracers.
            gross = float(
                np.sum(np.abs(self._cb_rate(diagnostics, out, cb_nm)) * dm_np)
            )
            np.testing.assert_allclose(
                np.sum(net, axis=0), 0.0, atol=max(1e-5 * gross, 1e-20),
                err_msg=str(pair),
            )

    def test_factory_door_scales_change_the_removal(self):
        # ``echam_physics(wetdep={...})`` -- what ``+physics.wetdep.<field>=``
        # builds -- must move the removal of the term it composes, each scale
        # on its own pathway: ``incloud_scale`` on the in-droplet (cloud-borne)
        # removal, ``impact_scale`` on the below-cloud impaction of the
        # interstitial aerosol. Compared through the AeroCom ``wet_*`` ledger,
        # the deposition diagnostic the retune scores.
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            _species_of)
        from jcm.physics.echam.echam_terms import echam_physics

        state, diagnostics, spec, mass_name = self._setup()
        species, short = spec.modes[0].species[0], spec.modes[0].short
        cb_key = mass_name(species, short, cloud_borne=True)
        wet_key = f"wet_{_species_of(mass_name(species, short))}"

        def run(**wetdep):
            physics = echam_physics(
                checkpoint_terms=False, aerosol_module="jam",
                cloud_scheme="2m", jam_microphysics="placeholder",
                wetdep=wetdep)
            term = next(t for t in physics.terms
                        if t.category == "aerosol_wetdep")
            _, out = term(state, diagnostics, None, None)
            return (np.asarray(out[wet_key]),
                    self._cb_rate(diagnostics, out, cb_key))

        wet, cb = run()
        self.assertTrue(bool((wet > 0.0).all()))
        self.assertTrue(bool((cb < 0.0).all()))

        wet_ic0, cb_ic0 = run(incloud_scale=0.0)
        self.assertTrue(bool((wet_ic0 < wet).all()))
        np.testing.assert_array_equal(cb_ic0, 0.0)

        wet_im0, cb_im0 = run(impact_scale=0.0)
        self.assertTrue(bool((wet_im0 < wet).all()))
        np.testing.assert_array_equal(cb_im0, cb)

        # A half-strength scale removes less than full strength but still
        # removes (the update is the implicit q*exp(-rate*dt), so the change
        # is monotone in the rate, not proportional to it).
        _, cb_half = run(incloud_scale=0.5)
        self.assertTrue(bool(((cb < cb_half) & (cb_half < 0.0)).all()))

    def test_conv_transport_door_scales_only_the_convective_ledger(self):
        # ``echam_physics(conv_transport={"conv_scav_scale": s})`` -- what
        # ``+physics.conv_transport.conv_scav_scale=`` builds -- scales the
        # in-plume convective scavenging the transport term publishes and
        # WetScavenging folds into the AeroCom ``wet_*`` ledger. At zero the
        # convective contribution is gone from the ledger, and the removal
        # WetScavenging computes itself (stratiform in-cloud, below-cloud) is
        # untouched: its tendencies are identical at every scale.
        import dataclasses

        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            _species_of)
        from jcm.physics.echam.echam_terms import echam_physics

        state, diagnostics, spec, mass_name = self._setup()
        nlev, ncols = state.temperature.shape
        diagnostics = self._attach_convection(diagnostics, nlev, ncols)
        conv = diagnostics["convection"]
        # A precipitating plume with a sub-cloud release: what the in-plume
        # removal reads besides the mass flux and condensate attached above.
        diagnostics["convection"] = dataclasses.replace(
            conv, precip_efficiency=jnp.where(conv.qc_conv > 0.0, 0.5, 0.0),
            precip_evap_fraction=jnp.full_like(conv.qc_conv, 0.3))
        diagnostics["_dt_seconds"] = 1800.0
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        species = _species_of(key)

        def run(scale):
            physics = echam_physics(
                checkpoint_terms=False, aerosol_module="jam",
                cloud_scheme="2m", jam_microphysics="placeholder",
                conv_transport={"conv_scav_scale": scale})
            transport = next(t for t in physics.terms
                             if t.name == "convective_tracer_transport")
            wetdep = next(t for t in physics.terms
                          if t.category == "aerosol_wetdep")
            _, published = transport(state, diagnostics, None, None)
            tend, out = wetdep(state, published, None, None)
            return published["_conv_scav_flux"], np.asarray(
                out[f"wet_{species}"]), tend

        flux1, wet1, tend1 = run(1.0)
        flux0, wet0, tend0 = run(0.0)
        flux_half, wet_half, _ = run(0.5)

        self.assertTrue(bool((np.asarray(flux1[key]) > 0.0).all()))
        for nm, f in flux0.items():
            np.testing.assert_array_equal(np.asarray(f), 0.0, err_msg=nm)
        self.assertTrue(bool(((0.0 < np.asarray(flux_half[key]))
                              & (np.asarray(flux_half[key])
                                 < np.asarray(flux1[key]))).all()))
        # The ledger moves with the published convective flux and nothing
        # else: at scale 1 it exceeds scale 0 by exactly the fluxes of this
        # species' tracers (every mode's mass of it).
        self.assertTrue(bool(((wet0 < wet_half) & (wet_half < wet1)).all()))
        folded = sum(np.asarray(f) for nm, f in flux1.items()
                     if _species_of(nm) == species)
        np.testing.assert_allclose(wet1 - wet0, folded, rtol=1e-4)
        # WetScavenging's own removal does not read the transport fluxes.
        for nm in tend1.tracers:
            np.testing.assert_array_equal(
                np.asarray(tend1.tracers[nm]), np.asarray(tend0.tracers[nm]),
                err_msg=nm)

    def test_cloud_borne_removed_at_full_incloud_rate(self):
        # Cloud-borne aerosol is entirely in-droplet: its stratiform removal
        # must not scale with the interstitial activated fraction, and must
        # be a strict sink wherever condensate converts to precip (#602).
        state, diagnostics, spec, mass_name = self._setup()
        from jcm.physics.aerosol.jam import number_name

        term = WetScavenging()
        _, out = term(state, diagnostics, None, None)
        cb_key = mass_name(spec.modes[0].species[0], spec.modes[0].short,
                           cloud_borne=True)
        nc_key = number_name(spec.modes[0].short, cloud_borne=True)
        self.assertTrue(
            bool((self._cb_rate(diagnostics, out, cb_key) < 0.0).all())
        )
        self.assertTrue(
            bool((self._cb_rate(diagnostics, out, nc_key) < 0.0).all())
        )
        # Independent of the interstitial activated fraction.
        diagnostics_af0 = dict(diagnostics)
        diagnostics_af0["activated_fraction"] = jnp.zeros_like(
            diagnostics["activated_fraction"]
        )
        _, out_af0 = term(state, diagnostics_af0, None, None)
        np.testing.assert_array_equal(
            self._cb_rate(diagnostics_af0, out_af0, cb_key),
            self._cb_rate(diagnostics, out, cb_key),
        )

    def test_incloud_pathway_moves_with_the_representation(self):
        # Explicit cloud-borne phase (default MAM4 spec): the interstitial
        # tracers keep only impaction — the activated fraction no longer
        # scales their removal. Implicit phase (cloud_borne=False): the
        # activated fraction does scale removal, and no mirror tendencies
        # are emitted at all.
        import dataclasses
        from jcm.physics.aerosol.jam import MAM4_SPEC

        state, diagnostics, spec, mass_name = self._setup()
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        lo = dict(diagnostics)
        lo["activated_fraction"] = jnp.full_like(
            diagnostics["activated_fraction"], 0.1
        )

        explicit = WetScavenging()
        t_hi, _ = explicit(state, diagnostics, None, None)   # af = 0.7
        t_lo, _ = explicit(state, lo, None, None)
        np.testing.assert_array_equal(
            np.asarray(t_hi.tracers[key]), np.asarray(t_lo.tracers[key]),
        )

        implicit = WetScavenging(
            spec=dataclasses.replace(MAM4_SPEC, cloud_borne=False)
        )
        t_hi, _ = implicit(state, diagnostics, None, None)
        t_lo, _ = implicit(state, lo, None, None)
        self.assertLess(
            float(t_hi.tracers[key].sum()), float(t_lo.tracers[key].sum()),
            "implicit treatment must scavenge more at higher activation",
        )
        self.assertFalse(
            any(nm.startswith(("mc_", "nc_")) for nm in t_hi.tracers),
            "implicit population must not emit mirror tendencies",
        )

    def test_equilibrium_removal_matches_between_representations(self):
        # At exchange equilibrium (q_cb = cf·af·q_tot) the explicit
        # representation removes rate_ic·q_cb and the implicit one removes
        # cf·af·rate_ic·q_tot — the SAME mass. Without the cf factor on the
        # implicit stratiform rate the two disagree by 1/cf (2x here), which
        # would poison the #602 A/B comparison with a difference that has
        # nothing to do with the representation. Below-cloud impaction is
        # switched off and the precip kept light so the exponential update
        # stays in its linear regime.
        import dataclasses
        from jcm.physics.aerosol.jam import MAM4_SPEC

        from jcm.physics.aerosol.jam import MAM4_SPEC as SPEC, number_name
        from jcm.physics.aerosol.jam.activation.arg_term import (
            JamActivationData,
        )

        state, diagnostics, spec, mass_name = self._setup(precip=1.0e-7)
        cf, q_tot, n_tot = 0.6, 1.0e-9, 1.0e8
        shape = state.temperature.shape
        # Distinct per-mode AND per-quantity fractions, so using the
        # aggregate (or the wrong one of the pair, or the wrong mode's)
        # anywhere breaks the match.
        n_modes = SPEC.n_modes()
        can = jnp.asarray(
            [float(m.can_activate) for m in SPEC.modes]
        ).reshape(-1, 1, 1)
        per_mode = can / (1.0 + jnp.arange(n_modes).reshape(-1, 1, 1))
        act = JamActivationData(
            number_frac=per_mode * jnp.full((n_modes,) + shape, 0.4),
            mass_frac=per_mode * jnp.full((n_modes,) + shape, 0.8),
        )
        diagnostics = dict(diagnostics)
        diagnostics["_jam_activation"] = act
        params = WetDepParameters(
            incloud_scale=jnp.asarray(1.0),
            sol_factb=jnp.asarray(0.0),
            mu_water_air=jnp.asarray(60.0),
            impact_scale=jnp.asarray(1.0),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(2.0),
        )

        # The (interstitial key, cloud-borne key, total, fraction) tuples
        # under test: mass and number of the first two activatable modes.
        cases = []
        for i in (0, 1):
            mode = SPEC.modes[i]
            fn = float(act.number_frac[i, 0, 0])
            fm = float(act.mass_frac[i, 0, 0])
            cases.append((number_name(mode.short),
                          number_name(mode.short, cloud_borne=True),
                          n_tot, fn))
            cases.append((mass_name(mode.species[0], mode.short),
                          mass_name(mode.species[0], mode.short,
                                    cloud_borne=True),
                          q_tot, fm))

        # Explicit: each pair partitioned at its own exchange equilibrium
        # (interstitial in the tracers, cloud-borne in the carry).
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        tracers = dict(state.tracers)
        carry = dict(diagnostics[CARRY_KEY])
        for key_int, key_cb, tot, frac in cases:
            q_cb = cf * frac * tot
            tracers[key_int] = jnp.full_like(tracers[key_int], tot - q_cb)
            carry[key_cb] = jnp.full_like(carry[key_cb], q_cb)
        diag_exp = {**diagnostics, CARRY_KEY: carry}
        tend_exp, out_exp = WetScavenging(params=params)(
            state.copy(tracers=tracers), diag_exp, None, None,
        )

        # Implicit: everything interstitial, per-mode-fraction scavenged.
        tracers = dict(state.tracers)
        for key_int, _, tot, _ in cases:
            tracers[key_int] = jnp.full_like(tracers[key_int], tot)
        implicit = WetScavenging(
            params=params,
            spec=dataclasses.replace(MAM4_SPEC, cloud_borne=False),
        )
        tend_imp, _ = implicit(
            state.copy(tracers=tracers), diagnostics, None, None,
        )

        for key_int, key_cb, tot, frac in cases:
            removed_explicit = -(
                np.asarray(tend_exp.tracers[key_int])
                + self._cb_rate(diag_exp, out_exp, key_cb)
            )
            removed_implicit = -np.asarray(tend_imp.tracers[key_int])
            self.assertGreater(float(removed_explicit.max()), 0.0, key_int)
            np.testing.assert_allclose(
                removed_implicit, removed_explicit, rtol=5e-3,
                err_msg=key_int,
            )

    def _mixed_phase(self, state, diagnostics, *, pice, f_wat_ic, f_ice_ic,
                     qni=0.0):
        """Rewrite the setup's cloud as mixed-phase with known conversions."""
        shape = state.temperature.shape
        clouds = diagnostics["clouds"]
        pool = 1.0e-3 / 0.6
        dt = 1800.0
        clouds = clouds.copy(
            incloud_liquid=jnp.full(shape, pool * (1.0 - pice)),
            incloud_ice=jnp.full(shape, pool * pice),
            incloud_rain_formation=jnp.full(
                shape, f_wat_ic * pool * (1.0 - pice) / dt),
            incloud_snow_formation=jnp.full(shape, f_ice_ic * pool * pice / dt),
            incloud_riming=jnp.zeros(shape),
        )
        tracers = {**state.tracers, "qni": jnp.full(shape, qni)}
        return state.copy(tracers=tracers), {**diagnostics, "clouds": clouds}

    def test_reservoir_phases_are_removed_at_their_own_conversion(self):
        # HAM scavenges the droplet-held and crystal-held aerosol apart
        # (ham_wetdep: water part x f_wat, ice part x f_ice). A reservoir the
        # exchange recorded as 30 % droplet-held loses 0.3 f_wat + 0.7 f_ice
        # of itself, not the condensate-weighted (1-p) f_wat + p f_ice.
        from jcm.physics.aerosol.jam import mass_name
        from jcm.physics.aerosol.jam.cloud_borne_store import LIQUID_SHARE_KEY

        state, diagnostics, spec, _ = self._setup(precip=1.0e-7)
        state, diagnostics = self._mixed_phase(
            state, diagnostics, pice=0.5, f_wat_ic=0.4, f_ice_ic=0.1)
        params = WetDepParameters(
            incloud_scale=jnp.asarray(1.0), sol_factb=jnp.asarray(0.0),
            mu_water_air=jnp.asarray(60.0), impact_scale=jnp.asarray(1.0),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(2.0))
        nm = mass_name("so4", "acc", cloud_borne=True)
        diag = {**diagnostics, LIQUID_SHARE_KEY: {nm: jnp.full(
            state.temperature.shape, 0.3)}}
        _, out = WetScavenging(params=params)(state, diag, None, None)
        lost = -self._cb_rate(diag, out, nm) * 1800.0 / 1.0e-9
        np.testing.assert_allclose(lost, 0.3 * 0.4 + 0.7 * 0.1, rtol=2e-3)

    def test_implicit_population_splits_the_phases(self):
        # Without an explicit phase the in-cloud removal is HAM's
        # cf * [(1-p) f_ARG f_wat + p f_ice_nuc f_ice]: in an ice cloud with
        # no crystals nothing activated by droplet rules is removed.
        import dataclasses
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.activation.arg_term import (
            JamActivationData)

        state, diagnostics, spec, _ = self._setup(precip=1.0e-7)
        params = WetDepParameters(
            incloud_scale=jnp.asarray(1.0), sol_factb=jnp.asarray(0.0),
            mu_water_air=jnp.asarray(60.0), impact_scale=jnp.asarray(1.0),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(2.0))
        n_modes = MAM4_SPEC.n_modes()
        shape = state.temperature.shape
        act = JamActivationData(
            number_frac=jnp.full((n_modes,) + shape, 0.5),
            mass_frac=jnp.full((n_modes,) + shape, 0.8))
        implicit = WetScavenging(
            params=params,
            spec=dataclasses.replace(MAM4_SPEC, cloud_borne=False))
        nm = mass_name("so4", "acc")
        removed = {}
        for pice in (0.0, 1.0):
            st, dg = self._mixed_phase(state, {**diagnostics,
                                               "_jam_activation": act},
                                       pice=pice, f_wat_ic=0.4, f_ice_ic=0.4)
            tend, _ = implicit(st, dg, None, None)
            removed[pice] = float(-np.asarray(tend.tracers[nm]).mean()) * 1800.0
        np.testing.assert_allclose(removed[0.0] / 1.0e-9, 0.6 * 0.8 * 0.4,
                                   rtol=2e-3)
        self.assertLess(removed[1.0], 1e-6 * removed[0.0])

    def test_grad_through_sol_factb(self):
        state, diagnostics, spec, mass_name = self._setup()

        def loss(coeff):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                sol_factb=coeff,
                mu_water_air=jnp.asarray(60.0),
                impact_scale=jnp.asarray(1.0),
                conv_scav_ratio=jnp.asarray(0.99),
                conv_updraft_velocity=jnp.asarray(2.0),
            )
            term = WetScavenging(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(0.1))
        self.assertTrue(np.isfinite(float(g)))

    def test_grad_through_conv_scav_ratio(self):
        # The convective scavenging ratio must be a live differentiable
        # knob: nonzero, finite gradient when convective cloud is present.
        state, diagnostics, spec, mass_name = self._setup()
        diagnostics = self._attach_convection(diagnostics, 4, 2)

        def loss(ratio):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                sol_factb=jnp.asarray(0.1),
                mu_water_air=jnp.asarray(60.0),
                impact_scale=jnp.asarray(1.0),
                conv_scav_ratio=ratio,
                conv_updraft_velocity=jnp.asarray(2.0),
            )
            term = WetScavenging(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(0.99))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)

    def test_grad_through_conv_updraft_velocity(self):
        # The updraft velocity sets the convective footprint, so with the
        # washout as the only sink a faster updraft (smaller area) must
        # remove less: a finite, strictly negative gradient of the squared
        # removal.
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        diagnostics = self._attach_convection(diagnostics, 4, 2)

        def loss(w_u):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                sol_factb=jnp.asarray(0.1),
                mu_water_air=jnp.asarray(60.0),
                impact_scale=jnp.asarray(1.0),
                conv_scav_ratio=jnp.asarray(0.99),
                conv_updraft_velocity=w_u,
            )
            term = WetScavenging(params=params, in_plume_convective=True)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(2.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertLess(float(g), 0.0)


if __name__ == "__main__":
    unittest.main()


class FormationLedgerTest(unittest.TestCase):
    """The process-time scavenging ledger pathway (#708).

    The in-cloud removal keys to the cloud scheme's HAMMOZ-interface
    ledger (bounded per-step fractions captured at process time), so a
    cell whose condensate fully converted to precipitation — which ends
    the step with cover 0, condensate 0, and a ZEROED in-cloud pool but a
    positive formation ledger — still scavenges, in exactly the step with
    the largest removal (a cover-keyed reconstruction would read zero
    there — the dead zone the ledger interface exists to close).
    """

    @staticmethod
    def _emptied_cell():
        """Build a one-step full conversion: ledger positive, pools zero."""
        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
        from jcm.physics.aerosol.jam.tracer_layout import (
            mass_name, number_name)
        from jcm.physics_interface import PhysicsState
        from jcm.physics.aerosol.jam.jam_state import JamAerosolState

        nlev, ncols = 4, 2
        spec = MAM4_SPEC
        tracers, carry = {}, {}
        for mode in spec.modes:
            tracers[number_name(mode.short)] = jnp.full(
                (nlev, ncols), 1e8)
            carry[number_name(mode.short, cloud_borne=True)] = jnp.full(
                (nlev, ncols), 1e7)
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full(
                    (nlev, ncols), 1e-9)
                carry[mass_name(sp, mode.short, cloud_borne=True)] = (
                    jnp.full((nlev, ncols), 1e-10))
        tracers["qni"] = jnp.zeros((nlev, ncols))   # the 2M crystal number
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 275.0), tracers=tracers)
        shape = (spec.n_modes(), nlev, ncols)
        aer = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6),
            r_wet=jnp.full(shape, 0.2e-6),
            rho=jnp.full(shape, 1800.0),
            kappa=jnp.full(shape, 0.5),
            mass=jnp.full(shape, 1e-9),
            number=jnp.full(shape, 1.0e8),
        )
        form_gm = jnp.zeros((nlev, ncols)).at[1].set(1.0e-7)
        clouds = CloudData.zeros((ncols,), nlev).copy(
            # POST-microphysics state of the emptied cell: no cover, no
            # condensate, zeroed pools — only the formation ledger and
            # the process-time cover remember what happened.
            precip_formation_rate=form_gm,
            rain_flux=jnp.cumsum(form_gm * 200.0, axis=0),
            precip_rain=(form_gm * 200.0).sum(axis=0),
            incloud_rain_formation=form_gm / 0.5,
            process_cloud_fraction=jnp.zeros(
                (nlev, ncols)).at[1].set(0.5),
        )
        diagnostics = {
            CARRY_KEY: carry,
            "_jam_state": aer,
            "activated_fraction": jnp.full((nlev, ncols), 0.7),
            "air_density": jnp.full((nlev, ncols), 1.0),
            "layer_thickness": jnp.full((nlev, ncols), 200.0),
            "clouds": clouds,
            "_dt_seconds": 1800.0,
            "reffl": jnp.zeros((nlev, ncols)),
            "reffi": jnp.zeros((nlev, ncols)),
        }
        return state, diagnostics, spec

    def test_emptied_cell_still_scavenges_cloud_borne(self):
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        from jcm.physics.aerosol.jam.tracer_layout import mass_name

        state, diagnostics, spec = self._emptied_cell()
        params = WetDepParameters(
            incloud_scale=jnp.asarray(1.0),
            sol_factb=jnp.asarray(0.0),          # isolate in-cloud
            mu_water_air=jnp.asarray(60.0),
            impact_scale=jnp.asarray(1.0),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(2.0),
        )
        cb_key = mass_name(spec.modes[0].species[0], spec.modes[0].short,
                           cloud_borne=True)
        q0 = np.asarray(diagnostics[CARRY_KEY][cb_key])

        _, out = WetScavenging(params=params)(
            state, diagnostics, None, None)
        q1 = np.asarray(out[CARRY_KEY][cb_key])
        # Fraction-1 marker: essentially ALL cloud-borne mass in the
        # emptied level leaves with the precip it formed...
        self.assertLess(q1[1].max(), 2e-6 * q0[1].max())
        # ...and untouched levels keep theirs.
        np.testing.assert_allclose(q1[0], q0[0], rtol=1e-6)

    def test_interstitial_share_keys_to_process_cover(self):
        # The implicit (no explicit phase) pathway must weight by the
        # PROCESS-TIME cover: post-write-back cover is 0 in the emptied
        # cell, so keying to it would zero the removal.
        import dataclasses
        from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
        from jcm.physics.aerosol.jam.tracer_layout import mass_name

        state, diagnostics, spec = self._emptied_cell()
        params = WetDepParameters(
            incloud_scale=jnp.asarray(1.0),
            sol_factb=jnp.asarray(0.0),
            mu_water_air=jnp.asarray(60.0),
            impact_scale=jnp.asarray(1.0),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(2.0),
        )
        implicit = WetScavenging(
            params=params,
            spec=dataclasses.replace(MAM4_SPEC, cloud_borne=False),
        )
        tend, _ = implicit(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        dq = np.asarray(tend.tracers[key])
        self.assertLess(dq[1].max(), 0.0)          # removal in the cell
        np.testing.assert_array_equal(dq[0], 0.0)  # none where no process

    def test_ledger_fractions_bounded(self):
        from types import SimpleNamespace
        from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
            incloud_scavenged_fractions)

        dt = 1800.0
        shape = (5,)
        clouds = SimpleNamespace(
            incloud_liquid=jnp.array([1e-3, 1e-3, 0.0, 0.0, 1e-15]),
            incloud_ice=jnp.array([0.0, 1e-4, 0.0, 1e-4, 0.0]),
            # Formation far exceeding the pool per step must clip at 1.
            incloud_rain_formation=jnp.array([1e-2, 1e-7, 1e-7, 0.0, 0.0]) / dt,
            incloud_snow_formation=jnp.array([0.0, 1e-5, 0.0, 1e-3, 0.0]) / dt,
            incloud_riming=jnp.zeros(shape),
        )
        f_wat, f_ice, pice = incloud_scavenged_fractions(clouds, dt)
        for arr in (f_wat, f_ice, pice):
            self.assertTrue(bool(jnp.all((arr >= 0.0) & (arr <= 1.0))))
        self.assertAlmostEqual(float(f_wat[0]), 1.0)   # clipped
        self.assertAlmostEqual(float(f_wat[2]), 1.0)   # emptied-pool marker
        self.assertAlmostEqual(float(f_ice[3]), 1.0)   # ice clip
        self.assertAlmostEqual(float(f_wat[4]), 0.0)   # sub-floor pool, no formation
        # Phase split falls back to the formation ledger in emptied cells.
        self.assertAlmostEqual(float(pice[2]), 0.0)


class ImpactionInWetScavengingTest(unittest.TestCase):
    """In-cloud impaction of interstitial aerosol (HAM ``ic_scav_imp``, #1067)."""

    _setup = WetDepTermTest._setup
    _mixed_phase = WetDepTermTest._mixed_phase
    _cb_rate = staticmethod(WetDepTermTest._cb_rate)

    DT = 1800.0
    CF, PICE, F_WAT, F_ICE = 0.6, 0.5, 0.4, 0.3

    @staticmethod
    def _params():
        # No below-cloud impaction, so in the explicit representation the
        # interstitial removal is the in-cloud impaction alone.
        return WetDepParameters(
            incloud_scale=jnp.asarray(1.0), sol_factb=jnp.asarray(0.0),
            mu_water_air=jnp.asarray(60.0), impact_scale=jnp.asarray(1.0),
            conv_scav_ratio=jnp.asarray(0.99),
            conv_updraft_velocity=jnp.asarray(2.0))

    def _cloudy(self, *, reffl=12.0, reffi=60.0, qni=2.0e7, r_wet=3.0e-6):
        """Build a mixed-phase cloud with known conversions, radii and crystals."""
        state, diag, spec, _ = self._setup(precip=1.0e-7)
        state, diag = self._mixed_phase(
            state, diag, pice=self.PICE, f_wat_ic=self.F_WAT,
            f_ice_ic=self.F_ICE, qni=qni)
        shape = state.temperature.shape
        aer = diag["_jam_state"]
        diag = {**diag, "reffl": jnp.full(shape, reffl),
                "reffi": jnp.full(shape, reffi),
                "_jam_state": aer.copy(r_wet=jnp.full_like(aer.r_wet, r_wet))}
        return state, diag, spec

    def _fractions(self, spec, i, moment, diag, state):
        """HAM's (droplet, crystal) impaction fractions for mode ``i``."""
        from jcm.physics.aerosol.jam.wetdep import incloud_impaction as ii

        mode = spec.modes[i]
        mr = ii.impaction_radius_um(diag["_jam_state"].r_wet[i],
                                    mode.geom_std_dev, moment)
        icnc_m3 = state.tracers["qni"] * diag["air_density"]
        return (ii.droplet_impaction_fraction(diag["reffl"], mr, moment),
                ii.crystal_impaction_fraction(diag["reffi"], icnc_m3, mr,
                                              self.DT))

    def test_explicit_phase_collects_the_interstitial_aerosol(self):
        """``cf·[(1 − p)·F_w·c_wat + p·F_i·c_ice]`` of each interstitial tracer."""
        from jcm.physics.aerosol.jam import mass_name, number_name

        state, diag, spec = self._cloudy()
        tend, _ = WetScavenging(params=self._params())(state, diag, None, None)
        off, _ = WetScavenging(params=self._params(), incloud_impaction="none")(
            state, diag, None, None)
        for i, mode in enumerate(spec.modes):
            for nm, moment in ((number_name(mode.short), "number"),
                               (mass_name(mode.species[0], mode.short), "mass")):
                f_w, f_i = self._fractions(spec, i, moment, diag, state)
                frac = self.CF * ((1.0 - self.PICE) * f_w * self.F_WAT
                                  + self.PICE * f_i * self.F_ICE)
                removed = -np.asarray(tend.tracers[nm]) * self.DT
                q = np.asarray(state.tracers[nm])
                np.testing.assert_allclose(removed, q * np.asarray(frac),
                                           rtol=1e-4, err_msg=nm)
                np.testing.assert_array_equal(np.asarray(off.tracers[nm]), 0.0)
                self.assertGreater(float(frac.min()), 0.0, nm)

    def test_cloud_borne_removal_is_unchanged(self):
        """Impaction adds an interstitial pathway; the reservoir's removal stays."""
        from jcm.physics.aerosol.jam import mass_name

        state, diag, _ = self._cloudy()
        nm = mass_name("so4", "acc", cloud_borne=True)
        _, on = WetScavenging(params=self._params())(state, diag, None, None)
        _, off = WetScavenging(params=self._params(),
                               incloud_impaction="none")(state, diag, None, None)
        np.testing.assert_array_equal(self._cb_rate(diag, on, nm),
                                      self._cb_rate(diag, off, nm))

    def test_implicit_population_sums_nucleation_and_impaction(self):
        """Without the phase: ``min(1, f_ARG + F_w)`` and ``min(1, f_ice + F_i)``."""
        import dataclasses

        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
        from jcm.physics.aerosol.jam.activation.arg_term import (
            JamActivationData)
        from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
            ice_phase_fractions)

        spec = dataclasses.replace(MAM4_SPEC, cloud_borne=False)
        # Large droplets and a 40 um mode drive the droplet mass fraction to
        # its clip, and the mass sum to its cap.
        for reffl, r_wet in ((12.0, 3.0e-6), (40.0, 4.0e-5)):
            state, diag, _ = self._cloudy(reffl=reffl, r_wet=r_wet)
            shape = state.temperature.shape
            n_modes = spec.n_modes()
            act = JamActivationData(
                number_frac=jnp.full((n_modes,) + shape, 0.5),
                mass_frac=jnp.full((n_modes,) + shape, 0.8))
            diag = {**diag, "_jam_activation": act}
            tend, _ = WetScavenging(params=self._params(), spec=spec)(
                state, diag, None, None)
            ice_n, ice_m = ice_phase_fractions(
                spec, [state.tracers[number_name(m.short)] for m in spec.modes],
                state.tracers["qni"])
            mode = spec.modes[2]                                   # coarse
            nm = mass_name(mode.species[0], mode.short)
            f_w, f_i = self._fractions(spec, 2, "mass", diag, state)
            frac = self.CF * (
                (1.0 - self.PICE) * jnp.minimum(0.8 + f_w, 1.0) * self.F_WAT
                + self.PICE * jnp.minimum(ice_m[2] + f_i, 1.0) * self.F_ICE)
            removed = -np.asarray(tend.tracers[nm]) * self.DT
            np.testing.assert_allclose(
                removed, np.asarray(state.tracers[nm] * frac), rtol=1e-4)
        self.assertEqual(float(f_w.min()), 1.0)    # the clipped regime ran

    def test_none_needs_no_radii_and_ham_refuses_without_them(self):
        state, diag, _ = self._cloudy()
        bare = {k: v for k, v in diag.items() if k not in ("reffl", "reffi")}
        WetScavenging(incloud_impaction="none")(state, bare, None, None)
        with self.assertRaisesRegex(ValueError, "reffl"):
            WetScavenging()(state, bare, None, None)
        with self.assertRaisesRegex(ValueError, "incloud_impaction"):
            WetScavenging(incloud_impaction="cam")
        self.assertIn("reffi", WetScavenging().requires)
        self.assertNotIn("reffl", WetScavenging(incloud_impaction="none").requires)

    def test_r7492_variant_reaches_the_term(self):
        from jcm.physics.aerosol.jam import mass_name

        state, diag, _ = self._cloudy(reffi=75.0)
        nm = mass_name("du", "cor")
        ham, _ = WetScavenging(params=self._params())(state, diag, None, None)
        old, _ = WetScavenging(params=self._params(),
                               incloud_impaction="ham_r7492")(
            state, diag, None, None)
        self.assertFalse(np.allclose(np.asarray(ham.tracers[nm]),
                                     np.asarray(old.tracers[nm]),
                                     rtol=1e-3, atol=0.0))

    def test_gradients_through_radii_and_crystal_number(self):
        """AD matches a central difference through the whole term."""
        from jcm.physics.aerosol.jam import mass_name
        from jcm.testing import check_gradients

        with jax.enable_x64(True):
            state, diag, _ = self._cloudy(reffl=13.7, reffi=63.4, qni=2.0e7)
            shape = state.temperature.shape
            nm = mass_name("du", "cor")
            term = WetScavenging(params=self._params())

            def removal(reffl, reffi, qni):
                st = state.copy(tracers={**state.tracers, "qni": qni})
                dg = {**diag, "reffl": reffl, "reffi": reffi}
                return term(st, dg, None, None)[0].tracers[nm]

            check_gradients(
                removal,
                (jnp.full(shape, 13.7), jnp.full(shape, 63.4),
                 jnp.full(shape, 2.0e7)),
                rtol=1e-5, live_inputs=("[0]", "[1]", "[2]"))
