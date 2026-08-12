"""Phase 5 tests: in-cloud + below-cloud scavenging and the term."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
    below_cloud_rate,
    conv_in_cloud_rate,
    in_cloud_rate,
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

    def test_in_cloud_rate_scales_with_activation(self):
        pf = jnp.full((1, 1), 1.0e-6)
        qc = jnp.full((1, 1), 1.0e-3)
        lo = in_cloud_rate(jnp.full((1, 1), 0.2), pf, qc)
        hi = in_cloud_rate(jnp.full((1, 1), 0.9), pf, qc)
        self.assertGreater(float(hi[0, 0]), float(lo[0, 0]))

    def test_below_cloud_size_dependence(self):
        precip = jnp.full((1, 1), 1.0e-4)
        cf = jnp.zeros((1, 1))
        params = WetDepParameters.default()
        accum = below_cloud_rate(precip, cf, jnp.full((1, 1), 0.1e-6), params)
        coarse = below_cloud_rate(precip, cf, jnp.full((1, 1), 2.0e-6), params)
        self.assertGreater(float(coarse[0, 0]), float(accum[0, 0]))

    def test_no_precip_no_below_cloud(self):
        params = WetDepParameters.default()
        rate = below_cloud_rate(
            jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.full((1, 1), 1e-6),
            params,
        )
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
        tracers = {}
        for mode in MAM4_SPEC.modes:
            for cb in (False, True):
                tracers[number_name(mode.short, cloud_borne=cb)] = jnp.full(
                    (nlev, ncols), 1.0e8
                )
                for sp in mode.species:
                    tracers[mass_name(sp, mode.short, cloud_borne=cb)] = (
                        jnp.full((nlev, ncols), 1e-9)
                    )
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
        )
        diagnostics = {
            "_jam_state": aer,
            "activated_fraction": jnp.full((nlev, ncols), 0.7),
            "air_density": jnp.full((nlev, ncols), 1.0),
            "layer_thickness": jnp.full((nlev, ncols), 200.0),
            "clouds": clouds,
        }
        return state, diagnostics, MAM4_SPEC, mass_name

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

    def _attach_convection(self, diagnostics, nlev, ncols, conv_precip=1.0e-4):
        from jcm.physics.convection.tiedtke_nordeng.types import ConvectionData

        import dataclasses
        # Convective cloud on levels 1..nlev-2: condensate + formation
        # there, none at the top level or the sub-cloud bottom level.
        prof = jnp.ones((nlev, ncols)).at[0].set(0.0).at[-1].set(0.0)
        conv = dataclasses.replace(
            ConvectionData.zeros((ncols,), nlev),
            precip_conv=jnp.full((ncols,), conv_precip),
            precip_formation=prof * 1.0e-4,
            qc_conv=prof * 1.0e-3,
        )
        diagnostics = dict(diagnostics)
        diagnostics["convection"] = conv
        return diagnostics

    def test_convective_precip_scavenges(self):
        # The convective pathway must strengthen removal vs the same state
        # without it: soluble modes via in-cloud + washout, the insoluble
        # pcm mode via washout only (below-cloud sees total precip).
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
        # With ONLY convective precip and a pressure diagnostic, levels
        # above the convective cloud top (no heating, lower pressure than
        # any active level) must see EXACTLY zero removal — rain
        # cannot collect aerosol above where it forms.
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        diagnostics = self._attach_convection(diagnostics, 4, 2)
        # Level 0 is the model top (200 hPa); heating is active on levels
        # 1..2, so the convective cloud top is at 500 hPa.
        diagnostics["pressure_full"] = (
            jnp.array([200.0, 500.0, 800.0, 1000.0])[:, None]
            * jnp.ones((1, 2)) * 100.0
        )
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        dq = np.asarray(tend.tracers[key])
        np.testing.assert_array_equal(dq[0], 0.0)      # above conv top
        self.assertTrue(np.all(dq[3] < 0.0))           # below cloud

    def test_conv_scavenging_no_convection_key_is_noop(self):
        # Without a "convection" diagnostic the term must fall back to the
        # stratiform-only behaviour (composability without a convection scheme).
        state, diagnostics, spec, mass_name = self._setup()
        self.assertNotIn("convection", diagnostics)
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(np.all(np.isfinite(np.asarray(tend.tracers[key]))))

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
        )
        # Isolate the in-cloud pathway: no impaction.
        params = WetDepParameters(
            incloud_scale=jnp.asarray(1.0),
            below_coeff=jnp.asarray(0.0),
            below_radius_ref=jnp.asarray(1.0e-7),
            conv_scav_ratio=jnp.asarray(0.99),
        )
        term = WetScavenging(params=params)
        tend, _ = term(state, diagnostics, None, None)

        mode = spec.modes[0]
        cb = np.asarray(
            tend.tracers[mass_name(mode.species[0], mode.short,
                                   cloud_borne=True)]
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
            net = sum(np.asarray(tend.tracers[nm]) * dm_np for nm in pair)
            # Tolerance relative to the gross removal: the evap fraction
            # carries f32 round-off from the cumsum/divide, so "everything
            # re-released" holds to ~1e-6 of what was scavenged, not to
            # absolute zero on 1e8-scale number tracers.
            gross = float(
                np.sum(np.abs(np.asarray(tend.tracers[pair[0]])) * dm_np)
            )
            np.testing.assert_allclose(
                np.sum(net, axis=0), 0.0, atol=max(1e-5 * gross, 1e-20),
                err_msg=str(pair),
            )

    def test_cloud_borne_removed_at_full_incloud_rate(self):
        # Cloud-borne aerosol is entirely in-droplet: its stratiform removal
        # must not scale with the interstitial activated fraction, and must
        # be a strict sink wherever condensate converts to precip (#602).
        state, diagnostics, spec, mass_name = self._setup()
        from jcm.physics.aerosol.jam import number_name

        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        cb_key = mass_name(spec.modes[0].species[0], spec.modes[0].short,
                           cloud_borne=True)
        self.assertIn(cb_key, tend.tracers)
        self.assertIn(number_name(spec.modes[0].short, cloud_borne=True),
                      tend.tracers)
        self.assertTrue(bool(jnp.all(tend.tracers[cb_key] < 0.0)))
        # Independent of the interstitial activated fraction.
        diagnostics_af0 = dict(diagnostics)
        diagnostics_af0["activated_fraction"] = jnp.zeros_like(
            diagnostics["activated_fraction"]
        )
        tend_af0, _ = term(state, diagnostics_af0, None, None)
        np.testing.assert_array_equal(
            np.asarray(tend_af0.tracers[cb_key]),
            np.asarray(tend.tracers[cb_key]),
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
            below_coeff=jnp.asarray(0.0),
            below_radius_ref=jnp.asarray(1.0e-7),
            conv_scav_ratio=jnp.asarray(0.99),
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

        # Explicit: each pair partitioned at its own exchange equilibrium.
        tracers = dict(state.tracers)
        for key_int, key_cb, tot, frac in cases:
            q_cb = cf * frac * tot
            tracers[key_int] = jnp.full_like(tracers[key_int], tot - q_cb)
            tracers[key_cb] = jnp.full_like(tracers[key_cb], q_cb)
        tend_exp, _ = WetScavenging(params=params)(
            state.copy(tracers=tracers), diagnostics, None, None,
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
                + np.asarray(tend_exp.tracers[key_cb])
            )
            removed_implicit = -np.asarray(tend_imp.tracers[key_int])
            self.assertGreater(float(removed_explicit.max()), 0.0, key_int)
            np.testing.assert_allclose(
                removed_implicit, removed_explicit, rtol=5e-3,
                err_msg=key_int,
            )

    def test_grad_through_below_coeff(self):
        state, diagnostics, spec, mass_name = self._setup()

        def loss(coeff):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                below_coeff=coeff,
                below_radius_ref=jnp.asarray(1.0e-7),
                conv_scav_ratio=jnp.asarray(0.99),
            )
            term = WetScavenging(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0e-4))
        self.assertTrue(np.isfinite(float(g)))

    def test_grad_through_conv_scav_ratio(self):
        # The convective scavenging ratio must be a live differentiable
        # knob: nonzero, finite gradient when convective cloud is present.
        state, diagnostics, spec, mass_name = self._setup()
        diagnostics = self._attach_convection(diagnostics, 4, 2)

        def loss(ratio):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                below_coeff=jnp.asarray(1.0e-4),
                below_radius_ref=jnp.asarray(1.0e-7),
                conv_scav_ratio=ratio,
            )
            term = WetScavenging(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(0.99))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)


if __name__ == "__main__":
    unittest.main()
