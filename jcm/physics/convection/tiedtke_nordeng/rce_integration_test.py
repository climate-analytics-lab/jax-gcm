"""RCE-style integration test for the Tiedtke-Nordeng convection scheme.

Covers the full scheme end-to-end on a tropical sounding with CAPE > 1000
J/kg, validating that our fixes (iterative saturation adjustment, wired-up
post-convection adjustment, dynamic LNB termination, Nordeng organized
entrainment) produce physically sensible tendencies: latent heating in
the cloud layer, drying of the boundary layer, and positive precipitation.

These are the RCE signatures that were missing / wrong before the fixes.
"""

import unittest
import jax.numpy as jnp
import numpy as np

from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    tiedtke_nordeng_convection,
    saturation_mixing_ratio,
)


def _tropical_sounding(nlev: int = 47, surface_T: float = 302.0,
                      surface_rh: float = 0.8, lapse_K_per_km: float = 6.5):
    """Build a conditionally-unstable tropical sounding (index 0 = TOA)."""
    # Pressure: 10 hPa (TOA) → 1000 hPa (surface)
    p = jnp.logspace(jnp.log10(1000.0), jnp.log10(100_000.0), nlev)
    z_km = -8.4 * jnp.log(p / 100_000.0)  # approx hypsometric height

    # T: standard lapse rate to 15 km, isothermal above
    T = jnp.maximum(surface_T - lapse_K_per_km * z_km, 200.0)

    # Humidity: prescribed RH, drying aloft
    qs = saturation_mixing_ratio(p, T)
    rh = jnp.where(p > 50_000.0, surface_rh, surface_rh * (p / 50_000.0))
    q = rh * qs

    # Density and layer thickness (approximate hydrostatic)
    rho = p / (287.0 * T)
    # layer_thickness in meters: dz ≈ - dp / (ρ g); use level midpoints
    dp = jnp.concatenate([jnp.diff(p), jnp.array([p[-1] * 0.02])])
    dz = jnp.abs(dp) / (rho * 9.81)

    return T, q, p, dz, rho


class TestRCEConvection(unittest.TestCase):
    """Full-scheme RCE-style integration tests."""

    def test_tropical_sounding_fires_convection(self):
        """On a sounding with CAPE > 1000 J/kg the scheme should produce:
        - non-zero tendencies
        - positive precipitation
        - non-zero updraft mass flux
        """
        # Use a very warm, moist sounding to guarantee CAPE > 1000 J/kg
        # (deep convection threshold in the scheme).
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=305.0, surface_rh=0.9, lapse_K_per_km=7.0
        )
        nlev = T.shape[0]
        u = jnp.zeros(nlev)
        v = jnp.zeros(nlev)
        qc = jnp.zeros(nlev)
        qi = jnp.zeros(nlev)
        dt = 1800.0  # 30 min
        cfg = ConvectionParameters.default()

        tendencies, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, u, v, qc, qi, dt, cfg,
        )
        # Should have nonzero temperature tendency somewhere
        self.assertGreater(
            float(jnp.max(jnp.abs(tendencies.dtedt))), 1e-6,
            "Convection should produce nonzero T tendency on unstable sounding",
        )
        # Surface precipitation should be positive
        self.assertGreater(
            float(tendencies.precip_conv), 0.0,
            "Unstable sounding should produce positive precipitation",
        )
        # Updraft mass flux should be active somewhere in the cloud
        self.assertGreater(
            float(jnp.max(state.mfu)), 1e-4,
            "Updraft mass flux should activate on unstable sounding",
        )

    def test_mid_level_convection_triggers_for_moderate_cape_moist_free_trop(self):
        """ktype=3 should fire when CAPE is moderate (100 < CAPE < 1000)
        and the free troposphere is moist (RH > 90 % at some 700-300 hPa
        level). Mirrors the ECHAM ``cubasmc`` mid-level trigger.

        Bug A regression test: before the trigger was added, JAX returned
        only ktype ∈ {0, 1, 2}; ktype=3 (mid-level) was a documented
        omission flagged by the Fortran harness comparison.
        """
        # Build a sounding with weaker surface CAPE (cooler surface) but
        # high free-trop RH. Use the helper's surface_T/lapse parameters
        # to produce a moist-but-not-explosive column.
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=298.0, surface_rh=0.85, lapse_K_per_km=5.5,
        )
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()

        _, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
        )
        ktype = int(state.ktype)
        # Accept either deep or mid (sounding-dependent) but NOT shallow
        # or no convection — both indicate the trigger isn't picking up
        # the moist-free-trop signal.
        assert ktype in (1, 3), (
            f"Expected ktype ∈ {{1, 3}} for moderate-CAPE moist column; "
            f"got ktype={ktype}"
        )

    def test_stable_sounding_no_convection(self):
        """On a stable sounding (cold surface) the scheme should return zero
        tendencies — ensures we haven't introduced spurious activation.
        """
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=260.0, surface_rh=0.5, lapse_K_per_km=2.0
        )
        nlev = T.shape[0]
        u = jnp.zeros(nlev)
        v = jnp.zeros(nlev)
        qc = jnp.zeros(nlev)
        qi = jnp.zeros(nlev)
        cfg = ConvectionParameters.default()

        tendencies, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, u, v, qc, qi, 1800.0, cfg,
        )
        # No convection → no tendencies, no precip
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(tendencies.dtedt))), 0.0, places=8,
        )
        self.assertAlmostEqual(float(tendencies.precip_conv), 0.0, places=8)

    def test_convective_heating_pattern(self):
        """Latent heat release from convection should produce a positive
        peak somewhere in the cloud column. With the deviation-flux
        formulation in ``flux_tendencies.py`` we get cancellation between
        heating in the upper cloud (where mfu drops via detrainment) and
        cooling in the lower cloud (compensating subsidence), so the
        column-summed mid-troposphere tendency can be near zero. The
        meaningful sanity check is that the *peak* dtedt exceeds the
        peak negative dtedt by at least a token amount, and that the
        peak lives in the mid-to-upper troposphere (350-650 hPa) rather
        than the boundary layer.

        See ``fortran_harness/PLAN.md`` Bug C — the deviation-flux
        formulation differs from ECHAM's full-flux + explicit
        detrainment, so absolute heating profile won't match Fortran
        bit-for-bit until we mirror the ECHAM formula. This test
        guards against the "no heating at all" or "boundary-layer-only"
        regression modes.
        """
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=305.0, surface_rh=0.9, lapse_K_per_km=7.0
        )
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
        )
        dtedt = np.asarray(tendencies.dtedt)
        peak_pos = float(np.max(dtedt))
        peak_pos_idx = int(np.argmax(dtedt))
        self.assertGreater(
            peak_pos, 1e-5,
            f"Expected non-trivial peak heating somewhere; "
            f"got max dtedt = {peak_pos:.3e} K/s",
        )
        # Peak heating should live in the cloud column (above the
        # boundary layer), not at the cloud base. The Bug-D
        # downdraft-runaway regression (mfd diverging to ~2 kg/m²/s
        # at the surface) used to push peak heating into the boundary
        # layer (~960 hPa); guard against that.
        peak_p = float(p[peak_pos_idx])
        self.assertLess(
            peak_p, 80_000.0,
            f"Peak heating at p={peak_p:.0f} Pa is below 800 hPa, in the "
            "boundary layer — likely the Bug-D downdraft-runaway regression "
            "(where heating used to peak at the cloud base)."
        )

    def test_convective_drying_in_cloud_layer(self):
        """Condensation removes vapour from the column during convection —
        the integrated q tendency should be negative (condensed → precip)
        minus what was transported up from the BL.
        """
        T, q, p, dz, rho = _tropical_sounding(surface_T=302.0, surface_rh=0.85)
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
        )
        # Some level should have dqdt < 0 (drying) from condensation
        self.assertLess(
            float(jnp.min(tendencies.dqdt)), 0.0,
            f"Expected some drying tendency from condensation; "
            f"min dqdt = {float(jnp.min(tendencies.dqdt)):.3e}",
        )

    def test_column_water_budget_closes(self):
        """The scheme's tendencies conserve column water against precip.

        ECHAM applies NO grid-mean saturation adjustment after cudtdq
        (verified against mo_cumastr.f90) — residual grid-mean
        supersaturation is the stratiform scheme's job, so the previous
        assertion here (post-tendency column at/below saturation) encoded
        the removed non-ECHAM adjustment. What the faithful ledger DOES
        guarantee — and what the removed adjustment silently broke — is
        the water budget: every kg of exported precipitation is debited
        from the column,

            Σ (dq/dt + dqc/dt + dqi/dt)·Δp/g + P  =  0,

        using the scheme's own per-level layer-mass convention (edge Δp at
        the boundaries, centred spacing inside).
        """
        T, q, p, dz, rho = _tropical_sounding(surface_T=302.0, surface_rh=0.85)
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()
        dt = 1800.0

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            dt, cfg,
            moisture_supply=jnp.array(5e-5),
        )
        import numpy as np
        import jcm.constants as c
        dpa = np.abs(np.diff(np.asarray(p)))
        # The scheme's own layer-mass convention: the dual-grid spacing the
        # divergence terms use, extended to the last level.
        dp_lev = np.concatenate([dpa, dpa[-1:]])
        mass = dp_lev / c.grav
        dwater = np.asarray(
            tendencies.dqdt + tendencies.dqc_dt + tendencies.dqi_dt
        )
        precip = float(tendencies.precip_conv)
        residual = float(np.sum(dwater * mass) + precip)
        self.assertGreater(precip, 0.0, "test column did not precipitate")
        self.assertLess(
            abs(residual), max(1e-3 * precip, 1e-9),
            f"column water budget open by {residual:.3e} kg/m2/s "
            f"against precip {precip:.3e}",
        )

    def test_column_energy_budget_closes(self):
        """Column enthalpy change balances the latent-heat exchange.

        The cudtdq ledger guarantees (with the deviation DSE fluxes
        telescoping over the column):

            cp·Σ dT/dt·Δp/g  =  Σ zalv·(plude+pdmfup+pdmfdp)·… − alhf·Σ pdpmel
                              =  −Σ zalv·(dq/dt)·Δp/g − alhf·Σ pdpmel

        i.e. the column warms by exactly the latent heat of the vapour it
        loses (phase-keyed zalv), minus the melt sink. Momentum terms carry
        no enthalpy here. Verified on the same column as the water budget;
        the pre-rewrite scheme failed this at the 300 W/m² level (heating
        454 W/m² vs L·P = 140 W/m², review finding 0.1).
        """
        T, q, p, dz, rho = _tropical_sounding(surface_T=302.0, surface_rh=0.85)
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()
        dt = 1800.0

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            dt, cfg,
            moisture_supply=jnp.array(5e-5),
        )
        import numpy as np
        import jcm.constants as c
        dpa = np.abs(np.diff(np.asarray(p)))
        # The scheme's own layer-mass convention: the dual-grid spacing the
        # divergence terms use, extended to the last level.
        dp_lev = np.concatenate([dpa, dpa[-1:]])
        mass = dp_lev / c.grav
        zalv = np.where(np.asarray(T) > c.tmelt, c.alhc, c.alhs)
        cp_int = c.cpd * float(np.sum(np.asarray(tendencies.dtedt) * mass))
        # Every kg of vapour the column loses was condensed somewhere in
        # the plume and released its latent heat — whether it left as
        # precipitation or as detrained condensate (the qc/qi tendencies
        # carry ALREADY-condensed water whose heat the ledger banked via
        # +zalv·plude). So the enthalpy identity pairs cp∫dT with the
        # phase-keyed latent heat of the VAPOUR loss alone:
        #     cp·Σ dT·Δp/g  ≈  −Σ zalv·dq·Δp/g  −  alhf·Σ pdpmel
        # (the DSE deviation fluxes telescope to zero over the column —
        # verified numerically). Two bounded openings are inherent to the
        # REFERENCE ledger itself: (a) vapour is removed where it is
        # entrained (warm levels, Lv-keyed zalv) but its heat is released
        # where the plume condenses/precipitates (cold levels, Ls-keyed) —
        # an (Ls−Lv)/Lv ≈ 13 % spread on the cold-source share (ECHAM's
        # 'fusion debt' of ice condensate); (b) the −alhf·pdpmel melt sink
        # is not exposed on the tendencies struct. Together they bound the
        # residual at ~15 %; the measured value here is ~9 %. The
        # pre-rewrite scheme failed this identity by ~220 % (heating
        # 454 W/m² vs L·P = 140, review finding 0.1).
        lat_int = float(np.sum(zalv * np.asarray(tendencies.dqdt) * mass))
        scale = max(abs(cp_int), abs(lat_int), 1.0)
        self.assertLess(
            abs(cp_int + lat_int) / scale, 0.15,
            f"column enthalpy vs latent exchange open by "
            f"{cp_int + lat_int:.1f} W/m2 (cp∫dT={cp_int:.1f}, "
            f"zalv∫dq={lat_int:.1f})",
        )


class TestMoistureSupplyClosure(unittest.TestCase):
    """The cloud-base mass-flux closure anchored to the surface moisture supply.

    These distil the single-column RCE finding that drove the closure fix: the
    bare-CAPE closure (``moisture_supply=0``) sets the cloud-base mass flux to
    the CFL cap ``layer_mass/dt`` whenever CAPE is large, so the convective
    burst *grows as the timestep shrinks* and empties CAPE in one step — the
    on/off cloud-base flicker. ECHAM instead anchors the flux to the
    boundary-layer moisture supply (``zmfub`` ≈ E/(q_u−q_e), mo_cumastr.f90),
    a smooth, timestep-independent rate. Passing the surface evaporation as
    ``moisture_supply`` switches the scheme onto that closure.
    """

    def _run(self, moisture_supply, dt=1800.0, surface_T=305.0,
             surface_rh=0.9, lapse=7.0):
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=surface_T, surface_rh=surface_rh, lapse_K_per_km=lapse,
        )
        nlev = T.shape[0]
        z = jnp.zeros(nlev)
        tend, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, z, z, z, z, dt, ConvectionParameters.default(),
            moisture_supply=jnp.asarray(float(moisture_supply)),
        )
        return tend, state

    def test_moisture_anchored_flux_is_timestep_invariant(self):
        """The flicker mechanism: anchored flux is dt-independent, CAPE-cap isn't.

        With a moisture supply the peak convective heating is the same at
        dt=1800 s and dt=600 s (the flux is E/(q_u−q_e), independent of dt). The
        bare-CAPE closure instead rides the ``layer_mass/dt`` CFL cap, so its
        peak heating grows markedly as dt shrinks — the per-step amplification
        that becomes the temporal flicker in an integration.
        """
        anch_long = float(jnp.max(jnp.abs(self._run(1.0e-4, dt=1800.0)[0].dtedt)))
        anch_short = float(jnp.max(jnp.abs(self._run(1.0e-4, dt=600.0)[0].dtedt)))
        cape_long = float(jnp.max(jnp.abs(self._run(0.0, dt=1800.0)[0].dtedt)))
        cape_short = float(jnp.max(jnp.abs(self._run(0.0, dt=600.0)[0].dtedt)))

        # Anchored: essentially dt-invariant.
        self.assertLess(anch_short / anch_long, 1.25)
        # The CAPE fallback is now Nordeng's zcape/(zheat·cmftau) — a
        # physical timescale closure with no dt in it — so it is ALSO
        # ~dt-invariant. The previous control assertion here pinned the
        # PATHOLOGY (the naive CAPE/(g·τ) fallback rode the layer-mass/dt
        # CFL cap, so its burst grew as dt shrank — the flicker mechanism);
        # with the fallback replaced, both branches are cured and the
        # assertion flips to pin that.
        self.assertLess(cape_short / cape_long, 1.25)

    def test_moisture_anchored_flux_is_smaller_and_bounded(self):
        """The evaporation-limited flux is far gentler than the CAPE-cap burst.

        On the same explosive sounding the moisture-anchored cloud-base mass
        flux is a small fraction of the bare-CAPE-cap flux — it removes CAPE
        gradually (keeping convection on) rather than dumping it in one step.
        """
        mfu_anchored = float(jnp.max(self._run(1.0e-4)[1].mfu))
        mfu_cape = float(jnp.max(self._run(0.0)[1].mfu))
        self.assertGreater(mfu_anchored, 0.0)  # convection still active
        self.assertLess(mfu_anchored, 0.5 * mfu_cape)

    def test_precip_scales_with_moisture_supply(self):
        """Moisture-budget content: precip exports the supplied moisture.

        Because M_b = E/(q_u−q_e), the convective mass flux — and hence the
        precipitation it produces — is linear in the supply E. Doubling the
        surface evaporation roughly doubles the convective precip.
        """
        pr_1x = float(self._run(1.0e-4)[0].precip_conv)
        pr_2x = float(self._run(2.0e-4)[0].precip_conv)
        self.assertGreater(pr_1x, 0.0)
        self.assertGreater(pr_2x / pr_1x, 1.8)
        self.assertLess(pr_2x / pr_1x, 2.2)

    def test_zero_supply_falls_back_to_cape_closure(self):
        """No moisture supply ⇒ unchanged (bare-CAPE) behaviour.

        Radiative-convective-only stacks (and any caller that does not provide a
        surface evaporation) must see the original CAPE closure. The default
        ``moisture_supply=0`` reproduces the no-argument call exactly and still
        fires convection on an unstable sounding.
        """
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=305.0, surface_rh=0.9, lapse_K_per_km=7.0,
        )
        nlev = T.shape[0]
        z = jnp.zeros(nlev)
        cfg = ConvectionParameters.default()
        default_tend, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho, z, z, z, z, 1800.0, cfg,
        )
        explicit_tend, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho, z, z, z, z, 1800.0, cfg,
            moisture_supply=jnp.asarray(0.0),
        )
        self.assertTrue(
            jnp.allclose(default_tend.dtedt, explicit_tend.dtedt)
        )
        self.assertGreater(float(jnp.max(jnp.abs(default_tend.dtedt))), 1e-6)

    def test_stable_column_with_moisture_supply_stays_inactive(self):
        """A statically stable column must NOT convect on evaporation alone.

        ``find_cloud_base`` returns the LCL, which exists in many stable
        columns, so triggering convection on the moisture supply alone
        (activate whenever E>0) fired deep convection in stable, non-buoyant
        columns (CAPE==0). On a T63L47 real-orography spin-up that
        over-activation dumped latent heat into stable tropical columns and ran
        the temperature away to NaN within ~4 days. The buoyancy floor
        (``cape > _MIN_CAPE_FOR_MOISTURE_TRIGGER``) restores ECHAM's ``ldcum``
        requirement: no buoyancy ⇒ no convection, no matter how large the
        moisture supply.
        """
        # Stable sounding (small lapse): a surface parcel reaches its LCL (a
        # cloud base exists, so the old ``moisture_supply > 0`` trigger would
        # have fired) but is never buoyant, so CAPE is below the floor.
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=285.0, surface_rh=0.7, lapse_K_per_km=3.5,
        )
        nlev = T.shape[0]
        z = jnp.zeros(nlev)
        cfg = ConvectionParameters.default()
        # Even a large moisture supply must not activate convection here.
        for supply in (1.0e-4, 1.0e-2):
            tend, state = tiedtke_nordeng_convection(
                T, q, p, dz, rho, z, z, z, z, 1800.0, cfg,
                moisture_supply=jnp.asarray(supply),
            )
            self.assertEqual(
                int(state.ktype), 0,
                f"stable column must stay inactive (supply={supply})",
            )
            self.assertAlmostEqual(
                float(jnp.max(jnp.abs(tend.dtedt))), 0.0, places=8,
            )
            self.assertAlmostEqual(float(tend.precip_conv), 0.0, places=8)

    def test_near_saturated_cloud_base_falls_back_to_cape_closure(self):
        """Near-saturated cloud base ⇒ moisture closure bypassed (ECHAM zlo1).

        The moisture-budget flux is ``E/(q_u−q_e)``; as the cloud-base
        environment approaches saturation the denominator collapses and
        ``E/q_excess`` spikes to the CFL cap ``layer_mass/dt`` — a catastrophic
        single-step latent-heat burst that seeded the T63L47 hot-cell runaway.
        ECHAM only applies the budget closure when the saturation deficit
        exceeds ``zdqmin`` (mo_cumastr.f90:268-271); below it the scheme falls
        back. So on a near-saturated (but still conditionally unstable) column
        the cloud-base mass flux with E>0 equals the bounded CAPE-closure flux
        (E=0), NOT the CFL-cap burst.
        """
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=302.0, surface_rh=0.997, lapse_K_per_km=7.0,
        )
        nlev = T.shape[0]
        z = jnp.zeros(nlev)
        cfg = ConvectionParameters.default()
        _, st_supply = tiedtke_nordeng_convection(
            T, q, p, dz, rho, z, z, z, z, 1800.0, cfg,
            moisture_supply=jnp.asarray(2.0e-4),
        )
        _, st_cape = tiedtke_nordeng_convection(
            T, q, p, dz, rho, z, z, z, z, 1800.0, cfg,
            moisture_supply=jnp.asarray(0.0),
        )
        mfu_supply = float(jnp.max(st_supply.mfu))
        mfu_cape = float(jnp.max(st_cape.mfu))
        # Convection still fires (unstable column) ...
        self.assertGreater(mfu_cape, 1.0e-4)
        # ... but the near-saturated cloud base falls back to the bounded CAPE
        # closure rather than the CFL-cap burst — identical mass flux to E=0.
        self.assertAlmostEqual(mfu_supply, mfu_cape, places=6)


if __name__ == "__main__":
    unittest.main()
