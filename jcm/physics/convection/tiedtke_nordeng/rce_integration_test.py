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

from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng_test import (  # noqa: E501
    deep_convection_drivers,
)
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

    # T: well-mixed (dry-adiabatic) boundary layer below 0.8 km, the standard
    # lapse rate above, isothermal above 15 km.
    #
    # The mixed layer is required, not cosmetic. ECHAM's ``cubase`` walks a
    # dry parcel upward and drops the column the moment it is not buoyant, so
    # a sounding running 6.5 K/km right down to the surface loses ~0.4 K of
    # parcel buoyancy per layer — more than any physical ``zlift`` — and
    # cannot trigger at all. Real boundary layers, and the ones jcm's own
    # vdiff produces in the coupled model, are near-neutral near the surface;
    # that is precisely the condition the trigger tests for.
    bl_top_km = 0.8
    dry_lapse = 9.81 / 1004.64          # K/m
    T = jnp.where(
        z_km < bl_top_km,
        surface_T - dry_lapse * 1000.0 * z_km,
        surface_T - dry_lapse * 1000.0 * bl_top_km
        - lapse_K_per_km * (z_km - bl_top_km),
    )
    T = jnp.maximum(T, 200.0)

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


def _interfaces(p):
    """Interface pressures for a sounding: full-level midpoints inside, the
    top and bottom extrapolated by half a layer (top clamped at 0 Pa).
    """
    from jcm.physics.convection.tiedtke_nordeng.half_levels import (
        reconstruct_pressure_half,
    )
    return reconstruct_pressure_half(p)


class TestRCEConvection(unittest.TestCase):
    """Full-scheme RCE-style integration tests."""

    def test_tropical_sounding_fires_convection(self):
        """A buoyant sounding fed by surface evaporation convects:
        - non-zero tendencies
        - positive precipitation
        - non-zero updraft mass flux

        ECHAM convects where ``cubase`` finds a buoyant cloud base AND the
        sub-cloud layer gains moisture (``zdqpbl > 0``, mo_cumastr.f90:565),
        so the column is given the evaporation and convergence of a deep
        tropical column, as the model would supply them.
        """
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
            **deep_convection_drivers({'pressure': p}),
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

    def test_ktype_follows_the_echam_triggers_not_a_cape_proxy(self):
        """Ktype records ECHAM's trigger + moisture-budget test (#697/#699).

        History: this test used to demand ktype ∈ {1, 3} for a
        moderate-CAPE column with a moist free troposphere, pinning an
        RH-based proxy that relabelled surface plumes "mid-level" to stand
        in for the missing ``cubasmc`` trigger. Both proxies are gone —
        the real ``cubasmc`` needs resolved ascent (omega < 0; covered in
        ``midlevel_trigger_test``), and deep-vs-shallow is the
        moisture-convergence integral. So the SAME column is now,
        correctly: shallow on its own surface flux, deep with resolved
        convergence beyond 0.1*E.
        """
        # 90 % boundary-layer humidity: a shallow plume entrains at
        # ``entrscv`` and ends its ascent at the first interface where the
        # diluted parcel no longer condenses, so at 85 % the supply-only
        # plume dies in its first layer (non-convective, as in ECHAM).
        atm_args = _tropical_sounding(
            surface_T=298.0, surface_rh=0.9, lapse_K_per_km=5.5,
        )
        T, q, p, dz, rho = atm_args
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()
        atm = {"rho": rho, "layer_thickness": dz, "pressure": p}

        drivers = deep_convection_drivers(atm)
        _, s_supply_only = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
            moisture_supply=drivers["moisture_supply"],
            moisture_tend_profile=drivers["moisture_tend_profile"],
        )
        _, s_convergent = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
            **drivers,
        )
        assert int(s_supply_only.ktype) == 2, (
            f"supply-only column must be shallow, got "
            f"{int(s_supply_only.ktype)}")
        assert int(s_convergent.ktype) == 1, (
            f"convergent column must be deep, got "
            f"{int(s_convergent.ktype)}")

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

        See ``fortran_harness/PLAN.md`` Bug C (on the
        ``origin/fortran-harness-vdiff`` branch, not in this tree) —
        the deviation-flux
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
            # Deep via ECHAM's zdqcv route (#699): the mid-troposphere heating
            # peak this test pins is a DEEP plume property.
            **deep_convection_drivers(
                {'rho': rho, 'layer_thickness': dz, 'pressure': p}),
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
            **deep_convection_drivers({'pressure': p}),
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

        with ``Δp/g`` the TRUE layer mass between the column's interfaces —
        the mass a host integrates the tendencies with (#530).
        """
        T, q, p, dz, rho = _tropical_sounding(surface_T=302.0, surface_rh=0.85)
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()
        dt = 1800.0
        p_half = _interfaces(p)

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            dt, cfg,
            moisture_supply=jnp.array(5e-5),
            pressure_half=p_half,
        )
        import numpy as np
        import jcm.constants as c
        mass = np.diff(np.asarray(p_half)) / c.grav
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

    def test_water_budget_closes_with_subcloud_evaporation(self):
        """Sub-cloud rain evaporation must cool/moisten, not just eat precip.

        With an elevated cloud base over a dry sub-cloud layer, the cuflx
        Kessler chain evaporates falling rain below cloud base (negative
        ``pdmfup`` increments). The Codex review on #550 caught that the
        original conv_mask truncated the per-level ledger at cloud base:
        the surface precip was depleted by the evaporation while its
        cooling/moistening was zeroed — silently drying the column. ECHAM's
        cudtdq loop runs to the SURFACE; the mask now applies only to the
        in-cloud flux-divergence terms, and this test pins both the closed
        budget and the sub-cloud moistening on exactly that configuration.
        """
        import numpy as np
        import jcm.constants as c
        # Moderate surface RH lifts the LCL to ~850 hPa, so the cloud base
        # is elevated and the (subsaturated) layers beneath it evaporate
        # the falling rain — the configuration the truncated mask broke.
        T, q, p, dz, rho = _tropical_sounding(surface_T=303.0, surface_rh=0.55)
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()

        # Deep classification via the zdqcv route (#699): the elevated
        # cloud base + rain-through-dry-layer configuration needs a deep
        # plume; supply-only is now (correctly) shallow with little rain.
        p_half = _interfaces(p)
        drivers = deep_convection_drivers(
            {"rho": rho, "layer_thickness": dz, "pressure_half": p_half},
            e_sfc=5e-5)
        tendencies, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
            pressure_half=p_half,
            **drivers,
        )
        precip = float(tendencies.precip_conv)

        mass = np.diff(np.asarray(p_half)) / c.grav
        dwater = np.asarray(
            tendencies.dqdt + tendencies.dqc_dt + tendencies.dqi_dt
        )
        gross = float(np.sum(np.abs(dwater) * mass)) + abs(precip)
        # Anti-vacuity: with the corrected ECHAM ice saturation (#547) the
        # deep dry sub-cloud layer evaporates ALL the rain before the
        # surface (precip is exactly 0 here) — the configuration this test
        # guards is precisely that evaporation, so the guard is that it
        # fired (sub-cloud moistening below), not that rain survives.
        self.assertGreater(gross, 0.0, "column did nothing — vacuous")
        residual = float(np.sum(dwater * mass) + precip)
        self.assertLess(
            abs(residual), max(1e-4 * gross, 1e-9),
            f"water budget open by {residual:.3e} with sub-cloud "
            f"evaporation active (precip {precip:.3e}, gross {gross:.3e}) "
            f"— the evaporation's moistening is being masked out",
        )
        # And the evaporation genuinely moistens the sub-cloud layer (the
        # pre-fix mask zeroed exactly these levels). ``kbase`` is the
        # cloud-base INTERFACE — the top of layer kbase — so the sub-cloud
        # layers are kbase and below. Their water budget is the total-water
        # flux they export upward through that interface (plume vapour and
        # condensate against the half-level environment, plus the downdraft
        # it receives) plus the rain evaporated into them, so the sum of the
        # sub-cloud water change and that export IS the evaporation, which
        # must be positive. (The cuflx sub-cloud taper spreads the plume's
        # cloud-base export through these layers, so their own tendency can
        # still be net drying.)
        from jcm.physics.convection.tiedtke_nordeng.updraft import (
            column_environment,
        )
        env = column_environment(T, q, p, pressure_half=p_half)
        kb = int(state.kbase)
        export = float(
            state.mfu[kb] * (state.qu[kb] + state.lu[kb] - env.qenh[kb])
            + state.mfd[kb] * (state.qd[kb] - env.qenh[kb])
        )
        sub_cloud = np.arange(nlev) >= kb
        evaporated = float(np.sum((dwater * mass)[sub_cloud])) + export
        self.assertGreater(
            evaporated, 0.0,
            "no sub-cloud moistening despite rain falling through a dry "
            "layer",
        )

    def test_column_energy_budget_closes(self):
        """Column enthalpy change balances the latent-heat exchange.

        The cudtdq ledger guarantees (with the deviation DSE fluxes
        telescoping over the column):

            Σ cp·dT/dt·Δp/g  =  Σ zalv·(plude+pdmfup+pdmfdp)·… − alhf·Σ pdpmel
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
        p_half = _interfaces(p)

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            dt, cfg,
            moisture_supply=jnp.array(5e-5),
            pressure_half=p_half,
        )
        import numpy as np
        import jcm.constants as c
        # The true layer mass between the column's interfaces.
        mass = np.diff(np.asarray(p_half)) / c.grav
        zalv = np.where(np.asarray(T) > c.tmelt, c.alhc, c.alhs)
        # The ledger converts heat to temperature with ECHAM's MOIST
        # ``pcpen = cpd·(1 + vtmpc2·q)`` (``zrcpm``, mo_cufluxdts.f90:648),
        # so the enthalpy it deposits is integrated with the same ``cp``.
        cp_moist = c.cpd * (1.0 + c.vtmpc2 * np.maximum(np.asarray(q), 0.0))
        cp_int = float(np.sum(cp_moist * np.asarray(tendencies.dtedt) * mass))
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
    """The cloud-base mass flux anchored to the sub-cloud moisture budget.

    ECHAM's first-guess cloud-base flux is ``zmfub = zdqpbl/(g·MAX(zqumqe,
    zdqmin))`` (mo_cumastr.f90:560-569): the moisture the sub-cloud layer
    gains per step, exported by the cloud-base parcel's water excess. The
    same budget decides whether a ``cubase`` column convects at all: the
    ``zlo1`` gate requires ``zdqpbl > 0`` and ``zqumqe > zdqmin``, and a
    column that fails it is not convective (``ldcum = .FALSE.``). A caller
    that gives only the surface evaporation ``moisture_supply`` has it
    delivered to the lowest layer, which is where vertical diffusion's
    surface row puts it.
    """

    def _run(self, moisture_supply, dt=1800.0, surface_T=305.0,
             surface_rh=0.9, lapse=7.0, deep=False):
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=surface_T, surface_rh=surface_rh, lapse_K_per_km=lapse,
        )
        nlev = T.shape[0]
        z = jnp.zeros(nlev)
        extra = {}
        if deep:
            # Resolved convergence > 0.1*supply so ECHAM's zdqcv test
            # (#699) classifies deep; the closure-path comparisons this class
            # makes are otherwise about the SUPPLY argument, which stays the
            # sole variable.
            sl = slice(nlev // 2, nlev - 4)
            conv = jnp.zeros(nlev).at[sl].set(
                1.3 * float(moisture_supply) / jnp.sum(rho[sl] * dz[sl]))
            extra["qte_dynamics"] = conv
        tend, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, z, z, z, z, dt, ConvectionParameters.default(),
            moisture_supply=jnp.asarray(float(moisture_supply)),
            **extra,
        )
        return tend, state

    def test_moisture_anchored_flux_is_timestep_invariant(self):
        """The anchored flux is ``zdqpbl/Δq``, which has no ``dt`` in it.

        The peak convective heating is the same at dt = 1800 s and 600 s to
        within the ``zmfmax = layer_mass/dt`` limiter, which can still touch
        the 600 s step on this explosive sounding (by design); a closure that
        rode that limiter would give a ratio of 3.
        """
        anch_long = float(jnp.max(jnp.abs(self._run(1.0e-4, dt=1800.0)[0].dtedt)))
        anch_short = float(jnp.max(jnp.abs(self._run(1.0e-4, dt=600.0)[0].dtedt)))
        self.assertGreater(anch_long, 0.0)
        self.assertLess(anch_short / anch_long, 1.5)

    def test_moisture_anchored_flux_is_bounded(self):
        """The evaporation-limited flux keeps convection on and bounded."""
        mfu_anchored = float(jnp.max(self._run(1.0e-4)[1].mfu))
        self.assertGreater(mfu_anchored, 0.0)
        self.assertLess(mfu_anchored, 5.0)

    def test_deep_precip_is_set_by_the_rescale_not_the_supply(self):
        """Deep precipitation is set by the Nordeng rescale, not the supply.

        The moisture budget is only the FIRST GUESS; Nordeng's
        ``zmfub1 = zcape·zmfub/(zheat·cmftau)`` sets every deep column's
        amplitude, and on this explosive sounding it saturates the closure
        clip at any supply, so doubling the supply leaves the convective
        precipitation almost unchanged, as in ECHAM.
        """
        pr_1x = float(self._run(2.0e-5, deep=True)[0].precip_conv)
        pr_2x = float(self._run(4.0e-5, deep=True)[0].precip_conv)
        self.assertGreater(pr_1x, 0.0)
        self.assertLess(abs(pr_2x / pr_1x - 1.0), 0.1)

    def test_no_sub_cloud_supply_no_convection(self):
        """A buoyant column whose sub-cloud layer gains no moisture is dry.

        ``zlo1`` fails where ``zdqpbl <= 0`` (mo_cumastr.f90:565): however
        buoyant the parcel, ECHAM makes the column non-convective. The
        default ``moisture_supply = 0`` reproduces the explicit zero.
        """
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=305.0, surface_rh=0.9, lapse_K_per_km=7.0,
        )
        nlev = T.shape[0]
        z = jnp.zeros(nlev)
        cfg = ConvectionParameters.default()
        for kwargs in ({}, {"moisture_supply": jnp.asarray(0.0)}):
            tend, state = tiedtke_nordeng_convection(
                T, q, p, dz, rho, z, z, z, z, 1800.0, cfg, **kwargs)
            self.assertEqual(int(state.ktype), 0)
            self.assertEqual(float(jnp.max(jnp.abs(tend.dtedt))), 0.0)
            self.assertEqual(float(tend.precip_conv), 0.0)
        # ... and the same column with a supply convects.
        self.assertGreater(int(self._run(1.0e-5)[1].ktype), 0)

    def test_stable_column_with_moisture_supply_stays_inactive(self):
        """A statically stable column does not convect on evaporation alone.

        A surface parcel on this sounding reaches its LCL but is not buoyant
        there, so ``cubase`` finds no cloud base, and no moisture supply
        makes the column convective (ECHAM's ``ldcum`` needs the cloud base).
        """
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


if __name__ == "__main__":
    unittest.main()
