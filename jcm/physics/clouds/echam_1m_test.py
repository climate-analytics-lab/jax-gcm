"""Unit tests for the ECHAM 1-moment cloud microphysics scheme."""

import jax.numpy as jnp
import jax
import numpy as np
import pytest
from .echam_1m import (
    MicrophysicsParameters,
    autoconversion, autoconversion_beheng, autoconversion_kk2000,
    ice_autoconversion, sedimentation_flux,
    cloud_microphysics_column_sweep,
)
from jcm.constants import tmelt


class TestAutoconversion:
    """Test autoconversion processes"""
    
    def test_autoconversion_no_water_no_rate(self):
        """Beheng autoconversion gives essentially zero rate at near-zero qc."""
        config = MicrophysicsParameters.default()
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(0.5)
        droplet_number = jnp.array(100e6)
        dt = 1800.0

        # qc = 0 → no autoconversion
        rate_zero = autoconversion_beheng(
            jnp.array(0.0), cloud_fraction, air_density, droplet_number, dt, config,
        )
        assert float(rate_zero) < 1e-15

        # qc = tiny (1e-7 kg/kg) → effectively no autoconversion
        rate_tiny = autoconversion_beheng(
            jnp.array(1e-7), cloud_fraction, air_density, droplet_number, dt, config,
        )
        assert float(rate_tiny) < 1e-12

        # qc = realistic post-convection (0.6 g/kg) → meaningful rate but
        # bounded by mass conservation (cannot deplete more than qc/dt).
        qc = jnp.array(0.6e-3)
        rate_high = autoconversion_beheng(
            qc, cloud_fraction, air_density, droplet_number, dt, config,
        )
        assert float(rate_high) > 0.0
        assert float(rate_high) <= float(qc) / dt + 1e-12, (
            "Beheng integral form must respect mass conservation: "
            "autoconv rate cannot exceed qc/dt."
        )

    def test_autoconversion_dependencies(self):
        """Beheng autoconversion: rate increases with qc, decreases with Nc.

        Note: with the implicit-integration formulation, large qc gets
        capped at the mass-conservation limit qc/dt, so the "rate
        increases with qc" check uses a short timestep where the rate
        hasn't saturated yet.
        """
        config = MicrophysicsParameters.default()
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(1.0)
        dt = 0.1  # short timestep so rate doesn't saturate at qc/dt

        rate_low_qc = autoconversion_beheng(
            jnp.array(0.4e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        rate_high_qc = autoconversion_beheng(
            jnp.array(0.8e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        assert float(rate_high_qc) > float(rate_low_qc), (
            "Higher qc → higher Beheng autoconversion rate"
        )

        # Droplet number dependence: more droplets (cleaner air) → slower
        # autoconversion (Nc^-3.3 in the formula).
        rate_few_droplets = autoconversion_beheng(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(50e6), dt, config,
        )
        rate_many_droplets = autoconversion_beheng(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(500e6), dt, config,
        )
        assert float(rate_few_droplets) > float(rate_many_droplets), (
            "Fewer cloud droplets → faster autoconversion (Beheng Nc^-3.3)"
        )


class TestKK2000Autoconversion:
    """KK2000 explicit-rate autoconversion + dispatcher tests."""

    def test_below_threshold_negligible(self):
        """Sub-threshold autoconversion is negligible, not exactly zero.

        The ccraut gate is a sigmoid ramp now (maintainability review
        B.2.5) so the threshold is calibratable; several widths below
        it the residual rate must be a vanishing fraction of the
        above-threshold rate, and at exactly zero cloud water the rate
        (and its gradient path) must be exactly zero.
        """
        config = MicrophysicsParameters.default(
            ccraut_kk_threshold=1e-3, autoconversion_scheme="kk2000",
        )
        # qc/cf = 2e-6 in-cloud, ~20 widths below the 1e-3 threshold.
        rate_below = autoconversion_kk2000(
            jnp.array(1e-6),
            jnp.array(0.5), jnp.array(1.0),
            jnp.array(100e6), 1800.0, config,
        )
        rate_above = autoconversion_kk2000(
            jnp.array(1e-3),               # in-cloud 2e-3, above threshold
            jnp.array(0.5), jnp.array(1.0),
            jnp.array(100e6), 1800.0, config,
        )
        assert float(rate_below) < 1e-6 * float(rate_above)
        # Exactly-zero cloud water stays exactly zero (double-where guard).
        rate_zero = autoconversion_kk2000(
            jnp.array(0.0), jnp.array(0.5), jnp.array(1.0),
            jnp.array(100e6), 1800.0, config,
        )
        assert float(rate_zero) == 0.0

    def test_dependencies(self):
        """KK2000: rate ∝ qc^2.47, ∝ Nc^-1.79 — same monotonicity as Beheng."""
        config = MicrophysicsParameters.default(
            ccraut_kk_threshold=1e-5, autoconversion_scheme="kk2000",
        )
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(1.0)
        dt = 1800.0

        rate_lo_qc = autoconversion_kk2000(
            jnp.array(0.4e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        rate_hi_qc = autoconversion_kk2000(
            jnp.array(0.8e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        assert float(rate_hi_qc) > float(rate_lo_qc)

        rate_few_drops = autoconversion_kk2000(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(50e6), dt, config,
        )
        rate_many_drops = autoconversion_kk2000(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(500e6), dt, config,
        )
        assert float(rate_few_drops) > float(rate_many_drops)

    def test_dispatcher_picks_scheme(self):
        """``autoconversion(...)`` dispatches by ``config.autoconversion_scheme``."""
        qc = jnp.array(0.6e-3)
        cloud_fraction = jnp.array(0.5)
        air_density = jnp.array(1.0)
        droplet_number = jnp.array(100e6)
        dt = 1800.0

        cfg_beheng = MicrophysicsParameters.default(autoconversion_scheme="beheng")
        cfg_kk2000 = MicrophysicsParameters.default(
            ccraut_kk_threshold=1e-5, autoconversion_scheme="kk2000",
        )

        rate_via_dispatcher_beheng = autoconversion(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_beheng,
        )
        rate_direct_beheng = autoconversion_beheng(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_beheng,
        )
        assert jnp.allclose(rate_via_dispatcher_beheng, rate_direct_beheng)

        rate_via_dispatcher_kk = autoconversion(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_kk2000,
        )
        rate_direct_kk = autoconversion_kk2000(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_kk2000,
        )
        assert jnp.allclose(rate_via_dispatcher_kk, rate_direct_kk)

        # Sanity: the two schemes give different rates on the same column
        assert not jnp.allclose(rate_via_dispatcher_beheng, rate_via_dispatcher_kk)

    def test_kk2000_active_at_defaults(self):
        """Selecting kk2000 WITHOUT overriding any threshold must convert.

        Regression for #674: when one ``ccraut`` field served as both the
        Beheng prefactor (15.0) and the KK2000 qc threshold, kk2000 at
        defaults evaluated sigmoid((qc - 15)/5e-5) = 0 for any physical qc
        — autoconversion silently off. With the split
        ``ccraut_kk_threshold`` (1e-5 kg/kg) default, physical stratiform
        cloud water (1e-4..1e-3 kg/kg in-cloud) must produce a clearly
        nonzero rate.
        """
        config = MicrophysicsParameters.default(autoconversion_scheme="kk2000")
        for qc_grid in (1e-4, 5e-4, 1e-3):
            rate = autoconversion_kk2000(
                jnp.array(qc_grid), jnp.array(1.0), jnp.array(1.0),
                jnp.array(100e6), 1800.0, config,
            )
            # The un-split parameter gave exactly 0.0 here; a meaningful
            # KK2000 rate at these qc is >> 1e-12 kg/kg/s.
            assert float(rate) > 1e-12, (
                f"kk2000 autoconversion dead at qc={qc_grid}"
            )

    def test_beheng_defaults_do_not_read_kk_threshold(self):
        """The Beheng path at defaults is unaffected by the KK threshold."""
        qc = jnp.array(0.6e-3)
        cf = jnp.array(0.5)
        rho = jnp.array(1.0)
        nc = jnp.array(100e6)
        cfg = MicrophysicsParameters.default()
        cfg_weird_kk = MicrophysicsParameters.default(ccraut_kk_threshold=123.0)
        r1 = autoconversion(qc, cf, rho, nc, 1800.0, cfg)
        r2 = autoconversion(qc, cf, rho, nc, 1800.0, cfg_weird_kk)
        assert float(r1) == float(r2)
        assert float(r1) > 0.0

    def test_scheme_int_alias(self):
        """SCHEME_BEHENG / SCHEME_KK2000 ints round-trip with string aliases."""
        cfg_str = MicrophysicsParameters.default(autoconversion_scheme="kk2000")
        cfg_int = MicrophysicsParameters.default(
            autoconversion_scheme=MicrophysicsParameters.SCHEME_KK2000,
        )
        assert int(cfg_str.autoconversion_scheme) == MicrophysicsParameters.SCHEME_KK2000
        assert int(cfg_int.autoconversion_scheme) == int(cfg_str.autoconversion_scheme)

    def test_legacy_kk2000_ccraut_override_raises(self):
        """A legacy ``ccraut``-as-threshold KK2000 config must fail loudly.

        Before the #674 split, KK2000 configs documented ``ccraut`` AS the qc
        threshold. Such a config now silently ignores the override (the KK2000
        branch reads ``ccraut_kk_threshold``), so construction must raise a
        migration error naming the new field rather than run at the 1e-5
        default.
        """
        with pytest.raises(ValueError, match="ccraut_kk_threshold"):
            MicrophysicsParameters.default(
                autoconversion_scheme="kk2000", ccraut=1e-3,
            )

    def test_kk2000_new_field_override_is_accepted(self):
        """Overriding the dedicated KK2000 field constructs cleanly."""
        cfg = MicrophysicsParameters.default(
            autoconversion_scheme="kk2000", ccraut_kk_threshold=1e-3,
        )
        assert float(cfg.ccraut_kk_threshold) == pytest.approx(1e-3)

    def test_beheng_ccraut_override_is_untouched_by_the_guard(self):
        """A ccraut override under Beheng (the field's real owner) is fine."""
        cfg = MicrophysicsParameters.default(
            autoconversion_scheme="beheng", ccraut=1e-3,
        )
        assert float(cfg.ccraut) == pytest.approx(1e-3)

    def test_ice_autoconversion(self):
        """Levkov aggregation properties (ECHAM mo_cloud.f90:996-1052).

        The previous placeholder had a −15 °C Gaussian efficiency peak
        and a hard 0.3 g/kg qi threshold — neither exists in the Levkov
        chain, whose rate grows with the ice content (Moss radius) and
        is temperature-independent at this stage (the T dependence sits
        in the downstream aggregation-by-snow collection efficiency).
        Pins: monotone in IWC, implicitly bounded (depletion ≤ qi even
        at absurd dt), substantial at cirrus-anvil ice contents (the
        placeholder's e-folding was ~30 days — effectively no sink).
        """
        config = MicrophysicsParameters.default()
        cloud_fraction = jnp.array(0.7)
        dt = 1800.0
        t = tmelt - 40.0
        rho = jnp.array(0.5)

        rate_lo = ice_autoconversion(0.2e-3 * 0.7, t, cloud_fraction, dt, config, air_density=rho)
        rate_hi = ice_autoconversion(1.0e-3 * 0.7, t, cloud_fraction, dt, config, air_density=rho)
        assert float(rate_hi) > float(rate_lo) > 0.0

        # Implicit integration: even with dt = 1 day the depletion cannot
        # exceed the available ice.
        rate_huge_dt = ice_autoconversion(
            1.0e-3 * 0.7, t, cloud_fraction, 86400.0, config, air_density=rho)
        assert float(rate_huge_dt) * 86400.0 <= 1.0e-3 * 0.7 + 1e-9

        # Physically meaningful sink: ≥ 10 % of the in-cloud ice per
        # 1800 s step at 1 g/kg in-cloud (the review measured ~57 %).
        depletion_frac = float(rate_hi) * dt / (1.0e-3 * 0.7)
        assert depletion_frac > 0.1


class TestSedimentation:
    """Test sedimentation processes"""
    
    def test_sedimentation_flux(self):
        """Test basic sedimentation flux calculation"""
        nlev = 10
        # Decreasing hydrometeor content with height (realistic)
        hydrometeor = jnp.linspace(1e-3, 0.1e-3, nlev)  # kg/kg
        air_density = jnp.ones(nlev) * 1.0     # kg/m³
        dz = jnp.ones(nlev) * 100.0            # m
        vt = jnp.ones(nlev) * 1.0              # m/s
        dt = 100.0  # Longer timestep to avoid CFL issues
        
        flux, tendency = sedimentation_flux(hydrometeor, air_density, dz, vt, dt)
        
        # Check flux shape
        assert flux.shape == (nlev + 1,)
        assert tendency.shape == (nlev,)
        
        # Top flux should be zero (no input from above)
        assert flux[0] == 0
        
        # Surface flux should be positive
        assert flux[-1] > 0
        
        # Top level loses mass (no input from above)
        assert tendency[0] < 0
        
        # Conservation check: total mass change equals surface flux
        # tendency is in kg/kg/s, need to convert to kg/m²/s
        total_mass_change = jnp.sum(tendency * air_density * dz)  # kg/m²/s
        # Surface flux is already in kg/m²/s
        assert jnp.abs(total_mass_change + flux[-1]) < 1e-6


class TestColumnSweepMicrophysics:
    """Tests for the ICON ``mo_cloud.f90`` column-sweep port.

    The column sweep propagates rain (``zrfl``) and snow (``zsfl``) as
    downward fluxes top-to-bottom inside a single timestep. These tests
    cover the column-budget invariants (surface flux = column source,
    closed water budget) that depend on that in-step flux coupling.
    """

    @staticmethod
    def _column(nlev=20, qc_top=None, qi_top=None, T_profile=None):
        """Build a column with optional cloud water/ice loading."""
        T = jnp.linspace(220.0, 295.0, nlev) if T_profile is None else T_profile
        p = jnp.linspace(20000.0, 100000.0, nlev)
        q = jnp.full(nlev, 5e-3)
        qc = jnp.zeros(nlev) if qc_top is None else qc_top
        qi = jnp.zeros(nlev) if qi_top is None else qi_top
        cf = jnp.where((qc + qi) > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        return T, q, p, qc, qi, cf, rho, dz, ndrop

    def test_no_clouds_no_precip(self):
        """A column with zero qc/qi must produce zero surface precip."""
        cfg = MicrophysicsParameters.default()
        T, q, p, qc, qi, cf, rho, dz, ndrop = self._column()
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        assert float(state.precip_rain) == 0.0
        assert float(state.precip_snow) == 0.0

    def test_warm_cloud_makes_surface_rain(self):
        """A liquid cloud aloft in a warm, near-saturated column produces rain.

        Background ``q`` is set to ~95% of saturation everywhere so that
        Rotstayn rain evaporation cannot consume the full precipitation
        flux before it reaches the surface.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.95 * qsw
        qc = jnp.zeros(nlev).at[5].set(2e-3)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        # Rain at surface, no snow (column never goes below freezing).
        assert float(state.precip_rain) > 1e-6
        assert float(state.precip_snow) == 0.0

    def test_accretion_rate_diagnosed_when_rain_falls_through_cloud(self):
        """Rain falling through a cloudy deck must report accretion.

        Regression for the Codex finding on #604: ``accretn`` was
        published as zero because the per-level accretion rate never
        left the sweep. A liquid deck spanning several near-saturated
        levels forms rain aloft that falls through cloud below, so
        both in-cloud (zrac2) and below-anvil (zrac1) accretion fire.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.95 * qsw
        qc = jnp.zeros(nlev).at[5:12].set(1e-3)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        assert state.accretion_rate.shape == (nlev,)
        assert float(jnp.max(state.accretion_rate)) > 0.0
        # No accretion outside the deck.
        assert float(jnp.max(state.accretion_rate[:5])) == 0.0

    def test_subsaturated_column_evaporates_rain(self):
        """A dry column under a cloud must evaporate falling rain.

        Setup: a near-saturated cloud aloft (so the in-sweep saturation
        adjustment doesn't immediately evaporate the cloud water before
        autoconv fires) with strongly sub-saturated layers below the
        cloud. Rotstayn rain evap should consume some of the falling
        rain so the surface flux is strictly less than the
        column-integrated rain source.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 300.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        # Near-saturated where the cloud lives (level 5); dry below.
        cloud_level = 5
        q = jnp.where(
            jnp.arange(nlev) == cloud_level, 0.95 * qsw, 0.3 * qsw,
        )
        qc = jnp.zeros(nlev).at[cloud_level].set(2e-3)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        rain_source_total = float(jnp.sum(state.rain_source))
        # surface precip should be strictly LESS than the local rain
        # source when rain evap is active in subsaturated air below cloud.
        assert rain_source_total > 0.0, "autoconv didn't fire — adjust q profile"
        assert float(state.precip_rain) < rain_source_total

    def test_rain_evap_flux_closes_the_warm_flux_ledger(self):
        """The exposed per-level rain evaporation closes the rain-flux ledger.

        On an all-warm column (no snow, no melt) the sweep's flux update is
        exactly ``rain_flux[k] = rain_flux[k-1] + rain_source[k] -
        rain_evap_flux[k]``, so the new #499 diagnostic must satisfy that
        identity level by level, with evaporation active (nonzero) in the
        sub-saturated layers below the cloud and never exceeding the
        available flux.
        """
        import numpy as np
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 300.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        # Moist enough below cloud that evap never zeroes the flux (the
        # ledger clamp stays slack), dry enough that evap is nonzero.
        cloud_level = 5
        q = jnp.where(
            jnp.arange(nlev) == cloud_level, 0.95 * qsw, 0.6 * qsw,
        )
        qc = jnp.zeros(nlev).at[cloud_level].set(2e-3)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        assert float(jnp.sum(state.snow_source)) == 0.0
        assert float(jnp.max(state.rain_evap_flux)) > 0.0
        assert float(jnp.min(state.rain_evap_flux)) >= 0.0
        rain_in = jnp.concatenate([jnp.zeros(1), state.rain_flux[:-1]])
        # atol covers f32 round-off relative to the ~1e-4 kg/m²/s flux
        # scale (observed residual ~2.5e-13 where evap consumes ~all of
        # the inflow).
        np.testing.assert_allclose(
            np.asarray(state.rain_flux),
            np.asarray(rain_in + state.rain_source - state.rain_evap_flux),
            rtol=1e-6, atol=1e-10,
        )
        assert float(state.rain_flux.min()) >= 0.0

    def test_snow_above_warm_layer_melts_to_rain(self):
        """Snow flux generated aloft melts as it falls into T>273K layers."""
        cfg = MicrophysicsParameters.default()
        nlev = 20
        # Cold above (level 3, 240K), warm below (>273K from level 8 down).
        T = jnp.concatenate([
            jnp.linspace(220.0, 260.0, 8),
            jnp.linspace(280.0, 295.0, nlev - 8),
        ])
        qi = jnp.zeros(nlev).at[3].set(5e-4).at[4].set(3e-4)
        T_p, q, p, qc, qi_arr, cf, rho, dz, ndrop = self._column(
            nlev=nlev, qi_top=qi, T_profile=T,
        )
        _, state = cloud_microphysics_column_sweep(
            T_p, q, p, qc, qi_arr, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        # The aloft ice → snow flux is small; what matters is that the
        # warm layers melt all of it before the surface, so surface snow
        # is nearly all melted while surface rain is positive. With the
        # corrected melt energetics (finding 2.11: melting now pays the
        # latent heat of fusion, capped by the layer's heat content) a
        # ~1 % residual of unmelted snow survives a single warm layer —
        # physical, unlike the pre-fix free melting that zeroed it.
        assert float(state.precip_snow) < 0.02 * float(state.precip_rain)
        # Some ice was autoconverted to snow → melted → rain.
        assert float(state.precip_rain) >= 0.0

    def test_zero_dt_dependence_on_thicker_column(self):
        """Sanity: thicker layers ≠ instability; precip should be finite."""
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 295.0, nlev)
        qc = jnp.zeros(nlev).at[5].set(2e-3)
        T_p, q, p, qc_arr, qi, cf, rho, dz, ndrop = self._column(
            nlev=nlev, qc_top=qc, T_profile=T,
        )
        for dz_val in (200.0, 1000.0, 2000.0):
            dz_v = jnp.full(nlev, dz_val)
            _, state = cloud_microphysics_column_sweep(
                T_p, q, p, qc_arr, qi, cf, rho, dz_v, ndrop, dt=1800.0, config=cfg,
            )
            assert jnp.isfinite(state.precip_rain)
            assert float(state.precip_rain) >= 0.0

    def test_jit_and_vmap(self):
        """Column sweep must be jit-able and vmap-able (matches per-level)."""
        cfg = MicrophysicsParameters.default()
        nlev = 15
        T_p, q, p, qc, qi, cf, rho, dz, ndrop = self._column(nlev=nlev)
        qc = qc.at[5].set(1e-3)

        f = jax.jit(cloud_microphysics_column_sweep, static_argnames=())
        _, state_jit = f(T_p, q, p, qc, qi, cf, rho, dz, ndrop, 1800.0, cfg)
        assert jnp.isfinite(state_jit.precip_rain)

        # Stack 4 columns and vmap over column axis 0.
        T_b = jnp.stack([T_p] * 4, axis=0)
        q_b = jnp.stack([q] * 4, axis=0)
        p_b = jnp.stack([p] * 4, axis=0)
        qc_b = jnp.stack([qc] * 4, axis=0)
        qi_b = jnp.stack([qi] * 4, axis=0)
        cf_b = jnp.stack([cf] * 4, axis=0)
        rho_b = jnp.stack([rho] * 4, axis=0)
        dz_b = jnp.stack([dz] * 4, axis=0)
        nd_b = jnp.stack([ndrop] * 4, axis=0)
        _, state_b = jax.vmap(
            cloud_microphysics_column_sweep,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )(T_b, q_b, p_b, qc_b, qi_b, cf_b, rho_b, dz_b, nd_b, 1800.0, cfg)
        assert state_b.precip_rain.shape == (4,)
        assert jnp.all(jnp.isfinite(state_b.precip_rain))

    def test_surface_flux_matches_source_when_no_evap(self):
        """In a saturated column, no rain evaporates: surface == ∑ source.

        Rain evaporation requires sub-saturation (``q < qsw``); set
        ``q = qsw`` everywhere so Rotstayn's ``zsusatw = min(0, …) = 0``
        and the propagating ``zrfl`` at the surface equals the column
        integrated local rain source exactly.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 15
        T = jnp.linspace(280.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        q = jax.vmap(saturation_specific_humidity)(p, T)
        qc = jnp.zeros(nlev).at[4].set(1.5e-3).at[7].set(8e-4)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        assert jnp.allclose(
            state.precip_rain, jnp.sum(state.rain_source), rtol=1e-5,
        )

    def test_column_water_budget_closes(self):
        """Column-integrated d/dt(q+qc+qi) = -precip_surface within dt.

        Locks the mass-conservation invariant of the merged column sweep:
        the per-layer (dq + dqc + dqi) tendencies integrated over column
        mass, plus the surface precip flux, must sum to zero — i.e. the
        only sink for total water in the column is the falling precip
        that exits at the surface. With the in-sweep saturation
        adjustment moving mass between q ↔ qc/qi this is the right
        invariant to track; the per-level scheme could not close it
        because it discarded rain each step.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(220.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.9 * qsw                     # near-saturated everywhere
        qc = jnp.zeros(nlev).at[10].set(1.5e-3).at[12].set(1e-3)
        qi = jnp.zeros(nlev).at[3].set(2e-4)
        cf = jnp.where((qc + qi) > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        dt = 1800.0
        tend, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=dt, config=cfg,
        )
        mref = rho * dz                                       # kg/m² per layer
        # Column-integrated total water tendency (kg/m²/s)
        total_water_tend = jnp.sum(
            (tend.dqdt + tend.dqcdt + tend.dqidt) * mref,
        )
        surface_precip = state.precip_rain + state.precip_snow    # kg/m²/s
        # Budget: ∫(dq+dqc+dqi) dm/dt = -surface_precip
        residual = float(total_water_tend + surface_precip)
        scale = float(jnp.maximum(jnp.abs(surface_precip), 1e-9))
        assert abs(residual) / scale < 1e-3, (
            f"column water budget residual {residual:.3e} kg/m²/s, "
            f"surface precip {float(surface_precip):.3e} kg/m²/s"
        )

    def test_cloud_free_cells_force_evaporate_condensate(self):
        """ECHAM ``zxlevap``/``zxievap`` (#668): cf=0 clears its condensate.

        A cloud-free cell's condensate must return to vapour
        unconditionally — even when the cell is supersaturated, the case
        where the grid-mean Newton adjustment previously did the opposite
        (the cell GAINED condensate, with no cf-weighted microphysical
        sink ever able to touch it). The cleared water is then the
        adjustment's to re-condense or not; either way it is vapour first,
        the budget closes, and the latent heat is paid.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 6
        T = jnp.linspace(250.0, 290.0, nlev)
        p = jnp.linspace(40000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        # 2 % SUPERsaturated in the orphaned-condensate cells — the
        # issue's probe case, where the old behaviour condensed further.
        q = 1.02 * qsw
        qc = jnp.zeros(nlev).at[3].set(3e-4)
        qi = jnp.zeros(nlev).at[1].set(2e-4)
        cf = jnp.zeros(nlev)                 # the whole column cloud-free
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        dt = 1800.0
        tend, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=dt, config=cfg,
        )
        import jcm.constants as c
        qc_new = qc + dt * tend.dqcdt
        T_new = T + dt * tend.dtedt
        q_new = q + dt * tend.dqdt
        # A supersaturated cf=0 cell retains condensate, and that is
        # FAITHFUL: clearing then re-capping is a thermodynamic identity
        # (total water and enthalpy are unchanged, so the cell returns to
        # the same equilibrium), and ECHAM's own 5.4 grid-box cap does the
        # same. What #668 guarantees is that the outcome is the unique
        # THERMODYNAMIC equilibrium of the cell's conserved quantities, not
        # a function of how the water happened to be split on entry. Pin it
        # with an enthalpy-consistent pair: state B is state A after
        # condensing 3e-4 (same total water, same moist enthalpy) — both
        # must land on the same (T, q, qc) to within Newton tolerance.
        dq_shift = 3e-4
        qB = q.at[3].add(-dq_shift)
        qcB = qc.at[3].add(dq_shift)
        TB = T.at[3].add(c.alhc * dq_shift / c.cpd)
        tendB, _ = cloud_microphysics_column_sweep(
            TB, qB, p, qcB, qi, cf, rho, dz, ndrop, dt=dt, config=cfg,
        )
        qcB_new = qcB + dt * tendB.dqcdt
        TB_new = TB + dt * tendB.dtedt
        assert abs(float(qcB_new[3]) - float(qc_new[3])) < 3e-5, (
            f"cf=0 outcome depends on the vapour/condensate split of an "
            f"enthalpy-identical state: {float(qc_new[3]):.3e} vs "
            f"{float(qcB_new[3]):.3e}")
        assert abs(float(TB_new[3]) - float(T_new[3])) < 0.15
        # No accumulating supersaturation: vapour ends at/below the 1 %
        # grid-box allowance.
        from jcm.physics.clouds.sundqvist import (
            saturation_specific_humidity as _qs)
        qs_new = jax.vmap(_qs)(p, T_new)
        assert float((q_new - 1.02 * qs_new).max()) < 5e-5, (
            "supersaturation above the ECHAM allowance survived the step")
        # And the column water budget still closes through the clearing.
        mref = rho * dz
        total_water_tend = jnp.sum(
            (tend.dqdt + tend.dqcdt + tend.dqidt) * mref)
        surface_precip = state.precip_rain + state.precip_snow
        residual = float(total_water_tend + surface_precip)
        assert abs(residual) < 1e-9, f"budget open by {residual:.3e}"

    def test_cloud_free_subsaturated_cell_clears_fully(self):
        """The other half of #668: a SUBsaturated cf=0 cell keeps nothing.

        Pre-#668 the grid-mean adjustment evaporated orphaned condensate
        only up to saturation and kept the remainder as cloud water forever
        (no cf-weighted microphysical sink can touch a cf=0 cell). ECHAM
        clears it unconditionally; after the evaporative cooling the cell
        here is still subsaturated, so nothing re-condenses and the store
        is genuinely gone.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 6
        T = jnp.linspace(250.0, 290.0, nlev)
        p = jnp.linspace(40000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.8 * qsw
        qc = jnp.zeros(nlev).at[3].set(3e-4)
        cf = jnp.zeros(nlev)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        dt = 1800.0
        tend, state = cloud_microphysics_column_sweep(
            T, q, p, qc, jnp.zeros(nlev), cf, rho, dz, ndrop, dt=dt,
            config=cfg,
        )
        qc_new = qc + dt * tend.dqcdt
        assert float(jnp.abs(qc_new).max()) < 1e-8, (
            f"subsaturated cf=0 condensate survived: {float(qc_new[3]):.2e}")
        # Latent heat was paid: the cell cooled by ~L*qc/cp.
        dT3 = float(dt * tend.dtedt[3])
        import jcm.constants as c
        expected = -c.alhc * 3e-4 / c.cpd
        assert abs(dT3 - expected) < 0.1 * abs(expected), (
            f"evaporative cooling {dT3:.3f} K, expected ~{expected:.3f} K")

    @staticmethod
    def _mixed_phase_column(nlev=20):
        """Ice cloud aloft + liquid cloud below in a near-saturated column.

        Exercises both the rain and the snow/falling-ice flux paths; the
        top layers carry no condensate so nothing can fall out of level 0.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        T = jnp.linspace(230.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.95 * qsw
        qc = jnp.zeros(nlev).at[8].set(2e-3)
        qi = jnp.zeros(nlev).at[4].set(5e-4)
        cf = jnp.where((qc + qi) > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        return T, q, p, qc, qi, cf, rho, dz, ndrop

    def test_flux_profiles_bottom_row_equals_surface_diagnostics(self):
        """COSP hook invariants: rain_flux/snow_flux are per-level fluxes.

        The profiles are the scan's through-layer fluxes, so the bottom
        level must equal the surface ``precip_rain`` / ``precip_snow``
        EXACTLY (same carry values), be non-negative everywhere, and be
        zero at the model top (level 0 in the physics-internal TOA-first
        frame — no condensate there, nothing can fall out of it).
        """
        cfg = MicrophysicsParameters.default()
        column = self._mixed_phase_column()
        _, state = cloud_microphysics_column_sweep(
            *column, dt=1800.0, config=cfg,
        )
        assert float(state.precip_rain) > 0.0, "column must actually rain"
        assert float(jnp.abs(state.rain_flux[-1] - state.precip_rain)) < 1e-12
        assert float(jnp.abs(state.snow_flux[-1] - state.precip_snow)) < 1e-12
        assert jnp.all(state.rain_flux >= 0.0)
        assert jnp.all(state.snow_flux >= 0.0)
        assert float(state.rain_flux[0]) == 0.0
        assert float(state.snow_flux[0]) == 0.0
        # The frozen profile includes the sedimenting cloud-ice flux, so
        # it must be positive at the ice-cloud level itself (flux leaving
        # level 4). How far below the source it survives depends on the
        # sedimentation numerics (the expm1-stable influx form absorbs it
        # within the next layer for this column), so only the source level
        # is asserted.
        assert float(state.snow_flux[4]) > 0.0

    def test_flux_profiles_column_and_vmap_agree(self):
        """A vmapped batch must reproduce the single-column flux profiles."""
        cfg = MicrophysicsParameters.default()
        column = self._mixed_phase_column()
        _, state_1 = cloud_microphysics_column_sweep(
            *column, dt=1800.0, config=cfg,
        )
        batched = tuple(jnp.stack([arr] * 3, axis=0) for arr in column)
        _, state_b = jax.vmap(
            cloud_microphysics_column_sweep,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )(*batched, 1800.0, cfg)
        assert state_b.rain_flux.shape == (3, column[0].shape[0])
        # rtol at float32 ulp scale plus an atol floor: the two layouts may
        # commute the same arithmetic differently, and a last-bit difference
        # on a ~1e-4 flux cascades through the near-cancelling exponential
        # tail into percent-level RELATIVE differences on fluxes of 1e-11
        # and below — physically zero precipitation (~1 mm/millennium), so
        # the floor treats them as such.
        for i in range(3):
            assert jnp.allclose(state_b.rain_flux[i], state_1.rain_flux,
                                rtol=2e-6, atol=1e-9)
            assert jnp.allclose(state_b.snow_flux[i], state_1.snow_flux,
                                rtol=2e-6, atol=1e-9)


class TestColumnSweepParameterGradients:
    """Regression tests for NaN parameter gradients through the sweep.

    ``x**frac`` at a zero base (the Marshall-Palmer ``zxrp1`` / ``zxsp1``
    concentrations and the Rotstayn rain-evap rate) has an infinite
    derivative, and where-masking the output alone produced
    ``d(precip)/d(params) = NaN`` even though finite differences gave a
    perfectly good ~1e-6. These tests fail on the unguarded code and pin
    the double-where fix.
    """

    @staticmethod
    def _precipitating_column(nlev=20):
        """Warm near-saturated column with a liquid cloud aloft.

        Same construction as
        ``TestColumnSweepMicrophysics.test_warm_cloud_makes_surface_rain``:
        q at 95 % saturation so rain reaches the surface.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        T = jnp.linspace(280.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.95 * qsw
        qc = jnp.zeros(nlev).at[5].set(2e-3)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        return T, q, p, qc, qi, cf, rho, dz, ndrop

    @staticmethod
    def _mixed_column(nlev=20):
        """Mixed-phase column (ice cloud aloft, liquid cloud below) so the
        snow path — and with it ``cvtfall`` — is exercised.

        The deck must be CONTIGUOUS (cf > 0 from the ice layers down through
        the liquid layers): with a clear gap between them, the precipitating
        cloud cover ``zclcpre`` collapses in the gap, ``snow_present`` is
        False where the liquid sits, and the snow-riming path through
        ``zxsp1`` (the one that carries ``cvtfall``) is never active. The
        previous disjoint fixture only passed because the epsilon-floored
        sedimentation VJP manufactured a spurious ``cvtfall`` sensitivity;
        with the stable expm1 form the true gradient there is ~1e-24, i.e.
        the path the test names was not exercised at all.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        T = jnp.linspace(230.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.95 * qsw
        qc = jnp.zeros(nlev).at[6].set(1e-3).at[7].set(2e-3).at[8].set(2e-3)
        qi = jnp.zeros(nlev).at[4].set(5e-4).at[5].set(3e-4)
        cf = jnp.zeros(nlev).at[4:9].set(0.7)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        return T, q, p, qc, qi, cf, rho, dz, ndrop

    @staticmethod
    def _total_precip(ccraut, cvtfall, column):
        T, q, p, qc, qi, cf, rho, dz, ndrop = column
        cfg = MicrophysicsParameters.default(ccraut=ccraut, cvtfall=cvtfall)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        return state.precip_rain + state.precip_snow

    def test_param_gradients_finite_and_match_fd(self):
        ccraut0, cvtfall0 = 15.0, 3.29
        column = self._precipitating_column()
        d_ccraut, d_cvtfall = jax.grad(self._total_precip, argnums=(0, 1))(
            ccraut0, cvtfall0, column,
        )
        # The essential regression: finite (the unguarded powers gave NaN).
        assert jnp.isfinite(d_ccraut), f"d(precip)/d(ccraut) = {d_ccraut}"
        assert jnp.isfinite(d_cvtfall), f"d(precip)/d(cvtfall) = {d_cvtfall}"
        # More autoconversion → more rain: strictly positive here.
        assert float(d_ccraut) > 0.0

        # Cross-check the autoconversion gradient against central finite
        # differences (well-conditioned: FD/AD agree to ~0.03 % here).
        h = ccraut0 * 1e-3
        fd = (
            float(self._total_precip(ccraut0 + h, cvtfall0, column))
            - float(self._total_precip(ccraut0 - h, cvtfall0, column))
        ) / (2.0 * h)
        assert jnp.isclose(d_ccraut, fd, rtol=0.05), (
            f"AD {float(d_ccraut)} vs FD {fd}"
        )

    def test_cvtfall_gradient_finite_with_snow_active(self):
        # cvtfall enters through the falling-snow concentration zxsp1 (and
        # the ice sedimentation fall speed); a contiguous mixed-phase deck
        # makes its gradient genuinely nonzero (previously NaN via the
        # unguarded x**(1/1.16)).
        column = self._mixed_column()
        d_cvtfall = jax.grad(self._total_precip, argnums=1)(
            15.0, 3.29, column,
        )
        assert jnp.isfinite(d_cvtfall), f"d(precip)/d(cvtfall) = {d_cvtfall}"
        assert float(d_cvtfall) != 0.0
        # Cross-check against central finite differences so a spurious
        # gradient (e.g. one manufactured by an ill-conditioned VJP, as the
        # epsilon-floored sedimentation used to) cannot pass as "nonzero".
        h = 1e-3
        fd = (
            float(self._total_precip(15.0, 3.29 + h, column))
            - float(self._total_precip(15.0, 3.29 - h, column))
        ) / (2.0 * h)
        assert jnp.isclose(d_cvtfall, fd, rtol=0.05), (
            f"AD {float(d_cvtfall)} vs FD {fd}"
        )


class TestEcham1MPublishesEffectiveRadius:
    """The term must publish an LWC-dependent ``clouds.r_eff_liq`` (#717).

    Without it RRTMGP falls back to ``effective_radius_liquid``, a constant
    ~11 um independent of liquid water content.
    """

    NLEV = 8
    NCOLS = 3

    def _run_term(self, qc_profile, cdnc_factor=None):
        from .echam_1m import Echam1MMicrophysics
        from .cloud_data import CloudData
        from jcm.physics.aerosol.aerosol_types import AerosolData
        from jcm.physics_interface import PhysicsState

        from .sundqvist import saturation_specific_humidity

        nlev, ncols = self.NLEV, self.NCOLS
        shape = (nlev, ncols)
        # Warm, saturated column so the sweep's saturation adjustment neither
        # evaporates the prescribed cloud nor freezes it.
        p_col = jnp.linspace(4e4, 1e5, nlev)
        t_col = jnp.linspace(280.0, 295.0, nlev)
        q_col = jax.vmap(saturation_specific_humidity)(p_col, t_col)
        pressure = p_col[:, None] * jnp.ones((1, ncols))
        temperature = t_col[:, None] * jnp.ones((1, ncols))
        specific_humidity = q_col[:, None] * jnp.ones((1, ncols))
        air_density = pressure / (287.05 * temperature)
        qc = jnp.asarray(qc_profile)
        cloud_fraction = jnp.where(qc > 0.0, 0.6, 0.0)

        clouds = CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=cloud_fraction, qc=qc, qi=jnp.zeros(shape),
        )
        aerosol = AerosolData.zeros((ncols,), nlev)
        if cdnc_factor is not None:
            aerosol = aerosol.copy(cdnc_factor=jnp.asarray(cdnc_factor))

        state = PhysicsState.zeros(
            shape,
            temperature=temperature,
            specific_humidity=specific_humidity,
            tracers={"qc": qc, "qi": jnp.zeros(shape)},
        )
        diagnostics = {
            "_dt_seconds": 600.0,
            "pressure_full": pressure,
            "air_density": air_density,
            "layer_thickness": jnp.full(shape, 500.0),
            "clouds": clouds,
            "aerosol": aerosol,
        }
        _, out = Echam1MMicrophysics()(state, diagnostics, None, None)
        return np.asarray(out["clouds"].r_eff_liq)

    def test_cloud_free_levels_are_exactly_zero(self):
        qc = jnp.zeros((self.NLEV, self.NCOLS)).at[5].set(3e-4)
        r_eff = self._run_term(qc)
        cloudy = np.zeros((self.NLEV, self.NCOLS), dtype=bool)
        cloudy[5] = True
        assert (r_eff[~cloudy] == 0.0).all()
        assert (r_eff[cloudy] > 0.0).all()

    def test_radius_is_not_the_constant_fallback(self):
        # ``effective_radius_liquid(1.0, 0.5)`` = 14*0.5 + 8*0.5 = 11 um.
        qc = jnp.zeros((self.NLEV, self.NCOLS)).at[5].set(3e-4)
        r_eff = self._run_term(qc)
        assert not np.allclose(r_eff[5], 11.0)
        assert np.all((r_eff[5] > 2.0) & (r_eff[5] < 30.0))

    def test_radius_increases_with_liquid_water_content(self):
        # Same CDNC in every column; only the LWC differs.
        qc = jnp.zeros((self.NLEV, self.NCOLS)).at[5].set(
            jnp.array([5e-5, 2e-4, 8e-4])
        )
        r_eff = self._run_term(qc)
        assert np.all(np.diff(r_eff[5]) > 0.0)

    def test_radius_varies_in_the_vertical(self):
        qc = jnp.zeros((self.NLEV, self.NCOLS)).at[3].set(1e-4).at[5].set(6e-4)
        r_eff = self._run_term(qc)
        assert np.all(r_eff[5] > r_eff[3])

    def test_twomey_smaller_droplets_for_more_aerosol(self):
        qc = jnp.zeros((self.NLEV, self.NCOLS)).at[5].set(3e-4)
        r_clean = self._run_term(qc, cdnc_factor=jnp.ones((self.NCOLS,)))
        r_polluted = self._run_term(qc, cdnc_factor=jnp.full((self.NCOLS,), 2.0))
        assert np.all(r_polluted[5] < r_clean[5])

class TestCloudFractionWriteBack1M:
    """The 1M term clears the cover of cells it empties (#687).

    ECHAM's 1M ``cloud`` routine writes the post-microphysics cover back
    to ``paclc`` (mo_cloud.f90:1280): a cell whose end-of-step condensate
    is below ``ccwmin`` in BOTH phases stops being cloudy. Without it,
    ``clouds.cloud_fraction`` meant the RH-diagnosed pre-microphysics
    cover under cloud_scheme='1m' but the post-microphysics cover under
    '2m', so every shared consumer (radiation, COSP, AeroCom, the JAM
    cloud-borne/aqueous/wetdep terms) switched semantics with the scheme.
    """

    @staticmethod
    def _drive_term(qc0, qi0, cf0, q_scale=0.95):
        from types import SimpleNamespace
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics_interface import PhysicsState

        nlev, ncols = 8, 2
        T = jnp.full((nlev, ncols), 285.0)
        p = jnp.linspace(3e4, 1e5, nlev)[:, None] * jnp.ones((1, ncols))
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        qsw = saturation_specific_humidity(p, T)
        q = q_scale * qsw            # subsaturated: condensate evaporates
        rho = p / (287.0 * T)
        zeros = jnp.zeros((nlev, ncols))

        state = PhysicsState(
            u_wind=zeros, v_wind=zeros, temperature=T,
            specific_humidity=q, geopotential=zeros,
            normalized_surface_pressure=jnp.ones(ncols),
            tracers={"qc": qc0, "qi": qi0},
        )
        clouds = CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=cf0, qc=qc0, qi=qi0)
        diagnostics = {
            "_dt_seconds": 1800.0,
            "pressure_full": p,
            "air_density": rho,
            "layer_thickness": jnp.full((nlev, ncols), 500.0),
            "clouds": clouds,
            "aerosol": SimpleNamespace(cdnc_factor=jnp.ones(ncols)),
        }
        term = Echam1MMicrophysics()
        _, diags_out = term(state, diagnostics, None, None)
        return diags_out["clouds"].cloud_fraction

    def test_emptied_cell_loses_cover_kept_cell_keeps_it(self):
        import numpy as np
        nlev, ncols = 8, 2
        qc0 = jnp.zeros((nlev, ncols))
        qi0 = jnp.zeros((nlev, ncols))
        cf0 = jnp.zeros((nlev, ncols))
        # Level 3: a wisp of liquid (5e-7) — the 5 % saturation deficit
        # evaporates it below ccwmin within the step. Level 5: a solid
        # deck (1e-3) the same deficit can only nibble at.
        qc0 = qc0.at[3].set(5e-7).at[5].set(1e-3)
        cf0 = cf0.at[3].set(0.4).at[5].set(0.7)

        cf_out = np.asarray(self._drive_term(qc0, qi0, cf0))
        assert np.all(cf_out[3] == 0.0), (
            f"emptied cell keeps cover {cf_out[3]} — no paclc write-back"
        )
        assert np.all(cf_out[5] > 0.0), (
            "a cell still holding condensate lost its cover"
        )
