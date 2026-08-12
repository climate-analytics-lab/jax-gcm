"""Unit tests for the ECHAM 1-moment cloud microphysics scheme."""

import jax.numpy as jnp
import jax
from .echam_1m import (
    MicrophysicsParameters, cloud_droplet_radius,
    autoconversion, autoconversion_beheng, autoconversion_kk2000,
    ice_autoconversion, sedimentation_flux,
    cloud_microphysics_column_sweep,
)
from jcm.constants import tmelt


class TestCloudDropletRadius:
    """Test cloud droplet radius calculations"""
    
    def test_typical_values(self):
        """Test with typical atmospheric values"""
        cloud_water = jnp.array(0.5e-3)  # 0.5 g/kg
        air_density = jnp.array(1.0)      # kg/m³
        droplet_number = jnp.array(100e6) # 100 per cm³ -> per kg
        config = MicrophysicsParameters.default()
        
        radius = cloud_droplet_radius(cloud_water, air_density, droplet_number, config)
        
        # Should be in reasonable range (5-20 microns)
        assert 5e-6 < radius < 20e-6
    
    def test_limits(self):
        """Test radius limits are applied"""
        config = MicrophysicsParameters.default()
        
        # Very high cloud water with very few droplets should hit max radius
        radius_high = cloud_droplet_radius(
            jnp.array(10e-3), jnp.array(1.0), jnp.array(1e5), config  # Very few droplets
        )
        assert jnp.allclose(radius_high, float(config.ceffmax) * 1e-6)
        
        # Very low cloud water with many droplets should hit min radius
        radius_low = cloud_droplet_radius(
            jnp.array(1e-6), jnp.array(1.0), jnp.array(1000e6), config  # Many droplets
        )
        assert jnp.allclose(radius_low, float(config.ceffmin) * 1e-6)


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
            ccraut=1e-3, autoconversion_scheme="kk2000",
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
            ccraut=1e-5, autoconversion_scheme="kk2000",
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
            ccraut=1e-5, autoconversion_scheme="kk2000",
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

    def test_scheme_int_alias(self):
        """SCHEME_BEHENG / SCHEME_KK2000 ints round-trip with string aliases."""
        cfg_str = MicrophysicsParameters.default(autoconversion_scheme="kk2000")
        cfg_int = MicrophysicsParameters.default(
            autoconversion_scheme=MicrophysicsParameters.SCHEME_KK2000,
        )
        assert int(cfg_str.autoconversion_scheme) == MicrophysicsParameters.SCHEME_KK2000
        assert int(cfg_int.autoconversion_scheme) == int(cfg_str.autoconversion_scheme)
    
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
        for i in range(3):
            assert jnp.allclose(state_b.rain_flux[i], state_1.rain_flux,
                                atol=1e-12)
            assert jnp.allclose(state_b.snow_flux[i], state_1.snow_flux,
                                atol=1e-12)


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
