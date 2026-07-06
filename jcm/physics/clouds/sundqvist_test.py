"""Unit tests for the Sundqvist cloud fraction and condensation helpers."""

import jax.numpy as jnp
import jax
from .sundqvist import (
    CloudParameters, saturation_vapor_pressure_water, saturation_vapor_pressure_ice,
    saturation_specific_humidity, calculate_cloud_fraction,
    condensation_evaporation, critical_relative_humidity, _qs_and_dqs_dt,
)
from jcm.constants import tmelt, eps, alhc, cpd




def _qs_for_cover(pressure, temperature):
    """Build qs consistent with the cover's lo2 phase switch for ice-free columns.

    The cloud-fraction decision now uses ECHAM's binary lo2 saturation
    (water unless T < cthomi or ice is present — review finding 2.27), so
    test humidities built as a fraction of qs must use the same surface,
    not the legacy mixed-phase blend.
    """
    from jcm.physics.clouds.sundqvist import _qs_cover
    return jax.vmap(lambda p_, t_: _qs_cover(p_, t_, jnp.zeros_like(t_)))(
        pressure, temperature)


class TestCondensationLinearisation:
    """The condensation step ports ECHAM ``mo_cloud.f90`` lines 696-784.

    These tests pin down the three changes that take the per-step heating
    at high supersaturation from ``+60 K`` (pre-fix) to ``+10 K``
    (post-fix, ECHAM-matching) on a Tibetan-style column:

    * Newton denominator ``1 + L/cp · dq_s/dT`` damps the step by ~6× in
      the warm troposphere.
    * Two-pass adjustment cleans up residual super-saturation.
    * 1 % over-saturation tolerance allows micro-residuals so we don't
      thrash near saturation on every column every step.
    """

    def _config(self):
        return CloudParameters.default()

    def test_warming_feedback_dampens_supersat_heating(self):
        """At 50 % supersaturation in the warm troposphere, the linearised
        Newton step (``1 + L/cp · dqs/dT`` denominator) must reduce the
        single-step heating by a factor close to ``1 + L/cp · dqs/dT``
        compared to the bare ``q_excess`` formula.
        """
        T = jnp.array(290.0)
        p = jnp.array(50000.0)
        cf = jnp.array(0.5)
        config = self._config()
        qs, dqs_dt = _qs_and_dqs_dt(p, T)
        q = 1.5 * qs   # 50 % supersat

        dT, _, _, _ = condensation_evaporation(
            T, q, jnp.array(0.0), jnp.array(0.0), cf, p, 1800.0, config,
        )
        dT_per_step = float(dT) * 1800.0

        # Naive (pre-fix) ΔT = L/cp · q_excess
        naive_dT = float(alhc / cpd * (q - qs))
        # Expected damping factor at this T
        damping = float(1.0 + alhc / cpd * dqs_dt)

        # Post-fix ΔT must be no larger than naive/damping × 1.2 (allow
        # 20 % slack for the cloud-fraction weighting + two-pass cleanup).
        assert dT_per_step <= naive_dT / damping * 1.2, (
            f"single-step heating {dT_per_step:.2f} K not damped enough "
            f"(naive {naive_dT:.2f}, damping {damping:.2f})"
        )

    def test_extreme_supersat_no_runaway_heating(self):
        """At 100 % supersaturation (the runaway condition that drove the
        T63 + Tibetan column failure in PR #458), the per-step heating
        must stay below 15 K. Pre-fix this hit 60+ K.
        """
        T = jnp.array(290.0)
        p = jnp.array(50000.0)
        cf = jnp.array(0.5)
        config = self._config()
        qs, _ = _qs_and_dqs_dt(p, T)
        q = 2.0 * qs

        dT, _, _, _ = condensation_evaporation(
            T, q, jnp.array(0.0), jnp.array(0.0), cf, p, 1800.0, config,
        )
        dT_per_step = float(dT) * 1800.0
        assert dT_per_step < 15.0, (
            f"single-step heating {dT_per_step:.2f} K — pre-fix it was 60 K, "
            f"the linearised port should bring it under 15 K"
        )

    def test_no_oversat_after_two_passes(self):
        """The two-pass cleanup must leave the column within
        ``1 + oversat_frac`` of saturation. Default ``oversat_frac=0.01``
        → no more than 1 % super-sat after the call.
        """
        T = jnp.array(290.0)
        p = jnp.array(50000.0)
        cf = jnp.array(1.0)   # full overcast — so pass 1 absorbs most
        config = self._config()
        qs, _ = _qs_and_dqs_dt(p, T)
        q = 1.20 * qs

        dT, dq, _, _ = condensation_evaporation(
            T, q, jnp.array(0.0), jnp.array(0.0), cf, p, 1800.0, config,
        )
        T_after = float(T) + float(dT) * 1800.0
        q_after = float(q) + float(dq) * 1800.0
        qs_after, _ = _qs_and_dqs_dt(p, jnp.array(T_after))
        rh_after = q_after / float(qs_after)
        assert rh_after <= 1.011, (
            f"post-condensation RH = {rh_after*100:.2f} % "
            f"(should be within 1 % oversat tolerance)"
        )

    def test_subsaturated_column_unchanged(self):
        """A subsaturated column with no cloud water/ice must produce zero
        condensation tendencies (no spontaneous evaporation when there's
        nothing to evaporate).
        """
        T = jnp.array(290.0)
        p = jnp.array(50000.0)
        config = self._config()
        qs, _ = _qs_and_dqs_dt(p, T)
        q = 0.5 * qs   # 50 % RH

        dT, dq, dqc, dqi = condensation_evaporation(
            T, q, jnp.array(0.0), jnp.array(0.0), jnp.array(0.5),
            p, 1800.0, config,
        )
        assert jnp.allclose(dT, 0.0)
        assert jnp.allclose(dq, 0.0)
        assert jnp.allclose(dqc, 0.0)
        assert jnp.allclose(dqi, 0.0)

    def test_moist_static_energy_per_step(self):
        """The per-step adjustment conserves moist static energy
        ``cp · ΔT + L · Δq = 0`` to within the linearisation residual
        (sub-1 % even at strong supersaturation).
        """
        T = jnp.array(290.0)
        p = jnp.array(50000.0)
        cf = jnp.array(0.5)
        config = self._config()
        qs, _ = _qs_and_dqs_dt(p, T)
        q = 1.30 * qs

        dT, dq, _, _ = condensation_evaporation(
            T, q, jnp.array(0.0), jnp.array(0.0), cf, p, 1800.0, config,
        )
        delta_T = float(dT) * 1800.0
        delta_q = float(dq) * 1800.0
        h_change = cpd * delta_T + alhc * delta_q
        h_baseline = cpd * float(T) + alhc * float(q)
        rel = abs(h_change) / h_baseline
        assert rel < 1e-2, f"moist static energy drift {rel*100:.3f} %"





class TestSaturationFunctions:
    """Test saturation vapor pressure and humidity calculations"""
    
    def test_saturation_vapor_pressure_water(self):
        """Test saturation vapor pressure over water"""
        # At 0°C, should be ~611 Pa
        es_0c = saturation_vapor_pressure_water(jnp.array(tmelt))
        assert jnp.abs(es_0c - 610.78) < 1.0
        
        # At 20°C, should be ~2339 Pa
        es_20c = saturation_vapor_pressure_water(jnp.array(tmelt + 20.0))
        assert 2300 < es_20c < 2400
        
        # Should increase with temperature
        temps = jnp.linspace(250, 310, 10)
        es_vals = jax.vmap(saturation_vapor_pressure_water)(temps)
        assert jnp.all(jnp.diff(es_vals) > 0)
    
    def test_saturation_vapor_pressure_ice(self):
        """Test saturation vapor pressure over ice"""
        # At 0°C, should be ~611 Pa
        es_0c = saturation_vapor_pressure_ice(jnp.array(tmelt))
        assert jnp.abs(es_0c - 610.78) < 1.0
        
        # At -20°C, should be ~103 Pa
        es_m20c = saturation_vapor_pressure_ice(jnp.array(tmelt - 20.0))
        assert 100 < es_m20c < 110
        
        # Should increase with temperature
        temps = jnp.linspace(220, 273, 10)
        es_vals = jax.vmap(saturation_vapor_pressure_ice)(temps)
        assert jnp.all(jnp.diff(es_vals) > 0)
    
    def test_saturation_specific_humidity(self):
        """Test saturation specific humidity calculation"""
        # Standard atmosphere at sea level
        p_sfc = 101325.0  # Pa
        t_sfc = 288.15    # K (15°C)
        
        qs = saturation_specific_humidity(jnp.array(p_sfc), jnp.array(t_sfc))
        
        # Should be around 10 g/kg
        assert 0.008 < qs < 0.012
        
        # Should increase as pressure decreases (at constant temperature)
        pressures = jnp.linspace(100000, 20000, 10)
        qs_vals = jax.vmap(lambda p: saturation_specific_humidity(p, t_sfc))(pressures)
        assert jnp.all(jnp.diff(qs_vals) > 0)  # qs increases as pressure decreases
        
        # Test mixed phase region
        t_mixed = 260.0  # K
        p_mid = 50000.0  # Pa
        qs_mixed = saturation_specific_humidity(jnp.array(p_mid), jnp.array(t_mixed))
        
        # Should be between pure ice and pure water values
        qs_ice = eps * saturation_vapor_pressure_ice(jnp.array(t_mixed)) / p_mid
        qs_water = eps * saturation_vapor_pressure_water(jnp.array(t_mixed)) / p_mid
        assert qs_ice <= qs_mixed <= qs_water


class TestCloudFraction:
    """Test cloud fraction calculations"""

    def test_critical_rh_matches_echam_mo_cover(self):
        """Pin ``mo_cover.f90`` critical-RH profile and parameter meanings."""
        config = CloudParameters.default()
        pressure = jnp.array([100000.0, 95000.0, 70000.0, 50000.0, 20000.0])
        p_sfc = 100000.0

        rhc = critical_relative_humidity(pressure, p_sfc, config)
        expected = config.crt + (config.crs - config.crt) * jnp.exp(
            1.0 - (p_sfc / pressure) ** config.nex
        )

        assert abs(float(config.crs) - 0.975) < 1e-7
        assert abs(float(config.crt) - 0.75) < 1e-7
        assert abs(float(config.nex) - 2.0) < 1e-7
        assert jnp.allclose(rhc, expected)
        assert jnp.isclose(rhc[0], config.crs)
        assert rhc[-1] < 0.751

    def test_cloud_fraction_uses_echam_threshold_profile(self):
        """A 70 kPa, 84.5% RH layer should cloud under ECHAM T63 defaults.

        This RH is above the ``mo_cover`` threshold but below the old
        sigma-interpolation threshold, so it catches the ordering/formula
        regression directly through ``calculate_cloud_fraction``.
        """
        config = CloudParameters.default()
        pressure = jnp.array([70000.0])
        temperature = jnp.array([260.0])
        p_sfc = 100000.0

        qs = _qs_for_cover(pressure, temperature)
        specific_humidity = 0.845 * qs
        cf, _ = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, p_sfc, config
        )

        assert cf[0] > 0.03
    
    def test_cloud_fraction_basic(self):
        """Test basic cloud fraction calculation"""
        config = CloudParameters.default()
        
        # Create test profile
        nlev = 20
        pressure = jnp.linspace(100000, 20000, nlev)
        temperature = jnp.linspace(288, 220, nlev)
        
        # Dry case - use 30% relative humidity everywhere
        # This creates a realistic dry atmosphere
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.3 * qs  # 30% RH everywhere
        cf, rh = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, 100000.0, config
        )
        
        # With 30% RH, should have no clouds anywhere
        assert jnp.all(cf < 0.01)  # No significant clouds
        assert jnp.all(rh < 0.35)  # RH should be around 30%
        
        # Saturated case - should have clouds
        qs = _qs_for_cover(pressure, temperature)
        specific_humidity = 0.95 * qs  # 95% relative humidity (cover qs)
        cf, rh = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, 100000.0, config
        )
        
        assert jnp.any(cf > 0.5)  # Should have significant clouds
        assert jnp.all(rh > 0.9)   # High relative humidity

    def test_stratospheric_cloud_cutoff(self):
        """No cloud above the cloud-top pressure (ECHAM ``jks``).

        A supersaturated column reaching into the stratosphere must form
        cloud in the troposphere but NOT above ``cloud_top_pressure_pa``.
        The RH closure otherwise fills the cold (qsat→0) stratosphere with
        spurious cloud (q/qsat ~80× at ~180 K), which inflates total cloud
        cover and corrupts the radiation cloud field.
        """
        # The default cutoff is now ECHAM-like 10 hPa (finding 2.26: the
        # 100 hPa stopgap deleted TTL cirrus); this test pins the cutoff
        # MECHANISM, so it sets 100 hPa explicitly.
        config = CloudParameters.default(cloud_top_pressure_pa=10000.0)
        pressure = jnp.array([90000.0, 50000.0, 30000.0, 8000.0, 5000.0, 3000.0])
        temperature = jnp.array([285.0, 250.0, 225.0, 200.0, 190.0, 185.0])
        qs = _qs_for_cover(pressure, temperature)
        specific_humidity = 1.2 * qs  # supersaturated at every level
        cf, _ = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, 100000.0, config
        )
        below = pressure >= config.cloud_top_pressure_pa
        above = pressure < config.cloud_top_pressure_pa
        assert jnp.all(cf[below] > 0.5), "tropospheric cloud should form"
        assert jnp.all(cf[above] == 0.0), "no cloud above the cutoff"

        # Disabling the cutoff (0 Pa) restores the (spurious) stratospheric
        # cloud — confirms the cutoff is what suppresses it.
        cfg_off = CloudParameters.default(cloud_top_pressure_pa=0.0)
        cf_off, _ = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, 100000.0, cfg_off
        )
        assert jnp.any(cf_off[above] > 0.5)

    def test_cloud_fraction_profile(self):
        """Test that critical RH varies with height"""
        config = CloudParameters.default()
        
        # Create pressure levels
        pressure = jnp.array([100000, 70000, 50000, 30000, 20000])
        temperature = jnp.array([288, 268, 248, 228, 218])
        p_sfc = 100000.0
        
        # Set constant relative humidity
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        rh_target = 0.8
        specific_humidity = rh_target * qs
        
        cf, rh = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, p_sfc, config
        )
        
        # Cloud fraction should increase with height at same RH
        # (because critical RH decreases with height)
        assert cf[0] < cf[-1]  # More clouds at top than bottom


class TestCondensationEvaporation:
    """Test condensation/evaporation processes"""
    
    def test_condensation(self):
        """Test condensation in supersaturated conditions"""
        config = CloudParameters.default()
        
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        cloud_fraction = jnp.array(0.5)
        cloud_water = jnp.array(0.0005)
        cloud_ice = jnp.array(0.0)
        dt = 1800.0  # 30 minutes
        
        # Create supersaturated conditions
        qs = saturation_specific_humidity(pressure, temperature)
        specific_humidity = 1.1 * qs  # 110% relative humidity
        
        dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
            temperature, specific_humidity, cloud_water, cloud_ice,
            cloud_fraction, pressure, dt, config
        )
        
        # Should have condensation
        assert dqdt < 0  # Humidity decreases
        assert dqcdt > 0  # Cloud water increases
        assert dtedt > 0  # Temperature increases (latent heat release)
    
    def test_evaporation(self):
        """Test evaporation in subsaturated conditions"""
        config = CloudParameters.default()
        
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        cloud_fraction = jnp.array(0.5)
        cloud_water = jnp.array(0.001)
        cloud_ice = jnp.array(0.0)
        dt = 1800.0
        
        # Create subsaturated conditions
        qs = saturation_specific_humidity(pressure, temperature)
        specific_humidity = 0.7 * qs  # 70% relative humidity
        
        dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
            temperature, specific_humidity, cloud_water, cloud_ice,
            cloud_fraction, pressure, dt, config
        )
        
        # Should have evaporation
        assert dqdt > 0   # Humidity increases
        assert dqcdt < 0  # Cloud water decreases
        assert dtedt < 0  # Temperature decreases (latent heat consumption)
        
        # Check evaporation doesn't exceed available cloud water
        assert dqcdt >= -cloud_water / dt


class TestJaxTransformations:
    """JIT/grad coverage for the live helpers the physics term composes."""

    def test_jit_and_grad_through_live_path(self):
        """JIT and grad work through cloud fraction + condensation."""
        config = CloudParameters.default()

        pressure = jnp.linspace(100000, 20000, 10)
        temperature = jnp.linspace(288, 220, 10)
        specific_humidity = jnp.ones(10) * 0.005
        cloud_water = jnp.zeros(10)
        cloud_ice = jnp.zeros(10)

        def live_path(t):
            cf, _ = calculate_cloud_fraction(
                t, specific_humidity, pressure, 100000.0, config
            )
            dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
                t, specific_humidity, cloud_water, cloud_ice,
                cf, pressure, 1800.0, config
            )
            return cf, dtedt

        cf, dtedt = jax.jit(live_path)(temperature)
        assert cf.shape == temperature.shape
        assert dtedt.shape == temperature.shape

        grad = jax.grad(lambda t: jnp.sum(live_path(t)[1] ** 2))(temperature)
        assert grad.shape == temperature.shape
        assert jnp.all(jnp.isfinite(grad))


class TestCondensationToCloudWater:
    """Tests for within-timestep condensation producing cloud condensate.

    ``condensation_evaporation`` must emit positive qc/qi tendencies from
    supersaturation even when initial qc/qi are zero, so that microphysics
    (called next in the term chain) can convert condensate to precipitation.
    """

    def test_supersaturated_column_produces_cloud_water(self):
        """A supersaturated column must produce a positive qc tendency.

        This is the key regression test: with cloud_water=0 but RH > 100%,
        the scheme must condense moisture into cloud water within the call.
        """
        config = CloudParameters.default()
        nlev = 20
        pressure = jnp.linspace(100000, 20000, nlev)
        temperature = jnp.linspace(290, 220, nlev)

        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = jnp.where(pressure > 60000, 1.05 * qs, 0.3 * qs)

        cloud_water = jnp.zeros(nlev)
        cloud_ice = jnp.zeros(nlev)

        _, _, dqcdt, _ = condensation_evaporation(
            temperature, specific_humidity, cloud_water, cloud_ice,
            jnp.zeros(nlev), pressure, 1800.0, config
        )

        assert jnp.max(dqcdt) > 0.0, \
            f"dqc/dt should be > 0 for supersaturated column, got {float(jnp.max(dqcdt)):.6e}"

    def test_subsaturated_column_no_cloud_water(self):
        """A dry subsaturated column with no initial condensate stays dry."""
        config = CloudParameters.default()
        nlev = 20
        pressure = jnp.linspace(100000, 20000, nlev)
        temperature = jnp.linspace(290, 220, nlev)

        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.3 * qs

        cloud_water = jnp.zeros(nlev)
        cloud_ice = jnp.zeros(nlev)

        _, _, dqcdt, dqidt = condensation_evaporation(
            temperature, specific_humidity, cloud_water, cloud_ice,
            jnp.zeros(nlev), pressure, 1800.0, config
        )

        assert jnp.allclose(dqcdt, 0.0), \
            "dqc/dt should remain 0 for dry column"
        assert jnp.allclose(dqidt, 0.0), \
            "dqi/dt should remain 0 for dry column"

    def test_cold_supersaturated_column_produces_cloud_ice(self):
        """A cold supersaturated column should produce a cloud-ice tendency."""
        config = CloudParameters.default()
        nlev = 10
        pressure = jnp.linspace(50000, 20000, nlev)
        temperature = jnp.full(nlev, 240.0)  # Below freezing

        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 1.1 * qs

        cloud_water = jnp.zeros(nlev)
        cloud_ice = jnp.zeros(nlev)

        _, _, _, dqidt = condensation_evaporation(
            temperature, specific_humidity, cloud_water, cloud_ice,
            jnp.zeros(nlev), pressure, 1800.0, config
        )

        assert jnp.max(dqidt) > 0.0, \
            f"dqi/dt should be > 0 for cold supersaturated column, got {float(jnp.max(dqidt)):.6e}"
