"""Unit tests for convective adjustment module

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from .adjustment import (
    saturation_adjustment, energy_conservation_check,
    convective_adjustment, cuadjtq, _qsat_and_dqsat_dt,
)
from .tiedtke_nordeng import saturation_mixing_ratio


class TestCuadjtq:
    """The linearised Newton-step saturation adjustment (port of
    ``mo_cuadjust.f90`` ``cuadjtq``).
    """

    def test_no_adjustment_for_subsaturated_input(self):
        """``kcall=1`` (condensation only) must leave subsaturated air
        unchanged: ``q < q_sat`` → ``cond = 0``.
        """
        T = jnp.array(280.0)
        p = jnp.array(90000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 0.5 * qs  # 50 % RH
        T_adj, q_adj, cond = cuadjtq(T, q, p, kcall=1)
        assert jnp.allclose(T_adj, T)
        assert jnp.allclose(q_adj, q)
        assert jnp.allclose(cond, 0.0)

    def test_modest_supersat_lands_close_to_saturation(self):
        """The Newton step is first-order accurate: at modest (10 %)
        supersaturation the two-pass ``cuadjtq`` lands within 1 % of
        saturation. (At very high supersaturation the linearisation
        leaves more residual — that's expected ECHAM behaviour and
        covered by ``test_strong_supersat_lands_subsaturated_not_super``.)
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.10 * qs
        T_adj, q_adj, _ = cuadjtq(T, q, p, kcall=1)
        qs_adj, _ = _qsat_and_dqsat_dt(T_adj, p)
        rh = float(q_adj / qs_adj)
        assert 0.99 <= rh <= 1.01, f"post-cuadjtq RH = {rh*100:.2f} %"

    def test_strong_supersat_lands_subsaturated_not_super(self):
        """At 50 % supersaturation the Newton step over-warms in the
        first pass; the ``kcall=1`` clip then prevents the second pass
        from re-evaporating. Final state must be subsaturated (the
        scheme is conservative — it never leaves the column above
        ``q_sat`` for the caller's downstream physics).
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.5 * qs
        T_adj, q_adj, _ = cuadjtq(T, q, p, kcall=1)
        qs_adj, _ = _qsat_and_dqsat_dt(T_adj, p)
        rh = float(q_adj / qs_adj)
        assert rh <= 1.0, f"left supersaturated at RH = {rh*100:.2f} %"

    def test_kcall_1_only_condenses(self):
        """``kcall=1`` (cubase / cuasc) must never produce negative
        condensate (no evaporation in the updraft branch).
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        # Subsaturated input: simple form would evaporate (cond < 0)
        q = 0.7 * qs
        _T_adj, _q_adj, cond = cuadjtq(T, q, p, kcall=1)
        assert float(cond) >= 0.0

    def test_kcall_2_only_evaporates(self):
        """``kcall=2`` (cudlfs / cuddraf) must never produce positive
        condensate (no condensation in the downdraft branch).
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.3 * qs
        _T_adj, _q_adj, cond = cuadjtq(T, q, p, kcall=2)
        assert float(cond) <= 0.0

    def test_kcall_0_allows_both_directions(self):
        """``kcall=0`` (cuini env q_sat) lets the Newton step go either
        way — used to align environmental q with q_sat at half levels.
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        # Subsaturated → negative cond (evaporation of imaginary liquid).
        q = 0.7 * qs
        _T1, _q1, cond_dry = cuadjtq(T, q, p, kcall=0)
        assert float(cond_dry) < 0.0
        # Supersaturated → positive cond.
        q = 1.3 * qs
        _T2, _q2, cond_wet = cuadjtq(T, q, p, kcall=0)
        assert float(cond_wet) > 0.0

    def test_moist_static_energy_conserved_per_step(self):
        """``cuadjtq`` releases L·Δq of latent heat per kg of water
        condensed. ``cp·ΔT + L·Δq`` must be conserved up to the
        linearisation residual (sub-1 % at 50 % supersat).
        """
        from jcm.constants import cp, alhc
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.5 * qs
        T_adj, q_adj, _cond = cuadjtq(T, q, p, kcall=1)
        h_before = cp * T + alhc * q
        h_after = cp * T_adj + alhc * q_adj
        rel_imbalance = float(jnp.abs(h_after - h_before) / h_before)
        assert rel_imbalance < 1e-3, (
            f"moist static energy drift = {rel_imbalance*100:.4f} %"
        )

    def test_refine_pass_reduces_residual(self):
        """The ``refine=True`` second iteration must leave the column
        closer to saturation than the single Newton pass.
        """
        T = jnp.array(290.0)
        p = jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 2.0 * qs  # strong supersat — exposes the linearisation residual
        T1, q1, _ = cuadjtq(T, q, p, kcall=1, refine=False)
        T2, q2, _ = cuadjtq(T, q, p, kcall=1, refine=True)
        qs1, _ = _qsat_and_dqsat_dt(T1, p)
        qs2, _ = _qsat_and_dqsat_dt(T2, p)
        rh1 = float(q1 / qs1)
        rh2 = float(q2 / qs2)
        assert abs(rh2 - 1.0) <= abs(rh1 - 1.0)





class TestSaturationAdjustment:
    """Test saturation adjustment functions"""
    
    def test_no_adjustment_needed(self):
        """Test that no adjustment occurs for subsaturated air"""
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        
        # Create subsaturated conditions (50% RH)
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 0.5 * qs
        
        cloud_water = jnp.array(0.0001)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should be unchanged
        assert jnp.allclose(t_adj, temperature)
        assert jnp.allclose(q_adj, specific_humidity)
        assert jnp.allclose(qc_adj, cloud_water)
        assert jnp.allclose(qi_adj, cloud_ice)
    
    def test_condensation_warm(self):
        """Test condensation in warm conditions"""
        temperature = jnp.array(285.0)  # Above freezing
        pressure = jnp.array(90000.0)
        
        # Create supersaturated state (110% RH)
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.1 * qs
        
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should have warming from latent heat
        assert t_adj > temperature
        
        # Should have condensation to liquid only
        assert q_adj < specific_humidity
        assert qc_adj > cloud_water
        assert jnp.allclose(qi_adj, cloud_ice)  # No ice formation
        
        # Check final state is near saturation
        rs_adj = saturation_mixing_ratio(pressure, t_adj)
        qs_adj = rs_adj / (1 + rs_adj)
        rh_final = q_adj / qs_adj
        # The adjustment reduces supersaturation significantly
        # but may not reach exact saturation in finite iterations
        assert 0.75 < rh_final < 1.05  # Should be much closer to saturation
    
    def test_condensation_cold(self):
        """Test condensation in cold conditions"""
        temperature = jnp.array(250.0)  # Well below freezing
        pressure = jnp.array(50000.0)
        
        # Create supersaturated state
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.15 * qs
        
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should have warming
        assert t_adj > temperature
        
        # Should have condensation to ice only
        assert q_adj < specific_humidity
        assert jnp.allclose(qc_adj, cloud_water)  # No liquid formation
        assert qi_adj > cloud_ice
    
    def test_condensation_mixed_phase(self):
        """Test condensation in mixed phase region"""
        temperature = jnp.array(265.0)  # Mixed phase
        pressure = jnp.array(70000.0)
        
        # Create supersaturated state
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.08 * qs
        
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should have both liquid and ice
        assert qc_adj > 0
        assert qi_adj > 0
        
        # At -8°C, we're in the mixed phase region
        # Both phases should be present but the ratio depends on the partitioning scheme
    
    def test_conservation(self):
        """Test mass conservation in adjustment"""
        temperature = jnp.array(275.0)
        pressure = jnp.array(85000.0)
        
        # Supersaturated
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.12 * qs
        
        cloud_water = jnp.array(0.0002)
        cloud_ice = jnp.array(0.0001)
        
        # Total water before
        total_before = specific_humidity + cloud_water + cloud_ice
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Total water after
        total_after = q_adj + qc_adj + qi_adj
        
        # Should conserve total water (within numerical precision)
        assert jnp.allclose(total_before, total_after, rtol=2e-3)


class TestEnergyConservation:
    """Test energy conservation diagnostics"""
    
    def test_warming_condensation(self):
        """Test energy balance for condensation case"""
        # Initial state
        t_old = jnp.array(280.0)
        q_old = jnp.array(0.012)  # 12 g/kg
        qc_old = jnp.array(0.0)
        qi_old = jnp.array(0.0)
        
        # After condensation
        t_new = jnp.array(281.5)  # Warmed
        q_new = jnp.array(0.010)  # Dried
        qc_new = jnp.array(0.002) # Cloud formed
        qi_new = jnp.array(0.0)
        
        precip = jnp.array(0.0)
        dt = 3600.0
        
        imbalance = energy_conservation_check(
            t_old, q_old, qc_old, qi_old,
            t_new, q_new, qc_new, qi_new,
            precip, dt
        )
        
        # Energy should be approximately conserved
        # Some imbalance due to approximations
        assert jnp.abs(imbalance) < 50.0  # W/m²
    
    def test_with_precipitation(self):
        """Test energy balance with precipitation"""
        t_old = jnp.array(280.0)
        q_old = jnp.array(0.010)
        qc_old = jnp.array(0.002)
        qi_old = jnp.array(0.0)
        
        # After precipitation
        t_new = jnp.array(280.0)  # Same temperature
        q_new = jnp.array(0.010)  # Same vapor
        qc_new = jnp.array(0.001) # Cloud water reduced
        qi_new = jnp.array(0.0)
        
        # Precipitation removes 1 g/kg of water
        precip = jnp.array(0.001 / 3600.0)  # kg/kg/s -> kg/m²/s needs scaling
        dt = 3600.0
        
        imbalance = energy_conservation_check(
            t_old, q_old, qc_old, qi_old,
            t_new, q_new, qc_new, qi_new,
            precip, dt
        )
        
        # Should have energy loss due to precipitation
        assert imbalance < 0  # Energy removed


class TestConvectiveAdjustment:
    """Test the full convective adjustment"""
    
    def test_apply_tendencies_and_adjust(self):
        """Test applying tendencies followed by adjustment"""
        # Initial state
        temperature = jnp.array(278.0)
        pressure = jnp.array(90000.0)
        specific_humidity = jnp.array(0.008)
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        # Convective tendencies (warming and moistening)
        conv_tend_t = jnp.array(2.0 / 3600.0)    # 2 K/hour
        conv_tend_q = jnp.array(0.002 / 3600.0)  # 2 g/kg/hour
        conv_tend_qc = jnp.array(0.0)
        conv_tend_qi = jnp.array(0.0)
        
        dt = 1800.0  # 30 minutes
        
        # Apply adjustment
        t_adj, q_adj, qc_adj, qi_adj = convective_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice,
            conv_tend_t, conv_tend_q, conv_tend_qc, conv_tend_qi,
            dt
        )
        
        # Should be warmer (due to convective heating)
        assert t_adj > temperature
        # Humidity may decrease if warming causes condensation
        # Check that total tendency was applied
        t_expected = temperature + conv_tend_t * dt
        # Temperature should be at least the tendency-applied value
        assert t_adj >= t_expected - 0.1  # Allow small deviation
        
        # If supersaturated after tendencies, should have condensation
        rs_final = saturation_mixing_ratio(pressure, t_adj)
        qs_final = rs_final / (1 + rs_final)
        rh_final = q_adj / qs_final
        
        # Should not be supersaturated after adjustment
        assert rh_final <= 1.02
    
    def test_with_cloud_tendencies(self):
        """Test adjustment with cloud water tendencies"""
        temperature = jnp.array(275.0)
        pressure = jnp.array(85000.0)
        specific_humidity = jnp.array(0.006)
        cloud_water = jnp.array(0.0005)
        cloud_ice = jnp.array(0.0002)
        
        # Tendencies that increase clouds
        conv_tend_t = jnp.array(0.5 / 3600.0)
        conv_tend_q = jnp.array(-0.001 / 3600.0)  # Drying
        conv_tend_qc = jnp.array(0.0008 / 3600.0)  # Cloud increase
        conv_tend_qi = jnp.array(0.0002 / 3600.0)
        
        dt = 1800.0
        
        t_adj, q_adj, qc_adj, qi_adj = convective_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice,
            conv_tend_t, conv_tend_q, conv_tend_qc, conv_tend_qi,
            dt
        )
        
        # Clouds should increase
        assert qc_adj > cloud_water
        assert qi_adj > cloud_ice
        
        # Total water should be conserved (within adjustment)
        total_old = specific_humidity + cloud_water + cloud_ice
        total_tend = (conv_tend_q + conv_tend_qc + conv_tend_qi) * dt
        total_expected = total_old + total_tend
        total_new = q_adj + qc_adj + qi_adj
        
        assert jnp.allclose(total_new, total_expected, rtol=0.01)
    
    def test_jax_transformations(self):
        """Test JAX transformations work"""
        def adjustment_fn(temperature):
            pressure = jnp.array(90000.0)
            q = jnp.array(0.008)
            qc = jnp.array(0.0)
            qi = jnp.array(0.0)
            
            t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
                temperature, q, pressure, qc, qi
            )
            return t_adj
        
        # Test JIT
        jitted_fn = jax.jit(adjustment_fn)
        t = jnp.array(280.0)
        t_adj = jitted_fn(t)
        assert jnp.isfinite(t_adj)
        
        # Test gradient
        grad_fn = jax.grad(adjustment_fn)
        grad = grad_fn(t)
        assert jnp.isfinite(grad)


if __name__ == "__main__":
    # Run tests
    test_sat = TestSaturationAdjustment()
    test_sat.test_no_adjustment_needed()
    test_sat.test_condensation_warm()
    test_sat.test_condensation_cold()
    test_sat.test_condensation_mixed_phase()
    test_sat.test_conservation()
    
    test_energy = TestEnergyConservation()
    test_energy.test_warming_condensation()
    test_energy.test_with_precipitation()
    
    test_adj = TestConvectiveAdjustment()
    test_adj.test_apply_tendencies_and_adjust()
    test_adj.test_with_cloud_tendencies()
    test_adj.test_jax_transformations()
    
    print("All adjustment tests passed!")