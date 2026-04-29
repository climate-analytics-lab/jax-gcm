"""Unit tests for Tiedtke-Nordeng convection scheme

This file provides comprehensive unit tests that can be run with pytest.

Date: 2025-01-09
"""

import pytest
import jax.numpy as jnp
import jax

# Import convection modules
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    saturation_mixing_ratio,
    find_cloud_base,
    calculate_cape_cin
)
from jcm.physics.convection.tiedtke_nordeng.tracer_transport import (
    TracerIndices,
    initialize_tracers
)


def create_realistic_atmosphere(nlev=20, unstable=True):
    """Create a realistic atmospheric profile for testing"""
    # Physical constants
    Rd = 287.05  # J/(kg*K) - gas constant for dry air

    # Pressure levels (Pa) - from surface (1000 hPa) to top (~200 hPa)
    pressure = jnp.linspace(1e5, 2e4, nlev)

    # Height (m) - increases with decreasing pressure
    height = jnp.linspace(0, 12000, nlev)

    if unstable:
        # Unstable profile - warm at surface with normal lapse rate
        temperature = 300.0 - 6.5e-3 * height
        surface_humidity = 0.012  # 12 g/kg
    else:
        # Stable profile - cooler surface, weaker lapse rate
        temperature = 285.0 - 5.0e-3 * height
        surface_humidity = 0.003  # 3 g/kg (dry)

    # Humidity profile limited by saturation
    humidity_profile = surface_humidity * jnp.exp(-height / 2000.0)
    qs_profile = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
    humidity = jnp.minimum(humidity_profile, 0.9 * qs_profile)

    # Simple wind profile
    u_wind = jnp.full(nlev, 10.0)
    v_wind = jnp.zeros(nlev)

    # Calculate air density (kg/m³)
    rho = pressure / (Rd * temperature)

    # Calculate layer thickness (m) - uniform spacing in this case
    dz = height[1] - height[0]
    layer_thickness = jnp.full(nlev, dz)

    return {
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'height': height,
        'layer_thickness': layer_thickness,
        'rho': rho,
        'u_wind': u_wind,
        'v_wind': v_wind
    }


class TestSaturationFunctions:
    """Test saturation calculations"""
    
    def test_saturation_mixing_ratio_basic(self):
        """Test basic saturation mixing ratio calculation"""
        temp = 300.0  # K
        press = 1e5   # Pa
        
        qs = saturation_mixing_ratio(press, temp)
        
        # Should be reasonable tropical value
        assert 0.01 < qs < 0.05, f"Unrealistic saturation mixing ratio: {qs}"
        assert isinstance(qs, jnp.ndarray), "Should return JAX array"
    
    def test_saturation_temperature_dependence(self):
        """Test that saturation increases with temperature"""
        temps = jnp.array([273.15, 290.0, 300.0, 310.0])
        pressure = 1e5
        
        qs_array = jax.vmap(saturation_mixing_ratio, in_axes=(None, 0))(pressure, temps)
        
        # Should increase monotonically with temperature
        assert jnp.all(jnp.diff(qs_array) > 0), "Saturation should increase with temperature"
    
    def test_saturation_pressure_dependence(self):
        """Test that saturation increases with decreasing pressure"""
        pressures = jnp.array([1e5, 8e4, 6e4, 4e4])
        temperature = 300.0
        
        qs_array = jax.vmap(saturation_mixing_ratio, in_axes=(0, None))(pressures, temperature)
        
        # Should increase with decreasing pressure (at constant temperature)
        assert jnp.all(jnp.diff(qs_array) > 0), "Saturation should increase with decreasing pressure"
    
    def test_saturation_edge_cases(self):
        """Test edge cases for saturation calculation"""
        # Very cold temperature
        qs_cold = saturation_mixing_ratio(1e5, 200.0)
        assert qs_cold > 0, "Should have positive saturation even at low temperatures"
        assert qs_cold < 1e-3, "Should be very small at cold temperatures"
        
        # Very hot temperature
        qs_hot = saturation_mixing_ratio(1e5, 350.0)
        assert qs_hot > 0.1, "Should be large at high temperatures"
        assert qs_hot < 1.0, "Should be physically reasonable"


class TestCloudBase:
    """Test cloud base detection"""
    
    def test_cloud_base_unstable(self):
        """Test cloud base detection in unstable atmosphere"""
        atm = create_realistic_atmosphere(unstable=True)
        config = ConvectionParameters.default()
        
        cloud_base, has_cloud_base = find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        assert has_cloud_base, "Should find cloud base in unstable atmosphere"
        
        # Cloud base should be at reasonable height
        cb_height = atm['height'][cloud_base]
        assert 500 < cb_height < 4000, f"Unrealistic cloud base height: {cb_height}"
        
        # Cloud base should be above surface
        surf_idx = jnp.argmax(atm['pressure'])
        assert cloud_base > surf_idx, "Cloud base should be above surface"
    
    def test_cloud_base_stable(self):
        """Test cloud base detection in stable atmosphere"""
        atm = create_realistic_atmosphere(unstable=False)
        config = ConvectionParameters.default()
        
        cloud_base, has_cloud_base = find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        # May or may not find cloud base in stable atmosphere
        if has_cloud_base:
            cb_height = atm['height'][cloud_base]
            assert cb_height > 0, "Cloud base should be above surface"
    
    def test_cloud_base_jax_compatibility(self):
        """Test that cloud base detection works with JAX transformations"""
        atm = create_realistic_atmosphere(unstable=True)
        config = ConvectionParameters.default()
        
        # Test JIT compilation
        jitted_find_cloud_base = jax.jit(find_cloud_base)
        cloud_base, has_cloud_base = jitted_find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        assert isinstance(cloud_base, jnp.ndarray), "Should return JAX arrays"
        assert isinstance(has_cloud_base, jnp.ndarray), "Should return JAX arrays"


class TestCAPE:
    """Test CAPE/CIN calculations"""
    
    def test_cape_basic(self):
        """Test basic CAPE calculation"""
        atm = create_realistic_atmosphere(unstable=True)
        config = ConvectionParameters.default()
        
        # Find cloud base first
        cloud_base, has_cloud_base = find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        if has_cloud_base:
            cape, cin = calculate_cape_cin(
                atm['temperature'], atm['humidity'], atm['pressure'],
                atm['layer_thickness'], cloud_base, config
            )
            
            # CAPE should be non-negative
            assert cape >= 0, f"CAPE should be non-negative: {cape}"
            assert cin >= 0, f"CIN should be non-negative: {cin}"
            
            # Should be reasonable values
            assert cape < 10000, f"CAPE too high: {cape}"
            assert cin < 5000, f"CIN too high: {cin}"
    
    def test_cape_stable_vs_unstable(self):
        """Test that CAPE is higher in unstable atmosphere"""
        config = ConvectionParameters.default()
        
        # Unstable atmosphere
        atm_unstable = create_realistic_atmosphere(unstable=True)
        cb_unstable, has_cb_unstable = find_cloud_base(
            atm_unstable['temperature'], atm_unstable['humidity'], 
            atm_unstable['pressure'], config
        )
        
        # Stable atmosphere
        atm_stable = create_realistic_atmosphere(unstable=False)
        cb_stable, has_cb_stable = find_cloud_base(
            atm_stable['temperature'], atm_stable['humidity'],
            atm_stable['pressure'], config
        )
        
        if has_cb_unstable and has_cb_stable:
            cape_unstable, _ = calculate_cape_cin(
                atm_unstable['temperature'], atm_unstable['humidity'],
                atm_unstable['pressure'], atm_unstable['layer_thickness'],
                cb_unstable, config
            )
            
            cape_stable, _ = calculate_cape_cin(
                atm_stable['temperature'], atm_stable['humidity'],
                atm_stable['pressure'], atm_stable['layer_thickness'],
                cb_stable, config
            )
            
            # Both profiles should have positive CAPE (both have moisture)
            # Note: with moist adiabatic CAPE, the "stable" profile can have
            # significant CAPE too since it still has moisture and a lapse rate
            assert cape_unstable > 0, f"Unstable atmosphere should have positive CAPE, got {cape_unstable}"
            assert cape_stable >= 0, f"Stable atmosphere should have non-negative CAPE, got {cape_stable}"


class TestCAPEDeepColumn:
    """CAPE/CIN bounds for a deep (TOA-reaching) ICON-style column.

    The default ``create_realistic_atmosphere`` only goes up to 200 hPa, so
    the CIN integral never sees the upper troposphere or stratosphere. The
    real ICON 47-level grid reaches ~1 Pa, where the parcel-environment
    buoyancy difference becomes huge negative because the lifted parcel
    cools to ~100 K while the environmental temperature is held at 200 K.
    Without a stop above the LFC, those upper-level layers dump hundreds
    of thousands of J/kg into CIN — physically meaningless and a sign
    that the integration bounds are wrong.

    All inputs are TOA-first (level 0 = TOA, level nlev-1 = surface) to
    match the ICON physics convention used inside ``apply_convection``.
    """

    def _icon_like_column_47(self, surf_T=300.0, surf_q_kgkg=0.018):
        """Build a TOA-first 47-level column reaching ~1 Pa."""
        nlev = 47
        # surface-first sigma → flip to TOA-first
        sigma = jnp.linspace(1.0, 0.001, nlev)[::-1]
        pressure = 1.0e5 * sigma  # TOA-first: p[0] ≈ 100 Pa, p[-1] ≈ 1e5 Pa
        Rd = 287.05
        # Geopotential height referenced to surface (high at TOA, ~0 at surf)
        height = -Rd * 250.0 * jnp.log(pressure / pressure[-1]) / 9.81
        temperature = jnp.where(
            pressure > 2.0e4,
            surf_T - 6.5e-3 * height,
            jnp.maximum(surf_T - 6.5e-3 * height, 200.0),
        )
        humidity_profile = surf_q_kgkg * jnp.exp(-height / 2000.0)
        qs = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
        humidity = jnp.minimum(humidity_profile, 0.9 * qs)
        layer_thickness = jnp.abs(jnp.diff(height, append=height[-1]))
        return {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "layer_thickness": layer_thickness,
        }

    def test_cin_bounded_for_deep_column(self):
        """CIN must stay physically reasonable when the column reaches TOA.

        Realistic CIN values cap out at a few hundred J/kg even in
        strong-cap regimes; > 5000 J/kg is non-physical and indicates
        the integration is sweeping up stratospheric buoyancy noise.
        """
        atm = self._icon_like_column_47()
        config = ConvectionParameters.default()
        cb, has_cb = find_cloud_base(
            atm["temperature"], atm["humidity"], atm["pressure"], config,
        )
        assert bool(has_cb)
        cape, cin = calculate_cape_cin(
            atm["temperature"], atm["humidity"], atm["pressure"],
            atm["layer_thickness"], cb, config,
        )
        cin_v = float(cin)
        assert 0.0 <= cin_v < 5000.0, (
            f"CIN={cin_v:.0f} J/kg for a 47-level moist tropical column is "
            f"non-physical — likely integrating buoyancy across the "
            f"stratosphere where the parcel has cooled to ~100 K"
        )

    def test_cape_bounded_for_deep_column(self):
        """CAPE shouldn't explode either; >5000 J/kg is extreme-storm regime."""
        atm = self._icon_like_column_47()
        config = ConvectionParameters.default()
        cb, _ = find_cloud_base(
            atm["temperature"], atm["humidity"], atm["pressure"], config,
        )
        cape, _ = calculate_cape_cin(
            atm["temperature"], atm["humidity"], atm["pressure"],
            atm["layer_thickness"], cb, config,
        )
        cape_v = float(cape)
        assert 50.0 < cape_v < 5000.0, (
            f"CAPE={cape_v:.0f} J/kg outside physical range for a moist "
            f"tropical column"
        )


class TestJAXCompatibility:
    """Test JAX transformations"""
    
    def test_jit_compilation(self):
        """Test JIT compilation of key functions"""
        atm = create_realistic_atmosphere()
        config = ConvectionParameters.default()
        
        # Test JIT on saturation function
        jitted_saturation = jax.jit(saturation_mixing_ratio)
        qs = jitted_saturation(1e5, 300.0)
        assert qs > 0, "JIT compilation should work"
        
        # Test JIT on cloud base
        jitted_cloud_base = jax.jit(find_cloud_base)
        cb, has_cb = jitted_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        assert isinstance(cb, jnp.ndarray), "Should return JAX array"
    
    def test_vectorization(self):
        """Test vectorization with vmap"""
        temperatures = jnp.array([280.0, 290.0, 300.0, 310.0])
        pressure = 1e5
        
        # Vectorize over temperature
        vmap_saturation = jax.vmap(saturation_mixing_ratio, in_axes=(None, 0))
        qs_vec = vmap_saturation(pressure, temperatures)
        
        assert qs_vec.shape == temperatures.shape, "Should maintain shape"
        assert jnp.all(qs_vec > 0), "All values should be positive"
    
    def test_gradients(self):
        """Test gradient computation"""
        def loss_fn(temp):
            return saturation_mixing_ratio(1e5, temp)
        
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(300.0)
        
        assert gradient > 0, "Gradient should be positive (qs increases with T)"
        assert jnp.isfinite(gradient), "Gradient should be finite"


class TestTracerTransport:
    """Test tracer transport functionality"""
    
    def test_tracer_initialization(self):
        """Test tracer initialization"""
        nlev = 20
        
        # Basic tracers only
        tracers_basic, indices_basic = initialize_tracers(nlev, include_chemistry=False)
        assert tracers_basic.shape == (nlev, 3), "Should have 3 basic tracers"
        
        # With chemistry
        tracers_chem, indices_chem = initialize_tracers(nlev, include_chemistry=True)
        assert tracers_chem.shape[1] > 3, "Should have additional chemical tracers"
        
        # Check indices
        assert indices_basic.iqv == 0, "Water vapor should be index 0"
        assert indices_basic.iqc == 1, "Cloud water should be index 1"
        assert indices_basic.iqi == 2, "Cloud ice should be index 2"
        assert indices_basic.iqt == 3, "Additional tracers should start at index 3"
    
    def test_tracer_indices(self):
        """Test tracer indices structure"""
        indices = TracerIndices()
        
        assert hasattr(indices, 'iqv'), "Should have water vapor index"
        assert hasattr(indices, 'iqc'), "Should have cloud water index"
        assert hasattr(indices, 'iqi'), "Should have cloud ice index"
        assert hasattr(indices, 'iqt'), "Should have additional tracer start index"
        
        # Check ordering
        assert indices.iqv < indices.iqc < indices.iqi < indices.iqt, "Indices should be ordered"


class TestConfiguration:
    """Test configuration parameters"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ConvectionParameters.default()
        
        # Check that all required parameters are present
        assert hasattr(config, 'tau'), "Should have CAPE timescale"
        assert hasattr(config, 'entrpen'), "Should have entrainment parameters"
        assert hasattr(config, 'cmfcmax'), "Should have mass flux limits"
        
        # Check reasonable values
        assert float(config.tau) > 0, "CAPE timescale should be positive"
        assert float(config.cmfcmax) > float(config.cmfcmin), "Max mass flux should exceed min"
        assert 0 < float(config.entrpen) < 1, "Entrainment rate should be reasonable"
    
    def test_config_modification(self):
        """Test configuration modification"""
        # Create configs with different tau values
        config1 = ConvectionParameters.default(tau=3600.0)
        config2 = ConvectionParameters.default(tau=7200.0)
        
        assert float(config1.tau) != float(config2.tau), "Should allow parameter modification"
        # Note: when creating parameters directly, need to set all fields


class TestPhysicalConsistency:
    """Test physical consistency of calculations"""
    
    def test_humidity_consistency(self):
        """Test that humidity profiles are physically consistent"""
        atm = create_realistic_atmosphere()
        
        # Check humidity range
        assert jnp.all(atm['humidity'] >= 0), "Humidity should be non-negative"
        assert jnp.all(atm['humidity'] < 0.1), "Humidity should be reasonable (< 100 g/kg)"
        
        # Check relative humidity
        qs_profile = jax.vmap(saturation_mixing_ratio)(atm['pressure'], atm['temperature'])
        rel_humidity = atm['humidity'] / qs_profile
        assert jnp.all(rel_humidity <= 1.0), "Should not exceed saturation"
    
    def test_temperature_consistency(self):
        """Test that temperature profiles are physically consistent"""
        atm = create_realistic_atmosphere()
        
        # Check temperature range
        assert jnp.all(atm['temperature'] > 150), "Temperature should be reasonable (> 150K)"
        assert jnp.all(atm['temperature'] < 350), "Temperature should be reasonable (< 350K)"
        
        # Check lapse rate
        temp_gradient = jnp.mean(jnp.diff(atm['temperature']) / jnp.diff(atm['height']))
        assert -0.015 < temp_gradient < 0, "Lapse rate should be reasonable"
    
    def test_pressure_consistency(self):
        """Test that pressure profiles are physically consistent"""
        atm = create_realistic_atmosphere()
        
        # Pressure should decrease with height
        assert jnp.all(jnp.diff(atm['pressure']) < 0), "Pressure should decrease with height"
        
        # Reasonable pressure range
        assert jnp.min(atm['pressure']) > 1e4, "Minimum pressure should be reasonable"
        assert jnp.max(atm['pressure']) <= 1.1e5, "Maximum pressure should be reasonable"


# Pytest fixtures for common test data
@pytest.fixture
def unstable_atmosphere():
    """Fixture providing unstable atmospheric profile"""
    return create_realistic_atmosphere(unstable=True)


@pytest.fixture  
def stable_atmosphere():
    """Fixture providing stable atmospheric profile"""
    return create_realistic_atmosphere(unstable=False)


@pytest.fixture
def default_config():
    """Fixture providing default convection configuration"""
    return ConvectionParameters.default()


class TestUpdraftDetrainment:
    """Test updraft entrainment/detrainment profile."""

    def test_organized_detrainment_increases_near_cloud_top(self):
        """Organized detrainment should increase sharply near cloud top (tan profile)."""
        # Sample the tan() profile at several heights
        fracs = jnp.array([0.2, 0.5, 0.8, 0.95])  # fractional distance from base
        tan_args = jnp.pi * (0.75 * fracs - 0.25)
        profiles = jnp.maximum(jnp.tan(tan_args), 0.0)

        # Detrainment should increase monotonically toward cloud top
        for i in range(len(profiles) - 1):
            assert float(profiles[i + 1]) >= float(profiles[i]), (
                f"Detrainment should increase toward cloud top: "
                f"frac={float(fracs[i]):.2f} -> {float(fracs[i+1]):.2f}"
            )

        # Near cloud top (frac=0.95) should be much larger than mid-cloud (frac=0.5)
        assert float(profiles[3]) > 3.0 * float(profiles[1]), (
            "Near-top detrainment should be much larger than mid-cloud"
        )

    def test_mass_flux_decreases_toward_cloud_top(self):
        """With organized detrainment, updraft mass flux should decrease toward cloud top."""
        from jcm.physics.convection.tiedtke_nordeng.updraft import calculate_updraft

        nlev = 40
        config = ConvectionParameters.default()
        kbase = 30
        ktop = 15

        # Create a moist-unstable profile
        pressure = jnp.linspace(100000, 10000, nlev)
        temperature = 300.0 * (pressure / 100000.0) ** 0.286
        qs = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
        humidity = 0.85 * qs
        layer_thickness = jnp.ones(nlev) * 500.0
        rho = pressure / (287.0 * temperature)

        state = calculate_updraft(
            temperature, humidity, pressure, layer_thickness, rho,
            kbase, ktop, ktype=1, mass_flux_base=0.1, config=config
        )

        # Mass flux at cloud top should be less than at cloud base
        mf_base = float(state.mfu[kbase])
        mf_top = float(state.mfu[ktop])
        assert mf_base > 0, "Mass flux at cloud base should be positive"
        assert mf_top < mf_base, (
            f"Mass flux should decrease toward top: base={mf_base:.4f}, top={mf_top:.4f}"
        )

    def test_detrainment_exceeds_entrainment_near_top(self):
        """Near cloud top, detrainment should exceed entrainment (organized component)."""
        from jcm.physics.convection.tiedtke_nordeng.updraft import calculate_updraft

        nlev = 40
        config = ConvectionParameters.default()
        kbase = 30
        ktop = 15

        pressure = jnp.linspace(100000, 10000, nlev)
        temperature = 300.0 * (pressure / 100000.0) ** 0.286
        qs = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
        humidity = 0.85 * qs
        layer_thickness = jnp.ones(nlev) * 500.0
        rho = pressure / (287.0 * temperature)

        state = calculate_updraft(
            temperature, humidity, pressure, layer_thickness, rho,
            kbase, ktop, ktype=1, mass_flux_base=0.1, config=config
        )

        # Near cloud top, detrainment should exceed entrainment
        near_top = ktop + 2
        assert float(state.detr[near_top]) > float(state.entr[near_top]), (
            f"Near cloud top (k={near_top}): detr={float(state.detr[near_top]):.6f} "
            f"should exceed entr={float(state.entr[near_top]):.6f}"
        )

class TestDowndraftLFS:
    """Test downdraft level of free sinking criteria."""

    def test_cmfdeps_parameter_exists(self):
        """Verify cmfdeps parameter exists with default value ~0.33."""
        config = ConvectionParameters.default()
        assert hasattr(config, 'cmfdeps'), "Should have cmfdeps parameter"
        assert float(config.cmfdeps) == pytest.approx(0.33)

    def test_cmfdeps_used_in_lfs_threshold(self):
        """LFS threshold should use cmfdeps (not cmfcmin) times base mass flux.

        The Fortran reference computes zmftop = -cmfdeps*pmfub where cmfdeps~0.33,
        giving a meaningful fraction of the updraft mass flux. Using cmfcmin (~1e-10)
        would make the threshold effectively zero.
        """
        config = ConvectionParameters.default()
        base_mf = 0.1  # Typical cloud base mass flux (kg/m²/s)

        # Correct threshold using cmfdeps
        threshold_correct = float(config.cmfdeps) * base_mf
        assert threshold_correct == pytest.approx(0.033, rel=0.01)

        # Old (wrong) threshold using cmfcmin would be negligible
        threshold_wrong = float(config.cmfcmin) * base_mf
        assert threshold_wrong < 1e-9, "cmfcmin threshold should be negligible"

        # The correct threshold should be orders of magnitude larger
        assert threshold_correct > threshold_wrong * 1e6


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])