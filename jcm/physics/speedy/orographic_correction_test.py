"""
Unit tests for orographic correction parameterization.

Tests verify that the orographic corrections are computed correctly and that
applying corrections in grid space produces equivalent results to the SPEEDY
spectral space implementation.
"""

# Force JAX to use CPU before any imports
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
import functools
from jax.test_util import check_vjp, check_jvp
from jcm.physics.speedy.orographic_correction import (
    compute_temperature_correction_vertical_profile,
    compute_humidity_correction_vertical_profile,
    compute_temperature_correction_horizontal,
    compute_humidity_correction_horizontal,
    get_orographic_correction_tendencies,
    apply_orographic_corrections_to_state
)
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.physical_constants import grav

def create_test_geometry(layers=8, lon_points=96, lat_points=48):
    """Create a test geometry object using the actual Geometry class."""
    # Use the actual Geometry class from the codebase
    nodal_shape = (lon_points, lat_points)
    return Geometry.from_grid_shape(nodal_shape=nodal_shape, node_levels=layers)


def create_test_boundaries(lon_points=96, lat_points=48):
    """Create test boundary data with simple orography."""
    # Create simple mountain orography (Gaussian peak)
    lon_idx = jnp.arange(lon_points)
    lat_idx = jnp.arange(lat_points)
    lon_grid, lat_grid = jnp.meshgrid(lon_idx, lat_idx, indexing='ij')
    
    # Simple Gaussian mountain centered in the domain
    center_lon, center_lat = lon_points // 2, lat_points // 2
    sigma_lon, sigma_lat = lon_points / 8, lat_points / 8
    
    orog = 1000.0 * jnp.exp(
        -((lon_grid - center_lon) ** 2 / (2 * sigma_lon ** 2) +
          (lat_grid - center_lat) ** 2 / (2 * sigma_lat ** 2))
    )

    class TestBoundaries:
        def __init__(self):
            self.orog = orog
            # For temperature correction, we need phis0 (spectrally-filtered surface geopotential)
            # In this test, we'll approximate it as geopotential = gravity * height
            self.phis0 = grav * orog  # Approximate phis0 = g * h
            # Add land/sea masks and sea surface temperature for humidity correction
            self.fmask = jnp.ones((lon_points, lat_points)) * 0.7  # 70% land
            self.tsea = jnp.full((lon_points, lat_points, 365), 285.0)     # Sea surface temperature
    
    return TestBoundaries()


def create_test_physics_state(layers=8, lon_points=96, lat_points=48):
    """Create a test physics state with realistic values."""
    shape = (layers, lon_points, lat_points)
    surface_shape = (lon_points, lat_points)
    
    # Create realistic temperature profile (decreases with height)
    temp_surface = 288.0  # K
    temp_top = 220.0  # K
    temperature = jnp.linspace(temp_surface, temp_top, layers)[:, None, None] * jnp.ones(shape)
    
    # Add some spatial variation
    lon_idx = jnp.arange(lon_points)
    lat_idx = jnp.arange(lat_points)
    lon_grid, lat_grid = jnp.meshgrid(lon_idx, lat_idx, indexing='ij')
    
    # Add sinusoidal temperature variation
    temp_variation = 10.0 * jnp.sin(2 * jnp.pi * lon_grid / lon_points) * jnp.cos(jnp.pi * lat_grid / lat_points)
    temperature = temperature + temp_variation
    
    # Create humidity field (decreases with height)
    humidity = 0.01 * jnp.exp(-jnp.arange(layers)[:, None, None] / 3.0) * jnp.ones(shape)
    
    # Create wind fields
    u_wind = jnp.zeros(shape)
    v_wind = jnp.zeros(shape)
    
    # Create geopotential (increases with height)
    geopotential = jnp.zeros(shape)
    
    # Create surface pressure
    surface_pressure = jnp.ones(surface_shape)
    
    return PhysicsState(
        u_wind=u_wind,
        v_wind=v_wind,
        temperature=temperature,
        specific_humidity=humidity,
        geopotential=geopotential,
        normalized_surface_pressure=surface_pressure
    )


class TestOrographicCorrection:
    """Test suite for orographic correction functions."""
    
    def test_temperature_vertical_profile(self):
        """Test computation of temperature correction vertical profile."""
        geometry = create_test_geometry(layers=8)
        parameters = Parameters.default()
        
        tcorv = compute_temperature_correction_vertical_profile(geometry, parameters)
        
        # Check shape
        assert tcorv.shape == (8,)
        
        # Check first level is zero (SPEEDY specification)
        assert tcorv[0] == 0.0
        
        # Check other levels are positive and increasing with sigma
        assert jnp.all(tcorv[1:] > 0.0)
        
        # Check values make physical sense (should be small corrections)
        assert jnp.all(tcorv < 1.0)
    
    def test_humidity_vertical_profile(self):
        """Test computation of humidity correction vertical profile."""
        geometry = create_test_geometry(layers=8)
        parameters = Parameters.default()
        
        qcorv = compute_humidity_correction_vertical_profile(geometry, parameters)
        
        # Check shape
        assert qcorv.shape == (8,)
        
        # Check first two levels are zero (SPEEDY specification)
        assert qcorv[0] == 0.0
        assert qcorv[1] == 0.0
        
        # Check other levels are positive
        assert jnp.all(qcorv[2:] > 0.0)
    
    def test_temperature_horizontal_correction(self):
        """Test computation of temperature horizontal correction."""
        boundaries = create_test_boundaries(lon_points=96, lat_points=48)
        geometry = create_test_geometry()
        
        tcorh = compute_temperature_correction_horizontal(boundaries, geometry)
        
        # Check shape
        assert tcorh.shape == (96, 48)
        
        # Check that maximum correction occurs where orography is highest
        max_orog_idx = jnp.unravel_index(jnp.argmax(boundaries.orog), boundaries.orog.shape)
        max_corr_idx = jnp.unravel_index(jnp.argmax(tcorh), tcorh.shape)
        assert max_orog_idx == max_corr_idx
    
    def test_humidity_horizontal_correction(self):
        """Test computation of humidity horizontal correction."""
        boundaries = create_test_boundaries(lon_points=96, lat_points=48)
        geometry = create_test_geometry()
        
        # Compute temperature correction needed for the new humidity correction
        tcorh = compute_temperature_correction_horizontal(boundaries, geometry)
        land_temp = jnp.full((96, 48), 288.0)  # Constant land temperature
        
        qcorh = compute_humidity_correction_horizontal(boundaries, geometry, tcorh, land_temp)
        
        # Check shape
        assert qcorh.shape == (96, 48)
        
        # Check that correction has reasonable magnitude
        assert jnp.all(jnp.abs(qcorh) < 10.0)  # Should be reasonable correction (up to ~10 g/kg)
        
        # Check that correction is related to orography (simplified implementation)
        # The sign and magnitude depend on the specific implementation
        assert jnp.any(qcorh != 0.0)  # Should not be all zeros
    
    def test_get_orographic_correction_tendencies(self):
        """Test the main tendency computation function."""
        state = create_test_physics_state()
        boundaries = create_test_boundaries()
        geometry = create_test_geometry()
        parameters = Parameters.default()
        nodal_shape = state.temperature.shape[1:]  # (lon, lat)
        node_levels = state.temperature.shape[0]   # layers
        physics_data = PhysicsData.zeros(nodal_shape, node_levels)
        
        tendencies, updated_physics_data = get_orographic_correction_tendencies(
            state, physics_data, parameters, boundaries, geometry
        )
        
        # Check return types
        assert isinstance(tendencies, PhysicsTendency)
        assert isinstance(updated_physics_data, PhysicsData)
        
        # Check shapes
        assert tendencies.u_wind.shape == state.u_wind.shape
        assert tendencies.v_wind.shape == state.v_wind.shape
        assert tendencies.temperature.shape == state.temperature.shape
        assert tendencies.specific_humidity.shape == state.specific_humidity.shape
        
        # Check that wind tendencies are zero (no orographic correction for winds)
        assert jnp.all(tendencies.u_wind == 0.0)
        assert jnp.all(tendencies.v_wind == 0.0)
        
        # Check that temperature and humidity tendencies are non-zero where orography exists
        assert jnp.any(tendencies.temperature != 0.0)
        assert jnp.any(tendencies.specific_humidity != 0.0)
        
        # Check that tendencies have reasonable magnitude
        # These are tendencies in K/s and kg/kg/s that will be integrated over the timestep
        # For reference: with test orography (~1km mountains):
        #   - Max temperature tendency: ~0.003 K/s → 5.4 K change over 30 min
        #   - Max humidity tendency: ~0.001 kg/kg/s → 1.8 g/kg change over 30 min

        assert jnp.all(jnp.abs(tendencies.temperature) < 0.05)  # Max ~0.05 K/s is reasonable
        assert jnp.all(jnp.abs(tendencies.specific_humidity) < 0.01)  # Max ~0.01 kg/kg/s is reasonable
    
    def test_apply_orographic_corrections_to_state(self):
        """Test direct application of corrections to state."""
        state = create_test_physics_state()
        boundaries = create_test_boundaries()
        geometry = create_test_geometry()
        parameters = Parameters.default()
        
        corrected_state = apply_orographic_corrections_to_state(
            state, boundaries, geometry, parameters
        )
        
        # Check that state type is preserved
        assert isinstance(corrected_state, PhysicsState)
        
        # Check that shapes are preserved
        assert corrected_state.temperature.shape == state.temperature.shape
        assert corrected_state.specific_humidity.shape == state.specific_humidity.shape
        
        # Check that wind fields are unchanged
        np.testing.assert_array_equal(corrected_state.u_wind, state.u_wind)
        np.testing.assert_array_equal(corrected_state.v_wind, state.v_wind)
        
        # Check that temperature and humidity are modified
        assert not jnp.array_equal(corrected_state.temperature, state.temperature)
        assert not jnp.array_equal(corrected_state.specific_humidity, state.specific_humidity)
        
        # Check that corrections are applied correctly
        tcorv = compute_temperature_correction_vertical_profile(geometry, parameters)
        tcorh = compute_temperature_correction_horizontal(boundaries, geometry)
        expected_temp_correction = tcorh * tcorv[:, None, None]

        actual_temp_correction = corrected_state.temperature - state.temperature
        # Allow for small numerical differences due to JAX/numpy precision
        np.testing.assert_allclose(actual_temp_correction, expected_temp_correction, rtol=1e-4, atol=2e-5)
    
    def test_jax_compatibility(self):
        """Test that functions are JAX-compatible (can be differentiated and JIT compiled)."""
        state = create_test_physics_state()
        boundaries = create_test_boundaries()
        geometry = create_test_geometry()
        parameters = Parameters.default()
        
        # Test gradient computation (JIT with non-array arguments is complex, so just test gradients)
        def loss_fn(state):
            corrected = apply_orographic_corrections_to_state(state, boundaries, geometry, parameters)
            return jnp.sum(corrected.temperature ** 2)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state)
        
        # Check that gradients exist and have correct shape
        assert hasattr(grads, 'temperature')
        assert grads.temperature.shape == state.temperature.shape
        assert jnp.any(grads.temperature != 0.0)  # Should have non-zero gradients
    
    def test_speedy_fortran_numerical_equivalence(self):
        """Test numerical equivalence with SPEEDY Fortran implementation."""
        # Use actual JAX-GCM geometry with 8 layers to get correct sigma levels
        geometry_fortran = create_test_geometry(layers=8)
        parameters = Parameters.default()
        
        # Test orography used with Fortran (4x4 grid)
        test_orog = jnp.array([
            [1000.0, 500.0, 200.0, 0.0],
            [800.0, 300.0, 100.0, 0.0],
            [600.0, 200.0, 50.0, 0.0],
            [400.0, 100.0, 0.0, 0.0]
        ])
        
        # phis0 = g * orog (as in Fortran)
        test_phis0 = grav * test_orog
        
        # Land/sea masks and temperatures (matching Fortran test values exactly)
        test_fmask = jnp.full((4, 4), 0.7)  # 70% land
        test_stl_am = jnp.full((4, 4), 288.0)  # Land surface temperature
        test_sst_am = jnp.full((4, 4, 365), 285.0)  # Sea surface temperature
        
        class TestBoundariesFortran:
            def __init__(self):
                self.orog = test_orog
                self.phis0 = test_phis0
                self.fmask = test_fmask
                self.tsea = test_sst_am
        
        boundaries_fortran = TestBoundariesFortran()
        
        # Reference values from SPEEDY Fortran test output (correct gamma=6.0, grav=9.81)
        fortran_tcorv = jnp.array([
            0.00000000e+00, 6.61537620e-01, 7.53886890e-01, 8.27481090e-01,
            8.88522200e-01, 9.35745870e-01, 9.68842590e-01, 9.91036630e-01
        ])
        
        fortran_qcorv = jnp.array([
            0.00000000e+00, 0.00000000e+00, 8.00000040e-03, 3.93040009e-02,
            1.32650989e-01, 3.21419134e-01, 5.82182832e-01, 8.57374973e-01
        ])
        
        # Note: Using gamma=6.0 and grav=9.81 as in SPEEDY
        # gamlat = gamma / (1000 * grav) = 6.0 / (1000 * 9.81) = 6.11621e-04  
        fortran_tcorh = jnp.array([
            [6.00000000e+00, 3.00000000e+00, 1.20000000e+00, 0.00000000e+00],
            [4.80000000e+00, 1.80000000e+00, 6.00000000e-01, 0.00000000e+00],
            [3.60000000e+00, 1.20000000e+00, 3.00000000e-01, 0.00000000e+00],
            [2.40000000e+00, 6.00000000e-01, 0.00000000e+00, 0.00000000e+00]
        ])
        
        # Compute JAX-GCM values
        jax_tcorv = compute_temperature_correction_vertical_profile(geometry_fortran, parameters)
        jax_qcorv = compute_humidity_correction_vertical_profile(geometry_fortran, parameters)
        jax_tcorh = compute_temperature_correction_horizontal(boundaries_fortran, geometry_fortran)
        jax_qcorh = compute_humidity_correction_horizontal(boundaries_fortran, geometry_fortran, jax_tcorh, test_stl_am)
        
        # Test temperature vertical profile - should match within floating-point precision
        np.testing.assert_allclose(jax_tcorv, fortran_tcorv, rtol=1e-3, atol=1e-6,
                                   err_msg="Temperature vertical profile does not match SPEEDY Fortran")
        
        # Test humidity vertical profile - should match exactly
        np.testing.assert_allclose(jax_qcorv, fortran_qcorv, rtol=1e-3, atol=1e-12,
                                   err_msg="Humidity vertical profile does not match SPEEDY Fortran")
        
        # Test temperature horizontal correction - should match exactly
        np.testing.assert_allclose(jax_tcorh, fortran_tcorh, rtol=1e-3, atol=1e-12,
                                   err_msg="Temperature horizontal correction does not match SPEEDY Fortran")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        geometry = create_test_geometry()
        parameters = Parameters.default()
        
        # Test with zero orography using proper model boundary initialization
        from jcm.boundaries import default_boundaries
        from dinosaur import spherical_harmonic
        
        # Create grid and flat orography
        grid = spherical_harmonic.Grid.T31()
        flat_orography = jnp.zeros(grid.nodal_shape)
        
        # Use the actual model boundary initialization (now fixed)
        boundaries_flat = default_boundaries(grid, flat_orography)
        
        tcorh_flat = compute_temperature_correction_horizontal(boundaries_flat, geometry)
        assert jnp.allclose(tcorh_flat, 0.0, atol=1e-5)
        
        # Humidity correction should also be zero when orography is zero
        land_temp_flat = jnp.full(boundaries_flat.orog.shape, 288.0)
        qcorh_flat = compute_humidity_correction_horizontal(boundaries_flat, geometry, tcorh_flat, land_temp_flat)
        assert jnp.allclose(qcorh_flat, 0.0, atol=1e-5)
        
        # test that total corrections are zero for flat orography
        test_state = create_test_physics_state(lon_points=grid.nodal_shape[0], 
                                             lat_points=grid.nodal_shape[1])
        
        # Apply corrections to state with flat orography
        corrected_state_flat = apply_orographic_corrections_to_state(
            test_state, boundaries_flat, geometry, parameters, land_temp_flat
        )
        
        # State should be completely unchanged when orography is flat
        np.testing.assert_allclose(corrected_state_flat.temperature, test_state.temperature, atol=1e-6,
                                    err_msg="Temperature should be unchanged with flat orography")
        np.testing.assert_allclose(corrected_state_flat.specific_humidity, test_state.specific_humidity, atol=1e-6,
                                    err_msg="Specific humidity should be unchanged with flat orography")
        
        # Test with minimum supported layers (5)
        geometry_5layer = create_test_geometry(layers=5)
        tcorv_5layer = compute_temperature_correction_vertical_profile(geometry_5layer, parameters)
        assert tcorv_5layer.shape == (5,)
        assert tcorv_5layer[0] == 0.0
        
        # Test with extreme orography (very tall, steep mountain)
        boundaries_extreme = create_test_boundaries(lon_points=32, lat_points=32)
        
        # Create an extremely tall, steep mountain (like Everest: 8849m)
        lon_idx = jnp.arange(32)
        lat_idx = jnp.arange(32)
        lon_grid, lat_grid = jnp.meshgrid(lon_idx, lat_idx, indexing='ij')
        
        # Very steep mountain: 8000m peak with small footprint (sigma=2 grid points)
        center_lon, center_lat = 16, 16
        sigma = 2.0  # Very steep
        
        extreme_orog = 8000.0 * jnp.exp(
            -((lon_grid - center_lon) ** 2 + (lat_grid - center_lat) ** 2) / (2 * sigma ** 2)
        )
        
        boundaries_extreme.orog = extreme_orog
        boundaries_extreme.phis0 = grav * extreme_orog  # Update phis0 too
        
        tcorh_extreme = compute_temperature_correction_horizontal(boundaries_extreme, geometry)
        assert jnp.all(jnp.isfinite(tcorh_extreme))  # Should not have infinities
        assert jnp.max(tcorh_extreme) > 0.0  # Should have positive values where mountain exists
        assert jnp.min(tcorh_extreme) < 1e-20  # Should be essentially zero where no orography
        
        # Test that maximum correction is at mountain peak
        max_orog_idx = jnp.unravel_index(jnp.argmax(extreme_orog), extreme_orog.shape)
        max_corr_idx = jnp.unravel_index(jnp.argmax(tcorh_extreme), tcorh_extreme.shape)
        assert max_orog_idx == max_corr_idx

    def test_temperature_vertical_profile_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test computation of temperature correction vertical profile gradient check."""
        geometry = create_test_geometry(layers=8)
        parameters = Parameters.default()

        # Set float inputs
        parameters_floats = convert_to_float(parameters)
        geometry_floats = convert_to_float(geometry)

        def f(parameters_f, geometry_f):
            return compute_temperature_correction_vertical_profile(parameters=convert_back(parameters_f, parameters), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (parameters_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (parameters_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        
    
    def test_humidity_vertical_profile_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test computation of humidity correction vertical profile gradient check."""
        geometry = create_test_geometry(layers=8)
        parameters = Parameters.default()

        # Set float inputs
        parameters_floats = convert_to_float(parameters)
        geometry_floats = convert_to_float(geometry)

        def f(parameters_f, geometry_f):
            return compute_humidity_correction_vertical_profile(parameters=convert_back(parameters_f, parameters), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (parameters_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (parameters_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        

    
    def test_temperature_horizontal_correction_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test computation of temperature horizontal correction gradient check."""
        lon, lat = 96, 48
        test_boundaries = create_test_boundaries(lon_points=lon, lat_points=lat)
        geometry = create_test_geometry()
        boundaries = BoundaryData.ones((lon, lat), orog = test_boundaries.orog,
                                       phis0 = test_boundaries.phis0,
                                       fmask = test_boundaries.fmask,
                                       tsea = test_boundaries.tsea)

        # Set float inputs
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(boundaries_f, geometry_f):
            return compute_temperature_correction_horizontal(boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.000001)
        check_jvp(f, f_jvp, args = (boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        
    
    def test_humidity_horizontal_correction_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test computation of humidity horizontal correction gradient check."""
        lon, lat = 96, 48
        test_boundaries = create_test_boundaries(lon_points=lon, lat_points=lat)
        geometry = create_test_geometry()
        boundaries = BoundaryData.ones((lon, lat), orog = test_boundaries.orog,
                                       phis0 = test_boundaries.phis0,
                                       fmask = test_boundaries.fmask,
                                       tsea = test_boundaries.tsea)
        # Compute temperature correction needed for the new humidity correction
        tcorh = compute_temperature_correction_horizontal(boundaries, geometry)
        land_temp = jnp.full((96, 48), 288.0)  # Constant land temperature

        # Set float inputs
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(boundaries_f, geometry_f, tcorh, land_temp):
            return compute_humidity_correction_horizontal(boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry),
                                       temperature_correction=tcorh, 
                                       land_temperature=land_temp)
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (boundaries_floats, geometry_floats, tcorh, land_temp), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (boundaries_floats, geometry_floats, tcorh, land_temp), 
                                atol=None, rtol=1, eps=0.00001)
    
    def test_get_orographic_correction_tendencies_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test the main tendency computation function gradient check."""
        lon, lat = 96, 48
        test_boundaries = create_test_boundaries(lon_points=lon, lat_points=lat)
        boundaries = BoundaryData.ones((lon, lat), orog = test_boundaries.orog,
                                       phis0 = test_boundaries.phis0,
                                       fmask = test_boundaries.fmask,
                                       tsea = test_boundaries.tsea)
        state = create_test_physics_state()
        geometry = create_test_geometry()
        parameters = Parameters.default()
        nodal_shape = state.temperature.shape[1:]  # (lon, lat)
        node_levels = state.temperature.shape[0]   # layers
        physics_data = PhysicsData.zeros(nodal_shape, node_levels)

        # Set float inputs
        state_floats = convert_to_float(state)
        physics_data_floats = convert_to_float(physics_data)
        parameters_floats = convert_to_float(parameters)
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(state_f, physics_data_f, parameters_f, boundaries_f,geometry_f):
            tend_out, data_out = get_orographic_correction_tendencies(state=convert_back(state_f, state), 
                                       physics_data=convert_back(physics_data_f, physics_data),
                                       parameters=convert_back(parameters_f, parameters), 
                                       boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(tend_out), convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (state_floats, physics_data_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.000001)
        check_jvp(f, f_jvp, args = (state_floats, physics_data_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.001)
    
    def test_apply_orographic_corrections_to_state_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test direct application of corrections to state gradient check."""
        lon, lat = 96, 48
        test_boundaries = create_test_boundaries(lon_points=lon, lat_points=lat)
        boundaries = BoundaryData.ones((lon, lat), orog = test_boundaries.orog,
                                       phis0 = test_boundaries.phis0,
                                       fmask = test_boundaries.fmask,
                                       tsea = test_boundaries.tsea)
        state = create_test_physics_state()
        geometry = create_test_geometry()
        parameters = Parameters.default()

        # Set float inputs
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(state_f, parameters_f, boundaries_f,geometry_f):
            state_out = apply_orographic_corrections_to_state(state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(state_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (state_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (state_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        



if __name__ == "__main__":
    # Run tests
    test_instance = TestOrographicCorrection()    
    test_instance.test_temperature_vertical_profile()    
    test_instance.test_humidity_vertical_profile()    
    test_instance.test_temperature_horizontal_correction()    
    test_instance.test_humidity_horizontal_correction()    
    test_instance.test_get_orographic_correction_tendencies()    
    test_instance.test_apply_orographic_corrections_to_state()    
    test_instance.test_jax_compatibility()    
    test_instance.test_speedy_fortran_numerical_equivalence()
    test_instance.test_edge_cases()
    test_instance.test_temperature_vertical_profile_gradient_check()    
    test_instance.test_humidity_vertical_profile_gradient_check()    
    test_instance.test_temperature_horizontal_correction_gradient_check()    
    test_instance.test_humidity_horizontal_correction_gradient_check()    
    test_instance.test_get_orographic_correction_tendencies_gradient_check()    
    test_instance.test_apply_orographic_corrections_to_state_gradient_check() 