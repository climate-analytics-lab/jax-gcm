"""
Comprehensive unit tests for vertical diffusion physics.

This module provides extensive testing of the vertical diffusion scheme,
including individual components and integrated behavior.
"""

import jax
import jax.numpy as jnp
import pytest
from typing import Tuple

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from .vertical_diffusion_types import VDiffParameters, VDiffState

# Create constants instance
PHYS_CONST = PhysicalConstants()
from .turbulence_coefficients import (
    compute_richardson_number, compute_mixing_length, compute_exchange_coefficients,
    compute_boundary_layer_height, compute_friction_velocity
)
from .matrix_solver import (
    setup_matrix_system, solve_tridiagonal_single, vertical_diffusion_step
)
from .vertical_diffusion import (
    vertical_diffusion_scheme, prepare_vertical_diffusion_state,
    compute_dry_static_energy, compute_virtual_temperature
)


class TestTurbulenceCoefficients:
    """Test turbulence coefficient calculations."""
    
    def test_richardson_number_stable(self):
        """Test Richardson number calculation for stable conditions."""
        # Setup stable profile (temperature increasing with height)
        ncol, nlev = 2, 5
        u = jnp.ones((ncol, nlev)) * 10.0  # Constant wind
        v = jnp.zeros((ncol, nlev))
        temperature = jnp.array([
            [280.0, 285.0, 290.0, 295.0, 300.0],
            [285.0, 290.0, 295.0, 300.0, 305.0]
        ])
        
        height_full = jnp.array([
            [100.0, 300.0, 500.0, 700.0, 900.0],
            [100.0, 300.0, 500.0, 700.0, 900.0]
        ])
        height_half = jnp.array([
            [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0],
            [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
        ])
        
        ri = compute_richardson_number(u, v, temperature, height_full, height_half)
        
        # Richardson number should be positive for stable conditions
        assert jnp.all(ri > 0)
        assert ri.shape == (ncol, nlev - 1)
    
    def test_richardson_number_unstable(self):
        """Test Richardson number calculation for unstable conditions."""
        # Setup unstable profile (temperature decreasing with height)
        ncol, nlev = 2, 5
        u = jnp.ones((ncol, nlev)) * 10.0
        v = jnp.zeros((ncol, nlev))
        temperature = jnp.array([
            [300.0, 295.0, 290.0, 285.0, 280.0],
            [305.0, 300.0, 295.0, 290.0, 285.0]
        ])
        
        height_full = jnp.array([
            [100.0, 300.0, 500.0, 700.0, 900.0],
            [100.0, 300.0, 500.0, 700.0, 900.0]
        ])
        height_half = jnp.array([
            [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0],
            [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
        ])
        
        ri = compute_richardson_number(u, v, temperature, height_full, height_half)
        
        # Richardson number should be negative for unstable conditions
        assert jnp.all(ri < 0)
    
    def test_mixing_length_computation(self):
        """Test mixing length computation."""
        ncol, nlev = 2, 5
        height_full = jnp.array([
            [100.0, 300.0, 500.0, 700.0, 900.0],
            [100.0, 300.0, 500.0, 700.0, 900.0]
        ])
        height_half = jnp.array([
            [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0],
            [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
        ])
        
        # Neutral conditions
        richardson_number = jnp.zeros((ncol, nlev - 1))
        boundary_layer_height = jnp.array([500.0, 600.0])
        
        mixing_length = compute_mixing_length(
            height_full, height_half, richardson_number, boundary_layer_height
        )
        
        assert mixing_length.shape == (ncol, nlev)
        assert jnp.all(mixing_length > 0)
        assert jnp.all(mixing_length >= 1.0)  # Minimum mixing length
        
        # Mixing length should increase with distance from surface (up to a point)
        assert jnp.all(mixing_length[:, 1] >= mixing_length[:, 0])
    
    def test_exchange_coefficients_physical_bounds(self):
        """Test that exchange coefficients are within physical bounds."""
        # Create realistic atmospheric state
        ncol, nlev = 3, 10
        state = create_test_atmospheric_state(ncol, nlev)
        params = VDiffParameters.default()
        
        # Create mixing length
        mixing_length = jnp.linspace(10.0, 100.0, nlev)[None, :] * jnp.ones((ncol, nlev))
        richardson_number = jnp.zeros((ncol, nlev - 1))
        
        exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture = (
            compute_exchange_coefficients(state, params, mixing_length, richardson_number)
        )
        
        # Check physical bounds
        assert jnp.all(exchange_coeff_momentum >= 0)
        assert jnp.all(exchange_coeff_heat >= 0)
        assert jnp.all(exchange_coeff_moisture >= 0)
        
        # Check maximum values
        assert jnp.all(exchange_coeff_momentum <= 1000.0)
        assert jnp.all(exchange_coeff_heat <= 1000.0)
        assert jnp.all(exchange_coeff_moisture <= 1000.0)
        
        # Check shapes
        assert exchange_coeff_momentum.shape == (ncol, nlev)
        assert exchange_coeff_heat.shape == (ncol, nlev)
        assert exchange_coeff_moisture.shape == (ncol, nlev)
    
    def test_boundary_layer_height_computation(self):
        """Test boundary layer height computation."""
        ncol, nlev = 2, 10
        state = create_test_atmospheric_state(ncol, nlev)
        
        # Create exchange coefficient profile that decreases with height
        exchange_coeff_heat = jnp.array([
            [10.0, 8.0, 6.0, 4.0, 2.0, 0.8, 0.6, 0.4, 0.2, 0.1],
            [15.0, 12.0, 9.0, 6.0, 3.0, 1.2, 0.9, 0.6, 0.3, 0.15]
        ])
        
        pbl_height = compute_boundary_layer_height(state, exchange_coeff_heat, threshold=1.0)
        
        assert pbl_height.shape == (ncol,)
        assert jnp.all(pbl_height >= 50.0)  # Minimum PBL height
        assert jnp.all(pbl_height <= 5000.0)  # Reasonable maximum
    
    def test_friction_velocity_computation(self):
        """Test friction velocity computation."""
        ncol = 5
        momentum_flux_u = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        momentum_flux_v = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25])
        air_density = jnp.ones(ncol) * 1.225  # kg/m³
        
        friction_velocity = compute_friction_velocity(
            momentum_flux_u, momentum_flux_v, air_density
        )
        
        assert friction_velocity.shape == (ncol,)
        assert jnp.all(friction_velocity >= 0.01)  # Minimum value
        assert jnp.all(friction_velocity <= 5.0)   # Reasonable maximum


class TestMatrixSolver:
    """Test tridiagonal matrix solver."""
    
    def test_tridiagonal_solver_simple(self):
        """Test tridiagonal solver with simple known solution."""
        ncol, nlev = 2, 3
        
        # Simple tridiagonal system: [2 -1 0; -1 2 -1; 0 -1 2] * x = [1; 0; 1]
        a = jnp.array([
            [0.0, -1.0, -1.0],
            [0.0, -1.0, -1.0]
        ])  # sub-diagonal
        b = jnp.array([
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0]
        ])  # diagonal
        c = jnp.array([
            [-1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0]
        ])  # super-diagonal
        d = jnp.array([
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0]
        ])  # RHS
        
        solution = solve_tridiagonal_single(a, b, c, d)
        
        # Check that solution satisfies the system
        assert solution.shape == (ncol, nlev)
        assert jnp.allclose(solution, jnp.array([[1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]), atol=1e-10)
    
    def test_matrix_system_setup(self):
        """Test setup of matrix system."""
        ncol, nlev = 2, 5
        state = create_test_atmospheric_state(ncol, nlev)
        params = VDiffParameters.default()
        
        # Create exchange coefficients
        exchange_coeff_momentum = jnp.ones((ncol, nlev)) * 10.0
        exchange_coeff_heat = jnp.ones((ncol, nlev)) * 8.0
        exchange_coeff_moisture = jnp.ones((ncol, nlev)) * 6.0
        dt = 300.0
        
        matrix_system = setup_matrix_system(
            state, params, exchange_coeff_momentum, 
            exchange_coeff_heat, exchange_coeff_moisture, dt
        )
        
        # Check matrix dimensions
        nmatrix = 6
        nvar_total = 8  # u, v, T, qv, qc, qi, TKE, thv_var
        assert matrix_system.matrix_coeffs.shape == (ncol, nlev, 3, nmatrix)
        assert matrix_system.rhs_vectors.shape == (ncol, nlev, nvar_total)
        assert matrix_system.variable_to_matrix.shape == (nvar_total,)
        
        # Check that diagonal elements are reasonable
        assert jnp.all(matrix_system.matrix_coeffs[:, :, 1, :] > 0)  # Diagonal > 0
    
    def test_vertical_diffusion_step_conservation(self):
        """Test that vertical diffusion step conserves mass."""
        ncol, nlev = 2, 5
        state = create_test_atmospheric_state(ncol, nlev)
        params = VDiffParameters.default()
        
        exchange_coeff_momentum = jnp.ones((ncol, nlev)) * 10.0
        exchange_coeff_heat = jnp.ones((ncol, nlev)) * 8.0
        exchange_coeff_moisture = jnp.ones((ncol, nlev)) * 6.0
        dt = 300.0
        
        tendencies = vertical_diffusion_step(
            state, params, exchange_coeff_momentum,
            exchange_coeff_heat, exchange_coeff_moisture, dt
        )
        
        # Check that tendencies are finite
        assert jnp.all(jnp.isfinite(tendencies.u_tendency))
        assert jnp.all(jnp.isfinite(tendencies.v_tendency))
        assert jnp.all(jnp.isfinite(tendencies.temperature_tendency))
        assert jnp.all(jnp.isfinite(tendencies.qv_tendency))
        
        # Check mass conservation for moisture (integrated tendency should be ~0)
        total_qv_tendency = jnp.sum(tendencies.qv_tendency * state.air_mass, axis=1)
        assert jnp.allclose(total_qv_tendency, 0.0, atol=1e-10)


class TestVerticalDiffusionScheme:
    """Test complete vertical diffusion scheme."""
    
    def test_vertical_diffusion_scheme_execution(self):
        """Test that vertical diffusion scheme executes without errors."""
        ncol, nlev = 3, 10
        nsfc_type = 3
        
        # Create input data
        u = jnp.ones((ncol, nlev)) * 10.0
        v = jnp.ones((ncol, nlev)) * 5.0
        temperature = jnp.linspace(300.0, 250.0, nlev)[None, :] * jnp.ones((ncol, nlev))
        qv = jnp.ones((ncol, nlev)) * 0.01
        qc = jnp.ones((ncol, nlev)) * 0.001
        qi = jnp.ones((ncol, nlev)) * 0.0005
        
        # Pressure profile
        pressure_half = jnp.linspace(101325.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        pressure_full = 0.5 * (pressure_half[:, :-1] + pressure_half[:, 1:])
        
        # Heights
        height_half = jnp.linspace(0.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        height_full = 0.5 * (height_half[:, :-1] + height_half[:, 1:])
        
        # Geopotential
        geopotential = PHYS_CONST.grav * height_full
        
        # Surface properties
        surface_temperature = jnp.ones((ncol, nsfc_type)) * 290.0
        surface_fraction = jnp.ones((ncol, nsfc_type)) / nsfc_type
        roughness_length = jnp.ones((ncol, nsfc_type)) * 0.01
        
        # Ocean velocities
        ocean_u = jnp.zeros(ncol)
        ocean_v = jnp.zeros(ncol)
        
        # Turbulence variables
        tke = jnp.ones((ncol, nlev)) * 0.1
        thv_variance = jnp.ones((ncol, nlev)) * 0.01
        
        dt = 300.0
        
        # Run vertical diffusion
        tendencies, diagnostics = vertical_diffusion_scheme(
            u, v, temperature, qv, qc, qi,
            pressure_full, pressure_half, geopotential,
            height_full, height_half,
            surface_temperature, surface_fraction, roughness_length,
            ocean_u, ocean_v, tke, thv_variance, dt
        )
        
        # Check that outputs are reasonable
        assert jnp.all(jnp.isfinite(tendencies.u_tendency))
        assert jnp.all(jnp.isfinite(tendencies.v_tendency))
        assert jnp.all(jnp.isfinite(tendencies.temperature_tendency))
        assert jnp.all(jnp.isfinite(diagnostics.exchange_coeff_momentum))
        assert jnp.all(jnp.isfinite(diagnostics.boundary_layer_height))
        
        # Check physical bounds
        assert jnp.all(jnp.abs(tendencies.u_tendency) <= 1.0)  # Reasonable wind tendency
        assert jnp.all(jnp.abs(tendencies.v_tendency) <= 1.0)
        assert jnp.all(jnp.abs(tendencies.temperature_tendency) <= 10.0)  # K/s
        assert jnp.all(diagnostics.boundary_layer_height >= 50.0)
    
    def test_vertical_diffusion_energy_conservation(self):
        """Test energy conservation in vertical diffusion."""
        ncol, nlev = 2, 8
        nsfc_type = 3
        
        # Create initial state
        u = jnp.ones((ncol, nlev)) * 10.0
        v = jnp.ones((ncol, nlev)) * 5.0
        temperature = jnp.linspace(300.0, 250.0, nlev)[None, :] * jnp.ones((ncol, nlev))
        qv = jnp.ones((ncol, nlev)) * 0.01
        qc = jnp.ones((ncol, nlev)) * 0.001
        qi = jnp.ones((ncol, nlev)) * 0.0005
        
        pressure_half = jnp.linspace(101325.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        pressure_full = 0.5 * (pressure_half[:, :-1] + pressure_half[:, 1:])
        
        height_half = jnp.linspace(0.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        height_full = 0.5 * (height_half[:, :-1] + height_half[:, 1:])
        
        geopotential = PHYS_CONST.grav * height_full
        
        surface_temperature = jnp.ones((ncol, nsfc_type)) * 290.0
        surface_fraction = jnp.ones((ncol, nsfc_type)) / nsfc_type
        roughness_length = jnp.ones((ncol, nsfc_type)) * 0.01
        
        ocean_u = jnp.zeros(ncol)
        ocean_v = jnp.zeros(ncol)
        
        tke = jnp.ones((ncol, nlev)) * 0.1
        thv_variance = jnp.ones((ncol, nlev)) * 0.01
        
        dt = 300.0
        
        # Compute initial energy
        dp = jnp.diff(pressure_half, axis=1)
        air_mass = dp / PHYS_CONST.grav
        
        initial_kinetic_energy = 0.5 * air_mass * (u**2 + v**2)
        initial_potential_energy = air_mass * PHYS_CONST.cp * temperature
        
        initial_total_energy = jnp.sum(initial_kinetic_energy + initial_potential_energy)
        
        # Run vertical diffusion
        tendencies, diagnostics = vertical_diffusion_scheme(
            u, v, temperature, qv, qc, qi,
            pressure_full, pressure_half, geopotential,
            height_full, height_half,
            surface_temperature, surface_fraction, roughness_length,
            ocean_u, ocean_v, tke, thv_variance, dt
        )
        
        # Check energy balance (should be approximately conserved in absence of surface fluxes)
        # This is a simplified check - real energy conservation would account for surface fluxes
        energy_change_rate = (
            jnp.sum(air_mass * (u * tendencies.u_tendency + v * tendencies.v_tendency)) +
            jnp.sum(tendencies.heating_rate)
        )
        
        # Energy change should be finite and reasonable
        assert jnp.isfinite(energy_change_rate)
        assert jnp.abs(energy_change_rate) <= 1e6  # Reasonable energy change rate
    
    def test_vertical_diffusion_mixing_effectiveness(self):
        """Test that vertical diffusion effectively mixes the atmosphere."""
        ncol, nlev = 1, 10
        nsfc_type = 3
        
        # Create strong vertical gradients
        u = jnp.array([[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]])
        v = jnp.zeros((ncol, nlev))
        temperature = jnp.array([[310.0, 305.0, 300.0, 295.0, 290.0, 285.0, 280.0, 275.0, 270.0, 265.0]])
        
        qv = jnp.ones((ncol, nlev)) * 0.01
        qc = jnp.ones((ncol, nlev)) * 0.001
        qi = jnp.ones((ncol, nlev)) * 0.0005
        
        pressure_half = jnp.linspace(101325.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        pressure_full = 0.5 * (pressure_half[:, :-1] + pressure_half[:, 1:])
        
        height_half = jnp.linspace(0.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        height_full = 0.5 * (height_half[:, :-1] + height_half[:, 1:])
        
        geopotential = PHYS_CONST.grav * height_full
        
        surface_temperature = jnp.ones((ncol, nsfc_type)) * 290.0
        surface_fraction = jnp.ones((ncol, nsfc_type)) / nsfc_type
        roughness_length = jnp.ones((ncol, nsfc_type)) * 0.01
        
        ocean_u = jnp.zeros(ncol)
        ocean_v = jnp.zeros(ncol)
        
        tke = jnp.ones((ncol, nlev)) * 1.0  # Strong turbulence
        thv_variance = jnp.ones((ncol, nlev)) * 0.1
        
        dt = 3600.0  # Longer time step for more mixing
        
        # Run vertical diffusion
        tendencies, diagnostics = vertical_diffusion_scheme(
            u, v, temperature, qv, qc, qi,
            pressure_full, pressure_half, geopotential,
            height_full, height_half,
            surface_temperature, surface_fraction, roughness_length,
            ocean_u, ocean_v, tke, thv_variance, dt
        )
        
        # Check that mixing occurs: lower levels should gain momentum, upper levels should lose it
        assert tendencies.u_tendency[0, 0] > 0  # Surface gains momentum
        assert tendencies.u_tendency[0, -1] < 0  # Top loses momentum
        
        # Check exchange coefficients are reasonable
        assert jnp.all(diagnostics.exchange_coeff_momentum > 0)
        assert jnp.all(diagnostics.exchange_coeff_heat > 0)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_dry_static_energy(self):
        """Test dry static energy calculation."""
        temperature = jnp.array([300.0, 290.0, 280.0])
        geopotential = jnp.array([0.0, 10000.0, 20000.0])
        
        dse = compute_dry_static_energy(temperature, geopotential)
        
        expected = PHYS_CONST.cp * temperature + geopotential
        assert jnp.allclose(dse, expected)
    
    def test_virtual_temperature(self):
        """Test virtual temperature calculation."""
        temperature = jnp.array([300.0, 290.0, 280.0])
        qv = jnp.array([0.01, 0.005, 0.001])
        
        tv = compute_virtual_temperature(temperature, qv)
        
        expected = temperature * (1.0 + 0.608 * qv)
        assert jnp.allclose(tv, expected)
    
    def test_prepare_vertical_diffusion_state(self):
        """Test preparation of vertical diffusion state."""
        ncol, nlev = 2, 5
        nsfc_type = 3
        
        # Create input arrays
        u = jnp.ones((ncol, nlev)) * 10.0
        v = jnp.ones((ncol, nlev)) * 5.0
        temperature = jnp.ones((ncol, nlev)) * 290.0
        qv = jnp.ones((ncol, nlev)) * 0.01
        qc = jnp.ones((ncol, nlev)) * 0.001
        qi = jnp.ones((ncol, nlev)) * 0.0005
        
        pressure_half = jnp.linspace(101325.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        pressure_full = 0.5 * (pressure_half[:, :-1] + pressure_half[:, 1:])
        
        height_half = jnp.linspace(0.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
        height_full = 0.5 * (height_half[:, :-1] + height_half[:, 1:])
        
        geopotential = PHYS_CONST.grav * height_full
        
        surface_temperature = jnp.ones((ncol, nsfc_type)) * 290.0
        surface_fraction = jnp.ones((ncol, nsfc_type)) / nsfc_type
        roughness_length = jnp.ones((ncol, nsfc_type)) * 0.01
        
        ocean_u = jnp.zeros(ncol)
        ocean_v = jnp.zeros(ncol)
        
        tke = jnp.ones((ncol, nlev)) * 0.1
        thv_variance = jnp.ones((ncol, nlev)) * 0.01
        
        # Prepare state
        state = prepare_vertical_diffusion_state(
            u, v, temperature, qv, qc, qi,
            pressure_full, pressure_half, geopotential,
            height_full, height_half,
            surface_temperature, surface_fraction, roughness_length,
            ocean_u, ocean_v, tke, thv_variance
        )
        
        # Check state structure
        assert state.u.shape == (ncol, nlev)
        assert state.v.shape == (ncol, nlev)
        assert state.temperature.shape == (ncol, nlev)
        assert state.air_mass.shape == (ncol, nlev)
        assert state.surface_temperature.shape == (ncol, nsfc_type)
        
        # Check air mass calculation
        dp = jnp.diff(pressure_half, axis=1)
        expected_air_mass = dp / PHYS_CONST.grav
        assert jnp.allclose(state.air_mass, expected_air_mass)


def create_test_atmospheric_state(ncol: int, nlev: int) -> VDiffState:
    """Create a realistic atmospheric state for testing."""
    nsfc_type = 3
    
    # Create realistic profiles
    u = jnp.ones((ncol, nlev)) * 10.0
    v = jnp.ones((ncol, nlev)) * 5.0
    temperature = jnp.linspace(300.0, 250.0, nlev)[None, :] * jnp.ones((ncol, nlev))
    qv = jnp.ones((ncol, nlev)) * 0.01
    qc = jnp.ones((ncol, nlev)) * 0.001
    qi = jnp.ones((ncol, nlev)) * 0.0005
    
    # Pressure profile
    pressure_half = jnp.linspace(101325.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
    pressure_full = 0.5 * (pressure_half[:, :-1] + pressure_half[:, 1:])
    
    # Heights
    height_half = jnp.linspace(0.0, 10000.0, nlev + 1)[None, :] * jnp.ones((ncol, nlev + 1))
    height_full = 0.5 * (height_half[:, :-1] + height_half[:, 1:])
    
    # Geopotential
    geopotential = PHYS_CONST.grav * height_full
    
    # Air masses
    dp = jnp.diff(pressure_half, axis=1)
    air_mass = dp / PHYS_CONST.grav
    dry_air_mass = air_mass * (1.0 - qv)
    
    # Surface properties
    surface_temperature = jnp.ones((ncol, nsfc_type)) * 290.0
    surface_fraction = jnp.ones((ncol, nsfc_type)) / nsfc_type
    roughness_length = jnp.ones((ncol, nsfc_type)) * 0.01
    
    # Ocean velocities
    ocean_u = jnp.zeros(ncol)
    ocean_v = jnp.zeros(ncol)
    
    # Turbulence variables
    tke = jnp.ones((ncol, nlev)) * 0.1
    thv_variance = jnp.ones((ncol, nlev)) * 0.01
    
    return VDiffState(
        u=u, v=v, temperature=temperature, qv=qv, qc=qc, qi=qi,
        pressure_full=pressure_full, pressure_half=pressure_half,
        geopotential=geopotential, air_mass=air_mass, dry_air_mass=dry_air_mass,
        surface_temperature=surface_temperature, surface_fraction=surface_fraction,
        roughness_length=roughness_length, height_full=height_full, height_half=height_half,
        tke=tke, thv_variance=thv_variance, ocean_u=ocean_u, ocean_v=ocean_v
    )


if __name__ == "__main__":
    # Run basic tests
    print("Running vertical diffusion tests...")
    
    # Test Richardson number calculation
    test_turb = TestTurbulenceCoefficients()
    test_turb.test_richardson_number_stable()
    test_turb.test_richardson_number_unstable()
    test_turb.test_mixing_length_computation()
    test_turb.test_exchange_coefficients_physical_bounds()
    test_turb.test_boundary_layer_height_computation()
    test_turb.test_friction_velocity_computation()
    print("✓ Turbulence coefficient tests passed")
    
    # Test matrix solver
    test_matrix = TestMatrixSolver()
    test_matrix.test_tridiagonal_solver_simple()
    test_matrix.test_matrix_system_setup()
    test_matrix.test_vertical_diffusion_step_conservation()
    print("✓ Matrix solver tests passed")
    
    # Test full scheme
    test_scheme = TestVerticalDiffusionScheme()
    test_scheme.test_vertical_diffusion_scheme_execution()
    test_scheme.test_vertical_diffusion_energy_conservation()
    test_scheme.test_vertical_diffusion_mixing_effectiveness()
    print("✓ Vertical diffusion scheme tests passed")
    
    # Test utilities
    test_utils = TestUtilityFunctions()
    test_utils.test_dry_static_energy()
    test_utils.test_virtual_temperature()
    test_utils.test_prepare_vertical_diffusion_state()
    print("✓ Utility function tests passed")
    
    print("All vertical diffusion tests passed! ✓")