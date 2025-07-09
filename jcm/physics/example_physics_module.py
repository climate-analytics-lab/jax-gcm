#!/usr/bin/env python3
"""
Example of how to implement a new physics module using JAX conversion patterns

This demonstrates the standard pattern for adding new physics modules
to the ICON physics framework with automatic vectorization.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.icon.icon_physics import IconPhysicsData
from jcm.physics.icon.constants import physical_constants

def example_radiation_module(state, physics_data, boundaries, geometry):
    """
    Example radiation module demonstrating JAX conversion patterns
    
    This shows how to implement a new physics module that:
    1. Works with 2D arrays [nlev, ncols] 
    2. Uses JAX-compatible operations
    3. Leverages centralized vectorization
    
    Args:
        state: PhysicsState with 2D arrays [nlev, ncols]
        physics_data: IconPhysicsData container
        boundaries: Boundary conditions
        geometry: Model geometry
        
    Returns:
        Tuple of (PhysicsTendency, IconPhysicsData)
    """
    
    # Single-column radiation function
    def radiation_single_column(temp_col, humid_col, pressure_col):
        """Apply radiation to a single atmospheric column"""
        
        # Example: Simple radiation scheme
        nlev = len(temp_col)
        
        # Initialize tendencies
        dtedt = jnp.zeros_like(temp_col)
        
        # Pattern 1: Simple conditional assignment
        # Surface heating
        surface_heating = jnp.where(
            pressure_col > 50000.0,  # Below 500 mb
            2.0e-5,  # 2 K/day heating
            0.0
        )
        
        # Pattern 7: Masked operations
        # Stratospheric cooling
        stratospheric_cooling = jnp.where(
            pressure_col < 10000.0,  # Above 100 mb
            -1.0e-5,  # -1 K/day cooling
            0.0
        )
        
        # Pattern 2: Conditional computation
        # Compute greenhouse effect only if water vapor present
        def compute_greenhouse():
            greenhouse = -3.0e-6 * humid_col * 1000  # Cooling proportional to humidity
            return greenhouse
        
        def no_greenhouse():
            return jnp.zeros_like(temp_col)
        
        greenhouse_effect = lax.cond(
            jnp.any(humid_col > 1e-6),
            compute_greenhouse,
            no_greenhouse
        )
        
        # Combine all effects
        dtedt = surface_heating + stratospheric_cooling + greenhouse_effect
        
        return dtedt
    
    # Pattern 10: Centralized vectorization
    # Apply to all columns using vmap (state is already 2D)
    from jcm.physics.speedy.physical_constants import p0
    
    # Calculate pressure levels
    surface_pressure = state.surface_pressure * p0
    sigma_levels = geometry.fsg
    pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
    
    # Vectorize over all columns
    heating_tendencies = jax.vmap(
        radiation_single_column,
        in_axes=(1, 1, 1),  # vmap over column dimension (axis 1)
        out_axes=1
    )(state.temperature, state.specific_humidity, pressure_levels)
    
    # Create physics tendencies (already in 2D format [nlev, ncols])
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=heating_tendencies,
        specific_humidity=jnp.zeros_like(state.specific_humidity)
    )
    
    # Update physics data
    updated_physics_data = physics_data.copy(
        radiation_data={
            'radiation_enabled': True,
            'last_applied': True
        }
    )
    
    return physics_tendencies, updated_physics_data


def example_cloud_module(state, physics_data, boundaries, geometry):
    """
    Example cloud microphysics module demonstrating more complex patterns
    
    Shows Pattern 5 (Accumulating scan) and Pattern 8 (Nested conditionals)
    """
    
    def cloud_single_column(temp_col, humid_col, pressure_col):
        """Apply cloud microphysics to a single column"""
        
        nlev = len(temp_col)
        
        # Initialize tendencies
        dtedt = jnp.zeros_like(temp_col)
        dqdt = jnp.zeros_like(humid_col)
        
        # Pattern 5: Accumulating scan for precipitation
        def precipitation_step(carry, level_inputs):
            k, temp, humid, pressure = level_inputs
            precip_above = carry
            
            # Saturation mixing ratio
            es = 611.2 * jnp.exp(17.67 * (temp - 273.15) / (temp - 29.65))
            qs = 0.622 * es / (pressure - 0.378 * es)
            
            # Pattern 8: Nested conditionals
            def compute_condensation():
                excess_humidity = humid - qs
                
                def large_excess():
                    condensation = 0.5 * excess_humidity  # 50% condensation
                    return condensation, 0.1 * condensation  # 10% falls as precip
                
                def small_excess():
                    condensation = 0.1 * excess_humidity  # 10% condensation
                    return condensation, 0.0  # No precipitation
                
                return lax.cond(
                    excess_humidity > 0.001,  # 1 g/kg threshold
                    large_excess,
                    small_excess
                )
            
            def no_condensation():
                return 0.0, 0.0
            
            condensation, new_precip = lax.cond(
                humid > qs,
                compute_condensation,
                no_condensation
            )
            
            # Latent heating
            heating = condensation * 2.5e6 / 1005.0  # L_v / c_p
            
            # Total precipitation
            total_precip = precip_above + new_precip
            
            return total_precip, (heating, -condensation)
        
        # Apply scan from top to bottom
        k_levels = jnp.arange(nlev)
        level_inputs = (k_levels, temp_col, humid_col, pressure_col)
        
        final_precip, (heating_profile, drying_profile) = lax.scan(
            precipitation_step,
            0.0,  # Initial precipitation
            level_inputs
        )
        
        return heating_profile, drying_profile
    
    # Calculate pressure levels
    from jcm.physics.speedy.physical_constants import p0
    surface_pressure = state.surface_pressure * p0
    sigma_levels = geometry.fsg
    pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
    
    # Vectorize over all columns
    heating_tendencies, drying_tendencies = jax.vmap(
        cloud_single_column,
        in_axes=(1, 1, 1),
        out_axes=(1, 1)
    )(state.temperature, state.specific_humidity, pressure_levels)
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=heating_tendencies,
        specific_humidity=drying_tendencies
    )
    
    # Update physics data
    updated_physics_data = physics_data.copy(
        cloud_data={
            'cloud_enabled': True,
            'last_applied': True
        }
    )
    
    return physics_tendencies, updated_physics_data


def demonstrate_module_integration():
    """
    Demonstrate how to integrate new modules into IconPhysics
    """
    
    print("JAX Physics Module Integration Example")
    print("=" * 50)
    
    # This is how you would add the modules to IconPhysics
    example_integration = """
    # In IconPhysics.__init__():
    if self.enable_radiation:
        terms.append(self._apply_radiation)
    
    if self.enable_clouds:
        terms.append(self._apply_clouds)
    
    # In IconPhysics class:
    def _apply_radiation(self, state, physics_data, boundaries, geometry):
        return example_radiation_module(state, physics_data, boundaries, geometry)
    
    def _apply_clouds(self, state, physics_data, boundaries, geometry):
        return example_cloud_module(state, physics_data, boundaries, geometry)
    """
    
    print("Integration pattern:")
    print(example_integration)
    
    print("\nKey benefits of this approach:")
    print("✅ Automatic vectorization via centralized compute_tendencies")
    print("✅ Clean separation of physics algorithms")
    print("✅ Consistent JAX compatibility patterns")
    print("✅ Easy testing and validation")
    print("✅ Reusable across different physics schemes")
    
    print("\nPerformance characteristics:")
    print("- Each module processes all columns simultaneously")
    print("- No explicit spatial loops in physics code")
    print("- JAX JIT compilation for optimal performance")
    print("- GPU/TPU ready with no code changes")
    
    return True


if __name__ == "__main__":
    demonstrate_module_integration()
    print("\n🎉 Physics module integration patterns demonstrated!")