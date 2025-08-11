"""
Radiative-Convective Equilibrium (RCE) Test

This module implements a single-column RCE test that mimics the swirl-jatmos approach:
- RK2 timestepping with T_rhs function
- Radiation-only heating (no convection like swirl-jatmos)
- Also RCE with simple convective adjustment
- Constant relative humidity adjustment
- Clear sky assumption (no clouds)
- Realistic atmospheric state configuration
- Proper solar geometry calculation from lat/lon/day/time
"""

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# Import JAX-GCM radiation schemes
from jcm.physics.icon.radiation.radiation_scheme_rrtmgp import radiation_scheme_rrtmgp
from jcm.physics.icon.radiation.radiation_scheme import radiation_scheme as radiation_scheme_icon
from jcm.physics.icon.radiation.radiation_types import RadiationParameters
from jcm.physics.icon.icon_physics_data import AerosolData
from jcm.physics.icon.constants.physical_constants import physical_constants
from jcm.physics.icon.radiation.radiation_scheme_test import create_test_atmosphere, create_default_aerosol_data
from jcm.physics.icon.radiation import cosine_solar_zenith_angle, calculate_solar_radiation_gcm
from jcm.physics.icon.surface import SurfaceParameters

# Constants
SURFACE_TEMPERATURE = 300.0  # Fixed surface temperature for RCE [K]
MOIST_ADIABATIC_LAPSE_RATE = 6.5  # Moist adiabatic lapse rate [K/km]


@dataclass
class RCESetup:
    """Configuration parameters for RCE test setup.
    
    This class contains all parameters needed to configure the RCE simulation including
    atmospheric configuration, simulation parameters, radiation scheme selection,
    solar conditions, and surface properties.
    """
    # Atmospheric configuration
    nlev: int = 32                    # Number of vertical levels
    
    # Simulation parameters
    dt: float = 86400.0               # Time step [s] (1 day)
    max_steps: int = 50              # Maximum integration steps
    convergence_threshold: float = 1e-4  # Temperature convergence threshold [K]
    
    # Radiation scheme selection
    radiation_scheme: str = "rrtmgp"    # "rrtmgp" or "icon"
    
    # Solar conditions (equatorial, noon, summer)
    day_of_year: float = 172.0        # Summer solstice
    seconds_since_midnight: float = 43200.0  # Noon
    latitude: float = 0.0             # Equator
    longitude: float = 0.0            # Prime meridian
    
    # Surface properties (like swirl-jatmos)
    surface_emissivity: float = 0.98  # Surface longwave emissivity
    surface_albedo_vis: float = 0.07  # Visible surface albedo (ocean-like)
    surface_albedo_nir: float = 0.07  # Near-IR surface albedo (ocean-like)
    
    # Convective adjustment
    convective_adjustment: bool = True  # Enable convective adjustment for RCE


@dataclass
class RCEState:
    """State variables for RCE simulation.
    
    Contains all atmospheric state variables, heating rates, and surface
    conditions needed for the RCE simulation.
    """
    # Atmospheric state
    temperature: jnp.ndarray          # Temperature [K] (nlev,)
    pressure: jnp.ndarray             # Pressure [Pa] (nlev,)
    specific_humidity: jnp.ndarray    # Specific humidity [kg/kg] (nlev,)
    cloud_water: jnp.ndarray          # Cloud liquid water [kg/kg] (nlev,)
    cloud_ice: jnp.ndarray            # Cloud ice [kg/kg] (nlev,)
    cloud_fraction: jnp.ndarray       # Cloud fraction [-] (nlev,)
    layer_thickness: jnp.ndarray      # Layer thickness [m] (nlev,)
    air_density: jnp.ndarray          # Air density [kg/m³] (nlev,)
    height: jnp.ndarray               # Height [m] (nlev,)
    
    # Heating rates
    radiative_heating: jnp.ndarray    # Radiative heating [K/s] (nlev,)
    total_heating: jnp.ndarray        # Total heating [K/s] (nlev,)
    
    # Surface state
    net_energy_flux: float           # Net energy flux at surface [W/m²]


def compute_saturation_humidity(temperature: jnp.ndarray, pressure: jnp.ndarray, relative_humidity: float = 0.75) -> jnp.ndarray:
    """Compute specific humidity given constant relative humidity.
    Implements thermodynamic saturation adjustment based on Clausius-Clapeyron equation.
    
    Args:
        temperature: Temperature [K]
        pressure: Pressure [Pa]
        relative_humidity: Relative humidity (default: 0.75 like swirl-jatmos)
        
    Returns:
        Specific humidity [kg/kg]
    """
    # Constants from JAX-GCM physical constants
    R_d = physical_constants.rd  # J/(kg·K) - gas constant for dry air
    R_v = physical_constants.rv  # J/(kg·K) - gas constant for water vapor
    
    # Saturation vapor pressure (Clausius-Clapeyron)
    L_v = physical_constants.alhc  # J/kg - latent heat of vaporization
    T_ref = physical_constants.t0  # K - reference temperature
    p_ref = 611.0  # Pa - reference vapor pressure at T_ref
    
    # Saturation vapor pressure
    p_v_sat = p_ref * jnp.exp(L_v / R_v * (1/T_ref - 1/temperature))
    
    # Specific humidity from relative humidity
    alpha = relative_humidity * R_d / R_v * p_v_sat / (pressure - p_v_sat)
    q_t = alpha / (1 + alpha)
    
    return q_t

class RCESolver:
    """Radiative-Convective Equilibrium solver.
    
    Implements a single-column RCE model following the swirl-jatmos approach.
    Uses RK2 timestepping.
    """
    
    def __init__(self, setup: RCESetup):
        """Initialize RCE solver with given configuration.
        
        Args:
            setup: Configuration parameters for the RCE simulation
        """
        self.setup = setup
        
        # Initialize radiation parameters with realistic values
        self.radiation_params = RadiationParameters.default(
            surface_emissivity=setup.surface_emissivity,
            surface_albedo_vis=setup.surface_albedo_vis,
            surface_albedo_nir=setup.surface_albedo_nir,
            solar_constant=calculate_solar_radiation_gcm(
                setup.day_of_year, setup.seconds_since_midnight, setup.longitude, setup.latitude
            )[0]  # Use solar irradiance (flux)
        )
        
        # Initialize aerosol data
        self.aerosol_data = create_default_aerosol_data(setup.nlev, self.radiation_params)
        
        # Select radiation scheme
        if setup.radiation_scheme.lower() == "rrtmgp":
            self.radiation_fn = radiation_scheme_rrtmgp
        else:
            self.radiation_fn = radiation_scheme_icon
        
        # Initialize surface parameters for energy balance calculations
        self.surface_params = SurfaceParameters.default()
    
    def create_initial_state(self) -> RCEState:
        """Create initial RCE state using test atmosphere.
        
        Returns:
            Initial RCE state with atmospheric profiles and zero heating rates
        """
        # Use test atmosphere
        atm = create_test_atmosphere(self.setup.nlev)
        
        # Initialize heating arrays
        radiative_heating = jnp.zeros(self.setup.nlev)
        total_heating = jnp.zeros(self.setup.nlev)
        
        # Clear sky like swirl-jatmos - no clouds
        cloud_water = jnp.zeros(self.setup.nlev)
        cloud_ice = jnp.zeros(self.setup.nlev)
        cloud_fraction = jnp.zeros(self.setup.nlev)
        
        return RCEState(
            temperature=atm['temperature'],
            pressure=atm['pressure_levels'],
            specific_humidity=atm['specific_humidity'],
            cloud_water=cloud_water,
            cloud_ice=cloud_ice,
            cloud_fraction=cloud_fraction,
            layer_thickness=atm['layer_thickness'],
            air_density=atm['air_density'],
            height=atm['height'],
            radiative_heating=radiative_heating,
            total_heating=total_heating,
            net_energy_flux=0.0
        )
    
    def compute_radiation(self, state: RCEState) -> Tuple[jnp.ndarray, float]:
        """Compute radiative heating using selected radiation scheme.
        
        Args:
            state: Current RCE state
            
        Returns:
            Tuple of (radiative_heating, net_surface_flux)
        """

        tendencies, diagnostics = self.radiation_fn(
            temperature=state.temperature,
            specific_humidity=state.specific_humidity,
            pressure_levels=state.pressure,
            layer_thickness=state.layer_thickness,
            air_density=state.air_density,
            cloud_water=state.cloud_water,
            cloud_ice=state.cloud_ice,
            cloud_fraction=state.cloud_fraction,
            day_of_year=self.setup.day_of_year,
            seconds_since_midnight=self.setup.seconds_since_midnight,
            latitude=self.setup.latitude,
            longitude=self.setup.longitude,
            parameters=self.radiation_params,
            aerosol_data=self.aerosol_data
        )
        
        # Extract heating rate and surface flux
        radiative_heating = tendencies.temperature_tendency
        net_surface_flux = diagnostics.surface_sw_down + diagnostics.surface_lw_down - diagnostics.surface_lw_up
        
        return radiative_heating, net_surface_flux
    
    def compute_convective_adjustment(self, state: RCEState) -> jnp.ndarray:
        """Compute convective adjustment to maintain stable lapse rate.
        
        Implements standard Manabe-Strickler convective adjustment that responds
        to actual atmospheric instability and maintains stable temperature profiles.
        
        Args:
            state: Current RCE state
            
        Returns:
            Convective heating rate [K/s]
        """
        # Constants
        dry_adiabatic_lapse_rate = 9.8  # K/km
        moist_adiabatic_lapse_rate = MOIST_ADIABATIC_LAPSE_RATE  # K/km
        
        # Calculate current lapse rate
        dT_dz = jnp.gradient(state.temperature) / jnp.gradient(state.height)
        current_lapse_rate = -dT_dz * 1000  # Convert to K/km
        
        # Initialize convective heating
        convective_heating = jnp.zeros_like(state.temperature)

        # Convective adjustment is applied when the lapse rate is greater than the dry + moist adiabatic lapse rate
        unstable_mask = current_lapse_rate > moist_adiabatic_lapse_rate
        
        if jnp.any(unstable_mask):
            # Target lapse rate: use moist adiabatic for adjustment
            target_lapse_rate = jnp.where(unstable_mask, moist_adiabatic_lapse_rate, current_lapse_rate)
            target_dT_dz = -target_lapse_rate / 1000  # Convert K/km to K/m
            
            # Calculate temperature adjustment needed
            temp_adjustment = (target_dT_dz - dT_dz) * state.layer_thickness
            
            # Convert to heating rate
            convective_heating = jnp.where(unstable_mask, temp_adjustment / self.setup.dt, 0.0)
        
        return convective_heating
    
    def compute_rhs(self, state: RCEState) -> jnp.ndarray:
        """Compute right-hand side of temperature equation.
        
        Implements radiative heating with optional convective adjustment.
        
        Args:
            state: Current RCE state
            
        Returns:
            Temperature tendency [K/s]
        """
        # Compute radiative heating
        radiative_heating, _ = self.compute_radiation(state)
        
        # Add convective adjustment if enabled
        if self.setup.convective_adjustment:
            convective_heating = self.compute_convective_adjustment(state)
            return radiative_heating + convective_heating
        
        # Pure radiative equilibrium (no convection like swirl-jatmos)
        return radiative_heating
    
    def step_rk2(self, state: RCEState, dt: float) -> RCEState:
        """Take one RK2 timestep.
        
        Implements second-order Runge-Kutta method with humidity adjustment
        following swirl-jatmos approach.
        
        Args:
            state: Current RCE state
            dt: Time step [s]
            
        Returns:
            Updated RCE state after one timestep
        """
        # Stage 1: Compute tendency at current state
        rhs1 = self.compute_rhs(state)
        
        # Update temperature for midpoint
        temp_mid = state.temperature + 0.5 * dt * rhs1
        
        # Update humidity at midpoint using saturation adjustment keeping fixed relative humidity
        q_mid = compute_saturation_humidity(temp_mid, state.pressure)
        
        # Create midpoint state
        state_mid = RCEState(
            temperature=temp_mid,
            pressure=state.pressure,
            specific_humidity=q_mid,
            cloud_water=state.cloud_water,
            cloud_ice=state.cloud_ice,
            cloud_fraction=state.cloud_fraction,
            layer_thickness=state.layer_thickness,
            air_density=state.air_density,
            height=state.height,
            radiative_heating=state.radiative_heating,
            total_heating=state.total_heating,
            net_energy_flux=state.net_energy_flux
        )
        
        # Stage 2: Compute tendency at midpoint state
        rhs2 = self.compute_rhs(state_mid)
        
        # Update temperature using RK2
        temp_new = state.temperature + dt * rhs2
        
        # Enforce surface boundary condition: keep bottom level at fixed temperature
        # This creates a true RCE setup with fixed surface temperature
        temp_new = temp_new.at[-1].set(SURFACE_TEMPERATURE)
        
        # Update humidity using saturation adjustment
        q_new = compute_saturation_humidity(temp_new, state.pressure)
        
        # Update heating rates for diagnostics
        radiative_heating, net_surface_flux = self.compute_radiation(state_mid)
        total_heating = radiative_heating
        
        # Create new state
        return RCEState(
            temperature=temp_new,
            pressure=state.pressure,
            specific_humidity=q_new,
            cloud_water=state.cloud_water,
            cloud_ice=state.cloud_ice,
            cloud_fraction=state.cloud_fraction,
            layer_thickness=state.layer_thickness,
            air_density=state.air_density,
            height=state.height,
            radiative_heating=radiative_heating,
            total_heating=total_heating,
            net_energy_flux=net_surface_flux
        )
    
    def run_simulation(self) -> Tuple[RCEState, dict]:
        """Run RCE simulation to convergence.
        
        Integrates the RCE equations using RK2 timestepping until convergence
        or maximum steps reached.
        
        Returns:
            Tuple of (final_state, history_dict)
        """
        # Initialize state
        state = self.create_initial_state()
        
        # Initialize history tracking
        history = {
            'step': [],
            'time': [],
            't_avg': [],
            't_surface': [],
            'q_avg': [],
            'rad_heating_max': [],
            'net_flux': [],
            'temperature_profiles': [],  # Store full temperature profiles
            'pressure_profiles': []      # Store pressure profiles
        }
        
        # Simulation loop
        for step in range(self.setup.max_steps):
            # Track history
            current_time = step * self.setup.dt / 3600.0  # hours
            t_avg = jnp.mean(state.temperature)
            t_atm_bottom = state.temperature[-2]
            q_avg = jnp.mean(state.specific_humidity)
            rad_heating_max = jnp.max(jnp.abs(state.radiative_heating))
            net_flux = state.net_energy_flux
            
            history['step'].append(step)
            history['time'].append(current_time)
            history['t_avg'].append(t_avg)
            history['t_surface'].append(t_atm_bottom)
            history['q_avg'].append(q_avg)
            history['rad_heating_max'].append(rad_heating_max)
            history['net_flux'].append(net_flux)
            history['temperature_profiles'].append(state.temperature.copy())  # Store full profile
            history['pressure_profiles'].append(state.pressure.copy())        # Store pressure profile
            
            # Print progress
            print(f"Step {step:4d}: T_avg={t_avg:.3f}K, T_surf={t_atm_bottom:.3f}K, "
                    f"q_avg={q_avg*1000:.2f}g/kg, rad_max={rad_heating_max*86400:.2f}K/day, "
                    f"flux={net_flux:.1f}W/m²")
            
            # Check convergence
            if step > 0:
                t_change = abs(t_avg - history['t_avg'][-2])
                if t_change < self.setup.convergence_threshold:
                    print(f"\nConverged at step {step} (T change = {t_change:.6f} K)")
                    print(f"Final state: T_avg={t_avg:.3f}K, T_surf={t_atm_bottom:.3f}K")
                    break
            
            # Take timestep
            state = self.step_rk2(state, self.setup.dt)
        
        else:
            print(f"\nReached maximum steps ({self.setup.max_steps}) without convergence")
            print(f"Final T change: {abs(t_avg - history['t_avg'][-2]):.6f} K")
        
        return state, history

def compare_results(state_re, state_rce, history_re, history_rce):
    """Compare radiative equilibrium vs radiative-convective equilibrium results.
    
    Args:
        state_re: Final state from radiative equilibrium
        state_rce: Final state from radiative-convective equilibrium
        history_re: History from radiative equilibrium
        history_rce: History from radiative-convective equilibrium
    """
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    plt.plot(state_re.temperature[:-1], state_re.pressure[:-1], 'b-', linewidth=2, label='Radiative Equilibrium')
    plt.plot(state_rce.temperature[:-1], state_rce.pressure[:-1], 'r-', linewidth=2, label=f'Radiative-Convective Equilibrium ({MOIST_ADIABATIC_LAPSE_RATE} K/km adjustment)')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Pressure [Pa]')
    plt.title('Temperature Profiles Comparison - RRTMGP Radiation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    plt.gca().invert_yaxis()
    plt.show()

def run_rce_test():
    """Run the RCE test with both radiative and radiative-convective equilibrium.
    
    Creates and runs two RCE simulations:
    1. Pure radiative equilibrium (no convection)
    2. Radiative-convective equilibrium (with convective adjustment)
    Then analyzes and plots the results.
    """
    # Run 1: Pure Radiative Equilibrium
    setup_re = RCESetup(radiation_scheme="rrtmgp", max_steps=15, convective_adjustment=False)
    solver_re = RCESolver(setup_re)
    final_state_re, history_re = solver_re.run_simulation()
    
    # Run 2: Radiative-Convective Equilibrium
    setup_rce = RCESetup(radiation_scheme="rrtmgp", max_steps=15, convective_adjustment=True)
    solver_rce = RCESolver(setup_rce)
    final_state_rce, history_rce = solver_rce.run_simulation()
    
    # Analyze and compare results
    compare_results(final_state_re, final_state_rce, history_re, history_rce)
    
    return final_state_re, final_state_rce, history_re, history_rce


if __name__ == "__main__":
    # Run the RCE test (both RE and RCE)
    final_state_re, final_state_rce, history_re, history_rce = run_rce_test()
