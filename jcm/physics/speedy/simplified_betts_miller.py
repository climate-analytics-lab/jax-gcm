"""
Simplified Betts-Miller convection scheme following Frierson (2007).
https://doi.org/10.1175/JAS3935.1

This implements the qref-formulation for the Simplified Betts-Miller scheme,
adapted from SpeedyWeather.jl for use in jax-gcm.
"""

import jax
from jax import jit, lax
import jax.numpy as jnp
from jcm.geometry import Geometry
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics_interface import PhysicsTendency, PhysicsState
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.physical_constants import (
    p0, alhc, grav, cp, rgas, R_vapour, mol_ratio
)
from jcm.physics.speedy.humidity import get_qsat


@jit
def compute_virtual_temperature(temp: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute virtual temperature from temperature and specific humidity.
    
    Args:
        temp: Temperature [K]
        q: Specific humidity [g/kg]
    
    Returns:
        Virtual temperature [K]
    """
    mu = (1.0 - mol_ratio) / mol_ratio  # ≈ 0.608
    q_kgkg = q / 1000.0  # Convert g/kg to kg/kg
    return temp * (1.0 + mu * q_kgkg)


@jit 
def pseudo_adiabat_step(carry, level_idx):
    """
    Single step of pseudo-adiabat calculation, going from level k+1 to k (upward).
    
    This is designed to be used with jax.lax.scan, iterating from surface upward.
    
    Args:
        carry: Tuple of (temp_parcel, humid_parcel, temp_virt_parcel, saturated, buoyant, temp_ref)
        level_idx: Current level index (0 = top, kx-1 = surface)
    
    Returns:
        Updated carry and output for this level
    """
    (temp_parcel, humid_parcel, temp_virt_parcel, saturated, buoyant,
     sigma, sigma_prev, pres, geopot_k, geopot_prev, temp_virt_env_k, R_cp, Lv, cₚ) = carry
    
    # Compute dry adiabatic temperature
    temp_parcel_dry = temp_parcel * (sigma / sigma_prev) ** R_cp
    
    # Check saturation at this level  
    qsat_dry = get_qsat(temp_parcel_dry, pres, sigma)
    would_saturate = humid_parcel >= qsat_dry
    now_saturated = saturated | would_saturate
    
    # Moist adiabatic calculation (only used if saturated)
    mu = (1.0 - mol_ratio) / mol_ratio
    q_kgkg = humid_parcel / 1000.0  # Convert to kg/kg for thermodynamic calculations
    
    # Moist adiabatic lapse rate parameters (Eq from SpeedyWeather)
    # dT/dΦ = -Γ/cp where Γ is the moist adiabatic lapse rate factor
    A = q_kgkg * Lv / ((1.0 - q_kgkg) ** 2 * rgas)
    B = q_kgkg * Lv ** 2 / ((1.0 - q_kgkg) ** 2 * cₚ * R_vapour)
    Gamma = (1.0 + A / temp_virt_parcel) / (1.0 + B / temp_parcel ** 2)
    
    delta_geopot = geopot_k - geopot_prev
    temp_parcel_moist = temp_parcel - delta_geopot / cₚ * Gamma
    
    # Get saturation humidity at new moist temperature
    qsat_moist = get_qsat(temp_parcel_moist, pres, sigma)
    
    # Select dry or moist path based on saturation
    new_temp_parcel = jnp.where(now_saturated, temp_parcel_moist, temp_parcel_dry)
    new_humid_parcel = jnp.where(now_saturated, qsat_moist, humid_parcel)
    
    # Virtual temperature of parcel
    new_temp_virt_parcel = compute_virtual_temperature(new_temp_parcel, new_humid_parcel)
    
    # Check buoyancy (parcel warmer than environment)
    still_buoyant = buoyant & (new_temp_virt_parcel > temp_virt_env_k)
    
    # Only update if still buoyant
    final_temp = jnp.where(still_buoyant, new_temp_parcel, jnp.nan)
    
    return (new_temp_parcel, new_humid_parcel, new_temp_virt_parcel, 
            now_saturated, still_buoyant), (final_temp, still_buoyant)


@jit
def compute_pseudo_adiabat(
    temp_parcel_init: jnp.ndarray,
    humid_parcel_init: jnp.ndarray,
    temp_virt_env: jnp.ndarray,
    geopot: jnp.ndarray,
    psa: jnp.ndarray,
    sigma_full: jnp.ndarray,
    sigma_half: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the pseudo-adiabatic temperature profile and level of zero buoyancy.
    
    The parcel starts at the surface with given temperature and humidity,
    follows the dry adiabat until saturation, then follows the moist pseudo-adiabat.
    The calculation stops at the level of zero buoyancy (LZB).
    
    Args:
        temp_parcel_init: Initial parcel temperature at surface [K], shape (ix, il)
        humid_parcel_init: Initial parcel specific humidity [g/kg], shape (ix, il)
        temp_virt_env: Virtual temperature of environment [K], shape (kx, ix, il)
        geopot: Geopotential [m²/s²], shape (kx, ix, il)
        psa: Normalized surface pressure, shape (ix, il)
        sigma_full: Full sigma levels, shape (kx,)
        sigma_half: Half sigma levels, shape (kx+1,)
    
    Returns:
        temp_ref_profile: Reference temperature profile [K], shape (kx, ix, il)
        level_zero_buoyancy: Index of LZB for each column, shape (ix, il)
    """
    kx, ix, il = temp_virt_env.shape
    
    # Thermodynamic constants
    R_cp = rgas / cp
    Lv = alhc * 1000.0  # Convert J/g to J/kg
    
    # Initialize reference profile with NaN (levels above LZB will stay NaN)
    temp_ref_profile = jnp.full((kx, ix, il), jnp.nan)
    
    # Set surface level
    temp_ref_profile = temp_ref_profile.at[kx-1].set(temp_parcel_init)
    
    # Initialize parcel state at surface
    temp_parcel = temp_parcel_init
    humid_parcel = humid_parcel_init
    temp_virt_parcel = compute_virtual_temperature(temp_parcel, humid_parcel)
    saturated = jnp.zeros((ix, il), dtype=bool)
    buoyant = jnp.ones((ix, il), dtype=bool)
    
    # Track level of zero buoyancy (initialize to surface)
    level_zero_buoyancy = jnp.full((ix, il), kx, dtype=jnp.int32)
    
    # Iterate from surface upward (k = kx-2 down to k = 0)
    def body_fn(k_from_top, state):
        """Process one level, going from surface upward."""
        temp_ref_profile, temp_parcel, humid_parcel, temp_virt_parcel, saturated, buoyant, lzb = state
        
        # k is the level we're computing (0=top, kx-1=surface)
        # k_from_surface goes from 1 to kx-1
        k = kx - 1 - k_from_top
        k_below = k + 1  # Level below (closer to surface)
        
        sigma_k = sigma_full[k]
        sigma_below = sigma_full[k_below]
        geopot_k = geopot[k]
        geopot_below = geopot[k_below]
        temp_virt_env_k = temp_virt_env[k]
        
        # Dry adiabatic ascent
        temp_parcel_dry = temp_parcel * (sigma_k / sigma_below) ** R_cp
        
        # Check saturation
        qsat_dry = get_qsat(temp_parcel_dry, psa, sigma_k)
        would_saturate = humid_parcel >= qsat_dry
        now_saturated = saturated | would_saturate
        
        # Moist adiabatic calculation
        mu = (1.0 - mol_ratio) / mol_ratio
        q_kgkg = humid_parcel / 1000.0
        
        A = q_kgkg * Lv / ((1.0 - q_kgkg) ** 2 * rgas)
        B = q_kgkg * Lv ** 2 / ((1.0 - q_kgkg) ** 2 * cp * R_vapour)
        Gamma = (1.0 + A / temp_virt_parcel) / (1.0 + B / temp_parcel ** 2)
        
        delta_geopot = geopot_k - geopot_below
        temp_parcel_moist = temp_parcel - delta_geopot / cp * Gamma
        
        # Saturation humidity at moist temperature
        qsat_moist = get_qsat(temp_parcel_moist, psa, sigma_k)
        
        # Select based on saturation state
        new_temp_parcel = jnp.where(now_saturated, temp_parcel_moist, temp_parcel_dry)
        new_humid_parcel = jnp.where(now_saturated, qsat_moist, humid_parcel)
        
        # Virtual temperature
        new_temp_virt_parcel = compute_virtual_temperature(new_temp_parcel, new_humid_parcel)
        
        # Buoyancy check
        still_buoyant = buoyant & (new_temp_virt_parcel > temp_virt_env_k)
        
        # Update reference profile only where still buoyant
        new_temp_ref = jnp.where(still_buoyant, new_temp_parcel, jnp.nan)
        temp_ref_profile = temp_ref_profile.at[k].set(new_temp_ref)
        
        # Update LZB: when buoyancy is first lost, record this level + 1
        # (LZB is the first level with buoyancy, so we add 1 when buoyancy is lost)
        lost_buoyancy = buoyant & ~still_buoyant
        new_lzb = jnp.where(lost_buoyancy, k + 1, lzb)
        
        return (temp_ref_profile, new_temp_parcel, new_humid_parcel, 
                new_temp_virt_parcel, now_saturated, still_buoyant, new_lzb)
    
    # Run the loop from surface upward
    init_state = (temp_ref_profile, temp_parcel, humid_parcel, 
                  temp_virt_parcel, saturated, buoyant, level_zero_buoyancy)
    
    final_state = lax.fori_loop(1, kx, body_fn, init_state)
    temp_ref_profile, _, _, _, _, final_buoyant, level_zero_buoyancy = final_state
    
    # If still buoyant at top, LZB is 1 (top level)
    level_zero_buoyancy = jnp.where(final_buoyant, 1, level_zero_buoyancy)
    
    return temp_ref_profile, level_zero_buoyancy


@jit
def compute_humid_ref_profile(
    temp_ref_profile: jnp.ndarray,
    psa: jnp.ndarray,
    sigma_full: jnp.ndarray,
    ref_rel_humidity: float
) -> jnp.ndarray:
    """
    Compute reference humidity profile from reference temperature.
    
    Args:
        temp_ref_profile: Reference temperature profile [K], shape (kx, ix, il)
        psa: Normalized surface pressure, shape (ix, il)
        sigma_full: Full sigma levels, shape (kx,)
        ref_rel_humidity: Reference relative humidity [0-1]
    
    Returns:
        humid_ref_profile: Reference humidity profile [g/kg], shape (kx, ix, il)
    """
    kx = sigma_full.shape[0]
    
    # Vectorized computation of qsat at each level
    def compute_qsat_at_level(k, temp_ref, psa, sigma):
        return get_qsat(temp_ref[k], psa, sigma[k])
    
    # Use vmap to compute qsat at all levels
    qsat_ref = jax.vmap(
        lambda k: get_qsat(temp_ref_profile[k], psa, sigma_full[k]),
        out_axes=0
    )(jnp.arange(kx))
    
    return ref_rel_humidity * qsat_ref


@jit
def get_sbm_convection_tendencies(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData = None,
    geometry: Geometry = None
) -> tuple[PhysicsTendency, PhysicsData]:
    """
    Compute convective tendencies using the Simplified Betts-Miller scheme.
    
    This scheme relaxes temperature and humidity profiles toward reference
    profiles computed from a pseudo-adiabatic ascent. The scheme distinguishes
    between deep convection (precipitating) and shallow convection (non-precipitating).
    
    Reference: Frierson (2007), https://doi.org/10.1175/JAS3935.1
    
    Args:
        state: Physics state containing temperature, humidity, etc.
        physics_data: Additional physics data including saturation humidity
        parameters: Model parameters including SBM-specific parameters
        forcing: External forcing data (unused)
        geometry: Grid geometry information
    
    Returns:
        physics_tendencies: Temperature and humidity tendencies
        physics_data: Updated physics data with convection diagnostics
    """
    kx, ix, il = state.temperature.shape
    
    # Extract fields
    temp = state.temperature  # [K]
    qa = state.specific_humidity  # [g/kg]
    geopot = state.geopotential  # [m²/s²]
    psa = state.normalized_surface_pressure
    
    # Get SBM parameters
    sbm = parameters.convection
    time_scale_sec = sbm.time_scale * 3600.0  # Convert hours to seconds
    ref_rel_humidity = sbm.relative_humidity
    
    # Compute sigma level information
    sigma_full = geometry.fsg  # (kx,)
    
    # Compute half levels and layer thickness from full levels
    sigma_half = jnp.concatenate([
        jnp.array([0.0]),
        0.5 * (sigma_full[:-1] + sigma_full[1:]),
        jnp.array([1.0])
    ])
    delta_sigma = jnp.diff(sigma_half)  # Layer thickness (kx,)
    
    # Virtual temperature of environment
    temp_virt_env = compute_virtual_temperature(temp, qa)
    
    # Surface parcel properties (from lowest layer)
    temp_parcel_init = temp[kx-1]
    humid_parcel_init = qa[kx-1]
    
    # Compute pseudo-adiabatic reference profile and level of zero buoyancy
    temp_ref_profile, level_zero_buoyancy = compute_pseudo_adiabat(
        temp_parcel_init, humid_parcel_init, temp_virt_env, geopot, 
        psa, sigma_full, sigma_half
    )
    
    # Compute reference humidity profile: q_ref = RH_ref * q_sat(T_ref)
    humid_ref_profile = compute_humid_ref_profile(
        temp_ref_profile, psa, sigma_full, ref_rel_humidity
    )
    
    # Create mask for levels at or below LZB (where convection acts)
    level_indices = jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis]
    conv_mask = level_indices >= level_zero_buoyancy[jnp.newaxis, :, :]
    
    # Replace NaN with current values for masked calculations
    temp_ref_safe = jnp.where(jnp.isnan(temp_ref_profile), temp, temp_ref_profile)
    humid_ref_safe = jnp.where(jnp.isnan(humid_ref_profile), qa, humid_ref_profile)
    
    # Compute Pq (precipitation from drying) and PT (precipitation from cooling)
    # Pq = Σ (q - q_ref) * Δσ  (positive when environment is moister than reference)
    # PT = -Σ (T - T_ref) * Δσ  (positive when environment is warmer than reference)
    delta_q = (qa - humid_ref_safe) * conv_mask
    delta_T = (temp - temp_ref_safe) * conv_mask
    
    Pq = jnp.sum(delta_q * delta_sigma[:, jnp.newaxis, jnp.newaxis], axis=0)
    PT = -jnp.sum(delta_T * delta_sigma[:, jnp.newaxis, jnp.newaxis], axis=0)
    
    # Determine convection type
    deep_convection = (Pq > 0) & (PT > 0)
    shallow_convection = (Pq <= 0) & (PT > 0)
    
    # Height of zero buoyancy level in sigma coordinates
    # sigma_half at LZB index gives the sigma value at top of convection
    sigma_half_expanded = jnp.broadcast_to(
        sigma_half[:-1, jnp.newaxis, jnp.newaxis], 
        (kx, ix, il)
    )
    sigma_lzb = jnp.take_along_axis(
        sigma_half_expanded,
        level_zero_buoyancy[jnp.newaxis, :, :],
        axis=0
    ).squeeze(0)
    delta_sigma_lzb = 1.0 - sigma_lzb  # σ_surface(=1) - σ_LZB
    delta_sigma_lzb = jnp.maximum(delta_sigma_lzb, 1e-6)  # Avoid division by zero
    
    # Latent heat ratio: Lv/cp in units of K per (g/kg)
    # alhc is in J/g, cp is in J/(kg·K)
    # Lv/cp = (J/g) / (J/(kg·K)) = K·kg/g = K·1000/(g/kg) 
    # So for q in g/kg: (Lv/cp) * q gives temperature in K
    Lv_cp = alhc / cp * 1000.0  # K per (g/kg)
    
    # === DEEP CONVECTION ADJUSTMENT ===
    # ΔT = (PT - Pq * Lv/cp) / Δσ_lzb  (Eq. 5 in Frierson 2007)
    delta_T_deep = (PT - Pq * Lv_cp) / delta_sigma_lzb
    
    # === SHALLOW CONVECTION ADJUSTMENT ===
    # Qref = -Σ q_ref * Δσ  (Eq. 11)
    # fq = 1 - Pq/Qref  (Eq. 12, note Pq is negative for shallow)
    # ΔT = PT / Δσ_lzb  (Eq. 14)
    Qref = -jnp.sum(humid_ref_safe * delta_sigma[:, jnp.newaxis, jnp.newaxis] * conv_mask, axis=0)
    Qref = jnp.where(jnp.abs(Qref) < 1e-10, -1e-10, Qref)  # Avoid division by zero
    fq = 1.0 - Pq / Qref
    fq = jnp.clip(fq, 0.0, 2.0)  # Reasonable bounds
    delta_T_shallow = PT / delta_sigma_lzb
    
    # Apply adjustments to reference profiles
    temp_ref_adjusted = temp_ref_safe
    humid_ref_adjusted = humid_ref_safe
    
    # Deep convection: T_ref -= ΔT (Eq. 6)
    temp_ref_adjusted = jnp.where(
        conv_mask & deep_convection[jnp.newaxis, :, :],
        temp_ref_adjusted - delta_T_deep[jnp.newaxis, :, :],
        temp_ref_adjusted
    )
    
    # Shallow convection: T_ref -= ΔT (Eq. 15), q_ref *= fq (Eq. 13)
    temp_ref_adjusted = jnp.where(
        conv_mask & shallow_convection[jnp.newaxis, :, :],
        temp_ref_adjusted - delta_T_shallow[jnp.newaxis, :, :],
        temp_ref_adjusted
    )
    humid_ref_adjusted = jnp.where(
        conv_mask & shallow_convection[jnp.newaxis, :, :],
        humid_ref_adjusted * fq[jnp.newaxis, :, :],
        humid_ref_adjusted
    )
    
    # Compute tendencies: relax toward reference profiles
    # dT/dt = -(T - T_ref) / τ
    # dq/dt = -(q - q_ref) / τ
    any_convection = deep_convection | shallow_convection
    
    ttend = jnp.where(
        conv_mask & any_convection[jnp.newaxis, :, :],
        -(temp - temp_ref_adjusted) / time_scale_sec,
        0.0
    )
    
    qtend = jnp.where(
        conv_mask & any_convection[jnp.newaxis, :, :],
        -(qa - humid_ref_adjusted) / time_scale_sec,
        0.0
    )
    
    # Compute convective precipitation (only for deep convection)
    # Integrate moisture tendency vertically: precip = -∫ dq/dt dp/g
    # In sigma coordinates: dp = pₛ dσ, so precip = -pₛ/g ∫ dq/dt dσ
    moisture_removal = jnp.maximum(-qtend, 0.0)  # Only count moisture removal
    precnv = jnp.where(
        deep_convection,
        jnp.sum(moisture_removal * delta_sigma[:, jnp.newaxis, jnp.newaxis], axis=0),
        0.0
    )
    # Convert to mass flux [g/(m² s)]
    # qtend is in g/kg/s, delta_sigma is dimensionless
    # psa * p0 gives pressure in Pa = kg/(m·s²)
    # Dividing by g gives kg/m²
    # Result: (g/kg/s) * (kg/m²) = g/(m² s) ... need to check units carefully
    precnv = precnv * psa * p0 / grav  # [g/(m² s)]
    
    # Update physics data with convection diagnostics
    convection_out = physics_data.convection.copy(
        iptop=level_zero_buoyancy,
        cbmf=jnp.zeros((ix, il)),  # Cloud base mass flux (not used in SBM)
        precnv=precnv
    )
    physics_data = physics_data.copy(convection=convection_out)
    
    # Create tendency output
    physics_tendencies = PhysicsTendency.zeros(
        shape=state.temperature.shape,
        temperature=ttend,
        specific_humidity=qtend
    )
    
    return physics_tendencies, physics_data


# Alternative entry point matching existing jax-gcm interface
#get_convection_tendencies = get_sbm_convection_tendencies
