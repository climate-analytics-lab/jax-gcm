"""Leapfrog time integration filters for JAX-GCM.

This module contains horizontal diffusion filters that are compatible with 
leapfrog time stepping, moved from the Dinosaur time_integration.py module.
"""

from typing import Callable, Dict, Union

try:
    from dinosaur import filtering
    from dinosaur import spherical_harmonic
    from dinosaur import typing
    from dinosaur.time_integration import leapfrog_step_filter
except ImportError:
    # Fallback imports if dinosaur is not available
    # Define minimal leapfrog_step_filter if dinosaur is not available
    def leapfrog_step_filter(state_filter):
        """Convert a state filter into a leapfrog time integration filter."""
        def _filter(u, u_next):
            del u  # unused
            current, future = u_next  # leapfrog state is a tuple of 2 time slices.
            future = state_filter(future)
            return (current, future)
        return _filter

import jax
import jax.numpy as jnp
import numpy as np

# Type aliases (these may need to be adjusted based on actual JAX-GCM types)
PyTreeState = typing.PyTreeState if 'typing' in globals() else jax.Array
PyTreeTermsFn = typing.PyTreeTermsFn if 'typing' in globals() else Callable
PyTreeStepFilterFn = typing.PyTreeStepFilterFn if 'typing' in globals() else Callable
Array = typing.Array if 'typing' in globals() else jax.Array


def horizontal_diffusion_leapfrog_step_filter(
    grid: 'spherical_harmonic.Grid',
    dt: float,
    tau: float,
    order: int = 1,
):
  """Returns a horizontal diffusion step filter.

  This filter simulates dampening on modes according to:

    (∂u_k / ∂t) ≈ -(u_k / 𝜏) * (((k * (k + 1)) / (L * (L + 1))) ** order)

  Where L is the maximum total wavenumber. For more details see
  `filtering.horizontal_diffusion_filter`.

  Args:
    grid: the `spherical_harmonic.Grid` to use for the computation.
    dt: size of the time step to be used for each filter application.
    tau: timescale over which the top mode decreases by a factor of `e ** (-1)`.
    order: controls the polynomial order of the exponential filter.

  Returns:
    A function that accepts a state and returns a filtered state.
  """
  eigenvalues = grid.laplacian_eigenvalues
  scale = dt / (tau * abs(eigenvalues[-1]) ** order)
  filter_fn = filtering.horizontal_diffusion_filter(grid, scale, order)
  return leapfrog_step_filter(filter_fn)


def multi_timescale_horizontal_diffusion_step_filter(
    grid: 'spherical_harmonic.Grid',
    dt: float,
    timescales: Dict[str, Union[float, Array]],
    order: Dict[str, Union[int, Array]] = None,
) -> PyTreeStepFilterFn:
  """Returns a horizontal diffusion step filter with field and level-specific timescales and orders.
  
  This filter applies horizontal diffusion with different timescales and diffusion orders
  for different fields and vertical levels in the dinosaur State object, while remaining 
  JAX-differentiable.
  
  Args:
    grid: the `spherical_harmonic.Grid` to use for the computation.
    dt: size of the time step.
    timescales: dictionary mapping field names to diffusion timescales.
                Field names should match those in the State object:
                - 'vorticity': timescale(s) for vorticity diffusion
                - 'divergence': timescale(s) for divergence diffusion  
                - 'temperature_variation': timescale(s) for temperature diffusion
                - 'log_surface_pressure': timescale for surface pressure diffusion
                - 'tracers': timescale(s) for tracer diffusion (applied to all tracers)
                
                Values can be:
                - float: same timescale for all levels
                - Array[levels]: different timescale per level
    order: dictionary mapping field names to diffusion orders (default: all 1).
           Values can be:
           - int: same order for all levels  
           - Array[levels]: different order per level
    
  Returns:
    A function that accepts a state and returns a filtered state.
    
  Example:
    # Uniform diffusion
    timescales = {
        'vorticity': 2.4 * 3600,  # 2.4 hours for all levels
        'divergence': 2.4 * 3600,
        'temperature_variation': 2.4 * 3600, 
    }
    
    # Level-specific diffusion (stratospheric vs tropospheric)
    levels = 8
    stratospheric_timescale = 12.0 * 3600  # 12 hours
    tropospheric_timescale = 2.4 * 3600    # 2.4 hours
    
    # Strong diffusion in stratosphere (first 2 levels), weaker in troposphere
    level_timescales = jnp.array([stratospheric_timescale, stratospheric_timescale] + 
                                 [tropospheric_timescale] * (levels - 2))
    level_orders = jnp.array([1, 1] + [4] * (levels - 2))  # del^2 vs del^8
    
    timescales = {
        'vorticity': level_timescales,
        'divergence': level_timescales,
        'temperature_variation': level_timescales,
    }
    orders = {
        'vorticity': level_orders,
        'divergence': level_orders, 
        'temperature_variation': level_orders,
    }
    
    filter_fn = multi_timescale_horizontal_diffusion_step_filter(
        grid, dt, timescales, orders)
  """
  
  # Default orders to 1 if not specified
  if order is None:
    order = {}
  
  # Pre-compute diffusion coefficients using SPEEDY-style approach
  # This avoids numerical issues with very small eigenvalues by using total wavenumbers
  _, total_wavenumber = grid.modal_axes
  trunc = grid.total_wavenumbers - 1
  rlap = 1.0 / float(trunc * (trunc + 1))
  
  diffusion_coeffs = {}
  
  def compute_speedy_style_coeffs(tau_val, order_val):
    """Compute diffusion coefficients using SPEEDY's exact approach."""
    coeffs = jnp.ones_like(total_wavenumber, dtype=float)
    
    # Compute coefficients for each wavenumber
    for j, twn in enumerate(total_wavenumber):
      if twn > 0:  # Skip the zero wavenumber mode
        # Normalized Laplacian: n*(n+1) / L*(L+1) where L is truncation
        elap = twn * (twn + 1) * rlap
        elapn = elap ** order_val
        
        # SPEEDY's diffusion coefficient (explicit)
        hdiff_equiv = 1.0 / tau_val
        dmp_val = hdiff_equiv * elapn
        
        # SPEEDY's implicit coefficient  
        dmp1_val = 1.0 / (1.0 + dt * dmp_val)
        
        # Multiplicative filter equivalent to SPEEDY's step:
        # new_field = old_field * (1 - dt * dmp * dmp1)
        # This is mathematically equivalent to SPEEDY's (tendency - dmp*field) * dmp1 
        # when integrated with dt
        coeff = 1.0 - dt * dmp_val * dmp1_val
        coeffs = coeffs.at[j].set(coeff)
    
    return coeffs
  
  for field_name, tau in timescales.items():
    field_order = order.get(field_name, 1)
    
    # Handle scalar vs array timescales and orders
    tau = jnp.asarray(tau)
    field_order = jnp.asarray(field_order)
    
    if tau.ndim == 0:  # Scalar timescale
      if field_order.ndim == 0:  # Scalar order
        diffusion_coeffs[field_name] = compute_speedy_style_coeffs(tau, field_order)
      else:  # Array order with scalar timescale
        coeffs_per_level = []
        for level_order in field_order:
          coeffs_per_level.append(compute_speedy_style_coeffs(tau, level_order))
        diffusion_coeffs[field_name] = jnp.stack(coeffs_per_level)
    else:  # Array timescale
      if field_order.ndim == 0:  # Scalar order with array timescale
        coeffs_per_level = []
        for level_tau in tau:
          coeffs_per_level.append(compute_speedy_style_coeffs(level_tau, field_order))
        diffusion_coeffs[field_name] = jnp.stack(coeffs_per_level)
      else:  # Both array timescale and array order
        coeffs_per_level = []
        for level_tau, level_order in zip(tau, field_order):
          coeffs_per_level.append(compute_speedy_style_coeffs(level_tau, level_order))
        diffusion_coeffs[field_name] = jnp.stack(coeffs_per_level)
  
  def filter_fn(state):
    """Apply field and level-specific horizontal diffusion to state variables."""
    
    def apply_diffusion(field_name, field_value):
      # Get diffusion coefficient for this field, default to no diffusion
      coeff = diffusion_coeffs.get(field_name)
      
      if coeff is None:
        return field_value
      
      # Handle tracer dictionaries
      if field_name == 'tracers' and isinstance(field_value, dict):
        tracer_coeff = diffusion_coeffs.get('tracers')
        if tracer_coeff is not None:
          return {k: apply_level_diffusion(v, tracer_coeff) for k, v in field_value.items()}
        else:
          return field_value
      
      # Apply diffusion if field is an array with appropriate shape
      if hasattr(field_value, 'shape') and len(field_value.shape) >= 2:
        return apply_level_diffusion(field_value, coeff)
      else:
        # Return unchanged for scalar or incompatible fields
        return field_value
    
    def apply_level_diffusion(field_value, coeff):
      """Apply level-specific diffusion coefficients to a field."""
      if coeff.ndim == 1:  # Same coefficient for all levels (uniform diffusion)
        # coeff shape: (wavenumber_modes,)
        # field_value shape: (levels, lon_modes, lat_modes) or (lon_modes, lat_modes)
        return field_value * coeff
      elif coeff.ndim == 2:  # Different coefficient per level (level-specific diffusion)
        # coeff shape: (levels, wavenumber_modes)
        # field_value shape: (levels, lon_modes, lat_modes)
        # Need to broadcast correctly: coeff[:, None, :] to match field dimensions
        if len(field_value.shape) == 3:  # (levels, lon_modes, lat_modes)
          expanded_coeff = coeff[:, None, :]  # (levels, 1, lat_modes)
          return field_value * expanded_coeff
        else:
          return field_value * coeff
      else:
        return field_value
    
    # Apply diffusion to all fields in the state
    if hasattr(state, '_replace'):
      # NamedTuple-like object
      state_dict = state._asdict()
    elif hasattr(state, 'asdict'):
      # Object with asdict method
      state_dict = state.asdict()
    elif hasattr(state, '__dict__'):
      # Regular object
      state_dict = state.__dict__
    else:
      # Assume it's already a dict
      state_dict = state
    
    # Apply diffusion to each field
    filtered_dict = {}
    for field_name, field_value in state_dict.items():
      filtered_dict[field_name] = apply_diffusion(field_name, field_value)
    
    # Return the same type as input if possible
    if hasattr(state, '_replace'):
      return state._replace(**filtered_dict)
    elif hasattr(state, 'copy') and callable(state.copy):
      try:
        return state.copy(**filtered_dict)
      except:
        pass
    elif hasattr(state, '__class__'):
      try:
        new_state = object.__new__(state.__class__)
        for key, value in filtered_dict.items():
          setattr(new_state, key, value)
        return new_state
      except:
        pass
    
    # Fallback to dict
    return filtered_dict
  
  return filter_fn