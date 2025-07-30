"""Leapfrog time integration filters for JAX-GCM.

This module contains horizontal diffusion filters that are compatible with 
leapfrog time stepping, moved from the Dinosaur time_integration.py module.
"""

from typing import Callable, Dict, Union

from dinosaur import filtering
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur.time_integration import leapfrog_step_filter
import jax
import jax.numpy as jnp

# Type aliases
PyTreeState = typing.PyTreeState
PyTreeTermsFn = typing.PyTreeTermsFn
PyTreeStepFilterFn = typing.PyTreeStepFilterFn
Array = typing.Array


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
    orders: Dict[str, Union[int, Array]] = None,
) -> Callable:
  """Returns a JIT-compatible horizontal diffusion step filter with field and level-specific timescales and orders.
  
  Args:
    grid: the `spherical_harmonic.Grid` to use for the computation.
    dt: size of the time step.
    timescales: dictionary mapping field names to diffusion timescales.
                Supported field names:
                - 'vorticity': timescale(s) for vorticity diffusion
                - 'divergence': timescale(s) for divergence diffusion  
                - 'temperature_variation': timescale(s) for temperature diffusion
                - 'log_surface_pressure': timescale for surface pressure diffusion
                - 'tracers': timescale(s) for tracer diffusion (applied to all tracers)
                
                Values can be:
                - float: same timescale for all levels
                - Array[levels]: different timescale per level

    orders: dictionary mapping field names to diffusion orders (default: all 4).
           Values can be:
           - int: same order for all levels  
           - Array[levels]: different order per level
    
  Returns:
    A JIT-compatible function that accepts a state dict and returns a filtered state dict.
    
  Example:
    # SPEEDY-style level-specific diffusion
    levels = 8
    strat_timescale = 12.0 * 3600  # 12 hours  
    trop_timescale = 2.4 * 3600    # 2.4 hours
    
    level_timescales = jnp.array([strat_timescale, strat_timescale] + 
                                 [trop_timescale] * (levels - 2))
    level_orders = jnp.array([1, 1] + [4] * (levels - 2))  # del^2 vs del^8
    
    timescales = {
        'vorticity': level_timescales,
        'divergence': level_timescales,
        'temperature_variation': level_timescales,
        'tracers': level_timescales,
    }
    orders = {
        'vorticity': level_orders,
        'divergence': level_orders, 
        'temperature_variation': level_orders,
        'tracers': level_orders,
    }
    
    filter_fn = multi_timescale_horizontal_diffusion_step_filter(
        grid, dt, timescales, orders)
        
    # The returned filter is JIT-compatible
  """
  
  # Default orders to 4 (del^8 diffusion) if not specified
  if orders is None:
    orders = {}
  
  # Pre-compute diffusion coefficients using SPEEDY-style approach
  _, total_wavenumber = grid.modal_axes
  trunc = grid.total_wavenumbers - 1
  rlap = 1.0 / float(trunc * (trunc + 1))
  
  def compute_speedy_style_coeffs(tau_val, order_val):
    """Compute diffusion coefficients - from SPEEDY horizontal_diffusion.f90."""
    twn = total_wavenumber
    mask = twn > 0  # Only apply to non-zero modes
    
    # Vectorized computation for all wavenumbers
    elap = twn * (twn + 1) * rlap
    elapn = elap ** order_val
    
    # SPEEDY's diffusion coefficient (explicit)
    hdiff_equiv = 1.0 / tau_val
    dmp_val = hdiff_equiv * elapn
    
    # SPEEDY's implicit coefficient  
    dmp1_val = 1.0 / (1.0 + dt * dmp_val)
    
    # Multiplicative filter equivalent to SPEEDY's step:
    # new_field = old_field * (1 - dt * dmp * dmp1)
    filter_coeff = 1.0 - dt * dmp_val * dmp1_val
    
    # Apply only to non-zero modes, keep zero mode unchanged
    return jnp.where(mask, filter_coeff, 1.0)
  
  # Pre-compute coefficients for all configured fields
  field_coeffs = {}
  
  for field_name, tau in timescales.items():
    field_order = orders.get(field_name, 4)  # Default to del^8
    
    # Convert to JAX arrays
    tau = jnp.asarray(tau)
    field_order = jnp.asarray(field_order)
    
    if tau.ndim == 0 and field_order.ndim == 0:
      # Scalar timescale and order - uniform across levels
      field_coeffs[field_name] = compute_speedy_style_coeffs(tau, field_order)
    elif tau.ndim == 1 and field_order.ndim == 0:
      # Level-specific timescales, uniform order
      coeffs_per_level = []
      for level_tau in tau:
        coeffs_per_level.append(compute_speedy_style_coeffs(level_tau, field_order))
      field_coeffs[field_name] = jnp.stack(coeffs_per_level)
    elif tau.ndim == 0 and field_order.ndim == 1:
      # Uniform timescale, level-specific orders
      coeffs_per_level = []
      for level_order in field_order:
        coeffs_per_level.append(compute_speedy_style_coeffs(tau, level_order))
      field_coeffs[field_name] = jnp.stack(coeffs_per_level)
    elif tau.ndim == 1 and field_order.ndim == 1:
      # Both level-specific
      coeffs_per_level = []
      for level_tau, level_order in zip(tau, field_order):
        coeffs_per_level.append(compute_speedy_style_coeffs(level_tau, level_order))
      field_coeffs[field_name] = jnp.stack(coeffs_per_level)
    else:
      raise ValueError(f"Unsupported timescale/order configuration for {field_name}")
  
  # Define the fields that will be processed
  configured_fields = set(field_coeffs.keys())
  
  @jax.jit
  def apply_diffusion_to_field(field_value, coeffs):
    """Apply diffusion coefficients to a field (JIT-compatible)."""
    if field_value.ndim == 2:
      # 2D field (lon_modes, lat_modes): broadcast coeffs
      return field_value * coeffs
    elif field_value.ndim == 3:  
      # 3D field (levels, lon_modes, lat_modes)
      if coeffs.ndim == 1:
        # Uniform coefficients: broadcast to all levels
        return field_value * coeffs[None, None, :]
      else:
        # Level-specific coefficients: broadcast properly
        return field_value * coeffs[:, None, :]
    else:
      # Return unchanged for other dimensions
      return field_value
  
  def filter_fn(state):
    """Apply diffusion to state fields (JIT-compatible, handles State objects and dicts)."""
    
    # Convert state to dictionary if needed
    if hasattr(state, 'asdict'):
      # Dinosaur State object
      state_dict = state.asdict()
      use_state_object = True
    elif hasattr(state, '_asdict'):
      # NamedTuple-like object
      state_dict = state._asdict()
      use_state_object = True
    else:
      # Already a dictionary
      state_dict = state
      use_state_object = False
    
    # Apply diffusion using JIT-compiled helper
    filtered_dict = _apply_diffusion_jit(state_dict)
    
    # Return same type as input
    if use_state_object:
      if hasattr(state, 'replace'):
        return state.replace(**filtered_dict)
      elif hasattr(state, '_replace'):
        return state._replace(**filtered_dict)
      else:
        # Fallback - return dictionary
        return filtered_dict
    else:
      return filtered_dict
  
  @jax.jit
  def _apply_diffusion_jit(state_dict):
    """JIT-compiled diffusion application (internal helper)."""
    filtered_state = {}
    
    # Process each field in the state dictionary
    for field_name, field_value in state_dict.items():
      if field_name in configured_fields:
        coeffs = field_coeffs[field_name]
        
        if field_name == 'tracers' and isinstance(field_value, dict):
          # Apply diffusion to all tracers using the tracer coefficients
          filtered_tracers = {}
          for tracer_name, tracer_value in field_value.items():
            filtered_tracers[tracer_name] = apply_diffusion_to_field(tracer_value, coeffs)
          filtered_state[field_name] = filtered_tracers
        else:
          # Apply diffusion to regular array field
          filtered_state[field_name] = apply_diffusion_to_field(field_value, coeffs)
      else:
        # Copy field unchanged if not configured for diffusion
        filtered_state[field_name] = field_value
    
    return filtered_state
  
  return filter_fn