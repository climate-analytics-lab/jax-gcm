"""Utility functions for computing the optical properties of clouds."""

import collections
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp

from jcm.rrtmgp.optics import lookup_cloud_optics
from jcm.rrtmgp.optics import optics_utils

Array: TypeAlias = jax.Array
Interpolant: TypeAlias = optics_utils.Interpolant
IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight

_EPSILON = 1e-6
_KG_TO_G_FACTOR = 1e3
_M_TO_MICRONS_FACTOR = 1e6


def _particle_size_interpolant(
    f: Array, lower_bnd: float, upper_bnd: float, n_size: int
) -> dict[str, Callable[..., Interpolant]]:
	"""Creates effective radius interpolant based on desired number of points."""
	interp = optics_utils.create_linear_interpolant(
		f, jnp.linspace(lower_bnd, upper_bnd, n_size)
	)
	return collections.OrderedDict({'r': lambda: interp})


def compute_optical_properties(
    lookup: lookup_cloud_optics.LookupCloudOptics,
    cloud_path_liq: Array,
    cloud_path_ice: Array,
    radius_eff_liq: Array,
    radius_eff_ice: Array,
    ibnd: Array,
) -> dict[str, Array]:
	"""Computes the optical properties of clouds from lookup tables.
	
	The optical depth, single-scattering albedo, and asymmetry factor are computed
	taking into account both the liquid and ice phase. The final optical depth
	is a simple sum of the optical depths of both phases. The final ssa is a
	a weighted sum (weighted by optical depth) of the ssa of both phases. The
	final asymmetry factor is also a weighted sum (weighted by ssa) of the
	asymmetry factors of both phases.
	
	Args:
		lookup: A `LookupCloudOptics` instance containing lookup tables for the
		  extinction coefficient, the single-scattering albedo, and the asymmetry
		  factor of condensates indexed by spectral band and effective radius of
		  cloud particles.
		cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
		  [kg/m²].
		cloud_path_ice: The cloud ice water path in each atmospheric grid cell
		  [kg/m²].
		radius_eff_liq: The effective radius of cloud droplets in each atmospheric
		  grid cell [m].
		radius_eff_ice: The effective radius of cloud ice particles in each
		  atmospheric grid cell [m].
		ibnd: The spectral band index.
	
	Returns:
		A dictionary containing:
		'optical_depth': The combined optical depth contribution from condensates.
		'ssa': The combined single-scattering albedo contribution from condensates.
		'g': The combined asymmetry factor contribution from condensates.
	"""
	# Convert effective radius from meter to microns to conform to the lookup
	# tables. Default to the lower bound to prevent an out-of-range error when
	# interpolating. These default values will later be eliminated by the cloud
	# mask constructed below.
	# Using the name `size` because the lookup tables for RRTMGP use radius for
	# cloud liquid and diameter for cloud ice.
	particle_size = [
		jnp.clip(
			_M_TO_MICRONS_FACTOR * radius_eff_liq,
			lookup.radius_liq_lower,
			lookup.radius_liq_upper
		),
		jnp.clip(
			_M_TO_MICRONS_FACTOR * 2.0 * radius_eff_ice,
			lookup.diameter_ice_lower,
			lookup.diameter_ice_upper
		)
	]
	
	# Create interpolant for cloud droplet effective radius.
	size_lower = [lookup.radius_liq_lower, lookup.diameter_ice_lower]
	size_upper = [lookup.radius_liq_upper, lookup.diameter_ice_upper]
	n_size = [lookup.n_size_liq, lookup.n_size_ice]
	
	interpolants = jax.tree.map(
		_particle_size_interpolant, particle_size, size_lower, size_upper, n_size
	)
	
	roughness = lookup.ice_roughness.value
	ext_tables = (lookup.ext_liq[ibnd, :], lookup.ext_ice[roughness, ibnd, :])
	ssa_tables = (lookup.ssa_liq[ibnd, :], lookup.ssa_ice[roughness, ibnd, :])
	asy_tables = (lookup.asy_liq[ibnd, :], lookup.asy_ice[roughness, ibnd, :])
	
	# Convert cloud path to g/m² to conform to the lookup tables.
	cloud_path = (
		cloud_path_liq * _KG_TO_G_FACTOR,
		cloud_path_ice * _KG_TO_G_FACTOR,
	)
	
	cld_mask_liq = jnp.where(
		cloud_path_liq * _KG_TO_G_FACTOR >= _EPSILON,
		jnp.ones_like(cloud_path_liq),
		jnp.zeros_like(cloud_path_liq),
	)
	
	cld_mask_ice = jnp.where(
		cloud_path_ice * _KG_TO_G_FACTOR >= _EPSILON,
		jnp.ones_like(cloud_path_ice),
		jnp.zeros_like(cloud_path_ice),
	)
	cld_mask = (cld_mask_liq, cld_mask_ice)
	
	optical_props = []
	for interp, ext, ssa, asy, cld_path, mask in zip(
		interpolants, ext_tables, ssa_tables, asy_tables, cloud_path, cld_mask
	):
	    props = {}
	    props['tau'] = optics_utils.interpolate(ext, interp) * cld_path * mask
	    # Weight the ssa by the optical depth.
	    props['tau_ssa'] = optics_utils.interpolate(ssa, interp) * props['tau']
	    # Weight the asymmetry factor by the single-scattering albedo.
	    props['tau_ssa_g'] = optics_utils.interpolate(asy, interp) * props['tau_ssa']
		
	    optical_props.append(props)

	combined_props = jax.tree.map(jnp.add, *optical_props)
	return {
		'optical_depth': combined_props['tau'],
		'ssa': jnp.where(
			combined_props['tau'] != 0,
			combined_props['tau_ssa'] / combined_props['tau'],
			0.0,
		),
		'asymmetry_factor': jnp.where(
			combined_props['tau_ssa'] != 0,
			combined_props['tau_ssa_g'] / combined_props['tau_ssa'],
			0.0,
		),
	}
