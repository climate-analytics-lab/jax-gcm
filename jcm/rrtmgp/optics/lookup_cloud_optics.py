"""Dataclass for loading and accessing cloud optics."""

import enum
import dataclasses
from collections.abc import Mapping
from typing import TypeAlias

import jax
from jcm.rrtmgp.optics import dataloader_base

Array: TypeAlias = jax.Array

class IceRoughness(enum.Enum):
	"""Defines the levels of ice roughness that index the ice lookup tables."""
	# No roughness.
	SMOOTH = 0
	# Medium roughness.
	MEDIUM = 1
	# Rough.
	ROUGH = 2


@dataclasses.dataclass(frozen=True)
class LookupCloudOptics:
	"""Dataclass for lookup table of cloud optical properties."""
	# Number of liquid particle sizes.
	n_size_liq: int
	# Number of ice particle sizes.
	n_size_ice: int
	# Liquid particle size lower bound for interpolation.
	radius_liq_lower: float
	# Liquid particle size upper bound for interpolation.
	radius_liq_upper: float
	# Ice particle size lower bound for interpolation.
	diameter_ice_lower: float
	# Ice particle size upper bound for interpolation.
	diameter_ice_upper: float
	
	# Lookup table for liquid extinction coefficient (`nbnd, nsize_liq`) in
	# m²/g.
	ext_liq: Array
	# Lookup table for liquid single-scattering albedo (`nbnd, nsize_liq`).
	ssa_liq: Array
	# Lookup table for liquid asymmetry parameter (`nbnd, nsize_liq`).
	asy_liq: Array
	
	# Lookup table for ice extinction coefficient (`nrghice, nband, nsize_ice`)
	# in m²/g.
	ext_ice: Array
	# Lookup table for ice single-scattering albedo
	# (`nrghice, nband, nsize_ice`).
	ssa_ice: Array
	# Lookup table for ice asymmetry parameter (`nrghice, nband, nsize_ice`).
	asy_ice: Array
	# Ice roughness.
	ice_roughness: IceRoughness = IceRoughness.MEDIUM


def _load_data(
    tables: Mapping[str, Array],
    dims: Mapping[str, int],
) -> dict[str, Array | int]:
	"""Preprocess the cloud optics data.
	
	Args:
	tables: The extracted lookup tables and other parameters as a dictionary.
	dims: A dictionary containing dimension information for the tables.
	
	Returns:
	A dictionary containing dimension information and the preprocessed RRTMGP
	data as Arrays or ints.
	"""
	data = {}
	data['n_size_liq'] = dims['nsize_liq']
	data['n_size_ice'] = dims['nsize_ice']
	data['radius_liq_lower'] = tables['radliq_lwr']
	data['radius_liq_upper'] = tables['radliq_upr']
	data['diameter_ice_lower'] = tables['radice_lwr']
	data['diameter_ice_upper'] = tables['radice_upr']
	data['ext_liq'] = tables['lut_extliq']
	data['ssa_liq'] = tables['lut_ssaliq']
	data['asy_liq'] = tables['lut_asyliq']
	data['ext_ice'] = tables['lut_extice']
	data['ssa_ice'] = tables['lut_ssaice']
	data['asy_ice'] = tables['lut_asyice']
	return data


def from_nc_file(path: str) -> LookupCloudOptics:
	"""Instantiate a `LookupCloudOptics` object from a netCDF file.
	
	The netCDF file should contain the lookup tables for the extinction
	coefficients, the single-scattering albedo, and the asymmetry factor for
	liquid and ice particles.
	
	Args:
	path: The full path of the netCDF file containing the cloud optics lookup
	  tables.
	
	Returns:
	A `LookupCloudOptics` instance.
	"""
	_, tables, dims = dataloader_base.parse_nc_file(path)
	kwargs = _load_data(tables, dims)
	return LookupCloudOptics(**kwargs)
