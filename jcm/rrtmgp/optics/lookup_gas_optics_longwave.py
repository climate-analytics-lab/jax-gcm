"""A Dataclass for longwave optical properties of gases."""

from collections.abc import Mapping
import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import netCDF4 as nc

from jcm.rrtmgp.optics import dataloader_base
from jcm.rrtmgp.optics import lookup_gas_optics_base

Array: TypeAlias = jax.Array


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsLongwave(lookup_gas_optics_base.AbstractLookupGasOptics):
	"""Lookup tables of gases' optical properties in the longwave bands."""
	
	# Planck fraction `(n_t_ref, n_p_ref, n_η, n_gpt)`.
	planck_fraction: Array
	# Number of reference temperatures, for Planck source calculations.
	n_t_plnk: int
	# reference temperatures for Planck source calculations `(n_t_plnk)`.
	t_planck: Array
	# total Planck source for each band `(n_bnd, n_t_plnk)`.
	totplnk: Array


def _load_data(
    ds: nc.Dataset, tables: Mapping[str, Array], dims: Mapping[str, int]
) -> dict[str, Any]:
	"""Preprocess the RRTMGP longwave gas optics data.
	
	Args:
	ds: The original netCDF Dataset containing the RRTMGP shortwave optics data.
	tables: The extracted data as a dictionary of `Array`s.
	dims: A dictionary containing dimension information for the tables.
	
	Returns:
	A dictionary containing dimension information and the preprocessed RRTMGP
	data as `Array`s.
	"""
	data = lookup_gas_optics_base.load_data(ds, tables, dims)
	data['n_t_plnk'] = dims['temperature_Planck']
	data['planck_fraction'] = tables['plank_fraction']
	# Similarly to the original RRTM Fortran code, here we assume that
	# temperature minimum and maximum are the same for the absorption
	# coefficient grid and the Planck grid and the Planck grid is equally
	# spaced.
	data['t_planck'] = jnp.linspace(
	  data['temperature_ref_min'],
	  data['temperature_ref_max'],
	  data['n_t_plnk'], dtype=jnp.float32
	)
	data['totplnk'] = tables['totplnk']
	return data


def from_nc_file(path: str) -> LookupGasOpticsLongwave:
	"""Instantiate a `LookupGasOpticsLongwave` object from zipped netCDF file.
	
	The compressed file should be netCDF parsable and contain the RRTMGP
	absorprtion coefficient lookup table for the longwave bands as well as all
	the auxiliary reference tables required to index into the lookup table.
	
	Args:
	path: The full path of the zipped netCDF file containing the longwave
	  absorption coefficient lookup table.
	
	Returns:
	A `LookupGasOpticsLongwave` object.
	"""
	ds, tables, dims = dataloader_base.parse_nc_file(path)
	kwargs = _load_data(ds, tables, dims)
	return LookupGasOpticsLongwave(**kwargs)
