"""Implementations of `OpticsScheme`s and a factory method."""

from collections.abc import Mapping
from typing import Callable, Literal, TypeAlias, cast
from typing_extensions import override

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from jcm.rrtmgp.config import radiative_transfer
from jcm.rrtmgp.optics import cloud_optics
from jcm.rrtmgp.optics import constants
from jcm.rrtmgp.optics import gas_optics
from jcm.rrtmgp.optics import lookup_cloud_optics
from jcm.rrtmgp.optics import lookup_gas_optics_base
from jcm.rrtmgp.optics import lookup_gas_optics_longwave
from jcm.rrtmgp.optics import lookup_gas_optics_shortwave
from jcm.rrtmgp.optics import lookup_volume_mixing_ratio
from jcm.rrtmgp.optics import optics_base

Array: TypeAlias = jax.Array
AbstractLookupGasOptics: TypeAlias = (lookup_gas_optics_base.AbstractLookupGasOptics)
LookupCloudOptics: TypeAlias = lookup_cloud_optics.LookupCloudOptics
LookupGasOpticsLongwave: TypeAlias = (lookup_gas_optics_longwave.LookupGasOpticsLongwave)
LookupGasOpticsShortwave: TypeAlias = (lookup_gas_optics_shortwave.LookupGasOpticsShortwave)
LookupVolumeMixingRatio: TypeAlias = (lookup_volume_mixing_ratio.LookupVolumeMixingRatio)

_EPSILON = 1e-6

def centered_difference(f: Array, dim: Literal[0, 1, 2]) -> Array:
	"""Compute the centered difference of f in the given dimension."""
	return jnp.roll(f, -1, axis=dim) - jnp.roll(f, 1, axis=dim)
	

class RRTMOptics(optics_base.OpticsScheme):
	"""The Rapid Radiative Transfer Model (RRTM) optics scheme implementation."""

	def __init__(
		self,
		vmr_lib: LookupVolumeMixingRatio,
		params: radiative_transfer.OpticsParameters,
	):
	    super().__init__()
	    assert isinstance(params.optics, radiative_transfer.RRTMOptics)
	    rrtm_params = params.optics
	    self.vmr_lib = vmr_lib
	    self.cloud_optics_lw = lookup_cloud_optics.from_nc_file(
	        rrtm_params.cloud_longwave_nc_filepath
	    )
	    self.cloud_optics_sw = lookup_cloud_optics.from_nc_file(
	        rrtm_params.cloud_shortwave_nc_filepath
	    )
	    self.gas_optics_lw = lookup_gas_optics_longwave.from_nc_file(
	        rrtm_params.longwave_nc_filepath
	    )
	    self.gas_optics_sw = lookup_gas_optics_shortwave.from_nc_file(
	        rrtm_params.shortwave_nc_filepath
	    )
	    # Type narrowing
	    assert self.gas_optics_lw is not None
	    assert self.gas_optics_sw is not None

	def _od_fn(
		self,
		is_lw: bool,
		igpt: Array,
		molecules: Array,
		temperature: Array,
		pressure: Array,
		vmr_fields: dict[int, Array] | None,
	) -> Array:
	    """The actual optical_depth calculation."""
	    logging.info('Calling optical depth graph.')
	    lookup_gas_optics = self.gas_optics_lw if is_lw else self.gas_optics_sw
	    return gas_optics.compute_minor_optical_depth(
	        lookup_gas_optics,
	        self.vmr_lib,
	        molecules,
	        temperature,
	        pressure,
	        igpt,
	        vmr_fields,
	    ) + gas_optics.compute_major_optical_depth(
	        lookup_gas_optics,
	        self.vmr_lib,
	        molecules,
	        temperature,
	        pressure,
	        igpt,
	        vmr_fields,
	    )

	def _optical_depth_fn(
		self, igpt: Array, is_lw: bool
	) -> Callable[[Array, Array, Array, dict[int, Array] | None], Array]:
		"""Create a function for computing the optical depth.
		
		Args:
			igpt: The g-point that will be used to index into the RRTMGP lookup table.
			is_lw: If `True`, uses the longwave lookup. Otherwise, uses the shortwave lookup.
		
		Returns:
			A callable that computes the optical depth given fields for the number of
			molecules per area, pressure, and volume mixing ratio of various gases.
		"""
		
		def od_fn(
			molecules: Array,
			temperature: Array,
			pressure: Array,
			vmr_fields: dict[int, Array] | None,
		) -> Array:
			return self._od_fn(
				is_lw, igpt, molecules, temperature, pressure, vmr_fields
			)
		
		return od_fn

	def _rayl_fn(
		self,
		igpt: Array,
		molecules: Array,
		temperature: Array,
		pressure: Array,
		vmr_fields: dict[int, Array] | None,
	) -> Array:
		"""The actual Rayleigh scattering calculation."""
		logging.info('Calling Rayleigh scattering graph.')
		return gas_optics.compute_rayleigh_optical_depth(
			self.gas_optics_sw,
			self.vmr_lib,
			molecules,
			temperature,
			pressure,
			igpt,
			vmr_fields,
		)

	def rayleigh_scattering_fn(
		self, igpt: Array
	) -> Callable[[Array, Array, Array, dict[int, Array] | None], Array]:
		"""Create a function for computing Rayleigh scattering.
		
		Args:
			igpt: The g-point that will be used to index into the RRTMGP lookup table.
		
		Returns:
			A callable that computes the Rayleigh scattering optical depth given
			fields for the number of molecules per area, temperature, pressure, and
			and volume mixing ratio of various gases.
		"""

		def rayl_fn(
			molecules: Array,
			temperature: Array,
			pressure: Array,
			vmr_fields: dict[int, Array] | None,
		) -> Array:
			return self._rayl_fn(igpt, molecules, temperature, pressure, vmr_fields)
			
		return rayl_fn

	def _pf_fn(
		self,
		igpt: Array,
		pressure: Array,
		temperature: Array,
		vmr_fields: dict[int, Array] | None = None,
	) -> Array:
		"""The actual Planck fraction calculation."""
		logging.info('Calling Planck fraction graph.')
		return gas_optics.compute_planck_fraction(
			self.gas_optics_lw,
			self.vmr_lib,
			pressure,
			temperature,
			igpt,
			vmr_fields,
		)

	def planck_fraction_fn(
		self,
		igpt: Array,
	):
		"""Create a function for computing the Planck fraction.
		
		Args:
			igpt: The g-point that will be used to index into the RRTMGP lookup table.
		
		Returns:
			A callable that computes the Planck fraction given fields for pressure,
			temperature and volume mixing ratio of various gases.
		"""

		def pf_fn(
			pressure: Array,
			temperature: Array,
			vmr_fields: dict[int, Array] | None = None,
		) -> Array:
			return self._pf_fn(igpt, pressure, temperature, vmr_fields)
			
		return pf_fn

	def _ps_fn(
		self,
		igpt: Array,
		planck_fraction: Array,
		temperature: Array,
	) -> Array:
		"""The actual Planck source calculation."""
		logging.info('Calling Planck source graph.')
		return gas_optics.compute_planck_sources(
			self.gas_optics_lw, planck_fraction, temperature, igpt
		)

	def planck_src_fn(
	  self,
	  igpt: Array,
	):
		"""Create a function for computing the Planck source.
		
		Args:
			igpt: The g-point that will be used to index into the RRTMGP lookup table.
		
		Returns:
			A callable that computes the Planck source given fields for the
			precomputed pointwise Planck fraction and temperature.
		"""

		def ps_fn(planck_fraction: Array, temperature: Array) -> Array:
			return self._ps_fn(igpt, planck_fraction, temperature)
			
		return ps_fn

	def _cloud_props(
		self, ibnd, is_lw, r_eff_liq, cloud_path_liq, r_eff_ice, cloud_path_ice
	) -> dict[str, Array]:
		"""The actual cloud optical properties calculation."""
		logging.info('Calling cloud optical properties graph.')
		cloud_lookup = self.cloud_optics_lw if is_lw else self.cloud_optics_sw
		return cloud_optics.compute_optical_properties(
			cloud_lookup,
			cloud_path_liq,
			cloud_path_ice,
			r_eff_liq,
			r_eff_ice,
			ibnd=ibnd,
		)

	def cloud_properties_fn(
		self,
		ibnd: Array,
		is_lw: bool,
	):
		"""Create a function for computing cloud optical properties.
		
		Args:
		  ibnd: The spectral band index that will be used to index into the lookup
			tables for cloud absorption coefficients.
		  is_lw: If `True`, uses the longwave lookup. Otherwise, uses the shortwave
			lookup.
		
		Returns:
		  A callable that returns a dictionary containing the cloud optical depth,
		  single-scattering albedo, and asymmetry factor.
		"""
	
		def cloud_props_fn(r_eff_liq, cloud_path_liq, r_eff_ice, cloud_path_ice):
			return self._cloud_props(
				ibnd,
				is_lw,
				r_eff_liq,
				cloud_path_liq,
				r_eff_ice,
				cloud_path_ice,
			)
			
		return cloud_props_fn

	def _apply_delta_scaling_for_cloud(
		self,
		cloud_optical_props: Mapping[str, Array],
	) -> dict[str, Array]:
		"""Delta-scales optical properties for shortwave bands."""
		optical_depth = cloud_optical_props['optical_depth']
		ssa = cloud_optical_props['ssa']
		g = cloud_optical_props['asymmetry_factor']
		
		# Apply delta scaling
		wf = ssa * g**2
		cloud_tau = (1 - wf) * optical_depth
		cloud_ssa = (ssa - wf) / jnp.maximum(1 - wf, _EPSILON)
		cloud_asy = (g - g**2) / jnp.maximum(1 - g**2, _EPSILON)
		
		return {
			'optical_depth': cloud_tau,
			'ssa': cloud_ssa,
			'asymmetry_factor': cloud_asy,
		}

	def _combine_gas_and_cloud_properties(
		self,
		igpt: Array,
		optical_props: Mapping[str, Array],
		is_lw: bool,
		radius_eff_liq: Array | None = None,
		cloud_path_liq: Array | None = None,
		radius_eff_ice: Array | None = None,
		cloud_path_ice: Array | None = None,
	) -> dict[str, Array]:
		"""Combine the gas optical properties with the cloud optical properties."""
		gas_lookup = self.gas_optics_lw if is_lw else self.gas_optics_sw
		assert gas_lookup is not None  # Type narrowing.
		
		# If any of the input cloud states are `None`, replace them with zeros.
		if radius_eff_liq is None:
			radius_eff_liq = jnp.zeros_like(optical_props['ssa'])
		if cloud_path_liq is None:
			cloud_path_liq = jnp.zeros_like(optical_props['ssa'])
		if radius_eff_ice is None:
			radius_eff_ice = jnp.zeros_like(optical_props['ssa'])
		if cloud_path_ice is None:
			cloud_path_ice = jnp.zeros_like(optical_props['ssa'])
		
		ibnd = gas_lookup.g_point_to_bnd[igpt]
		
		compute_cloud_properties_fn = self.cloud_properties_fn(ibnd, is_lw)
		cloud_optical_props = compute_cloud_properties_fn(
			radius_eff_liq, cloud_path_liq, radius_eff_ice, cloud_path_ice
		)
		
		if not is_lw:
			cloud_optical_props = self._apply_delta_scaling_for_cloud(cloud_optical_props)
		return self.combine_optical_properties(optical_props, cloud_optical_props)

	@override
	def compute_lw_optical_properties(
		self,
		pressure: Array,
		temperature: Array,
		molecules: Array,
		igpt: Array,
		vmr_fields: dict[int, Array] | None = None,
		cloud_r_eff_liq: Array | None = None,
		cloud_path_liq: Array | None = None,
		cloud_r_eff_ice: Array | None = None,
		cloud_path_ice: Array | None = None,
	) -> dict[str, Array]:
		"""Compute the monochromatic longwave optical properties.
		
		Uses the RRTM optics scheme to compute the longwave optical depth, albedo,
		and asymmetry factor. These raw optical properties can be further
		transformed downstream to better suit the assumptions of the particular
		radiative transfer solver being used.
		
		Args:
			pressure: The pressure field [Pa].
			temperature: The temperature [K].
			molecules: The number of molecules in an atmospheric grid cell per area [molecules / m^2].
			igpt: The spectral interval index, or g-point.
			vmr_fields: An optional dictionary containing precomputed volume mixing
				ratio fields, keyed by gas index, that will overwrite the global means.
			cloud_r_eff_liq: The effective radius of cloud droplets [m].
			cloud_path_liq: The cloud liquid water path in each atmospheric grid cell [kg/m²].
			cloud_r_eff_ice: The effective radius of cloud ice particles [m].
			cloud_path_ice: The cloud ice water path in each atmospheric grid cell [kg/m²].
		
		Returns:
			A dictionary containing (for a single g-point):
				'optical_depth': The longwave optical depth.
				'ssa': The longwave single-scattering albedo.
				'asymmetry_factor': The longwave asymmetry factor.
		"""
		optical_depth_fn = self._optical_depth_fn(igpt, is_lw=True)
		optical_depth_lw = optical_depth_fn(
			molecules, temperature, pressure, vmr_fields
		)
		optical_props = {
			'optical_depth': optical_depth_lw,
			'ssa': jnp.zeros_like(optical_depth_lw),
			'asymmetry_factor': jnp.zeros_like(optical_depth_lw),
		}
		
		if cloud_path_liq is not None or cloud_path_ice is not None:
			return self._combine_gas_and_cloud_properties(
				igpt,
				optical_props,
				is_lw=True,
				radius_eff_liq=cloud_r_eff_liq,
				cloud_path_liq=cloud_path_liq,
				radius_eff_ice=cloud_r_eff_ice,
				cloud_path_ice=cloud_path_ice
			)
		return optical_props

	@override
	def compute_sw_optical_properties(
		self,
		pressure: Array,
		temperature: Array,
		molecules: Array,
		igpt: Array,
		vmr_fields: dict[int, Array] | None = None,
		cloud_r_eff_liq: Array | None = None,
		cloud_path_liq: Array | None = None,
		cloud_r_eff_ice: Array | None = None,
		cloud_path_ice: Array | None = None,
	) -> dict[str, Array]:
		"""Compute the monochromatic shortwave optical properties.
		
		Uses the RRTM optics scheme to compute the shortwave optical depth, albedo,
		and asymmetry factor. These raw optical properties can be further
		transformed downstream to better suit the assumptions of the particular
		radiative transfer solver being used.
		
		Args:
			pressure: The pressure field [Pa].
			temperature: The temperature [K].
			molecules: The number of molecules in an atmospheric grid cell per area [molecules / m^2].
			igpt: The spectral interval index, or g-point.
			vmr_fields: An optional dictionary containing precomputed volume mixing
				ratio fields, keyed by gas index, that will overwrite the global means.
			cloud_r_eff_liq: The effective radius of cloud droplets [m].
			cloud_path_liq: The cloud liquid water path in each atmospheric grid cell [kg/m²].
			cloud_r_eff_ice: The effective radius of cloud ice particles [m].
			cloud_path_ice: The cloud ice water path in each atmospheric grid cell [kg/m²].
		
		Returns:
			A dictionary containing (for a single g-point):
				'optical_depth': The shortwave optical depth.
				'ssa': The shortwave single-scattering albedo.
				'asymmetry_factor': The shortwave asymmetry factor.
		"""
		optical_depth_fn = self._optical_depth_fn(igpt, is_lw=False)
		optical_depth_sw = optical_depth_fn(
			molecules, temperature, pressure, vmr_fields
		)
		
		rayl_fn = self.rayleigh_scattering_fn(igpt)
		rayleigh_scattering = rayl_fn(molecules, temperature, pressure, vmr_fields)
		
		optical_depth_sw = optical_depth_sw + rayleigh_scattering
		
		ssa = jnp.where(
			optical_depth_sw > 0,
			rayleigh_scattering / optical_depth_sw,
			jnp.zeros_like(rayleigh_scattering),
		)
		
		optical_props = {
			'optical_depth': optical_depth_sw,
			'ssa': ssa,
			'asymmetry_factor': jnp.zeros_like(ssa),
		}
		if cloud_path_liq is not None or cloud_path_ice is not None:
			return self._combine_gas_and_cloud_properties(
				igpt,
				optical_props,
				is_lw=False,
				radius_eff_liq=cloud_r_eff_liq,
				cloud_path_liq=cloud_path_liq,
				radius_eff_ice=cloud_r_eff_ice,
				cloud_path_ice=cloud_path_ice
			)
		return optical_props

	@override
	def compute_planck_sources(
		self,
		pressure: Array,
		temperature: Array,
		igpt: Array,
		vmr_fields: dict[int, Array] | None = None,
		sfc_temperature: Array | None = None,
	) -> dict[str, Array]:
		"""Compute the monochromatic Planck sources given the atmospheric state.
		
		This requires interpolating the temperature to z faces.
		
		Args:
			pressure: The pressure field [Pa].
			temperature: The temperature [K].
			igpt: The spectral interval index, or g-point.
			vmr_fields: An optional dictionary containing precomputed volume mixing
				ratio fields, keyed by gas index, that will overwrite the global means.
			sfc_temperature: The surface temperature [K], 2D field.
		
		Returns:
			A dictionary containing the Planck source at the cell center
			(`planck_src`), the top cell boundary (`planck_src_top`), the bottom cell
			boundary (`planck_src_bottom`) and, if a `sfc_temperature` argument was
			provided, the surface cell boundary (`planck_src_sfc`). Note that the
			surface source will only be valid for the replicas in the first
			computational layer, as the local temperature field is used to compute it.
		"""
		assert sfc_temperature is not None, 'sfc_temperature is required.'
		
		temperature_bottom, temperature_top = optics_base.reconstruct_face_values(
			temperature, f_lower_bc=sfc_temperature
		)
		planck_fraction_fn = self.planck_fraction_fn(igpt)
		planck_fraction = planck_fraction_fn(pressure, temperature, vmr_fields)
		
		planck_src_fn = self.planck_src_fn(igpt)
		planck_src = planck_src_fn(planck_fraction, temperature)
		planck_src_top = planck_src_fn(planck_fraction, temperature_top)
		planck_src_bottom = planck_src_fn(planck_fraction, temperature_bottom)
		
		planck_srcs = {
			'planck_src': planck_src,
			'planck_src_top': planck_src_top,
			'planck_src_bottom': planck_src_bottom,
		}
		
		if sfc_temperature is not None:
			# Extract the first interior node for surface calculations.
			planck_fraction_0 = planck_fraction[:, :, 1]
			
			planck_srcs['planck_src_sfc'] = planck_src_fn(
				planck_fraction_0, sfc_temperature
			)
		
		return planck_srcs

	@override
	@property
	def n_gpt_lw(self) -> int:
		"""The number of g-points in the longwave bands."""
		self.gas_optics_lw = cast(LookupGasOpticsLongwave, self.gas_optics_lw)
		return self.gas_optics_lw.n_gpt

	@override
	@property
	def n_gpt_sw(self) -> int:
		"""The number of g-points in the shortwave bands."""
		self.gas_optics_sw = cast(LookupGasOpticsShortwave, self.gas_optics_sw)
		return self.gas_optics_sw.n_gpt

	@override
	@property
	def solar_fraction_by_gpt(self) -> Array:
		"""Mapping from g-point to the fraction of total solar radiation."""
		self.gas_optics_sw = cast(LookupGasOpticsShortwave, self.gas_optics_sw)
		return self.gas_optics_sw.solar_src_scaled


class GrayAtmosphereOptics(optics_base.OpticsScheme):
	"""Implementation of the gray atmosphere optics scheme."""
	
	def __init__(
		self,
		params: radiative_transfer.OpticsParameters,
	):
		super().__init__()
		optics = params.optics
		assert isinstance(optics, radiative_transfer.GrayAtmosphereOptics)
		self._p0 = optics.p0
		self._alpha = optics.alpha
		self._d0_lw = optics.d0_lw
		self._d0_sw = optics.d0_sw

	@override
	def compute_lw_optical_properties(
		self,
		pressure: Array,
		*args,
		**kwargs,
	) -> dict[str, Array]:
		"""Compute longwave optical properties based on pressure and lapse rate.
		
		See Schneider 2004, J. Atmos. Sci. (2004) 61 (12): 1317–1340.
		DOI: https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2
		To obtain the local optical depth of the layer, the expression for
		cumulative optical depth (from the top of the atmosphere to an arbitrary
		pressure level) was differentiated with respect to the pressure and
		multiplied by the pressure difference across the grid cell.
		
		Args:
			pressure: The pressure field [Pa].
			*args: Miscellaneous inherited arguments.
			**kwargs: Miscellaneous inherited keyword arguments.
		
		Returns:
			A dictionary containing the optical depth (`optical_depth`), the single-
			scattering albedo (`ssa`), and the asymmetry factor (`asymmetry_factor`)
			for longwave radiation.
		"""
		# Compute the centered pressure difference in z: (p_{k+1} - p_{k-1}) / 2.
		dp = 0.5 * centered_difference(pressure, dim=2)
		# Compute the pointwise optical depth as a function of pressure only.
		alpha, d0_lw, p0 = self._alpha, self._d0_lw, self._p0
		tau = jnp.abs(alpha * d0_lw * (pressure / p0) ** alpha / pressure * dp)
		
		return {
			'optical_depth': tau,
			'ssa': jnp.zeros_like(pressure),
			'asymmetry_factor': jnp.zeros_like(pressure),
		}

	@override
	def compute_sw_optical_properties(
		self,
		pressure: Array,
		*args,
		**kwargs,
	) -> dict[str, Array]:
		"""Compute the shortwave optical properties of a gray atmosphere.
		
		See O'Gorman 2008, Journal of Climate Vol 21, Page(s): 3815–3832.
		DOI: https://doi.org/10.1175/2007JCLI2065.1. In particular, the cumulative
		optical depth expression shown in equation 3 inside the exponential is
		differentiated with respect to pressure and scaled by the pressure
		difference across the grid cell.
		
		Args:
			pressure: The pressure field [Pa].
			*args: Miscellaneous inherited arguments.
			**kwargs: Miscellaneous inherited keyword arguments.
		
		Returns:
			A dictionary containing the optical depth (`optical_depth`), the single-
			scattering albedo (`ssa`), and the asymmetry factor (`asymmetry_factor`)
			for shortwave radiation.
		"""
		# Compute the centered pressure difference in z:
		#   dp_{i,j,k} = (p_{i,j,k+1} - p_{i,j,k-1}) / 2.
		dp = 0.5 * centered_difference(pressure, dim=2)
		
		# Compute the pointwise optical depth as a function of pressure only.
		d0_sw, p0 = self._d0_sw, self._p0
		tau = jnp.abs(2 * d0_sw * (pressure / p0) * (dp / p0))
		
		return {
			'optical_depth': tau,
			'ssa': jnp.zeros_like(pressure),
			'asymmetry_factor': jnp.zeros_like(pressure),
		}

	@override
	def compute_planck_sources(
		self,
		pressure: Array,
		temperature: Array,
		*args,
		sfc_temperature: Array | None = None,
	) -> dict[str, Array]:
		"""Compute the Planck sources used in the longwave problem.
		
		The computation is based on Stefan-Boltzmann's law, which states that the
		thermal radiation emitted from a blackbody is directly proportional to the
		4-th power of its absolute temperature.
		
		Args:
			pressure: The pressure field [Pa].
			temperature: The temperature [K].
			*args: Miscellaneous inherited arguments.
			sfc_temperature: The optional surface temperature [K], 2D field.
		
		Returns:
			A dictionary containing the Planck source at the cell center
			(`planck_src`), the top cell boundary (`planck_src_top`), and the bottom
			cell boundary (`planck_src_bottom`).
		"""
		del pressure
		assert sfc_temperature is not None, 'sfc_temperature is required.'
		
		def src_fn(t: Array) -> Array:
			return constants.STEFAN_BOLTZMANN * t**4 / np.pi
		
		# Interpolate temperature from (ccc) to (ccf), and also provide a shifted
		# copy.
		temperature_bottom, temperature_top = optics_base.reconstruct_face_values(
			temperature, f_lower_bc=sfc_temperature
		)
		
		planck_srcs = {
			'planck_src': src_fn(temperature),
			'planck_src_top': src_fn(temperature_top),
			'planck_src_bottom': src_fn(temperature_bottom),
		}
		if sfc_temperature is not None:
			planck_srcs['planck_src_sfc'] = src_fn(sfc_temperature)
		return planck_srcs

	@property
	@override
	def n_gpt_lw(self) -> int:
		"""The number of g-points in the longwave bands."""
		return 1

	@property
	@override
	def n_gpt_sw(self) -> int:
		"""The number of g-points in the shortwave bands."""
		return 1

	@override
	@property
	def solar_fraction_by_gpt(self) -> Array:
		"""Mapping from g-point to the fraction of total solar radiation."""
		return jnp.array([1.0], dtype=jnp.float32)


def optics_factory(
    params: radiative_transfer.OpticsParameters,
    vmr_lib: LookupVolumeMixingRatio | None = None,
) -> optics_base.OpticsScheme:
	"""Construct an instance of `OpticsScheme`.
	
	Args:
		params: The optics parameters.
		vmr_lib: An instance of `LookupVolumeMixingRatio` containing gas
		  concentrations.
	
	Returns:
		An instance of `OpticsScheme`.
	"""
	if isinstance(params.optics, radiative_transfer.RRTMOptics):
		assert vmr_lib is not None, '`vmr_lib` is required for `RRTMOptics`.'
		return RRTMOptics(vmr_lib, params)
	elif isinstance(params.optics, radiative_transfer.GrayAtmosphereOptics):
		return GrayAtmosphereOptics(params)
	else:
		raise ValueError('Unsupported optics scheme.')
