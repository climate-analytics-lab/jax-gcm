"""Abstract base class defining the interface of an optics scheme."""

import abc
from collections.abc import Mapping
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

def shift_from_plus(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """output_i = f_{i+1}."""
  return jnp.roll(f, -1, axis=dim)


def shift_from_minus(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """output_i = f_{i-1}."""
  return jnp.roll(f, 1, axis=dim)

	
def _shift_up(f: Array) -> Array:
	"""output_i = f_{i-1}."""
	return shift_from_minus(f, 2)


def _shift_down(f: Array) -> Array:
	"""output_i = f_{i+1}."""
	return shift_from_plus(f, 2)


##### Below this, WENO5 interpolation special for RRTMGP #####
def _weno5_nonlinear_weights(
    f_node: Array, dim: Literal[0, 1, 2], wall_bc: bool
) -> tuple[Array, Array, Array, Array, Array, Array]:
	"""Compute the nonlinear weights for WENO5."""
	# The following quantities are all on nodes.
	f = f_node
	f_iminus3 = jnp.roll(f, 3, axis=dim)
	f_iminus2 = jnp.roll(f, 2, axis=dim)
	f_iminus1 = jnp.roll(f, 1, axis=dim)
	f_iplus1 = jnp.roll(f, -1, axis=dim)
	f_iplus2 = jnp.roll(f, -2, axis=dim)
	
	# Compute beta, which are the smoothness indicators.
	beta1_plus = (
	  13 / 12 * (f_iminus3 - 2 * f_iminus2 + f_iminus1) ** 2
	  + 1 / 4 * (f_iminus3 - 4 * f_iminus2 + 3 * f_iminus1) ** 2
	)
	beta2_plus = (
	  13 / 12 * (f_iminus2 - 2 * f_iminus1 + f) ** 2
	  + 1 / 4 * (f_iminus2 - f) ** 2
	)
	beta3_plus = (
	  13 / 12 * (f_iminus1 - 2 * f + f_iplus1) ** 2
	  + 1 / 4 * (3 * f_iminus1 - 4 * f + f_iplus1) ** 2
	)
	
	beta1_minus = (
	  13 / 12 * (f - 2 * f_iplus1 + f_iplus2) ** 2
	  + 1 / 4 * (3 * f - 4 * f_iplus1 + f_iplus2) ** 2
	)
	beta2_minus = (
	  13 / 12 * (f_iminus1 - 2 * f + f_iplus1) ** 2
	  + 1 / 4 * (f_iminus1 - f_iplus1) ** 2
	)
	beta3_minus = (
	  13 / 12 * (f_iminus2 - 2 * f_iminus1 + f) ** 2
	  + 1 / 4 * (f_iminus2 - 4 * f_iminus1 + 3 * f) ** 2
	)
	
	c1, c2, c3 = 0.1, 0.6, 0.3  # Optimal linear weights for WENO5-JS.
	epsilon = 1e-5
	
	alpha1_plus = c1 / (beta1_plus + epsilon)**2
	alpha2_plus = c2 / (beta2_plus + epsilon)**2
	alpha3_plus = c3 / (beta3_plus + epsilon)**2
	
	alpha1_minus = c1 / (beta1_minus + epsilon)**2
	alpha2_minus = c2 / (beta2_minus + epsilon)**2
	alpha3_minus = c3 / (beta3_minus + epsilon)**2
	
	# Deal with boundaries, if we have boundaries instead of periodic BCs.
	# Here we, ASSUME the values in the halos are usable with legitimate values.
	# Strategy: For the first interior face, we set the unnormalized weight alpha1
	# to 0 because there are not enough points for its stencil.  E.g., note that
	# beta1_plus (and hence alpha1_plus) uses f_iminus3, which is not defined for
	# the first interior face.  For the last interior face we do the same thing --
	# set alpha1_minus to 0.
	# The result is that WENO5 will adapt to using the other stencils.
	# For the second interior face, alpha1_plus uses f_iminus3 which is the halo
	# node.  So if the halo node value is ok, then this should be fine.
	hw = 1  # Assumed halo width of 1.
	if wall_bc:
		alpha1_plus = alpha1_plus.at[:, :, hw + 1].set(0)
		alpha1_minus = alpha1_minus.at[:, :, -hw - 1].set(0)

	# Compute the nonlinear weights.
	w1_plus = alpha1_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
	w2_plus = alpha2_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
	w3_plus = alpha3_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
	
	w1_minus = alpha1_minus / (alpha1_minus + alpha2_minus + alpha3_minus)
	w2_minus = alpha2_minus / (alpha1_minus + alpha2_minus + alpha3_minus)
	w3_minus = alpha3_minus / (alpha1_minus + alpha2_minus + alpha3_minus)
	return w1_plus, w2_plus, w3_plus, w1_minus, w2_minus, w3_minus


def _weno5_local_reconstructions(
    f_node: Array, dim: Literal[0, 1, 2]
) -> tuple[Array, Array, Array, Array, Array, Array]:
	"""Compute the local reconstructions from different stencils for WENO5.
	
	Args:
	f_node: A 3D array, evaluated on nodes.
	dim: The dimension along with the interpolation is performed.
	
	Returns:
	A tuple of six 3D arrays, the local reconstructions of the input nodal array
	on the face i - 1/2, for different stencils.
	"""
	# The following quantities are on nodes.
	f = f_node
	f_iminus3 = jnp.roll(f, 3, axis=dim)
	f_iminus2 = jnp.roll(f, 2, axis=dim)
	f_iminus1 = jnp.roll(f, 1, axis=dim)
	f_iplus1 = jnp.roll(f, -1, axis=dim)
	f_iplus2 = jnp.roll(f, -2, axis=dim)
	
	# Compute the local reconstructions from the various stencils.
	# These are approximations on the face i - 1/2.  We could name the variable
	# with an additiona subscript _face_iminushalf, but that would be verbose.
	f1_plus = 1/3 * f_iminus3 - 7/6 * f_iminus2 + 11/6 * f_iminus1
	f2_plus = -1/6 * f_iminus2 + 5/6 * f_iminus1 + 1/3 * f
	f3_plus = 1/3 * f_iminus1 + 5/6 * f - 1/6 * f_iplus1
	
	f1_minus = 11/6 * f - 7/6 * f_iplus1 + 1/3 * f_iplus2
	f2_minus = 1/3 * f_iminus1 + 5/6 * f - 1/6 * f_iplus1
	f3_minus = -1/6 * f_iminus2 + 5/6 * f_iminus1 + 1/3 * f
	return f1_plus, f2_plus, f3_plus, f1_minus, f2_minus, f3_minus


def weno5_node_to_face_for_rrtmgp(
    f_node: Array,
    dim: Literal[0, 1, 2],
    f_lower_bc: Array | None = None,
    neumann_upper_bc: bool = False,
) -> tuple[Array, Array]:
	"""Perform WENO5-JS interpolation from nodes to faces.
	
	* An array evaluated on nodes has index i <==> coordinate location x_i
	* An array evaluated on faces has index i <==> coordinate location x_{i-1/2}
	
	See also QUICK interpolation in convection.py.
	
	When dealing with the boundaries, for now we are USING the values in the halos
	(halo width = 1) as part of the process, and that the values in the halos are
	set appropriately.  This is not ideal, and is inconsistent with the rest of
	the code (which does not use halos values).  We should get rid of the use of
	halo values later.
	
	Refs: Jiang and Shu, "Efficient Implementation of Weighted ENO Schemes",
	JCP 126, 202-228 (1996).
	
	Args:
		f_node: A 3D array, evaluated on nodes.
		dim: The dimension along with the interpolation is performed.
		f_lower_bc: If not None, this value is used as the boundary condition for f
		  on the lower face (the wall).  This should be a 2D array.
		neumann_upper_bc: If True, then a Neumann BC is used for f on the upper
		  face.
	
	Returns:
		A tuple of two 3D arrays interpolated from `f_node`, which is evaluted on
		faces in dimension `dim`. The first array is the "plus" (left-biased)
		interpolation, and the second array is the "minus" (right-biased)
		interpolation.
	"""
	if f_lower_bc is not None and neumann_upper_bc:
		# We have walls on both faces of the domain.
		wall_bc = True
	else:
		# We don't have walls on both faces; revert to periodic treatment.
		wall_bc = False

	w1_plus, w2_plus, w3_plus, w1_minus, w2_minus, w3_minus = (
		_weno5_nonlinear_weights(f_node, dim, wall_bc)
	)
	# Get various local reconstructions of f on the face i - 1/2.
	f1_plus, f2_plus, f3_plus, f1_minus, f2_minus, f3_minus = (
		_weno5_local_reconstructions(f_node, dim)
	)
	# Obtain the WENO reconstruction by combining the nonlinear weights with the
	# local reconstructions on the faces.
	f_face_plus = w1_plus * f1_plus + w2_plus * f2_plus + w3_plus * f3_plus
	f_face_minus = w1_minus * f1_minus + w2_minus * f2_minus + w3_minus * f3_minus

	# Deal with boundaries.  Here, assume a possible Dirichlet lower BC and a
	# Neumann upper BC.
	# These two `if`s only deal with the face value on the halos, not interior
	# faces.  Note: we really should be assigning the lower BC to the wall-face,
	# not a halo node, but let's fix that up later.
	hw = 1  # Assumed halo width.
	if f_lower_bc is not None:
		# Assign value in the halo ...
		f_face_plus = f_face_plus.at[:, :, 0].set(f_lower_bc)
		f_face_minus = f_face_minus.at[:, :, 0].set(f_lower_bc)

	if neumann_upper_bc:
		f_face_minus = f_face_minus.at[:, :, -hw].set(f_node[:, :, -hw - 1])

	return f_face_plus, f_face_minus


def reconstruct_face_values(f: Array, f_lower_bc: Array) -> tuple[Array, Array]:
	"""Reconstruct the face values using a high-order scheme.
	
	This function performs an interpolation from nodes to faces.
	
	Args:
	f: The cell-center values that will be interpolated.
	f_lower_bc: The boundary condition for f on the lower face (wall).
	
	Returns:
	A tuple with the reconstructed temperature at the bottom and top face,
	respectively.
	"""
	# f is f_ccc.
	# Use WENO5.
	
	# Set halo value to the BC wall value.
	f = f.at[:, :, 0].set(f_lower_bc)
	
	# Enforce Neumann BC on the upper halo value.
	f = f.at[:, :, -1].set(f[:, :, -2])
	
	f_face_plus, f_face_minus = weno5_node_to_face_for_rrtmgp(
	  f, dim=2, f_lower_bc=f_lower_bc, neumann_upper_bc=True
	)
	# To get the final interpolation, just take the average of the plus and
	# minus reconstructions.
	f_bottom_ccf = 0.5 * (f_face_plus + f_face_minus)
	
	# Use centered interpolation: UNSTABLE.
	# f_bottom_ccf = interpolation.z_c_to_f(f)
	
	f_top = _shift_down(f_bottom_ccf)
	# Update the f_top face (wall) value.
	# Look into what is the appropriate BC here.
	f_top = f_top.at[:, :, -1].set(f_top[:, :, -2])
	return f_bottom_ccf, f_top


	
class OpticsScheme(abc.ABC):
	"""Abstract base class for optics scheme."""
	
	_EPSILON = 1e-6
	
	def __init__(
	  self,
	):
		self._halo_width = 1  # halos in z
		# self._face_interp_scheme_order = params.face_interp_scheme_order
		
		self.cloud_optics_lw = None
		self.cloud_optics_sw = None
		self.gas_optics_lw = None
		self.gas_optics_sw = None

	@abc.abstractmethod
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
		"""Computes the monochromatic longwave optical properties.
		
		Args:
		  pressure: The pressure field [Pa].
		  temperature: The temperature [K].
		  molecules: The number of molecules in an atmospheric grid cell per area
			[molecules/m²].
		  igpt: The spectral interval index, or g-point.
		  vmr_fields: An optional dictionary containing precomputed volume mixing
			ratio fields, keyed by gas index, that will overwrite the global means.
		  cloud_r_eff_liq: The effective radius of cloud droplets [m].
		  cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
			[kg/m²].
		  cloud_r_eff_ice: The effective radius of cloud ice particles [m].
		  cloud_path_ice: The cloud ice water path in each atmospheric grid cell
			[kg/m²].
		
		Returns:
		  A dictionary containing (for a single g-point):
			'optical_depth': The longwave optical depth.
			'ssa': The longwave single-scattering albedo.
			'asymmetry_factor': The longwave asymmetry factor.
		"""

	@abc.abstractmethod
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
		"""Computes the monochromatic shortwave optical properties.
		
		Args:
		  pressure: The pressure field [Pa].
		  temperature: The temperature [K].
		  molecules: The number of molecules in an atmospheric grid cell per area
			[molecules/m²].
		  igpt: The spectral interval index, or g-point.
		  vmr_fields: An optional dictionary containing precomputed volume mixing
			ratio fields, keyed by gas index, that will overwrite the global means.
		  cloud_r_eff_liq: The effective radius of cloud droplets [m].
		  cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
			[kg/m²].
		  cloud_r_eff_ice: The effective radius of cloud ice particles [m].
		  cloud_path_ice: The cloud ice water path in each atmospheric grid cell
			[kg/m²].
		
		Returns:
		  A dictionary containing (for a single g-point):
			'optical_depth': The shortwave optical depth.
			'ssa': The shortwave single-scattering albedo.
			'asymmetry_factor': The shortwave asymmetry factor.
		"""

		@abc.abstractmethod
		def compute_planck_sources(
		  self,
		  pressure: Array,
		  temperature: Array,
		  igpt: Array,
		  vmr_fields: dict[int, Array] | None = None,
		  sfc_temperature: Array | None = None,
		) -> dict[str, Array]:
			"""Computes the Planck sources used in the longwave problem.
			
			Args:
			  pressure: The pressure field [Pa].
			  temperature: The temperature [K].
			  igpt: The spectral interval index, or g-point.
			  vmr_fields: An optional dictionary containing precomputed volume mixing
				ratio fields, keyed by gas index, that will overwrite the global means.
			  sfc_temperature: An optional 2D plane for the surface temperature [K].
			
			Returns:
			  A dictionary containing the Planck source at the cell center
			  (`planck_src`), the top cell boundary (`planck_src_top`), and the bottom
			  cell boundary (`planck_src_bottom`).
			"""
	
	@property
	@abc.abstractmethod
	def n_gpt_lw(self) -> int:
		"""The number of g-points in the longwave bands."""
	
	@property
	@abc.abstractmethod
	def n_gpt_sw(self) -> int:
		"""The number of g-points in the shortwave bands."""
	
	@property
	@abc.abstractmethod
	def solar_fraction_by_gpt(self) -> Array:
		"""Mapping from g-point to the fraction of total solar radiation."""
	
	def combine_optical_properties(
		self,
		optical_props_1: Mapping[str, Array],
		optical_props_2: Mapping[str, Array],
	) -> dict[str, Array]:
		"""Combines the optical properties from two separate parameterizations."""
		tau1 = optical_props_1['optical_depth']
		tau2 = optical_props_2['optical_depth']
		ssa1 = optical_props_1['ssa']
		ssa2 = optical_props_2['ssa']
		g1 = optical_props_1['asymmetry_factor']
		g2 = optical_props_2['asymmetry_factor']
		
		# Combine optical depths
		tau = tau1 + tau2
		
		# Combine single-scattering albedos.
		ssa_unnormalized = tau1 * ssa1 + tau2 * ssa2
		
		# Combine asymmetry factors.
		g = (tau1 * ssa1 * g1 + tau2 * ssa2 * g2) / jnp.maximum(
			ssa_unnormalized, self._EPSILON
		)
		
		return {
			'optical_depth': tau,
			'ssa': ssa_unnormalized / jnp.maximum(tau, self._EPSILON),
			'asymmetry_factor': g,
		}
