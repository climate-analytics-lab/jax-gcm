"""ECHAM physics term functions.

Standalone functions implementing the individual ECHAM parameterizations
(``apply_radiation``, ``apply_clouds_and_microphysics``, etc.). These are
wrapped by ``ComposablePhysics`` term classes in ``echam_terms.py``; there
is no monolithic orchestrator class — use ``echam_physics()`` from
``echam_terms`` to build a composable ECHAM physics package.

As schemes migrate to scheme-named ``PhysicsTerm`` classes living next to
their underlying numerics (Phase 2+ of the composable refactor), the
``apply_*`` functions in this file shrink. ``apply_convection`` has been
extracted to :class:`jcm.physics.convection.tiedtke_nordeng.TiedtkeConvection`.
"""

import logging

from jax import jit
import jax.numpy as jnp
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm import constants as physical_constants

# Import physics modules (will be implemented progressively)
from jcm.physics.echam.parameters import Parameters
from jcm.physics.echam.echam_physics_data import PhysicsData

logger = logging.getLogger(__name__)


def _column_vector(value: jnp.ndarray, ncols: int) -> jnp.ndarray:
    """Return a vmapped scalar diagnostic as one value per column."""
    return jnp.reshape(value, (ncols,))


@jit
def _prepare_common_physics_state(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Prepare common physics variables that are used by multiple physics modules.
    
    This reduces code duplication by computing pressure levels, heights, air density,
    and other commonly needed variables once for all physics modules.
    
    Args:
        state: Physics state variables (already in 2D format [nlev, ncols])
        boundaries: Boundary conditions (already updated with time-varying conditions)
        geometry: Model geometry
        
    Returns:
        Dictionary with common physics variables

    """
    p0 = physical_constants.p0
    
    # Calculate pressure levels from surface pressure and hybrid (a, b) coefficients.
    # Works for pure sigma (a=0, b=sigma) and ICON hybrid (a + b*P_s).
    surface_pressure = state.normalized_surface_pressure * p0  # Convert to Pa
    pressure_levels = physics_data.echam_coords.calculate_pressure_full(surface_pressure)
    pressure_half = physics_data.echam_coords.calculate_pressure_half(surface_pressure)
    
    # Convert geopotential to height
    height_levels = state.geopotential / physical_constants.grav

    # Calculate height at interfaces (half levels)
    # Internal interfaces are midpoints between full levels
    height_half_internal = (height_levels[1:] + height_levels[:-1]) / 2

    # Top interface: extrapolate using the same spacing as the top layer
    # This maintains consistent layer thickness at the top
    top_layer_thickness = height_levels[0] - height_half_internal[0]
    height_top = height_levels[0] + top_layer_thickness

    # Surface interface: use actual surface height (from geopotential at lowest level)
    # For sigma coordinates, assume surface is at orography height
    # A reasonable approximation is half the lowest layer below the lowest full level
    bottom_layer_thickness = height_half_internal[-1] - height_levels[-1]
    height_surface = height_levels[-1] - bottom_layer_thickness

    height_half = jnp.concatenate((
        height_top[jnp.newaxis],
        height_half_internal,
        height_surface[jnp.newaxis]), axis=0)

    # Calculate air density
    rho = pressure_levels / (physical_constants.rd * state.temperature)
    
    # Calculate layer thickness (clamp to minimum 10m for numerical stability
    # with thin uniform sigma layers)
    dp = jnp.diff(pressure_half, axis=0)
    dz_full = jnp.maximum(dp / (rho * physical_constants.grav), 10.0)
    
    # Calculate relative humidity (Tetens formula; clip T only enough to avoid
    # divide-by-zero at T=29.65K and exp overflow)
    # Wide math-safety clip; NOT a physical-range bound
    T_clip = jnp.clip(state.temperature, 50.0, 500.0)
    q_clip = jnp.maximum(state.specific_humidity, 0.0)
    es = 611.2 * jnp.exp(17.67 * (T_clip - 273.15) / (T_clip - 29.65))
    e = q_clip * pressure_levels / (0.622 + 0.378 * q_clip)
    rel_humidity = e / jnp.maximum(es, 1e-3)

    diagnostic_data = physics_data.diagnostics.copy(
        pressure_full=pressure_levels,
        pressure_half=pressure_half,
        height_full=height_levels,
        height_half=height_half,
        relative_humidity=rel_humidity,
        surface_pressure=surface_pressure,
        air_density=rho,
        layer_thickness=dz_full,
    )

    # Note: chemistry is intentionally not initialized here. ``apply_forcing_data``
    # (the next term in the physics sequence) unconditionally overwrites
    # ``physics_data.chemistry`` with constant GHG concentrations every step,
    # so any initialization work done here would be immediately discarded.
    updated_physics_data = physics_data.copy(diagnostics=diagnostic_data)

    zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return zero_tendencies, updated_physics_data

# Physics term methods


# ``apply_radiation`` was extracted to
# :class:`jcm.physics.radiation.grey_two_stream.GreyTwoStreamRadiation`
# (Phase 3 of the scheme-named-terms refactor). The shared caching gate
# now lives in
# :func:`jcm.physics.radiation.grey_two_stream.radiation_should_compute`
# / :func:`cached_radiation_tendency` so the RRTMGP and NN-emulator
# radiation terms can call them too.


# ``apply_radiation_rrtmgp`` was extracted to
# :class:`jcm.physics.radiation.rrtmgp.RRTMGPRadiation` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_radiation_emulated`` was extracted to
# :class:`jcm.physics.radiation.nn_emulator_scheme.NNEmulatorRadiation`
# (Phase 3 of the scheme-named-terms refactor).


# ``_apply_radiation_inner`` was moved into
# :class:`jcm.physics.radiation.grey_two_stream.GreyTwoStreamRadiation`
# as the term's ``_compute_full`` method.


# ``_apply_radiation_rrtmgp_inner`` was moved into
# :class:`jcm.physics.radiation.rrtmgp.RRTMGPRadiation` as the term's
# ``_compute_full`` method.


# ``_apply_radiation_emulated_inner`` was moved into
# :class:`jcm.physics.radiation.nn_emulator_scheme.NNEmulatorRadiation`
# as the term's ``_compute_full`` method.


# ``apply_convection`` was extracted to
# ``jcm.physics.convection.tiedtke_nordeng.TiedtkeConvection`` (Phase 2 of
# the scheme-named-terms refactor). The numerical implementation moved
# verbatim into the term ``__call__``.


# ``apply_cloud_fraction`` was extracted to
# :class:`jcm.physics.clouds.sundqvist.SundqvistCloudFraction` (Phase 3
# of the scheme-named-terms refactor).


# ``apply_microphysics_1m`` was extracted to
# :class:`jcm.physics.clouds.echam_1m.Echam1MMicrophysics` (Phase 3 of
# the scheme-named-terms refactor).


# ``apply_microphysics_2m`` was extracted to
# :class:`jcm.physics.clouds.lohmann_2m.Lohmann2MMicrophysics` (Phase 3
# of the scheme-named-terms refactor).


# ``apply_clouds_and_microphysics`` (the deprecated single-term variant)
# was retired together with its EchamCloudsAndMicrophysics wrapper —
# the ``cloud_scheme="1m"`` / ``cloud_scheme="2m"`` factory paths now
# use :class:`Echam1MMicrophysics` / :class:`Lohmann2MMicrophysics`.


# ``apply_vertical_diffusion`` was extracted to
# :class:`jcm.physics.vertical_diffusion.tte_tke.TteTkeVerticalDiffusion`
# (Phase 3 of the scheme-named-terms refactor).


# ``apply_surface`` was extracted to
# :class:`jcm.physics.surface.echam.surface_physics.EchamSurface`
# (Phase 3 of the scheme-named-terms refactor).

# ``apply_simple_gwd`` was extracted to
# :class:`jcm.physics.gravity_waves.simple.SimpleGwd` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_hines`` was extracted to
# :class:`jcm.physics.gravity_waves.hines.HinesGwd` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_sso`` was extracted to
# :class:`jcm.physics.gravity_waves.sso.LottMillerSso` (Phase 3 of the
# scheme-named-terms refactor).


# ``apply_chemistry`` was extracted to
# :class:`jcm.physics.chemistry.SimpleChemistry` (Phase 3 of the
# scheme-named-terms refactor).