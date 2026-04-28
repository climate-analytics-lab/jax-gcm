"""PhysicsTerm wrappers for existing ICON physics functions.

Each wrapper delegates to the original ICON function, translating between
the composable ``diagnostics`` dict and the legacy typed ``PhysicsData``
struct. The numerical implementation is untouched.

The ICON physics operates in column-vectorized format (nlev, ncols) rather
than 3D grid format (nlev, nlon, nlat). Column vectorization is handled by
``ComposablePhysics(vectorize_columns=True)``, so individual term wrappers
work in column format throughout.

Date: 2026-04-13
"""

from __future__ import annotations

from typing import ClassVar

from flax import nnx

from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.date import DateData
from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.physics.icon.icon_coords import IconCoords
from jcm.physics.icon.parameters import Parameters
from jcm.physics.composable_physics import ComposablePhysics


# ------------------------------------------------------------------
# Helpers for diagnostics ↔ PhysicsData translation
# ------------------------------------------------------------------

def _data_from_diagnostics(
    diagnostics: dict, coords: IconCoords,
    col_shape: tuple, num_levels: int,
) -> PhysicsData:
    """Reconstruct ICON PhysicsData from the diagnostics dict."""
    date = diagnostics.get("_date", DateData.zeros())

    data = PhysicsData.zeros(
        col_shape, num_levels,
        icon_coords=coords, date=date,
    )

    if "_radiation" in diagnostics:
        data = data.copy(radiation=diagnostics["_radiation"])
    if "_convection" in diagnostics:
        data = data.copy(convection=diagnostics["_convection"])
    if "_clouds" in diagnostics:
        data = data.copy(clouds=diagnostics["_clouds"])
    if "_vertical_diffusion" in diagnostics:
        data = data.copy(
            vertical_diffusion=diagnostics["_vertical_diffusion"],
        )
    if "_surface" in diagnostics:
        data = data.copy(surface=diagnostics["_surface"])
    if "_aerosol" in diagnostics:
        data = data.copy(aerosol=diagnostics["_aerosol"])
    if "_chemistry" in diagnostics:
        data = data.copy(chemistry=diagnostics["_chemistry"])
    if "_diagnostics" in diagnostics:
        data = data.copy(diagnostics=diagnostics["_diagnostics"])

    return data


def _diagnostics_from_data(
    diagnostics: dict, data: PhysicsData,
) -> dict:
    """Store all ICON PhysicsData sub-structs into the diagnostics dict."""
    return {
        **diagnostics,
        "_radiation": data.radiation,
        "_convection": data.convection,
        "_clouds": data.clouds,
        "_vertical_diffusion": data.vertical_diffusion,
        "_surface": data.surface,
        "_aerosol": data.aerosol,
        "_chemistry": data.chemistry,
        "_diagnostics": data.diagnostics,
    }


# ------------------------------------------------------------------
# Base class for ICON term wrappers
# ------------------------------------------------------------------

class IconTermBase(PhysicsTerm):
    """Base for ICON term wrappers.

    Handles IconCoords caching and provides the translation helpers.
    Each term accesses the full ICON Parameters from diagnostics
    (injected by ComposableIconPhysics) to ensure all terms share
    the same parameter state (including timestep).
    """

    def __init__(self):
        """Initialize base ICON term."""
        self._coords_cached = False

    def cache_coords(self, coords):
        """Cache IconCoords from the coordinate system."""
        self._icon_coords = nnx.Variable(
            IconCoords.from_coordinate_system(coords),
        )
        nodal_shape = self._icon_coords.get_value().nodal_shape
        self._num_levels = nodal_shape[0]
        self._col_shape = (nodal_shape[1] * nodal_shape[2],)
        self._nodal_shape_3d = nodal_shape
        self._coords_cached = True

    def _build_data(self, diagnostics: dict) -> PhysicsData:
        """Reconstruct PhysicsData from diagnostics."""
        return _data_from_diagnostics(
            diagnostics, self._icon_coords.get_value(),
            self._col_shape, self._num_levels,
        )

    def _get_params(self, diagnostics: dict) -> Parameters:
        """Get full ICON Parameters from diagnostics."""
        return diagnostics.get("_icon_params", Parameters.default())


# ------------------------------------------------------------------
# Concrete ICON term wrappers
# ------------------------------------------------------------------

class IconPrepareState(IconTermBase):
    """Compute common diagnostic fields (pressure, height, density)."""

    name: ClassVar[str] = "icon_prepare_state"
    category: ClassVar[str] = "prepare"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute diagnostic fields from state."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import (
            _prepare_common_physics_state,
        )
        tend, data = _prepare_common_physics_state(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconForcing(IconTermBase):
    """Set time-varying boundary conditions."""

    name: ClassVar[str] = "icon_forcing"
    category: ClassVar[str] = "forcing"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Apply forcing boundary conditions."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.forcing import apply_forcing_data
        tend, data = apply_forcing_data(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconAerosol(IconTermBase):
    """MACv2-SP simple plume aerosol scheme."""

    name: ClassVar[str] = "icon_aerosol"
    category: ClassVar[str] = "aerosol"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute aerosol optical properties."""
        data = self._build_data(diagnostics)
        from jcm.physics.aerosol.macv2_sp import (
            get_simple_aerosol,
        )
        tend, data = get_simple_aerosol(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconChemistry(IconTermBase):
    """Simple chemistry scheme for ozone, methane, CO2."""

    name: ClassVar[str] = "icon_chemistry"
    category: ClassVar[str] = "chemistry"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Update chemistry tracers."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_chemistry
        tend, data = apply_chemistry(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconRadiation(IconTermBase):
    """Grey-body radiation scheme."""

    name: ClassVar[str] = "icon_radiation"
    category: ClassVar[str] = "radiation"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute radiative heating rates and fluxes."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_radiation
        tend, data = apply_radiation(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconRadiationRRTMGP(IconTermBase):
    """RRTMGP full-spectrum radiation scheme."""

    name: ClassVar[str] = "icon_radiation_rrtmgp"
    category: ClassVar[str] = "radiation"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute RRTMGP radiative heating rates."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import (
            apply_radiation_rrtmgp,
        )
        tend, data = apply_radiation_rrtmgp(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconRadiationEmulated(IconTermBase):
    """Neural network radiation emulator (bidirectional GRU).

    Uses a pre-trained neural network to emulate radiative transfer,
    providing a fast, differentiable alternative to RRTMGP.
    See ``jcm.physics.icon.radiation.nn_emulator`` for details.
    """

    name: ClassVar[str] = "icon_radiation_emulated"
    category: ClassVar[str] = "radiation"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute NN-emulated radiative heating rates."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import (
            apply_radiation_emulated,
        )
        tend, data = apply_radiation_emulated(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconConvection(IconTermBase):
    """Tiedtke-Nordeng convection scheme."""

    name: ClassVar[str] = "icon_convection"
    category: ClassVar[str] = "convection"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute convective tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_convection
        tend, data = apply_convection(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class SundqvistCloudFraction(IconTermBase):
    """Sundqvist (1989) / Lohmann-Roeckner (1996) diagnostic cloud fraction.

    Diagnoses cloud fraction as ``cc = 1 - sqrt(1 - b0)`` with
    ``b0 = (RH - RH_crit) / (1 - RH_crit)`` and emits the associated
    condensation tendencies. Originally the ICON shallow-cloud step;
    renamed to reflect the underlying scheme rather than the package.
    """

    name: ClassVar[str] = "sundqvist_cloud_fraction"
    category: ClassVar[str] = "cloud_fraction"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute condensation tendencies and cloud-fraction diagnostics."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_cloud_fraction
        tend, data = apply_cloud_fraction(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconCloudsAndMicrophysics2M(IconTermBase):
    """ICON 2-moment cloud microphysics (Phase 5a: warm-rain only).

    Declares the full 2M prognostic tracer set — qc, qi, qnc, qni, qr, qs —
    via :meth:`required_tracers`. The qnc/qni number concentrations are
    stored per kg of air and carry ``nondimensionalize=False`` so they
    round-trip through the modal/nodal converters without the gram/kg scaling
    that mass mixing ratios get.

    Only the Khairoutdinov-Kogan warm-rain autoconversion is wired in at this
    stage; ice-phase and sedimentation work is tracked in issue #341. Must be
    composed downstream of :class:`SundqvistCloudFraction`.
    """

    name: ClassVar[str] = "icon_clouds_microphysics_2m"
    category: ClassVar[str] = "clouds"

    @classmethod
    def required_tracers(cls):
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
            TracerSpec("qnc", units="kg^-1", nondimensionalize=False),
            TracerSpec("qni", units="kg^-1", nondimensionalize=False),
            TracerSpec("qr", units="kg/kg"),
            TracerSpec("qs", units="kg/kg"),
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute 2-moment microphysics tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_microphysics_2m
        tend, data = apply_microphysics_2m(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconCloudsAndMicrophysics1M(IconTermBase):
    """ICON 1-moment cloud microphysics (autoconversion + precipitation).

    Reads post-condensation ``qc``/``qi``/``cloud_fraction`` from the
    diagnostics dict; must be composed downstream of an
    :class:`SundqvistCloudFraction` term.
    """

    name: ClassVar[str] = "icon_clouds_microphysics_1m"
    category: ClassVar[str] = "clouds"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute microphysics tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_microphysics_1m
        tend, data = apply_microphysics_1m(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconCloudsAndMicrophysics(IconTermBase):
    """Coupled cloud fraction and microphysics scheme (legacy single-term).

    Deprecated: use :class:`SundqvistCloudFraction` + :class:`IconCloudsAndMicrophysics1M`
    instead. Kept for backward compat with existing call sites.
    """

    name: ClassVar[str] = "icon_clouds_microphysics"
    category: ClassVar[str] = "clouds"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute cloud and microphysics tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import (
            apply_clouds_and_microphysics,
        )
        tend, data = apply_clouds_and_microphysics(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconVerticalDiffusion(IconTermBase):
    """TKE-based vertical diffusion and boundary layer."""

    name: ClassVar[str] = "icon_vertical_diffusion"
    category: ClassVar[str] = "vertical_diffusion"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute vertical diffusion tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import (
            apply_vertical_diffusion,
        )
        tend, data = apply_vertical_diffusion(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconSurface(IconTermBase):
    """Surface fluxes for ocean, sea ice, and land."""

    name: ClassVar[str] = "icon_surface"
    category: ClassVar[str] = "surface"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute surface flux tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_surface
        tend, data = apply_surface(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class IconGravityWaves(IconTermBase):
    """Orographic gravity wave drag."""

    name: ClassVar[str] = "icon_gravity_waves"
    category: ClassVar[str] = "gravity_waves"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute gravity wave drag tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.icon.icon_physics import apply_gravity_waves
        tend, data = apply_gravity_waves(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


# ------------------------------------------------------------------
# Helper to build ICON Parameters with overrides
# ------------------------------------------------------------------

def _icon_params_with(**overrides) -> Parameters:
    """Build ICON Parameters from defaults with specific overrides."""
    p = Parameters.default()
    return Parameters(
        convection=overrides.get("convection", p.convection),
        clouds=overrides.get("clouds", p.clouds),
        microphysics=overrides.get("microphysics", p.microphysics),
        microphysics_2m=overrides.get("microphysics_2m", p.microphysics_2m),
        gravity_waves=overrides.get("gravity_waves", p.gravity_waves),
        radiation=overrides.get("radiation", p.radiation),
        vertical_diffusion=overrides.get(
            "vertical_diffusion", p.vertical_diffusion,
        ),
        surface=overrides.get("surface", p.surface),
        aerosol=overrides.get("aerosol", p.aerosol),
    )


# ------------------------------------------------------------------
# ComposableIconPhysics — ICON parameter management
# ------------------------------------------------------------------

class ComposableIconPhysics(ComposablePhysics):
    """ComposablePhysics with ICON shared parameter management.

    Column vectorization is handled by the parent class via
    ``vectorize_columns=True``. This subclass adds ICON-specific
    parameter storage and timestep management.

    The full ICON ``Parameters`` is stored and injected into the
    diagnostics dict as ``_icon_params`` so all terms share it.
    """

    def __init__(self, terms, checkpoint_terms=True, parameters=None):
        """Initialize with ICON-specific parameter storage."""
        super().__init__(
            terms, checkpoint_terms, vectorize_columns=True,
        )
        self._icon_parameters = nnx.Variable(
            parameters or Parameters.default(),
        )

    @property
    def parameters(self) -> Parameters:
        """Read access to the shared ICON parameters struct."""
        return self._icon_parameters.get_value()

    def replace(self, category, new_term):
        """Replace a term, preserving ComposableIconPhysics type."""
        new_terms = []
        inserted = False
        for t in self.terms:
            if t.category == category:
                if not inserted:
                    new_terms.append(new_term)
                    inserted = True
            else:
                new_terms.append(t)
        if not inserted:
            raise ValueError(
                f"No term with category {category!r} found.",
            )
        return ComposableIconPhysics(
            terms=new_terms,
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._icon_parameters.get_value(),
        )

    def remove(self, category):
        """Remove terms, preserving ComposableIconPhysics type."""
        return ComposableIconPhysics(
            terms=[
                t for t in self.terms if t.category != category
            ],
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._icon_parameters.get_value(),
        )

    def apply_timestep(self, dt_seconds: float):
        """Update timestep on the shared ICON parameters.

        This mirrors ``IconPhysics.parameters.with_timestep()``.

        """
        p = self._icon_parameters.get_value()
        self._icon_parameters = nnx.Variable(
            p.with_timestep(dt_seconds),
        )

    def _compute_tendencies_columns(
        self, state, forcing, terrain, date,
        prev_physics_data=None,
    ):
        """Override to inject ICON parameters into diagnostics."""
        import jax
        import jax.numpy as jnp
        from jcm.physics.composable_physics import (
            _reshape_state_to_columns,
            _accumulate,
            _reshape_tendencies_to_3d,
        )

        nlev, nlon, nlat = state.temperature.shape
        ncols = nlat * nlon

        vectorized_state = _reshape_state_to_columns(
            state, nlev, ncols,
        )

        diagnostics: dict = {}
        if prev_physics_data is not None:
            diagnostics = {**prev_physics_data}

        diagnostics["_date"] = date
        diagnostics["_icon_params"] = self._icon_parameters.get_value()

        tracer_tends = {
            name: jnp.zeros((nlev, ncols))
            for name in state.tracers
        }
        acc = {
            "u_wind": jnp.zeros((nlev, ncols)),
            "v_wind": jnp.zeros((nlev, ncols)),
            "temperature": jnp.zeros((nlev, ncols)),
            "specific_humidity": jnp.zeros((nlev, ncols)),
            "tracers": tracer_tends,
        }

        for term in self.terms:
            call_fn = (
                jax.checkpoint(term)
                if self.checkpoint_terms
                else term
            )
            tend, diagnostics = call_fn(
                vectorized_state, diagnostics, forcing, terrain,
            )
            acc = _accumulate(acc, tend)

        tendencies = _reshape_tendencies_to_3d(acc, nlev, nlat, nlon)
        return tendencies, diagnostics


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def icon_physics(
    parameters: Parameters | None = None,
    checkpoint_terms: bool = True,
    radiation_scheme: str = "grey",
    cloud_scheme: str = "1m",
):
    """Create a ComposableIconPhysics with standard ICON ordering.

    Args:
        parameters: Optional ICON Parameters. Uses defaults if None.
        checkpoint_terms: Whether to checkpoint terms.
        radiation_scheme: "grey" (default), "rrtmgp", or "emulated".
        cloud_scheme: "1m" (default, single-moment) or "2m" (two-moment
            warm-rain; see issue #341 for ongoing scheme completion).

    Returns:
        A ComposableIconPhysics instance with all ICON terms.

    """
    p = parameters or Parameters.default()

    if radiation_scheme == "rrtmgp":
        rad_term = IconRadiationRRTMGP()
    elif radiation_scheme == "grey":
        rad_term = IconRadiation()
    elif radiation_scheme == "emulated":
        rad_term = IconRadiationEmulated()
    else:
        raise ValueError(
            f"Unknown radiation_scheme={radiation_scheme!r}. "
            "Choose 'grey', 'rrtmgp', or 'emulated'."
        )

    if cloud_scheme == "1m":
        micro_term = IconCloudsAndMicrophysics1M()
    elif cloud_scheme == "2m":
        micro_term = IconCloudsAndMicrophysics2M()
    else:
        raise ValueError(
            f"Unknown cloud_scheme={cloud_scheme!r}. Choose '1m' or '2m'."
        )

    return ComposableIconPhysics(
        terms=[
            IconPrepareState(),
            IconForcing(),
            IconAerosol(),
            IconChemistry(),
            rad_term,
            IconConvection(),
            SundqvistCloudFraction(),
            micro_term,
            IconVerticalDiffusion(),
            IconSurface(),
            IconGravityWaves(),
        ],
        checkpoint_terms=checkpoint_terms,
        parameters=p,
    )
