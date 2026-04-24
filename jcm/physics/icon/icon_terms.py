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

from jcm.physics.physics_term import PhysicsTerm
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


class IconCloudsAndMicrophysics(IconTermBase):
    """Coupled cloud fraction and microphysics scheme."""

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

    def __add__(self, other):
        """Compose preserving the ComposableIconPhysics subclass.

        The base ``__add__`` returns a plain ``ComposablePhysics``, which
        loses the ICON parameter store and our custom ``data_struct_to_dict``
        (used for writing precip / surface flux diagnostics to xarray).
        """
        if hasattr(other, "terms"):
            other_terms = list(other.terms)
        elif hasattr(other, "category") and callable(other):
            other_terms = [other]
        else:
            return NotImplemented
        return ComposableIconPhysics(
            terms=list(self.terms) + other_terms,
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._icon_parameters.get_value(),
        )

    def data_struct_to_dict(self, struct, nodal_shape=None, sep="."):  # noqa: D401
        """Expose ICON-specific diagnostic fields for xarray output.

        The composable diagnostics dict uses ``_<sub>`` keys for each ICON
        sub-struct (radiation, convection, clouds, surface, ...). The parent
        class filters those out. Here we unpack a curated set of useful
        scalar diagnostics into top-level keys so they appear in the output
        Dataset for analysis (precip, evap, surface fluxes, cloud water).

        Fields coming out of column-vectorized terms have a trailing
        ``ncols = nlon * nlat`` axis; we reshape to ``(..., nlon, nlat)``
        so ``data_to_xarray`` can resolve the dims.
        """
        out = super().data_struct_to_dict(struct, nodal_shape=nodal_shape, sep=sep)

        if not isinstance(struct, dict):
            return out

        # If caller passed a 3-D nodal_shape (nlev, nlon, nlat), we want the
        # 2-D (nlon, nlat) view. Otherwise accept it as-is.
        nodal_2d = None
        if nodal_shape is not None:
            if len(nodal_shape) == 3:
                nodal_2d = (nodal_shape[1], nodal_shape[2])
            elif len(nodal_shape) == 2:
                nodal_2d = tuple(nodal_shape)

        def _reshape_to_nodal(arr):
            """Reshape trailing ncols axis → (nlon, nlat) when possible."""
            if nodal_2d is None:
                return arr
            ncols = nodal_2d[0] * nodal_2d[1]
            s = arr.shape
            if s and s[-1] == ncols:
                return arr.reshape(s[:-1] + nodal_2d)
            return arr

        # Walk the internal sub-structs and pick fields worth persisting.
        def _pick(sub, attrs, prefix):
            if sub is None:
                return
            for a in attrs:
                v = getattr(sub, a, None)
                if v is None:
                    continue
                out[f"{prefix}{a}"] = _reshape_to_nodal(v)

        _pick(struct.get("_convection"), ["precip_conv"], "convection.")
        _pick(struct.get("_clouds"),
              ["precip_rain", "precip_snow", "cloud_fraction", "qc", "qi"],
              "clouds.")
        _pick(struct.get("_surface"),
              ["latent_heat_flux", "sensible_heat_flux", "evaporation",
               "surface_temperature", "momentum_flux_u", "momentum_flux_v"],
              "surface.")
        _pick(struct.get("_vertical_diffusion"),
              ["tke", "pbl_height", "surface_friction_velocity",
               "monin_obukhov_length"],
              "vdiff.")
        _pick(struct.get("_radiation"),
              ["toa_lw_up", "toa_sw_up", "toa_sw_down",
               "surface_lw_down", "surface_sw_down", "surface_lw_up"],
              "radiation.")

        # Post-pass: normalize remaining sub-struct arrays. ``super()`` emits
        # every array in every typed sub-struct, including bulky fields on
        # half levels (``radiation.sw_flux_up`` has shape ``(..., nlev+1, ncols)``)
        # that ``data_to_xarray`` doesn't know how to label. Reshape trailing
        # ncols to ``(nlon, nlat)`` where possible, and drop any array whose
        # trailing two dims aren't the nodal grid — the curated picks above
        # already cover anything a user is likely to want.
        nlev_check = nodal_shape[0] if (nodal_shape and len(nodal_shape) == 3) else None
        if nodal_2d is not None:
            for k in list(out.keys()):
                v = out[k]
                if not hasattr(v, "shape"):
                    continue
                v = _reshape_to_nodal(v)
                out[k] = v
                s = v.shape
                # Require the trailing two dims to match the nodal grid so
                # ``data_to_xarray`` can resolve them. If there is an
                # additional vertical axis immediately before, it must equal
                # ``nlev`` (or 1 for surface-with-explicit-axis); anything
                # else — e.g. half-level ``nlev+1`` radiation fluxes — is
                # dropped.
                if len(s) < 2 or s[-2:] != nodal_2d:
                    del out[k]
                    continue
                # Only the ``(time, nlev, nlon, nlat)`` 4-D layout carries an
                # explicit vertical axis; 2-D and 3-D layouts don't. Catch
                # half-level fluxes by requiring the vertical dim (if
                # present) to equal nlev or 1 (surface-with-axis).
                if nlev_check is not None and len(s) == 4:
                    vert = s[-3]
                    if vert not in (nlev_check, 1):
                        del out[k]
        return out

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

        # PhysicsState carries specific_humidity (and mass-mixing-ratio
        # tracers) in g/kg — a SPEEDY legacy convention. ICON physics is
        # written for kg/kg (see formulas like e = q·p/(0.622 + 0.378·q)
        # in icon_physics.py, which requires q in kg/kg or else vapor
        # pressure exceeds total pressure at realistic moisture levels).
        # Convert on entry and scale the tendency back on exit so the
        # interface's g/kg/s → nondim step applies the correct units.
        vectorized_state = vectorized_state.copy(
            specific_humidity=vectorized_state.specific_humidity * 1e-3,
            tracers={
                name: tracer * 1e-3
                for name, tracer in vectorized_state.tracers.items()
            },
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

        # Scale q tendencies back kg/kg/s → g/kg/s to match PhysicsTendency
        # convention expected by physics_tendency_to_dynamics_tendency.
        acc = {
            **acc,
            "specific_humidity": acc["specific_humidity"] * 1e3,
            "tracers": {
                name: t * 1e3 for name, t in acc["tracers"].items()
            },
        }

        tendencies = _reshape_tendencies_to_3d(acc, nlev, nlat, nlon)
        return tendencies, diagnostics


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def icon_physics(
    parameters: Parameters | None = None,
    checkpoint_terms: bool = True,
    radiation_scheme: str = "grey",
):
    """Create a ComposableIconPhysics with standard ICON ordering.

    Args:
        parameters: Optional ICON Parameters. Uses defaults if None.
        checkpoint_terms: Whether to checkpoint terms.
        radiation_scheme: "grey" (default), "rrtmgp", or "emulated".

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

    return ComposableIconPhysics(
        terms=[
            IconPrepareState(),
            IconForcing(),
            IconAerosol(),
            IconChemistry(),
            rad_term,
            IconConvection(),
            IconCloudsAndMicrophysics(),
            IconVerticalDiffusion(),
            IconSurface(),
            IconGravityWaves(),
        ],
        checkpoint_terms=checkpoint_terms,
        parameters=p,
    )
