"""PhysicsTerm wrappers for existing ECHAM physics functions.

Each wrapper delegates to the original ECHAM function, translating between
the composable ``diagnostics`` dict and the legacy typed ``PhysicsData``
struct. The numerical implementation is untouched.

The ECHAM physics operates in column-vectorized format (nlev, ncols) rather
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
from jcm.physics.echam.echam_physics_data import PhysicsData
from jcm.physics.echam.echam_coords import EchamCoords
from jcm.physics.echam.parameters import Parameters
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.convection.tiedtke_nordeng import TiedtkeConvection
from jcm.physics.diagnostics.moist_air_state import (
    MOIST_AIR_FIELDS,
    MoistAirColumnState,
)
from jcm.physics.aerosol import Macv2SpAerosol
from jcm.physics.chemistry import SimpleChemistry
from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
from jcm.physics.clouds.sundqvist import SundqvistCloudFraction
from jcm.physics.gravity_waves.hines import HinesGwd
from jcm.physics.gravity_waves.sso import LottMillerSso
from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
from jcm.physics.forcing.echam_boundary_conditions import (
    EchamBoundaryConditions,
)


# ------------------------------------------------------------------
# Helpers for diagnostics ↔ PhysicsData translation
# ------------------------------------------------------------------

def _data_from_diagnostics(
    diagnostics: dict, coords: EchamCoords,
    col_shape: tuple, num_levels: int,
) -> PhysicsData:
    """Reconstruct ECHAM PhysicsData from the diagnostics dict.

    The moist-air diagnostics (pressure_full, height_full, …) are now
    written by :class:`MoistAirColumnState` as top-level public keys.
    Reassemble them into ``data.diagnostics`` here so legacy ``apply_*``
    consumers continue to see the typed sub-struct they expect. Falls
    back to a legacy ``_diagnostics`` typed key if any caller still
    writes it (defence in depth during the deprecation window).
    """
    date = diagnostics.get("_date", DateData.zeros())

    data = PhysicsData.zeros(
        col_shape, num_levels,
        echam_coords=coords,
        model_step=date.model_step,
        dt_seconds=date.dt_seconds,
    )

    diag_overrides = {
        f: diagnostics[f] for f in MOIST_AIR_FIELDS if f in diagnostics
    }
    if diag_overrides:
        data = data.copy(
            diagnostics=data.diagnostics.copy(**diag_overrides),
        )
    elif "_diagnostics" in diagnostics:
        data = data.copy(diagnostics=diagnostics["_diagnostics"])

    # ``radiation`` lives under a public top-level key after Phase 3
    # (``GreyTwoStreamRadiation`` / ``EchamBoundaryConditions``); fall back to
    # the legacy ``_radiation`` typed key for safety.
    if "radiation" in diagnostics:
        data = data.copy(radiation=diagnostics["radiation"])
    elif "_radiation" in diagnostics:
        data = data.copy(radiation=diagnostics["_radiation"])
    # ``clouds`` lives under a public top-level key after Phase 3
    # (``SundqvistCloudFraction`` / microphysics terms); fall back to
    # the legacy ``_clouds`` typed key for safety.
    if "clouds" in diagnostics:
        data = data.copy(clouds=diagnostics["clouds"])
    elif "_clouds" in diagnostics:
        data = data.copy(clouds=diagnostics["_clouds"])
    if "_vertical_diffusion" in diagnostics:
        data = data.copy(
            vertical_diffusion=diagnostics["_vertical_diffusion"],
        )
    if "_surface" in diagnostics:
        data = data.copy(surface=diagnostics["_surface"])
    # ``aerosol`` lives under a public top-level key after Phase 3
    # (``Macv2SpAerosol``); fall back to the legacy ``_aerosol`` typed key.
    if "aerosol" in diagnostics:
        data = data.copy(aerosol=diagnostics["aerosol"])
    elif "_aerosol" in diagnostics:
        data = data.copy(aerosol=diagnostics["_aerosol"])
    # ``chemistry`` lives under a public top-level key after Phase 3
    # (``SimpleChemistry`` / ``EchamBoundaryConditions``); fall back to
    # the legacy ``_chemistry`` typed key for safety.
    if "chemistry" in diagnostics:
        data = data.copy(chemistry=diagnostics["chemistry"])
    elif "_chemistry" in diagnostics:
        data = data.copy(chemistry=diagnostics["_chemistry"])

    return data


def _diagnostics_from_data(
    diagnostics: dict, data: PhysicsData,
) -> dict:
    """Store ECHAM PhysicsData sub-structs into the diagnostics dict.

    Moist-air diagnostics are written as top-level public keys (no
    leading underscore, so they appear in user-facing xarray output as
    ``pressure_full`` / ``height_full`` / … instead of the old
    ``diagnostics.pressure_full`` / ``diagnostics.height_full``). This
    propagates any in-step mutation a legacy ``apply_*`` made to
    ``data.diagnostics`` (e.g. ``apply_cloud_fraction`` updating
    ``relative_humidity``).
    """
    out = {
        **diagnostics,
        "radiation": data.radiation,
        "clouds": data.clouds,
        "_vertical_diffusion": data.vertical_diffusion,
        "_surface": data.surface,
        "aerosol": data.aerosol,
        "chemistry": data.chemistry,
    }
    for field in MOIST_AIR_FIELDS:
        out[field] = getattr(data.diagnostics, field)
    return out


# ------------------------------------------------------------------
# Base class for ECHAM term wrappers
# ------------------------------------------------------------------

class EchamTermBase(PhysicsTerm):
    """Base for ECHAM term wrappers.

    Handles EchamCoords caching and provides the translation helpers.
    Each term accesses the full ECHAM Parameters from diagnostics
    (injected by ComposableEchamPhysics) to ensure all terms share
    the same parameter state (including timestep).
    """

    def __init__(self):
        """Initialize base ECHAM term."""
        self._coords_cached = False

    def cache_coords(self, coords):
        """Cache EchamCoords from the coordinate system."""
        self._echam_coords = nnx.Variable(
            EchamCoords.from_coordinate_system(coords),
        )
        nodal_shape = self._echam_coords.get_value().nodal_shape
        self._num_levels = nodal_shape[0]
        self._col_shape = (nodal_shape[1] * nodal_shape[2],)
        self._nodal_shape_3d = nodal_shape
        self._coords_cached = True

    def _build_data(self, diagnostics: dict) -> PhysicsData:
        """Reconstruct PhysicsData from diagnostics."""
        return _data_from_diagnostics(
            diagnostics, self._echam_coords.get_value(),
            self._col_shape, self._num_levels,
        )

    def _get_params(self, diagnostics: dict) -> Parameters:
        """Get full ECHAM Parameters from diagnostics."""
        return diagnostics.get("_echam_params", Parameters.default())


# ------------------------------------------------------------------
# Concrete ECHAM term wrappers
# ------------------------------------------------------------------

class EchamRadiationEmulated(EchamTermBase):
    """Neural network radiation emulator (bidirectional GRU).

    Uses a pre-trained neural network to emulate radiative transfer,
    providing a fast, differentiable alternative to RRTMGP.
    See ``jcm.physics.echam.radiation.nn_emulator`` for details.
    """

    name: ClassVar[str] = "echam_radiation_emulated"
    category: ClassVar[str] = "radiation"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute NN-emulated radiative heating rates."""
        data = self._build_data(diagnostics)
        from jcm.physics.echam.echam_physics import (
            apply_radiation_emulated,
        )
        tend, data = apply_radiation_emulated(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class EchamCloudsAndMicrophysics2M(EchamTermBase):
    """ECHAM 2-moment cloud microphysics (Phase 5a: warm-rain only).

    Declares the full 2M prognostic tracer set — qc, qi, qnc, qni, qr, qs —
    via :meth:`required_tracers`. The qnc/qni number concentrations are
    stored per kg of air and carry ``nondimensionalize=False`` so they
    round-trip through the modal/nodal converters without the gram/kg scaling
    that mass mixing ratios get.

    Only the Khairoutdinov-Kogan warm-rain autoconversion is wired in at this
    stage; ice-phase and sedimentation work is tracked in issue #341. Must be
    composed downstream of :class:`SundqvistCloudFraction`.
    """

    name: ClassVar[str] = "echam_clouds_microphysics_2m"
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
        from jcm.physics.echam.echam_physics import apply_microphysics_2m
        tend, data = apply_microphysics_2m(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class EchamCloudsAndMicrophysics(EchamTermBase):
    """Coupled cloud fraction and microphysics scheme (legacy single-term).

    Deprecated: use :class:`SundqvistCloudFraction` + :class:`EchamCloudsAndMicrophysics1M`
    instead. Kept for backward compat with existing call sites.
    """

    name: ClassVar[str] = "echam_clouds_microphysics"
    category: ClassVar[str] = "clouds"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute cloud and microphysics tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.echam.echam_physics import (
            apply_clouds_and_microphysics,
        )
        tend, data = apply_clouds_and_microphysics(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class EchamVerticalDiffusion(EchamTermBase):
    """TKE-based vertical diffusion and boundary layer."""

    name: ClassVar[str] = "echam_vertical_diffusion"
    category: ClassVar[str] = "vertical_diffusion"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute vertical diffusion tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.echam.echam_physics import (
            apply_vertical_diffusion,
        )
        tend, data = apply_vertical_diffusion(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


class EchamSurface(EchamTermBase):
    """Surface fluxes for ocean, sea ice, and land."""

    name: ClassVar[str] = "echam_surface"
    category: ClassVar[str] = "surface"

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute surface flux tendencies."""
        data = self._build_data(diagnostics)
        from jcm.physics.echam.echam_physics import apply_surface
        tend, data = apply_surface(
            state, data,
            self._get_params(diagnostics), forcing, terrain,
        )
        return tend, _diagnostics_from_data(diagnostics, data)


# ``EchamSimpleGwd`` was extracted to
# :class:`jcm.physics.gravity_waves.simple.SimpleGwd` (Phase 3 of the
# scheme-named-terms refactor). It was never wired into the default
# ``echam_physics()`` factory; users who want the cheap GWD now compose
# ``SimpleGwd()`` in directly.


# ------------------------------------------------------------------
# Helper to build ECHAM Parameters with overrides
# ------------------------------------------------------------------

def _echam_params_with(**overrides) -> Parameters:
    """Build ECHAM Parameters from defaults with specific overrides."""
    p = Parameters.default()
    return Parameters(
        convection=overrides.get("convection", p.convection),
        clouds=overrides.get("clouds", p.clouds),
        microphysics=overrides.get("microphysics", p.microphysics),
        microphysics_2m=overrides.get("microphysics_2m", p.microphysics_2m),
        hines=overrides.get("hines", p.hines),
        sso=overrides.get("sso", p.sso),
        simple_gwd=overrides.get("simple_gwd", p.simple_gwd),
        radiation=overrides.get("radiation", p.radiation),
        vertical_diffusion=overrides.get(
            "vertical_diffusion", p.vertical_diffusion,
        ),
        surface=overrides.get("surface", p.surface),
        aerosol=overrides.get("aerosol", p.aerosol),
    )


# ------------------------------------------------------------------
# ComposableEchamPhysics — ECHAM parameter management
# ------------------------------------------------------------------

class ComposableEchamPhysics(ComposablePhysics):
    """ComposablePhysics with ECHAM shared parameter management.

    Column vectorization is handled by the parent class via
    ``vectorize_columns=True``. This subclass adds ECHAM-specific
    parameter storage and timestep management.

    The full ECHAM ``Parameters`` is stored and injected into the
    diagnostics dict as ``_echam_params`` so all terms share it.
    """

    def __init__(self, terms, checkpoint_terms=True, parameters=None):
        """Initialize with ECHAM-specific parameter storage."""
        super().__init__(
            terms, checkpoint_terms, vectorize_columns=True,
        )
        self._echam_parameters = nnx.Variable(
            parameters or Parameters.default(),
        )

    @property
    def parameters(self) -> Parameters:
        """Read access to the shared ECHAM parameters struct."""
        return self._echam_parameters.get_value()

    def replace(self, category, new_term):
        """Replace a term, preserving ComposableEchamPhysics type."""
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
        return ComposableEchamPhysics(
            terms=new_terms,
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._echam_parameters.get_value(),
        )

    def remove(self, category):
        """Remove terms, preserving ComposableEchamPhysics type."""
        return ComposableEchamPhysics(
            terms=[
                t for t in self.terms if t.category != category
            ],
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._echam_parameters.get_value(),
        )

    def __add__(self, other):
        """Append term(s), preserving ComposableEchamPhysics type.

        Without this override the parent ``ComposablePhysics.__add__``
        returns a plain ``ComposablePhysics``; ``Model.__init__`` would
        then skip the ``apply_timestep`` call (it's gated on
        ``isinstance(..., ComposableEchamPhysics)``) and ECHAM terms
        would silently keep the default ``dt_conv = 3600 s`` regardless
        of the actual model timestep — corrupting any flux that uses
        ``dt_conv`` (e.g. surface implicit damping factor).
        """
        if hasattr(other, "terms"):
            other_terms = list(other.terms)
        elif hasattr(other, "category") and callable(other):
            other_terms = [other]
        else:
            return NotImplemented
        return ComposableEchamPhysics(
            terms=list(self.terms) + other_terms,
            checkpoint_terms=self.checkpoint_terms,
            parameters=self._echam_parameters.get_value(),
        )

    def apply_timestep(self, dt_seconds: float):
        """Update timestep on the shared ECHAM parameters.

        This mirrors ``EchamPhysics.parameters.with_timestep()``.

        """
        p = self._echam_parameters.get_value()
        self._echam_parameters = nnx.Variable(
            p.with_timestep(dt_seconds),
        )

    def _compute_tendencies_columns(
        self, state, forcing, terrain, date,
        prev_physics_data=None,
    ):
        """Override to inject ECHAM parameters into diagnostics."""
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
        diagnostics["_echam_params"] = self._echam_parameters.get_value()

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

def echam_physics(
    parameters: Parameters | None = None,
    checkpoint_terms: bool = True,
    radiation_scheme: str | PhysicsTerm = "grey",
    cloud_scheme: str = "1m",
):
    """Create a ComposableEchamPhysics with standard ECHAM ordering.

    Args:
        parameters: Optional ECHAM Parameters. Uses defaults if None.
        checkpoint_terms: Whether to checkpoint terms.
        radiation_scheme: "grey" (default), "rrtmgp", "emulated", or a
            custom ``PhysicsTerm`` with category "radiation".
        cloud_scheme: "1m" (default, single-moment) or "2m" (two-moment
            warm-rain; see issue #341 for ongoing scheme completion).

    Returns:
        A ComposableEchamPhysics instance with all ECHAM terms.

    """
    p = parameters or Parameters.default()

    if isinstance(radiation_scheme, PhysicsTerm):
        if radiation_scheme.category != "radiation":
            raise ValueError(
                "Custom radiation_scheme terms must have category "
                "'radiation'."
            )
        rad_term = radiation_scheme
    elif radiation_scheme == "rrtmgp":
        rad_term = RRTMGPRadiation(params=p.radiation)
    elif radiation_scheme == "grey":
        rad_term = GreyTwoStreamRadiation(params=p.radiation)
    elif radiation_scheme == "emulated":
        rad_term = EchamRadiationEmulated()
    else:
        raise ValueError(
            f"Unknown radiation_scheme={radiation_scheme!r}. "
            "Choose 'grey', 'rrtmgp', 'emulated', or pass a radiation "
            "PhysicsTerm."
        )

    if cloud_scheme == "1m":
        micro_term = Echam1MMicrophysics(params=p.microphysics)
    elif cloud_scheme == "2m":
        micro_term = EchamCloudsAndMicrophysics2M()
    else:
        raise ValueError(
            f"Unknown cloud_scheme={cloud_scheme!r}. Choose '1m' or '2m'."
        )

    return ComposableEchamPhysics(
        terms=[
            MoistAirColumnState(),
            EchamBoundaryConditions(),
            Macv2SpAerosol(params=p.aerosol),
            SimpleChemistry(),
            rad_term,
            TiedtkeConvection(params=p.convection),
            SundqvistCloudFraction(params=p.clouds),
            micro_term,
            EchamVerticalDiffusion(),
            EchamSurface(),
            HinesGwd(params=p.hines),
            LottMillerSso(params=p.sso),
        ],
        checkpoint_terms=checkpoint_terms,
        parameters=p,
    )
