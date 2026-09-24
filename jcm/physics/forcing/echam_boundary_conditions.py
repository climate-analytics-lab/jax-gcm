"""``EchamBoundaryConditions`` — ECHAM time-varying boundary condition term.

Sets the surface and chemistry inputs that the rest of the ECHAM physics
expects to find in the typed ``RadiationData`` / ``SurfaceData`` /
``ChemistryData`` sub-structs:

- Surface albedo (visible + NIR) and emissivity: the land / sea-ice / ocean
  tile average of ECHAM 6.3's per-tile albedo schemes
  (:mod:`jcm.physics.surface.echam.albedo` — background albedo ``alb0`` with
  prescribed ``snowc_am`` snow cover, forest masking and glaciers over land;
  temperature-dependent sea ice; zenith-dependent open water) and per-tile
  emissivities, all held in a differentiable
  :class:`SurfaceOpticsParameters` (#347, #672).
- Surface temperature (land = ``forcing.stl_am``; ocean = ``forcing.sea_surface_temperature``).
- Roughness length (1 cm over land, 0.1 mm over ocean).
- CO2 and CH4 from ``forcing.co2_vmr`` / ``forcing.ch4_vmr``; O3 from the
  ``forcing.ozone_climatology`` when loaded, else this term's analytical
  profile.

The numerical implementation matches what was previously in
``apply_forcing_data`` (echam/forcing.py); this term is the ECHAM-specific
home for that routine. The ``Echam`` prefix is intentional — the albedo
schemes and the analytic-ozone fallback are ECHAM choices, not generic
boundary conditions.

Typed sub-structs are written into the diagnostics dict under the legacy
``_radiation`` / ``_surface`` / ``_chemistry`` keys so that the legacy
``apply_*`` consumer terms (which still build a full ``PhysicsData`` via
``_data_from_diagnostics``) see the same shape they always have. As those
consumer terms migrate to scheme-named terms in later phases, this term
will move to writing scheme-public keys directly.

"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx, struct

from jcm.forcing import ForcingData
from jcm.physics.chemistry.simple_chemistry import ChemistryData
from jcm.physics.coords_util import column_lat_lon
from jcm.physics.radiation import current_cos_zenith
from jcm.physics.radiation.radiation_types import RadiationData
from jcm.physics.surface.echam import albedo as albedo_scheme
from jcm.physics.surface.echam.albedo import EchamSurfaceAlbedoParameters
from jcm.physics.surface.echam.surface_types import SurfaceData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


@struct.dataclass
class SurfaceOpticsParameters:
    """Surface albedo and emissivity constants (#347, #672).

    Albedo follows ECHAM 6.3's per-tile schemes
    (:mod:`jcm.physics.surface.echam.albedo`); emissivity is a per-tile
    constant. Every numeric value is a differentiable pytree leaf.
    """

    albedo: EchamSurfaceAlbedoParameters = struct.field(
        default_factory=EchamSurfaceAlbedoParameters)
    land_emissivity: jnp.ndarray = 0.95
    ocean_emissivity: jnp.ndarray = 0.98
    seaice_emissivity: jnp.ndarray = 0.95


def _tile_partition(land_fraction, sea_ice_fraction):
    """``(land, sea_ice, ocean)`` fractions that sum to one.

    Sea ice is clipped against the land fraction so the three tiles are a
    partition of the box. The forcing bundle flags permanent land ice as
    ``icec = 1`` where ``lsm = 1``, and an unclipped blend then sums to 2
    over polar land: emissivity reaches 1.9 and the surface reflectance
    ``1 - eps`` goes negative. The other two consumers of ``sice_am``
    (``surface/echam/surface_physics.py``,
    ``vertical_diffusion/tte_tke/vertical_diffusion.py``) clip the same way,
    so radiation and the surface tiles see one partition (#703).
    """
    sea_ice_fraction = jnp.clip(sea_ice_fraction, 0.0, 1.0 - land_fraction)
    ocean_fraction = jnp.maximum(
        1.0 - land_fraction - sea_ice_fraction, 0.0,
    )
    return land_fraction, sea_ice_fraction, ocean_fraction


def _surface_optical_properties(
    land_fraction: jnp.ndarray,
    sea_ice_fraction: jnp.ndarray,
    p: SurfaceOpticsParameters,
    *,
    background_albedo: jnp.ndarray,
    snow_fraction: jnp.ndarray,
    land_temperature: jnp.ndarray,
    ice_temperature: jnp.ndarray,
    cos_zenith: jnp.ndarray,
    forest_fraction: jnp.ndarray | float = 0.0,
    glacier_fraction: jnp.ndarray | float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Grid-box visible / near-IR albedo and emissivity.

    ECHAM averages the tile albedos band by band with the tile fractions
    (``mo_surface.f90`` ``average_tiles`` of ``albedo_vis``/``albedo_nir``).
    Land and sea ice are broadband, so the same value enters both bands;
    open water carries its direct-beam band offsets (merged with the diffuse
    albedo, see :data:`~jcm.physics.surface.echam.albedo.OCEAN_DIRECT_WEIGHT`).

    Snow cover is prescribed: ``snow_fraction`` is the climatological
    ``snowc_am`` and there is no snow on the sea ice, so the ice takes the
    bare-ice constants. Canopy snow and the leaf area index are not carried
    either; ``land_albedo`` then uses JSBACH's own ``MAX(lai, 2)`` floor and
    a snow-free canopy. Prognostic snow is the open half of #672.
    """
    land_fraction, sea_ice_fraction, ocean_fraction = _tile_partition(
        land_fraction, sea_ice_fraction)
    a = p.albedo
    land = albedo_scheme.land_albedo(
        background_albedo, snow_fraction, land_temperature, a,
        forest_fraction=forest_fraction, glacier_fraction=glacier_fraction,
    )
    ice = albedo_scheme.sea_ice_albedo(ice_temperature, a)
    ocean_vis, ocean_nir = albedo_scheme.ocean_albedo_per_band(cos_zenith, a)
    albedo_vis = (
        land_fraction * land
        + ocean_fraction * ocean_vis
        + sea_ice_fraction * ice
    )
    albedo_nir = (
        land_fraction * land
        + ocean_fraction * ocean_nir
        + sea_ice_fraction * ice
    )
    emissivity = (
        land_fraction * p.land_emissivity
        + ocean_fraction * p.ocean_emissivity
        + sea_ice_fraction * p.seaice_emissivity
    )
    return albedo_vis, albedo_nir, emissivity


def sea_ice_surface_temperature(sea_surface_temperature, sea_ice_fraction):
    """Prescribed sea-ice tile temperature, ``min(SST, ctfreez)``.

    The same value the vertical diffusion and ``EchamSurface`` use for the
    ice tile, so the albedo and the turbulent fluxes see one ice surface.
    """
    return jnp.where(
        sea_ice_fraction > 0.0,
        jnp.minimum(sea_surface_temperature, albedo_scheme.CTFREEZ),
        sea_surface_temperature,
    )


class EchamBoundaryConditions(PhysicsTerm):
    """Apply ECHAM time-varying boundary conditions to the diagnostics dict.

    Operates on column-vectorized state ``(nlev, ncols)``. Writes ECHAM
    radiation/surface/chemistry typed sub-structs under the legacy
    ``_radiation`` / ``_surface`` / ``_chemistry`` keys for downstream
    legacy ``apply_*`` consumers.
    """

    name: ClassVar[str] = "echam_boundary_conditions"
    category: ClassVar[str] = "forcing"
    requires: ClassVar[tuple[str, ...]] = (
        # Needed by the analytical ozone profile (Fortuin & Kelder-style)
        # that seeds ``chemistry.ozone_vmr`` each step. ``MoistAirColumnState``
        # populates both diagnostics; this term must run after it.
        "pressure_full", "surface_pressure",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "radiation", "surface", "chemistry",
    )
    # Carry seeded as zeros by the base class. The first
    # ``compute_tendencies`` call overwrites every boundary field from
    # ``ForcingData`` at the top of the term loop, so the zero seed
    # never leaks into downstream physics.
    carry_slots: ClassVar[dict[str, type]] = {
        "radiation": RadiationData,
        "surface": SurfaceData,
        "chemistry": ChemistryData,
    }

    def __init__(
        self,
        ozone_peak_ppmv: float = 8.0,
        ozone_peak_height_m: float = 20_000.0,
        ozone_scale_height_m: float = 7_000.0,
        surface_optics: SurfaceOpticsParameters | None = None,
    ):
        """Hold the analytical-ozone profile parameters.

        Defaults broadly track Fortuin & Kelder 1998 zonal-mean
        climatology near the equator: ~8 ppmv peak at ~20 km height with
        a 7 km e-folding scale above (and a linear ramp from surface to
        peak below). Override per-Hydra to drive a different vertical
        profile.

        Args:
            ozone_peak_ppmv: Stratospheric peak ozone volume mixing
                ratio (ppmv).
            ozone_peak_height_m: Altitude of the ozone maximum (m).
            ozone_scale_height_m: e-folding height for decay above the
                peak (m).
            surface_optics: Surface albedo constants and per-tile
                emissivities; ECHAM 6.3 defaults when ``None``. Held in an
                ``nnx.Param`` so every value is a differentiable leaf
                (#347, #672).

        """
        # Store as plain Python floats; the ``ChemistryParameters``
        # struct (a ``tree_math.struct`` of JAX arrays) is built fresh
        # inside ``__call__`` so it stays a JIT-friendly closure value
        # rather than a stored pytree on this flax ``nnx`` term.
        self._ozone_peak_ppmv = float(ozone_peak_ppmv)
        self._ozone_peak_height_m = float(ozone_peak_height_m)
        self._ozone_scale_height_m = float(ozone_scale_height_m)
        self.surface_optics = nnx.Param(
            surface_optics or SurfaceOpticsParameters())

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (deg) for the ocean albedo's solar zenith.

        Same columns and flattening as the radiation terms' own cache, so the
        zenith angle the ocean albedo sees is the one the shortwave solve uses.
        """
        lat, lon = column_lat_lon(coords.horizontal)
        self._lats = nnx.Variable(lat * 180.0 / jnp.pi)
        self._lons = nnx.Variable(lon * 180.0 / jnp.pi)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Populate radiation, surface, and chemistry inputs."""
        nlev, ncols = state.temperature.shape
        if not hasattr(self, "_lats"):
            raise RuntimeError(
                "EchamBoundaryConditions needs cache_coords(coords) before it "
                "is called: the ocean albedo depends on the solar zenith "
                "angle of each column. ComposablePhysics.cache_coords does "
                "this for every term.")

        def col(x):
            """Grid (nlon, nlat) / (1, ncols) field -> column (ncols,)."""
            return jnp.asarray(x).reshape(ncols)

        def optional_col(x):
            # Bundles built before #672 carry no forest / glacier map; their
            # absence means "none", which is what ECHAM sees with a zero map.
            return 0.0 if x is None else col(x)

        land_fraction = col(terrain.fmask)
        sea_ice_fraction = col(forcing.sice_am)
        sst = col(forcing.sea_surface_temperature)
        land_temperature = col(forcing.stl_am)
        cos_zenith = current_cos_zenith(
            forcing.solar, self._lons.get_value(), self._lats.get_value(),
        ).reshape(ncols)

        albedo_vis, albedo_nir, emissivity = _surface_optical_properties(
            land_fraction, sea_ice_fraction, self.surface_optics.get_value(),
            background_albedo=col(forcing.alb0),
            # ``snowc_am`` is a cover fraction on the mirror bundles
            # (``jcm.data.mirror.bundles``); the clip guards the legacy
            # packaged files, whose ``snowc`` shares the variable name.
            snow_fraction=jnp.clip(col(forcing.snowc_am), 0.0, 1.0),
            land_temperature=land_temperature,
            ice_temperature=sea_ice_surface_temperature(
                sst, jnp.clip(sea_ice_fraction, 0.0, 1.0 - land_fraction)),
            cos_zenith=cos_zenith,
            forest_fraction=optional_col(forcing.forest_fraction),
            glacier_fraction=optional_col(forcing.glacier_fraction),
        )
        surface_temperature = jnp.where(
            land_fraction > 0.5, land_temperature, sst,
        )
        roughness_length = jnp.where(
            land_fraction > 0.5,
            0.01,    # 1 cm over land
            0.0001,  # 0.1 mm over ocean
        )

        # CH4 is PRESCRIBED: overwritten here from ``ForcingData`` (#347)
        # every step, so SimpleChemistry's OH-scaled decay survives only as
        # the ``methane_loss`` diagnostic — there is no evolving CH4 budget.
        # CO2 is not seeded here: radiation reads ``forcing.co2_vmr``
        # directly, so the chemistry diagnostic never carries it.
        ch4_vmr_value = forcing.ch4_vmr

        # O3: prefer the realistic CMIP6/ECHAM-style climatology carried
        # on ``forcing.ozone_climatology`` (loaded from a netCDF in
        # ``build_forcing``). Without that file, fall back to the
        # analytical Fortuin & Kelder-style surrogate driven by this
        # term's ``ozone_peak_*`` constructor kwargs. The analytical
        # profile is known to be a poor match for the real climatology
        # (peak in the wrong place, troposphere overestimated by ~50x,
        # mesopause underestimated by ~30x — see validation against
        # T63_ozone_picontrol.nc), so it should be treated as a
        # placeholder for unit tests / SCM where no climatology is
        # available.
        #
        # ``chemistry.ozone_vmr`` is consumed by RRTMGP as ppmv (a
        # ``* 1e-6`` converts to mole fraction inside that term).
        if forcing.ozone_climatology.is_loaded():
            # Pre-interpolated to the model's hybrid grid offline (see
            # ``jcm.data.bc.interpolate_ozone``) — straight slice, no
            # online vertical interp.
            ozone_vmr_ppmv = forcing.ozone_climatology.o3_ppmv
            # Enforce the OzoneClimatology contract here, at trace time,
            # rather than letting a violation travel. Unlike the oxidant
            # fields -- whose consumer reshapes to ``temperature.shape`` --
            # this array is handed to RRTMGP's ``lev_to_col`` as a plain
            # transpose, so an extra horizontal axis does not fail here: it
            # fails much later inside the radiation halo padder with a shape
            # error that names neither ozone nor the loader that produced it.
            # ``forcing`` reaches terms UNFLATTENED, so every loader owes the
            # flattened ``(nlev, ncols)`` layout the state already has.
            if ozone_vmr_ppmv.ndim != state.temperature.ndim:
                raise ValueError(
                    "forcing.ozone_climatology.o3_ppmv has "
                    f"{ozone_vmr_ppmv.ndim} dims {ozone_vmr_ppmv.shape}, but "
                    f"the physics view is {state.temperature.ndim}-D "
                    f"{state.temperature.shape}. OzoneClimatology must be "
                    "stored with the horizontal already flattened to the "
                    "term view -- (ntime, nlev, ncols) pre-select. Fix the "
                    "loader that built this forcing."
                )
        else:
            from jcm.physics.chemistry.simple_chemistry import (
                ChemistryParameters,
                fixed_ozone_distribution,
            )
            defaults = ChemistryParameters.default()
            ozone_params = ChemistryParameters(
                ozone_scale_height=jnp.asarray(self._ozone_scale_height_m),
                ozone_max_vmr=jnp.asarray(self._ozone_peak_ppmv),
                ozone_tropopause_height=jnp.asarray(self._ozone_peak_height_m),
                ozone_stratosphere_coeff=defaults.ozone_stratosphere_coeff,
                methane_surface_vmr=defaults.methane_surface_vmr,
                methane_lifetime=defaults.methane_lifetime,
                methane_oh_scaling=defaults.methane_oh_scaling,
            )
            ozone_vmr_ppmv = fixed_ozone_distribution(
                pressure=diagnostics["pressure_full"],
                surface_pressure=diagnostics["surface_pressure"],
                temperature=state.temperature,
                config=ozone_params,
            )

        # Start from whatever the previous step (or upstream term) left us
        # so we don't clobber radiation cache or other sub-struct fields.
        radiation = diagnostics.get(
            "radiation", RadiationData.zeros((ncols,), nlev),
        ).copy(
            surface_albedo_vis=albedo_vis,
            surface_albedo_nir=albedo_nir,
            surface_emissivity=emissivity,
        )
        surface = diagnostics.get(
            "surface", SurfaceData.zeros((ncols,), nlev),
        ).copy(
            surface_temperature=surface_temperature,
            skin_temperature=surface_temperature,
            roughness_length=roughness_length,
        )
        chemistry_zero = diagnostics.get(
            "chemistry", ChemistryData.zeros((ncols,), nlev),
        )
        chemistry = chemistry_zero.copy(
            methane_vmr=jnp.ones_like(chemistry_zero.methane_vmr)
            * ch4_vmr_value,
            ozone_vmr=ozone_vmr_ppmv,
        )

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {
            **diagnostics,
            "radiation": radiation,
            "surface": surface,
            "chemistry": chemistry,
        }
