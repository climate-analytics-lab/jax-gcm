"""Wind-erosion dust emission (Tegen et al. 2002 flux physics).

The full HAMMOZ BGC/Tegen scheme (``mo_ham_dust``) carries a 17-soil-type
database, soil size distributions and preferential-source maps to derive a
per-gridcell threshold friction velocity and sandblasting efficiency. That
soil pre-processing produces, in effect, a **dust source/erodibility field**;
here we read that field from forcing and apply the Tegen *emission physics*
to it — the saltation horizontal flux and its sandblasting conversion to a
vertical (emitted) flux, gated by friction velocity exceeding the threshold:

    G = scale · (ρ_air/g) · u*³ · (1 + u*t/u*) · (1 − (u*t/u*)²)   (u* > u*t)
    F = source · mobilization · α · G

with ``u*`` the surface friction velocity (from the ``vertical_diffusion``
diagnostic), ``u*t`` the threshold and ``α`` the sandblasting efficiency. The
same ``(1−r)(1+r)²`` White (1979) form is what CLM's ``DustEmisZender2003``
uses, so the surface gating below follows CLM/CAM directly:

* ``source`` — the prescribed erodibility from ``forcing.dust_source``. For
  CAM's geomorphic basin factor (``mbl_bsn_fct_geo``, 0–5.7) CAM zeroes cells
  below ``soil_erod_threshold = 0.1`` and applies **no upper bound**
  (``dust_model.F90``); a Tegen/HAMMOZ potential-source map is instead a
  fraction in [0, 1] whose own preprocessing already embeds the land-cover
  mask. ``source_kind`` selects which convention the field follows.
* ``mobilization`` — CLM ``lnd_frc_mbl``: the land fraction that is snow-free
  and unfrozen. CLM's vegetation term ``1 − VAI/0.3`` has no counterpart here
  because no LAI/vegetation boundary field is carried (#777); a vegetation-
  masked monthly source map is the ``tegen_potential`` alternative.
* ``u*t`` — raised over moist soil by the Fecan et al. (1999) factor CLM
  applies in ``frc_thr_wet_fct``.

Emitted mass is split accum/coarse with CAM's ``dust_emis_sclfctr`` and
converted to number at the emission bins' volume-mean diameters
(``dust_common::dust_set_params``), not at the modes' equilibrium sizes.

References: Tegen et al. (2002), JGR 107; Marticorena & Bergametti (1995);
Zender et al. (2003); Fecan et al. (1999).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

import jcm.constants as c
from jcm.constants import grav as _G
from jcm.physics.aerosol.jam.emissions.distributors import distribute_surface_flux
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm
from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
    accumulate_emission_fluxes, emission_flux_keys)

#: Source-field conventions ``forcing.dust_source`` may follow.
SOURCE_KINDS = ("cam_erodibility", "tegen_potential")

#: Gravimetric soil-water content at saturation [kg/kg], used to read the
#: model's relative soil wetness in the units Fecan's threshold is defined in
#: (porosity 0.45 over a 1485 kg/m³ dry bulk density, CLM's ``watsat``/``bd``).
_GWC_SATURATED = 0.30

#: Width of the frozen-ground ramp below the melting point [K]. CLM uses the
#: top soil layer's liquid fraction; with only a land surface temperature
#: available the liquid fraction is ramped over the last 2 K.
_FREEZE_RANGE = 2.0


@tree_math.struct
class DustParameters:
    """Calibratable knobs for the Tegen wind-erosion flux."""

    scale: jnp.ndarray             # overall horizontal-flux scale
    alpha: jnp.ndarray             # sandblasting efficiency [1/m]
    u_threshold: jnp.ndarray       # dry threshold friction velocity [m/s]
    accum_fraction: jnp.ndarray    # fraction of emitted dust into accum (rest coarse)
    u_star_default: jnp.ndarray    # fallback friction velocity [m/s]
    source_threshold: jnp.ndarray  # erodibility below this emits nothing [-]
    soil_moisture_threshold: jnp.ndarray  # Fecan gravimetric threshold [kg/kg]
    emission_diameter: jnp.ndarray  # (accum, coarse) volume-mean diameter [m]

    @classmethod
    def default(cls) -> "DustParameters":
        return cls(
            scale=jnp.asarray(1.0),
            alpha=jnp.asarray(1.0e-5),
            u_threshold=jnp.asarray(0.2),
            # CAM ``dust_emis_sclfctr`` for MAM4 over the (0.1–1, 1–10) µm
            # emission bins; its 1.65e-5 Aitken share has no home in a
            # population whose Aitken mode carries no dust.
            accum_fraction=jnp.asarray(0.021),
            u_star_default=jnp.asarray(0.3),
            source_threshold=jnp.asarray(0.1),
            # Fecan gwc_thr for ~20 % clay soil (CLM derives it per gridcell
            # from clay content, which no boundary field here carries).
            soil_moisture_threshold=jnp.asarray(0.04),
            # ``dust_common::dust_set_params`` mass-weighted diameters of the
            # emitted lognormal (D_vma = 3.5 µm, σ = 2) over those two bins.
            emission_diameter=jnp.asarray([0.7806e-6, 3.8983e-6]),
        )


def horizontal_flux(
    u_star: jnp.ndarray, u_threshold: jnp.ndarray, air_density: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """Tegen saltation horizontal flux G [kg/m/s] (zero below threshold)."""
    u = jnp.maximum(u_star, 1.0e-3)
    ratio = u_threshold / u
    g_flux = scale * (air_density / _G) * u ** 3 * (1.0 + ratio) * (1.0 - ratio ** 2)
    return jnp.where(u_star > u_threshold, g_flux, 0.0)


def source_weight(source: jnp.ndarray, source_kind: str,
                  threshold: jnp.ndarray) -> jnp.ndarray:
    """Prescribed source field → dimensionless erodibility weight.

    ``cam_erodibility``: CAM's geomorphic basin factor, zeroed below
    ``soil_erod_threshold`` and **unbounded above** (``dust_model.F90``) — the
    factor exceeds 1 in the large closed basins that are the strongest dust
    sources. ``tegen_potential``: a potential-source *fraction*, [0, 1].
    """
    s = jnp.maximum(source, 0.0)
    if source_kind == "cam_erodibility":
        return jnp.where(s < threshold, 0.0, s)
    return jnp.minimum(s, 1.0)


def mobilization_fraction(land_fraction: jnp.ndarray, snow_cover: jnp.ndarray,
                          land_temperature: jnp.ndarray) -> jnp.ndarray:
    """Fraction of a gridcell that can mobilize dust (CLM ``lnd_frc_mbl``).

    Land only, reduced by snow cover, and shut off over frozen ground (CLM's
    ``liqfrac``, here ramped over the 2 K below the melting point since no soil
    ice content is carried). Ocean and sea ice contribute nothing.

    The frozen-ground term is what masks the ice sheets: ``snowc_am`` is zeroed
    on permanent snow by construction (their albedo lives in ``alb`` instead),
    so a snow gate alone would emit dust from Antarctica.
    """
    liquid = jnp.clip(
        (land_temperature - (c.tmelt - _FREEZE_RANGE)) / _FREEZE_RANGE, 0.0, 1.0)
    return (jnp.clip(land_fraction, 0.0, 1.0)
            * (1.0 - jnp.clip(snow_cover, 0.0, 1.0)) * liquid)


def wet_threshold_factor(soil_wetness: jnp.ndarray,
                         gwc_threshold: jnp.ndarray) -> jnp.ndarray:
    """Fecan (1999) moist-soil increase of u*t (CLM ``frc_thr_wet_fct``).

    ``√(1 + 1.21·(100·(w − w_thr))^0.68)`` above the threshold gravimetric
    water content, 1 below it. The model carries a relative soil wetness, so
    ``w = wetness · _GWC_SATURATED``.
    """
    gwc = jnp.clip(soil_wetness, 0.0, 1.0) * _GWC_SATURATED
    excess = gwc - gwc_threshold
    # Guarded operand: x**0.68 has an infinite derivative at x = 0, which the
    # ``where`` masks in value but not in the reverse pass.
    safe = jnp.maximum(excess, 1.0e-12)
    return jnp.where(excess > 0.0,
                     jnp.sqrt(1.0 + 1.21 * (100.0 * safe) ** 0.68), 1.0)


class DustEmissions(PhysicsTerm):
    """Wind-erosion dust emission (Tegen flux × gated prescribed source)."""

    name: ClassVar[str] = "jam_dust_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = emission_flux_keys()

    def __init__(
        self,
        params: DustParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
        source_kind: str = "cam_erodibility",
    ):
        """Hold params, the population and the source-field convention."""
        if source_kind not in SOURCE_KINDS:
            raise ValueError(
                f"Unknown dust source_kind {source_kind!r}; "
                f"choose one of {list(SOURCE_KINDS)}."
            )
        self.params = nnx.Param(params or DustParameters.default())
        self._spec = spec or MAM4_SPEC
        self._source_kind = source_kind

    def _u_star(self, diagnostics, ncols, params):
        if "vertical_diffusion" in diagnostics:
            return diagnostics["vertical_diffusion"].surface_friction_velocity
        return jnp.full((ncols,), params.u_star_default)

    @staticmethod
    def _surface_field(obj, name, ncols, default):
        """Read a per-column surface field from forcing/terrain, else ``default``."""
        v = getattr(obj, name, None) if obj is not None else None
        if v is not None and jnp.size(v) == ncols:
            return jnp.ravel(v)
        return jnp.full((ncols,), default)

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape

        source = source_weight(
            self._surface_field(forcing, "dust_source", ncols, 0.0),
            self._source_kind, p.source_threshold)
        # Land, snow-free, unfrozen fraction. Absent boundary fields default to
        # bare warm land so a source map alone still emits (aquaplanet runs
        # supply no source and stay inert either way).
        mobilization = mobilization_fraction(
            self._surface_field(terrain, "fmask", ncols, 1.0),
            self._surface_field(forcing, "snowc_am", ncols, 0.0),
            self._surface_field(forcing, "stl_am", ncols, c.tmelt + 15.0))
        u_threshold = p.u_threshold * wet_threshold_factor(
            self._surface_field(forcing, "soilw_am", ncols, 0.0),
            p.soil_moisture_threshold)

        u_star = self._u_star(diagnostics, ncols, p)
        g_flux = horizontal_flux(u_star, u_threshold, air_density[-1], p.scale)
        flux = source * mobilization * p.alpha * g_flux        # kg/m²/s

        # The population owns *which* classes receive primary dust (and their
        # default split); the tunable ``accum_fraction`` overrides the fraction
        # for this 2-class fine/coarse scheme. No literal mode names here.
        # Number comes from each bin's emission volume-mean diameter, CAM's
        # ``x_mton``, not from the mode's equilibrium size.
        (fine, _), (coarse, _) = self._spec.primary_split("du")
        fluxes = [
            ("du", fine.short, flux * p.accum_fraction, p.emission_diameter[0]),
            ("du", coarse.short, flux * (1.0 - p.accum_fraction),
             p.emission_diameter[1]),
        ]
        tracer_tends = distribute_surface_flux(self._spec, fluxes, air_density, dz)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        # Publish this term's contribution to the AeroCom per-species
        # emission fluxes (accumulated across all emitting terms).
        diagnostics = accumulate_emission_fluxes(
            diagnostics, tracer_tends,
            diagnostics["air_density"],
            diagnostics["layer_thickness"])

        return tendency, diagnostics
