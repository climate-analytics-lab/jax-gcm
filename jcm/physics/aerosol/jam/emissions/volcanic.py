"""``VolcanicEmissions`` — volcanic SO₂ from continuous degassing and eruptions (#1190).

Reads the point sources :func:`jcm.forcing.read_volcanic_emissions` mapped onto
model columns (``forcing.volcanic_emissions``) and emits them over the column's
own layer heights (``height_half``, geopotential height a.s.l.):

* **Continuous degassing** (CMIP6/CESM2 ``contvolcano``) carries its own
  profile on fixed altitude layers. Each inventory layer is split over the
  model layers it overlaps; the part below the model's (smoothed) surface goes
  into the lowest layer and any part above the lid into the top one, so every
  source's mass is emitted. The product holds the inventory's total sulfur as
  SO₂; the share ``so4_primary_fraction`` (2.5 %, the share CESM's files carry
  as SO₄ and HAM's ``zfacso2``) is emitted as primary sulfate, the rest as
  ``g_so2``.
* **Eruptions** (CMIP7 ``utsvolcemis``) follow the Aubry et al. (2026, §V.2)
  recommendation: the SO₂ is injected over the eruption's UTC day, in the
  column holding the volcano, with the profile ``exp(−((z − H)/D)²)`` around
  the injection altitude ``H`` a.s.l. (``D`` the inventory's depth). The
  profile is integrated exactly over each layer (``erf``), with the same
  below-surface/above-lid rule, so a plume thin against the layers keeps its
  mass and its height derivatives. Eruptions emit SO₂ only.

Primary sulfate goes where the population puts it: MAM4's ``primary_split``
(50 % Aitken, 50 % accumulation, the mass split CESM gives the same
inventory's sulfate; the number follows each mode's own size), and for M7 HAM's
default sector class (``"fossil"``: 50 % soluble
Aitken at 30 nm, 50 % soluble accumulation at 75 nm), which every sector HAM's
emission routine does not special-case takes.

Eruptions are selected each step from ``forcing.solar.calendar_year`` and
``day_of_year``, so no per-day forcing field is stored. All three numbers in
:class:`VolcanicParameters` are differentiable.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import tree_math
from flax import nnx
from jax.scipy.special import erf

from jcm.physics.aerosol.jam.emissions.distributors import particle_mean_mass
from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
    accumulate_emission_fluxes, emission_flux_keys)
from jcm.physics.aerosol.jam.emissions.ham_sectors import (
    cmr_to_emission_diameter)
from jcm.physics.aerosol.jam.emissions.sectors import SO4_PRIMARY_FRACTION
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name, mass_name, number_name)
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

#: HAM sector class whose primary-sulfate targets volcanic SO₂ takes on M7.
HAM_VOLCANIC_CLASS = "fossil"


@tree_math.struct
class VolcanicParameters:
    """Differentiable volcanic-emission knobs."""

    degassing_scale: jnp.ndarray       # multiplier on the degassing inventory
    eruption_scale: jnp.ndarray        # multiplier on the eruption inventory
    so4_primary_fraction: jnp.ndarray  # degassing SO₂ emitted as primary SO₄

    @classmethod
    def default(cls) -> "VolcanicParameters":
        return cls(degassing_scale=jnp.asarray(1.0),
                   eruption_scale=jnp.asarray(1.0),
                   so4_primary_fraction=jnp.asarray(SO4_PRIMARY_FRACTION))


def layer_overlap_fractions(z_half, source_interface):
    """Share of each source layer falling in each model layer, ``(nlev, nsrc, ncol)``.

    ``z_half`` is ``(nlev+1, ncol)`` model interface heights, top first;
    ``source_interface`` the ``(nsrc+1,)`` source layer interfaces, bottom
    first, in the same height coordinate. Shares below the model surface go to
    the lowest layer and above the lid to the top one, so each source layer's
    shares sum to 1.
    """
    top, bot = z_half[:-1, None, :], z_half[1:, None, :]
    lo = source_interface[None, :-1, None]
    hi = source_interface[None, 1:, None]
    thick = hi - lo
    frac = jnp.clip(jnp.minimum(top, hi) - jnp.maximum(bot, lo), 0.0) / thick
    surface, lid = z_half[-1][None, :], z_half[0][None, :]
    lo2, hi2, thick2 = lo[0], hi[0], thick[0]
    below = jnp.clip(jnp.minimum(surface, hi2) - lo2, 0.0) / thick2
    above = jnp.clip(hi2 - jnp.maximum(lid, lo2), 0.0) / thick2
    return frac.at[-1].add(below).at[0].add(above)


def gaussian_layer_fractions(z_half, center, depth):
    """Exact layer integrals of ``exp(−((z − center)/depth)²)``, ``(nlev, ncol)``.

    ``z_half`` is ``(nlev+1, ncol)`` interface heights, top first; ``center``
    and ``depth`` are ``(ncol,)``. The profile's tails below the surface and
    above the lid are added to the lowest and top layers, so the fractions sum
    to 1 exactly however thin the plume is against the layers.
    """
    cdf = 0.5 * erf((z_half - center[None, :]) / depth[None, :])
    frac = cdf[:-1] - cdf[1:]
    return frac.at[-1].add(cdf[-1] + 0.5).at[0].add(0.5 - cdf[0])


class VolcanicEmissions(PhysicsTerm):
    """Volcanic SO₂: continuous degassing plus daily eruption injections."""

    name: ClassVar[str] = "jam_volcanic_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = (
        "air_density", "layer_thickness", "height_half")
    provides: ClassVar[tuple[str, ...]] = emission_flux_keys()

    def __init__(self, params: VolcanicParameters | None = None, *,
                 spec: ModalAerosolSpec | None = None):
        """Hold the (differentiable) parameters and the population."""
        self.params = nnx.Param(params or VolcanicParameters.default())
        self._spec = spec or MAM4_SPEC
        self._so2_to_so4_mass = (self._spec.species_props("so4").molar_mass
                                 / GAS_SPECIES["so2"].molar_mass)

    def validate_forcing(self, forcing, run_window=None) -> None:
        """Warn when the term is composed but the run has no volcanic sources."""
        # ``warnings`` rather than a log line: a chunked run validates every
        # chunk, and the warning filter shows it once.
        if forcing is None:
            return
        volc = getattr(forcing, "volcanic_emissions", None)
        if volc is None:
            warnings.warn(
                "VolcanicEmissions is composed but forcing.volcanic_emissions "
                "is unset, so the run has no volcanic SO2 (#1190): set "
                "forcing.volcanic_file (the JAM presets use 'auto').",
                stacklevel=2)
            return
        span = volc.get("eruption_span") if run_window is not None else None
        if span is not None:
            first, last = (float(x) for x in np.asarray(span))
            if run_window[0] < first or run_window[1] > last:
                day = lambda t: np.datetime64(int(t), "s").astype("datetime64[D]")  # noqa: E731
                warnings.warn(
                    f"The volcanic eruption list covers {day(first)} to "
                    f"{day(last)}; outside it the run emits volcanic degassing "
                    "only. forcing.volcanic_eruption_year replays one listed "
                    "year instead.", stacklevel=2)

    def _sulfate_targets(self):
        """``(mode, mass_fraction, mass per particle)`` for primary SO₄."""
        density = self._spec.species_props("so4").density
        policy = self._spec.sector_emission
        if policy is None:
            return tuple((mode, frac, density / mode.number_factor)
                         for mode, frac in self._spec.primary_split("so4"))
        out = []
        for target in policy.targets[HAM_VOLCANIC_CLASS]["so4"]:
            mode = self._spec.mode(target.mode)
            diameter = cmr_to_emission_diameter(target.cmr_m, mode.geom_std_dev)
            out.append((mode, target.mass_fraction,
                        particle_mean_mass(mode, density,
                                           emission_diameter=diameter)))
        return tuple(out)

    def __call__(self, state, diagnostics, forcing, terrain):
        shape = state.temperature.shape
        nlev = shape[0]
        rho = diagnostics["air_density"].reshape(nlev, -1)
        dz = diagnostics["layer_thickness"].reshape(nlev, -1)
        z_half = diagnostics["height_half"].reshape(nlev + 1, -1)
        dtype = state.temperature.dtype
        tends: dict[str, jnp.ndarray] = {}

        volc = (getattr(forcing, "volcanic_emissions", None)
                if forcing is not None else None)
        if volc is not None:
            p = self.params.get_value()
            zeros = jnp.zeros(rho.shape)

            # Degassing: per-model-layer SO2 flux [kg m-2 s-1] in its columns.
            d_cell = volc["degassing_cell"]
            share = layer_overlap_fractions(
                z_half[:, d_cell], volc["degassing_altitude_interface"])
            degas = p.degassing_scale * jnp.einsum(
                "ksc,cs->kc", share, volc["degassing_flux"])
            degas = zeros.at[:, d_cell].add(degas)

            # Eruptions active on this UTC day.
            solar = forcing.solar
            active = ((volc["eruption_day"] == solar.day_of_year)
                      & ((volc["eruption_year"] < 0)
                         | (volc["eruption_year"] == solar.calendar_year)))
            e_cell = volc["eruption_cell"]
            profile = gaussian_layer_fractions(
                z_half[:, e_cell], volc["eruption_height"],
                volc["eruption_depth"])
            rate = p.eruption_scale * jnp.where(active, volc["eruption_flux"], 0.0)
            erupt = zeros.at[:, e_cell].add(profile * rate[None, :])

            frac = p.so4_primary_fraction
            inv = 1.0 / (rho * dz)
            tends[gas_name("so2")] = ((1.0 - frac) * degas + erupt) * inv
            so4 = frac * degas * self._so2_to_so4_mass * inv
            for mode, mass_fraction, particle_mass in self._sulfate_targets():
                for name, value in (
                        (mass_name("so4", mode.short), so4 * mass_fraction),
                        (number_name(mode.short),
                         so4 * mass_fraction / particle_mass)):
                    tends[name] = tends.get(name, zeros) + value
            # Pinned to the state's dtype: the parameters are float64 under
            # x64 while pySES runs physics in float32 (#770).
            tends = {k: v.astype(dtype).reshape(shape) for k, v in tends.items()}

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tends,
        )
        diagnostics = accumulate_emission_fluxes(
            diagnostics, tends, diagnostics["air_density"],
            diagnostics["layer_thickness"])
        return tendency, diagnostics
