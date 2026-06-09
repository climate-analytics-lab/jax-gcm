"""``JamOpticsTerm`` — online aerosol optics from the modal population.

Computes per-radiation-band aerosol optical depth, single-scattering albedo
and asymmetry parameter from the JAM population and writes them into the
``aerosol`` diagnostic, giving online aerosol a **direct radiative effect**
(#495). For each mode and band: form the volume-mixed complex refractive index
(dry species + aerosol water), the size parameter ``x = 2π r_wet/λ``, look up
the Mie efficiencies, and accumulate the extinction across modes;
single-scattering albedo and asymmetry are extinction-/scattering-weighted.

The expensive Mie evaluation is paid once at construction (``mie_lut``); the
per-step path is a differentiable table interpolation. SW band optics overwrite
MACv2-SP's fields (so the SW direct effect flows immediately); LW band optics
populate the new ``aerosol`` LW fields consumed by RRTMGP. MACv2-SP is kept
only for its indirect (Twomey/SPA) fields for now.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.optics.mie_lut import default_mie_lut, interp_mie
from jcm.physics.aerosol.jam.optics.refractive_index import refractive_index_at
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

_TINY = 1.0e-30
_FOUR_THIRDS_PI = 4.0 / 3.0 * math.pi


@dataclasses.dataclass(frozen=True)
class _OpticsCache:
    """Per-band centers and refractive indices (static; nnx treats as a leaf)."""

    sw_nm: np.ndarray
    lw_nm: np.ndarray
    ri_sw: dict          # species -> (n[bands], k[bands])
    ri_lw: dict


class JamOpticsTerm(PhysicsTerm):
    """Online aerosol SW+LW optics written into the ``aerosol`` diagnostic."""

    name: ClassVar[str] = "jam_optics"
    category: ClassVar[str] = "aerosol_optics"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "aerosol", "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = ("aerosol",)

    def __init__(self, *, spec: ModalAerosolSpec | None = None):
        """Build the Mie lookup table and hold the population."""
        self._spec = spec or MAM4_SPEC
        self._lut = default_mie_lut()
        self._cache = None   # set by cache_band_config

    def cache_band_config(self, band_config) -> None:
        """Precompute band centers and per-species refractive indices."""
        sw_nm = np.asarray(band_config.sw_band_centers_nm, np.float64)
        lw_nm = np.asarray(band_config.lw_band_centers_nm, np.float64)
        species = {sp for m in self._spec.modes for sp in m.species} | {"h2o"}
        ri_sw, ri_lw = {}, {}
        for sp in species:
            n_sw, k_sw = refractive_index_at(sp, jnp.asarray(sw_nm))
            n_lw, k_lw = refractive_index_at(sp, jnp.asarray(lw_nm))
            ri_sw[sp] = (np.asarray(n_sw), np.asarray(k_sw))
            ri_lw[sp] = (np.asarray(n_lw), np.asarray(k_lw))
        self._cache = _OpticsCache(sw_nm, lw_nm, ri_sw, ri_lw)

    def _band_optics(self, state, aer, num_per_area, centers_nm, ri):
        """Per-band ``(aod, ssa, asy)``, each ``(n_band, nlev, ncols)``.

        The bands are independent and share the whole modal geometry
        (volumes, wet radii, number), so the band axis is mapped with a
        single ``jax.vmap`` rather than a Python loop: only the wavelength
        and the per-species refractive index ``(n, k)`` vary across bands.
        The inner loops over modes/species stay explicit — they are ragged
        (each mode carries a different species set) and small.
        """
        n_band = centers_nm.shape[0]
        if n_band == 0:
            empty = jnp.zeros((0,) + state.temperature.shape)
            return empty, empty, empty

        zeros = jnp.zeros_like(state.temperature)
        lam_all = jnp.asarray(centers_nm, state.temperature.dtype) * 1.0e-9
        # ri: species -> (n[n_band], k[n_band]); vmap maps the band axis.
        ri_j = {sp: (jnp.asarray(n), jnp.asarray(k)) for sp, (n, k) in ri.items()}

        def one_band(lam_m, ri_band):
            aod = jnp.zeros_like(state.temperature)
            scat = jnp.zeros_like(state.temperature)
            gscat = jnp.zeros_like(state.temperature)
            for i, mode in enumerate(self._spec.modes):
                r_wet = aer.r_wet[i]
                vol_n = jnp.zeros_like(state.temperature)
                vol_k = jnp.zeros_like(state.temperature)
                vol_tot = jnp.zeros_like(state.temperature)
                for sp in mode.species:
                    mass = state.tracers.get(mass_name(sp, mode.short), zeros)
                    v = mass / self._spec.species_props(sp).density
                    n_sp, k_sp = ri_band[sp]
                    vol_n = vol_n + v * n_sp
                    vol_k = vol_k + v * k_sp
                    vol_tot = vol_tot + v
                v_water = aer.number[i] * _FOUR_THIRDS_PI * jnp.maximum(
                    r_wet ** 3 - aer.r_dry[i] ** 3, 0.0
                )
                n_w, k_w = ri_band["h2o"]
                vol_n = vol_n + v_water * n_w
                vol_k = vol_k + v_water * k_w
                vol_tot = vol_tot + v_water

                safe = jnp.maximum(vol_tot, _TINY)
                m_n = jnp.where(vol_tot > _TINY, vol_n / safe, 1.5)
                m_k = jnp.where(vol_tot > _TINY, vol_k / safe, 1.0e-8)
                x = 2.0 * math.pi * r_wet / lam_m
                q_ext, ssa, g = interp_mie(self._lut, x, m_n, m_k)
                aod_i = num_per_area[i] * q_ext * math.pi * r_wet ** 2
                aod = aod + aod_i
                scat = scat + ssa * aod_i
                gscat = gscat + g * ssa * aod_i
            return (
                aod,
                scat / jnp.maximum(aod, _TINY),
                gscat / jnp.maximum(scat, _TINY),
            )

        return jax.vmap(one_band)(lam_all, ri_j)

    def __call__(self, state, diagnostics, forcing, terrain):
        aer = diagnostics["_jam_state"]
        aerosol = diagnostics["aerosol"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        c = self._cache

        # Number per unit area [m^-2] per mode (number is kg^-1).
        num_per_area = aer.number * (air_density * dz)[jnp.newaxis]

        aod_sw, ssa_sw, asy_sw = self._band_optics(
            state, aer, num_per_area, c.sw_nm, c.ri_sw
        )
        aod_lw, ssa_lw, asy_lw = self._band_optics(
            state, aer, num_per_area, c.lw_nm, c.ri_lw
        )

        new_aerosol = aerosol.copy(
            aod_sw_per_band=aod_sw, ssa_sw_per_band=ssa_sw, asy_sw_per_band=asy_sw,
            aod_lw_per_band=aod_lw, ssa_lw_per_band=ssa_lw, asy_lw_per_band=asy_lw,
        )
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "aerosol": new_aerosol}
