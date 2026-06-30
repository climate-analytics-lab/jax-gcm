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


#: Reference wavelength for the column AOD diagnostic — 550 nm is the
#: standard visible wavelength for satellite (MODIS/MISR) and AERONET AOD,
#: so it is the natural observable to validate the scheme against.
_AOD_REF_NM = 550.0


@dataclasses.dataclass(frozen=True)
class _OpticsCache:
    """Per-band centers and refractive indices (static; nnx treats as a leaf)."""

    sw_nm: np.ndarray
    lw_nm: np.ndarray
    ri_sw: dict          # species -> (n[bands], k[bands])
    ri_lw: dict
    aod_band_idx: int    # SW band index whose centre is closest to 550 nm
    aod_band_nm: float   # that band's actual centre wavelength [nm]


class JamOpticsTerm(PhysicsTerm):
    """Online aerosol SW+LW optics written into the ``aerosol`` diagnostic."""

    name: ClassVar[str] = "jam_optics"
    category: ClassVar[str] = "aerosol_optics"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "aerosol", "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = ("aerosol", "aerosol_optical_depth")

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
        # SW band whose centre is closest to 550 nm — the band the column AOD
        # diagnostic reports (exact 550 nm with RRTMGP's banding; the single
        # broadband centre for grey radiation).
        if sw_nm.size:
            aod_idx = int(np.argmin(np.abs(sw_nm - _AOD_REF_NM)))
            aod_nm = float(sw_nm[aod_idx])
        else:
            aod_idx, aod_nm = 0, float("nan")
        self._cache = _OpticsCache(sw_nm, lw_nm, ri_sw, ri_lw, aod_idx, aod_nm)

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
            # Clamp the extinction-/scattering-weighted SSA and asymmetry to
            # their physical [0, 1] range. With a non-negative per-mode AOD
            # (number floored at 0 in ``__call__``) these ratios are already
            # bounded, but a tiny Mie-LUT edge overshoot in ``q_ext``/``ssa``
            # could still nudge them out of range, and RRTMGP's two-stream
            # solver NaNs on an SSA outside [0, 1] — so clamp defensively.
            return (
                aod,
                jnp.clip(scat / jnp.maximum(aod, _TINY), 0.0, 1.0),
                jnp.clip(gscat / jnp.maximum(scat, _TINY), 0.0, 1.0),
            )

        return jax.vmap(one_band)(lam_all, ri_j)

    def __call__(self, state, diagnostics, forcing, terrain):
        aer = diagnostics["_jam_state"]
        aerosol = diagnostics["aerosol"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        c = self._cache

        # Number per unit area [m^-2] per mode (number is kg^-1). Floor the
        # modal number at 0 before it enters the optics: spectral advection of
        # the aerosol-number tracers leaves small NEGATIVE number on the
        # near-zero cold-start field (Gibbs ringing). A negative number gives a
        # negative per-mode extinction, which can drive the band AOD ≤ 0 and
        # then the extinction-weighted SSA (= scat / AOD) and asymmetry to
        # ±huge — RRTMGP's two-stream solver NaNs on an out-of-range SSA. As the
        # aerosol burden grows the ringing crosses zero within the first day,
        # so this is a hard stability requirement, not a cosmetic floor. With
        # number ≥ 0 every derived optic is physical (AOD ≥ 0 ⇒ SSA, g ∈
        # [0, 1]); consistent with the AOD-550 diagnostic floor below.
        num_per_area = jnp.maximum(aer.number, 0.0) * (air_density * dz)[jnp.newaxis]

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

        # Column aerosol optical depth at ~550 nm: the total-column extinction
        # optical depth (sum of the per-layer band AOD over the vertical axis 0)
        # in the SW band closest to 550 nm. This is the standard satellite /
        # AERONET observable, so it is the cleanest external check on the scheme.
        # ``aod_sw`` is ``(n_band, nlev, *horiz)``; the column AOD is ``(*horiz)``.
        # Clamp at 0: optical depth is non-negative by definition, but spectral
        # transport leaves small negative aerosol number on the near-zero
        # cold-start field (Gibbs ringing), which can drive a tiny negative
        # per-layer extinction. The floor keeps the reported observable physical.
        if aod_sw.shape[0]:
            aod_550 = jnp.maximum(jnp.sum(aod_sw[c.aod_band_idx], axis=0), 0.0)
        else:
            aod_550 = jnp.zeros_like(state.temperature[0])

        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {
            **diagnostics,
            "aerosol": new_aerosol,
            "aerosol_optical_depth": aod_550,
        }
