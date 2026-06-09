"""Per-species complex refractive indices vs wavelength.

Representative complex refractive indices ``m = n − i·k`` (radiative
convention: positive ``k`` is absorbing) for the JAM species, tabulated at a
handful of anchor wavelengths spanning the shortwave and longwave and
interpolated in ``log10(λ)`` (constant extrapolation outside the anchors).

These are **first-cut representative values** drawn from the standard
literature — OPAC / Hess et al. (1998), Stier et al. (2005), Sokolik & Toon
(1999) for dust, Hale & Querry (1973) for water — not full spectral tables.
They capture the dominant contrasts (BC strongly absorbing and ~grey; sulfate/
sea-salt transparent in the SW and absorbing in the LW; dust moderately
absorbing with a strong LW silicate feature; organics weakly absorbing). A
band-resolved spectral upgrade is a follow-up.

``refractive_index_at(species, wavelength_nm)`` returns ``(n, k)`` arrays.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Anchor wavelengths (µm).
_LAM_UM = np.array([0.30, 0.55, 1.0, 3.0, 10.0, 25.0])

# token -> (n[anchors], k[anchors]). Same length as _LAM_UM.
_RI: dict[str, tuple[list[float], list[float]]] = {
    # sulfate: transparent SW, absorbing LW (sulfate ν3 band near 9 µm).
    "so4": ([1.45, 1.43, 1.42, 1.39, 1.85, 1.90],
            [1e-8, 1e-8, 1e-6, 1.6e-2, 0.46, 0.20]),
    "nh4": ([1.52, 1.50, 1.48, 1.40, 1.80, 1.85],
            [1e-7, 1e-7, 1e-4, 2e-2, 0.40, 0.20]),
    "no3": ([1.55, 1.53, 1.50, 1.42, 1.75, 1.80],
            [1e-6, 1e-6, 1e-4, 3e-2, 0.30, 0.20]),
    # black carbon: strongly absorbing, weakly dispersive.
    "bc": ([1.80, 1.85, 1.90, 2.10, 2.40, 2.50],
           [0.66, 0.71, 0.79, 0.93, 1.00, 1.00]),
    # primary / secondary / marine organics: weakly absorbing.
    "poa": ([1.55, 1.53, 1.52, 1.48, 1.55, 1.60],
            [3e-2, 6e-3, 5e-3, 2e-2, 0.10, 0.12]),
    "soa": ([1.50, 1.49, 1.48, 1.46, 1.52, 1.56],
            [5e-3, 2e-3, 2e-3, 1.5e-2, 0.09, 0.11]),
    "moa": ([1.53, 1.52, 1.51, 1.47, 1.53, 1.58],
            [1e-2, 5e-3, 5e-3, 2e-2, 0.10, 0.12]),
    # sea salt: transparent SW, LW bands.
    "ss": ([1.51, 1.50, 1.49, 1.48, 1.40, 1.60],
           [1e-8, 1e-8, 1e-6, 1e-3, 5e-2, 0.10]),
    # dust: moderately absorbing SW, strong LW silicate (~9–10 µm) feature.
    "du": ([1.56, 1.53, 1.52, 1.50, 1.90, 2.10],
           [3e-2, 3e-3, 1e-3, 5e-3, 0.40, 0.30]),
    # aerosol water (condensed phase).
    "h2o": ([1.35, 1.33, 1.32, 1.42, 1.22, 1.50],
            [1e-8, 1e-9, 1e-6, 1e-2, 5e-2, 0.40]),
}

_LOG_LAM = np.log10(_LAM_UM)


def refractive_index_at(species: str, wavelength_nm):
    """Interpolated ``(n, k)`` for ``species`` at ``wavelength_nm`` (array).

    ``wavelength_nm`` may be any shape; ``n``/``k`` are returned with that
    shape. Interpolation is linear in ``log10(λ)`` with constant ends.
    """
    n_anchor, k_anchor = _RI[species]
    log_lam = jnp.log10(jnp.asarray(wavelength_nm) / 1000.0)  # nm → µm → log10
    xp = jnp.asarray(_LOG_LAM)
    n = jnp.interp(log_lam, xp, jnp.asarray(n_anchor))
    k = jnp.interp(log_lam, xp, jnp.asarray(k_anchor))
    return n, k
