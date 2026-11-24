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
    # The sulfate tracer represents ammonium bisulfate (115 g/mol; see
    # species.py and chemistry/aqueous.py), not fully neutralised ammonium
    # sulfate. Li et al. (2001), JAS 58, 193–209, use n=1.473 for NH4HSO4:
    # https://doi.org/10.1175/1520-0469(2001)058<0193:POTOPO>2.0.CO;2
    # Treat the transparent 0.3–1 µm region as nondispersive at this
    # representative value; these anchors are not a measured spectrum.
    # Water dilution is applied separately by the modal volume mixing rule,
    # so these must be DRY indices, not those of an aqueous sulfate solution.
    # Sulfate remains absorbing in the LW (ν3 band near 9 µm).
    "so4": ([1.473, 1.473, 1.473, 1.39, 1.85, 1.90],
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
    # HAM's "OC" token (M7's organic-carbon species, distinct from MAM4's
    # poa/soa — see microphysics/m7_data.py): mo_ham_rad_data.f90's OC
    # table (``iradoc``), evaluated at jcm's six anchor wavelengths by
    # finding the RRTM SW/LW band whose wavelength interval CONTAINS each
    # anchor. Band edges are the RRTM wavenumber tables (``wavenum1``/
    # ``wavenum2``; SW: mo_srtm_setup.f90:93-97, LW: mo_lrtm_setup.f90:95-99
    # — lambda[um] = 1e4/wavenumber[cm-1]):
    #   0.30 um -> SW band 27 (wavenumber 29000-38000 -> 0.263-0.345 um,
    #     the 12th SW array slot, mid-wavelength 0.30 um in the
    #     mo_ham_rad_data.f90:194 comment): cnr/cni(1:14,iradoc) line
    #     243/248, index 12 -> n=1.443, k=1.63e-2.
    #   0.55 um -> the dedicated 550 nm optional-wavelength slot
    #     (lambda_sw_opt(1), exact match rather than an interval pick):
    #     cnr/cni(15:16,iradoc) line 255/258, index 1 -> n=1.53, k=5.50e-3.
    #   1.0 um -> SW band 23 (8050-12850 -> 0.778-1.242 um, mid-wavelength
    #     1.01 um, SW array index 8): n=1.420, k=2.01e-2.
    #   3.0 um -> SW band 17 (3250-4000 -> 2.5-3.077 um, mid-wavelength
    #     2.79 um, SW array index 2): n=1.510, k=7.33e-3.
    #   10.0 um -> LW band 7 (980-1080 -> 9.259-10.204 um): cnr/cni
    #     (17:32,iradoc) line 362/367, LW-array index 7 -> n=1.81, k=4.54e-2.
    #   25.0 um -> LW band 2 (350-500 -> 20-28.57 um): LW-array index 2 ->
    #     n=1.95, k=2.35e-1.
    "oc": ([1.443, 1.53, 1.420, 1.510, 1.81, 1.95],
           [1.63e-2, 5.50e-3, 2.01e-2, 7.33e-3, 4.54e-2, 2.35e-1]),
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
