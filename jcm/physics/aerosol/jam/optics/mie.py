"""Bohren & Huffman (1983) Mie kernel (NumPy).

Computes the extinction efficiency, single-scattering albedo and asymmetry
parameter of a homogeneous sphere of size parameter ``x = 2πr/λ`` and complex
refractive index ``m = mr + i·mi``.

This is deliberately **NumPy, not JAX**: it is only ever evaluated to *build a
lookup table once* at term construction (``mie_lut``); the per-step optics
interpolate that table differentiably. Keeping it off the JAX path avoids
forcing the whole model into float64 (the recurrences need float64 precision).
Validated to match an independent scipy-Bessel Mie implementation to 1e-4
across Rayleigh, resonance, large-x and strongly absorbing regimes.

Algorithm: Bohren & Huffman, *Absorption and Scattering of Light by Small
Particles* (1983), appendix ``BHMIE``.
"""

from __future__ import annotations

import numpy as np

X_MAX = 100.0   # size parameter is clipped here; large-x optics asymptote


def mie_efficiencies(x: float, mr: float, mi: float) -> tuple[float, float, float]:
    """Return ``(q_ext, ssa, g)`` for size parameter ``x`` and index ``mr+i·mi``."""
    x = float(np.clip(x, 1.0e-6, X_MAX))
    m = complex(mr, mi)
    y = m * x

    nmax = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 2
    nmx = max(nmax, int(abs(y))) + 16

    # Logarithmic derivative D_n(y), downward recurrence (stable).
    d = np.zeros(nmx + 1, dtype=complex)
    for k in range(nmx, 0, -1):
        d[k - 1] = k / y - 1.0 / (d[k] + k / y)
    d_n = d[1:nmax + 1]                                   # D_1 … D_{nmax}

    n = np.arange(1, nmax + 1)

    # Riccati–Bessel ψ_n(x), χ_n(x), upward recurrence; ψ_{-1}=cos, ψ_0=sin.
    psi = np.zeros(nmax + 1)
    chi = np.zeros(nmax + 1)
    psi[0], chi[0] = np.sin(x), np.cos(x)
    psi_prev, chi_prev = np.cos(x), -np.sin(x)            # ψ_{-1}, χ_{-1}
    for nn in range(1, nmax + 1):
        psi[nn] = (2 * nn - 1) / x * psi[nn - 1] - psi_prev
        chi[nn] = (2 * nn - 1) / x * chi[nn - 1] - chi_prev
        psi_prev, chi_prev = psi[nn - 1], chi[nn - 1]

    psi_n, psi_nm1 = psi[1:], psi[:-1]
    chi_n, chi_nm1 = chi[1:], chi[:-1]
    xi_n = psi_n - 1j * chi_n
    xi_nm1 = psi_nm1 - 1j * chi_nm1

    fac_a = d_n / m + n / x
    a_n = (fac_a * psi_n - psi_nm1) / (fac_a * xi_n - xi_nm1)
    fac_b = d_n * m + n / x
    b_n = (fac_b * psi_n - psi_nm1) / (fac_b * xi_n - xi_nm1)

    tn = 2 * n + 1
    q_ext = (2.0 / x ** 2) * np.sum(tn * np.real(a_n + b_n))
    q_sca = (2.0 / x ** 2) * np.sum(tn * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2))

    a_np1 = np.append(a_n[1:], 0.0)
    b_np1 = np.append(b_n[1:], 0.0)
    g_qsca = (4.0 / x ** 2) * (
        np.sum(n * (n + 2) / (n + 1) * np.real(a_n * np.conj(a_np1) + b_n * np.conj(b_np1)))
        + np.sum(tn / (n * (n + 1)) * np.real(a_n * np.conj(b_n)))
    )

    q_sca = max(q_sca, 1.0e-30)
    q_ext = max(q_ext, q_sca)
    return float(q_ext), float(min(max(q_sca / q_ext, 0.0), 1.0)), \
        float(min(max(g_qsca / q_sca, -1.0), 1.0))


def mie_efficiencies_grid(x: float, mr: np.ndarray, mi: np.ndarray):
    """Batched ``mie_efficiencies`` over a (real RI, imag RI) grid at one

    fixed scalar size parameter ``x``. Same Bohren-Huffman recurrences as
    :func:`mie_efficiencies`, broadcast over ``mr``/``mi`` (same shape)
    instead of evaluated one triple at a time, because the per-species-
    pair Mie table a HAM-style lookup needs (``ham_mie_tables.py``: up to
    101 x 201 real/imag-RI points, at every size-parameter node of a
    lognormal quadrature) is ~1e4-1e5x too many scalar calls to finish in
    anything short of hours. ``psi``/``chi`` depend on ``x`` alone, so they
    are computed once and broadcast; the downward ``D_n`` recurrence is
    elementwise in ``y = m*x``, which varies over the grid, and is run to a
    single ``nmx`` covering the whole grid's worst case -- downward
    recurrence for the logarithmic derivative is unconditionally stable
    however far above the eventually-needed order it starts, so this is
    exact, not an approximation, for every grid point that needed fewer
    terms. Matches :func:`mie_efficiencies` to float64 precision (see
    ``mie_test.py``).

    Returns ``(q_ext, ssa, g)``, each with ``mr``'s shape.
    """
    x = float(np.clip(x, 1.0e-6, X_MAX))
    mr = np.asarray(mr, dtype=np.float64)
    mi = np.asarray(mi, dtype=np.float64)
    m = mr + 1j * mi
    y = m * x

    nmax = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 2
    nmx = max(nmax, int(np.max(np.abs(y)))) + 16

    d = np.zeros((nmx + 1,) + m.shape, dtype=complex)
    for k in range(nmx, 0, -1):
        d[k - 1] = k / y - 1.0 / (d[k] + k / y)
    d_n = d[1:nmax + 1]                                    # (nmax, *grid)

    pad = (1,) * m.ndim
    n = np.arange(1, nmax + 1).reshape((-1,) + pad)

    psi = np.zeros(nmax + 1)
    chi = np.zeros(nmax + 1)
    psi[0], chi[0] = np.sin(x), np.cos(x)
    psi_prev, chi_prev = np.cos(x), -np.sin(x)
    for nn in range(1, nmax + 1):
        psi[nn] = (2 * nn - 1) / x * psi[nn - 1] - psi_prev
        chi[nn] = (2 * nn - 1) / x * chi[nn - 1] - chi_prev
        psi_prev, chi_prev = psi[nn - 1], chi[nn - 1]

    psi_n, psi_nm1 = psi[1:].reshape((-1,) + pad), psi[:-1].reshape((-1,) + pad)
    chi_n, chi_nm1 = chi[1:].reshape((-1,) + pad), chi[:-1].reshape((-1,) + pad)
    xi_n = psi_n - 1j * chi_n
    xi_nm1 = psi_nm1 - 1j * chi_nm1

    fac_a = d_n / m + n / x
    a_n = (fac_a * psi_n - psi_nm1) / (fac_a * xi_n - xi_nm1)
    fac_b = d_n * m + n / x
    b_n = (fac_b * psi_n - psi_nm1) / (fac_b * xi_n - xi_nm1)

    tn = 2 * n + 1
    q_ext = (2.0 / x ** 2) * np.sum(tn * np.real(a_n + b_n), axis=0)
    q_sca = (2.0 / x ** 2) * np.sum(tn * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2), axis=0)

    zero_row = np.zeros((1,) + m.shape, dtype=complex)
    a_np1 = np.concatenate([a_n[1:], zero_row], axis=0)
    b_np1 = np.concatenate([b_n[1:], zero_row], axis=0)
    g_qsca = (4.0 / x ** 2) * (
        np.sum(n * (n + 2) / (n + 1) * np.real(a_n * np.conj(a_np1) + b_n * np.conj(b_np1)), axis=0)
        + np.sum(tn / (n * (n + 1)) * np.real(a_n * np.conj(b_n)), axis=0)
    )

    q_sca = np.maximum(q_sca, 1.0e-30)
    q_ext = np.maximum(q_ext, q_sca)
    return (q_ext,
            np.clip(q_sca / q_ext, 0.0, 1.0),
            np.clip(g_qsca / q_sca, -1.0, 1.0))
