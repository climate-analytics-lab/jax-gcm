"""jcm's HAM below-cloud scavenging against the compiled ECHAM6.3-HAM2.3 routine (#1017).

``jcm/data/test/echam_cloud_reference/hamwetdep_bc.npz`` holds the outputs of
the UNMODIFIED r7492 ``mo_ham_wetdep.f90::bc_rain``/``bc_snow``
(``kscavBCtype=3``) -- including the ``indexy1``/``indexy2`` index-computing
snippet that normally runs inline in ``ham_wetdep`` just before calling them
-- on a designed grid of (precip flux, wet radius) test points
(``hamwetdep_bc_README.md``, ``hamwetdep_bc_provenance.json``). This is a
genuine end-to-end check (wet radius -> bin index -> bilinear table lookup ->
rate), not merely a recompilation of the table data: it is what caught both
the missing 50 um radius clip and the ``Q12``/``Q21`` corner-swap quirk in
``bc_rain``'s own data-filling loop documented in ``ham_below_cloud.py``.

Measured max relative error over the 96 cases: float64 0.0 (exact); float32
3.5e-7 (rain), 1.6e-7 (snow) -- the tolerance below is set from that measured
max, not a guessed round number.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import numpy as np
import pytest

from jcm.physics.aerosol.jam.wetdep.ham_below_cloud import (
    bc_rain_rate,
    bc_snow_rate,
    load_croft_tables,
    rain_rate_bin,
)

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "hamwetdep_bc.npz")
_PHASE = {1: "number", 2: "mass"}
# float64 matches the compiled reference to round-off (it is a direct table
# lookup + bilinear interpolation, a closed form of the inputs); float32's
# tolerance is the measured max relative error over all 96 cases (6.0e-7),
# not a guessed round number.
RTOL = {"float64": 1e-12, "float32": 5e-7}


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_bc_rain_rate_matches_compiled_reference(prec):
    z = load()
    with jax.enable_x64(prec == "float64"):
        tables = load_croft_tables()
        dtype = np.float64 if prec == "float64" else np.float32
        for flux, radius, phase, ref_rain in zip(z["flux"], z["radius"], z["phase"],
                                                   z["rain_rate"]):
            got = float(bc_rain_rate(np.asarray(flux, dtype), np.asarray(radius, dtype),
                                     phase=_PHASE[int(phase)], tables=tables,
                                     variant="ham_r7492"))
            np.testing.assert_allclose(got, ref_rain, rtol=RTOL[prec], atol=1e-30)


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_bc_snow_rate_matches_compiled_reference(prec):
    z = load()
    with jax.enable_x64(prec == "float64"):
        tables = load_croft_tables()
        dtype = np.float64 if prec == "float64" else np.float32
        for flux, radius, ref_snow in zip(z["flux"], z["radius"], z["snow_rate"]):
            got = float(bc_snow_rate(np.asarray(flux, dtype), np.asarray(radius, dtype),
                                     tables=tables, variant="ham_r7492"))
            np.testing.assert_allclose(got, ref_snow, rtol=RTOL[prec], atol=1e-30)


def test_bc_rain_rate_vectorizes_over_a_column():
    """The per-tracer call also has to run broadcast over a (nlev,) column."""
    with jax.enable_x64(True):
        z = load()
        tables = load_croft_tables()
        flux = np.asarray(z["flux"][:8], np.float64)
        radius = np.asarray(z["radius"][:8], np.float64)
        got = np.asarray(bc_rain_rate(flux, radius, phase="number", tables=tables,
                                      variant="ham_r7492"))
        mask = z["phase"][:8] == 1
        np.testing.assert_allclose(got[mask], z["rain_rate"][:8][mask], rtol=1e-12)


def test_below_50um_clip_matters():
    """Without the clip (mo_ham_wetdep.f90:272), a radius above the 50 um cap
    would extrapolate past caerorad's top node (83.23 um) instead of reading
    the clamped value -- confirm the clip actually changes the result.
    """
    with jax.enable_x64(True):
        tables = load_croft_tables()
        capped = float(bc_rain_rate(np.asarray(0.01), np.asarray(50e-6),
                                    phase="mass", tables=tables))
        above = float(bc_rain_rate(np.asarray(0.01), np.asarray(120e-6),
                                   phase="mass", tables=tables))
        assert capped == above, "radii above the 50 um cap must clamp to the cap's rate"


def _bilinear_reference(tables, table, pfrain, mr_um):
    """Return the textbook bilinear interpolant of ``table`` in (rain rate, radius)."""
    cr = np.asarray(tables.crainrate)
    ca = np.asarray(tables.caerorad)
    i = int(np.clip(np.searchsorted(cr, pfrain, side="right") - 1, 0, len(cr) - 2))
    j = int(np.clip(np.searchsorted(ca, mr_um, side="right") - 1, 0, len(ca) - 2))
    wx = (pfrain - cr[i]) / (cr[i + 1] - cr[i])
    wy = (mr_um - ca[j]) / (ca[j + 1] - ca[j])
    t = np.asarray(table)
    return ((1 - wx) * (1 - wy) * t[i, j] + wx * (1 - wy) * t[i + 1, j]
            + (1 - wx) * wy * t[i, j + 1] + wx * wy * t[i + 1, j + 1])


@pytest.mark.parametrize("phase", ("number", "mass"))
def test_corrected_lookup_is_the_bilinear_interpolant(phase):
    """Inside the table the default lookup is the bilinear interpolant, at
    points inside each bin (clear of the nodes' 4-digit rounding).
    """
    with jax.enable_x64(True):
        tables = load_croft_tables()
        table = tables.cscavbcrn if phase == "number" else tables.cscavbcrm
        cr = np.asarray(tables.crainrate)
        ca = np.asarray(tables.caerorad)
        for i in range(1, len(cr) - 1):
            pfrain = np.sqrt(cr[i] * cr[i + 1])
            for j in range(20, 58, 3):
                mr_um = np.sqrt(ca[j] * ca[j + 1])
                got = float(bc_rain_rate(np.asarray(pfrain), np.asarray(mr_um * 1e-6),
                                         phase=phase, tables=tables))
                want = _bilinear_reference(tables, table, pfrain, mr_um)
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-30,
                                           err_msg=f"node ({i}, {j})")


def test_corrected_rain_pair_reaches_zero_continuously():
    """Below the first rain node r7492's pair collapses onto the zero-rain row;
    the corrected rate interpolates to it, continuous across the switch and
    linear in the rain rate below the first node.
    """
    with jax.enable_x64(True):
        tables = load_croft_tables()
        cr1 = float(tables.crainrate[1])
        switch = 10.0 ** -2.5 / 3600.0     # where 2 log10(3600 p) + 5 = 0
        mr = np.asarray(0.5e-6)
        below, above = np.asarray(switch * (1 - 1e-9)), np.asarray(switch * (1 + 1e-9))
        assert rain_rate_bin(below, "ham_r7492") == (0, 0)
        assert rain_rate_bin(below) == (0, 1)
        r7492 = [float(bc_rain_rate(p, mr, phase="number", tables=tables,
                                    variant="ham_r7492")) for p in (below, above)]
        fixed = [float(bc_rain_rate(p, mr, phase="number", tables=tables))
                 for p in (below, above)]
        assert r7492[0] == 0.0 and r7492[1] > 0.0
        np.testing.assert_allclose(fixed[0], fixed[1], rtol=1e-8)
        at_node = float(bc_rain_rate(np.asarray(cr1), mr, phase="number", tables=tables))
        tiny = float(bc_rain_rate(np.asarray(cr1 * 1e-3), mr, phase="number",
                                  tables=tables))
        np.testing.assert_allclose(tiny, 1e-3 * at_node, rtol=1e-9)


def test_corrected_lookup_interpolates_along_radius_at_the_rain_clamp():
    """Above 100 mm/h the rain axis clamps to its last node; the radius axis
    must still interpolate (r7492's swapped corners return the upper radius
    node's value throughout the bin instead).
    """
    with jax.enable_x64(True):
        tables = load_croft_tables()
        ca = np.asarray(tables.caerorad)
        row = np.asarray(tables.cscavbcrn)[-1]
        j = 40
        mr_um = np.sqrt(ca[j] * ca[j + 1])
        wy = (mr_um - ca[j]) / (ca[j + 1] - ca[j])
        got = float(bc_rain_rate(np.asarray(0.05), np.asarray(mr_um * 1e-6),
                                 phase="number", tables=tables))
        np.testing.assert_allclose(got, (1 - wy) * row[j] + wy * row[j + 1],
                                   rtol=1e-12)


def test_corrected_lookup_gradients_are_finite():
    import jax.numpy as jnp

    with jax.enable_x64(True):
        tables = load_croft_tables()

        def rate(log_p, mr):
            return bc_rain_rate(jnp.exp(log_p), mr, phase="mass", tables=tables)

        for p in (1e-8, 5e-7, 3e-5, 0.01, 0.05):
            for mr in (1e-8, 2e-7, 3e-6, 4e-5):
                g = jax.grad(rate, argnums=(0, 1))(jnp.log(p), jnp.asarray(mr))
                assert all(np.isfinite(float(x)) for x in g), (p, mr)


def test_corrected_lookup_is_continuous_across_the_aerosol_nodes():
    """r7492's aerosol index assumes the nominal 2**(1/3) spacing of nodes the
    table rounds to four digits, so its rate jumps where the index switches;
    the corrected bracket is the table's own and the rate is continuous.
    """
    with jax.enable_x64(True):
        tables = load_croft_tables()
        jumps = {"ham": 0.0, "ham_r7492": 0.0}
        for k in range(2, 59):
            switch_m = 1e-4 * 2 ** ((k - 1) / 3) * 1e-6
            for variant in jumps:
                lo, hi = (float(bc_rain_rate(np.asarray(3e-5), np.asarray(switch_m * f),
                                             phase="number", tables=tables,
                                             variant=variant))
                          for f in (1 - 1e-12, 1 + 1e-12))
                if max(lo, hi) > 0:
                    jumps[variant] = max(jumps[variant], abs(hi - lo) / max(lo, hi))
        assert jumps["ham"] < 1e-9
        assert jumps["ham_r7492"] > 1e-3
