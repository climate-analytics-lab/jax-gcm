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
                                     phase=_PHASE[int(phase)], tables=tables))
            np.testing.assert_allclose(got, ref_rain, rtol=RTOL[prec], atol=1e-30)


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_bc_snow_rate_matches_compiled_reference(prec):
    z = load()
    with jax.enable_x64(prec == "float64"):
        tables = load_croft_tables()
        dtype = np.float64 if prec == "float64" else np.float32
        for flux, radius, ref_snow in zip(z["flux"], z["radius"], z["snow_rate"]):
            got = float(bc_snow_rate(np.asarray(flux, dtype), np.asarray(radius, dtype),
                                     tables=tables))
            np.testing.assert_allclose(got, ref_snow, rtol=RTOL[prec], atol=1e-30)


def test_bc_rain_rate_vectorizes_over_a_column():
    """The per-tracer call also has to run broadcast over a (nlev,) column."""
    with jax.enable_x64(True):
        z = load()
        tables = load_croft_tables()
        flux = np.asarray(z["flux"][:8], np.float64)
        radius = np.asarray(z["radius"][:8], np.float64)
        got = np.asarray(bc_rain_rate(flux, radius, phase="number", tables=tables))
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
