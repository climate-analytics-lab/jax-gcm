"""jcm's Kaercher-Lohmann cirrus port against compiled ECHAM6.3-HAM2.3 r7492
(jax-gcm#1017 task 2 part 2a).

``hamcirrus.npz`` holds the compiled, UNMODIFIED ``mo_cirrus.f90::xfrzmstr``
(which calls ``xfrzhom``/``xicehom``) run on 24 independently designed
single-level cells: temperature 190-238 K, ice supersaturation below/at/
above ``SCRHOM(T)``, updraft 0.01-2 m/s, aerosol number 1e6-1e10 m-3 (plus
the orchestrator's own depletion floor), and both sides of the ``cthomi``
gate. See ``hamcirrus_README.md``/``hamcirrus_provenance.json`` for the
harness and build details.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.clouds.lohmann_2m.cirrus import xfrzmstr
from jcm.physics.clouds.lohmann_2m_fortran_reference_test import (
    echam_constants, echam_params, precision,
)

REF = (Path(__file__).resolve().parents[3] / "data" / "test"
       / "echam_cloud_reference" / "hamcirrus.npz")


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_xfrzmstr_matches_compiled_fortran(prec):
    """``xfrzmstr`` on all 24 designed cells, compared to the compiled chain.

    float64 matches to round-off (max relative error 8.0e-15, well inside
    the task's 1e-12 target); float32 to 8.8e-7 (float32 round-off). Both
    measured, not guessed; the margin below is >=100x the measured value.
    """
    z = load()
    dtype = jnp.float64 if prec == "float64" else jnp.float32

    with jax.enable_x64(prec == "float64"), echam_constants(), precision(prec):
        params = echam_params()
        j = functools.partial(lambda v: jnp.asarray(np.asarray(v, np.float64), dtype))
        susati = j(z["in/susati"])
        updraft_ms = j(z["in/verv_cms"] / 100.0)
        apn_cm3 = j(z["in/apn_cm3"])
        t = j(z["in/t"])
        p_pa = j(z["in/p_pa"])
        dt = jnp.asarray(float(z["config/ztmst"]), dtype)
        ri, nicex = xfrzmstr(susati, updraft_ms, apn_cm3, t, p_pa, dt, params)

    ri = np.asarray(ri, np.float64)
    nicex = np.asarray(nicex, np.float64)
    ref_ri = z["out/zri_m"]
    ref_nicex = z["out/znicex_m3"]

    rtol = {"float64": 1e-12, "float32": 1e-4}[prec]
    atol_ri = {"float64": 1e-19, "float32": 1e-12}[prec]
    atol_nicex = {"float64": 1e-9, "float32": 1.0}[prec]

    np.testing.assert_allclose(ri, ref_ri, rtol=rtol, atol=atol_ri,
                                err_msg="zri")
    np.testing.assert_allclose(nicex, ref_nicex, rtol=rtol, atol=atol_nicex,
                                err_msg="znicex")

    # The "no freeze" cell must be EXACTLY zero (the Fortran's own
    # zero-initialisation, mo_cirrus.f90:300-301, not a tiny residual).
    names = [str(x) for x in z["meta/names"]]
    no_freeze = names.index("temp_above_cthomi_no_freeze")
    assert ri[no_freeze] == 0.0
    assert nicex[no_freeze] == 0.0

    # At least one cell must show genuinely non-zero nucleation -- the
    # thing this port exists to produce.
    assert float(np.max(nicex)) > 0.0


def test_gradient_well_behaved_across_every_gate():
    """Gradients stay finite across every gate this module double-``where``‐s.

    Perturbs susati/updraft/aerosol/temperature by a tiny amount around
    EVERY designed cell (including the ones that gate out entirely, or that
    sit exactly at a branch boundary) and checks no NaN/inf appears --
    JAX_gotchas.md's class of discarded-branch singularity.
    """
    z = load()
    with jax.enable_x64(True), echam_constants(), precision("float64"):
        params = echam_params()

        def total(susati, updraft_ms, apn_cm3, t, p_pa, dt):
            ri, nicex = xfrzmstr(susati, updraft_ms, apn_cm3, t, p_pa, dt, params)
            return jnp.sum(ri) + jnp.sum(nicex) * 1e-7

        susati = jnp.asarray(z["in/susati"])
        updraft_ms = jnp.asarray(z["in/verv_cms"] / 100.0)
        apn_cm3 = jnp.asarray(z["in/apn_cm3"])
        t = jnp.asarray(z["in/t"])
        p_pa = jnp.asarray(z["in/p_pa"])
        dt = jnp.asarray(float(z["config/ztmst"]))

        grads = jax.grad(total, argnums=(0, 1, 2, 3))(
            susati, updraft_ms, apn_cm3, t, p_pa, dt)
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g))), g


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
