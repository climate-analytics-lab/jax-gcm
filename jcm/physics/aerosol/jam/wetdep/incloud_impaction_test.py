"""Tests for ``wetdep.incloud_impaction`` (ECHAM-HAM ``ic_scav_imp``).

The compiled-Fortran reference and how it was made are described in
``jcm/data/test/echam_cloud_reference/hamimpaction_README.md``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.wetdep import incloud_impaction as ii
from jcm.physics.aerosol.jam.wetdep import incloud_impaction_tables as tables
from jcm.testing import check_gradients

REFERENCE = (Path(__file__).resolve().parents[4] / "data" / "test"
             / "echam_cloud_reference" / "hamimpaction_T63L47.npz")

#: Reference variant -> jcm variant. ``fixed`` is r7492 compiled with the
#: three corrections applied to its source (see the README).
VARIANTS = {"r7492": "ham_r7492", "fixed": "ham"}


def _load():
    with np.load(REFERENCE) as z:
        return {k: z[k] for k in z.files}


def _jcm_fractions(z, variant):
    """Return the (n_cells, n_modes, phase, moment) fractions jcm gives the reference cells."""
    out = np.zeros(z["out_r7492_sfimp"].shape)
    reffl = jnp.asarray(z["reffl"])
    reffi = jnp.asarray(z["reffi"])
    icnc_m3 = jnp.asarray(z["icnc"] * z["rho"])
    dt = float(z["dt"])
    for m, sigma in enumerate(z["geom_std_dev"]):
        r_wet = jnp.asarray(z["rwet"][:, m])
        for k, moment in enumerate(("number", "mass")):
            mr = ii.impaction_radius_um(r_wet, float(sigma), moment)
            out[:, m, 0, k] = ii.droplet_impaction_fraction(
                reffl, mr, moment, variant)
            out[:, m, 1, k] = ii.crystal_impaction_fraction(
                reffi, icnc_m3, mr, dt, variant)
    return out


def test_tables_are_the_compiled_module_values():
    """The packaged tables equal the values the compiled r7492 module holds."""
    z = _load()
    for name, key in (("CAERORAD_UM", "table_caerorad"),
                      ("CDROPRAD_UM_R7492", "table_cdroprad"),
                      ("CPLATERAD_UM", "table_cplaterad"),
                      ("SCAVDROPN", "table_scavdropn"),
                      ("SCAVDROPM", "table_scavdropm"),
                      ("SCAVICEPLATE", "table_scaviceplate")):
        np.testing.assert_array_equal(getattr(tables, name), z[key], name)
    expected = z["table_cdroprad"].copy()
    expected[6] = 30.0
    np.testing.assert_array_equal(tables.CDROPRAD_UM, expected)


@pytest.mark.parametrize("ref_variant", sorted(VARIANTS))
def test_matches_compiled_ic_scav_imp(ref_variant):
    """Every cell, mode, phase and moment against the compiled routine.

    ``r7492`` is the unmodified ``ic_scav_imp`` (with the ``mr``/``indexy``
    block of ``ham_wetdep`` and ``scavcoef_bilinterp``); ``fixed`` is the same
    source with the three corrections. HAM's fraction is clipped to [0, 1] by
    ``get_icscavfrac``, so the reference is too. The ice fraction is
    ``1 - exp(-x)`` in Fortran and ``-expm1(-x)`` here: for tiny ``x`` the
    Fortran form carries an absolute error of order 1e-16, hence the
    ``atol``.
    """
    z = _load()
    with jax.enable_x64(True):
        got = _jcm_fractions(z, VARIANTS[ref_variant])
        mr = np.stack([
            np.stack([np.asarray(ii.impaction_radius_um(
                jnp.asarray(z["rwet"][:, m]), float(s), mom))
                for mom in ("number", "mass")], axis=-1)
            for m, s in enumerate(z["geom_std_dev"])], axis=1)
    np.testing.assert_allclose(mr, z["out_mr_um"], rtol=1e-14, atol=0.0)
    want = np.clip(z[f"out_{ref_variant}_sfimp"], 0.0, 1.0)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_matches_compiled_ic_scav_imp_in_float32():
    """The float32 path the model runs agrees with the compiled routine.

    Bin choices can differ from float64 only for inputs within float32
    round-off of a node, which the reference cells are not.
    """
    z = _load()
    with jax.enable_x64(False):
        got = _jcm_fractions(z, "ham")
    want = np.clip(z["out_fixed_sfimp"], 0.0, 1.0)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-10)


def test_reference_cells_exercise_every_branch():
    """The reference is not vacuous: each regime of the lookup is present."""
    z = _load()
    reffl, reffi, icnc = z["reffl"], z["reffi"], z["icnc"]
    assert np.any((reffl > 0) & (reffl < 5))
    assert np.any((reffl >= 25) & (reffl < 35))          # cdroprad node 6
    assert np.any(reffl >= 45)                           # droplet clamp
    assert np.any(reffl == 0) and np.any(reffi == 0)
    assert np.any((reffi >= 1) & (reffi < 5))
    assert np.any((reffi >= 50) & (reffi < 100))         # plate index /50
    assert np.any((reffi >= 100) & (reffi < 650))
    assert np.any((reffi > 0) & (icnc == 0))             # no crystals
    mr_m = z["out_mr_um"][..., 1]
    assert np.any(mr_m == 50.0)                          # the 50 um cap
    # The corrections move the result where they apply.
    diff = ~np.isclose(z["out_r7492_sfimp"], z["out_fixed_sfimp"],
                       rtol=1e-6, atol=0.0)
    assert diff[:, :, 0].any() and diff[:, :, 1].any()
    # The real columns carry both phases with non-trivial fractions.
    assert np.max(z["out_fixed_sfimp"][:, :, 0]) > 1e-3
    assert np.max(z["out_fixed_sfimp"][:, :, 1]) > 1e-3


@pytest.mark.parametrize("moment", ("number", "mass"))
def test_default_interpolates_the_table(moment):
    """At the table nodes the default returns the entry; between them it is continuous.

    This is what the corner order and the droplet axis fix: r7492 returns
    the transposed off-diagonal entry at the (x2, y1) node.
    """
    with jax.enable_x64(True):
        table = tables.SCAVDROPN if moment == "number" else tables.SCAVDROPM
        x = jnp.asarray([10.0, 30.0, 40.0])                # droplet nodes 2, 6, 8
        rows = np.array([2, 6, 8])
        j = 35                                             # aerosol node
        y = jnp.full(3, tables.CAERORAD_UM[j])
        got = ii.droplet_impaction_fraction(x, y, moment)
        np.testing.assert_allclose(got, table[rows, j], rtol=1e-12)
        # Continuity across a droplet node (left and right limits agree).
        eps = 1e-9
        left = ii.droplet_impaction_fraction(x - eps, y * 1.07, moment)
        right = ii.droplet_impaction_fraction(x + eps, y * 1.07, moment)
        np.testing.assert_allclose(left, right, rtol=1e-6)
        # r7492 at the (x2, y1) corner of a cell returns the (x1, y2) entry.
        # The aerosol nodes are four-figure roundings of 1e-4 * 2**((i-1)/3),
        # so pick a node HAM's index formula maps to itself (rounded up).
        k = next(i for i in range(30, 50)
                 if np.floor(3.0 * np.log2(1e4 * tables.CAERORAD_UM[i]) + 1.0)
                 == i)
        r7492 = ii.droplet_impaction_fraction(
            jnp.asarray([15.0 - 1e-12]), jnp.asarray([tables.CAERORAD_UM[k]]),
            moment, "ham_r7492")
        np.testing.assert_allclose(r7492, table[2, k + 1], rtol=1e-9)
        fixed = ii.droplet_impaction_fraction(
            jnp.asarray([15.0 - 1e-12]), jnp.asarray([tables.CAERORAD_UM[k]]),
            moment)
        np.testing.assert_allclose(fixed, table[3, k], rtol=1e-9)


def test_plate_lookup_brackets_large_crystals():
    """Above 50 um the default reads the bracketing 25 um plate nodes."""
    with jax.enable_x64(True):
        j = 40
        y = jnp.full(4, tables.CAERORAD_UM[j])
        r = jnp.asarray([50.0, 75.0, 100.0, 125.0])         # plate nodes 10-13
        icnc = jnp.full(4, 1.0e5)
        dt = 600.0
        got = ii.crystal_impaction_fraction(r, icnc, y, dt)
        k = tables.SCAVICEPLATE[10:14, j]
        np.testing.assert_allclose(got, -np.expm1(-k * 1e-6 * 1e5 * dt),
                                   rtol=1e-12)


def _fractions(reffl, reffi, icnc_m3, r_wet, dt):
    """Both phases and moments of the coarse mode, the gradient test target."""
    sigma = MAM4_SPEC.modes[2].geom_std_dev
    out = []
    for moment in ("number", "mass"):
        mr = ii.impaction_radius_um(r_wet, sigma, moment)
        out.append(ii.droplet_impaction_fraction(reffl, mr, moment))
        out.append(ii.crystal_impaction_fraction(reffi, icnc_m3, mr, dt))
    return jnp.stack(out)


def test_gradients_inside_the_table():
    """AD matches a central difference in the radii, the crystal number and dt."""
    with jax.enable_x64(True):
        args = (jnp.asarray([7.3, 12.6, 21.9, 31.4]),     # reffl [um]
                jnp.asarray([13.2, 37.7, 62.4, 112.8]),   # reffi [um]
                jnp.asarray([2.0e4, 8.0e4, 3.0e5, 1.0e6]),  # ICNC [m-3]
                jnp.asarray([0.31e-6, 0.83e-6, 1.7e-6, 2.9e-6]),  # r_wet [m]
                jnp.asarray(720.0))
        check_gradients(_fractions, args, rtol=1e-5,
                        live_inputs=("[0]", "[1]", "[2]", "[3]", "[4]"))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_gradients_are_finite_at_the_table_edges(dtype):
    """Clamped and gated inputs give finite gradients, never NaN.

    No liquid or ice (radius 0), no crystals, a crystal below 1 um, a droplet
    beyond the last node, a crystal beyond the largest plate, an empty mode
    (radius 0) and a mode above the 50 um cap.
    """
    with jax.enable_x64(dtype == jnp.float64):
        reffl = jnp.asarray([0.0, 0.0, 60.0, 3.0, 47.0, 12.0, 12.0], dtype)
        reffi = jnp.asarray([0.0, 0.5, 30.0, 700.0, 0.0, 30.0, 30.0], dtype)
        icnc = jnp.asarray([0.0, 1e5, 0.0, 1e5, 1e5, 1e5, 1e5], dtype)
        r_wet = jnp.asarray([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0.0, 1e-4], dtype)
        dt = jnp.asarray(720.0, dtype)

        def total(*a):
            return jnp.sum(_fractions(*a))

        grads = jax.grad(total, argnums=(0, 1, 2, 3, 4))(
            reffl, reffi, icnc, r_wet, dt)
        for g in grads:
            assert np.all(np.isfinite(np.asarray(g)))


def test_variant_and_moment_are_checked():
    with pytest.raises(ValueError, match="variant"):
        ii.droplet_impaction_fraction(1.0, 1.0, "number", "cam")
    with pytest.raises(ValueError, match="moment"):
        ii.impaction_radius_um(1e-6, 1.8, "volume")
