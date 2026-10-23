"""ECHAM's ``lccnclim`` climatological freezing inputs (#1149): the port and its wiring.

``ccnclim_freezing_aerosol`` (``ccnclim.py``) is verified by hand against the
Fortran it ports, not a compiled-Fortran run: ``mo_ccnclim.f90::
ccnclim_IN_setup``'s mixed-phase block (F 547-558) performs no arithmetic at
all on its eight outputs -- each is a straight assignment from a
module-level ``CCNCLIMCTL`` namelist default (F 71-79) -- so reading those
DATA-statement values off the source IS the reference; running the compiled
routine could only reproduce the identical literal assignment (see
``ccnclim.py``'s module docstring for the full argument, including why the
mixed-phase block has no location/season dependence to exercise a harness
against). The eight-input, already-Fortran-verified ``het_mxphase_freezing``
(``lohmann_2m_freezing_reference_test.py``, #953) is what actually turns
these fractions into freezing rates; this file checks only that
:func:`ccnclim_freezing_aerosol` supplies the right eight numbers in the
right shape, that it is differentiable (CLAUDE.md: tunables stay
gradient-visible), and that ``cloud_microphysics_2m``'s section 6.2 reaches
it by default and gets out of its way for a prognostic aerosol or the
explicit ``AerosolFreeFreezing.DEMOTT`` fallback.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics import thermodynamics
from jcm.physics.clouds.lohmann_2m.ccnclim import ccnclim_freezing_aerosol
from jcm.physics.clouds.lohmann_2m.scheme import cloud_microphysics_2m
from jcm.physics.clouds.lohmann_2m.types import HeterogeneousFreezingAerosol
from jcm.physics.clouds.lohmann_2m_params import AerosolFreeFreezing, CloudParams2M
from jcm.testing import check_gradients

# mo_ccnclim.f90 F 71-79 -- the module-level CCNCLIMCTL namelist defaults
# ccnclim_IN_setup's mixed-phase block assigns verbatim (F 549-558).
_FORTRAN_DEFAULTS = dict(
    dust_soluble=0.010, dust_insoluble_accumulation=0.094,
    dust_insoluble_coarse=0.229, bc_soluble=0.042, bc_insoluble=0.0,
    wet_radius_insoluble_aitken=1.0, wet_radius_insoluble_accumulation=2.59e-08,
    wet_radius_insoluble_coarse=7.97e-08,
)

NLEV = 16
DT = 1800.0


# ===========================================================================
# ccnclim_freezing_aerosol itself
# ===========================================================================
def test_matches_the_fortran_namelist_defaults():
    """Every field is exactly mo_ccnclim.f90's own DATA-statement value."""
    p = CloudParams2M.default()
    out = ccnclim_freezing_aerosol(jnp.zeros((4,)), p)
    for name, value in _FORTRAN_DEFAULTS.items():
        got = np.asarray(getattr(out, name))
        # float32 (the scheme's working precision): rtol below float32 eps (1.2e-7).
        np.testing.assert_allclose(got, np.full(got.shape, value), rtol=1e-6)


def test_constants_are_differentiable_leaves_not_hardcoded():
    """The eight constants are CloudParams2M leaves, not bare floats baked into ccnclim.py."""
    p = CloudParams2M.default(ccnclim_fracdusol=0.5, ccnclim_rwetai=1e-6)
    out = ccnclim_freezing_aerosol(jnp.zeros((3,)), p)
    np.testing.assert_allclose(float(out.dust_soluble[0]), 0.5, rtol=1e-6)
    np.testing.assert_allclose(
        float(out.wet_radius_insoluble_accumulation[0]), 1e-6, rtol=1e-6)


@pytest.mark.parametrize("shape", [(5,), (5, 3), (5, 2, 2)])
def test_broadcasts_to_any_horizontal_rank(shape):
    """Broadcasting-native (CLAUDE.md): vertical on axis 0, any trailing shape, no vmap/reshape."""
    p = CloudParams2M.default()
    ref = jnp.zeros(shape, dtype=jnp.float32)
    out = ccnclim_freezing_aerosol(ref, p)
    for leaf in jax.tree_util.tree_leaves(out):
        assert leaf.shape == shape
        assert leaf.dtype == ref.dtype


def test_single_column_and_vectorized_block_agree_per_column():
    """A (kx,) column and the matching column of a (kx, ncols) block give identical values."""
    p = CloudParams2M.default()
    block = ccnclim_freezing_aerosol(jnp.zeros((5, 3)), p)
    column = ccnclim_freezing_aerosol(jnp.zeros((5,)), p)
    for leaf_block, leaf_col in zip(jax.tree_util.tree_leaves(block),
                                     jax.tree_util.tree_leaves(column)):
        np.testing.assert_array_equal(np.asarray(leaf_block[:, 1]), np.asarray(leaf_col))


def test_ccnclim_freezing_aerosol_gradients_match_differences():
    """AD vs. a central difference for the eight tunables (jcm.testing's harness)."""
    with jax.enable_x64(True):
        p = CloudParams2M.default()
        ref = jnp.ones((4,), dtype=jnp.float64)
        names = ("ccnclim_fracdusol", "ccnclim_fracduai", "ccnclim_fracduci",
                 "ccnclim_fracbcsol", "ccnclim_fracbcinsol", "ccnclim_rwetki",
                 "ccnclim_rwetai", "ccnclim_rwetci")
        leaves = {n: getattr(p, n) for n in names}

        def f(vals):
            return jax.tree_util.tree_leaves(ccnclim_freezing_aerosol(ref, p.replace(**vals)))

        check_gradients(f, (leaves,), rtol=1e-6)


# ===========================================================================
# section 6.2 wiring: the three-way static switch
# ===========================================================================
def _mixed_phase_column():
    """Build a column with a mixed-phase supercooled-liquid layer (T in (cthomi, tmelt)).

    Matches ``scheme_test.py``'s own ``_column`` layout (temperature linearly
    spanning 215-295 K top to bottom over 16 levels): levels 9-10 (T = 263.0,
    268.3 K) hold supercooled liquid inside ``ll_mxfrz``'s window, level 10 in
    particular sitting OUTSIDE the DeMott (2010) closure's 238-264 K active
    range -- so the default (lccnclim) and the DEMOTT fallback are expected to
    disagree there, not just by magnitude.
    """
    temperature = jnp.linspace(215.0, 295.0, NLEV)
    pressure = jnp.linspace(2.0e4, 1.0e5, NLEV)
    rho = pressure / (287.04 * temperature)
    qsat = thermodynamics.saturation_specific_humidity_and_derivative(
        temperature, pressure, phase="water")[0]
    q = 0.85 * qsat
    idx = jnp.arange(NLEV)
    qc = jnp.where((idx >= 9) & (idx < 14), 4e-4, 0.0)
    cf = jnp.where(qc > 0, 0.6, 0.0)
    return dict(
        temperature_m1=temperature, specific_humidity_m1=q, pressure=pressure,
        qc_m1=qc, qi_m1=jnp.zeros(NLEV),
        qnc_m1=jnp.where(qc > 0, 8e7 / rho, 0.0), qni_m1=jnp.zeros(NLEV),
        cloud_fraction=cf, air_density=rho,
        layer_thickness=jnp.full(NLEV, 500.0), tke=jnp.full(NLEV, 0.1),
        activated_cdnc=jnp.full(NLEV, 8e7), ice_nuclei=jnp.zeros(NLEV),
        ice_nuclei_deposition=jnp.zeros(NLEV),
    )


def _call(column, params, **kw):
    col = dict(column)
    args = [col.pop(k) for k in (
        "temperature_m1", "specific_humidity_m1", "pressure", "qc_m1", "qi_m1",
        "qnc_m1", "qni_m1", "cloud_fraction", "air_density", "layer_thickness",
        "tke", "activated_cdnc", "ice_nuclei", "ice_nuclei_deposition")]
    return cloud_microphysics_2m(*args, DT, params, **col, **kw)


def test_default_without_a_prognostic_aerosol_matches_explicit_lccnclim():
    """The implicit default path is bit-identical to explicitly building and passing
    ``ccnclim_freezing_aerosol``'s own output as ``freezing_aerosol``.
    """
    column = _mixed_phase_column()
    p = CloudParams2M.default()
    implicit = _call(column, p)
    explicit = _call(column, p,
                     freezing_aerosol=ccnclim_freezing_aerosol(column["air_density"], p))
    for a, b in zip(jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_default_differs_from_the_demott_fallback():
    """Confirm lccnclim (the default) is a genuinely different closure from DEMOTT.

    Not a relabeling of the same numbers -- level 10's liquid sits outside
    DeMott's temperature window (see ``_mixed_phase_column``'s docstring), so
    DEMOTT freezes nothing there while lccnclim's rate-based closure does.
    """
    column = _mixed_phase_column()
    lccnclim = _call(column, CloudParams2M.default())
    demott = _call(column, CloudParams2M.default(aerosol_free_freezing=AerosolFreeFreezing.DEMOTT))
    diff = np.abs(np.asarray(lccnclim[0].dqnidt) - np.asarray(demott[0].dqnidt))
    assert float(np.max(diff)) > 0.0
    assert float(diff[10]) > 0.0 and float(np.asarray(demott[0].dqnidt)[10]) == 0.0


def test_prognostic_aerosol_overrides_the_aerosol_free_default():
    """With ``freezing_aerosol`` supplied, ``aerosol_free_freezing`` has no effect at
    all: the JAM path always wins over both aerosol-free closures.
    """
    column = _mixed_phase_column()
    fa = HeterogeneousFreezingAerosol(**{
        k: jnp.full((NLEV,), v) for k, v in dict(
            dust_soluble=0.3, dust_insoluble_accumulation=0.0, dust_insoluble_coarse=0.0,
            bc_soluble=0.1, bc_insoluble=0.0, wet_radius_insoluble_aitken=0.0,
            wet_radius_insoluble_accumulation=0.0, wet_radius_insoluble_coarse=0.0,
        ).items()})
    with_lccnclim_default = _call(column, CloudParams2M.default(), freezing_aerosol=fa)
    with_demott_selected = _call(
        column, CloudParams2M.default(aerosol_free_freezing=AerosolFreeFreezing.DEMOTT),
        freezing_aerosol=fa)
    for a, b in zip(jax.tree_util.tree_leaves(with_lccnclim_default),
                    jax.tree_util.tree_leaves(with_demott_selected)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_default_aerosol_free_gradient_through_the_scheme_is_finite():
    """The lccnclim fractions stay live, differentiable parameters all the way through
    section 6.2's ``het_mxphase_freezing`` call -- not just inside ``ccnclim.py``.
    """
    column = _mixed_phase_column()

    def loss(fracdusol, rwetai):
        p = CloudParams2M.default(ccnclim_fracdusol=fracdusol, ccnclim_rwetai=rwetai)
        tend = _call(column, p)[0]
        return jnp.sum(tend.dqidt ** 2) + jnp.sum(tend.dqnidt ** 2)

    g = jax.grad(loss, argnums=(0, 1))(0.010, 2.59e-8)
    assert all(np.isfinite(float(x)) for x in g)


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
