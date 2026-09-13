"""Below-cloud impaction coefficients against CAM's ``calc_1_impact_rate``.

The reference numbers were produced by compiling CAM's own
``calc_1_impact_rate`` (ESCOMP/CAM ``cam_development``,
``src/chemistry/modal_aero/aero_model.F90``) as a standalone program and
running it at the tabulation state the routine is used at (273.16 K,
0.75e6 dyne/cm², the mode's dry material density).
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_jvp, check_vjp

from jcm.physics.aerosol.jam.wetdep.impaction import (
    DLN_DG,
    GROW_MAX,
    GROW_MIN,
    IMPACT_SCALE_DEFAULT,
    MU_WATER_AIR_DEFAULT,
    bcscavcoef,
    build_impaction_table,
    impaction_scavenging_rates,
    table_log_coefficients,
)

#: CAM ``mo_constants`` in cgs. jcm's shared constants differ in the 6th-7th
#: significant digit, so the algorithmic parity below is checked against CAM's
#: own values and the constant swap is bounded separately.
CAM_CGS = dict(boltz_cgs=1.38065e-16, rgas_cgs=8.3144724e7, m_air_cgs=28.966)

#: ``(dg_wet [m], sigma_g, density [kg/m³]) -> (number, volume)`` [1/mm].
CAM_REFERENCE = (
    ((0.011e-6, 1.8, 1770.0), (9.1991872722e-03, 2.4933704370e-03)),
    ((0.110e-6, 1.8, 1770.0), (6.8273473378e-04, 8.1447204069e-04)),
    ((0.260e-6, 1.6, 1770.0), (6.5570039628e-04, 1.1249622455e-03)),
    ((0.440e-6, 1.8, 1770.0), (1.4865672054e-03, 6.1732071538e-02)),
    ((1.000e-6, 1.8, 2600.0), (5.8402937958e-02, 7.7942248519e-01)),
    ((2.000e-6, 1.8, 2600.0), (4.0869856241e-01, 1.7861883989e+00)),
    ((4.000e-6, 1.8, 2600.0), (1.2854641228e+00, 2.6034014003e+00)),
    ((10.00e-6, 1.8, 2600.0), (2.5163956909e+00, 3.0433277257e+00)),
)


class ImpactionRateTest(unittest.TestCase):

    def test_matches_cam_calc_1_impact_rate(self):
        """The algorithm, checked against CAM with CAM's own constants."""
        for args, (ref_num, ref_vol) in CAM_REFERENCE:
            num, vol = impaction_scavenging_rates(*args, **CAM_CGS)
            self.assertAlmostEqual(num / ref_num, 1.0, places=8, msg=str(args))
            self.assertAlmostEqual(vol / ref_vol, 1.0, places=8, msg=str(args))

    def test_shared_constants_move_the_coefficient_negligibly(self):
        """Shared constants vs CAM's: a 6th-digit difference, not a physics one.

        Production uses ``jcm.constants`` (CLAUDE.md's single source of truth),
        whose ak / r_universal / m_air differ from CAM ``mo_constants`` by
        <= 7.4e-6 relative. That is the floor on the parity above, so it is
        bounded here rather than folded into it.
        """
        for args, _ in CAM_REFERENCE:
            cam = impaction_scavenging_rates(*args, **CAM_CGS)
            jcm = impaction_scavenging_rates(*args)
            for got, ref in zip(jcm, cam):
                self.assertLess(abs(got - ref) / ref, 1.0e-4, msg=str(args))

    def test_greenfield_minimum(self):
        # Brownian diffusion falls and inertial impaction rises with size, so
        # the coefficient has a minimum in the 0.05-0.5 µm accumulation range
        # — the feature an r² extrapolation of a submicron coefficient cannot
        # represent.
        diameters = np.geomspace(5.0e-9, 2.0e-5, 40)
        vol = np.array([
            impaction_scavenging_rates(d, 1.8, 1770.0)[1] for d in diameters])
        d_min = diameters[int(np.argmin(vol))]
        self.assertGreater(d_min, 5.0e-8)
        self.assertLess(d_min, 5.0e-7)

    def test_bounded_and_saturating_with_size(self):
        # Slinn's efficiency is capped at 1, so the coefficient saturates at
        # the rain's geometric sweep-out rate rather than growing as r². A
        # 5x size increase past the coarse mode must change it by well under
        # the 25x an r² law would give.
        _, vol_10 = impaction_scavenging_rates(10.0e-6, 1.8, 2000.0)
        _, vol_50 = impaction_scavenging_rates(50.0e-6, 1.8, 2000.0)
        self.assertGreater(vol_50, vol_10)
        self.assertLess(vol_50 / vol_10, 2.0)
        self.assertLess(vol_50, 10.0)

    def test_monotone_above_the_gap(self):
        diameters = np.geomspace(1.0e-6, 2.0e-5, 12)
        vol = np.array([
            impaction_scavenging_rates(d, 1.8, 2000.0)[1] for d in diameters])
        self.assertTrue(bool(np.all(np.diff(vol) > 0.0)))


class ImpactionTableTest(unittest.TestCase):

    def setUp(self):
        self.table = build_impaction_table(2.0e-6, 1.8, 2600.0)

    def _lookup(self, mu=MU_WATER_AIR_DEFAULT, scale=IMPACT_SCALE_DEFAULT):
        """``(dgnum, ln_number, ln_volume)`` at the given knob values."""
        ln_n, ln_v = table_log_coefficients(self.table, mu, scale)
        return self.table.dgnum, ln_n, ln_v

    def test_table_nodes_reproduce_the_integral(self):
        # Look up at exactly a table node: the interpolation weight is zero,
        # so the result must be the integral itself.
        for jgrow in (GROW_MIN, 0, 5, GROW_MAX):
            dg = 2.0e-6 * np.exp(jgrow * DLN_DG)
            ref_num, ref_vol = impaction_scavenging_rates(dg, 1.8, 2600.0)
            num, vol = bcscavcoef(jnp.asarray(0.5 * dg), *self._lookup())
            self.assertAlmostEqual(float(num) / ref_num, 1.0, places=4)
            self.assertAlmostEqual(float(vol) / ref_vol, 1.0, places=4)

    def test_interpolation_is_bracketed(self):
        # Between two nodes the log-linear value stays between them.
        dg = 2.0e-6 * np.exp(2.5 * DLN_DG)
        lo = impaction_scavenging_rates(2.0e-6 * np.exp(2 * DLN_DG), 1.8, 2600.0)
        hi = impaction_scavenging_rates(2.0e-6 * np.exp(3 * DLN_DG), 1.8, 2600.0)
        _, vol = bcscavcoef(jnp.asarray(0.5 * dg), *self._lookup())
        self.assertGreater(float(vol), lo[1])
        self.assertLess(float(vol), hi[1])

    def test_clamped_below_the_table(self):
        # CAM clamps at the low end; a vanishing radius must not blow up.
        _, vol = bcscavcoef(jnp.asarray(1.0e-12), *self._lookup())
        self.assertTrue(np.isfinite(float(vol)))
        self.assertAlmostEqual(
            float(vol), float(jnp.exp(self._lookup()[2][0])), places=6)

    def test_broadcasts_over_column_and_grid_shapes(self):
        # Broadcasting-native: the vertical is axis 0, everything after it
        # broadcasts (a bare column and a column block must agree).
        r_col = jnp.geomspace(0.2e-6, 4.0e-6, 6)
        r_block = jnp.tile(r_col[:, None], (1, 3))
        _, vol_col = bcscavcoef(r_col, *self._lookup())
        _, vol_block = bcscavcoef(r_block, *self._lookup())
        self.assertEqual(vol_block.shape, (6, 3))
        np.testing.assert_allclose(
            np.asarray(vol_block),
            np.broadcast_to(np.asarray(vol_col)[:, None], (6, 3)), rtol=1e-6)

    def test_gradient_is_finite_and_positive_above_the_gap(self):
        def f(r):
            return bcscavcoef(r, *self._lookup())[1]

        g = jax.grad(f)(jnp.asarray(1.5e-6))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreater(float(g), 0.0)

    def test_vjp_jvp_agree_with_numerics(self):
        # Differentiate a size SCALE rather than the metre-scale radius, so
        # the finite-difference step stays inside one table interval (the
        # log-linear interpolation is only piecewise differentiable).
        base = jnp.array([0.31e-6, 1.37e-6, 2.90e-6])

        def f(scale):
            num, vol = bcscavcoef(base * scale, *self._lookup())
            return jnp.sum(num + vol)

        args = (jnp.asarray(1.0),)
        check_vjp(f, functools.partial(jax.vjp, f), args=args,
                  rtol=1e-3, atol=1e-3)
        check_jvp(f, functools.partial(jax.jvp, f), args=args,
                  rtol=1e-3, atol=1e-3)


class ImpactionKnobTest(unittest.TestCase):
    """The two collection-integral knobs are live differentiable leaves."""

    def setUp(self):
        self.table = build_impaction_table(2.0e-6, 1.8, 2600.0)

    def test_defaults_reproduce_the_reference_integral(self):
        """At the default knobs the traced kernel is the float64 integral."""
        ln_n, ln_v = table_log_coefficients(
            self.table, MU_WATER_AIR_DEFAULT, IMPACT_SCALE_DEFAULT)
        for k, jgrow in enumerate(range(GROW_MIN, GROW_MAX + 1)):
            dg = 2.0e-6 * np.exp(jgrow * DLN_DG)
            ref_num, ref_vol = impaction_scavenging_rates(dg, 1.8, 2600.0)
            self.assertAlmostEqual(float(jnp.exp(ln_n[k])) / ref_num, 1.0,
                                   places=4, msg=f"node {jgrow}")
            self.assertAlmostEqual(float(jnp.exp(ln_v[k])) / ref_vol, 1.0,
                                   places=4, msg=f"node {jgrow}")

    def test_both_knobs_carry_a_gradient(self):
        def loss(mu, scale):
            ln_n, ln_v = table_log_coefficients(self.table, mu, scale)
            return jnp.sum(ln_n) + jnp.sum(ln_v)

        g_mu, g_scale = jax.grad(loss, argnums=(0, 1))(
            MU_WATER_AIR_DEFAULT, IMPACT_SCALE_DEFAULT)
        for name, g in (("mu_water_air", g_mu), ("impact_scale", g_scale)):
            self.assertTrue(np.isfinite(float(g)), name)
            self.assertNotEqual(float(g), 0.0, name)

    def test_knob_gradients_match_central_differences(self):
        """Autodiff against a central difference, in the knobs' own units.

        ``check_vjp``'s unit-normal probe is not usable here: the efficiency
        carries a ``min(E, 1)`` kink, so a random step straddles it, and the
        sum is evaluated in float32. A central difference at a step well
        above float32 noise and below the kink's scale is the meaningful
        comparison.
        """
        def loss(mu, scale):
            ln_n, ln_v = table_log_coefficients(self.table, mu, scale)
            return jnp.sum(ln_n) + jnp.sum(ln_v)

        mu0, s0 = MU_WATER_AIR_DEFAULT, IMPACT_SCALE_DEFAULT
        g_mu, g_scale = jax.grad(loss, argnums=(0, 1))(mu0, s0)
        for got, fd in (
            (g_mu, (loss(mu0 * 1.01, s0) - loss(mu0 * 0.99, s0))
                   / (0.02 * mu0)),
            (g_scale, (loss(mu0, s0 * 1.01) - loss(mu0, s0 * 0.99))
                      / (0.02 * s0)),
        ):
            self.assertAlmostEqual(float(got) / float(fd), 1.0, delta=0.05)

    def test_impact_scale_raises_collection_monotonically(self):
        """``impact_scale`` only adds a non-negative term, under the E<=1 cap.

        ``mu_water_air`` is deliberately not asserted monotone: in Slinn's
        interception term ``(1 + 2μχ)/(1 + μ/√Re)`` the sign of ∂/∂μ is that
        of ``2χ - 1/√Re``, so raising μ *lowers* collection for particles
        small enough that ``χ < 1/(2√Re)``. It is checked for liveness and
        effect instead.
        """
        base = table_log_coefficients(
            self.table, MU_WATER_AIR_DEFAULT, IMPACT_SCALE_DEFAULT)[1]
        more = table_log_coefficients(
            self.table, MU_WATER_AIR_DEFAULT, 2.0 * IMPACT_SCALE_DEFAULT)[1]
        less = table_log_coefficients(
            self.table, MU_WATER_AIR_DEFAULT, 0.5 * IMPACT_SCALE_DEFAULT)[1]
        self.assertTrue(bool(jnp.all(more >= base - 1e-5)))
        self.assertTrue(bool(jnp.all(less <= base + 1e-5)))
        self.assertGreater(float(jnp.sum(more - base)), 0.0)

        other_mu = table_log_coefficients(
            self.table, 2.0 * MU_WATER_AIR_DEFAULT, IMPACT_SCALE_DEFAULT)[1]
        self.assertGreater(float(jnp.max(jnp.abs(other_mu - base))), 1e-4)


if __name__ == "__main__":
    unittest.main()
