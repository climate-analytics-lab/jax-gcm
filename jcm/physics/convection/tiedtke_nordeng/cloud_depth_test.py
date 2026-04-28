"""Tests for ``cloud_depth_for_target_top``.

The updraft scan ceiling used by ``tiedtke_nordeng_convection`` is
computed from a target cloud-top *pressure* rather than a fixed level
count, so the same physics runs unchanged across vertical resolutions.
These tests guard against three regression modes:

1. **Resolution dependence.** A fixed ``cloud_depth=35`` would silently
   clamp deep convection on coarse grids (e.g. 8-level sigma) and
   under-cap it on fine grids (e.g. 90-level).
2. **Wrong target.** Picking the wrong level near the target pressure
   (e.g. one above instead of just below) shifts the scan range by
   ~22 hPa and biases reported cloud tops.
3. **Edge cases.** Targets above the surface or below the model top
   should not crash or produce nonsensical depths.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    cloud_depth_for_target_top,
)


def _logspace_pressure(nlev: int, p_top_hpa: float = 10.0,
                       p_surf_hpa: float = 1000.0) -> jnp.ndarray:
    """TOA-first pressure profile spanning ``[p_top_hpa, p_surf_hpa]``."""
    p_pa = jnp.logspace(
        jnp.log10(p_top_hpa * 100.0),
        jnp.log10(p_surf_hpa * 100.0),
        nlev,
    )
    return p_pa


class TestCloudDepthForTargetTop(unittest.TestCase):
    """Resolution-independent scan-ceiling derivation."""

    def test_47_level_deep_reaches_above_200_hPa(self):
        """On a 47-level ICON-like grid, the deep target (150 hPa) should
        give a depth that lets the scan reach pressures < 200 hPa from a
        surface cloud base — matching the ECHAM cumastr reach observed
        in the harness on the same RCE column.
        """
        p = _logspace_pressure(47)
        cloud_base = jnp.array(46)  # surface
        depth = cloud_depth_for_target_top(p, cloud_base, 15_000.0)
        # ktop = cloud_base - depth
        ktop = int(cloud_base) - int(depth)
        self.assertLess(
            float(p[ktop]) / 100.0, 200.0,
            f"Scan ceiling at p={float(p[ktop])/100:.1f} hPa is below 200 hPa; "
            "deep convection scan range too shallow."
        )

    def test_8_level_does_not_exceed_nlev_minus_2(self):
        """On a coarse 8-level grid, the depth must clamp to ``nlev-2``
        (= 6) even though the full pressure range is much smaller per
        level. Otherwise the scan extends to TOA, wasting compute and
        risking unphysical extension into the stratosphere.
        """
        p = _logspace_pressure(8)
        cloud_base = jnp.array(7)  # surface
        depth = cloud_depth_for_target_top(p, cloud_base, 15_000.0)
        self.assertLessEqual(
            int(depth), 8 - 2,
            f"depth={int(depth)} on an 8-level grid exceeds nlev-2=6.",
        )

    def test_90_level_scales_up(self):
        """A 90-level grid should produce a much larger ``cloud_depth``
        than 47 levels, since each level is finer in pressure. Using a
        fixed 35 here would cap the scan at ~700 hPa, missing real deep
        convection.
        """
        p_47 = _logspace_pressure(47)
        p_90 = _logspace_pressure(90)
        depth_47 = int(cloud_depth_for_target_top(
            p_47, jnp.array(46), 15_000.0,
        ))
        depth_90 = int(cloud_depth_for_target_top(
            p_90, jnp.array(89), 15_000.0,
        ))
        # 90-level grid has ~roughly twice the pressure resolution.
        self.assertGreater(
            depth_90, depth_47,
            f"90-level depth ({depth_90}) should exceed 47-level depth "
            f"({depth_47}) since each level is finer in pressure."
        )

    def test_shallow_target_is_smaller_than_deep_target(self):
        """A 700 hPa shallow target should give a smaller depth than a
        150 hPa deep target on the same column.
        """
        p = _logspace_pressure(47)
        cloud_base = jnp.array(46)
        deep = int(cloud_depth_for_target_top(p, cloud_base, 15_000.0))
        shallow = int(cloud_depth_for_target_top(p, cloud_base, 70_000.0))
        self.assertLess(
            shallow, deep,
            f"shallow depth ({shallow}) should be smaller than deep depth "
            f"({deep}) — they're using a more permissive (shallower) target."
        )

    def test_min_layers_clamp(self):
        """Even if the target is very close to the cloud base, ``depth``
        must be at least ``min_layers`` (default 2) so the updraft has a
        non-degenerate column to scan.
        """
        p = _logspace_pressure(47)
        cloud_base = jnp.array(46)
        # Target close to cloud_base pressure
        target_close = float(p[44])  # only 2 levels above kbase
        depth = cloud_depth_for_target_top(
            p, cloud_base, target_close, min_layers=5,
        )
        self.assertGreaterEqual(
            int(depth), 5,
            f"depth={int(depth)} below the requested min_layers=5.",
        )

    def test_target_below_surface_does_not_crash(self):
        """A target pressure higher than the surface (nonsensical) should
        not crash; depth gracefully falls back to ``min_layers``.
        """
        p = _logspace_pressure(47)
        cloud_base = jnp.array(46)
        # Target = 1500 hPa is below all model levels
        depth = cloud_depth_for_target_top(p, cloud_base, 150_000.0)
        # Either ``min_layers`` or the level closest to surface
        # (cloud_base itself, depth=0 → clamped to min_layers).
        self.assertGreaterEqual(int(depth), 2)
        self.assertLess(int(depth), 47 - 1)

    def test_target_above_TOA_does_not_crash(self):
        """A target pressure below the model top (zero or below) should
        give a depth that lets the scan reach near-TOA, capped at
        ``nlev-2``.
        """
        p = _logspace_pressure(47)
        cloud_base = jnp.array(46)
        # Target = 0 Pa is below all model levels (above TOA)
        depth = cloud_depth_for_target_top(p, cloud_base, 0.0)
        # Should be allowed up to nlev-2
        self.assertLessEqual(int(depth), 47 - 2)

    def test_cloud_base_at_mid_column(self):
        """Cloud base above the surface — depth should be measured
        relative to ``cloud_base``, not the surface.
        """
        p = _logspace_pressure(47)
        cloud_base = jnp.array(30)  # mid-column (~600 hPa on this grid)
        # Deep target should still allow the scan to reach 150 hPa
        depth = cloud_depth_for_target_top(p, cloud_base, 15_000.0)
        ktop = int(cloud_base) - int(depth)
        self.assertGreaterEqual(ktop, 0)
        self.assertLess(
            float(p[ktop]) / 100.0, 200.0,
            "Scan ceiling didn't reach 200 hPa from a mid-column "
            "cloud base."
        )

    def test_jit_compatible(self):
        """The function must compose with ``jax.jit`` so it can be called
        inside the convection scheme's compiled scan.
        """
        import jax
        jit_depth = jax.jit(
            cloud_depth_for_target_top,
            static_argnames=("min_layers",),
        )
        p = _logspace_pressure(47)
        cloud_base = jnp.array(46)
        depth = jit_depth(p, cloud_base, 15_000.0)
        self.assertGreater(int(depth), 0)


class TestCloudDepthIntegration(unittest.TestCase):
    """End-to-end: the resolution-independent depth lets the convection
    scheme actually run on different grids without artificial truncation.
    """

    def test_47_level_rce_reaches_above_650_hPa(self):
        """The whole tiedtke_nordeng_convection pipeline on an RCE column
        should produce an updraft whose top is above 650 hPa. The
        regression mode this guards against is the original
        ``cloud_depth=15`` cap that limited deep convection to ~750 hPa
        on the 47-level grid (the dynamic termination then often kicks
        in above that, so 650 hPa leaves room for normal termination
        while still flagging the cap regression).
        """
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            tiedtke_nordeng_convection, ConvectionParameters,
            saturation_mixing_ratio,
        )

        nlev = 47
        rd = 287.04
        grav = 9.80665
        p0 = 101325.0
        sigma_bnds = jnp.linspace(1000.0 / p0, 1.0, nlev + 1)
        p_half = sigma_bnds * p0
        p_full = 0.5 * (p_half[:-1] + p_half[1:])

        # Tropical RCE-like sounding: 305 K surface, 7 K/km lapse,
        # 90 % RH (clipped to small at top).
        z = -8400.0 * jnp.log(p_full / p0)
        T = jnp.maximum(305.0 - 7.0e-3 * z, 200.0)
        qs = jnp.array([
            float(saturation_mixing_ratio(jnp.asarray(p_full[k]),
                                           jnp.asarray(T[k])))
            for k in range(nlev)
        ])
        q = 0.9 * qs
        Tv = T * (1.0 + 0.608 * q)
        rho = p_full / (rd * Tv)
        dlnp = jnp.diff(jnp.log(p_half))
        dz = rd * Tv / grav * dlnp

        cfg = ConvectionParameters.default(
            dt_conv=1800.0, entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4,
            entrdd=2.0e-4, tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10,
            cprcon=2.5e-4, cevapcu=2.0e-5, cmfctop=0.20, cmfdeps=0.30,
        )
        tend, state = tiedtke_nordeng_convection(
            T, q, p_full, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
        )
        # Find topmost level with nonzero updraft mass flux
        mfu = np.asarray(state.mfu)
        active = mfu > 1e-10
        if not np.any(active):
            self.skipTest("convection didn't fire on this RCE column")
        top_idx = int(np.where(active)[0].min())
        top_p_hpa = float(p_full[top_idx]) / 100.0
        self.assertLess(
            top_p_hpa, 650.0,
            f"Cloud top reached only {top_p_hpa:.1f} hPa; the deep cloud "
            "scan ceiling is artificially truncating updraft (Bug B "
            "regression — original cloud_depth=15 cap put it at ~750 hPa)."
        )


if __name__ == "__main__":
    unittest.main()
