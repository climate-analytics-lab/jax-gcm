"""ECHAM ``cubasmc`` — the mid-level convection trigger (#697).

``cubase`` (the surface-parcel ``klab`` walk, #684) is deliberately strict:
the parcel must stay buoyant from the lowest level all the way to its LCL,
so it only fires under a moist, well-mixed boundary layer. ECHAM's second
trigger is what covers everything else — elevated convection above a stable
layer, warm-conveyor and frontal ascent, nocturnal elevated convection over
land. These tests pin the reference's four gates, the environment-seeded
parcel, the omega-derived mass flux, and the ``zlift`` bonus that becomes
reachable again only through this path.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.saturation import (
    saturation_specific_humidity_and_derivative,
)
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    find_cloud_base,
    find_midlevel_cloud_base,
    midlevel_mass_flux,
    tiedtke_nordeng_convection,
)
from jcm.physics.convection.tiedtke_nordeng.types import ConvectionParameters


NLEV = 40
DZ = 400.0


def _elevated_column(rh_mid=0.95, w_mid=-0.3, mid_slice=slice(12, 15)):
    """Surface-first column with an inversion below and a rising moist layer.

    The strong low-level inversion is the point: it guarantees ``cubase``
    finds nothing, so anything that convects here came from ``cubasmc``.
    Above the inversion the column is statically stable (6.5 K/km on the
    column's own hydrostatic heights): the half-level environment ECHAM's
    ``cuini`` builds enforces a non-decreasing dry static energy upward, so a
    dry-unstable jump above the inversion would carry the inversion's warmth
    up the whole column.
    """
    p = np.linspace(101325.0, 5000.0, NLEV)
    z = -7.6e3 * np.log(p / p[0])               # hypsometric height [m]
    t = np.empty(NLEV)
    t[:4] = 300.0 + 2.0 * np.arange(4)          # capping inversion
    t[4:] = t[3] - 6.5e-3 * (z[4:] - z[3])
    qs, _ = saturation_specific_humidity_and_derivative(jnp.array(t), jnp.array(p))
    qs = np.asarray(qs)
    q = 0.5 * qs
    q[mid_slice] = rh_mid * qs[mid_slice]
    w = np.zeros(NLEV)
    w[mid_slice] = w_mid
    dz = np.full(NLEV, DZ)
    return (jnp.array(t), jnp.array(q), jnp.array(p), jnp.array(w),
            jnp.array(dz))


class TestMidLevelTriggerGates(unittest.TestCase):
    """The four ``cubasmc`` conditions (mo_cuascent.f90:631-634)."""

    def setUp(self):
        self.cfg = ConvectionParameters.default()

    def _find(self, t, q, p, w, dz, cfg=None):
        return find_midlevel_cloud_base(t, q, p, w, dz, cfg or self.cfg)

    def test_fires_where_cubase_cannot(self):
        t, q, p, w, dz = _elevated_column()
        _, has_sfc = find_cloud_base(t, q, p, self.cfg, None, dz)
        self.assertFalse(bool(has_sfc),
                         "the capping inversion should defeat cubase")

        base, found = self._find(t, q, p, w, dz)
        self.assertTrue(bool(found))
        # The LOWEST qualifying level wins (ECHAM scans bottom-up and the
        # first hit sets kcbot).
        self.assertEqual(int(base), 12)

    def test_requires_ascent(self):
        """``pverv < 0``. Subsidence, or no omega at all, means no trigger."""
        t, q, p, _, dz = _elevated_column()
        for name, w in (("subsidence", jnp.full(NLEV, +0.3)),
                        ("no provider", jnp.zeros(NLEV))):
            with self.subTest(name):
                _, found = self._find(t, q, p, w, dz)
                self.assertFalse(bool(found))

    def test_requires_near_saturation(self):
        """``pqen > 0.90 pqsen``: 85 % RH is not enough, 95 % is."""
        t, q, p, w, dz = _elevated_column(rh_mid=0.85)
        _, found = self._find(t, q, p, w, dz)
        self.assertFalse(bool(found))

        t, q, p, w, dz = _elevated_column(rh_mid=0.95)
        _, found = self._find(t, q, p, w, dz)
        self.assertTrue(bool(found))

    def test_excludes_the_boundary_layer(self):
        """``pgeoh(kk)/grav > 1500 m`` keeps the trigger off cubase's turf.

        A saturated, rising layer at 400-1200 m is inside the boundary
        layer, which is the surface parcel's domain; ECHAM will not start a
        mid-level plume there.
        """
        t, q, p, w, dz = _elevated_column(mid_slice=slice(1, 3))
        _, found = self._find(t, q, p, w, dz)
        self.assertFalse(bool(found))

    def test_excludes_bases_above_300_hpa(self):
        """ECHAM's ``nmctop``: no mid-level cloud base at or above 300 hPa."""
        t, q, p, w, dz = _elevated_column(mid_slice=slice(32, 35))
        self.assertLess(float(p[33]), 30_000.0)   # fixture really is up there
        _, found = self._find(t, q, p, w, dz)
        self.assertFalse(bool(found))

    def test_lmfmid_switch_disables_it(self):
        """ECHAM's own namelist switch, the escape hatch for an omega-less
        backend (see ``TiedtkeConvection.__init__``).
        """
        t, q, p, w, dz = _elevated_column()
        off = ConvectionParameters.default(cu_lmfmid=False)
        _, found = self._find(t, q, p, w, dz, cfg=off)
        self.assertFalse(bool(found))

    def test_ordering_agnostic(self):
        """Same answer whether the column arrives surface-first or TOA-first."""
        t, q, p, w, dz = _elevated_column()
        base_sf, found_sf = self._find(t, q, p, w, dz)
        flip = lambda a: a[::-1]
        base_tf, found_tf = self._find(*(flip(a) for a in (t, q, p, w, dz)))
        self.assertEqual(bool(found_sf), bool(found_tf))
        # Same physical level, expressed in each ordering.
        self.assertEqual(int(base_tf), NLEV - 1 - int(base_sf))


class TestMidLevelSurvivalRetry(unittest.TestCase):
    """ECHAM retries one level up when a seeded plume dies immediately.

    ``cubasmc`` seeds at the lowest qualifying level, but if the plume is
    unbuoyant at its first ascent step the ascent sets ``klab = 0`` there and
    the next loop iteration lets ``cubasmc`` seed again one level higher. The
    net rule is "lowest qualifying level whose plume survives its first
    step", which is what the survival term encodes.
    """

    def test_skips_a_level_capped_from_above(self):
        cfg = ConvectionParameters.default()
        t, q, p, w, dz = _elevated_column(mid_slice=slice(12, 18))
        base_ref, found_ref = find_midlevel_cloud_base(t, q, p, w, dz, cfg)
        self.assertTrue(bool(found_ref))
        self.assertEqual(int(base_ref), 12)

        # Hold layer 12 at 91 % RH: still an eligible seed (> 0.90), but the
        # one-layer lift of its first ascent step (to the layer's top
        # interface) cools it too little to condense, so ECHAM sets
        # ``klab = 0`` there and ``cubasmc`` re-seeds in the next layer up,
        # which at 95 % RH does condense and is buoyant.
        qs, _ = saturation_specific_humidity_and_derivative(t, p)
        q_dry12 = q.at[12].set(0.91 * qs[12])
        base, found = find_midlevel_cloud_base(t, q_dry12, p, w, dz, cfg)
        self.assertTrue(bool(found), "the trigger should retry, not give up")
        self.assertEqual(int(base), 13)

    def test_no_qualifying_level_survives(self):
        """A uniformly capped moist layer gets nothing at all."""
        cfg = ConvectionParameters.default()
        t, q, p, w, dz = _elevated_column(mid_slice=slice(12, 15))
        # Make the whole candidate range strongly stable above each seed.
        t_stable = t.at[12:17].add(jnp.array([0.0, 15.0, 30.0, 45.0, 60.0]))
        _, found = find_midlevel_cloud_base(t_stable, q, p, w, dz, cfg)
        self.assertFalse(bool(found))


class TestMidLevelMassFlux(unittest.TestCase):
    """``zzzmb = MIN(cmfcmax, MAX(cmfcmin, -pverv/grav))``."""

    def test_is_the_resolved_ascent(self):
        cfg = ConvectionParameters.default()
        for omega in (-0.05, -0.3, -1.0):
            with self.subTest(omega=omega):
                self.assertAlmostEqual(
                    float(midlevel_mass_flux(jnp.array(omega), cfg)),
                    -omega / c.grav, places=6,
                )

    def test_clipped_to_the_echam_bounds(self):
        cfg = ConvectionParameters.default()
        # cmfcmax = 1.0 kg/m2/s; -omega/g reaches that at omega ~ -9.81 Pa/s.
        self.assertAlmostEqual(
            float(midlevel_mass_flux(jnp.array(-50.0), cfg)), 1.0, places=6)
        # Subsidence would give a negative flux; the floor is cmfcmin.
        self.assertAlmostEqual(
            float(midlevel_mass_flux(jnp.array(+1.0), cfg)),
            float(cfg.cmfcmin), places=12)


class TestMidLevelEndToEnd(unittest.TestCase):
    """The whole column scheme, driven only by the mid-level trigger."""

    def _run(self, omega):
        t, q, p, w, dz = _elevated_column()
        w = omega if omega is not None else w
        rho = np.asarray(p) / (c.rd * np.asarray(t))
        zeros = jnp.zeros(NLEV)
        return tiedtke_nordeng_convection(
            t, q, p, dz, jnp.array(rho), zeros, zeros, zeros, zeros,
            dt=1800.0, config=ConvectionParameters.default(),
            omega=w,
        )

    def test_convects_and_is_labelled_mid_level(self):
        _, state = self._run(omega=None)
        self.assertEqual(int(state.ktype), 3,
                         "a cubasmc plume is ktype=3 by definition")
        self.assertEqual(int(state.kbase), 12)

    def test_parcel_is_seeded_from_the_environment_not_the_surface(self):
        """``pqu = pqen(kk)`` — the plume has no surface connection.

        Seeding from the surface would hand this plume the boundary layer's
        much larger mixing ratio — water it never had. ``cubasmc`` seeds at
        the BOTTOM interface of the seeding layer ``kk = kcbot``, which in
        this surface-first column is the top interface of layer ``kk − 1``;
        the seed temperature is layer kk's brought dry-adiabatically down
        that half layer.
        """
        t, q, p, _, _ = _elevated_column()
        _, state = self._run(omega=None)
        kb = int(state.kbase)
        seed = kb - 1
        self.assertAlmostEqual(float(state.qu[seed]), float(q[kb]), places=9)
        self.assertGreater(float(state.tu[seed]), float(t[kb]))
        self.assertLess(float(state.tu[seed]), float(t[kb]) + 2.0)
        # Not a tautology: the two candidate parcels are far apart, so the
        # assertions above genuinely discriminate between them.
        self.assertGreater(abs(float(q[kb]) - float(q[0])), 0.3 * float(q[0]))
        self.assertLess(float(t[kb]), float(t[0]) - 5.0)

    def test_mass_flux_comes_from_omega_not_a_cape_closure(self):
        """Halving the ascent halves the cloud-base mass flux exactly.

        A CAPE or moisture-budget closure would be indifferent to omega;
        this is the signature of ECHAM leaving ``ktype == 3`` alone (the
        Nordeng rescale is gated on ``ktype == 1``, mo_cumastr.f90:898).
        """
        t, q, p, w, dz = _elevated_column()
        _, s_full = self._run(omega=w)
        _, s_half = self._run(omega=w * 0.5)
        kb = int(s_full.kbase)
        self.assertEqual(kb, int(s_half.kbase))
        self.assertAlmostEqual(
            float(s_half.mfu[kb]) / float(s_full.mfu[kb]), 0.5, places=5)

    def test_no_ascent_no_convection(self):
        _, state = self._run(omega=jnp.zeros(NLEV))
        self.assertEqual(int(state.ktype), 0)

    def test_omega_defaults_to_dormant(self):
        """The ``omega=None`` caller path is off, not accidentally on."""
        t, q, p, w, dz = _elevated_column()
        rho = np.asarray(p) / (c.rd * np.asarray(t))
        zeros = jnp.zeros(NLEV)
        _, state = tiedtke_nordeng_convection(
            t, q, p, dz, jnp.array(rho), zeros, zeros, zeros, zeros,
            dt=1800.0, config=ConvectionParameters.default(),
        )
        self.assertEqual(int(state.ktype), 0)


class TestSurfacePathUnaffected(unittest.TestCase):
    """``cubase`` wins wherever it fires — ECHAM only reaches ``cubasmc``
    where ``klab == 0``, i.e. where no surface plume exists.
    """

    def test_surface_plume_keeps_its_own_base_and_type(self):
        # Well-mixed boundary layer under a moist, rising mid-level layer:
        # both triggers would fire, and the surface one must win.
        p = np.linspace(101325.0, 5000.0, NLEV)
        z = -7.6e3 * np.log(p / p[0])                  # hypsometric height
        t = 300.0 - 9.7e-3 * z                         # near-dry-adiabatic PBL
        t[3:] = t[3] - 6.0e-3 * (z[3:] - z[3])
        qs, _ = saturation_specific_humidity_and_derivative(
            jnp.array(t), jnp.array(p))
        qs = np.asarray(qs)
        q = 0.80 * qs
        q[12:15] = 0.95 * qs[12:15]
        w = np.zeros(NLEV)
        w[12:15] = -0.3
        dz = np.full(NLEV, DZ)
        cfg = ConvectionParameters.default()

        base_sfc, has_sfc = find_cloud_base(
            jnp.array(t), jnp.array(q), jnp.array(p), cfg, None, jnp.array(dz))
        base_mid, has_mid = find_midlevel_cloud_base(
            jnp.array(t), jnp.array(q), jnp.array(p), jnp.array(w),
            jnp.array(dz), cfg)
        self.assertTrue(bool(has_sfc), "fixture must have a surface plume")
        self.assertTrue(bool(has_mid), "fixture must also qualify mid-level")

        rho = p / (c.rd * t)
        zeros = jnp.zeros(NLEV)
        _, state = tiedtke_nordeng_convection(
            jnp.array(t), jnp.array(q), jnp.array(p), jnp.array(dz),
            jnp.array(rho), zeros, zeros, zeros, zeros,
            dt=1800.0, config=cfg, omega=jnp.array(w),
            moisture_supply=jnp.array(1e-4),
        )
        self.assertEqual(int(state.kbase), int(base_sfc))
        self.assertNotEqual(int(state.ktype), 3)
        self.assertNotEqual(int(base_sfc), int(base_mid))


if __name__ == "__main__":
    unittest.main()
