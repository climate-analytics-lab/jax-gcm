"""ECHAM's deep/shallow split — the moisture-convergence test (#699).

ECHAM classifies a ``cubase`` plume deep (ktype=1) iff the column moisture
convergence exceeds 1.1x the surface latent flux (mo_cumastr.f90:571-574),
and demotes deep to shallow when the realized cloud is thinner than 200 hPa
(line 752). jcm previously used a CAPE sigmoid at 1000 J/kg with no ECHAM
counterpart; these tests pin the reference test, the demotion, and the
lagged reconstruction of the dynamics part of ``pqte`` from the
``_prev_step`` carry.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.saturation import (
    saturation_specific_humidity_and_derivative,
)
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    tiedtke_nordeng_convection,
)
from jcm.physics.convection.tiedtke_nordeng.types import ConvectionParameters

NLEV = 40
DZ = 400.0
E_SFC = 3.0e-5     # surface evaporation [kg/m2/s]


def _unstable_column(cap_hpa=None):
    """TOA-first conditionally unstable column with a well-mixed PBL.

    ``cap_hpa``: optionally make the profile strongly stable above this
    pressure, capping any plume there.
    """
    p = np.linspace(5000.0, 101325.0, NLEV)
    dz = np.full(NLEV, DZ)
    t = np.empty(NLEV)
    t[-1] = 301.0
    for k in range(NLEV - 2, -1, -1):
        lapse = 9.7e-3 if k > NLEV - 5 else 6.3e-3
        if cap_hpa is not None and p[k] < cap_hpa * 100.0:
            lapse = -2.0e-3          # inversion above the cap
        t[k] = t[k + 1] - lapse * dz[k]
    qs, _ = saturation_specific_humidity_and_derivative(jnp.array(t), jnp.array(p))
    q = 0.9 * np.asarray(qs)
    rho = p / (c.rd * t)
    return (jnp.array(t), jnp.array(q), jnp.array(p), jnp.array(dz),
            jnp.array(rho))


def _run(column, qte_dynamics=None, config=None):
    t, q, p, dz, rho = column
    rho_np = np.asarray(rho)
    # The vdiff-supply profile lives in the lowest layers, wherever the
    # SURFACE is in this column's ordering (highest pressure).
    prof = np.zeros(NLEV)
    sfc = slice(-4, None) if float(p[-1]) >= float(p[0]) else slice(0, 4)
    prof[sfc] = E_SFC / (rho_np[sfc] * np.asarray(dz)[sfc]).sum()
    zeros = jnp.zeros(NLEV)
    return tiedtke_nordeng_convection(
        t, q, p, dz, rho, zeros, zeros, zeros, zeros,
        dt=1800.0, config=config or ConvectionParameters.default(),
        moisture_supply=jnp.array(E_SFC),
        moisture_tend_profile=jnp.array(prof),
        qte_dynamics=qte_dynamics,
    )


def _convergence_profile(rho, dz, fraction_of_e):
    """Build a mid-level moisture-convergence profile integrating to f*E."""
    conv = np.zeros(NLEV)
    sl = slice(20, 32)
    conv[sl] = fraction_of_e * E_SFC / (
        np.asarray(rho)[sl] * np.asarray(dz)[sl]).sum()
    return jnp.array(conv)


class TestMoistureConvergenceSplit(unittest.TestCase):
    def test_no_convergence_is_shallow(self):
        """A column fed only by its own surface flux is shallow.

        The vdiff supply integrates to exactly E, so zdqcv - 1.1E = -0.1E
        and ECHAM's test says ktype=2 regardless of CAPE.
        """
        _, state = _run(_unstable_column())
        self.assertEqual(int(state.ktype), 2)

    def test_convergence_makes_it_deep(self):
        """Large-scale moisture convergence beyond 0.1*E flips it deep —
        and the deep plume (entrpen, Nordeng closure) is measurably
        deeper and rains harder than the shallow one.
        """
        col = _unstable_column()
        _, s_sh = _run(col)
        _, s_dp = _run(col, qte_dynamics=_convergence_profile(col[4], col[3], 0.5))
        self.assertEqual(int(s_dp.ktype), 1)
        p = np.asarray(col[2])
        def depth(s):
            mfu = np.asarray(s.mfu)
            top = np.where(mfu > 1e-6)[0]
            return p[int(s.kbase)] - p[top].min() if top.size else 0.0
        self.assertGreater(depth(s_dp), depth(s_sh) + 1.0e4)
        self.assertGreater(float(s_dp.prate), 10.0 * float(s_sh.prate))

    def test_switch_threshold_is_1p1_e(self):
        """The switch sits at zdqcv = 1.1*E: 0.05*E of extra convergence
        stays shallow, 0.2*E goes deep (width 2e-7 << 0.05*E here).
        """
        col = _unstable_column()
        _, s_below = _run(col, qte_dynamics=_convergence_profile(col[4], col[3], 0.05))
        _, s_above = _run(col, qte_dynamics=_convergence_profile(col[4], col[3], 0.2))
        self.assertEqual(int(s_below.ktype), 2)
        self.assertEqual(int(s_above.ktype), 1)

    def test_depth_demotion_relabels_thin_deep(self):
        """ECHAM mo_cumastr.f90:752: deep with a cloud thinner than 200 hPa
        is relabelled shallow, however strong the convergence.
        """
        col = _unstable_column(cap_hpa=850.0)
        _, state = _run(col, qte_dynamics=_convergence_profile(col[4], col[3], 0.5))
        p = np.asarray(col[2])
        mfu = np.asarray(state.mfu)
        top = np.where(mfu > 1e-6)[0]
        if top.size:
            self.assertLess(p[int(state.kbase)] - p[top].min(), 2.0e4,
                            "fixture failed to cap the plume below 200 hPa")
        self.assertNotEqual(int(state.ktype), 1)


class TestOrientationCanonicalization(unittest.TestCase):
    """A surface-first column must give the mirror of the TOA-first result.

    The ascent/descent scans are TOA-first internals; the public entry
    point canonicalizes. Before it did, a surface-first column's plume
    could never propagate past its own base (the scan looked for the
    arriving parcel on the wrong side) — flagged by Codex on the
    mid-level path, but it applied to every trigger.
    """

    def test_surface_first_mirrors_toa_first(self):
        col = _unstable_column()          # TOA-first
        t, q, p, dz, rho = col
        conv = _convergence_profile(rho, dz, 0.5)
        tend_a, state_a = _run(col, qte_dynamics=conv)
        flip = lambda a: a[::-1]
        col_sf = tuple(flip(a) for a in col)
        tend_b, state_b = _run(col_sf, qte_dynamics=flip(conv))

        self.assertEqual(int(state_a.ktype), 1, "fixture must convect deep")
        self.assertEqual(int(state_b.ktype), int(state_a.ktype))
        self.assertEqual(int(state_b.kbase), NLEV - 1 - int(state_a.kbase))
        np.testing.assert_allclose(float(state_b.prate),
                                   float(state_a.prate), rtol=1e-5)
        for field in ("dtedt", "dqdt", "dqc_dt", "dqi_dt"):
            np.testing.assert_allclose(
                np.asarray(getattr(tend_b, field)),
                np.asarray(getattr(tend_a, field))[::-1],
                rtol=1e-5, atol=1e-12, err_msg=field)
        np.testing.assert_allclose(
            np.asarray(state_b.mfu), np.asarray(state_a.mfu)[::-1],
            rtol=1e-5, atol=1e-12)


class TestLaggedDynamicsReconstruction(unittest.TestCase):
    """The wrapper rebuilds dyn = (q_now - q_prev)/dt - q_tend_prev from the
    ``_prev_step`` carry that ComposablePhysics publishes.

    The fake scheme runs under ``jax.vmap``, so the probe value must flow
    OUT through a returned tendency rather than a Python side effect — the
    same pattern as the existing wrapper tests: ``qte_dynamics`` rides
    ``dqdt`` (zero ``dtedt`` keeps the heating cap inactive).
    """

    def _capture(self, diagnostics_extra, monkey):
        import jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng as tn
        from types import SimpleNamespace
        from jcm.physics.convection.tiedtke_nordeng.types import (
            ConvectionTendencies)
        from jcm.physics_interface import PhysicsState
        from jcm.physics.clouds.cloud_data import CloudData

        nlev, ncols = 8, 2
        shape = (nlev, ncols)

        def fake_convection(temperature, humidity, pressure, layer_thickness,
                            air_density, u_wind, v_wind, qc, qi, dt_seconds,
                            params, land_fraction, moisture_supply,
                            moisture_tend_profile, thvsig, omega,
                            qte_dynamics):
            zeros = jnp.zeros_like(temperature)
            return ConvectionTendencies(
                dtedt=zeros, dqdt=qte_dynamics, dudt=zeros, dvdt=zeros,
                qc_conv=zeros, qi_conv=zeros,
                precip_formation=zeros,
                precip_conv=jnp.zeros((), temperature.dtype),
                dqc_dt=zeros, dqi_dt=zeros,
            ), tn.initialize_convection(
                temperature, humidity, pressure, u_wind, v_wind, params)

        monkey.setattr(tn, "tiedtke_nordeng_convection", fake_convection)
        state = PhysicsState.zeros(
            shape,
            temperature=jnp.full(shape, 280.0),
            specific_humidity=jnp.full(shape, 5e-3),
            tracers={"qc": jnp.zeros(shape), "qi": jnp.zeros(shape)},
        )
        pf = jnp.broadcast_to(
            jnp.linspace(10000.0, 100000.0, nlev)[:, None], shape)
        diagnostics = {
            "_dt_seconds": 900.0,
            "pressure_full": pf,
            "layer_thickness": jnp.full(shape, 800.0),
            "air_density": pf / (c.rd * 280.0),
            "clouds": CloudData.zeros((ncols,), nlev),
            **diagnostics_extra,
        }
        terrain = SimpleNamespace(fmask=jnp.zeros(ncols))
        tend, _ = tn.TiedtkeConvection()(
            state, diagnostics, forcing=None, terrain=terrain)
        return tend.specific_humidity, state       # (nlev, ncols)

    def test_reconstruction_from_prev_step(self):
        import pytest
        monkey = pytest.MonkeyPatch()
        try:
            nlev, ncols, dt = 8, 2, 900.0
            q_prev = jnp.full((nlev, ncols), 4.5e-3)
            q_tend_prev = jnp.full((nlev, ncols), 1e-7)
            qte, state = self._capture({
                "_prev_step": {"specific_humidity": q_prev,
                               "q_tendency": q_tend_prev},
            }, monkey)
            expected = (state.specific_humidity - q_prev) / dt - q_tend_prev
            np.testing.assert_allclose(np.asarray(qte), np.asarray(expected),
                                       rtol=1e-6)
        finally:
            monkey.undo()

    def test_zero_template_reads_as_no_information(self):
        """Step 1: the scan carry template is all zeros — a state the model
        cannot produce — and must NOT be read as a huge drying tendency.
        """
        import pytest
        monkey = pytest.MonkeyPatch()
        try:
            nlev, ncols = 8, 2
            qte, _ = self._capture({
                "_prev_step": {
                    "specific_humidity": jnp.zeros((nlev, ncols)),
                    "q_tendency": jnp.zeros((nlev, ncols))},
            }, monkey)
            np.testing.assert_array_equal(np.asarray(qte), 0.0)
        finally:
            monkey.undo()

    def test_absent_carry_reads_as_no_information(self):
        import pytest
        monkey = pytest.MonkeyPatch()
        try:
            qte, _ = self._capture({}, monkey)
            np.testing.assert_array_equal(np.asarray(qte), 0.0)
        finally:
            monkey.undo()


class TestPrevStepPublication(unittest.TestCase):
    def test_composable_publishes_and_output_excludes(self):
        """ComposablePhysics publishes ``_prev_step`` {q, dq/dt} and the
        xarray flattener drops it.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.forcing import ForcingData
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData

        physics = echam_physics(radiation_scheme="grey")
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        physics.cache_coords(coords)
        nlev = 8
        nlon, nlat = coords.horizontal.nodal_shape
        state = PhysicsState.zeros(
            (nlev, nlon, nlat),
            temperature=jnp.full((nlev, nlon, nlat), 280.0),
            specific_humidity=jnp.full((nlev, nlon, nlat), 4e-3),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={spec.name: jnp.zeros((nlev, nlon, nlat))
                     for spec in physics.required_tracers()},
        )
        tend, diag = physics.compute_tendencies(
            state, ForcingData.zeros((nlon, nlat)),
            TerrainData.aquaplanet(coords),
        )
        self.assertIn("_prev_step", diag)
        ncols = nlon * nlat
        np.testing.assert_allclose(
            np.asarray(diag["_prev_step"]["specific_humidity"]),
            np.asarray(state.specific_humidity).reshape(nlev, ncols))
        np.testing.assert_allclose(
            np.asarray(diag["_prev_step"]["q_tendency"]),
            np.asarray(tend.specific_humidity).reshape(nlev, ncols),
            atol=1e-12)
        flat = physics.data_struct_to_dict(diag)
        self.assertFalse(
            [k for k in flat if "prev_step" in k],
            "the _prev_step carry leaked into user-facing output")


if __name__ == "__main__":
    unittest.main()
