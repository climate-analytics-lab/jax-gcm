"""Tests for the AeroCom phase-4 diagnostics term."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.diagnostics.aerocom import (
    OVERLAP_MAXIMUM,
    OVERLAP_MAXIMUM_RANDOM,
    OVERLAP_RANDOM,
    THRES_CLD,
    THRES_COD,
    AerocomDiagnostics,
    _column_integral,
    _interp_to_pressure,
    _lognormal_number_above,
    cloud_top_sample,
)


def _reference_cloud_top(cod3d, f3d, t3d, phase3d, cdr3d, icr3d, cdnc3d, overlap):
    """Literal NumPy transcription of the protocol's Fortran, for comparison.

    Deliberately written as the nested loop the protocol publishes, so it is
    an independent check on the vectorised ``lax.scan`` implementation
    rather than a restatement of it.
    """
    nz, nx = f3d.shape
    if overlap in (OVERLAP_RANDOM, OVERLAP_MAXIMUM_RANDOM):
        clt = np.ones(nx)
    else:
        clt = np.zeros(nx)
    icc = np.zeros(nx)
    lcc = np.zeros(nx)
    ttop = np.zeros(nx)
    cdr = np.zeros(nx)
    icr = np.zeros(nx)
    cdnc = np.zeros(nx)

    for i in range(nx):
        for k in range(1, nz):  # uppermost layer assumed cloud-free
            if cod3d[k, i] > THRES_COD and f3d[k, i] > THRES_CLD:
                if overlap == OVERLAP_MAXIMUM:
                    flag_max = -1.0
                    ftmp = max(clt[i], f3d[k, i])
                elif overlap == OVERLAP_RANDOM:
                    flag_max = 1.0
                    ftmp = clt[i] * (1 - f3d[k, i])
                else:
                    flag_max = 1.0
                    ftmp = (clt[i] * (1 - max(f3d[k, i], f3d[k - 1, i]))
                            / (1 - min(f3d[k - 1, i], 1 - THRES_CLD)))
                w = (clt[i] - ftmp) * flag_max
                ttop[i] += t3d[k, i] * w
                icr[i] += icr3d[k, i] * (1 - phase3d[k, i]) * w
                icc[i] += (1 - phase3d[k, i]) * w
                cdr[i] += cdr3d[k, i] * phase3d[k, i] * w
                cdnc[i] += cdnc3d[k, i] * phase3d[k, i] * w
                lcc[i] += phase3d[k, i] * w
                clt[i] = ftmp
        if overlap in (OVERLAP_RANDOM, OVERLAP_MAXIMUM_RANDOM):
            clt[i] = 1.0 - clt[i]
    return dict(clt=clt, ttop=ttop, icr=icr, icc=icc, cdr=cdr, cdnc=cdnc, lcc=lcc)


class CloudTopSamplingTest(unittest.TestCase):
    """The cloud-top scan must reproduce the protocol's published Fortran."""

    def _random_profile(self, seed=0, nz=12, nx=7):
        rng = np.random.default_rng(seed)
        f3d = rng.uniform(0.0, 1.0, (nz, nx))
        f3d[0] = 0.0  # protocol assumes the uppermost layer is cloud-free
        cod3d = rng.uniform(0.0, 5.0, (nz, nx))
        t3d = rng.uniform(200.0, 300.0, (nz, nx))
        phase3d = rng.uniform(0.0, 1.0, (nz, nx))
        cdr3d = rng.uniform(1e-6, 2e-5, (nz, nx))
        icr3d = rng.uniform(1e-5, 1e-4, (nz, nx))
        cdnc3d = rng.uniform(1e6, 3e8, (nz, nx))
        return cod3d, f3d, t3d, phase3d, cdr3d, icr3d, cdnc3d

    def test_matches_protocol_reference_all_overlaps(self):
        arrays = self._random_profile()
        for overlap in (OVERLAP_MAXIMUM, OVERLAP_RANDOM, OVERLAP_MAXIMUM_RANDOM):
            with self.subTest(overlap=overlap):
                got = cloud_top_sample(
                    *[jnp.asarray(a) for a in arrays], overlap=overlap)
                want = _reference_cloud_top(*arrays, overlap=overlap)
                for key in want:
                    np.testing.assert_allclose(
                        np.asarray(got[key]), want[key], rtol=1e-5, atol=1e-8,
                        err_msg=f"{overlap}/{key}")

    def test_clear_column_gives_zero_cover(self):
        """No visible cloud anywhere -> clt = 0 and all sums zero."""
        nz, nx = 8, 3
        z = jnp.zeros((nz, nx))
        got = cloud_top_sample(cod3d=z, f3d=z, t3d=jnp.full((nz, nx), 250.0),
                               phase3d=z, cdr3d=z, icr3d=z, cdnc3d=z)
        for key in ("clt", "ttop", "cdr", "icr", "cdnc", "lcc", "icc"):
            np.testing.assert_allclose(np.asarray(got[key]), 0.0, atol=1e-12)

    def test_single_opaque_overcast_layer(self):
        """One overcast opaque liquid layer -> clt=1 and ttop is its temperature."""
        nz, nx = 6, 2
        f = np.zeros((nz, nx))
        f[3] = 1.0
        cod = np.zeros((nz, nx))
        cod[3] = 10.0
        t = np.full((nz, nx), 250.0)
        t[3] = 275.0
        phase = np.zeros((nz, nx))
        phase[3] = 1.0  # all liquid
        cdr = np.zeros((nz, nx))
        cdr[3] = 1e-5
        cdnc = np.zeros((nz, nx))
        cdnc[3] = 1e8
        got = cloud_top_sample(
            cod3d=jnp.asarray(cod), f3d=jnp.asarray(f), t3d=jnp.asarray(t),
            phase3d=jnp.asarray(phase), cdr3d=jnp.asarray(cdr),
            icr3d=jnp.zeros((nz, nx)), cdnc3d=jnp.asarray(cdnc))
        np.testing.assert_allclose(np.asarray(got["clt"]), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(got["ttop"]), 275.0, atol=1e-4)
        np.testing.assert_allclose(np.asarray(got["lcc"]), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(got["icc"]), 0.0, atol=1e-6)
        # Grid-mean (not in-cloud): overcast so they coincide here.
        np.testing.assert_allclose(np.asarray(got["cdr"]), 1e-5, rtol=1e-5)

    def test_thresholds_exclude_thin_and_transparent_cloud(self):
        """Layers below either threshold contribute nothing."""
        nz, nx = 5, 1
        f = np.zeros((nz, nx))
        cod = np.zeros((nz, nx))
        f[2] = THRES_CLD / 2.0      # too thin
        cod[2] = 10.0
        f[3] = 1.0
        cod[3] = THRES_COD / 2.0    # too transparent
        got = cloud_top_sample(
            cod3d=jnp.asarray(cod), f3d=jnp.asarray(f),
            t3d=jnp.full((nz, nx), 260.0), phase3d=jnp.ones((nz, nx)),
            cdr3d=jnp.ones((nz, nx)), icr3d=jnp.ones((nz, nx)),
            cdnc3d=jnp.ones((nz, nx)))
        np.testing.assert_allclose(np.asarray(got["clt"]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(got["ttop"]), 0.0, atol=1e-12)

    def test_gradients_are_finite(self):
        """Reverse-mode through the scan must stay finite (degenerate inputs)."""
        nz, nx = 6, 3
        f = jnp.zeros((nz, nx)).at[2:4].set(0.5)
        cod = jnp.zeros((nz, nx)).at[2:4].set(1.0)

        def loss(cloud_frac):
            out = cloud_top_sample(
                cod3d=cod, f3d=cloud_frac, t3d=jnp.full((nz, nx), 260.0),
                phase3d=jnp.full((nz, nx), 0.5), cdr3d=jnp.full((nz, nx), 1e-5),
                icr3d=jnp.full((nz, nx), 3e-5), cdnc3d=jnp.full((nz, nx), 1e8))
            return jnp.sum(out["ttop"]) + jnp.sum(out["clt"])

        g = jax.grad(loss)(f)
        self.assertTrue(bool(jnp.isfinite(g).all()))

    def test_rejects_unknown_overlap(self):
        z = jnp.zeros((3, 2))
        with self.assertRaises(ValueError):
            cloud_top_sample(z, z, z, z, z, z, z, overlap="bogus")


class HelperTest(unittest.TestCase):

    def test_column_integral_of_uniform_field(self):
        """Int q dp/g of a constant q is q * (ps - ptop) / g."""
        import jcm.constants as c
        nlev, nx = 10, 4
        p_half = jnp.linspace(1000.0, 100000.0, nlev + 1)[:, None] * jnp.ones((1, nx))
        q = jnp.full((nlev, nx), 3e-3)
        got = _column_integral(q, p_half)
        want = 3e-3 * (100000.0 - 1000.0) / c.grav
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6)

    def test_interp_to_pressure_recovers_linear_in_logp(self):
        """A field linear in log(p) is interpolated exactly."""
        nlev, nx = 20, 3
        p = jnp.exp(jnp.linspace(jnp.log(1000.0), jnp.log(101000.0), nlev))
        p_full = p[:, None] * jnp.ones((1, nx))
        field = 2.0 * jnp.log(p_full) + 1.0
        for target in (20000.0, 70000.0, 50000.0):
            got = _interp_to_pressure(field, p_full, target)
            want = 2.0 * np.log(target) + 1.0
            np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5)

    def test_interp_clamps_below_ground(self):
        """A target below the surface returns the near-surface value, not an
        extrapolation.
        """
        nlev, nx = 8, 2
        p = jnp.linspace(5000.0, 60000.0, nlev)[:, None] * jnp.ones((1, nx))
        field = jnp.arange(nlev, dtype=jnp.float32)[:, None] * jnp.ones((1, nx))
        got = _interp_to_pressure(field, p, 90000.0)  # below the deepest level
        np.testing.assert_allclose(np.asarray(got), float(nlev - 1), rtol=1e-6)

    def test_lognormal_number_above_limits(self):
        """Threshold far below/above the mode returns all/none of the number."""
        n = jnp.full((3,), 1e8)
        r = jnp.full((3,), 50e-9)     # 100 nm diameter mode
        sg = jnp.asarray(1.6)
        almost_all = _lognormal_number_above(n, r, sg, 1e-12)
        almost_none = _lognormal_number_above(n, r, sg, 1e-3)
        np.testing.assert_allclose(np.asarray(almost_all), 1e8, rtol=1e-4)
        self.assertLess(float(jnp.max(almost_none)), 1.0)
        # At the median diameter exactly half the number is above.
        half = _lognormal_number_above(n, r, sg, 100e-9)
        np.testing.assert_allclose(np.asarray(half), 0.5e8, rtol=1e-4)

    def test_lognormal_zero_mode_is_inert_and_grad_safe(self):
        """An empty mode contributes nothing and does not poison gradients."""
        n = jnp.zeros((2,))
        r = jnp.zeros((2,))
        out = _lognormal_number_above(n, r, jnp.asarray(1.8), 70e-9)
        np.testing.assert_allclose(np.asarray(out), 0.0)
        g = jax.grad(lambda nn: jnp.sum(
            _lognormal_number_above(nn, r, jnp.asarray(1.8), 70e-9)))(n)
        self.assertTrue(bool(jnp.isfinite(g).all()))


class TermConfigTest(unittest.TestCase):

    def test_rejects_unknown_group(self):
        with self.assertRaises(ValueError):
            AerocomDiagnostics(groups=("cloud", "not-a-group"))

    def test_rejects_unknown_overlap(self):
        with self.assertRaises(ValueError):
            AerocomDiagnostics(overlap="sideways")

    def test_declared_metadata(self):
        term = AerocomDiagnostics()
        self.assertEqual(term.category, "diagnostics")
        self.assertIn("clouds", term.requires)
        self.assertIn("aerocom_clt", term.provides)


if __name__ == "__main__":
    unittest.main()


class EndToEndTest(unittest.TestCase):
    """The term must run inside a real ECHAM physics step and emit finite fields."""

    def test_runs_in_echam_physics_and_emits_diagnostics(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        model = Model(
            coords=coords, terrain=TerrainData.aquaplanet(coords), time_step=900.0,
            physics=echam_physics(
                radiation_scheme="grey", cloud_scheme="2m",
                enable_aerocom=True,
                aerocom_groups=("cloud", "column", "plev")),
        )
        names = [t.name for t in model.physics.terms]
        self.assertIn("aerocom_diagnostics", names)
        # The diagnostics term must be terminal — nothing may depend on it.
        self.assertEqual(names[-1], "aerocom_diagnostics")

        ds = model.run(total_time=0.05, save_interval=0.05).to_xarray()
        emitted = [k for k in ds.data_vars if "aerocom" in k]
        self.assertTrue(emitted,
                        f"no aerocom_* diagnostics in output: {list(ds.data_vars)[:5]}")
        for key in emitted:
            arr = np.asarray(ds[key])
            self.assertTrue(np.isfinite(arr).all(), f"{key} not finite")
        # Physical sanity on the headline fields.
        clt = np.asarray(ds["aerocom_clt"])
        self.assertGreaterEqual(clt.min(), -1e-6)
        self.assertLessEqual(clt.max(), 1.0 + 1e-6)
        for key in ("aerocom_lwp", "aerocom_iwp", "aerocom_prw", "aerocom_cod"):
            self.assertGreaterEqual(np.asarray(ds[key]).min(), -1e-9, key)

    def test_no_tendency_is_applied(self):
        """Adding the term must not change the model trajectory."""
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        def run(enable):
            coords = get_coords(get_echam_levels(47), spectral_truncation=21)
            m = Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                      time_step=900.0,
                      physics=echam_physics(radiation_scheme="grey",
                                            cloud_scheme="2m",
                                            enable_aerocom=enable))
            return m.run(total_time=0.05, save_interval=0.05).to_xarray()

        off, on = run(False), run(True)
        np.testing.assert_allclose(
            np.asarray(on["temperature"]), np.asarray(off["temperature"]),
            rtol=1e-10, atol=1e-10,
            err_msg="AerocomDiagnostics perturbed the trajectory; it must be "
                    "diagnostic-only")


class CodexRegressionTest(unittest.TestCase):
    """Regressions for the P1 findings on PR #582."""

    def test_cdnc_uses_prognostic_qnc_when_2m_is_active(self):
        """2M leaves CloudData.droplet_number zero; cdnc must not be zero.

        Codex #582: the advertised echam-jam-aerocom preset runs the
        2-moment scheme, which carries qnc (kg^-1) and never populates
        CloudData.droplet_number, so reading the latter reported zero CDNC
        throughout.
        """
        term = AerocomDiagnostics()

        class _Clouds:
            droplet_number = jnp.zeros((4, 3))  # as the 2M scheme leaves it

        class _State:
            tracers = {"qnc": jnp.full((4, 3), 2.0e8), "qni": jnp.full((4, 3), 1.0e6)}

        rho = jnp.full((4, 3), 1.2)
        cdnc_m3, qnc, qni = term._number_concentrations(
            _State(), {"air_density": rho}, _Clouds())
        np.testing.assert_allclose(np.asarray(cdnc_m3), 2.0e8 * 1.2, rtol=1e-6)
        self.assertIsNotNone(qni)

    def test_cdnc_falls_back_to_1m_volumetric_field(self):
        """With no qnc tracer (1M scheme) the m^-3 CloudData field is used."""
        term = AerocomDiagnostics()

        class _Clouds:
            droplet_number = jnp.full((4, 3), 5.0e7)  # m^-3, as 1M writes it

        class _State:
            tracers: dict = {}

        cdnc_m3, qnc, _ = term._number_concentrations(
            _State(), {"air_density": jnp.full((4, 3), 1.0)}, _Clouds())
        np.testing.assert_allclose(np.asarray(cdnc_m3), 5.0e7, rtol=1e-6)
        # And the per-mass form is recovered for the dp/g column integral.
        np.testing.assert_allclose(np.asarray(qnc), 5.0e7, rtol=1e-6)

    def test_column_number_integrates_per_mass_tracer_with_dp_over_g(self):
        """Column number must be sum(qnc[kg^-1] * dp/g), giving m^-2."""
        import jcm.constants as c
        nlev, nx = 6, 2
        p_half = jnp.linspace(1000.0, 101000.0, nlev + 1)[:, None] * jnp.ones((1, nx))
        qnc = jnp.full((nlev, nx), 1.0e8)
        term = AerocomDiagnostics(groups=("column",))

        class _State:
            specific_humidity = jnp.zeros((nlev, nx))

        out = term._column_group(_State(), {}, p_half, qnc, None)
        want = 1.0e8 * (101000.0 - 1000.0) / c.grav
        np.testing.assert_allclose(np.asarray(out["aerocom_cdnum"]), want, rtol=1e-6)

    def test_icnum_is_emitted_from_the_2m_ice_tracer(self):
        """CloudData has no ice_number; icnum must come from the qni tracer."""
        nlev, nx = 5, 2
        p_half = jnp.linspace(1000.0, 101000.0, nlev + 1)[:, None] * jnp.ones((1, nx))
        term = AerocomDiagnostics(groups=("column",))

        class _State:
            specific_humidity = jnp.zeros((nlev, nx))

        out = term._column_group(_State(), {}, p_half, None, jnp.full((nlev, nx), 1e6))
        self.assertIn("aerocom_icnum", out)
        self.assertGreater(float(jnp.min(out["aerocom_icnum"])), 0.0)

    def test_burdens_include_cloud_borne_mass(self):
        """Both interstitial m_* and cloud-borne mc_* must be emitted."""
        nlev, nx = 4, 2
        p_half = jnp.linspace(1000.0, 101000.0, nlev + 1)[:, None] * jnp.ones((1, nx))
        term = AerocomDiagnostics(groups=("aerosol",))

        class _State:
            tracers = {"m_so4_acc": jnp.full((nlev, nx), 1e-9),
                       "mc_so4_acc": jnp.full((nlev, nx), 2e-9)}

        out = term._aerosol_group(_State(), {}, p_half)
        # One total per species, summing interstitial + cloud-borne.
        import jcm.constants as c
        want = (1e-9 + 2e-9) * (101000.0 - 1000.0) / c.grav
        np.testing.assert_allclose(
            np.asarray(out["aerocom_burden_so4"]), want, rtol=1e-6,
            err_msg="cloud-borne aerosol mass must be included or species "
                    "burdens undercount once aerosol is activated")


class CmorWriterTest(unittest.TestCase):
    """The post-processor must not mislabel units."""

    def _dataset(self, **extra):
        import xarray as xr
        base = {"geopotential": ("level", np.array([9806.65, 19613.3])),
                "surface_pressure": ((), np.float64(101325.0))}
        base.update(extra)
        return xr.Dataset({k: xr.DataArray(v[1], dims=v[0] if v[0] else ())
                           for k, v in base.items()})

    def test_geopotential_is_converted_to_height(self):
        """Height zg is metres; geopotential is m2/s2 and must be divided by g."""
        import tempfile
        from tools.aerocom_cmor import convert
        import xarray as xr
        ds = self._dataset()
        with tempfile.TemporaryDirectory() as td:
            import pathlib
            convert(ds, "JCM-t", "all_2000", "2010", "monthly", pathlib.Path(td))
            got = xr.open_dataset(pathlib.Path(td) /
                                  "aerocom_JCM-t_all_2000_zg_ModelLevel_2010_monthly.nc")
        # 9806.65 m2/s2 / 9.80665 = 1000 m
        np.testing.assert_allclose(np.asarray(got["zg"]), [1000.0, 2000.0], rtol=1e-4)

    def test_height_full_preferred_over_geopotential(self):
        """When both are present the direct height field wins."""
        import pathlib
        import tempfile
        import xarray as xr
        from tools.aerocom_cmor import convert
        ds = self._dataset(height_full=("level", np.array([1234.0, 5678.0])))
        with tempfile.TemporaryDirectory() as td:
            convert(ds, "JCM-t", "all_2000", "2010", "monthly", pathlib.Path(td))
            got = xr.open_dataset(pathlib.Path(td) /
                                  "aerocom_JCM-t_all_2000_zg_ModelLevel_2010_monthly.nc")
        np.testing.assert_allclose(np.asarray(got["zg"]), [1234.0, 5678.0], rtol=1e-6)

    def test_species_burden_is_renamed_for_submission(self):
        import xarray as xr
        from tools.aerocom_cmor import _collect_burdens
        ds = xr.Dataset({"aerocom_burden_so4": xr.DataArray(np.array([3.0, 3.0]))})
        got = _collect_burdens(ds)
        np.testing.assert_allclose(np.asarray(got["burden_so4"]), [3.0, 3.0])

    def test_organic_optics_species_are_summed_not_overwritten(self):
        """AeroCom reports ONE organic component, but jcm carries primary,
        secondary and marine organics separately — so the three must add
        into od550oa. Overwriting instead of summing would silently report
        only whichever species happened to be visited last (jax-gcm#584).
        """
        import xarray as xr
        from tools.aerocom_cmor import _collect_optics
        ds = xr.Dataset({
            "od550_poa": xr.DataArray(np.array([1.0])),
            "od550_soa": xr.DataArray(np.array([2.0])),
            "od550_moa": xr.DataArray(np.array([4.0])),
            "od550_du": xr.DataArray(np.array([0.5])),
            "od550_wat": xr.DataArray(np.array([0.25])),
            "abs550_bc": xr.DataArray(np.array([0.1])),
            "od550_mode_acc": xr.DataArray(np.array([0.7])),
        })
        got = _collect_optics(ds)
        np.testing.assert_allclose(np.asarray(got["od550oa"]), [7.0])
        np.testing.assert_allclose(np.asarray(got["od550dust"]), [0.5])
        np.testing.assert_allclose(np.asarray(got["od550aerh2o"]), [0.25])
        np.testing.assert_allclose(np.asarray(got["abs550bc"]), [0.1])
        # Per-mode fields are a JAM extra, passed through under their own name.
        np.testing.assert_allclose(np.asarray(got["od550_mode_acc"]), [0.7])

    def test_spectral_optics_reach_submission_files(self):
        """End-to-end: the #584 fields must come out of ``convert`` as their
        own CMOR files with a real CF standard_name, not be dropped for want
        of a NAME_MAP entry.
        """
        import pathlib
        import tempfile
        import xarray as xr
        from tools.aerocom_cmor import convert
        ds = self._dataset(
            ang4487aer=((), np.float64(1.4)),
            od550aer=((), np.float64(0.21)),
            abs550aer=((), np.float64(0.01)),
            od550_bc=((), np.float64(0.004)),
        )
        with tempfile.TemporaryDirectory() as td:
            convert(ds, "JCM-t", "all_2000", "2010", "monthly", pathlib.Path(td))
            root = pathlib.Path(td)
            got = xr.open_dataset(
                root / "aerocom_JCM-t_all_2000_od550aer_Column_2010_monthly.nc")
            self.assertAlmostEqual(float(got["od550aer"]), 0.21, places=6)
            self.assertEqual(
                got["od550aer"].attrs["standard_name"],
                "atmosphere_optical_thickness_due_to_ambient_aerosol_particles")
            # The per-species field is grouped to its AeroCom component name.
            bc = xr.open_dataset(
                root / "aerocom_JCM-t_all_2000_od550bc_Column_2010_monthly.nc")
            self.assertAlmostEqual(float(bc["od550bc"]), 0.004, places=6)


class AerosolGroupEndToEndTest(unittest.TestCase):
    """The aerosol group must survive a real JAM step.

    Regression: the group emitted its N70/N100/PM keys only when
    ``_jam_state`` was present, but that key is absent from the initial
    carry probe — so the scan rejected the changing carry pytree. The
    earlier end-to-end test used only cloud/column/plev and missed it.
    """

    def test_aerosol_group_runs_with_jam(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        model = Model(
            coords=coords, terrain=TerrainData.aquaplanet(coords), time_step=900.0,
            physics=echam_physics(
                radiation_scheme="grey", cloud_scheme="2m", aerosol_module="jam",
                enable_aerocom=True, aerocom_groups=("aerosol",)),
        )
        ds = model.run(total_time=0.05, save_interval=0.05).to_xarray()
        for key in ("aerocom_N70", "aerocom_N100", "aerocom_PM1", "aerocom_PM10"):
            self.assertIn(key, ds.data_vars)
            self.assertTrue(np.isfinite(np.asarray(ds[key])).all(), key)
        for spec in ("so4", "bc", "ss", "du"):
            key = f"aerocom_burden_{spec}"
            self.assertIn(key, ds.data_vars)
            self.assertTrue(np.isfinite(np.asarray(ds[key])).all(), key)


class AerocomOpticsConfigTest(unittest.TestCase):
    def test_optics_diagnostics_require_the_jam_module(self):
        """MACv2-SP has no species tracers, so the per-species split cannot
        be produced. Fail at construction rather than write an output file
        that is silently missing the requested fields (jax-gcm#584).
        """
        from jcm.physics.echam.echam_terms import echam_physics
        with self.assertRaisesRegex(ValueError, "aerosol_module='jam'"):
            echam_physics(aerosol_module="macv2sp", aerocom_optics=True)
