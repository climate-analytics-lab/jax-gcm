"""Tests for the AeroCom omnibus additions (#583, #586, #581 residuals)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.diagnostics.aerocom import AerocomDiagnostics


class NearSurfaceGroupTest(unittest.TestCase):
    """The nearsurface group: 2 m/10 m, psl, precip split, wbase."""

    def _diag(self, nlev=6, nx=3, z0=0.1, orog=0.0, t_low=290.0,
              conv_precip=2e-5, tke=0.5):
        class _Clouds:
            cloud_fraction = jnp.zeros((nlev, nx)).at[3].set(0.6)
            precip_snow = jnp.full((nx,), 1e-5)
            precip_rain = jnp.full((nx,), 3e-5)

        class _Surface:
            surface_temperature = jnp.full((nx,), 288.0)
            roughness_length = jnp.full((nx,), z0)

        class _Conv:
            precip_conv = jnp.full((nx,), conv_precip)

        class _Vdiff:
            pass
        _Vdiff.tke = jnp.full((nx, nlev), tke)  # (ncol, nlev) layout

        class _Terrain:
            pass
        _Terrain.orog = jnp.full((nx,), orog)

        # Lowest level-centre pressure deliberately BELOW the surface
        # pressure (p0), as in any real hybrid grid — so a psl computed
        # from p_full[-1] (the Codex #604 P1) cannot pass the sea-level
        # identity test below.
        p_full = jnp.linspace(20000.0, 99600.0, nlev)[:, None] * jnp.ones((1, nx))
        z_full = jnp.linspace(15000.0, 50.0, nlev)[:, None] * jnp.ones((1, nx)) + orog
        z_half = jnp.linspace(16000.0, 0.0, nlev + 1)[:, None] * jnp.ones((1, nx)) + orog
        diagnostics = {
            "clouds": _Clouds(), "surface": _Surface(),
            "convection": _Conv(), "vertical_diffusion": _Vdiff(),
            "height_full": z_full, "height_half": z_half,
        }

        class _State:
            temperature = jnp.full((nlev, nx), 260.0).at[-1].set(t_low)
            specific_humidity = jnp.full((nlev, nx), 5e-3)
            u_wind = jnp.full((nlev, nx), 10.0)
            v_wind = jnp.full((nlev, nx), -5.0)
            # psl/dew2 read the true surface pressure (Codex on #604).
            normalized_surface_pressure = jnp.ones((nx,))
            tracers: dict = {}

        term = AerocomDiagnostics(groups=("nearsurface",))
        state = _State()
        out = term._nearsurface_group(
            state, diagnostics, _Terrain(), state.temperature, p_full)
        return out, state, diagnostics

    def test_neutral_log_law_wind_ratio_is_exact(self):
        out, state, diag = self._diag(z0=0.1)
        z_agl = float(diag["height_full"][-1, 0] - diag["height_half"][-1, 0])
        want = 10.0 * np.log(10.0 / 0.1) / np.log(z_agl / 0.1)
        np.testing.assert_allclose(np.asarray(out["aerocom_uas"]), want,
                                   rtol=1e-6)

    def test_tas_lies_between_skin_and_lowest_level(self):
        out, state, _ = self._diag(t_low=290.0)
        tas = np.asarray(out["aerocom_tas"])
        self.assertTrue(((tas >= 288.0) & (tas <= 290.0)).all())

    def test_dew_point_never_exceeds_tas(self):
        out, _, _ = self._diag()
        self.assertTrue((np.asarray(out["aerocom_dew2"])
                         <= np.asarray(out["aerocom_tas"]) + 1e-6).all())

    def test_psl_equals_ps_at_sea_level(self):
        import jcm.constants as c
        out, _, _ = self._diag(orog=0.0)
        np.testing.assert_allclose(np.asarray(out["aerocom_psl"]), c.p0,
                                   rtol=1e-6)

    def test_psl_exceeds_ps_over_orography(self):
        import jcm.constants as c
        out, _, _ = self._diag(orog=1500.0)
        self.assertTrue((np.asarray(out["aerocom_psl"]) > c.p0).all())

    def test_warm_surface_conv_precip_is_rain(self):
        out, _, _ = self._diag(t_low=290.0, conv_precip=2e-5)
        np.testing.assert_allclose(np.asarray(out["aerocom_prcr"]), 2e-5)
        np.testing.assert_allclose(np.asarray(out["aerocom_prcs"]), 0.0)
        np.testing.assert_allclose(np.asarray(out["aerocom_prsn"]), 1e-5)

    def test_cold_surface_conv_precip_is_snow(self):
        out, _, _ = self._diag(t_low=260.0, conv_precip=2e-5)
        np.testing.assert_allclose(np.asarray(out["aerocom_prcs"]), 2e-5)
        np.testing.assert_allclose(np.asarray(out["aerocom_prsn"]), 3e-5)

    def test_wbase_is_the_activation_updraft_at_cloud_base(self):
        out, _, _ = self._diag(tke=0.5)
        want = 0.7 * np.sqrt(2.0 * 0.5)
        np.testing.assert_allclose(np.asarray(out["aerocom_wbase"]), want,
                                   rtol=1e-6)

    def test_no_cloud_gives_zero_wbase(self):
        out, state, diag = self._diag()
        diag["clouds"].cloud_fraction = jnp.zeros_like(
            diag["clouds"].cloud_fraction)
        term = AerocomDiagnostics(groups=("nearsurface",))
        p_full = jnp.linspace(20000.0, 100000.0, 6)[:, None] * jnp.ones((1, 3))
        out2 = term._nearsurface_group(
            state, diag, None, state.temperature, p_full)
        np.testing.assert_allclose(np.asarray(out2["aerocom_wbase"]), 0.0)


class PlevOmegaTest(unittest.TestCase):
    """wap/w500/w700 from the dycore omega provider (jax-gcm#409)."""

    nlev, nx = 12, 6

    def _plev(self, omega=None):
        term = AerocomDiagnostics(groups=("plev",))
        nlev, nx = self.nlev, self.nx
        p_full = (jnp.linspace(20000.0, 99600.0, nlev)[:, None]
                  * jnp.ones((1, nx)))

        class _State:
            u_wind = jnp.zeros((nlev, nx))
            v_wind = jnp.zeros((nlev, nx))
            temperature = jnp.full((nlev, nx), 280.0)

        diagnostics = {}
        if omega is not None:
            diagnostics["_dycore_fields"] = {"omega": omega}
        return term._plev_group(_State(), diagnostics,
                                _State.temperature, p_full), p_full

    def test_absent_provider_zero_fills_statically(self):
        out, p_full = self._plev(omega=None)
        self.assertEqual(out["aerocom_wap"].shape, (self.nlev, self.nx))
        self.assertEqual(float(jnp.max(jnp.abs(out["aerocom_wap"]))), 0.0)
        for key in ("aerocom_w500", "aerocom_w700"):
            self.assertEqual(out[key].shape, (self.nx,))
            self.assertEqual(float(jnp.max(jnp.abs(out[key]))), 0.0)

    def test_wap_passthrough_and_slices_exact_in_log_p(self):
        # An omega linear in log(p) makes the log-linear interpolation
        # exact, so w500/w700 have closed-form expectations.
        _, p_full = self._plev(omega=None)
        alpha = 0.037
        omega = alpha * jnp.log(p_full)
        out, _ = self._plev(omega=omega)
        np.testing.assert_allclose(np.asarray(out["aerocom_wap"]),
                                   np.asarray(omega))
        np.testing.assert_allclose(
            np.asarray(out["aerocom_w500"]),
            alpha * np.log(50000.0) * np.ones(self.nx), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(out["aerocom_w700"]),
            alpha * np.log(70000.0) * np.ones(self.nx), rtol=1e-6)

    def test_provider_keys_are_declared(self):
        for key in ("aerocom_wap", "aerocom_w500", "aerocom_w700"):
            self.assertIn(key, AerocomDiagnostics.provides)

    def test_runner_defaults_omega_on_for_aerocom_plev(self):
        from omegaconf import OmegaConf

        from jcm.runners import _want_omega
        plev_cfg = {"physics": {"enable_aerocom": True,
                                "aerocom_groups": ["cloud", "plev"]}}
        self.assertTrue(_want_omega(OmegaConf.create(plev_cfg)))
        self.assertFalse(_want_omega(OmegaConf.create(
            {"physics": {"enable_aerocom": True,
                         "aerocom_groups": ["cloud"]}})))
        self.assertFalse(_want_omega(OmegaConf.create({})))
        # An explicit dycore.compute_omega always wins, either way.
        self.assertFalse(_want_omega(OmegaConf.create(
            {**plev_cfg, "dycore": {"compute_omega": False}})))
        self.assertTrue(_want_omega(OmegaConf.create(
            {"dycore": {"compute_omega": True}})))
        # The shipped yaml null means "decide from the physics config".
        self.assertTrue(_want_omega(OmegaConf.create(
            {**plev_cfg, "dycore": {"compute_omega": None}})))

    def test_end_to_end_echam_run_with_the_provider(self):
        """Probe-vs-step structure and real values through a scanned run.

        The get_empty_data probe runs without dycore-field injection (the
        zero-fill branch) while the scanned steps see the injected omega;
        the run only compiles if both produce the same carry structure.
        """
        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        dycore = DinosaurDycore(
            coords=coords, terrain=TerrainData.aquaplanet(coords),
            dt_seconds=900.0, compute_omega=True)
        model = Model(
            dycore=dycore, time_step=15.0,
            physics=echam_physics(radiation_scheme="grey",
                                  enable_aerocom=True,
                                  aerocom_groups=("plev",)),
        )
        ds = model.run(total_time=0.05, save_interval=0.05).to_xarray()
        for key in ("aerocom_wap", "aerocom_w500", "aerocom_w700"):
            arr = np.asarray(ds[key])
            self.assertTrue(np.isfinite(arr).all(), f"{key} not finite")
        # The balanced-start ECHAM state has real vertical motion within a
        # few steps; all-zero wap would mean the provider never reached
        # the term.
        self.assertGreater(float(np.abs(ds["aerocom_wap"]).max()), 0.0)
        self.assertLess(float(np.abs(ds["aerocom_wap"]).max()), 1e2)


class CodexRound2RegressionTest(unittest.TestCase):
    """Regressions for the Codex findings on PR #604."""

    def test_tropopause_window_derived_from_level_pressures(self):
        """An L95-like grid must search near 40-550 hPa, not indices 13-35."""
        term = AerocomDiagnostics(groups=("plev",))

        class _Vert:
            # 95 sigma centres crowding the stratosphere like the L95 grid.
            centers = np.concatenate([
                np.geomspace(3e-6, 0.05, 60), np.linspace(0.06, 0.995, 35)])

        class _Coords:
            vertical = _Vert()

        term.cache_coords(_Coords())
        ref = term._nominal_level_pressures(95)
        self.assertIsNotNone(ref)
        ncctop = int(np.searchsorted(ref, 4000.0))
        nccbot = int(np.searchsorted(ref, 55000.0))
        # The window must cover the real tropopause pressures, far from
        # the L47-tuned defaults (13..35 sit at 31-501 Pa on this grid).
        self.assertGreater(ncctop, 35)
        self.assertGreater(nccbot, ncctop + 2)
        self.assertLess(ref[ncctop], 5000.0)
        self.assertGreater(ref[nccbot - 1], 10000.0)

    def test_save_predictions_writes_the_snapshot_file(self):
        import pathlib
        import tempfile

        import xarray as xr
        from jcm.runners import save_predictions

        class _Preds:
            def to_xarray(self):
                return xr.Dataset({"t": xr.DataArray(np.zeros(3),
                                                     dims=("x",))})

            def snapshot_dataset(self):
                return xr.Dataset({"clt": xr.DataArray(
                    np.zeros((2, 3)), dims=("snap_time", "x"))})

        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "run.nc"
            save_predictions(_Preds(), out)
            self.assertTrue(out.exists())
            self.assertTrue((pathlib.Path(td) / "run_snapshots.nc").exists())

    def test_run_config_declares_snapshot_keys(self):
        import yaml
        cfg = yaml.safe_load(open("jcm/config/run/default.yaml"))
        self.assertIn("snapshot_interval", cfg)
        self.assertIn("snapshot_variables", cfg)


class FluxFamilyTest(unittest.TestCase):
    """Deposition fluxes and the emission/deposition/burden budget."""

    def test_deposition_flux_sign_and_species(self):
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            accumulate_deposition_fluxes)
        nlev, nx = 4, 2
        rho = jnp.full((nlev, nx), 1.0)
        dz = jnp.full((nlev, nx), 100.0)
        tends = {"m_bc_a4": jnp.full((nlev, nx), -1e-12),
                 "n_a4": jnp.full((nlev, nx), -1.0)}
        out = accumulate_deposition_fluxes({}, tends, rho, dz, kind="dry")
        # Removal (negative tendency) reports a POSITIVE downward flux.
        np.testing.assert_allclose(np.asarray(out["dry_bc"]),
                                   nlev * 100.0 * 1e-12, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(out["dry_so4"]), 0.0)

    def test_reset_covers_the_whole_flux_family(self):
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            ResetEmissionFluxes, all_flux_keys)
        self.assertIn("emi_bb_bc", all_flux_keys())
        self.assertIn("dry_du", all_flux_keys())
        self.assertIn("wet_so4", all_flux_keys())
        self.assertEqual(set(ResetEmissionFluxes.provides), set(all_flux_keys()))

    def test_inert_species_budget_closes_over_one_jam_step(self):
        """Emission - dry - wet ~= d(burden)/dt for BC over one step.

        BC has no chemistry source/sink, so over a single physics step the
        column budget must close to roundoff: the tracers' total tendency
        IS emission + transportless-removal, and the flux diagnostics
        integrate exactly those tendencies.
        """
        import jcm.physics.aerosol.jam.emissions.flux_diagnostic as fd
        nlev, nx = 5, 2
        rho = jnp.full((nlev, nx), 1.0)
        dz = jnp.full((nlev, nx), 200.0)
        emis = {"m_bc_a4": jnp.zeros((nlev, nx)).at[-1].set(2e-12)}
        dep = {"m_bc_a4": jnp.full((nlev, nx), -3e-13)}
        wet = {"m_bc_a4": jnp.full((nlev, nx), -1e-13)}
        d = fd.accumulate_emission_fluxes({}, emis, rho, dz)
        d = fd.accumulate_deposition_fluxes(d, dep, rho, dz, kind="dry")
        d = fd.accumulate_deposition_fluxes(d, wet, rho, dz, kind="wet")
        total_tend = jax.tree_util.tree_map(
            lambda *xs: sum(xs), emis, dep, wet)
        dburden_dt = jnp.sum(total_tend["m_bc_a4"] * rho * dz, axis=0)
        closure = (np.asarray(d["emi_bc"]) - np.asarray(d["dry_bc"])
                   - np.asarray(d["wet_bc"]))
        # Both sides are exactly balanced here (closure = 0), so the
        # comparison needs an absolute floor: rtol alone cannot compare
        # a float32-roundoff residual against an exact zero.
        np.testing.assert_allclose(closure, np.asarray(dburden_dt),
                                   rtol=1e-6, atol=1e-16)


class SnapshotStreamTest(unittest.TestCase):
    """The 3-hourly instantaneous output stream (#586)."""

    def test_snapshots_match_a_snapshot_mode_run(self):
        """Strided snapshots equal the instantaneous states a fine-grained
        snapshot-mode run saves at the same times (SPEEDY T31, 4 steps).
        """
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        dt_min = 30.0
        dt_days = dt_min / 1440.0
        model = Model(coords=get_speedy_coords(), time_step=dt_min)
        # Averaged run: one save interval of 4 steps, snapshots every 2.
        preds = model.run(save_interval=4 * dt_days, total_time=4 * dt_days,
                          output_averages=True,
                          snapshot_interval=2 * dt_days,
                          snapshot_variables=("_shortwave_rad.cloudc",))
        snaps = preds.snapshots
        self.assertIsNotNone(snaps)
        arr = np.asarray(snaps["_shortwave_rad.cloudc"])
        self.assertEqual(arr.shape[0], 2)  # 4 steps / stride 2

        # Reference: fine-grained snapshot-mode run saving every 2 steps —
        # its saved frames ARE the instantaneous states at the same times.
        model2 = Model(coords=get_speedy_coords(), time_step=dt_min)
        ref = model2.run(save_interval=2 * dt_days, total_time=4 * dt_days,
                         output_averages=False)
        ref_cc = np.asarray(jax.device_get(ref.physics["_shortwave_rad"].cloudc))
        np.testing.assert_allclose(arr, ref_cc.reshape(arr.shape), rtol=1e-6)

    def test_snapshot_dataset_axes(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        dt_days = 30.0 / 1440.0
        model = Model(coords=get_speedy_coords(), time_step=30.0)
        preds = model.run(save_interval=4 * dt_days, total_time=4 * dt_days,
                          output_averages=True,
                          snapshot_interval=2 * dt_days,
                          snapshot_variables=("_shortwave_rad.cloudc",))
        ds = preds.snapshot_dataset()
        self.assertIsNotNone(ds)
        self.assertIn("snap_time", ds.dims)
        self.assertEqual(ds["_shortwave_rad_cloudc"].dims,
                         ("snap_time", "lon", "lat"))

    def test_snapshots_require_averaged_mode(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        dt_days = 30.0 / 1440.0
        model = Model(coords=get_speedy_coords(), time_step=30.0)
        with self.assertRaises(ValueError):
            model.run(save_interval=4 * dt_days, total_time=4 * dt_days,
                      output_averages=False,
                      snapshot_interval=2 * dt_days,
                      snapshot_variables=("_shortwave_rad.cloudc",))


class CmorOmnibusTest(unittest.TestCase):
    def test_new_fields_reach_submission_files(self):
        import pathlib
        import tempfile

        import xarray as xr
        from tools.aerocom_cmor import convert

        ds = xr.Dataset({
            "radiation.toa_sw_up_noa": xr.DataArray(np.full((3, 4), 90.0),
                                                    dims=("lat", "lon")),
            "aerocom_tas": xr.DataArray(np.full((3, 4), 288.0),
                                        dims=("lat", "lon")),
            "dry_bc": xr.DataArray(np.full((3, 4), 1e-12),
                                   dims=("lat", "lon")),
            "emi_bb_oc": xr.DataArray(np.full((3, 4), 2e-12),
                                      dims=("lat", "lon")),
            "aerocom_wap": xr.DataArray(np.full((5, 3, 4), -0.02),
                                        dims=("level", "lat", "lon")),
            "aerocom_w500": xr.DataArray(np.full((3, 4), -0.05),
                                         dims=("lat", "lon")),
        })
        with tempfile.TemporaryDirectory() as td:
            written, skipped = convert(
                ds, "JCM-t", "all_2000", "2010", "monthly",
                pathlib.Path(td), na_aliases=True)
        names = "\n".join(written)
        for var in ("rsutnoa", "rsut_na", "tas", "drybc", "emi_bb_oc",
                    "wap", "w500"):
            self.assertIn(f"_{var}_", names)
        # wap goes out on model levels, the slices as 2-D column fields.
        self.assertIn("_wap_ModelLevel_", names)
        self.assertIn("_w500_Column_", names)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()


class AerosolFreeRadiationTest(unittest.TestCase):
    """The #583 noa fluxes: a second solve with aerosol optics zeroed."""

    def test_grey_scheme_is_rejected(self):
        from jcm.physics.echam.echam_terms import echam_physics
        with self.assertRaises(ValueError):
            echam_physics(radiation_scheme="grey",
                          aerosol_free="exact")

    def test_flag_off_leaves_noa_fields_zero(self):
        from jcm.physics.radiation.radiation_types import RadiationData
        r = RadiationData.zeros((8,), 5)
        np.testing.assert_allclose(np.asarray(r.toa_sw_up_noa), 0.0)


import pytest  # noqa: E402


@pytest.mark.slow
class AerosolFreeRadiationSlowTest(unittest.TestCase):
    """One real RRTMGP step: noa differs from all-sky iff aerosol is present.

    Runs the full ECHAM composition (RRTMGP + 2M clouds + MACv2-SP with
    ``aerosol_free="exact"``) in the single-column host rather than
    a T21 ``Model``: the claim under test — the second, aerosol-free
    RRTMGP solve and its published ``*_noa`` fluxes — is per-column
    physics, and running 2048 spectral columns through the most expensive
    double-compile in the suite bought only wall clock (#627, ~7 min of
    every PR gate). The column sits in the East-Asian MACv2-SP plume
    (30°N, 117°E) and is sunlit at the default start time, so the
    anthropogenic AOD is nonzero and the SW must respond.
    """

    def test_noa_fluxes_differ_with_aerosol_and_match_without(self):
        import jax.numpy as jnp

        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.rce import rce_initial_state
        from jcm.single_column_model import SingleColumnModel

        physics = echam_physics(
            radiation_scheme="rrtmgp", cloud_scheme="2m",
            aerosol_module="macv2sp", aerosol_free="exact",
            checkpoint_terms=False)
        vertical = get_echam_levels(47)
        scm = SingleColumnModel(
            physics=physics, vertical=vertical,
            lat_deg=30.0, lon_deg=117.0, dt_seconds=720.0)
        column = rce_initial_state(vertical, sst=300.0)
        # Seed every tracer the composition declares (the 2M scheme carries
        # number concentrations beyond rce_initial_state's qc/qi).
        column = column.copy(tracers={
            spec.name: jnp.zeros(47) for spec in physics.required_tracers()})
        preds = scm.run([column])
        rad = preds.physics_data["radiation"]
        # Guard the geometry assumption explicitly: a dark column would make
        # the SW assertion below fail for the wrong reason.
        self.assertGreater(float(np.asarray(rad.cos_zenith).max()), 0.0)
        sw_noa = np.asarray(jax.device_get(rad.toa_sw_up_noa))
        sw_all = np.asarray(jax.device_get(rad.toa_sw_up))
        self.assertTrue(np.isfinite(sw_noa).all())
        # MACv2-SP puts nonzero anthropogenic AOD in 2005 defaults, so the
        # aerosol-free SW must differ measurably in this sunlit plume column...
        self.assertGreater(np.abs(sw_noa - sw_all).max(), 1e-3)
        # ...while the LW (MACv2-SP models SW effects only) stays equal.
        lw_noa = np.asarray(jax.device_get(rad.toa_lw_up_noa))
        lw_all = np.asarray(jax.device_get(rad.toa_lw_up))
        np.testing.assert_allclose(lw_noa, lw_all, rtol=1e-5)
