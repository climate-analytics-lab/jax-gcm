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

        p_full = jnp.linspace(20000.0, 100000.0, nlev)[:, None] * jnp.ones((1, nx))
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
        out, _, _ = self._diag(orog=0.0)
        np.testing.assert_allclose(np.asarray(out["aerocom_psl"]), 100000.0,
                                   rtol=1e-6)

    def test_psl_exceeds_ps_over_orography(self):
        out, _, _ = self._diag(orog=1500.0)
        self.assertTrue((np.asarray(out["aerocom_psl"]) > 100000.0).all())

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
        })
        with tempfile.TemporaryDirectory() as td:
            written, skipped = convert(
                ds, "JCM-t", "all_2000", "2010", "monthly",
                pathlib.Path(td), na_aliases=True)
        names = "\n".join(written)
        for var in ("rsutnoa", "rsut_na", "tas", "drybc", "emi_bb_oc"):
            self.assertIn(f"_{var}_", names)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()


class AerosolFreeRadiationTest(unittest.TestCase):
    """The #583 noa fluxes: a second solve with aerosol optics zeroed."""

    def test_grey_scheme_is_rejected(self):
        from jcm.physics.echam.echam_terms import echam_physics
        with self.assertRaises(ValueError):
            echam_physics(radiation_scheme="grey",
                          aerosol_free_radiation=True)

    def test_flag_off_leaves_noa_fields_zero(self):
        from jcm.physics.radiation.radiation_types import RadiationData
        r = RadiationData.zeros((8,), 5)
        np.testing.assert_allclose(np.asarray(r.toa_sw_up_noa), 0.0)


import pytest  # noqa: E402


@pytest.mark.slow
class AerosolFreeRadiationSlowTest(unittest.TestCase):
    """One real RRTMGP step: noa differs from all-sky iff aerosol is present."""

    def test_noa_fluxes_differ_with_aerosol_and_match_without(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.runners import inject_jw_profile
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        physics = echam_physics(
            radiation_scheme="rrtmgp", cloud_scheme="2m",
            aerosol_module="macv2sp", aerosol_free_radiation=True,
            checkpoint_terms=False)
        model = Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                      physics=physics, time_step=12.0)
        inject_jw_profile(model, rh=0.6)
        dt_days = 12.0 / 1440.0
        preds = model.resume(save_interval=dt_days, total_time=dt_days)
        rad = preds.physics["radiation"]
        sw_noa = np.asarray(jax.device_get(rad.toa_sw_up_noa))
        sw_all = np.asarray(jax.device_get(rad.toa_sw_up))
        self.assertTrue(np.isfinite(sw_noa).all())
        # MACv2-SP puts nonzero anthropogenic AOD in 2005 defaults, so the
        # aerosol-free SW must differ measurably somewhere sunlit...
        self.assertGreater(np.abs(sw_noa - sw_all).max(), 1e-3)
        # ...while the LW (MACv2-SP models SW effects only) stays equal.
        lw_noa = np.asarray(jax.device_get(rad.toa_lw_up_noa))
        lw_all = np.asarray(jax.device_get(rad.toa_lw_up))
        np.testing.assert_allclose(lw_noa, lw_all, rtol=1e-5)
