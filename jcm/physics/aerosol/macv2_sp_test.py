"""Tests for the MACv2-SP (Simple Plumes) aerosol scheme.

Anchored to the v1 reference (mo_simple_plumes_v1.f90 + MACv2.0-SP_v1.nc,
Stevens et al. 2017 GMD supplement): real plume parameters, dz-weighted
orography-truncated vertical profiles, per-feature time weighting inside
the spatial sum, anthropogenic-AOD-weighted per-band optics with the
ssa→1 / asy→0 zero-AOD limits, and a column-scalar natural background
that feeds only dNovrN/Nccn.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jcm.forcing import ForcingData
from jcm.physics.aerosol.aerosol_types import AerosolData
from jcm.physics.aerosol.macv2_sp import (
    _per_feature_plume_gaussians,
    get_CDNC,
    get_dNovrN,
    get_plume_column_weights,
    get_plume_spatial_distribution,
    get_simple_aerosol,
    get_vertical_profiles,
)
from jcm.physics.aerosol.macv2_sp_params import AerosolParameters

NPLUMES, NFEATURES = 9, 2


def _column_grid(nlev=30, ncols=4, oro=None):
    """Uniform-dz test columns, level 0 = model top (jcm convention)."""
    z = jnp.linspace(250.0, 14750.0, nlev)[::-1]
    height = jnp.broadcast_to(z[:, None], (nlev, ncols))
    dz = jnp.full((nlev, ncols), 500.0)
    if oro is None:
        oro = jnp.zeros(ncols)
    return height, dz, oro


def _forcing_ones():
    class _F:
        aerosol_year_weight = jnp.ones(NPLUMES)
        aerosol_ann_cycle = jnp.ones((NFEATURES, NPLUMES))
    return _F()


def _run_scheme(lats, lons, oro=None, nlev=30,
                bands=jnp.asarray([550.0]), forcing=None):
    ncols = lats.shape[0]
    height, dz, oro = _column_grid(nlev, ncols, oro)
    data = AerosolData.zeros((ncols,), nlev, n_bnd_sw=bands.shape[0],
                             n_bnd_lw=1)
    return get_simple_aerosol(
        height, dz, oro, lats, lons, data,
        AerosolParameters.default(), forcing or _forcing_ones(), bands,
    )


class TestAerosolParameters:
    """The defaults are the real MACv2.0-SP_v1.nc values."""

    def test_reference_values(self):
        """Spot-check hard values transcribed from the v1 parameter file."""
        p = AerosolParameters.default()
        assert p.plume_lat.shape[0] == 9 and p.ftr_weight.shape[0] == 2
        # Plume 1 is Europe (the 260-degree wrap case), plume 3 East Asia.
        np.testing.assert_allclose(p.plume_lat[0], 49.4)
        np.testing.assert_allclose(p.plume_lon[0], 20.6)
        np.testing.assert_allclose(p.plume_lon[2], 114.0)
        np.testing.assert_allclose(p.aod_spmx[2], 0.636)
        # Biomass plumes 5-8 share the darker ssa550.
        np.testing.assert_allclose(p.ssa550[4:8], 0.87)
        np.testing.assert_allclose(p.asy550, 0.63)
        np.testing.assert_allclose(p.angstrom, 2.0)
        # ftr_weight sums to 1 per plume in the file.
        np.testing.assert_allclose(jnp.sum(p.ftr_weight, axis=0), 1.0,
                                   rtol=1e-5)

    def test_parameter_ranges(self):
        """Physical ranges; longitudes use the file's 0-360 convention."""
        p = AerosolParameters.default()
        assert jnp.all(p.plume_lon >= 0.0) and jnp.all(p.plume_lon < 360.0)
        assert jnp.all(p.plume_lat >= -90.0) and jnp.all(p.plume_lat <= 90.0)
        assert jnp.all(p.ssa550 > 0.0) and jnp.all(p.ssa550 <= 1.0)
        assert jnp.all(p.asy550 > 0.0) and jnp.all(p.asy550 < 1.0)
        assert jnp.all(p.beta_a > 0.0) and jnp.all(p.beta_b > 0.0)
        assert jnp.all(p.aod_spmx > 0.0) and jnp.all(p.aod_fmbg > 0.0)


class TestVerticalProfiles:
    """dz-weighted, orography-truncated beta profiles (findings 2.29/2.33c)."""

    def test_shape_and_sea_level_normalization(self):
        height, dz, oro = _column_grid()
        prof = get_vertical_profiles(height, dz, oro,
                                     AerosolParameters.default())
        assert prof.shape == (NPLUMES, 30, 4)
        # Sea-level columns: mask is a no-op, level sums are exactly 1.
        np.testing.assert_allclose(jnp.sum(prof, axis=1), 1.0, rtol=1e-5)

    def test_aod_split_invariant_to_level_refinement(self):
        """The below-3km AOD share must not depend on the vertical grid.

        The reference weights the beta density by dz before normalizing;
        the pre-fix port normalized the density alone, so refining or
        stretching the grid moved column AOD between layers (54% vs 28%
        below 3 km in the review's probe). Compare the below-3km mass on
        a uniform grid against a 2x-stretched grid.
        """
        p = AerosolParameters.default()

        def below_3km_share(z_edges):
            z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])[::-1]      # top-down
            dz = jnp.diff(z_edges)[::-1]
            prof = get_vertical_profiles(
                z_mid[:, None], dz[:, None], jnp.zeros(1), p,
            )
            return jnp.sum(jnp.where(z_mid[None, :, None] < 3000.0, prof, 0.0),
                           axis=1)[:, 0]

        uniform = below_3km_share(jnp.linspace(0.0, 15000.0, 31))
        # Geometrically stretched edges (fine near the surface).
        r = jnp.cumsum(1.12 ** jnp.arange(30))
        stretched = below_3km_share(15000.0 * jnp.concatenate([jnp.zeros(1), r]) / r[-1])
        np.testing.assert_allclose(uniform, stretched, atol=0.02)

    def test_orography_truncation(self):
        """Levels below the surface carry zero AOD; the mass is removed.

        Fortran applies the z >= oro mask AFTER normalization (line 300)
        — over elevated terrain the column sum is < 1, not renormalized.
        """
        height, dz, _ = _column_grid(ncols=2)
        oro = jnp.array([0.0, 4000.0])
        prof = get_vertical_profiles(height, dz, oro,
                                     AerosolParameters.default())
        below = height < oro[None, :]
        assert jnp.all(jnp.where(below[None], prof, 0.0) == 0.0)
        sums = jnp.sum(prof, axis=1)
        np.testing.assert_allclose(sums[:, 0], 1.0, rtol=1e-5)
        assert jnp.all(sums[:, 1] < 1.0)
        # Above-ground levels keep their sea-level values (mask does not
        # redistribute mass): compare both columns under the mountain
        # column's own mask.
        above_mtn = ~below[:, 1]
        np.testing.assert_allclose(
            jnp.where(above_mtn[None, :], prof[:, :, 1], 0.0),
            jnp.where(above_mtn[None, :], prof[:, :, 0], 0.0), rtol=1e-5,
        )


class TestSpatialDistribution:
    """Rotated anisotropic Gaussians with the Europe wrap case."""

    def test_plume_centers_maximum(self):
        p = AerosolParameters.default()
        gauss = _per_feature_plume_gaussians(p.plume_lat, p.plume_lon, p)
        # At its own center every feature Gaussian is exactly 1.
        for ip in range(NPLUMES):
            np.testing.assert_allclose(gauss[:, ip, ip], 1.0, rtol=1e-6)

    def test_europe_wrap_continuity_across_greenwich(self):
        """The 260-degree wrap keeps Europe's tail continuous across 0 E.

        With the real parameters (Europe at 20.6 E, feature-2 westward
        sigma 35 degrees) the trans-Atlantic tail crosses the 0/360 seam;
        the plume-1 wrap threshold prevents a jump there.
        """
        p = AerosolParameters.default()
        lats = jnp.full(4, 49.4)
        # Two 0.2-degree steps: one across the 0/360 seam, one just west
        # of it. Without the plume-1 wrap the across-seam step jumps
        # (delta_lon flips from -20.5 to +339.3); with it, both steps
        # show only the smooth 0.2-degree Gaussian gradient.
        lons = jnp.array([359.9, 0.1, 359.5, 359.7])
        gauss = _per_feature_plume_gaussians(lats, lons, p)
        seam_step = jnp.abs(gauss[:, 0, 1] - gauss[:, 0, 0])
        off_step = jnp.abs(gauss[:, 0, 3] - gauss[:, 0, 2])
        assert jnp.all(seam_step < 3.0 * off_step + 1e-6)
        assert float(gauss[1, 0, 0]) > 0.3   # tail is genuinely alive there

    def test_collapsed_distribution_bounded(self):
        p = AerosolParameters.default()
        lats = jnp.linspace(-90, 90, 50)
        lons = jnp.linspace(0, 360, 50)
        dist = get_plume_spatial_distribution(lats, lons, p)
        assert dist.shape == (NPLUMES, 50)
        assert jnp.all(dist >= 0.0) and jnp.all(dist <= 1.0)


class TestColumnWeights:
    """Per-feature time weighting inside the spatial sum (finding 2.31)."""

    def test_feature_cycles_do_not_commute(self):
        """Features with different ann_cycle and different Gaussians must
        be combined per feature — the collapsed [avg cycle]x[summed
        Gaussian] product is measurably wrong wherever the two feature
        Gaussians differ (the pre-fix behavior).
        """
        p = AerosolParameters.default()
        # A point where plume-6 feature Gaussians differ strongly.
        lats, lons = jnp.array([-15.0]), jnp.array([290.0])
        gauss = _per_feature_plume_gaussians(lats, lons, p)
        yw = jnp.ones(NPLUMES)
        # Feature-asymmetric cycle for plume 6 (index 5).
        ann = jnp.ones((NFEATURES, NPLUMES))
        ann = ann.at[0, 5].set(0.1).at[1, 5].set(1.0)
        cw_an, _ = get_plume_column_weights(p, yw, ann, gauss)
        # Reference: sum_f (yw*ann_f*fw_f*g_f) * aod_spmx
        fw = p.ftr_weight
        expected = p.aod_spmx[5] * (
            ann[0, 5] * fw[0, 5] * gauss[0, 5, 0]
            + ann[1, 5] * fw[1, 5] * gauss[1, 5, 0]
        )
        np.testing.assert_allclose(cw_an[5, 0], expected, rtol=1e-6)
        # The collapsed (pre-fix) combination differs at this point.
        collapsed = (
            p.aod_spmx[5]
            * float(jnp.sum(fw[:, 5] * ann[:, 5]) / jnp.sum(fw[:, 5]))
            * float(jnp.sum(fw[:, 5] * gauss[:, 5, 0]))
        )
        assert not np.isclose(float(cw_an[5, 0]), collapsed, rtol=1e-3)

    def test_background_omits_year_weight(self):
        """cw_bg uses the annual cycle only (Fortran time_weight_bg)."""
        p = AerosolParameters.default()
        lats, lons = jnp.array([49.4]), jnp.array([20.6])
        gauss = _per_feature_plume_gaussians(lats, lons, p)
        ann = jnp.ones((NFEATURES, NPLUMES))
        _, bg_at_0 = get_plume_column_weights(p, jnp.zeros(NPLUMES), ann, gauss)
        _, bg_at_1 = get_plume_column_weights(p, jnp.ones(NPLUMES), ann, gauss)
        np.testing.assert_allclose(bg_at_0, bg_at_1)
        an_at_0, _ = get_plume_column_weights(p, jnp.zeros(NPLUMES), ann, gauss)
        assert jnp.all(an_at_0 == 0.0)


class TestFullScheme:
    """End-to-end reference behavior of get_simple_aerosol."""

    def test_column_aod_at_plume_center(self):
        """Sea-level column AOD at a plume center ~ aod_spmx (+ tails).

        Pins the removal of the double-Gaussian defect: the pre-fix path
        multiplied the plume-summed column AOD by each plume's Gaussian
        again, so the value at the East-Asia center was far from
        aod_spmx.
        """
        out = _run_scheme(jnp.array([30.0]), jnp.array([114.0]))
        aod = float(out.aod_total[0])
        assert 0.6 < aod < 0.75, aod    # 0.636 + neighboring-plume tails

    def test_remote_ocean_is_clean_and_finite(self):
        """Mid-Pacific: near-zero AOD, no NaN (finding 2.33e), reference
        zero-AOD optics limits from the completing division.
        """
        out = _run_scheme(jnp.array([0.0]), jnp.array([180.0]))
        assert float(out.aod_total[0]) < 1e-3
        assert jnp.all(jnp.isfinite(out.ssa_profile))
        assert jnp.all(jnp.isfinite(out.asy_profile))
        assert float(out.cdnc_factor[0]) < 1.01
        # dNovrN -> 1 and background -> 0.02 far from every plume.
        np.testing.assert_allclose(out.aod_background[0], 0.02, atol=1e-3)

    def test_zero_aod_optics_limits(self):
        """Ssa -> 1 and asy -> 0 where plume AOD vanishes (Fortran 366-367)."""
        out = _run_scheme(jnp.array([0.0]), jnp.array([180.0]),
                          oro=jnp.array([8000.0]))
        # Levels below the 8 km surface have exactly zero AOD.
        assert float(jnp.min(out.aod_profile)) == 0.0
        zero = out.aod_profile == 0.0
        assert jnp.all(jnp.where(zero, out.ssa_profile, 1.0) == 1.0)
        assert jnp.all(jnp.where(zero, out.asy_profile, 0.0) == 0.0)

    def test_biomass_optics_are_darker(self):
        """Columns under a biomass plume inherit ssa550 = 0.87 by AOD
        weighting (not the spatial-Gaussian blend of finding 2.32).
        """
        out = _run_scheme(jnp.array([-3.5]), jnp.array([16.0]))
        k = int(jnp.argmax(out.aod_profile[:, 0]))
        np.testing.assert_allclose(out.ssa_profile[k, 0], 0.87, atol=0.005)

    def test_background_never_enters_radiative_aod(self):
        """aod_total is anthropogenic-only: zero year_weight -> zero AOD
        even though the fine-mode background AOD stays finite.
        """
        class _F:
            aerosol_year_weight = jnp.zeros(NPLUMES)
            aerosol_ann_cycle = jnp.ones((NFEATURES, NPLUMES))
        out = _run_scheme(jnp.array([49.4]), jnp.array([20.6]), forcing=_F())
        assert float(jnp.max(jnp.abs(out.aod_profile))) == 0.0
        assert float(out.aod_total[0]) == 0.0
        assert float(out.aod_background[0]) > 0.05   # plume-shaped fm bg
        np.testing.assert_allclose(out.cdnc_factor[0], 1.0, rtol=1e-6)

    def test_mountain_column_loses_aod(self):
        """Orography truncation removes below-ground plume AOD."""
        lats = jnp.array([30.0, 30.0])
        lons = jnp.array([114.0, 114.0])
        out = _run_scheme(lats, lons, oro=jnp.array([0.0, 4000.0]))
        assert float(out.aod_total[1]) < float(out.aod_total[0])

    def test_per_band_scaling(self):
        """Angstrom scaling: AOD falls with wavelength; 550 nm band
        reproduces the 550 nm diagnostic profile.
        """
        bands = jnp.asarray([442.0, 550.0, 1020.0])
        out = _run_scheme(jnp.array([30.0]), jnp.array([114.0]), bands=bands)
        col = jnp.sum(out.aod_sw_per_band[:, :, 0], axis=1)
        assert float(col[0]) > float(col[1]) > float(col[2])
        np.testing.assert_allclose(
            out.aod_sw_per_band[1], out.aod_profile, rtol=1e-5,
        )
        np.testing.assert_allclose(
            out.ssa_sw_per_band[1], out.ssa_profile, rtol=1e-5,
        )

    def test_dnovrn_magnitude(self):
        """DNovrN in the Stevens 2017 range: ~1 remote, 1.3-1.7 at the
        East-Asia 2005 maximum, largest where background is clean.
        """
        out = _run_scheme(jnp.array([30.0, 0.0]), jnp.array([114.0, 180.0]))
        assert 1.2 < float(out.cdnc_factor[0]) < 1.8
        assert abs(float(out.cdnc_factor[1]) - 1.0) < 0.01


class TestCDNC:
    """Twomey-factor and absolute-CCN helpers."""

    def test_dnovrn_formula(self):
        aod_sp, aod_bg = jnp.array([0.2]), jnp.array([0.05])
        expected = np.log(1000.0 * 0.25 + 1.0) / np.log(1000.0 * 0.05 + 1.0)
        np.testing.assert_allclose(get_dNovrN(aod_sp, aod_bg)[0], expected,
                                   rtol=1e-6)

    def test_dnovrn_no_anthropogenic(self):
        np.testing.assert_allclose(
            get_dNovrN(jnp.zeros(3), jnp.full(3, 0.02)), 1.0, rtol=1e-6,
        )

    def test_cdnc_monotone(self):
        aods = jnp.array([0.0, 0.1, 0.5, 1.0])
        cdnc = get_CDNC(aods)
        assert float(cdnc[0]) == 1.0
        assert jnp.all(jnp.diff(cdnc) > 0.0)


class TestJAXCompatibility:
    """jit / grad through the full scheme."""

    def test_jit_full_scheme(self):
        height, dz, oro = _column_grid(ncols=2)
        lats, lons = jnp.array([30.0, 0.0]), jnp.array([114.0, 180.0])
        data = AerosolData.zeros((2,), 30, n_bnd_sw=1, n_bnd_lw=1)
        p = AerosolParameters.default()
        f = ForcingData.ones((1, 2))

        @jax.jit
        def run(height, lats, lons):
            return get_simple_aerosol(
                height, dz, oro, lats, lons, data, p, f,
                jnp.asarray([550.0]),
            ).aod_total

        out = run(height, lats, lons)
        assert out.shape == (2,) and jnp.all(jnp.isfinite(out))

    def test_gradient_wrt_plume_amplitude(self):
        """d(column AOD)/d(aod_spmx) is finite and positive — the
        calibration path the parameter threading exists for.
        """
        height, dz, oro = _column_grid(ncols=1)
        lats, lons = jnp.array([30.0]), jnp.array([114.0])
        data = AerosolData.zeros((1,), 30, n_bnd_sw=1, n_bnd_lw=1)
        f = _forcing_ones()

        def total_aod(aod_spmx):
            p = AerosolParameters.default()
            p = type(p)(**{**{k: getattr(p, k) for k in (
                'plume_lat', 'plume_lon', 'beta_a',
                'beta_b', 'aod_fmbg', 'asy550', 'ssa550', 'angstrom',
                'sig_lon_E', 'sig_lon_W', 'sig_lat_E', 'sig_lat_W', 'theta',
                'ftr_weight', 'background_aod', 'spa_prefactor',
                'spa_exponent', 'spa_cap_smoothing')}, 'aod_spmx': aod_spmx})
            out = get_simple_aerosol(height, dz, oro, lats, lons, data, p,
                                     f, jnp.asarray([550.0]))
            return jnp.sum(out.aod_total)

        g = jax.grad(total_aod)(AerosolParameters.default().aod_spmx)
        assert g.shape == (NPLUMES,)
        assert jnp.all(jnp.isfinite(g))
        assert float(g[2]) > 0.9   # East-Asia column sits on plume 3

    def test_gradient_finite_through_zero_aod_columns(self):
        """Gradients survive columns/levels with zero anthropogenic AOD.

        Codex review on #555: a bare where() around the 550 nm profile
        divisions still differentiates the inactive 0/0 branch, NaN-ing
        any gradient that touches ssa/asy_profile (grey radiation does).
        A mid-Pacific column plus an orography-masked column exercise
        exactly those zero-denominator cells.
        """
        height, dz, _ = _column_grid(ncols=2)
        oro = jnp.array([0.0, 8000.0])
        lats, lons = jnp.array([0.0, 0.0]), jnp.array([180.0, 180.0])
        data = AerosolData.zeros((2,), 30, n_bnd_sw=1, n_bnd_lw=1)
        f = _forcing_ones()

        def loss(aod_spmx):
            p = AerosolParameters.default()
            p = type(p)(**{**{k: getattr(p, k) for k in (
                'plume_lat', 'plume_lon', 'beta_a',
                'beta_b', 'aod_fmbg', 'asy550', 'ssa550', 'angstrom',
                'sig_lon_E', 'sig_lon_W', 'sig_lat_E', 'sig_lat_W', 'theta',
                'ftr_weight', 'background_aod', 'spa_prefactor',
                'spa_exponent', 'spa_cap_smoothing')}, 'aod_spmx': aod_spmx})
            out = get_simple_aerosol(height, dz, oro, lats, lons, data, p,
                                     f, jnp.asarray([550.0]))
            # Sum the fields whose divisions had the NaN-gradient risk.
            return (jnp.sum(out.ssa_profile) + jnp.sum(out.asy_profile)
                    + jnp.sum(out.aod_profile))

        g = jax.grad(loss)(AerosolParameters.default().aod_spmx)
        assert jnp.all(jnp.isfinite(g)), g

    def test_gradient_wrt_coordinates(self):
        p = AerosolParameters.default()

        def total(lats, lons):
            gauss = _per_feature_plume_gaussians(lats, lons, p)
            cw_an, _ = get_plume_column_weights(
                p, jnp.ones(NPLUMES), jnp.ones((NFEATURES, NPLUMES)), gauss,
            )
            return jnp.sum(cw_an)

        lats = jnp.linspace(-80, 80, 20)
        lons = jnp.linspace(0, 350, 20)
        gl, gn = jax.grad(total, argnums=(0, 1))(lats, lons)
        assert jnp.all(jnp.isfinite(gl)) and jnp.all(jnp.isfinite(gn))
