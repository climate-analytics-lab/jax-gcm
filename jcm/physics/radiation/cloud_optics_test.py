"""Unit tests for cloud optics calculations

Tests cloud optical properties including extinction, scattering,
and asymmetry parameters for both water and ice clouds.

Date: 2025-01-10
"""

import jax
import pytest
import jax.numpy as jnp
from jcm.physics.radiation.cloud_optics import (
    cloud_optics,
    effective_radius_liquid,
    effective_radius_ice,
    get_band_wavelength,
)
from jcm.physics.radiation.constants import (
    SW_BAND_LIMITS,
    LW_BAND_LIMITS,
    N_SW_BANDS,
    N_LW_BANDS,
)
from jcm.testing import check_gradients


@pytest.mark.parametrize("is_sw", [True, False], ids=["sw", "lw"])
def test_band_wavelength_within_band_limits(is_sw):
    """Every band's representative wavelength lies inside that band.

    This single assertion catches the #678 band-mapping bug: the cloud
    optics loops over ``N_SW_BANDS``/``N_LW_BANDS`` but the old code indexed
    hardcoded 6-SW/8-LW wavelength tables, so band 0 (the near-IR band that
    carries the SW cloud absorption) was evaluated at 0.245 um and absorbed
    nothing. Deriving the wavelength from the band's own wavenumber limits
    guarantees ``lambda(b)`` falls between ``1e4/wn_hi`` and ``1e4/wn_lo``.
    """
    limits = SW_BAND_LIMITS if is_sw else LW_BAND_LIMITS
    n_bands = N_SW_BANDS if is_sw else N_LW_BANDS
    for band in range(n_bands):
        wn_lo, wn_hi = limits[band]
        wl_lo_um, wl_hi_um = 1.0e4 / wn_hi, 1.0e4 / wn_lo
        wl = float(get_band_wavelength(band, is_sw=is_sw))
        assert wl_lo_um <= wl <= wl_hi_um, (
            f"band {band} wavelength {wl} um outside "
            f"[{wl_lo_um}, {wl_hi_um}] um"
        )


def test_near_ir_sw_band_absorbs():
    """The near-IR SW band must carry real cloud absorption (ssa < 1).

    Regression for #678: with the old 0.245 um mapping both SW bands had
    ssa ~ 0.99999 (zero absorption). Band 0 is now the near-IR band
    (0.69-2.5 um, 1.08 um centre), where liquid water absorbs, so its
    single-scatter albedo must be measurably below the UV/visible band's.
    """
    nlev = 1
    cwp = jnp.array([0.1])          # 100 g/m2 liquid layer
    cip = jnp.zeros(nlev)
    dz = jnp.array([1000.0])
    sw_optics, _ = cloud_optics(cwp, cip, dz, jnp.array(1.0))
    ssa_near_ir = float(sw_optics.single_scatter_albedo[0, 0])
    ssa_uv_vis = float(sw_optics.single_scatter_albedo[0, 1])
    assert ssa_near_ir < ssa_uv_vis
    assert ssa_near_ir < 0.9999


def test_effective_radius_liquid():
    """The fallback radius is a constant scaled by the Twomey factor.

    There is deliberately no land/ocean contrast: it was a CCN proxy that
    double-counted with cdnc_factor, and ECHAM's land term is a 6% spectral
    breadth factor superseded wherever a droplet number exists (#670).
    """
    r_eff_clean = effective_radius_liquid(jnp.array(1.0))
    assert r_eff_clean.shape == ()
    assert float(r_eff_clean) == pytest.approx(11.0)

    # More droplets at fixed water -> smaller droplets, as N^(-1/3).
    r_eff_polluted = effective_radius_liquid(jnp.array(8.0))
    assert float(r_eff_polluted) == pytest.approx(11.0 / 2.0)


def test_effective_radius_ice():
    """Moss/Foot power law: r_eff = 83.8 * IWC^0.216 (in-cloud IWC, g/m3).

    ECHAM mo_cloud_optics.f90:358 reference values.
    """
    # 0.01 g/m3 -> 83.8 * 0.01**0.216 ~ 31 um
    r_mid = effective_radius_ice(jnp.array(0.01))
    assert jnp.allclose(r_mid, 83.8 * 0.01**0.216, rtol=1e-6)
    assert 30.0 < r_mid < 32.0

    # Thin cirrus, 1e-4 g/m3 -> ~11.4 um (the fabricated T-ramp formula
    # produced 40-160 um here, saturating the RRTMGP LUT edge).
    r_thin = effective_radius_ice(jnp.array(1e-4))
    assert jnp.allclose(r_thin, 83.8 * 1e-4**0.216, rtol=1e-6)
    assert r_thin < 15.0

    # Monotonically increasing with IWC, profile shape preserved
    iwc = jnp.logspace(-5, 0, 10)
    r_eff = effective_radius_ice(iwc)
    assert r_eff.shape == (10,)
    assert jnp.all(jnp.diff(r_eff) > 0)

    # Zero-IWC guard: finite, positive value and a finite gradient
    # (double-where around the x**0.216 singularity at x = 0).
    import jax
    r_zero, grad_zero = jax.value_and_grad(
        lambda x: effective_radius_ice(x)
    )(jnp.array(0.0))
    assert jnp.isfinite(r_zero) and r_zero > 0
    assert jnp.isfinite(grad_zero)


def test_cloud_optics_integration():
    """Test the main cloud_optics function"""
    nlev = 15
    
    # Create mixed cloud profile
    cloud_water_path = jnp.zeros(nlev)
    cloud_ice_path = jnp.zeros(nlev)
    
    # Water clouds in lower levels
    cloud_water_path = cloud_water_path.at[10:].set(0.1)
    
    # Ice clouds in upper levels
    cloud_ice_path = cloud_ice_path.at[2:8].set(0.05)
    
    layer_thickness = jnp.full(nlev, 500.0)  # m

    # Calculate cloud optics
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Check output shapes - now using fixed bands
    from jcm.physics.radiation.constants import N_SW_BANDS, N_LW_BANDS
    assert sw_optics.optical_depth.shape == (nlev, N_SW_BANDS)
    assert sw_optics.single_scatter_albedo.shape == (nlev, N_SW_BANDS)
    assert sw_optics.asymmetry_factor.shape == (nlev, N_SW_BANDS)
    
    assert lw_optics.optical_depth.shape == (nlev, N_LW_BANDS)
    assert lw_optics.single_scatter_albedo.shape == (nlev, N_LW_BANDS)
    assert lw_optics.asymmetry_factor.shape == (nlev, N_LW_BANDS)
    
    # Physical constraints
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(lw_optics.optical_depth >= 0)
    
    assert jnp.all(sw_optics.single_scatter_albedo >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo <= 1)
    assert jnp.all(lw_optics.single_scatter_albedo >= 0)
    assert jnp.all(lw_optics.single_scatter_albedo <= 1)
    
    # No NaN values
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))
    
    # Clear-sky levels should have zero optical depth
    assert jnp.all(sw_optics.optical_depth[0, :] == 0)
    assert jnp.all(lw_optics.optical_depth[0, :] == 0)
    
    # Cloudy levels should have non-zero optical depth
    assert jnp.any(sw_optics.optical_depth[5, :] > 0)  # Ice cloud level
    assert jnp.any(sw_optics.optical_depth[12, :] > 0)  # Water cloud level


def test_cloud_optics_no_clouds():
    """Test cloud optics with no clouds"""
    nlev = 10
    cloud_water_path = jnp.zeros(nlev)
    cloud_ice_path = jnp.zeros(nlev)
    layer_thickness = jnp.full(nlev, 500.0)

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Should have zero optical depth everywhere
    assert jnp.allclose(sw_optics.optical_depth, 0.0)
    assert jnp.allclose(lw_optics.optical_depth, 0.0)
    
    # Single scattering albedo should be physical (but not used when tau=0)
    assert jnp.all(sw_optics.single_scatter_albedo >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo <= 1)


def test_cloud_optics_extreme_values():
    """Test cloud optics with extreme cloud water/ice paths"""
    nlev = 5
    layer_thickness = jnp.full(nlev, 500.0)

    # Very small cloud water/ice
    cloud_water_path = jnp.ones(nlev) * 1e-8
    cloud_ice_path = jnp.ones(nlev) * 1e-8

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )

    # Should handle small values without NaN
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))

    # Very large cloud water/ice
    cloud_water_path = jnp.ones(nlev) * 10.0  # Very thick clouds
    cloud_ice_path = jnp.ones(nlev) * 5.0

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Should handle large values
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))
    
    # Should have high optical depths (only first 2 bands for SW, 3 for LW)
    assert jnp.all(sw_optics.optical_depth[:, :2] > 1.0)
    assert jnp.all(lw_optics.optical_depth[:, :3] > 0.1)


def test_cloud_optics_iwc_dependence():
    """Ice optics respond to in-cloud IWC via the Moss/Foot r_eff.

    Same ice path spread over a thinner layer means higher in-cloud IWC,
    hence larger crystals — the optical properties must differ.
    """
    nlev = 10
    cloud_water_path = jnp.zeros(nlev)
    cloud_ice_path = jnp.ones(nlev) * 0.05

    # Thick layers: low IWC, small crystals
    sw_thick, lw_thick = cloud_optics(
        cloud_water_path, cloud_ice_path, jnp.full(nlev, 2000.0),
        jnp.array(1.0),
    )
    # Thin layers: high IWC, large crystals
    sw_thin, lw_thin = cloud_optics(
        cloud_water_path, cloud_ice_path, jnp.full(nlev, 100.0),
        jnp.array(1.0),
    )

    assert sw_thick.optical_depth.shape == sw_thin.optical_depth.shape
    assert jnp.all(sw_thick.optical_depth >= 0)
    assert jnp.all(sw_thin.optical_depth >= 0)
    # The r_eff difference must actually reach the optics
    assert not jnp.allclose(
        sw_thick.optical_depth, sw_thin.optical_depth
    )


def test_cloud_optics_mixed_phase():
    """Test mixed-phase clouds (both water and ice)"""
    layer_thickness = jnp.full(8, 500.0)

    # Mixed phase: water and ice coexist
    cloud_water_path = jnp.array([0.0, 0.1, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0])
    cloud_ice_path = jnp.array([0.0, 0.0, 0.05, 0.1, 0.15, 0.1, 0.05, 0.0])

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Total optical depth should be combination of water and ice
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(lw_optics.optical_depth >= 0)
    
    # Levels with both water and ice should have higher optical depth
    mixed_level = 3  # Both water and ice present
    
    # Mixed phase should have substantial optical depth (only first n_bands)
    assert jnp.all(sw_optics.optical_depth[mixed_level, :2] > 0)
    assert jnp.all(lw_optics.optical_depth[mixed_level, :3] > 0)


def test_cloud_optics_band_variations():
    """Test spectral variations across bands"""
    nlev = 5
    cloud_water_path = jnp.ones(nlev) * 0.2
    cloud_ice_path = jnp.zeros(nlev)
    layer_thickness = jnp.full(nlev, 500.0)

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Should have variations across bands
    # (Exact variations depend on parameterization)
    for i in range(nlev):
        if cloud_water_path[i] > 0:
            # SW bands should have some optical depth
            sw_tau_level = sw_optics.optical_depth[i, :]
            assert jnp.any(sw_tau_level > 0)
            
            # LW bands should have some optical depth
            lw_tau_level = lw_optics.optical_depth[i, :]
            assert jnp.any(lw_tau_level > 0)
    
    # Check that all bands have reasonable values
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(lw_optics.optical_depth >= 0)
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))


def test_cloud_optics_scattering_properties():
    """Test scattering properties of clouds"""
    nlev = 5
    cloud_water_path = jnp.ones(nlev) * 0.2
    cloud_ice_path = jnp.ones(nlev) * 0.1
    layer_thickness = jnp.full(nlev, 500.0)

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # SW should have high single scattering albedo (clouds scatter well in visible) - only first 2 bands
    assert jnp.all(sw_optics.single_scatter_albedo[:, :2] > 0.8)
    
    # LW should have lower single scattering albedo (more absorption in IR)
    # Note: Different number of bands, so compare averages
    lw_ssa_avg = jnp.mean(lw_optics.single_scatter_albedo, axis=1)
    sw_ssa_avg = jnp.mean(sw_optics.single_scatter_albedo, axis=1)
    assert jnp.all(lw_ssa_avg <= sw_ssa_avg)
    
    # Asymmetry factor should be physical (only first bands have values)
    assert jnp.all(sw_optics.asymmetry_factor[:, :2] >= -1)
    assert jnp.all(sw_optics.asymmetry_factor[:, :2] <= 1)
    assert jnp.all(lw_optics.asymmetry_factor[:, :3] >= -1)
    assert jnp.all(lw_optics.asymmetry_factor[:, :3] <= 1)
    
    # Clouds typically have forward scattering (g > 0)
    cloudy_levels = cloud_water_path + cloud_ice_path > 0
    if jnp.any(cloudy_levels):
        assert jnp.any(sw_optics.asymmetry_factor > 0)

class TestCloudOpticsGradients:
    """AD against a central difference through ``cloud_optics`` (#820).

    Green wherever a two-sided derivative exists. Which is not everywhere:
    **zero condensate is a discontinuity of this function, by construction.**
    A layer with ``cloud_water_path == cloud_ice_path == 0`` gets the
    clear-sky fill values ``ssa = 1`` and ``g = 0``
    (``cloud_optics.py:621/631``), while the limit of the tau-weighted
    combination as the paths go to zero is the condensate's own ``ssa`` and
    ``g`` — about 0.9999 and 0.80 for the ice deck in
    ``test_cloud_optics_integration``. Crossing zero therefore jumps ``g`` by
    0.8, not by an epsilon, and a central difference across it reports
    ``jump/eps``. ``tau`` itself is continuous there but kinked, by the
    ``jnp.maximum(tau, 0.0)`` floors at ``:357`` and ``:436``.

    Neither is a defect: every consumer weights ``ssa`` and ``g`` by ``tau``,
    which is zero in exactly those layers, so the jump is inert. So the
    comparisons below are made on profiles that are cloudy at **every**
    level, and the all-clear and mixed clear/cloudy columns are checked for
    finiteness instead — which is the property that matters for a column
    that evolves clear mid-rollout.

    The ``sqrt(12/r_eff)`` and ``sqrt(35/r_eff)`` size factors
    (``:497``/``:555``) are safe at zero condensate for a reason worth
    stating: ``effective_radius_ice`` double-``where``s its ``iwc**0.216``
    and returns a finite 83.8 um at zero IWC, and the liquid radius does not
    depend on the water path at all, so neither square root ever meets a zero
    denominator.
    """

    NLEV = 12

    @staticmethod
    def _layer_thickness(nlev=NLEV):
        return jnp.full(nlev, 500.0)

    @pytest.mark.parametrize("seed", [0, 4])
    @pytest.mark.parametrize("scale", [1.0, 1.0e-5], ids=["thick", "small"])
    def test_gradients_match_a_central_difference(self, scale, seed):
        """Cloudy at every level, at deck and at near-threshold amounts.

        The ``small`` case is 1e-5 of the ``thick`` one — in-cloud paths of
        O(1e-7) kg/m2, far below anything a radiation call cares about but
        still strictly positive, so it probes the approach to the zero
        boundary without sitting on it.
        """
        cloud_water_path = scale * jnp.linspace(0.01, 0.09, self.NLEV)
        cloud_ice_path = scale * jnp.linspace(0.002, 0.03, self.NLEV)
        # rtol is 5e-3, not 1e-3: the reference here is a FLOAT32 central
        # difference (the suite stays f32 by design, #729), and the near-IR SW
        # band now carries real liquid absorption (#678) -- its Mie
        # geometric-optics ``1 - exp(-4*pi*n_imag*r/lambda)`` term makes the
        # tau-weighted ssa/g respond to the paths, raising the f32
        # cancellation floor of the finite difference to ~2e-3 of the
        # derivative. The analytic gradient is verified correct: rerun in
        # float64 and the one-sided secants agree to ~1e-6, i.e. the function
        # is smooth here and only the f32 FD reference is noisy.
        check_gradients(
            cloud_optics,
            (cloud_water_path, cloud_ice_path, self._layer_thickness(),
             jnp.array(1.2)),
            rtol=5e-3, seed=seed)

    @pytest.mark.parametrize("kind", ["clear", "decks"])
    def test_gradients_are_finite_at_zero_condensate(self, kind):
        """No input may return a non-finite gradient at zero condensate.

        ``clear`` is an entirely cloud-free column; ``decks`` is the mixed
        profile a real column has, with exact zeros above, between and below
        two decks. Both put layers on the ``tau == 0`` boundary where the
        ``ssa``/``g`` safe-denominator guards (``:616-630``) select their
        clear-sky branch, and the assertion is that the discarded branch does
        not leak a ``0 * inf`` back through the ``jnp.where``.
        """
        if kind == "clear":
            cloud_water_path = jnp.zeros(self.NLEV)
            cloud_ice_path = jnp.zeros(self.NLEV)
        else:
            cloud_water_path = jnp.zeros(self.NLEV).at[8:11].set(0.08)
            cloud_ice_path = jnp.zeros(self.NLEV).at[2:5].set(0.03)
        args = (cloud_water_path, cloud_ice_path, self._layer_thickness(),
                jnp.array(1.2))

        def total(*a):
            return sum(jnp.sum(leaf ** 2)
                       for leaf in jax.tree.leaves(cloud_optics(*a)))

        gradients = jax.grad(total, argnums=(0, 1, 2, 3))(*args)
        names = ("cloud_water_path", "cloud_ice_path", "layer_thickness",
                 "cdnc_factor")
        for name, gradient in zip(names, gradients):
            assert jnp.all(jnp.isfinite(gradient)), (
                f"d/d{name} is not finite in a {kind} column: {gradient}")
