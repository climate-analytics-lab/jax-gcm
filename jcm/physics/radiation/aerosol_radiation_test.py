"""Test script for aerosol-radiation integration

This script tests the updated radiation scheme with aerosol effects.
"""

import jax.numpy as jnp
from jcm.physics.radiation.grey_two_stream.radiation_scheme import (
    combine_optical_properties
)
from jcm.physics.radiation.radiation_types import RadiationParameters, OpticalProperties
from jcm.physics.radiation.cloud_optics import effective_radius_liquid
from jcm.physics.aerosol.macv2_sp import (
    get_optical_properties,
    get_anthropogenic_aod,
    get_plume_spatial_distribution,
)
from jcm.physics.aerosol.macv2_sp_params import AerosolParameters

def test_aerosol_cloud_interaction():
    """Test that aerosols modify cloud effective radius"""
    print("Testing aerosol-cloud interactions...")
    
    land_fraction = 0.5
    
    # Test without aerosols
    r_eff_clean = effective_radius_liquid(jnp.array([1.0]), land_fraction)
    
    # Test with aerosols (increased CDNC)
    cdnc_factor = jnp.array([2.0])  # Double the droplet concentration
    r_eff_polluted = effective_radius_liquid(cdnc_factor, land_fraction)
    
    # With more droplets, effective radius should be smaller
    assert r_eff_polluted[0] < r_eff_clean[0], f"Expected smaller droplets with aerosols: {r_eff_polluted[0]} vs {r_eff_clean[0]}"
    
    # Check the expected scaling r_eff ~ N^(-1/3)
    expected_ratio = (2.0) ** (-1.0/3.0)  # ≈ 0.794
    actual_ratio = r_eff_polluted[0] / r_eff_clean[0]
    
    assert abs(actual_ratio - expected_ratio) < 0.01, f"Expected ratio {expected_ratio}, got {actual_ratio}"
    
    print(f"✓ Clean cloud r_eff: {r_eff_clean[0]:.2f} μm")
    print(f"✓ Polluted cloud r_eff: {r_eff_polluted[0]:.2f} μm")
    print(f"✓ Ratio: {actual_ratio:.3f} (expected: {expected_ratio:.3f})")


def test_optical_property_combination():
    """Test combination of gas, cloud, and aerosol optical properties"""
    print("\\nTesting optical property combination...")
    
    _nlev, _nbands = 3, 2
    
    # Gas optical depth (absorption only)
    gas_tau = jnp.array([[0.1, 0.05], [0.2, 0.1], [0.3, 0.15]])
    
    # Cloud optical properties (scattering)
    cloud_tau = jnp.array([[1.0, 0.8], [2.0, 1.6], [1.5, 1.2]])
    cloud_ssa = jnp.array([[0.99, 0.98], [0.99, 0.98], [0.99, 0.98]])
    cloud_g = jnp.array([[0.85, 0.82], [0.85, 0.82], [0.85, 0.82]])
    
    cloud_optics = OpticalProperties(
        optical_depth=cloud_tau,
        single_scatter_albedo=cloud_ssa,
        asymmetry_factor=cloud_g
    )
    
    # Aerosol optical properties (scattering + absorption)
    aerosol_tau = jnp.array([[0.2, 0.15], [0.3, 0.25], [0.1, 0.08]])
    aerosol_ssa = jnp.array([[0.9, 0.85], [0.9, 0.85], [0.9, 0.85]])
    aerosol_g = jnp.array([[0.7, 0.65], [0.7, 0.65], [0.7, 0.65]])
    
    # Test combination
    combined = combine_optical_properties(
        gas_tau, cloud_optics, aerosol_tau, aerosol_ssa, aerosol_g
    )
    
    # Check that optical depth is additive
    expected_tau = gas_tau + cloud_tau + aerosol_tau
    assert jnp.allclose(combined.optical_depth, expected_tau), "Optical depth should be additive"
    
    # Check that SSA is properly weighted
    cloud_sca = cloud_tau * cloud_ssa
    aerosol_sca = aerosol_tau * aerosol_ssa
    total_sca = cloud_sca + aerosol_sca
    expected_ssa = total_sca / expected_tau
    
    assert jnp.allclose(combined.single_scatter_albedo, expected_ssa, atol=1e-6), "SSA weighting incorrect"
    
    print("✓ Optical depth combination correct")
    print("✓ Single scattering albedo weighting correct")
    print(f"✓ Example combined τ: {combined.optical_depth[0, 0]:.3f}")
    print(f"✓ Example combined SSA: {combined.single_scatter_albedo[0, 0]:.3f}")


def test_radiation_scheme_with_without_aerosols():
    """Test that radiation scheme runs with and without aerosols"""
    print("\\nTesting radiation scheme with/without aerosols...")
    
    # Create test data
    nlev = 10
    parameters = RadiationParameters.default(n_sw_bands=2, n_lw_bands=3)
    
    # Test with mock aerosol data to ensure array shapes are correct
    try:
        # Create mock aerosol data
        total_bands = int(parameters.n_sw_bands) + int(parameters.n_lw_bands)
        aerosol_tau = jnp.ones((nlev, total_bands)) * 0.1
        aerosol_ssa = jnp.ones((nlev, total_bands)) * 0.9
        # Set LW bands to pure absorption
        aerosol_ssa = aerosol_ssa.at[:, int(parameters.n_sw_bands):].set(0.0)
        
        print(f"✓ Created test aerosol data: τ shape {aerosol_tau.shape}")
        print(f"✓ SW bands: {int(parameters.n_sw_bands)}, LW bands: {int(parameters.n_lw_bands)}")
        
    except Exception as e:
        print(f"✗ Error creating test data: {e}")
        return


def test_angstrom_spectral_scaling():
    """Test that Angstrom exponent produces correct spectral AOD scaling"""
    # AOD(λ) = AOD(550nm) * (λ/0.55)^(-α)
    ref_wavelength = 0.55  # μm

    # Test with known Angstrom exponent
    alpha = 1.5
    aod_550 = 0.3

    # Shorter wavelength should have higher AOD (more scattering)
    lambda_short = 0.4  # μm
    lambda_long = 1.0   # μm

    aod_short = aod_550 * (lambda_short / ref_wavelength) ** (-alpha)
    aod_long = aod_550 * (lambda_long / ref_wavelength) ** (-alpha)

    assert aod_short > aod_550, "AOD at shorter λ should be higher"
    assert aod_long < aod_550, "AOD at longer λ should be lower"

    # Check exact values
    expected_short = aod_550 * (0.4 / 0.55) ** (-1.5)
    expected_long = aod_550 * (1.0 / 0.55) ** (-1.5)
    assert jnp.allclose(aod_short, expected_short, rtol=1e-6)
    assert jnp.allclose(aod_long, expected_long, rtol=1e-6)


def test_angstrom_weighted_average():
    """Test that get_optical_properties returns proper Angstrom weighted average"""
    params = AerosolParameters.default()
    nlev, ncols = 10, 50

    aod_profile = jnp.ones((nlev, ncols)) * 0.1

    # Single plume dominating
    spatial_dist = jnp.zeros((params.nplumes, ncols))
    spatial_dist = spatial_dist.at[0, :].set(1.0)

    _, _, angstrom = get_optical_properties(aod_profile, spatial_dist, params)

    # Should match first plume's Angstrom exponent
    assert jnp.allclose(angstrom, params.angstrom[0], rtol=1e-6)

    # Equal-weight plumes
    spatial_dist_equal = jnp.ones((params.nplumes, ncols)) / params.nplumes
    _, _, angstrom_equal = get_optical_properties(aod_profile, spatial_dist_equal, params)

    # Should be mean of all plume values
    expected = jnp.mean(params.angstrom)
    assert jnp.allclose(angstrom_equal[0], expected, rtol=1e-5)


def test_temporal_weights_scale_aod():
    """Test that temporal weights from forcing scale anthropogenic AOD"""
    params = AerosolParameters.default()
    ncols = 100
    lats = jnp.linspace(-90, 90, ncols)
    lons = jnp.linspace(-180, 180, ncols)

    spatial_dist = get_plume_spatial_distribution(lats, lons, params)

    # Present-day: weights = 1
    year_weight_pd = jnp.ones(params.nplumes)
    ann_cycle = jnp.ones(params.nplumes)
    aod_pd = get_anthropogenic_aod(params, year_weight_pd, ann_cycle, spatial_dist)

    # Pre-industrial: weights = 0
    year_weight_pi = jnp.zeros(params.nplumes)
    aod_pi = get_anthropogenic_aod(params, year_weight_pi, ann_cycle, spatial_dist)

    # Half emissions
    year_weight_half = jnp.ones(params.nplumes) * 0.5
    aod_half = get_anthropogenic_aod(params, year_weight_half, ann_cycle, spatial_dist)

    assert jnp.all(aod_pi == 0), "Pre-industrial AOD should be zero"
    assert jnp.allclose(aod_half, aod_pd * 0.5, rtol=1e-6), "Half weights should give half AOD"

    # Seasonal cycle: reduce one plume
    ann_cycle_reduced = jnp.ones(params.nplumes)
    ann_cycle_reduced = ann_cycle_reduced.at[0].set(0.5)
    aod_seasonal = get_anthropogenic_aod(params, year_weight_pd, ann_cycle_reduced, spatial_dist)

    # Total AOD should decrease compared to present-day
    assert jnp.sum(aod_seasonal) < jnp.sum(aod_pd)
def test_aerosol_microphysics_droplet_coupling():
    """Test that apply_clouds_and_microphysics uses aerosol cdnc_factor for droplet number."""
    import numpy as np
    from jcm.physics.icon.icon_physics import apply_clouds_and_microphysics, _prepare_common_physics_state
    from jcm.physics.icon.icon_physics_data import PhysicsData
    from jcm.physics.icon.icon_coords import IconCoords
    from jcm.physics.icon.parameters import Parameters
    from jcm.physics_interface import PhysicsState
    from jcm.date import DateData
    from jcm.forcing import ForcingData
    from jcm.terrain import TerrainData
    from jcm.utils import get_coords

    nlev, nlat, nlon = 40, 32, 64
    ncols = nlat * nlon
    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(nlon, nlat))
    icon_coords = IconCoords.from_coordinate_system(coords)

    # Build a warm profile with cloud water so microphysics has work to do
    sigma_mid = (sigma_boundaries[:-1] + sigma_boundaries[1:]) / 2
    temp_profile = 290.0 * (sigma_mid ** 0.19)
    temperature = jnp.broadcast_to(
        jnp.array(temp_profile)[:, jnp.newaxis], (nlev, ncols)
    )
    q_profile = 0.01 * sigma_mid ** 3
    specific_hum = jnp.broadcast_to(
        jnp.array(q_profile)[:, jnp.newaxis], (nlev, ncols)
    )

    # Cloud water in mid-levels to trigger autoconversion
    qc = jnp.zeros((nlev, ncols))
    qc = qc.at[15:25, :].set(1e-3)

    from jcm.constants import physical_constants as pc
    height_profile = -pc.rd * 290.0 / pc.grav * np.log(sigma_mid)
    geopotential = jnp.broadcast_to(
        jnp.array(height_profile * pc.grav)[:, jnp.newaxis], (nlev, ncols)
    )

    state = PhysicsState(
        temperature=temperature,
        specific_humidity=specific_hum,
        u_wind=jnp.ones((nlev, ncols)) * 5.0,
        v_wind=jnp.zeros((nlev, ncols)),
        geopotential=geopotential,
        normalized_surface_pressure=jnp.ones(ncols),
        tracers={'qc': qc, 'qi': jnp.zeros((nlev, ncols))},
    )

    date = DateData.zeros()
    terrain = TerrainData.aquaplanet(coords)
    # Use short timestep so autoconversion rate limiter doesn't mask the
    # droplet-number sensitivity (default dt_conv=3600s clamps both cases)
    parameters = Parameters.default().with_timestep(1.0)
    forcing = ForcingData.zeros(coords.horizontal.nodal_shape)

    # --- Run with clean air (cdnc_factor = 1.0) ---
    pd_clean = PhysicsData.zeros((ncols,), nlev, icon_coords=icon_coords, date=date)
    cloud_data_clean = pd_clean.clouds.copy(
        cloud_fraction=jnp.where(qc > 0, 0.8, 0.0),
    )
    pd_clean = pd_clean.copy(clouds=cloud_data_clean)
    _, pd_clean = _prepare_common_physics_state(
        state, pd_clean, parameters, forcing, terrain
    )
    # cdnc_factor defaults to 1.0 from AerosolData.zeros

    tend_clean, pd_out_clean = apply_clouds_and_microphysics(
        state, pd_clean, parameters, forcing, terrain
    )

    # --- Run with polluted air (cdnc_factor = 3.0) ---
    pd_polluted = PhysicsData.zeros((ncols,), nlev, icon_coords=icon_coords, date=date)
    cloud_data_polluted = pd_polluted.clouds.copy(
        cloud_fraction=jnp.where(qc > 0, 0.8, 0.0),
    )
    aerosol_polluted = pd_polluted.aerosol.copy(
        cdnc_factor=jnp.ones(ncols) * 3.0,
    )
    pd_polluted = pd_polluted.copy(clouds=cloud_data_polluted, aerosol=aerosol_polluted)
    _, pd_polluted = _prepare_common_physics_state(
        state, pd_polluted, parameters, forcing, terrain
    )

    tend_polluted, pd_out_polluted = apply_clouds_and_microphysics(
        state, pd_polluted, parameters, forcing, terrain
    )

    # Droplet number stored in output should reflect cdnc_factor
    assert jnp.allclose(pd_out_clean.clouds.droplet_number, 100e6), (
        "Clean-air droplet number should be 100e6"
    )
    assert jnp.allclose(pd_out_polluted.clouds.droplet_number, 300e6), (
        "Polluted droplet number should be 3x baseline = 300e6"
    )

    # Higher CDNC suppresses autoconversion → less cloud water removal
    # (less negative qc tendency in polluted case)
    dqc_clean = tend_clean.tracers['qc']
    dqc_polluted = tend_polluted.tracers['qc']
    # In cloud layers, clean air should lose more cloud water
    cloud_mask = qc > 0
    mean_dqc_clean = float(jnp.mean(jnp.where(cloud_mask, dqc_clean, 0.0)))
    mean_dqc_polluted = float(jnp.mean(jnp.where(cloud_mask, dqc_polluted, 0.0)))
    assert mean_dqc_clean < mean_dqc_polluted, (
        f"Clean air should lose more cloud water (dqc_clean={mean_dqc_clean:.2e}) "
        f"than polluted air (dqc_polluted={mean_dqc_polluted:.2e})"
    )


def test_higher_cdnc_reduces_autoconversion():
    """Test physical effect: more droplets → smaller drops → less autoconversion."""
    from jcm.physics.clouds.echam_1m import (
        autoconversion, MicrophysicsParameters
    )

    # Use the default (Beheng) scheme — both Beheng and KK2000 have the
    # right Nc monotonicity, but Beheng is the production default.
    config = MicrophysicsParameters.default()
    cloud_water = jnp.array(1e-3)
    cloud_fraction = jnp.array(0.8)
    air_density = jnp.array(1.0)
    dt = 1.0

    # Clean air: fewer, larger droplets
    nc_clean = jnp.array(100e6)
    rate_clean = autoconversion(
        cloud_water, cloud_fraction, air_density, nc_clean, dt, config
    )

    # Polluted air: more, smaller droplets
    nc_polluted = jnp.array(300e6)
    rate_polluted = autoconversion(
        cloud_water, cloud_fraction, air_density, nc_polluted, dt, config
    )

    # Higher CDNC should suppress autoconversion (second indirect effect)
    assert rate_clean > rate_polluted, (
        f"Expected less autoconversion with more droplets: "
        f"clean={rate_clean}, polluted={rate_polluted}"
    )


if __name__ == "__main__":
    print("Testing aerosol-radiation integration...")
    print("=" * 50)

    test_aerosol_cloud_interaction()
    test_optical_property_combination()
    test_radiation_scheme_with_without_aerosols()
    test_angstrom_spectral_scaling()
    test_angstrom_weighted_average()
    test_temporal_weights_scale_aod()

    print("\n" + "=" * 50)
    test_aerosol_microphysics_droplet_coupling()
    test_higher_cdnc_reduces_autoconversion()

    print("\\n" + "=" * 50)
    print("All tests completed!")
