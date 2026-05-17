"""Unit tests for the ICON two-moment cloud microphysics scheme.

Ported from icon-physics-v1-amq; narrowed to the tests that exercise the
process functions currently wired into ``cloud_microphysics_2m_minimal``
(CloudUtils, FreezingBelow238K, Autoconversion_2M / precip_formation_warm).
The remaining 2M test classes from the amq branch (mixed-phase deposition,
sedimentation, update_tendencies, etc.) will be ported alongside Phase 5b
as the full ECHAM6 sequence is wired into the orchestrator — see #341.
"""

import jax
import jax.numpy as jnp
from math import pi

from .cloud_utils import (
    get_util_var,
    get_cloud_bounds,
    eff_ice_crystal_radius,
    minimum_CDNC,
)
from .lohmann_2m import (
    precip_formation_warm,
    precip_formation_cold,
    demott2010_inp,
    freezing_below_238K,
    het_mxphase_freezing,
    WBF_process,
    melting_snow_and_ice,
    sublimation_snow_and_ice_evaporation_rain,
    sedimentation_ice,
    mixed_phase_deposition_and_corrections,
    update_in_cloud_water,
    update_precip_fluxes,
    update_tendencies_and_important_vars,
    microphysics_dt_constants,
)
from .lohmann_2m_params import (
    cqtmin,
    ldyn_cdnc_min,
    rcd_vol_max,
    cdnc_min_fixed,
    cdnc_min_lower,
    cdnc_min_upper,
    fact_PK,
    pow_PK,
    tmelt,
    icemin,
    eps,
    clc_min,
)
from jcm.constants import rhoh2o, alhs, alhc, t0, rv


def _zeros(n: int) -> jnp.ndarray:
    return jnp.zeros((n,), dtype=jnp.float32)


def _full(n: int, v: float) -> jnp.ndarray:
    return jnp.full((n,), v, dtype=jnp.float32)


class TestCloudUtils:
    """Test utility functions for cloud microphysics"""

    def test_get_util_var(self):
        """Test utility variable calculations."""
        nproma, nbdim, ntdia, nlev, nlevp1 = 1, 1, 0, 3, 4
        paphm1 = jnp.array([[700.0, 800.0, 900.0, 1000.0]])  # Pressure at half levels
        pgeo = jnp.array([[300.0, 200.0, 100.0]])  # Geopotential at full levels
        papm1 = jnp.array([[750.0, 850.0, 950.0]])  # Pressure at full levels
        ptm1 = jnp.array([[260.0, 270.0, 280.0]])  # Temperature at full levels

        pgeoh, pdp, pdpg, pdz, paaa, pviscos = get_util_var(
            nproma, nbdim, ntdia, nlev, nlevp1, paphm1, pgeo, papm1, ptm1
        )

        # Check geopotential at half levels
        expected_pgeoh = jnp.array([[350.0, 250.0, 150.0, 0.0]])
        assert jnp.allclose(pgeoh, expected_pgeoh), f"Expected {expected_pgeoh}, got {pgeoh}"

        # Check pressure differences
        expected_pdp = jnp.array([[100.0, 100.0, 100.0]])
        assert jnp.allclose(pdp, expected_pdp), f"Expected {expected_pdp}, got {pdp}"

        # Check height differences
        expected_pdz = jnp.array([[10.19367991845056, 10.19367991845056, 15.2905199]])
        assert jnp.allclose(pdz, expected_pdz), f"Expected {expected_pdz}, got {pdz}"

        # Check air density correction
        expected_paaa = jnp.array([[1.8467386, 1.7793932, 1.7196922]])
        assert jnp.allclose(paaa, expected_paaa), f"Expected {expected_paaa}, got {paaa}"

        # Check dynamic viscosity
        expected_pviscos = jnp.array([[1.65162e-05, 1.70362e-05, 1.75562e-05]])
        assert jnp.allclose(pviscos, expected_pviscos), f"Expected {expected_pviscos}, got {pviscos}"

    def test_get_cloud_bounds(self):
        """Test the get_cloud_bounds function."""
        nproma = 1  # Number of columns
        nbdim = 1   # Number of rows
        ntdia = 0   # Starting level index
        nlev = 7    # Number of levels

        # Cloud cover array (paclc)
        paclc = jnp.array([[0.0, 0.8, 0.6, 0.0, 0.8, 0.6, 0.5]])  # Cloud between levels 1 to 2 and 4 to 6

        # Call the function
        ktop, kbas, kcl_minustop, kcl_minusbas = get_cloud_bounds(nproma, nbdim, ntdia, nlev, paclc)

        # Expected outputs
        expected_ktop = jnp.array([[0, 1, 0, 0, 4, 0, 0]])  # Cloud top at level 1 & 4
        expected_kbas = jnp.array([[0, 0, 2, 0, 0, 0, 6]])  # Cloud base at level 2 & 6
        expected_kcl_minustop = jnp.array([[0, 0, 1, 0, 0, 4, 4]])  # Cloud levels excluding top
        expected_kcl_minusbas = jnp.array([[0, 2, 0, 0, 6, 6, 0]])  # Cloud levels excluding base

        # Assertions
        assert jnp.array_equal(ktop, expected_ktop), f"ktop: Expected {expected_ktop}, got {ktop}"
        assert jnp.array_equal(kbas, expected_kbas), f"kbas: Expected {expected_kbas}, got {kbas}"
        assert jnp.array_equal(kcl_minustop, expected_kcl_minustop), f"lcl_minustop: Expected {expected_kcl_minustop}, got {kcl_minustop}"
        assert jnp.array_equal(kcl_minusbas, expected_kcl_minusbas), f"kcl_minusbas: Expected {expected_kcl_minusbas}, got {kcl_minusbas}"
    
    def test_eff_ice_crystal_radius(self):
        # Positive, non-degenerate inputs so the eps-guards do not affect the result
        pxice = jnp.array([0.1, 1.0, 10.0], dtype=jnp.float32)   # [g/m^3]
        picnc = jnp.array([1e5, 1e6, 1e7], dtype=jnp.float32)    # [1/m^3]

        got = eff_ice_crystal_radius(pxice, picnc)
        expected = 0.5e4 * (pxice / (fact_PK * picnc)) ** (1.0 / pow_PK)

        assert got.shape == expected.shape
        assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)
    
    def test_minimum_CDNC(self):
        pxwat = jnp.array([0.0, 1e-6, 1e-4, 1e-2], dtype=jnp.float32)  # [kg/m^3]
        got = minimum_CDNC(pxwat)

        if ldyn_cdnc_min:
            expected = rcd_vol_max ** (-3.0) * (3.0 / (4.0 * pi * rhoh2o)) * pxwat
            expected = jnp.clip(expected, cdnc_min_lower, cdnc_min_upper)
        else:
            expected = jnp.full_like(pxwat, cdnc_min_fixed * 1.0e6)  # cm^-3 -> m^-3

        assert got.shape == pxwat.shape
        assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)

        # extra invariant: dynamic branch must be within clip bounds
        if ldyn_cdnc_min:
            assert jnp.all(got >= cdnc_min_lower)
            assert jnp.all(got <= cdnc_min_upper)


class TestFreezingBelow238K:
    """Unit tests for the freezing_below_238K function."""

    def _base_inputs(self, n: int = 4):
        """Generate base inputs for the freezing_below_238K function."""
        return dict(
            freezing_condition=jnp.array([True, False, True, False]),  # Alternating freezing conditions
            cloud_cover=jnp.full((n,), 0.8),  # Cloud cover fraction
            min_cdnc=jnp.full((n,), 1e6),  # Minimum CDNC [1/m^3]
            ice_crystal_number=jnp.full((n,), 5e5),  # Initial ICNC [1/m^3]
            droplet_freezing_rate=jnp.full((n,), 1e4),  # Initial freezing rate [m^-3/s]
            droplet_number=jnp.full((n,), 1e7),  # Initial CDNC [1/m^3]
            freezing_rate=jnp.full((n,), 0.0),  # Initial freezing rate [kg/kg]
            cloud_ice=jnp.full((n,), 0.001),  # Cloud ice mixing ratio [kg/kg]
            cloud_liquid=jnp.full((n,), 0.002),  # Cloud liquid water mixing ratio [kg/kg]
            timestep=60.0,  # Time step [s]
            min_liquid_threshold=cqtmin,  # Minimum liquid water threshold [kg/kg]
        )

    def test_freezing_updates_correctly(self):
        """Test that freezing updates cloud ice, liquid, and droplet properties correctly."""
        inputs = self._base_inputs()
        outputs = freezing_below_238K(**inputs)

        # Extract outputs
        ice_crystal_number, droplet_freezing_rate, droplet_number, freezing_rate, cloud_ice, cloud_liquid = outputs

        # Check that freezing occurred where the condition is True
        assert jnp.all(cloud_liquid[inputs["freezing_condition"]] == 0.0)  # Liquid water should be zero where freezing occurs
        assert jnp.all(cloud_ice[inputs["freezing_condition"]] > inputs["cloud_ice"][inputs["freezing_condition"]])  # Ice should increase
        assert jnp.all(droplet_number[inputs["freezing_condition"]] == cqtmin)  # Droplet number should be reduced to the minimum threshold

        # Check that no changes occurred where the condition is False
        assert jnp.all(cloud_liquid[~inputs["freezing_condition"]] == inputs["cloud_liquid"][~inputs["freezing_condition"]])
        assert jnp.all(cloud_ice[~inputs["freezing_condition"]] == inputs["cloud_ice"][~inputs["freezing_condition"]])
        assert jnp.all(droplet_number[~inputs["freezing_condition"]] == inputs["droplet_number"][~inputs["freezing_condition"]])

    def test_no_freezing_when_condition_false(self):
        """Test that no freezing occurs when the freezing condition is False everywhere."""
        inputs = self._base_inputs()
        inputs["freezing_condition"] = jnp.full((4,), False)  # No freezing condition
        outputs = freezing_below_238K(**inputs)

         # Map outputs to their corresponding keys
        output_keys = [
            "ice_crystal_number",
            "droplet_freezing_rate",
            "droplet_number",
            "freezing_rate",
            "cloud_ice",
            "cloud_liquid",
        ]

        # Outputs should match inputs
        for key, output in zip(output_keys, outputs):
            assert jnp.all(output == inputs[key]), f"Mismatch for key: {key}"

    def test_freezing_with_min_cdnc(self):
        """Test that droplet number concentration is reduced to the minimum threshold."""
        inputs = self._base_inputs()
        inputs["droplet_number"] = jnp.array([1e7, 5e5, 2e6, 1e6])  # Varying initial CDNC
        outputs = freezing_below_238K(**inputs)

        # Check that droplet number is reduced to the minimum threshold where freezing occurs
        droplet_number = outputs[2]
        assert jnp.all(droplet_number[inputs["freezing_condition"]] == cqtmin)
        assert jnp.all(droplet_number[~inputs["freezing_condition"]] == inputs["droplet_number"][~inputs["freezing_condition"]])

    def test_freezing_rate_accumulation(self):
        """Test that the freezing rate accumulates correctly."""
        inputs = self._base_inputs()
        inputs["freezing_rate"] = jnp.array([0.0, 0.1, 0.2, 0.3])  # Initial freezing rates
        outputs = freezing_below_238K(**inputs)

        # outputs: ice_crystal_number, droplet_freezing_rate, droplet_number, freezing_rate, cloud_ice, cloud_liquid
        droplet_freezing_rate = outputs[1]
        droplet_number = outputs[2]
        freezing_rate_mass = outputs[3]

        mask = inputs["freezing_condition"]
        assert jnp.any(mask)

        # mass-based freezing_rate should increase where freezing occurs
        assert jnp.all(freezing_rate_mass[mask] > inputs["freezing_rate"][mask] + 1e-12)

        # droplet number should not increase where freezing occurs (may be reduced to cqtmin)
        assert jnp.all(droplet_number[mask] <= inputs["droplet_number"][mask] + 1e-12)

        # the droplet_freezing_rate diagnostic may decrease depending on semantics; just ensure it's finite
        assert jnp.all(jnp.isfinite(droplet_freezing_rate))


    def test_jittable(self): # FAILED iterable error, TODO might need to convert inputs to tuples or something else that is hashable for jit
        """Test that the function is JIT-compatible."""
        inputs = self._base_inputs()
        freezing_below_238K_jit = jax.jit(freezing_below_238K)
        outputs = freezing_below_238K_jit(**inputs)

        # Ensure outputs are finite and consistent
        for output in outputs:
            assert jnp.all(jnp.isfinite(output))

class TestAutoconversion_2M:
    def test_precip_formation_warm_mask_false_no_change(self):
        """If warm_precip_mask is False everywhere, outputs should be zero rates and unchanged inputs."""
        # config = CloudParams2M.default()

        shape = (5,)
        warm_precip_mask = jnp.zeros(shape, dtype=bool)

        autoconversion_factor = jnp.ones(shape)
        cloud_fraction = jnp.full(shape, 0.5)
        minimum_cloud_precip_fraction = jnp.full(shape, 0.1)
        air_density = jnp.full(shape, 1.0)
        rain_water = jnp.full(shape, 1e-4)
        minimum_droplet_number = jnp.full(shape, 1e6)
        droplet_number_in = jnp.full(shape, 2e6)
        cloud_water_in = jnp.full(shape, 1e-3)
        dt = jnp.full(shape, 10.0)

        droplet_number, cloud_water, pmratepr, prpr, prprn = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt
        )

        assert jnp.allclose(droplet_number, droplet_number_in)
        assert jnp.allclose(cloud_water, cloud_water_in)
        assert jnp.allclose(pmratepr, jnp.zeros_like(cloud_water_in))
        assert jnp.allclose(prpr, jnp.zeros_like(cloud_water_in))
        assert jnp.allclose(prprn, jnp.zeros_like(cloud_water_in))


    def test_precip_formation_warm_mask_true_reduces_cloud_water_and_nonnegative_rates(self):
        """If mask is True and cloud water is present, cloud water should not increase; rates should be >= 0."""
        # config = MicrophysicsParameters_2M.default()

        shape = (6,)
        warm_precip_mask = jnp.ones(shape, dtype=bool)

        autoconversion_factor = jnp.ones(shape)
        cloud_fraction = jnp.linspace(0.1, 1.0, shape[0])
        minimum_cloud_precip_fraction = jnp.full(shape, 0.2)
        air_density = jnp.full(shape, 1.0)
        rain_water = jnp.full(shape, 5e-4)
        minimum_droplet_number = jnp.full(shape, 1e6)

        droplet_number_in = jnp.full(shape, 2e6)
        cloud_water_in = jnp.full(shape, 2e-3)
        dt = jnp.full(shape, 10.0)

        droplet_number, cloud_water, pmratepr, prpr, prprn = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt
            # config=config,
        )

        # Cloud water is reduced by autoconversion and accretion terms; should not increase.
        assert jnp.all(cloud_water <= cloud_water_in + 1e-12)

        # Formation rates should be nonnegative for physically meaningful inputs.
        assert jnp.all(pmratepr >= -1e-12)
        assert jnp.all(prpr >= -1e-12)
        assert jnp.all(prprn >= -1e-12)

        # Droplet number should not increase (autoconversion removes droplets); allow tiny eps.
        assert jnp.all(droplet_number <= droplet_number_in + 1e-8)

    def test_precip_formation_warm_mixed_mask_only_updates_true_elements(self):
        """Only elements where mask is True should be modified."""
        # config = MicrophysicsParameters_2M.default()

        warm_precip_mask = jnp.array([True, False, True, False])

        autoconversion_factor = jnp.ones_like(warm_precip_mask, dtype=jnp.float32)
        cloud_fraction = jnp.full((4,), 0.5)
        minimum_cloud_precip_fraction = jnp.full((4,), 0.1)
        air_density = jnp.full((4,), 1.0)
        rain_water = jnp.full((4,), 1e-4)
        minimum_droplet_number = jnp.full((4,), 1e6)

        droplet_number_in = jnp.full((4,), 2e6)
        cloud_water_in = jnp.full((4,), 1e-3)
        dt = jnp.full((4,), 10.0)

        droplet_number, cloud_water, pmratepr, prpr, prprn = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt
            # config=config,
        )

        false_idx = jnp.where(~warm_precip_mask)[0]

        assert jnp.allclose(droplet_number[false_idx], droplet_number_in[false_idx])
        assert jnp.allclose(cloud_water[false_idx], cloud_water_in[false_idx])
        assert jnp.allclose(pmratepr[false_idx], 0.0)
        assert jnp.allclose(prpr[false_idx], 0.0)
        assert jnp.allclose(prprn[false_idx], 0.0)

    def test_precip_formation_cold_basic_invariants_and_shapes(self):
        """Smoke/invariant test for precip_formation_cold.

        Checks:
        - output shapes match input shapes
        - outputs are finite
        - non-negativity for formation rates (pspr, psacl, psacln, psprn, pmsnowacl)
        - droplet_number is not reduced below cqtmin
        - in-cloud condensates are not negative
        """
        n = 6
        dt = jnp.array(60.0, dtype=jnp.float32)

        # Make 3 points "active" (cloudy with ice+liquid+snow) and 3 "inactive"
        cloud_mask = jnp.array([True, True, True, False, True, False])

        cloud_fraction = jnp.array([0.3, 0.5, 0.1, 0.0, 0.2, 0.0], dtype=jnp.float32)
        autoconversion_factor = jnp.array([1.0, 0.7, 0.3, 0.0, 0.5, 0.0], dtype=jnp.float32)
        minimum_cloud_precip_fraction = jnp.minimum(cloud_fraction, jnp.array([0.2] * n, dtype=jnp.float32))

        air_density = jnp.array([1.2] * n, dtype=jnp.float32)
        inv_air_density = 1.0 / air_density
        inv_air_density_rcp = 1.0 / air_density  # keep identical for test

        temperature = jnp.array([260.0, 255.0, 268.0, 280.0, 250.0, 275.0], dtype=jnp.float32)
        dynamic_viscosity = jnp.array([1.8e-5] * n, dtype=jnp.float32)

        # Snow from above: present only for active points to trigger riming/accretion
        snow_mass_mmr_from_above = jnp.array([1e-5, 2e-5, 5e-6, 0.0, 1e-5, 0.0], dtype=jnp.float32)

        # In-cloud ice and liquid: positive for active points
        in_cloud_ice = jnp.array([2e-4, 1e-4, 5e-5, 0.0, 2e-4, 0.0], dtype=jnp.float32)
        in_cloud_liquid = jnp.array([1e-4, 2e-4, 1e-4, 0.0, 5e-5, 0.0], dtype=jnp.float32)

        # Number concentrations
        ice_number = jnp.array([1e5, 2e5, 5e4, 1e5, 3e5, 1e5], dtype=jnp.float32)
        droplet_number = jnp.array([5e7, 2e7, 1e7, 5e7, 4e7, 5e7], dtype=jnp.float32)

        # Minimum droplet number (pcdnc_min)
        minimum_droplet_number = jnp.array([1e6] * n, dtype=jnp.float32)

        snow_rate_in_cloud = jnp.zeros((n,), dtype=jnp.float32)

        outs = precip_formation_cold(
            cloud_mask=cloud_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            inverse_air_density=inv_air_density,
            inverse_air_density_rcp=inv_air_density_rcp,
            temperature=temperature,
            dynamic_viscosity=dynamic_viscosity,
            snow_mass_mmr_from_above=snow_mass_mmr_from_above,
            air_density=air_density,
            minimum_droplet_number=minimum_droplet_number,
            ice_number=ice_number,
            droplet_number=droplet_number,
            snow_rate_in_cloud=snow_rate_in_cloud,
            in_cloud_ice=in_cloud_ice,
            in_cloud_liquid=in_cloud_liquid,
            dt=dt,
        )

        assert len(outs) == 10
        (
            ice_number_o,
            droplet_number_o,
            snow_rate_in_cloud_o,
            in_cloud_ice_o,
            in_cloud_liquid_o,
            psprn,
            psacl,
            psacln,
            pmsnowacl,
            pspr,
        ) = outs

        for arr in outs:
            assert arr.shape == (n,)
            assert jnp.all(jnp.isfinite(arr)), "All outputs must be finite"

        # Invariants / basic physical bounds
        assert jnp.all(in_cloud_ice_o >= 0.0)
        assert jnp.all(in_cloud_liquid_o >= 0.0)
        assert jnp.all(droplet_number_o >= cqtmin)
        assert jnp.all(ice_number_o >= 0.0)

        # Formation/accretion diagnostics should never be negative
        assert jnp.all(pspr >= 0.0)
        assert jnp.all(psprn >= 0.0)
        assert jnp.all(psacl >= 0.0)
        assert jnp.all(psacln >= 0.0)
        assert jnp.all(pmsnowacl >= 0.0)

        # If a point is completely non-cloudy, outputs should remain "quiet" (rates zero)
        inactive = ~cloud_mask
        assert jnp.all(pspr[inactive] == 0.0)
        assert jnp.all(psacl[inactive] == 0.0)
        assert jnp.all(psacln[inactive] == 0.0)
        assert jnp.all(psprn[inactive] == 0.0)



class TestMeltingSnowIce_2M:
    def test_melting_snow_and_ice(self):
        dt = jnp.array(60.0, dtype=jnp.float32)

        temperature_previous = jnp.array([tmelt + 1.0, tmelt - 1.0], dtype=jnp.float32)
        melt_mask = temperature_previous > tmelt

        pressure_thickness = jnp.array([1.0e4, 1.0e4], dtype=jnp.float32)
        lsdcp = jnp.array([2.8e3, 2.8e3], dtype=jnp.float32)
        lvdcp = jnp.array([2.5e3, 2.5e3], dtype=jnp.float32)

        ice_cloud_previous = jnp.array([1e-4, 1e-4], dtype=jnp.float32)
        ice_tendency = jnp.array([1e-6, 1e-6], dtype=jnp.float32)

        icncq = jnp.array([2e5, 2e5], dtype=jnp.float32)
        icnc = jnp.array([1e6, 1e6], dtype=jnp.float32)
        cdnc = jnp.array([1e8, 1e8], dtype=jnp.float32)
        qmel = jnp.array([0.0, 0.0], dtype=jnp.float32)

        rain_flux = jnp.array([1e-5, 1e-5], dtype=jnp.float32)
        snow_flux = jnp.array([2e-5, 2e-5], dtype=jnp.float32)

        ice_flux = jnp.array([1.0e-5, 1.0e-5], dtype=jnp.float32)
        ice_flux_n = jnp.array([1.0e7, 1.0e7], dtype=jnp.float32)

        (
            icnc_o, qmel_o, cdnc_o,
            rain_flux_o, snow_flux_o,
            ice_flux_o, ice_flux_n_o,
            ice_tendency_o, pimlt, psmlt, pximlt,
        ) = melting_snow_and_ice(
            melt_mask=melt_mask,
            temperature_previous=temperature_previous,
            ice_cloud_previous=ice_cloud_previous,
            pressure_thickness=pressure_thickness,
            icncq=icncq, lsdcp=lsdcp, lvdcp=lvdcp,
            icnc=icnc, qmel=qmel, cdnc=cdnc,
            rain_flux=rain_flux, snow_flux=snow_flux,
            ice_flux=ice_flux, ice_flux_n=ice_flux_n,
            ice_tendency=ice_tendency, dt=dt,
        )

        assert icnc_o.shape == (2,)
        assert jnp.all(jnp.isfinite(icnc_o))
        assert jnp.all(jnp.isfinite(rain_flux_o))
        assert jnp.all(jnp.isfinite(snow_flux_o))

        # Melt point: ICNC -> icemin, transferred number into CDNC
        assert float(icnc_o[0]) == float(icemin)
        assert float(cdnc_o[0]) == float(cdnc[0] + icncq[0])
        assert float(qmel_o[0]) == float(qmel[0] + dt * icncq[0])

        # Non-melt point: numbers unchanged
        assert float(icnc_o[1]) == float(icnc[1])
        assert float(cdnc_o[1]) == float(cdnc[1])
        assert float(qmel_o[1]) == float(qmel[1])

        # Diagnostics non-negative
        assert float(pimlt[0]) >= 0.0
        assert float(psmlt[0]) >= 0.0
        assert float(pximlt[0]) >= 0.0
        assert jnp.all(ice_flux_n_o >= 0.0)
        assert jnp.all(ice_flux_o >= 0.0)


class TestSublimationSnowIceEvapRain_2M:
    def _common_inputs(self, n: int):
        dt = jnp.array(60.0, dtype=jnp.float32)
        return dict(
            dt=dt,
            specific_humidity_prev=_full(n, 1.0e-3),
            temperature_prev=_full(n, 260.0),
            precip_fraction=_full(n, 0.5),
            falling_ice_fraction=_full(n, 0.5),
            pressure_thickness=_full(n, 1.0e4),
            dp_over_g=_full(n, 1.0e3),
            subsat_wrt_ice=_full(n, -1e-5),
            lsdcp=_full(n, 2.8e3),
            inv_air_density=1.0 / _full(n, 1.2),
            qsat_ice=_full(n, 2.0e-3),
            inv_air_density_rcp=1.0 / _full(n, 1.2),
            snow_flux=_zeros(n),
            air_density=_full(n, 1.2),
            qsat_water_prev=_full(n, 2.0e-3),
            rain_flux=_zeros(n),
            subsat_wrt_water_evap=_full(n, -1e-5),
            thermo_term_water=_full(n, 1.0),
            ice_flux=_zeros(n),
            ice_flux_n=_full(n, 1.0e7),
        )

    def test_snow_sublimation_only(self):
        n = 4
        x = self._common_inputs(n)
        precip_mask = jnp.array([True, True, False, True])
        falling_ice_mask = jnp.array([False, False, False, False])
        x["snow_flux"] = jnp.array([2.0e-4, 1.0e-4, 2.0e-4, 0.0], dtype=jnp.float32)
        x["ice_flux_n"] = _zeros(n)

        ice_flux_o, ice_flux_n_o, ice_sublim, snow_sublim, rain_evap = (
            sublimation_snow_and_ice_evaporation_rain(
                precip_mask=precip_mask, falling_ice_mask=falling_ice_mask, **x,
            )
        )

        assert float(snow_sublim[0]) > 0.0
        assert float(snow_sublim[1]) > 0.0
        assert float(snow_sublim[2]) == 0.0
        assert float(snow_sublim[3]) == 0.0
        assert jnp.all(ice_sublim == 0.0)
        assert jnp.all(rain_evap == 0.0)
        assert jnp.allclose(ice_flux_o, x["ice_flux"])
        assert jnp.all(snow_sublim >= 0.0)

    def test_falling_ice_sublimation_reduces_fluxes(self):
        n = 4
        x = self._common_inputs(n)
        precip_mask = jnp.array([False, False, False, False])
        falling_ice_mask = jnp.array([True, True, False, True])
        ice_flux_in = jnp.array([2.0e-4, 1.0e-4, 5.0e-4, 2.0e-4], dtype=jnp.float32)
        ice_flux_n_in = jnp.array([2.0e7, 1.0e7, 1.0e7, 2.0e7], dtype=jnp.float32)
        x["ice_flux"] = ice_flux_in
        x["ice_flux_n"] = ice_flux_n_in

        ice_flux_o, ice_flux_n_o, ice_sublim, snow_sublim, rain_evap = (
            sublimation_snow_and_ice_evaporation_rain(
                precip_mask=precip_mask, falling_ice_mask=falling_ice_mask, **x,
            )
        )

        assert float(ice_sublim[0]) > 0.0
        assert float(ice_sublim[1]) > 0.0
        assert float(ice_sublim[2]) == 0.0
        assert float(ice_sublim[3]) > 0.0
        assert float(ice_flux_o[0]) < float(ice_flux_in[0])
        assert float(ice_flux_o[2]) == float(ice_flux_in[2])
        assert jnp.all(snow_sublim == 0.0)
        assert jnp.all(rain_evap == 0.0)
        assert jnp.all(ice_flux_o >= 0.0)
        assert jnp.all(ice_flux_n_o >= 0.0)

    def test_rain_evaporation_only(self):
        n = 4
        x = self._common_inputs(n)
        precip_mask = jnp.array([True, True, False, True])
        falling_ice_mask = jnp.array([False, False, False, False])
        x["rain_flux"] = jnp.array([3.0e-4, 1.0e-4, 2.0e-4, 0.0], dtype=jnp.float32)

        ice_flux_o, ice_flux_n_o, ice_sublim, snow_sublim, rain_evap = (
            sublimation_snow_and_ice_evaporation_rain(
                precip_mask=precip_mask, falling_ice_mask=falling_ice_mask, **x,
            )
        )

        assert float(rain_evap[0]) > 0.0
        assert float(rain_evap[1]) > 0.0
        assert float(rain_evap[2]) == 0.0
        assert float(rain_evap[3]) == 0.0
        assert jnp.all(snow_sublim == 0.0)
        assert jnp.all(ice_sublim == 0.0)
        assert jnp.all(rain_evap >= 0.0)


class TestSedimentationIce_2M:
    def _realistic_inputs(self, n: int):
        air_density = jnp.full((n,), 0.45, dtype=jnp.float32)
        cloud_fraction = jnp.array([0.8, 0.3, 0.0, 0.95], dtype=jnp.float32)
        ice_mmr_in_cloud = 5e-5
        ice_mmr_gridmean = jnp.array(
            [cloud_fraction[i] * ice_mmr_in_cloud if i != 2 else 0.0 for i in range(n)],
            dtype=jnp.float32,
        )
        icnc_in_cloud = jnp.array([5.0e4, 5.0e4, 5.0e4, 1.0e5], dtype=jnp.float32)
        vfall_typical = 0.3
        ice_flux_in = jnp.array(
            [vfall_typical * 0.45 * float(cloud_fraction[i]) * ice_mmr_in_cloud if i != 2 else 0.0 for i in range(n)],
            dtype=jnp.float32,
        )
        mean_crystal_mass = jnp.array(
            [0.45 * ice_mmr_in_cloud / float(icnc_in_cloud[i]) if i != 2 else 1e-12 for i in range(n)],
            dtype=jnp.float32,
        )
        ice_flux_n_in = ice_flux_in / jnp.maximum(mean_crystal_mass, 1e-20)
        ice_flux_n_in = ice_flux_n_in.at[2].set(0.0)
        return dict(
            cloud_fraction=cloud_fraction,
            air_density_correction=jnp.full((n,), 1.0, dtype=jnp.float32),
            pressure_thickness=jnp.full((n,), 3000.0, dtype=jnp.float32),
            air_density=air_density,
            inv_air_density_rcp=1.0 / air_density,
            ice_mmr_gridmean=ice_mmr_gridmean,
            icnc_in_cloud=icnc_in_cloud,
            ice_flux=ice_flux_in,
            ice_flux_n=ice_flux_n_in,
            falling_ice_fraction=jnp.array([0.5, 0.2, 0.0, 0.7], dtype=jnp.float32),
        )

    def test_sedimentation_reduces_cloud_ice_and_increases_flux(self):
        n = 4
        x = self._realistic_inputs(n)
        dt = jnp.asarray(60.0, dtype=jnp.float32)

        ice_mmr_o, icnc_o, ice_flux_o, ice_flux_n_o, falling_ice_frac_o, pmrateps_o = (
            sedimentation_ice(**x, dt=dt)
        )

        for arr in (ice_mmr_o, icnc_o, ice_flux_o, ice_flux_n_o, falling_ice_frac_o, pmrateps_o):
            assert jnp.all(jnp.isfinite(arr))

        assert jnp.all(ice_mmr_o >= 0.0)
        assert jnp.all(ice_flux_o >= 0.0)
        assert jnp.all(ice_flux_n_o >= 0.0)
        assert jnp.all(pmrateps_o >= 0.0)
        assert jnp.all(falling_ice_frac_o >= 0.0)
        assert jnp.all(falling_ice_frac_o <= 1.0)

        cloudy = x["cloud_fraction"] > clc_min
        assert jnp.all(ice_mmr_o[cloudy] <= x["ice_mmr_gridmean"][cloudy] + 1e-12)
        # No-cloud point (idx 2): unchanged
        assert jnp.allclose(ice_mmr_o[2], x["ice_mmr_gridmean"][2], atol=1e-10)
        # Flux from sedimentation should never decrease
        assert jnp.all(ice_flux_o - x["ice_flux"] >= -1e-12)

    def test_no_ice_no_sedimentation(self):
        n = 4
        x = self._realistic_inputs(n)
        dt = jnp.asarray(60.0, dtype=jnp.float32)
        x["ice_mmr_gridmean"] = jnp.zeros(n, dtype=jnp.float32)
        x["ice_flux"] = jnp.zeros(n, dtype=jnp.float32)
        x["ice_flux_n"] = jnp.zeros(n, dtype=jnp.float32)

        ice_mmr_o, icnc_o, ice_flux_o, ice_flux_n_o, _, pmrateps_o = (
            sedimentation_ice(**x, dt=dt)
        )

        assert jnp.allclose(ice_mmr_o, 0.0, atol=1e-12)
        assert jnp.allclose(ice_flux_o, 0.0, atol=1e-12)
        assert jnp.allclose(ice_flux_n_o, 0.0, atol=1e-12)
        assert jnp.allclose(pmrateps_o, 0.0, atol=1e-12)


class TestMixedPhaseDepositionAndCorrections2M:
    def _base_inputs(self, n: int = 4):
        T = jnp.full((n,), 240.0, dtype=jnp.float32)
        p = jnp.full((n,), 40000.0, dtype=jnp.float32)
        rho = jnp.full((n,), 0.45, dtype=jnp.float32)
        T_val = 240.0
        ztmp_ice = (alhs / rv) * (1.0 / t0 - 1.0 / T_val)
        ztmp_water = (alhc / rv) * (1.0 / t0 - 1.0 / T_val)
        esi_correct = 611 * jnp.exp(ztmp_ice)
        esw_correct = 611 * jnp.exp(ztmp_water)
        esi = jnp.full((n,), esi_correct, dtype=jnp.float32)
        esw = jnp.full((n,), esw_correct, dtype=jnp.float32)
        vtmpc1 = 0.608
        qsat_ice_internal = esi_correct / (float(p[0]) - (1.0 - 1.0 / (1.0 + vtmpc1)) * esi_correct)
        return dict(
            pressure=p,
            icnc=jnp.full((n,), 5e4, dtype=jnp.float32),
            specific_humidity_prev=jnp.full((n,), qsat_ice_internal * 0.98, dtype=jnp.float32),
            cloud_fraction=jnp.full((n,), 0.7, dtype=jnp.float32),
            sat_vap_pres_ice=esi,
            sat_vap_pres_water=esw,
            bergeron_variable=jnp.full((n,), 1e-3, dtype=jnp.float32),
            tompkins_genti=jnp.zeros((n,), dtype=jnp.float32),
            lsdcp=jnp.full((n,), 2.836e6 / 1004.0, dtype=jnp.float32),
            lvdcp=jnp.full((n,), 2.501e6 / 1004.0, dtype=jnp.float32),
            specific_humidity=jnp.full((n,), qsat_ice_internal * 1.5, dtype=jnp.float32),
            qsat_prev=jnp.full((n,), qsat_ice_internal, dtype=jnp.float32),
            air_density=rho,
            temperature=T,
            ice_evaporation=jnp.zeros((n,), dtype=jnp.float32),
            ice_mmr_gridmean=jnp.full((n,), 3e-5, dtype=jnp.float32),
            ice_detrainment_tendency=jnp.zeros((n,), dtype=jnp.float32),
            updraft_velocity=jnp.full((n,), 0.001, dtype=jnp.float32),
            condensation_rate=jnp.zeros((n,), dtype=jnp.float32),
            deposition_rate=jnp.zeros((n,), dtype=jnp.float32),
            dt=jnp.asarray(60.0, dtype=jnp.float32),
        )

    def _warm_inputs(self, n: int = 4):
        x = self._base_inputs(n)
        T = jnp.full((n,), 285.0, dtype=jnp.float32)
        p = jnp.full((n,), 85000.0, dtype=jnp.float32)
        esw = jnp.full((n,), 1400.0, dtype=jnp.float32)
        esi = jnp.full((n,), 1350.0, dtype=jnp.float32)
        qsw = esw / p
        x.update(
            pressure=p, temperature=T,
            air_density=jnp.full((n,), 1.0, dtype=jnp.float32),
            sat_vap_pres_ice=esi, sat_vap_pres_water=esw,
            specific_humidity=qsw * 1.03,
            specific_humidity_prev=qsw * 0.99, qsat_prev=qsw,
            updraft_velocity=jnp.full((n,), 1e6, dtype=jnp.float32),
            ice_mmr_gridmean=jnp.zeros((n,), dtype=jnp.float32),
            icnc=jnp.full((n,), 1e8, dtype=jnp.float32),
        )
        return x

    def test_outputs_finite_and_correct_shape_ice(self):
        x = self._base_inputs()
        outs = mixed_phase_deposition_and_corrections(**x)
        for arr in outs:
            assert arr.shape == (4,)
            assert jnp.all(jnp.isfinite(arr))

    def test_outputs_finite_and_correct_shape_liquid(self):
        x = self._warm_inputs()
        outs = mixed_phase_deposition_and_corrections(**x)
        for arr in outs:
            assert arr.shape == (4,)
            assert jnp.all(jnp.isfinite(arr))

    def test_ice_phase_produces_deposition_not_condensation(self):
        x = self._base_inputs()
        pcnd_o, pdep_o, *_ = mixed_phase_deposition_and_corrections(**x)
        assert jnp.all(pdep_o > 0.0)
        assert jnp.all(pcnd_o == 0.0)

    def test_liquid_phase_produces_condensation_not_deposition(self):
        x = self._warm_inputs()
        pcnd_o, pdep_o, *_ = mixed_phase_deposition_and_corrections(**x)
        assert jnp.all(pcnd_o > 0.0)
        assert jnp.all(pdep_o == 0.0)

    def test_temperature_thermodynamic_consistency_ice(self):
        x = self._base_inputs()
        pcnd_o, pdep_o, T_o, _, _ = mixed_phase_deposition_and_corrections(**x)
        T_expected = x["temperature"] + x["lsdcp"] * pdep_o + x["lvdcp"] * pcnd_o
        assert jnp.allclose(T_o, T_expected, atol=1e-4)

    def test_moisture_conservation_ice(self):
        x = self._base_inputs()
        pcnd_o, pdep_o, _, q_o, _ = mixed_phase_deposition_and_corrections(**x)
        q_expected = x["specific_humidity"] - pcnd_o - pdep_o
        assert jnp.allclose(q_o, q_expected, atol=1e-9)

    def test_pre_existing_deposition_is_accumulated(self):
        x = self._base_inputs()
        pdep_initial = jnp.full_like(x["deposition_rate"], 1e-6)
        x = {**x, "deposition_rate": pdep_initial}
        _, pdep_o, *_ = mixed_phase_deposition_and_corrections(**x)
        assert jnp.all(pdep_o >= pdep_initial - 1e-10)


class TestDeMott2010INP:
    def test_zero_outside_valid_range(self):
        T = jnp.array([280.0, 265.0, 237.0, 200.0], dtype=jnp.float32)
        n_inp = demott2010_inp(T, 0.5)
        assert float(n_inp[0]) == 0.0  # too warm
        assert float(n_inp[2]) == 0.0  # too cold
        assert float(n_inp[3]) == 0.0  # way too cold

    def test_nonzero_in_valid_range(self):
        T = jnp.array([260.0, 250.0, 240.0], dtype=jnp.float32)
        n_inp = demott2010_inp(T, 0.5)
        assert jnp.all(n_inp > 0.0)
        # Colder → more INP
        assert float(n_inp[2]) > float(n_inp[1]) > float(n_inp[0])

    def test_more_aerosol_more_inp(self):
        T = jnp.array([250.0], dtype=jnp.float32)
        n_low = demott2010_inp(T, 0.1)
        n_high = demott2010_inp(T, 2.0)
        assert float(n_high[0]) > float(n_low[0])

    def test_output_in_per_m3(self):
        T = jnp.array([250.0], dtype=jnp.float32)
        n_inp = demott2010_inp(T, 0.5)
        # Should be order 1e3–1e6 per m³ for typical conditions
        assert float(n_inp[0]) > 1.0
        assert float(n_inp[0]) < 1e10


class TestHetMxphaseFreezing:
    def _base_inputs(self, n: int = 4):
        return dict(
            freezing_condition=jnp.array([True, False, True, False]),
            pressure=jnp.full((n,), 90000.0),
            tke=jnp.full((n,), 0.1),
            vertical_velocity=jnp.full((n,), 0.2),
            cloud_cover=jnp.full((n,), 0.8),
            bc_soluble_fraction=jnp.full((n,), 0.1),
            bc_insoluble_fraction=jnp.full((n,), 0.05),
            dust_soluble_fraction=jnp.full((n,), 0.2),
            dust_accumulation_fraction=jnp.full((n,), 0.15),
            dust_coarse_fraction=jnp.full((n,), 0.1),
            air_density=jnp.full((n,), 1.0),
            inv_air_density=jnp.full((n,), 1.0),
            wet_radius_aitken=jnp.full((n,), 1e-7),
            wet_radius_accumulation=jnp.full((n,), 2e-7),
            wet_radius_coarse=jnp.full((n,), 3e-7),
            temperature=jnp.full((n,), 250.0),
            min_cdnc=jnp.full((n,), 1e6),
            ice_crystal_number=jnp.full((n,), 5e5),
            droplet_number=jnp.full((n,), 1e7),
            freezing_rate=jnp.full((n,), 0.0),
            cloud_ice=jnp.full((n,), 0.001),
            cloud_liquid=jnp.full((n,), 0.002),
            timestep=60.0,
            min_liquid_threshold=cqtmin,
        )

    def test_mxphase_no_freezing_when_condition_false(self):
        inputs = self._base_inputs()
        inputs["freezing_condition"] = jnp.full((4,), False)
        outputs = het_mxphase_freezing(**inputs)
        for key, output in zip(
            ["ice_crystal_number", "droplet_number", "freezing_rate", "cloud_ice", "cloud_liquid"],
            outputs[:5],
        ):
            assert jnp.all(output == inputs[key])

    def test_mxphase_min_cdnc_limit(self):
        inputs = self._base_inputs()
        inputs["droplet_number"] = jnp.array([1e7, 5e5, 2e6, 1e6])
        outputs = het_mxphase_freezing(**inputs)
        droplet_number = outputs[1]
        assert jnp.all(droplet_number[inputs["freezing_condition"]] >= cqtmin)
        assert jnp.all(
            droplet_number[~inputs["freezing_condition"]]
            == inputs["droplet_number"][~inputs["freezing_condition"]]
        )


class TestWBFProcess:
    def _base_inputs(self, n: int = 4):
        return dict(
            wbf_mask=jnp.array([True, False, True, False]),
            cloud_fraction=jnp.array([0.6, 0.6, 0.3, 0.0], dtype=jnp.float32),
            lsdcp=jnp.full((n,), 2.836e6 / 1004.0, dtype=jnp.float32),
            lvdcp=jnp.full((n,), 2.501e6 / 1004.0, dtype=jnp.float32),
            cdnc=jnp.array([5e7, 5e7, 5e7, 5e7], dtype=jnp.float32),
            cloud_liquid_in_cloud=jnp.array([2e-3, 1e-3, 5e-4, 1e-6], dtype=jnp.float32),
            cloud_ice_in_cloud=jnp.array([1e-4, 2e-4, 3e-4, 0.0], dtype=jnp.float32),
            cloud_liquid_tendency=jnp.array([1e-6, 2e-6, 3e-6, 4e-6], dtype=jnp.float32),
            cloud_ice_tendency=jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            temp_tendency=jnp.array([0.0, 1e-7, 2e-7, 3e-7], dtype=jnp.float32),
            dt=jnp.array(60.0, dtype=jnp.float32),
        )

    def test_wbf_applies_transfer_and_tendencies(self):
        inputs = self._base_inputs()
        cdnc_o, ql_o, qi_o, qlt_o, qit_o, t_o = WBF_process(**inputs)
        ztmst_rcp = 1.0 / jnp.maximum(inputs["dt"], eps)
        ztmp1 = ztmst_rcp * inputs["cloud_liquid_in_cloud"] * inputs["cloud_fraction"]
        mask = inputs["wbf_mask"]
        assert jnp.all(ql_o[mask] == 0.0)
        assert jnp.all(ql_o[~mask] == inputs["cloud_liquid_in_cloud"][~mask])
        assert jnp.allclose(
            qi_o[mask],
            inputs["cloud_ice_in_cloud"][mask] + inputs["cloud_liquid_in_cloud"][mask],
        )
        assert jnp.allclose(qlt_o[mask], inputs["cloud_liquid_tendency"][mask] - ztmp1[mask])
        assert jnp.allclose(qit_o[mask], inputs["cloud_ice_tendency"][mask] + ztmp1[mask])
        delta = (inputs["lsdcp"] - inputs["lvdcp"]) * ztmp1
        assert jnp.allclose(t_o[mask], inputs["temp_tendency"][mask] + delta[mask])

    def test_wbf_sets_cdnc_min_and_preserves_where_false(self):
        inputs = self._base_inputs()
        inputs["cdnc"] = jnp.array([1e8, 1e8, 1e5, 1e8], dtype=jnp.float32)
        cdnc_o, *_ = WBF_process(**inputs)
        mask = inputs["wbf_mask"]
        assert jnp.all(cdnc_o[mask] == cqtmin)
        assert jnp.all(cdnc_o[~mask] == inputs["cdnc"][~mask])

    def test_wbf_noop_when_mask_false_everywhere(self):
        inputs = self._base_inputs()
        inputs["wbf_mask"] = jnp.full((4,), False)
        before = {
            k: v.copy() for k, v in inputs.items()
            if k in (
                "cdnc", "cloud_liquid_in_cloud", "cloud_ice_in_cloud",
                "cloud_liquid_tendency", "cloud_ice_tendency", "temp_tendency",
            )
        }
        cdnc_o, ql_o, qi_o, qlt_o, qit_o, t_o = WBF_process(**inputs)
        assert jnp.allclose(cdnc_o, before["cdnc"])
        assert jnp.allclose(ql_o, before["cloud_liquid_in_cloud"])
        assert jnp.allclose(qi_o, before["cloud_ice_in_cloud"])
        assert jnp.allclose(qlt_o, before["cloud_liquid_tendency"])
        assert jnp.allclose(qit_o, before["cloud_ice_tendency"])
        assert jnp.allclose(t_o, before["temp_tendency"])


class TestUpdatePrecipFluxes_2M:
    def _base_inputs(self, n=4, dt=jnp.array(60.0, dtype=jnp.float32)):
        return dict(
            cloud_fraction=_zeros(n),
            pressure_thickness=_full(n, 1.0e4),
            rain_evap_mmr=_zeros(n),
            lsdcp=_full(n, 2.836e6 / 1004.0),
            lvdcp=_full(n, 2.501e6 / 1004.0),
            rain_formation=_zeros(n),
            snow_accretion=_zeros(n),
            snow_formation=_zeros(n),
            snow_sublimation_mmr=_zeros(n),
            temp_tmp=_full(n, 270.0),
            ice_flux_from_above=_zeros(n),
            precip_cover=_zeros(n),
            rain_flux=_zeros(n),
            snow_flux=_zeros(n),
            snow_melt=_zeros(n),
            dt=dt,
        )

    def test_no_sources_leaves_fluxes_unchanged(self):
        inp = self._base_inputs(4)
        out = update_precip_fluxes(**inp)
        for o, name in zip(out[:4], ("precip_cover", "rain_flux", "snow_flux", "snow_melt")):
            assert jnp.allclose(o, inp[name]), f"{name} changed"
        for arr in out[4:]:
            assert jnp.allclose(arr, 0.0)

    def test_rain_evaporation_reduces_rain_flux(self):
        n = 3
        inp = self._base_inputs(n)
        inp.update({
            "cloud_fraction": jnp.ones(n),
            "precip_cover": jnp.ones(n),
            "rain_flux": jnp.array([1e-4, 2e-4, 0.0], dtype=jnp.float32),
            "rain_evap_mmr": jnp.array([1e-4, 0.0, 5e-5], dtype=jnp.float32),
        })
        _, _, _, _, pfevapr, _, _, _ = update_precip_fluxes(**inp)
        _, _, _, zcons2, _ = microphysics_dt_constants(inp["dt"])
        expected_evap = (zcons2 * inp["pressure_thickness"] * inp["rain_evap_mmr"]).astype(pfevapr.dtype)
        precip_mask = pfevapr > 0.0
        assert jnp.allclose(pfevapr[precip_mask], expected_evap[precip_mask], atol=1e-6)
        assert jnp.all(pfevapr[~precip_mask] == 0.0)

    def test_incoming_ice_can_melt_into_rain_at_top(self):
        n = 2
        inp = self._base_inputs(n)
        inp.update({
            "cloud_fraction": _full(n, 0.8),
            "temp_tmp": jnp.full((n,), float(tmelt) + 2.0, dtype=jnp.float32),
            "ice_flux_from_above": jnp.array([1e-5, 0.0], dtype=jnp.float32),
        })
        _, rain_flux_o, _, snow_melt_o, *_ = update_precip_fluxes(**inp)
        assert float(rain_flux_o[0]) > 0.0
        assert float(snow_melt_o[0]) > 0.0
        assert float(rain_flux_o[1]) == 0.0
        assert float(snow_melt_o[1]) == 0.0


class TestUpdateInCloudWater_2M:
    def _base_inputs(self, n=4):
        dt = jnp.array(60.0, dtype=jnp.float32)
        flag_pattern = jnp.array([True, False, True, False], dtype=bool)
        cloud_flag = jnp.tile(flag_pattern, (n + flag_pattern.size - 1) // flag_pattern.size)[:n]
        cloud_fraction = jnp.where(cloud_flag, _full(n, 0.2), _zeros(n))
        return dict(
            pressure=_full(n, 8e4),
            activated_cdnc=_full(n, 1.0e6),
            condensation_rate=_zeros(n),
            deposition_rate=_zeros(n),
            tompkins_genti=_zeros(n),
            tompkins_gentl=_zeros(n),
            newly_formed_ice=_zeros(n),
            specific_humidity_tmp=_full(n, 1.0e-2),
            sat_spec_humidity_tmp=_full(n, 2.0e-2),
            air_density=_full(n, 1.2),
            ice_radius_mean=_full(n, 20e-6),
            temp_prev=_full(n, 280.0),
            cloud_flag=cloud_flag,
            ice_crystal_number=_full(n, 1.0),
            nucleation_rate=_zeros(n),
            droplet_number=_full(n, 1.0e5),
            cloud_fraction=cloud_fraction,
            cloud_ice_in_cloud=_zeros(n),
            cloud_liquid_in_cloud=_full(n, 1e-4),
            dt=dt,
        )

    def test_shapes_and_finite(self):
        inputs = self._base_inputs(6)
        outs = update_in_cloud_water(**inputs)
        assert isinstance(outs, tuple) and len(outs) == 8
        for out in outs:
            assert out.shape == inputs["pressure"].shape
            assert jnp.all(jnp.isfinite(out))

    def test_cloud_creation_initializes_incloud_values(self):
        n = 3
        inputs = self._base_inputs(n)
        inputs["cloud_flag"] = jnp.array([False, False, False])
        inputs["cloud_fraction"] = _zeros(n)
        inputs["condensation_rate"] = jnp.array([1e-6, 0.0, 1e-6], dtype=jnp.float32)
        outs = update_in_cloud_water(**inputs)
        cloud_flag_o, _, _, _, cloud_fraction_o, _, pxlb_o, _ = outs
        created_mask = inputs["condensation_rate"] > 0.0
        assert jnp.all(cloud_flag_o[created_mask])
        assert jnp.all(cloud_fraction_o[created_mask] > 0.0)
        assert jnp.any(pxlb_o[created_mask] > 0.0)

    def test_activation_increases_cdnc_and_accumulates(self):
        n = 2
        inputs = self._base_inputs(n)
        inputs["cloud_flag"] = jnp.array([True, True])
        inputs["cloud_fraction"] = _full(n, 0.3)
        inputs["cloud_liquid_in_cloud"] = _full(n, 5e-4)
        inputs["activated_cdnc"] = jnp.array([5e6, 5e6], dtype=jnp.float32)
        inputs["droplet_number"] = jnp.array([1e4, 2e5], dtype=jnp.float32)
        inputs["nucleation_rate"] = _zeros(n)
        before_cdnc = inputs["droplet_number"].copy()
        before_pqnuc = inputs["nucleation_rate"].copy()
        _, _, pqnuc_o, cdnc_o, _, _, _, _ = update_in_cloud_water(**inputs)
        assert jnp.all(cdnc_o >= before_cdnc)
        assert jnp.all(pqnuc_o >= before_pqnuc)

    def test_icnc_minimum_enforced_when_ice_present(self):
        n = 4
        inputs = self._base_inputs(n)
        inputs["cloud_flag"] = jnp.array([True, True, False, True])
        inputs["cloud_ice_in_cloud"] = jnp.array([0.0, 2e-4, 0.0, 5e-4], dtype=jnp.float32)
        inputs["ice_crystal_number"] = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        inputs["newly_formed_ice"] = jnp.full((n,), 1e8, dtype=jnp.float32)
        cloud_flag_o, icnc_o, _, _, _, pxib_o, _, _ = update_in_cloud_water(**inputs)
        mask = jnp.logical_and(cloud_flag_o, pxib_o > cqtmin)
        assert jnp.all(icnc_o[mask] >= icemin)
        assert jnp.all(icnc_o[~mask] == cqtmin)
        assert jnp.all(jnp.isfinite(icnc_o))


class TestUpdateTendencies_2M:
    def test_tracer_tendencies_and_shapes(self):
        from .lohmann_2m_params import ccwmin
        n = 4
        dt = jnp.array(60.0, dtype=jnp.float32)
        air_density = _full(n, 1.2)

        out = update_tendencies_and_important_vars(
            icnc=_full(n, 5e4), cdnc=_full(n, 1e8),
            ice_mmr_prev=_full(n, ccwmin * 1.1), liq_mmr_prev=_zeros(n),
            tracer_tm1_cdnc=_zeros(n), tracer_tm1_icnc=_zeros(n),
            condensation_rate=_full(n, 1e-6), deposition_rate=_zeros(n),
            rain_evap_mmr=_zeros(n), freezing_rate=_zeros(n),
            tompkins_ice=_zeros(n), tompkins_liq=_zeros(n),
            incloud_ice_melt=_zeros(n),
            lsdcp=_full(n, 2.836e6 / 1004.0), lvdcp=_full(n, 2.501e6 / 1004.0),
            air_density=air_density, inv_air_density=1.0 / air_density,
            rain_formation=_zeros(n), snow_accretion=_zeros(n),
            snow_formation=_zeros(n), cloud_ice_evap=_zeros(n),
            ice_flux_melt=_zeros(n), pxitec=_zeros(n), pxlevap=_zeros(n),
            pxltec=_zeros(n), pxisub=_zeros(n),
            snow_sublimation_mmr=_zeros(n), snow_melt=_zeros(n),
            cloud_ice_in_cloud=_zeros(n), cloud_liquid_in_cloud=_zeros(n),
            temp_tmp=_full(n, 280.0),
            liquid_cloud_flag=jnp.ones((n,), dtype=bool),
            ice_cloud_flag=jnp.ones((n,), dtype=bool),
            cloud_fraction=_full(n, 0.5),
            specific_humidity_tendency=_zeros(n), temp_tendency=_zeros(n),
            ice_tendency=_zeros(n), liq_tendency=_zeros(n),
            tracer_tendency_cdnc=_zeros(n), tracer_tendency_icnc=_zeros(n),
            incloud_liq_before_rain=_full(n, 1e-4),
            incloud_ice_before_snow=_full(n, 1e-4),
            dt=dt,
        )

        assert len(out) == 11
        for a in out:
            assert a.shape == (n,)
            assert jnp.all(jnp.isfinite(a))

        _, ztmst_rcp, _, _, _ = microphysics_dt_constants(dt)
        expected_tte_cdnc = ztmst_rcp * (_full(n, 1e8) * (1.0 / air_density) - _zeros(n))
        expected_tte_icnc = ztmst_rcp * (_full(n, 5e4) * (1.0 / air_density) - _zeros(n))
        assert jnp.allclose(out[5], expected_tte_cdnc)
        assert jnp.allclose(out[6], expected_tte_icnc)

    def test_small_cloud_fraction_zeroes_incloud_accumulators(self):
        n = 3
        dt = jnp.array(60.0, dtype=jnp.float32)

        out = update_tendencies_and_important_vars(
            icnc=_zeros(n), cdnc=_zeros(n),
            ice_mmr_prev=_zeros(n), liq_mmr_prev=_zeros(n),
            tracer_tm1_cdnc=_zeros(n), tracer_tm1_icnc=_zeros(n),
            condensation_rate=_zeros(n), deposition_rate=_zeros(n),
            rain_evap_mmr=_zeros(n), freezing_rate=_zeros(n),
            tompkins_ice=_zeros(n), tompkins_liq=_zeros(n),
            incloud_ice_melt=_zeros(n),
            lsdcp=_full(n, 2.836e6 / 1004.0), lvdcp=_full(n, 2.501e6 / 1004.0),
            air_density=_full(n, 1.2), inv_air_density=_full(n, 1.0 / 1.2),
            rain_formation=_zeros(n), snow_accretion=_zeros(n),
            snow_formation=_zeros(n), cloud_ice_evap=_zeros(n),
            ice_flux_melt=_zeros(n), pxitec=_zeros(n), pxlevap=_zeros(n),
            pxltec=_zeros(n), pxisub=_zeros(n),
            snow_sublimation_mmr=_zeros(n), snow_melt=_zeros(n),
            cloud_ice_in_cloud=_zeros(n), cloud_liquid_in_cloud=_zeros(n),
            temp_tmp=_full(n, 280.0),
            liquid_cloud_flag=jnp.zeros((n,), dtype=bool),
            ice_cloud_flag=jnp.zeros((n,), dtype=bool),
            cloud_fraction=_zeros(n),
            specific_humidity_tendency=_full(n, 1e-6),
            temp_tendency=_full(n, 1e-6),
            ice_tendency=_full(n, 1e-6), liq_tendency=_full(n, 1e-6),
            tracer_tendency_cdnc=_full(n, 1e-6),
            tracer_tendency_icnc=_full(n, 1e-6),
            incloud_liq_before_rain=_full(n, 1e-21),
            incloud_ice_before_snow=_full(n, 1e-21),
            dt=dt,
        )

        assert jnp.all(out[0] == 0.0)   # cloud_fraction
        assert jnp.all(out[7] == 0.0)   # incloud_liq
        assert jnp.all(out[8] == 0.0)   # incloud_ice

    def test_effective_radii_respect_cloud_flags(self):
        n = 2
        dt = jnp.array(60.0, dtype=jnp.float32)
        air_density = _full(n, 1.2)

        out = update_tendencies_and_important_vars(
            icnc=_full(n, 5e4), cdnc=_full(n, 1e7),
            ice_mmr_prev=_zeros(n), liq_mmr_prev=_zeros(n),
            tracer_tm1_cdnc=_zeros(n), tracer_tm1_icnc=_zeros(n),
            condensation_rate=_zeros(n), deposition_rate=_zeros(n),
            rain_evap_mmr=_zeros(n), freezing_rate=_zeros(n),
            tompkins_ice=_zeros(n), tompkins_liq=_zeros(n),
            incloud_ice_melt=_zeros(n),
            lsdcp=_full(n, 2.836e6 / 1004.0), lvdcp=_full(n, 2.501e6 / 1004.0),
            air_density=air_density, inv_air_density=1.0 / air_density,
            rain_formation=_zeros(n), snow_accretion=_zeros(n),
            snow_formation=_zeros(n), cloud_ice_evap=_zeros(n),
            ice_flux_melt=_zeros(n), pxitec=_zeros(n), pxlevap=_zeros(n),
            pxltec=_zeros(n), pxisub=_zeros(n),
            snow_sublimation_mmr=_zeros(n), snow_melt=_zeros(n),
            cloud_ice_in_cloud=_full(n, 2e-4),
            cloud_liquid_in_cloud=_full(n, 1e-4),
            temp_tmp=_full(n, 270.0),
            liquid_cloud_flag=jnp.zeros((n,), dtype=bool),
            ice_cloud_flag=jnp.zeros((n,), dtype=bool),
            cloud_fraction=_full(n, 0.5),
            specific_humidity_tendency=_zeros(n), temp_tendency=_zeros(n),
            ice_tendency=_zeros(n), liq_tendency=_zeros(n),
            tracer_tendency_cdnc=_zeros(n), tracer_tendency_icnc=_zeros(n),
            incloud_liq_before_rain=_zeros(n),
            incloud_ice_before_snow=_zeros(n),
            dt=dt,
        )

        assert jnp.all(out[9] == 0.0)   # liq_eff_radius
        assert jnp.all(out[10] == 0.0)  # ice_eff_radius


class TestIcon2MPipeline:
    """End-to-end checks that the 2M term composes into a runnable Model."""

    def test_factory_declares_six_tracers(self):
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(cloud_scheme="2m")
        names = {spec.name for spec in physics.required_tracers()}
        assert names == {"qc", "qi", "qnc", "qni", "qr", "qs"}
        nondim_flags = {
            spec.name: spec.nondimensionalize
            for spec in physics.required_tracers()
        }
        assert nondim_flags["qnc"] is False
        assert nondim_flags["qni"] is False
        assert nondim_flags["qc"] is True

    def test_model_runs_with_2m_and_stays_finite(self):
        """Short SPEEDY-grid run with the 2M composable physics; no NaNs."""
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        physics = echam_physics(cloud_scheme="2m", checkpoint_terms=False)
        model = Model(coords=get_speedy_coords(), physics=physics, time_step=180)
        preds = model.run(save_interval=(1 / 24.0), total_time=(2 / 24.0))

        assert jnp.all(jnp.isfinite(preds.dynamics.temperature))
        assert jnp.all(jnp.isfinite(preds.dynamics.specific_humidity))
        # Initial state should have seeded all six required tracers.
        assert set(model._final_dycore_state.tracers.keys()) >= {
            "specific_humidity", "qc", "qi", "qnc", "qni", "qr", "qs",
        }
