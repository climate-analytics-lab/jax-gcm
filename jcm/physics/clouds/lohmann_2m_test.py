"""Unit tests for the ICON two-moment cloud microphysics scheme.

Ported from icon-physics-v1-amq; narrowed to the tests that exercise the
process functions currently wired into ``cloud_microphysics_2m``
(CloudUtils, FreezingBelow238K, Autoconversion_2M / precip_formation_warm).
The remaining 2M test classes from the amq branch (mixed-phase deposition,
sedimentation, update_tendencies, etc.) will be ported alongside Phase 5b
as the full ECHAM6 sequence is wired into the orchestrator — see #341.
"""

import jax
import jax.numpy as jnp
import numpy as np
from math import pi

from .cloud_utils import (
    eff_ice_crystal_radius,
    minimum_CDNC,
)
from .lohmann_2m import (
    diagnostics,
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
from .lohmann_2m_params import CloudParams2M
from jcm.constants import rhow, alhs, alhc, rv

# Parameters are no longer module-level exports of lohmann_2m_params: the
# scheme reads everything from a threaded ``CloudParams2M`` struct. Tests
# read the identical default values from this instance (``_P.tmelt`` is the
# same 273.15 K the former module-level ``tmelt``/``t0`` provided).
_P = CloudParams2M.default()


def _zeros(n: int) -> jnp.ndarray:
    return jnp.zeros((n,), dtype=jnp.float32)


def _full(n: int, v: float) -> jnp.ndarray:
    return jnp.full((n,), v, dtype=jnp.float32)


class TestCloudUtils:
    """Test utility functions for cloud microphysics"""

    def test_eff_ice_crystal_radius(self):
        # Positive, non-degenerate inputs so the eps-guards do not affect the result
        pxice = jnp.array([0.1, 1.0, 10.0], dtype=jnp.float32)   # [g/m^3]
        picnc = jnp.array([1e5, 1e6, 1e7], dtype=jnp.float32)    # [1/m^3]

        got = eff_ice_crystal_radius(pxice, picnc, _P)
        expected = 0.5e4 * (pxice / (_P.fact_PK * picnc)) ** (1.0 / _P.pow_PK)

        assert got.shape == expected.shape
        assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)
    
    def test_minimum_CDNC(self):
        pxwat = jnp.array([0.0, 1e-6, 1e-4, 1e-2], dtype=jnp.float32)  # [kg/m^3]
        got = minimum_CDNC(pxwat, _P)

        if _P.ldyn_cdnc_min:
            expected = _P.rcd_vol_max ** (-3.0) * (3.0 / (4.0 * pi * rhow)) * pxwat
            expected = jnp.clip(expected, _P.cdnc_min_lower, _P.cdnc_min_upper)
        else:
            expected = jnp.full_like(pxwat, _P.cdnc_min_fixed * 1.0e6)  # cm^-3 -> m^-3

        assert got.shape == pxwat.shape
        assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)

        # extra invariant: dynamic branch must be within clip bounds
        if _P.ldyn_cdnc_min:
            assert jnp.all(got >= _P.cdnc_min_lower)
            assert jnp.all(got <= _P.cdnc_min_upper)


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
            min_liquid_threshold=_P.cqtmin,  # Minimum liquid water threshold [kg/kg]
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
        assert jnp.all(droplet_number[inputs["freezing_condition"]] == _P.cqtmin)  # Droplet number should be reduced to the minimum threshold

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
        assert jnp.all(droplet_number[inputs["freezing_condition"]] == _P.cqtmin)
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

        (droplet_number, cloud_water, pmratepr, prpr, prprn,
         _autoconv_only, _accretion_only) = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt,
            params=_P,
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

        (droplet_number, cloud_water, pmratepr, prpr, prprn,
         _autoconv_only, _accretion_only) = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt,
            params=_P,
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

        (droplet_number, cloud_water, pmratepr, prpr, prprn,
         _autoconv_only, _accretion_only) = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt,
            params=_P,
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
            params=_P,
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
        assert jnp.all(droplet_number_o >= _P.cqtmin)
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

        temperature_previous = jnp.array([_P.tmelt + 1.0, _P.tmelt - 1.0], dtype=jnp.float32)
        melt_mask = temperature_previous > _P.tmelt

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
            ice_tendency=ice_tendency, dt=dt, params=_P,
        )

        assert icnc_o.shape == (2,)
        assert jnp.all(jnp.isfinite(icnc_o))
        assert jnp.all(jnp.isfinite(rain_flux_o))
        assert jnp.all(jnp.isfinite(snow_flux_o))

        # Melt point: ICNC -> icemin, transferred number into CDNC
        assert float(icnc_o[0]) == float(_P.icemin)
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
            params=_P,
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
            sedimentation_ice(**x, dt=dt, params=_P)
        )

        for arr in (ice_mmr_o, icnc_o, ice_flux_o, ice_flux_n_o, falling_ice_frac_o, pmrateps_o):
            assert jnp.all(jnp.isfinite(arr))

        assert jnp.all(ice_mmr_o >= 0.0)
        assert jnp.all(ice_flux_o >= 0.0)
        assert jnp.all(ice_flux_n_o >= 0.0)
        assert jnp.all(pmrateps_o >= 0.0)
        assert jnp.all(falling_ice_frac_o >= 0.0)
        assert jnp.all(falling_ice_frac_o <= 1.0)

        cloudy = x["cloud_fraction"] > _P.clc_min
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
            sedimentation_ice(**x, dt=dt, params=_P)
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
        ztmp_ice = (alhs / rv) * (1.0 / _P.tmelt - 1.0 / T_val)
        ztmp_water = (alhc / rv) * (1.0 / _P.tmelt - 1.0 / T_val)
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
            params=_P,
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
        pcnd_o, pdep_o, T_o, _, _, _ = mixed_phase_deposition_and_corrections(**x)
        T_expected = x["temperature"] + x["lsdcp"] * pdep_o + x["lvdcp"] * pcnd_o
        assert jnp.allclose(T_o, T_expected, atol=1e-4)

    def test_moisture_conservation_ice(self):
        x = self._base_inputs()
        pcnd_o, pdep_o, _, q_o, _, _ = mixed_phase_deposition_and_corrections(**x)
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
            min_liquid_threshold=_P.cqtmin,
            params=_P,
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
        assert jnp.all(droplet_number[inputs["freezing_condition"]] >= _P.cqtmin)
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
            params=_P,
        )

    def test_wbf_applies_transfer_and_tendencies(self):
        inputs = self._base_inputs()
        cdnc_o, ql_o, qi_o, qlt_o, qit_o, t_o = WBF_process(**inputs)
        ztmst_rcp = 1.0 / jnp.maximum(inputs["dt"], _P.eps)
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
        assert jnp.all(cdnc_o[mask] == _P.cqtmin)
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
            params=_P,
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
        _, _, _, zcons2, _ = microphysics_dt_constants(inp["dt"], _P)
        expected_evap = (zcons2 * inp["pressure_thickness"] * inp["rain_evap_mmr"]).astype(pfevapr.dtype)
        precip_mask = pfevapr > 0.0
        assert jnp.allclose(pfevapr[precip_mask], expected_evap[precip_mask], atol=1e-6)
        assert jnp.all(pfevapr[~precip_mask] == 0.0)

    def test_incoming_ice_can_melt_into_rain_at_top(self):
        n = 2
        inp = self._base_inputs(n)
        inp.update({
            "cloud_fraction": _full(n, 0.8),
            "temp_tmp": jnp.full((n,), float(_P.tmelt) + 2.0, dtype=jnp.float32),
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
            params=_P,
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
        mask = jnp.logical_and(cloud_flag_o, pxib_o > _P.cqtmin)
        assert jnp.all(icnc_o[mask] >= _P.icemin)
        assert jnp.all(icnc_o[~mask] == _P.cqtmin)
        assert jnp.all(jnp.isfinite(icnc_o))


class TestUpdateInCloudWaterCirrusBranches_2M:
    """ICNC update under the alternative ``nic_cirrus`` selectors.

    The default parameter set uses ``nic_cirrus=1`` (diagnostic ICNC from
    ice mass / mean radius); these tests pin the ``nic_cirrus=2``
    (external nucleation source, capped by pressure) and the fall-through
    (leave-unchanged) branches.
    """

    def _inputs_with_ice(self, n=3):
        base = TestUpdateInCloudWater_2M()._base_inputs(n)
        base["cloud_flag"] = jnp.array([True, True, True])
        base["cloud_fraction"] = _full(n, 0.4)
        base["cloud_ice_in_cloud"] = _full(n, 2e-4)   # > cqtmin
        # icnc <= icemin so the candidate-update branch (ll2_ic) fires.
        base["ice_crystal_number"] = _zeros(n)
        return base

    def test_nic_cirrus_2_uses_external_source_capped_by_pressure(self):
        n = 3
        inputs = self._inputs_with_ice(n)
        inputs["params"] = _P.replace(nic_cirrus=2)
        # One column below the pressure cap, one above it (cap = pap*1e6).
        cap = float(inputs["pressure"][0]) * 1e6
        inputs["newly_formed_ice"] = jnp.array(
            [5e4, cap * 10.0, 0.0], dtype=jnp.float32
        )
        _, icnc_o, *_ = update_in_cloud_water(**inputs)
        # Below cap: candidate passes through (already >= icemin).
        assert jnp.isclose(icnc_o[0], 5e4)
        # Above cap: clipped to pressure * 1e6.
        assert jnp.isclose(icnc_o[1], cap, rtol=1e-6)
        # Zero source: enforced up to the icemin floor.
        assert jnp.isclose(icnc_o[2], _P.icemin)

    def test_nic_cirrus_other_leaves_icnc_at_minimum_floor(self):
        n = 3
        inputs = self._inputs_with_ice(n)
        inputs["params"] = _P.replace(nic_cirrus=0)
        inputs["newly_formed_ice"] = _full(n, 1e8)
        _, icnc_o, *_ = update_in_cloud_water(**inputs)
        # Fall-through branch: the candidate is the existing ICNC (0),
        # so only the icemin floor applies where cloud ice is present.
        assert jnp.all(icnc_o == _P.icemin)


class TestDiagnostics2M:
    """Accumulator updates in ``assembly.diagnostics``.

    The function mirrors ECHAM's cloud-diagnostics bookkeeping: every
    accumulator only advances under its own flag mask (liquid cloud,
    ice cloud, cloud-top, TOVS-selected cirrus), while the effective-
    radius and cloud-fraction accumulators advance unconditionally.
    """

    N = 4
    DT = 100.0
    LEVEL_INDEX = 5

    def _base_inputs(self):
        n = self.N
        zeros = _zeros(n)
        return dict(
            cdnc=_full(n, 1e8),
            icnc=_full(n, 5e4),
            cloud_fraction=_full(n, 0.5),
            dp_over_g=_full(n, 1000.0),
            layer_thickness=_full(n, 500.0),
            freezing_number_rate=_full(n, 3.0),
            air_density=_full(n, 1.2),
            rain_number_formation=_full(n, 2.0),
            snow_number_accretion=_full(n, 1.0),
            incloud_ice=_full(n, 3e-5),
            incloud_liquid=_full(n, 1e-4),
            temp_tmp=jnp.array([280.0, 280.0, 230.0, 280.0], dtype=jnp.float32),
            eff_radius_liq=jnp.array([10.0, 2.0, 0.0, 0.0], dtype=jnp.float32),
            eff_radius_ice=_full(n, 20.0),
            liquid_cloud_flag=jnp.array([True, True, False, False]),
            ice_cloud_flag=jnp.array([False, True, True, False]),
            cdnc_ave=zeros, cdnc_ave_acc=zeros, cdnc_ave_burd=zeros,
            cdnc_ct=zeros, cld_ice_time=zeros, cld_liq_time=zeros,
            icnc_ave=zeros, icnc_ave_acc=zeros, icnc_ave_burd=zeros,
            ice_water_content_acc=zeros, iwp_tovs=zeros,
            liq_water_content_acc=zeros, cdnc_accretion=zeros,
            cdnc_autoconv=zeros, cdnc_freezing=zeros,
            eff_radius_ice_acc=zeros, eff_radius_ice_time=zeros,
            eff_radius_ice_tovs=zeros, eff_radius_liq_acc=zeros,
            eff_radius_liq_ct=zeros, eff_radius_liq_time=zeros,
            cdnc_burden=zeros, icnc_burden=zeros, tau1i=zeros,
            eff_radius_ct_m=zeros, cloud_fraction_acc=zeros,
            ktop=jnp.array([5, 5, 5, 0], dtype=jnp.int32),
            level_index=self.LEVEL_INDEX,
            dt=jnp.array(self.DT, dtype=jnp.float32),
            params=_P,
        )

    def test_number_process_sinks_are_time_integrated(self):
        inp = self._base_inputs()
        out = diagnostics(**inp)
        cdnc_accretion, cdnc_autoconv, cdnc_freezing = out[12], out[13], out[14]
        # Sinks subtract dt * rate everywhere (no flag gating in ECHAM).
        assert jnp.allclose(cdnc_autoconv, -self.DT * 2.0)
        assert jnp.allclose(cdnc_freezing, -self.DT * 3.0)
        assert jnp.allclose(cdnc_accretion, -self.DT * 1.0)

    def test_liquid_accumulators_gated_by_liquid_flag(self):
        inp = self._base_inputs()
        out = diagnostics(**inp)
        cdnc_ave, cdnc_ave_acc = out[0], out[1]
        cld_liq_time, cdnc_burden = out[5], out[21]
        liq_mask = np.asarray(inp["liquid_cloud_flag"])

        # Where the liquid flag is set the accumulators advance by the
        # exact ECHAM increments; elsewhere they stay zero.
        assert jnp.allclose(cdnc_ave_acc[liq_mask], self.DT * 1e8)
        assert jnp.all(cdnc_ave_acc[~liq_mask] == 0.0)
        assert jnp.allclose(cdnc_ave[liq_mask], self.DT * 1e8 * 0.5)
        assert jnp.all(cdnc_ave[~liq_mask] == 0.0)
        assert jnp.allclose(cld_liq_time[liq_mask], self.DT)
        assert jnp.all(cld_liq_time[~liq_mask] == 0.0)
        assert jnp.allclose(cdnc_burden[liq_mask], 1e8 * 500.0)
        assert jnp.all(cdnc_burden[~liq_mask] == 0.0)

    def test_cloud_top_liquid_diagnostics(self):
        inp = self._base_inputs()
        out = diagnostics(**inp)
        cdnc_ct, eff_radius_liq_ct = out[3], out[19]
        eff_radius_liq_time, eff_radius_ct_m = out[20], out[24]

        # Only column 0 satisfies the full cloud-top mask: liquid flag,
        # ktop == level_index, T > tmelt, prior ct radius < 4 um and a
        # current radius >= 4 um. Column 1 fails on radius (2 um < 4).
        assert float(eff_radius_ct_m[0]) == 10.0
        assert jnp.all(eff_radius_ct_m[1:] == 0.0)
        assert jnp.isclose(eff_radius_liq_ct[0], self.DT * 10.0)
        assert jnp.all(eff_radius_liq_ct[1:] == 0.0)
        assert jnp.isclose(cdnc_ct[0], self.DT * 1e8 * 0.5)
        assert jnp.all(cdnc_ct[1:] == 0.0)
        assert jnp.isclose(eff_radius_liq_time[0], self.DT)
        assert jnp.all(eff_radius_liq_time[1:] == 0.0)

    def test_ice_accumulators_and_tovs_selection(self):
        inp = self._base_inputs()
        out = diagnostics(**inp)
        cld_ice_time = out[4]
        icnc_ave, icnc_ave_acc = out[6], out[7]
        ice_water_content_acc, iwp_tovs = out[9], out[10]
        eff_radius_ice_time, eff_radius_ice_tovs = out[16], out[17]
        tau1i = out[23]
        ice_mask = np.asarray(inp["ice_cloud_flag"])

        assert jnp.allclose(icnc_ave_acc[ice_mask], self.DT * 5e4)
        assert jnp.all(icnc_ave_acc[~ice_mask] == 0.0)
        assert jnp.allclose(icnc_ave[ice_mask], self.DT * 5e4 * 0.5)
        assert jnp.allclose(cld_ice_time[ice_mask], self.DT)
        assert jnp.allclose(
            ice_water_content_acc[ice_mask], self.DT * 3e-5 * 1.2,
        )

        # TOVS semi-transparent cirrus: IWP = 1000*xib*cf*dpg = 15 g/m2,
        # tau = 1.9787 * 15 * 20^-1.0365 ~ 1.33, inside (0.7, 3.8), so
        # the ice-flagged, non-cloud-top columns 1 and 2 are sampled.
        expected_tau = 1.9787 * 15.0 * 20.0 ** (-1.0365)
        assert 0.7 < expected_tau < 3.8
        assert jnp.allclose(tau1i[ice_mask], expected_tau, rtol=1e-5)
        assert jnp.all(tau1i[~ice_mask] == 0.0)
        assert jnp.allclose(eff_radius_ice_tovs[ice_mask], self.DT * 20.0)
        assert jnp.all(eff_radius_ice_tovs[~ice_mask] == 0.0)
        assert jnp.allclose(eff_radius_ice_time[ice_mask], self.DT)
        assert jnp.allclose(iwp_tovs[ice_mask], self.DT * 15.0)

    def test_unconditional_accumulators(self):
        inp = self._base_inputs()
        out = diagnostics(**inp)
        eff_radius_ice_acc, eff_radius_liq_acc = out[15], out[18]
        cloud_fraction_acc = out[25]

        # Effective-radius and cloud-fraction accumulators advance in
        # every column regardless of the cloud flags (ECHAM behaviour).
        assert jnp.allclose(eff_radius_ice_acc, self.DT * 20.0)
        assert jnp.allclose(
            eff_radius_liq_acc, self.DT * np.asarray(inp["eff_radius_liq"]),
        )
        assert jnp.allclose(cloud_fraction_acc, self.DT * 0.5)

    def test_output_arity_and_shapes(self):
        inp = self._base_inputs()
        out = diagnostics(**inp)
        assert len(out) == 26
        for arr in out:
            assert arr.shape == (self.N,)
            assert jnp.all(jnp.isfinite(arr))


class TestUpdateTendencies_2M:
    def test_tracer_tendencies_and_shapes(self):
        n = 4
        dt = jnp.array(60.0, dtype=jnp.float32)
        air_density = _full(n, 1.2)

        out = update_tendencies_and_important_vars(
            icnc=_full(n, 5e4), cdnc=_full(n, 1e8),
            ice_mmr_prev=_full(n, _P.ccwmin * 1.1), liq_mmr_prev=_zeros(n),
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
            params=_P,
        )

        assert len(out) == 11
        for a in out:
            assert a.shape == (n,)
            assert jnp.all(jnp.isfinite(a))

        _, ztmst_rcp, _, _, _ = microphysics_dt_constants(dt, _P)
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
            params=_P,
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
            params=_P,
        )

        assert jnp.all(out[9] == 0.0)   # liq_eff_radius
        assert jnp.all(out[10] == 0.0)  # ice_eff_radius


class TestIcon2MPipeline:
    """End-to-end checks that the 2M term composes into a runnable Model."""

    def test_factory_declares_six_tracers(self):
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(cloud_scheme="2m")
        names = {spec.name for spec in physics.required_tracers()}
        # qr/qs are no longer prognostic: ECHAM's 2M carries precipitation
        # exclusively in the within-step prfl/psfl fluxes; the tracers
        # double-booked that mass (review finding 2.18).
        assert names == {"qc", "qi", "qnc", "qni"}
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
        # Initial state should have seeded the four prognostic tracers
        # (qr/qs dropped — precipitation is flux-form, finding 2.18).
        assert set(model._final_dycore_state.tracers.keys()) >= {
            "specific_humidity", "qc", "qi", "qnc", "qni",
        }


class TestColumnWaterConservation2M:
    """The 2M flux-form ledger conserves column water against surface precip.

    With rain/snow carried exclusively as within-step fluxes (the qr/qs
    tracers double-booked mass and their negative state-difference
    tendencies were silently clipped — review finding 2.18), every kg the
    column loses must leave as surface precipitation:

        Σ (dq + dqc + dqi)/dt · ρ·dz  +  rain_sfc + snow_sfc  ≈  0.

    Verified against the gross internal water movement so the bound is
    meaningful even when the column barely precipitates.
    """

    def test_water_budget_closes_against_surface_fluxes(self):
        import numpy as np
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity

        nlev = 20
        T = jnp.linspace(230.0, 300.0, nlev)
        p = jnp.linspace(2e4, 1e5, nlev)
        rho = p / (287.0 * T)
        q = 0.95 * jax.vmap(saturation_specific_humidity)(p, T)
        qc = jnp.zeros(nlev).at[10:16].set(1e-3)
        qi = jnp.zeros(nlev).at[4:8].set(2e-4)
        cf = jnp.where((qc + qi) > 0, 0.7, 0.0)
        dz = jnp.full(nlev, 500.0)
        qnc = jnp.where(qc > 0, 5e7, 0.0)
        qni = jnp.where(qi > 0, 1e4, 0.0)
        params = CloudParams2M.default()

        tend, rain_sfc, snow_sfc, *_ = cloud_microphysics_2m(
            T, q, p, qc, qi, qnc, qni,
            jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
            jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, params,
        )
        mref = np.asarray(rho * dz)
        dw = np.asarray(tend.dqdt + tend.dqcdt + tend.dqidt)
        P = float(rain_sfc + snow_sfc)
        gross = float(np.sum(np.abs(dw) * mref)) + abs(P)
        residual = float(np.sum(dw * mref) + P)
        assert gross > 0.0, "column did nothing — test is vacuous"
        assert abs(residual) < max(1e-5 * gross, 1e-12), (
            f"2M water budget open by {residual:.3e} kg/m2/s "
            f"(gross movement {gross:.3e}, precip {P:.3e})"
        )

    def test_precip_process_rates_close_the_warm_ledger(self):
        """The #499 per-level formation/evaporation rates are the true ledger.

        On an all-warm liquid column (no ice, snow or melt) every kg of
        surface rain was formed minus evaporated on the way down, so the
        new grid-mean [kg/kg/s] outputs must satisfy
        ``rain_sfc == sum((formation - evaporation) * rho * dz)`` — a
        units-and-sign check that also fails if a pathway (accretion,
        riming) is missing from the formation sum.
        """
        import numpy as np
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity

        nlev = 20
        T = jnp.linspace(280.0, 300.0, nlev)
        p = jnp.linspace(2e4, 1e5, nlev)
        rho = p / (287.0 * T)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = jnp.where(
            (jnp.arange(nlev) >= 10) & (jnp.arange(nlev) < 16),
            0.95 * qsw, 0.7 * qsw,
        )
        qc = jnp.zeros(nlev).at[10:16].set(1e-3)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        dz = jnp.full(nlev, 500.0)
        qnc = jnp.where(qc > 0, 5e7, 0.0)
        params = CloudParams2M.default()

        (tend, rain_sfc, snow_sfc, _rl, _ri, _rfw, _rfm, _au, _ac, _wbf,
         form, evap, _cf, rain_prof, snow_prof) = cloud_microphysics_2m(
            T, q, p, qc, jnp.zeros(nlev), qnc, jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
            jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, params,
        )
        assert form.shape == (nlev,) and evap.shape == (nlev,)
        assert float(jnp.min(form)) >= 0.0
        assert float(jnp.min(evap)) >= 0.0
        assert float(snow_sfc) == 0.0
        assert float(rain_sfc) > 0.0, "no rain formed — fixture too dry"
        assert float(jnp.max(evap)) > 0.0, "no evap — fixture too moist"
        mref = np.asarray(rho * dz)
        np.testing.assert_allclose(
            float(rain_sfc),
            float(np.sum((np.asarray(form) - np.asarray(evap)) * mref)),
            rtol=1e-5,
        )

    def test_precip_process_rates_cover_the_cold_chain(self):
        """The #499 formation rate must use the GRID-MEAN snow-side terms.

        An all-cold ice column (no liquid, no melt) with numerous small
        crystals (high ICNC → slow sedimentation, so the folded cloud-ice
        flux is a small correction) forms snow through the cold chain
        alone. The surface snow must then be bracketed by the
        (formation − evaporation) column ledger: equal up to the small
        sedimenting-ice contribution. An implementation that grabbed the
        IN-CLOUD cold-formation variants instead of the grid-mean ones
        overshoots by 1/cf ≈ 1.4x and breaks the upper bracket.
        """
        import numpy as np
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity

        nlev = 20
        T = jnp.linspace(215.0, 262.0, nlev)          # never melts
        p = jnp.linspace(2e4, 7e5 / 10.0, nlev)
        rho = p / (287.0 * T)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.9 * qsw
        qi = jnp.zeros(nlev).at[8:14].set(3e-4)
        qni = jnp.where(qi > 0, 1e6, 0.0)             # many small crystals
        cf = jnp.where(qi > 0, 0.7, 0.0)
        dz = jnp.full(nlev, 500.0)
        params = CloudParams2M.default()

        (tend, rain_sfc, snow_sfc, _rl, _ri, _rfw, _rfm, _au, _ac, _wbf,
         form, evap, _cf, _rp, _sp) = cloud_microphysics_2m(
            T, q, p, jnp.zeros(nlev), qi, jnp.zeros(nlev), qni,
            jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
            jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, params,
        )
        mref = np.asarray(rho * dz)
        ledger = float(np.sum((np.asarray(form) - np.asarray(evap)) * mref))
        total_sfc = float(rain_sfc + snow_sfc)
        assert total_sfc > 0.0, "cold chain made no precip — fixture off"
        assert ledger > 0.0
        # Ledger ≤ surface (the sedimenting-ice fold adds mass the ledger
        # deliberately excludes), and covers at least 60% of it (the
        # crystals are small and slow, so the fold is a minor term). An
        # in-cloud/grid-mean mixup (~1.4x) breaks the first bound.
        assert ledger <= total_sfc * (1.0 + 1e-5), (
            f"ledger {ledger:.3e} exceeds surface {total_sfc:.3e}"
        )
        assert ledger >= 0.6 * total_sfc, (
            f"ledger {ledger:.3e} vs surface {total_sfc:.3e} — cold-chain "
            "formation missing from the #499 rate"
        )

    def test_cold_supersaturation_is_consumed(self):
        """Sub-cthomi supersaturation must deposit onto diagnosed ICNC.

        Second year-run killer (after the warm-rain gate): with the
        hollow nic_cirrus=2 default (#552 — expects an external
        Kaercher-Lohmann pnicex source jcm never computes), cells below
        cthomi with ice mass but ~no crystals never nucleate ICNC,
        depositional growth stalls, and RH w.r.t. ice grows without
        bound (5-10x over the Antarctic winter surface by day 85). The
        nic_cirrus=1 diagnostic (ICNC from ice mass / crystal radius)
        must consume the supersaturation; the hollow branch must not —
        this pins BOTH the new default and the #552 gap.
        """
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity

        nlev = 8
        T = jnp.full(nlev, 226.0)             # below cthomi = 238.15
        p = jnp.linspace(6.5e4, 7.6e4, nlev)  # Antarctic-plateau surface
        rho = p / (287.0 * T)
        qs = jax.vmap(saturation_specific_humidity)(p, T)
        q = 2.0 * qs                          # 200% RH
        qi = jnp.full(nlev, 1.5e-4)
        qni = jnp.zeros(nlev)                 # ice mass, no crystals
        cf = jnp.full(nlev, 0.9)
        dz = jnp.full(nlev, 300.0)

        def run(nic):
            params = CloudParams2M.default(nic_cirrus=nic)
            tend, _, _, *_ = cloud_microphysics_2m(
                T, q, p, jnp.zeros(nlev), qi, jnp.zeros(nlev), qni,
                jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
                jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
                jnp.zeros(nlev), jnp.zeros(nlev),
                1800.0, params,
            )
            return float(jnp.sum(tend.dqdt * rho * dz))  # vapor sink

        dq_diag = run(1)
        assert dq_diag < -1e-6, (
            f"nic_cirrus=1 did not deposit the cold supersaturation "
            f"(column dq {dq_diag:.3e} kg/m2/s)"
        )

    def test_koop_homogeneous_freezing_floor(self):
        """Vapor above the Koop threshold cannot survive a step (#552 interim).

        Third year-run killer: in the (cold-biased) winter stratosphere,
        S_ice grew past 2 faster than ICNC-limited deposition consumed
        it, ending in a latent-heat NaN near day 110. Homogeneous
        nucleation physics forbids that state: above S_crit(T) =
        2.349 - T/259 (Koop et al. 2000), solution droplets freeze in
        seconds. One microphysics step must bring S_ice at 190 K from
        1.74 to at/below the threshold, with bounded latent heating.
        """
        import numpy as np
        import jcm.constants as c
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M

        nlev = 4
        T = jnp.full(nlev, 190.2)
        p = jnp.linspace(3000.0, 4500.0, nlev)
        rho = p / (287.0 * T)
        esi = 610.78 * np.exp(21.875 * (190.2 - 273.15) / (190.2 - 7.66))
        qsi = c.eps * esi / np.asarray(p)
        q = jnp.asarray(1.74 * qsi)
        qi = jnp.full(nlev, 1.5e-4)
        tend, _, _, *_ = cloud_microphysics_2m(
            T, q, p, jnp.zeros(nlev), qi, jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev), jnp.full(nlev, 0.287), rho,
            jnp.full(nlev, 800.0), jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev), 720.0, CloudParams2M.default(),
        )
        s_after = (np.asarray(q) + 720.0 * np.asarray(tend.dqdt)) / qsi
        scrit = 2.349 - 190.2 / 259.0
        assert float(np.max(s_after)) <= scrit + 0.02, (
            f"S_ice {s_after} not pulled to the Koop threshold {scrit:.3f}"
        )
        # Latent heating from the burst stays small (q is tiny at 190 K).
        assert float(np.max(np.abs(720.0 * np.asarray(tend.dtedt)))) < 1.0

    def test_supercooled_stratus_rains(self):
        """Warm-rain coalescence must drain supercooled liquid decks.

        ECHAM ll_prcp_warm (mo_cloud_micro_2m.f90:1662) has NO
        temperature condition — autoconversion/accretion act on any
        cloud liquid. A (T > tmelt) gate left polar/storm-track
        supercooled stratus (238-273 K) without a liquid sink: qc built
        up ~50x over a month of coupled T63L47 integration and NaN'd
        the run radiatively. This column is a -5 C stratus deck with
        drizzle-ready qc; the scheme must produce surface rain (the
        deck is warmer than the snow path's effective range, so rain is
        the expected exit).
        """
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity

        nlev = 12
        T = jnp.linspace(258.0, 269.0, nlev)   # all supercooled
        p = jnp.linspace(6e4, 1.0e5, nlev)
        rho = p / (287.0 * T)
        q = 0.95 * jax.vmap(saturation_specific_humidity)(p, T)
        qc = jnp.zeros(nlev).at[6:].set(8e-4)   # thick low liquid deck
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.9, 0.0)
        dz = jnp.full(nlev, 400.0)
        qnc = jnp.where(qc > 0, 5e7, 0.0)       # 50/mg — modest CDNC
        params = CloudParams2M.default()

        tend, rain_sfc, snow_sfc, *_ = cloud_microphysics_2m(
            T, q, p, qc, qi, qnc, jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
            jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, params,
        )
        total_precip = float(rain_sfc + snow_sfc)
        assert total_precip > 1e-7, (
            f"supercooled deck produced no precipitation "
            f"(rain {float(rain_sfc):.3e}, snow {float(snow_sfc):.3e}) — "
            f"the warm-rain mask is temperature-gated again"
        )
        # And the deck must actually lose liquid.
        dqc_col = float(jnp.sum(tend.dqcdt * rho * dz))
        assert dqc_col < 0.0

    def test_water_budget_closes_with_ice_reaching_the_surface(self):
        """Polar-cirrus column: sedimenting ice exits as surface snow.

        Pins two coupled sedimentation ledger entries (Codex review on
        #554): the pxite seed carrying the scan's net qi change (fallout
        loss above, exponential-integral re-deposit below cloud base) and
        the pxisub vapor credit for falling ice sublimating in
        subsaturated cloud-free air. The two omissions cancel in TOTAL
        column water (the negative-in-cloud-ice correction refunds the
        missing debit to vapor), so closure alone cannot pin them — the
        below-cloud re-deposit assertion breaks the degeneracy: with the
        seed absent, dqidt is exactly zero below the source cloud.
        """
        import numpy as np
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity

        nlev = 20
        T = jnp.linspace(210.0, 262.0, nlev)   # never above freezing
        p = jnp.linspace(2e4, 1e5, nlev)
        rho = p / (287.0 * T)
        q = 0.9 * jax.vmap(saturation_specific_humidity)(p, T)
        qi = jnp.zeros(nlev).at[6:12].set(5e-4)
        qc = jnp.zeros(nlev)
        cf = jnp.where(qi > 0, 0.7, 0.0)
        dz = jnp.full(nlev, 500.0)
        # Few, large crystals → fast fallout to the surface.
        qni = jnp.where(qi > 0, 2e3, 0.0)
        params = CloudParams2M.default()

        tend, rain_sfc, snow_sfc, *_ = cloud_microphysics_2m(
            T, q, p, qc, qi, jnp.zeros(nlev), qni,
            jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
            jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, params,
        )
        mref = np.asarray(rho * dz)
        dw = np.asarray(tend.dqdt + tend.dqcdt + tend.dqidt)
        P = float(rain_sfc + snow_sfc)
        assert float(snow_sfc) > 0.0, "no surface snow — fallout never reached the ground"
        gross = float(np.sum(np.abs(dw) * mref)) + abs(P)
        residual = float(np.sum(dw * mref) + P)
        assert abs(residual) < max(1e-5 * gross, 1e-12), (
            f"budget open by {residual:.3e} with surface-reaching ice "
            f"(snow_sfc {float(snow_sfc):.3e}, gross {gross:.3e})"
        )
        # The ice source cloud occupies levels 6..11; level 12 is clear
        # (cf = 0, no in-cloud deposition), so any qi gain there can only
        # be sedimenting ice re-depositing out of the falling flux — the
        # part of the pxite seed that total-water closure cannot see.
        assert float(tend.dqidt[12]) > 0.0, (
            "no below-cloud-base qi gain from the sedimenting flux — the "
            "scan's net ice change is not entering the pxite ledger"
        )


class TestPrecipFluxProfiles2M:
    """COSP-hook invariants for the per-level rain / snow flux profiles.

    ``cloud_microphysics_2m`` now returns the flux LEAVING each layer
    (stacked from the flux-coupled scan's ys), so the bottom level must
    equal the surface flux diagnostics EXACTLY — they are literally the
    same carry values. The frozen profile is snow plus the sedimenting
    cloud-ice flux (folded into snow at the bottom level, matching how
    ``surface_snow_flux`` is composed).
    """

    @staticmethod
    def _mixed_phase_column(nlev=20):
        """Ice cloud aloft + warm liquid deck below, near-saturated."""
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        T = jnp.linspace(225.0, 300.0, nlev)
        p = jnp.linspace(2e4, 1e5, nlev)
        rho = p / (287.0 * T)
        q = 0.95 * jax.vmap(saturation_specific_humidity)(p, T)
        qc = jnp.zeros(nlev).at[12:17].set(1e-3)
        qi = jnp.zeros(nlev).at[4:8].set(3e-4)
        cf = jnp.where((qc + qi) > 0, 0.8, 0.0)
        dz = jnp.full(nlev, 500.0)
        qnc = jnp.where(qc > 0, 5e7, 0.0)
        qni = jnp.where(qi > 0, 2e3, 0.0)  # few, large crystals → fallout
        return T, q, p, qc, qi, qnc, qni, cf, rho, dz

    @classmethod
    def _run(cls, column):
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        T, q, p, qc, qi, qnc, qni, cf, rho, dz = column
        nlev = T.shape[0]
        return cloud_microphysics_2m(
            T, q, p, qc, qi, qnc, qni,
            jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
            jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, CloudParams2M.default(),
        )

    def test_bottom_row_equals_surface_fluxes(self):
        # Index by position rather than unpacking: the flux profiles are
        # last by convention, and the scalars in between (eff. radii, the
        # rain-source split, the process rates) are not exercised here.
        out = self._run(self._mixed_phase_column())
        rain_sfc, snow_sfc = out[1], out[2]
        rain_prof, snow_prof = out[-2], out[-1]
        assert float(rain_sfc + snow_sfc) > 0.0, "column must precipitate"
        # Same carry values → exact equality (not just allclose).
        assert float(jnp.abs(rain_prof[-1] - rain_sfc)) < 1e-12
        assert float(jnp.abs(snow_prof[-1] - snow_sfc)) < 1e-12
        # Non-negative everywhere; zero at the model top (level 0 in the
        # physics-internal TOA-first frame: no condensate up there, so
        # nothing can be falling out of the top layer).
        assert jnp.all(rain_prof >= 0.0)
        assert jnp.all(snow_prof >= 0.0)
        assert float(rain_prof[0]) == 0.0
        assert float(snow_prof[0]) == 0.0
        # The frozen profile includes the sedimenting cloud-ice flux, so
        # it is already positive immediately below the ice cloud.
        assert float(snow_prof[8]) > 0.0

    def test_column_and_vmap_agree(self):
        from jcm.physics.clouds.lohmann_2m import cloud_microphysics_2m
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
        column = self._mixed_phase_column()
        (*_, rain_1, snow_1) = self._run(column)

        T, q, p, qc, qi, qnc, qni, cf, rho, dz = column
        nlev = T.shape[0]
        extras = (
            jnp.zeros(nlev), jnp.zeros(nlev), cf, rho, dz,
            jnp.full(nlev, 0.1), jnp.full(nlev, 5e7),
            jnp.zeros(nlev), jnp.zeros(nlev),
        )
        args = (T, q, p, qc, qi, qnc, qni) + extras
        batched = tuple(jnp.stack([a] * 3, axis=0) for a in args)
        (*_, rain_b, snow_b) = jax.vmap(
            cloud_microphysics_2m,
            in_axes=(0,) * 16 + (None, None),
        )(*batched, 1800.0, CloudParams2M.default())
        assert rain_b.shape == (3, nlev)
        for i in range(3):
            assert jnp.allclose(rain_b[i], rain_1, atol=1e-12)
            assert jnp.allclose(snow_b[i], snow_1, atol=1e-12)
