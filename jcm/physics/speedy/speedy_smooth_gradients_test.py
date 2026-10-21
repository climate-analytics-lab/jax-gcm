"""Gradient tests for SPEEDY's surrogate-gradient switches.

Every SPEEDY switch keeps the reference value at any width; the width only
gives its derivative that of a smooth function
(``jcm.physics.speedy.smoothing``). The primitive tests check that contract
for every helper (``check_surrogate_gradient``). The scheme-level tests pin
the two halves where each switch is formed: the scheme's output is
bit-identical at width 0 and at a positive width, and a gradient that is
exactly zero (or unbounded) under the reference derivative is finite and
nonzero with a positive width -- the humidity trigger (with its column
moisture budget), the vertical-diffusion onset, the condensation cap, the
stratiform clip and the drizzle corner of the cloud cover, and the dry-land
evaporation hinge. The convective precipitation onset is a single
``surrogate_pos`` and is covered by the primitive tests and, through the
trigger's budget test, by its tangent in a switching-on plume. If a value
test regresses the forward model has been changed; if a gradient test, the
switch has been re-hardened.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.speedy.smoothing import (
    smooth_clip01, smooth_gate, smooth_max, smooth_min, smooth_pos,
    surrogate_clip01, surrogate_gate, surrogate_max, surrogate_min,
    surrogate_pos, surrogate_sqrt,
)
from jcm.testing import check_surrogate_gradient

# NB: no jax_enable_x64 here: the flag is process-global and would
# repin the f32 reference-trajectory tolerances of the regression suite.


class TestSmoothingPrimitives:
    """Width-0 exactness and smooth-gradient survival for the helpers."""

    def test_width_zero_reproduces_hard_ops(self):
        x = jnp.linspace(-2.0, 3.0, 41)
        assert jnp.array_equal(smooth_pos(x, 0.0), jnp.maximum(x, 0.0))
        assert jnp.array_equal(smooth_gate(x, 0.5, 0.0), (x > 0.5).astype(x.dtype))
        assert jnp.array_equal(smooth_min(x, 1.0, 0.0), jnp.minimum(x, 1.0))
        assert jnp.array_equal(smooth_max(x, 1.0, 0.0), jnp.maximum(x, 1.0))
        assert jnp.array_equal(smooth_clip01(x, 0.0), jnp.clip(x, 0.0, 1.0))

    def test_width_zero_gradients_are_finite_at_corners(self):
        # The double-where guard must keep the width-0 branch free of NaN
        # cotangents exactly at the corner points.
        for fn, at in (
            (lambda v: smooth_pos(v, 0.0), 0.0),
            (lambda v: smooth_min(v, 1.0, 0.0), 1.0),
            (lambda v: smooth_max(v, 1.0, 0.0), 1.0),
            (lambda v: smooth_clip01(v, 0.0), 0.0),
            (lambda v: smooth_clip01(v, 0.0), 1.0),
        ):
            g = jax.grad(fn)(jnp.asarray(at))
            assert jnp.isfinite(g), f"NaN corner gradient in {fn} at {at}"

    def test_smooth_gradients_survive_the_clipped_side(self):
        w = 0.1
        for fn, at in (
            (lambda v: smooth_pos(v, w), -0.3),
            (lambda v: smooth_gate(v, 0.5, w) * 2.0, 0.2),
            (lambda v: smooth_min(v, 1.0, w), 1.3),
            (lambda v: smooth_clip01(v, w), 1.3),
        ):
            g = jax.grad(fn)(jnp.asarray(at))
            assert jnp.isfinite(g) and g != 0.0, (
                f"smooth gradient dead on the clipped side of {fn}"
            )


class TestSurrogatePrimitives:
    """Hard value, smooth derivative: the with_surrogate_gradient contract."""

    W = 0.1

    def test_value_is_hard_and_derivatives_are_the_smooth_functions(self):
        x = jnp.linspace(-1.5, 2.5, 33)
        w = self.W
        cases = (
            (surrogate_gate, smooth_gate, (x, jnp.asarray(0.5))),
            (surrogate_pos, smooth_pos, (x,)),
            (surrogate_min, smooth_min, (x, jnp.asarray(1.0))),
            (surrogate_max, smooth_max, (x, jnp.asarray(1.0))),
            (surrogate_clip01, smooth_clip01, (x,)),
        )
        for surrogate, smooth, operands in cases:
            check_surrogate_gradient(
                lambda *xs: surrogate(*xs, w),
                lambda *xs: smooth(*xs, 0.0),
                lambda *xs: smooth(*xs, w),
                operands,
            )

    def test_width_has_no_gradient_and_zero_width_is_the_reference(self):
        x = jnp.asarray(0.3)
        assert jax.grad(lambda w: surrogate_pos(x, w))(jnp.asarray(0.1)) == 0.0
        assert jax.grad(lambda v: surrogate_gate(v, 0.5, 0.0))(x) == 0.0
        assert jax.grad(lambda v: surrogate_pos(v, 0.0))(jnp.asarray(-0.3)) == 0.0

    def test_sqrt_value_is_floored_and_slope_is_bounded_at_the_corner(self):
        floor, offset = 1e-9, 0.04
        x = jnp.asarray([0.0, 1e-8, 1e-3, 4.0])
        np.testing.assert_array_equal(
            np.asarray(surrogate_sqrt(x, floor, offset)),
            np.asarray(jnp.sqrt(jnp.maximum(x, floor))),
        )
        slope = jax.vmap(jax.grad(lambda v: surrogate_sqrt(v, floor, offset)))(x)
        reference = jax.vmap(jax.grad(lambda v: surrogate_sqrt(v, floor, 0.0)))(x)
        assert jnp.all(jnp.isfinite(slope))
        assert float(slope.max()) <= 1.0 / (2.0 * np.sqrt(offset)) + 1e-6
        assert float(reference[1]) > 1e3  # the corner the surrogate removes
        # Far from the corner the slope is the reference's to O(offset/x).
        np.testing.assert_allclose(slope[-1], reference[-1], rtol=offset / 4.0)


def _convection_column(kx=8, rh_pbl_top=0.85):
    """Build a (kx, 1, 1) column engineered into the case-2 humidity trigger.

    Statically stable in saturation MSE aloft (ktop1 valid) but with dry
    intermediate levels (ktop2 invalid), so activation rides entirely on
    the boundary-layer RH criterion; ``rh_pbl_top`` sets how close the
    PBL-top humidity sits to the rhbl = 0.9 threshold.
    """
    from jcm.physics.speedy.speedy_coords import SpeedyCoords

    coords = SpeedyCoords.single_column_coords(num_levels=kx)
    # Saturation MSE aloft (~345 kJ/kg) sits BETWEEN the PBL-top actual
    # MSE (~338 kJ/kg) and the surface saturation MSE (~362 kJ/kg):
    # conditionally unstable (ktop1 valid) without actual-MSE instability
    # (ktop2 invalid), so activation rides on the RH trigger alone.
    se = jnp.full((kx, 1, 1), 320e3)
    se = se.at[-1].set(300e3)
    se = se.at[-2].set(304e3)
    qsat = jnp.full((kx, 1, 1), 10.0)
    qsat = qsat.at[-1].set(25.0)
    qsat = qsat.at[-2].set(15.0)
    qa = 0.3 * qsat
    # Surface layer above threshold; PBL-top layer set by rh_pbl_top.
    qa = qa.at[-1].set(0.95 * qsat[-1])
    qa = qa.at[-2].set(rh_pbl_top * qsat[-2])
    psa = jnp.ones((1, 1))
    return psa, se, qa, qsat, coords


def _diagnose(psa, se, qa, qsat, coords, trigger_smoothing):
    from jcm.physics.speedy.params import Parameters
    from jcm.physics.speedy.physics_data import PhysicsData
    from jcm.physics.convection.speedy_convection import diagnose_convection

    kx = se.shape[0]
    parameters = Parameters.default()
    parameters = dataclasses.replace(
        parameters,
        convection=dataclasses.replace(
            parameters.convection,
            trigger_smoothing=jnp.array(trigger_smoothing),
        ),
    )
    physics_data = PhysicsData.zeros((1, 1), kx, speedy_coords=coords)
    return diagnose_convection(psa, se, qa, qsat, parameters, physics_data)


class TestConvectionTriggerSurrogate:
    def test_trigger_value_is_the_hard_trigger_at_any_width(self):
        for rh in (0.80, 0.88, 0.895, 0.905, 0.99):
            psa, se, qa, qsat, coords = _convection_column(rh_pbl_top=rh)
            iptop0, qdif0 = _diagnose(psa, se, qa, qsat, coords, 0.0)
            iptop1, qdif1 = _diagnose(psa, se, qa, qsat, coords, 0.02)
            # Hard trigger: active iff both RH criteria exceed rhbl = 0.9.
            assert (float(qdif0[0, 0]) > 0.0) == (rh > 0.9)
            np.testing.assert_array_equal(np.asarray(qdif1), np.asarray(qdif0))
            np.testing.assert_array_equal(np.asarray(iptop1), np.asarray(iptop0))

    def test_surrogate_derivative_sees_the_trigger_jump(self):
        # On either side of the RH threshold the reference derivative of
        # qdif with respect to the PBL-top humidity is exactly zero: below
        # it qdif is 0, above it qdif is the SURFACE-layer excess. The jump
        # between them is what the surrogate derivative resolves.
        def qdif_of_dq(dq, width, rh):
            psa, se, qa, qsat, coords = _convection_column(rh_pbl_top=rh)
            qa = qa.at[-2].add(dq)
            _, qdif = _diagnose(psa, se, qa, qsat, coords, width)
            return qdif[0, 0]

        for rh in (0.89, 0.91):
            hard_grad = jax.grad(qdif_of_dq)(jnp.array(0.0), 0.0, rh)
            smooth_grad = jax.grad(qdif_of_dq)(jnp.array(0.0), 0.02, rh)
            assert hard_grad == 0.0
            assert jnp.isfinite(smooth_grad) and smooth_grad > 0.0

    def test_trigger_derivative_survives_when_both_rh_tests_fail(self):
        """A dry column (surface and PBL-top RH both below rhbl) still sees the trigger."""
        def qdif_of_dq(dq, width):
            psa, se, qa, qsat, coords = _convection_column(rh_pbl_top=0.85)
            qa = qa.at[-1].set(0.85 * qsat[-1])  # surface RH below rhbl = 0.9 too
            qa = qa.at[-1].add(dq)
            _, qdif = _diagnose(psa, se, qa, qsat, coords, width)
            return qdif[0, 0]

        assert float(qdif_of_dq(jnp.array(0.0), 0.0)) == 0.0
        assert float(qdif_of_dq(jnp.array(0.0), 0.05)) == 0.0
        assert float(jax.grad(qdif_of_dq)(jnp.array(0.0), 0.0)) == 0.0
        assert float(jax.grad(qdif_of_dq)(jnp.array(0.0), 0.05)) > 0.0

    @staticmethod
    def _column_budget(dq, rh, width, qsat_aloft_scale=1.0):
        """Column moisture tendency and convective rain on the trigger column."""
        import jcm.constants as c
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import HumidityData, PhysicsData
        from jcm.physics_interface import PhysicsState
        from jcm.physics.convection.speedy_convection import get_convection_tendencies

        psa, se, qa, qsat, coords = _convection_column(rh_pbl_top=rh)
        qsat = qsat.at[:-2].multiply(qsat_aloft_scale)
        qa = qa.at[-2].add(dq)
        kx = se.shape[0]
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters, convection=dataclasses.replace(
                parameters.convection, trigger_smoothing=jnp.array(width)))
        physics_data = PhysicsData.zeros(
            (1, 1), kx, speedy_coords=coords,
            humidity=HumidityData.zeros((1, 1), kx, qsat=qsat))
        state = PhysicsState.zeros(
            (kx, 1, 1), temperature=se / c.cpd, specific_humidity=qa,
            normalized_surface_pressure=psa)
        tend, out = get_convection_tendencies(state, physics_data, parameters)
        column_moisture = jnp.sum(tend.specific_humidity * coords.dhs[:, None, None])
        return column_moisture, out.convection.precnv[0, 0]

    def test_trigger_tangent_closes_the_column_moisture_budget(self):
        """Below the threshold the derivative is a whole plume switching on.

        Convection converts column moisture into rain at a fixed ratio (the
        values above the threshold show it). The trigger's surrogate
        derivative has to respect that ratio on both sides of the
        threshold: a moisture tangent without the matching rain tangent
        would be a sink the scheme does not have.
        """
        moisture, rain = self._column_budget(0.0, 0.95, 0.0)
        ratio = float(moisture / rain)
        for rh in (0.89, 0.91):
            (_, _), (d_moisture, d_rain) = jax.jvp(
                lambda d: self._column_budget(d, rh, 0.02), (jnp.array(0.0),), (jnp.array(1.0),))
            assert float(d_rain) > 0.0
            np.testing.assert_allclose(float(d_moisture), ratio * float(d_rain), rtol=1e-4)


class TestVdiffGateSmoothing:
    def _tendencies(self, rh_gate_smoothing, drh_scale):
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import (
            ConvectionData, HumidityData, PhysicsData,
        )
        from jcm.physics.speedy.speedy_coords import SpeedyCoords
        from jcm.physics_interface import PhysicsState
        from jcm.physics.vertical_diffusion.speedy_vdiff import (
            get_vertical_diffusion_tend,
        )
        from jcm.terrain import TerrainData

        kx, ix, il = 8, 1, 1
        coords = SpeedyCoords.single_column_coords(num_levels=kx)
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters,
            vertical_diffusion=dataclasses.replace(
                parameters.vertical_diffusion,
                rh_gate_smoothing=jnp.array(rh_gate_smoothing),
            ),
        )
        # Stable PBL (dmse < 0), RH contrast at the lowest interface just
        # BELOW the drh0 onset (drh0 = rhgrad * dsigma ~ 0.066), scaled by
        # drh_scale.
        se = jnp.linspace(340e3, 310e3, kx)[:, None, None] * jnp.ones((kx, ix, il))
        qsat = jnp.full((kx, ix, il), 10.0)
        rh = jnp.full((kx, ix, il), 0.5)
        rh = rh.at[-1].set(0.5 + drh_scale)
        qa = rh * qsat
        phi = jnp.linspace(150e3, 0.0, kx)[:, None, None] * jnp.ones((kx, ix, il))
        humidity = HumidityData.zeros((ix, il), kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(
            (ix, il), kx, iptop=jnp.full((ix, il), kx + 1, dtype=int), se=se
        )
        physics_data = PhysicsData.zeros(
            (ix, il), kx, humidity=humidity, convection=convection,
            speedy_coords=coords,
        )
        state = PhysicsState.zeros((kx, ix, il), specific_humidity=qa, geopotential=phi)
        tend, _ = get_vertical_diffusion_tend(
            state, physics_data, parameters, ForcingData.ones((ix, il)),
            TerrainData.single_column(),
        )
        return tend.specific_humidity

    def test_gate_value_is_the_hard_jump_at_any_width(self):
        # drh0 at the PBL interface is rhgrad * (fsg[-1] - fsg[-2]) ~ 0.0575.
        just_below, just_above = 0.050, 0.065
        hard_lo = self._tendencies(0.0, just_below)
        hard_hi = self._tendencies(0.0, just_above)
        # The hard gate switches a finite flux on: the tendency jumps,
        # whatever the width.
        assert float(jnp.abs(hard_lo).max()) == 0.0
        assert float(jnp.abs(hard_hi).max()) > 1e-7
        for drh in (just_below, just_above):
            np.testing.assert_array_equal(
                np.asarray(self._tendencies(0.02, drh)),
                np.asarray(self._tendencies(0.0, drh)))

    def test_surrogate_gate_derivative_sees_the_onset(self):
        just_below = 0.050

        def pbl_qtend(drh_scale, width):
            return self._tendencies(width, drh_scale)[-1, 0, 0]

        hard_grad = jax.grad(pbl_qtend)(jnp.asarray(just_below), 0.0)
        smooth_grad = jax.grad(pbl_qtend)(jnp.asarray(just_below), 0.02)
        assert hard_grad == 0.0
        assert jnp.isfinite(smooth_grad) and smooth_grad != 0.0


class TestChainedGates:
    def test_chain_has_the_product_value_and_the_first_failures_derivative(self):
        from jcm.physics.speedy.smoothing import chain_gates

        zero, w = jnp.asarray(0.0), jnp.asarray(0.2)

        def chain(x, y):
            return chain_gates(surrogate_gate(x, zero, w), surrogate_gate(y, zero, w))

        for x, y in ((0.3, 0.4), (0.3, -0.4), (-0.3, 0.4), (-0.3, -0.4)):
            assert float(chain(jnp.asarray(x), jnp.asarray(y))) == float((x > 0) * (y > 0))
        # Both fail: the plain product's gradient is zero, the chain's is the
        # first gate's.
        x, y = jnp.asarray(-0.3), jnp.asarray(-0.4)
        plain = jax.grad(lambda x_: surrogate_gate(x_, zero, w) * surrogate_gate(y, zero, w))(x)
        chained = jax.grad(chain, argnums=0)(x, y)
        assert float(plain) == 0.0
        np.testing.assert_allclose(
            float(chained), float(jax.grad(lambda x_: smooth_gate(x_, zero, w))(x)), rtol=1e-6)


class TestVdiffSeFluxOneSided:
    def test_stable_column_gets_no_reversed_heat_flux(self):
        """The shallow-convection SE flux stays one-sided at any width.

        gate * dmse would go negative below the threshold (a reversed
        heat flux in stable columns); the hinge, whose value is the hard
        maximum, keeps it >= 0 (Codex review, PR #567).
        """
        import dataclasses

        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import (
            ConvectionData, HumidityData, PhysicsData,
        )
        from jcm.physics.speedy.speedy_coords import SpeedyCoords
        from jcm.physics_interface import PhysicsState
        from jcm.physics.vertical_diffusion.speedy_vdiff import (
            get_vertical_diffusion_tend,
        )
        from jcm.terrain import TerrainData

        kx, ix, il = 8, 1, 1
        coords = SpeedyCoords.single_column_coords(num_levels=kx)
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters,
            vertical_diffusion=dataclasses.replace(
                parameters.vertical_diffusion,
                mse_gate_smoothing=jnp.array(2000.0),
            ),
        )
        # Stable PBL: dmse ~ -4 kJ/kg, within a few widths of threshold.
        se = jnp.linspace(340e3, 310e3, kx)[:, None, None] * jnp.ones((kx, ix, il))
        qsat = jnp.full((kx, ix, il), 10.0)
        rh = jnp.full((kx, ix, il), 0.5)
        qa = rh * qsat
        qa = qa.at[-1].set(qsat[-1] - 1.5)  # dmse = -4286 + 2501*(8.5-10) < 0
        phi = jnp.linspace(150e3, 0.0, kx)[:, None, None] * jnp.ones((kx, ix, il))
        humidity = HumidityData.zeros((ix, il), kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(
            (ix, il), kx, iptop=jnp.full((ix, il), kx + 1, dtype=int), se=se
        )
        physics_data = PhysicsData.zeros(
            (ix, il), kx, humidity=humidity, convection=convection,
            speedy_coords=coords,
        )
        state = PhysicsState.zeros((kx, ix, il), specific_humidity=qa, geopotential=phi)
        tend, _ = get_vertical_diffusion_tend(
            state, physics_data, parameters, ForcingData.ones((ix, il)),
            TerrainData.single_column(),
        )
        # The shallow-convection SE flux warms level kx-2 and cools the
        # surface layer; with the hinge it must not reverse.
        assert float(tend.temperature[-2, 0, 0]) >= 0.0
        assert float(tend.temperature[-1, 0, 0]) <= 0.0


class TestLscCapSmoothing:
    def _heating(self, cap_smoothing, rhlsc):
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import (
            ConvectionData, HumidityData, PhysicsData,
        )
        from jcm.physics.speedy.speedy_coords import SpeedyCoords
        from jcm.physics_interface import PhysicsState
        from jcm.physics.clouds.speedy_condensation import (
            get_large_scale_condensation_tendencies,
        )
        from jcm.terrain import TerrainData

        kx, ix, il = 8, 1, 1
        coords = SpeedyCoords.single_column_coords(num_levels=kx)
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters,
            condensation=dataclasses.replace(
                parameters.condensation,
                cap_smoothing=jnp.array(cap_smoothing),
                rhlsc=rhlsc,
            ),
        )
        qsat = jnp.full((kx, ix, il), 10.0)
        qa = 5.0 * qsat  # wildly supersaturated: the heating cap engages
        humidity = HumidityData.zeros((ix, il), kx, qsat=qsat)
        convection = ConvectionData.zeros(
            (ix, il), kx, iptop=jnp.full((ix, il), kx + 1, dtype=int)
        )
        physics_data = PhysicsData.zeros(
            (ix, il), kx, humidity=humidity, convection=convection,
            speedy_coords=coords,
        )
        state = PhysicsState.zeros((kx, ix, il), specific_humidity=qa)
        state = state.copy(normalized_surface_pressure=jnp.ones((ix, il)))
        tend, _ = get_large_scale_condensation_tendencies(
            state, physics_data, parameters, ForcingData.ones((ix, il)),
            TerrainData.single_column(),
        )
        # Level kx-2: the bottom level's rhref is dominated by the
        # rhblsc floor, which would zero d/d(rhlsc) structurally.
        return tend.temperature[-2, 0, 0]

    def test_capped_heating_gradient_survives_with_smoothing(self):
        hard_grad = jax.grad(self._heating, argnums=1)(0.0, jnp.array(0.9))
        smooth_grad = jax.grad(self._heating, argnums=1)(0.05, jnp.array(0.9))
        assert hard_grad == 0.0, "cap not engaged: test is vacuous"
        assert jnp.isfinite(smooth_grad) and smooth_grad != 0.0

    def test_capped_heating_is_the_hard_cap_at_any_width(self):
        hard = float(self._heating(0.0, jnp.array(0.9)))
        assert float(self._heating(0.05, jnp.array(0.9))) == hard


class TestCoverSmoothing:
    def _clstr(self, cover_smoothing, gse_s1):
        """Stratiform cover on a column whose stability saturates fstab."""
        return self._clouds(cover_smoothing, gse_s1).cloudstr[0, 0]

    def _clouds(self, cover_smoothing, gse_s1, precnv=0.0):
        """Run the cloud diagnosis on that column, with a convective rain rate."""
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import (
            CondensationData, ConvectionData, HumidityData, PhysicsData,
        )
        from jcm.physics.speedy.speedy_coords import SpeedyCoords
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.radiation.speedy_shortwave import clouds
        from jcm.terrain import TerrainData

        kx, ix, il = 8, 1, 1
        coords = SpeedyCoords.single_column_coords(num_levels=kx)
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters,
            shortwave_radiation=dataclasses.replace(
                parameters.shortwave_radiation,
                cover_smoothing=jnp.array(cover_smoothing),
                gse_s1=gse_s1,
            ),
        )
        # Stable column with gse ~ 0.47, just past gse_s1 = 0.40: the hard
        # fstab clip saturates at 1 (zero gradient) while the smooth tail
        # is still well within float range. Dry air keeps the RH cover at
        # zero.
        phi = jnp.linspace(60e3, 0.0, kx)[:, None, None] * jnp.ones((kx, ix, il))
        se = 300e3 + 0.47 * phi
        qsat = jnp.full((kx, ix, il), 10.0)
        rh = jnp.full((kx, ix, il), 0.2)
        humidity = HumidityData.zeros((ix, il), kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(
            (ix, il), kx, iptop=jnp.full((ix, il), kx + 1, dtype=int), se=se,
            precnv=jnp.full((ix, il), precnv),
        )
        condensation = CondensationData.zeros((ix, il), kx)
        physics_data = PhysicsData.zeros(
            (ix, il), kx, humidity=humidity, convection=convection,
            condensation=condensation, speedy_coords=coords,
        )
        state = PhysicsState.zeros(
            (kx, ix, il), specific_humidity=rh * qsat, geopotential=phi
        )
        operand = (
            state, physics_data, parameters,
            ForcingData.ones((ix, il)), TerrainData.single_column(),
            PhysicsTendency.zeros(shape=(kx, ix, il)),
        )
        _, pd, *_ = clouds(operand)
        return pd.shortwave_rad

    def test_saturated_fstab_gradient_survives_with_smoothing(self):
        hard_grad = jax.grad(self._clstr, argnums=1)(0.0, jnp.array(0.40))
        smooth_grad = jax.grad(self._clstr, argnums=1)(0.05, jnp.array(0.40))
        assert hard_grad == 0.0, "fstab not saturated: test is vacuous"
        assert jnp.isfinite(smooth_grad) and smooth_grad != 0.0

    def test_drizzle_cover_slope_is_bounded_with_smoothing(self):
        """The sqrt(precipitation) corner: unbounded reference slope, bounded surrogate's.

        A precipitation rate of 1e-8 g/(m^2 s) (~1e-6 mm/day) sits on the
        corner; the reference d(cover)/d(precnv) there is hundreds of times
        the surrogate's bound wpcl * 86.4 / (2 * w * pmaxcl).
        """
        from jcm.physics.speedy.params import ShortwaveRadiationParameters

        sw = ShortwaveRadiationParameters.default()
        width = 0.05

        def cover(precnv, w):
            return self._clouds(w, jnp.array(0.40), precnv).cloudc[0, 0]

        rain = jnp.array(1e-8)
        assert float(cover(rain, width)) == float(cover(rain, 0.0))
        reference = float(jax.grad(cover)(rain, 0.0))
        surrogate = float(jax.grad(cover)(rain, width))
        bound = float(sw.wpcl * 86.4 / (2.0 * width * sw.pmaxcl))
        assert reference > 100.0 * bound
        assert 0.0 < surrogate <= bound * (1 + 1e-5)

    def test_cover_is_the_hard_cover_at_any_width(self):
        # The sqrt-corner and every clip keep their reference values: the
        # width changes no cover, only its derivatives.
        hard = float(self._clstr(0.0, jnp.array(0.40)))
        assert np.isfinite(hard)
        assert float(self._clstr(0.05, jnp.array(0.40))) == hard


class TestSurfaceEvapSmoothing:
    def _dry_land_evap(self, evap_smoothing, soilw=0.7):
        """Land evaporation on a column whose evap hinge is firmly closed."""
        import dataclasses

        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import (
            ConvectionData, HumidityData, LWRadiationData, PhysicsData,
            SurfaceFluxData, SWRadiationData,
        )
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        from jcm.physics_interface import PhysicsState
        from jcm.physics.surface.speedy_surface_flux import get_surface_fluxes
        from jcm.terrain import TerrainData

        kx, ix, il = 8, 64, 32
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)
        xy, zxy = (ix, il), (kx, ix, il)
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters,
            surface_flux=dataclasses.replace(
                parameters.surface_flux,
                evap_smoothing=jnp.array(evap_smoothing),
            ),
        )
        # Mildly dry land: soilw*qsat(tskin) - q1 sits a few tenths of a
        # g/kg below the hinge across the grid (tskin varies with
        # latitude), so the hard hinge gives exactly zero evaporation
        # while the softplus tail is small but ALIVE. That is the regime
        # where an inconsistent hard evap > 0 mask in the energy balance
        # hands the dry tail the full latent sensitivity; columns far
        # below the hinge cannot discriminate because the tail
        # underflows to zero and the mask never engages.
        qa = jnp.full(zxy, 10.5)
        state = PhysicsState.zeros(
            zxy, jnp.ones(zxy), jnp.ones(zxy), jnp.full(zxy, 300.0), qa,
            jnp.ones(zxy) * (jnp.arange(kx))[::-1][:, None, None], jnp.ones(xy),
        )
        terrain = TerrainData.from_coords(
            coords, orography=jnp.zeros(xy), fmask=jnp.ones(xy), lfluxland=True
        )
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        physics_data = PhysicsData.zeros(
            xy, kx,
            convection=ConvectionData.zeros(xy, kx),
            humidity=HumidityData.zeros(xy, kx, rh=jnp.full(zxy, 0.9)),
            surface_flux=SurfaceFluxData.zeros(xy, rlds=jnp.full(xy, 400.0)),
            shortwave_rad=SWRadiationData.zeros(xy, kx, rsds=jnp.full(xy, 400.0)),
            longwave_rad=LWRadiationData.zeros(xy, kx),
            speedy_coords=speedy_c,
        )
        forcing = ForcingData.ones(
            xy,
            sea_surface_temperature=jnp.full(xy, 292.0),
            soilw_am=jnp.full(xy, 1.0) * soilw,
            stl_am=jnp.full(xy, 288.0),
        )
        _, pd = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        # fmask = 1 everywhere, so the published grid mean is the land value.
        return jnp.max(jnp.abs(pd.surface_flux.evap))

    def test_dry_column_evaporation_is_the_hard_hinge_at_any_width(self):
        # The hinge and the energy balance's activity weight keep their hard
        # values, so a dry column evaporates exactly nothing at any width
        # (no softplus tail, no latent correction from the skin balance).
        hard = float(self._dry_land_evap(0.0))
        assert hard == 0.0, "hinge not closed: test is vacuous"
        assert float(self._dry_land_evap(0.1)) == 0.0

    def test_dry_column_evaporation_derivative_opens_with_smoothing(self):
        # The column sits a few tenths of a g/kg below the hinge: the
        # reference evaporation has zero derivative in the soil moisture,
        # the surrogate a positive one.
        def total_evap(soilw, width):
            return self._dry_land_evap(width, soilw)

        assert float(jax.grad(total_evap)(jnp.array(0.7), 0.0)) == 0.0
        assert float(jax.grad(total_evap)(jnp.array(0.7), 0.1)) > 0.0


class TestLscBoundaryThresholdSurrogate:
    """``max(rhref[-1], rhblsc)``: the bottom level's threshold selection."""

    def _bottom_drying(self, bl_threshold_smoothing, rhlsc, rhblsc):
        """Bottom-level drying tendency of a saturated column."""
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import (
            ConvectionData, HumidityData, PhysicsData,
        )
        from jcm.physics.speedy.speedy_coords import SpeedyCoords
        from jcm.physics_interface import PhysicsState
        from jcm.physics.clouds.speedy_condensation import (
            get_large_scale_condensation_tendencies,
        )
        from jcm.terrain import TerrainData

        kx, ix, il = 8, 1, 1
        coords = SpeedyCoords.single_column_coords(num_levels=kx)
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters,
            condensation=dataclasses.replace(
                parameters.condensation,
                bl_threshold_smoothing=jnp.array(bl_threshold_smoothing),
                rhlsc=rhlsc, rhblsc=rhblsc,
            ),
        )
        qsat = jnp.full((kx, ix, il), 10.0)
        qa = qsat  # saturated: the bottom level condenses at either threshold
        humidity = HumidityData.zeros((ix, il), kx, qsat=qsat)
        convection = ConvectionData.zeros(
            (ix, il), kx, iptop=jnp.full((ix, il), kx + 1, dtype=int)
        )
        physics_data = PhysicsData.zeros(
            (ix, il), kx, humidity=humidity, convection=convection,
            speedy_coords=coords,
        )
        state = PhysicsState.zeros((kx, ix, il), specific_humidity=qa)
        state = state.copy(normalized_surface_pressure=jnp.ones((ix, il)))
        tend, _ = get_large_scale_condensation_tendencies(
            state, physics_data, parameters, ForcingData.ones((ix, il)),
            TerrainData.single_column(),
        )
        return tend.specific_humidity[-1, 0, 0]

    # On the 8-level column the bottom layer sits at sigma 0.95, so the
    # profile's own bottom-level threshold is rhlsc - 0.1 * (1 - 0.95^2) =
    # rhlsc - 0.00975: rhblsc = 0.85 loses to rhlsc = 0.9, rhblsc = 0.95 wins.
    LOSES = (jnp.array(0.9), jnp.array(0.85))
    WINS = (jnp.array(0.9), jnp.array(0.95))

    def test_value_is_the_hard_selection_at_any_width(self):
        rtlsc = 1.0 / (4.0 * 3600.0)
        for (rhlsc, rhblsc), threshold in ((self.LOSES, 0.9 - 0.00975),
                                           (self.WINS, 0.95)):
            hard = float(self._bottom_drying(0.0, rhlsc, rhblsc))
            np.testing.assert_allclose(hard, (threshold - 1.0) * 10.0 * rtlsc, rtol=1e-5)
            assert float(self._bottom_drying(0.03, rhlsc, rhblsc)) == hard

    def test_losing_parameter_regains_a_derivative(self):
        d_rhblsc = jax.grad(self._bottom_drying, argnums=2)
        d_rhlsc = jax.grad(self._bottom_drying, argnums=1)
        # rhblsc below the profile value has no effect on the model: the
        # reference derivative is exactly zero. The surrogate's is positive
        # and decays with the 0.04 gap -- about a tenth of the winning slope
        # at width 0.03 -- so an optimiser that stepped rhblsc below the
        # threshold is still pulled back toward it.
        assert float(d_rhblsc(0.0, *self.LOSES)) == 0.0
        winning = float(d_rhblsc(0.0, *self.WINS))
        assert winning > 0.0
        losing = float(d_rhblsc(0.03, *self.LOSES))
        assert 0.0 < losing < 0.2 * winning
        # The mirror image: rhlsc has no bottom-level derivative while
        # rhblsc wins the selection.
        assert float(d_rhlsc(0.0, *self.WINS)) == 0.0
        assert float(d_rhlsc(0.03, *self.WINS)) > 0.0


class TestStabilityClipSurrogate:
    """The +/- dtheta clip of the surface-to-air excess in the stability factor."""

    def _factor(self, excess, dtheta, width):
        from jcm.physics.speedy.params import SurfaceFluxParameters
        from jcm.physics.surface.speedy_surface_flux import _stability_factor

        sfp = dataclasses.replace(
            SurfaceFluxParameters.default(), dtheta=dtheta,
            stability_smoothing=jnp.array(width))
        t2 = jnp.asarray(290.0)
        return _stability_factor(t2 + excess, t2, sfp)

    def test_value_is_the_hard_clip_at_any_width(self):
        fstab, dtheta = 0.67, jnp.asarray(3.0)
        # (excess, clipped dth): the stable side is damped by astab = 0.5
        # before the clip, so -8 K clips at -3 and -3 K sits at -1.5.
        for excess, dth in ((-8.0, -3.0), (-3.0, -1.5), (-1.0, -0.5),
                            (0.5, 0.5), (2.0, 2.0), (5.0, 3.0)):
            hard = float(self._factor(jnp.asarray(excess), dtheta, 0.0))
            np.testing.assert_allclose(hard, 1.0 + dth * fstab / 3.0, rtol=1e-6)
            assert float(self._factor(jnp.asarray(excess), dtheta, 0.5)) == hard

    def test_clipped_columns_regain_a_dtheta_derivative(self):
        d_dtheta = jax.grad(self._factor, argnums=1)
        for excess in (5.0, -8.0):  # clipped on the unstable and the stable side
            hard = float(d_dtheta(jnp.asarray(excess), jnp.asarray(3.0), 0.0))
            smooth = float(d_dtheta(jnp.asarray(excess), jnp.asarray(3.0), 0.5))
            # The factor is 1 +/- fstab whatever dtheta is: the reference
            # derivative is the round-off of fstab/dtheta - dtheta*fstab/dtheta^2.
            np.testing.assert_allclose(hard, 0.0, atol=1e-6)
            assert np.isfinite(smooth) and smooth != 0.0

    def test_unclipped_columns_keep_the_reference_derivative(self):
        d_dtheta = jax.grad(self._factor, argnums=1)
        excess = jnp.asarray(1.0)  # four widths inside the +dtheta clip
        hard = float(d_dtheta(excess, jnp.asarray(3.0), 0.0))
        smooth = float(d_dtheta(excess, jnp.asarray(3.0), 0.5))
        assert hard != 0.0
        np.testing.assert_allclose(smooth, hard, rtol=0.1)

    def test_block_is_a_surrogate(self):
        from jcm.physics.surface.speedy_surface_flux import (
            _clipped_excess, _surrogate_clipped_excess,
        )

        w = jnp.asarray(0.5)
        operands = (jnp.asarray([-8.0, -2.0, 0.5, 5.0]), jnp.asarray(3.0),
                    jnp.asarray(0.5))
        check_surrogate_gradient(
            lambda *xs: _surrogate_clipped_excess(*xs, w),
            lambda *xs: _clipped_excess(*xs, 0.0),
            lambda *xs: _clipped_excess(*xs, w),
            operands,
        )


class TestConvectionPressureSwitchSurrogate:
    """``psa > psmin`` and the clipped ramp on the cloud-base mass flux."""

    @staticmethod
    def _run(psa_value, psmin_smoothing, psmin=0.8):
        """Convection on the RH-trigger column at a given surface pressure."""
        import jcm.constants as c
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.physics_data import HumidityData, PhysicsData
        from jcm.physics_interface import PhysicsState
        from jcm.physics.convection.speedy_convection import get_convection_tendencies

        psa, se, qa, qsat, coords = _convection_column(rh_pbl_top=0.95)
        psa = psa * psa_value
        kx = se.shape[0]
        parameters = Parameters.default()
        parameters = dataclasses.replace(
            parameters, convection=dataclasses.replace(
                parameters.convection, psmin=jnp.asarray(psmin),
                psmin_smoothing=jnp.array(psmin_smoothing)))
        physics_data = PhysicsData.zeros(
            (1, 1), kx, speedy_coords=coords,
            humidity=HumidityData.zeros((1, 1), kx, qsat=qsat))
        state = PhysicsState.zeros(
            (kx, 1, 1), temperature=se / c.cpd, specific_humidity=qa,
            normalized_surface_pressure=psa)
        tend, out = get_convection_tendencies(state, physics_data, parameters)
        column_moisture = jnp.sum(tend.specific_humidity * coords.dhs[:, None, None])
        return tend, out.convection, column_moisture

    def test_value_is_the_hard_switch_at_any_width(self):
        kx = 8
        for psa_value in (0.75, 0.85, 0.95):
            tend0, conv0, _ = self._run(jnp.asarray(psa_value), 0.0)
            tend1, conv1, _ = self._run(jnp.asarray(psa_value), 0.02)
            for a, b in zip(jax.tree.leaves((tend0, conv0)),
                            jax.tree.leaves((tend1, conv1))):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            below = psa_value < 0.8
            assert (int(conv0.iptop[0, 0]) == kx + 1) == below
            assert (float(conv0.cbmf[0, 0]) == 0.0) == below
            assert (float(conv0.qdif[0, 0]) == 0.0) == below
            if below:
                assert float(jnp.abs(tend0.specific_humidity).max()) == 0.0
                assert float(jnp.abs(tend0.temperature).max()) == 0.0

    def _rain(self, psmin, psa_value, width):
        _, conv, _ = self._run(jnp.asarray(psa_value), width, psmin=psmin)
        return conv.precnv[0, 0]

    def test_sub_threshold_column_sees_the_switch(self):
        # 0.05 below psmin: 2.5 widths. Raising psmin pushes the switch
        # further away, so the surrogate derivative is negative.
        d_psmin = jax.grad(self._rain)
        assert float(d_psmin(jnp.asarray(0.8), 0.75, 0.0)) == 0.0
        smooth = float(d_psmin(jnp.asarray(0.8), 0.75, 0.02))
        assert np.isfinite(smooth) and smooth < 0.0

    def test_capped_ramp_regains_a_derivative(self):
        # psa = 0.95 is past the cap (1 + psmin)/2 = 0.9: the mass flux is
        # psa * 1 whatever psmin is.
        d_psmin = jax.grad(self._rain)
        assert float(d_psmin(jnp.asarray(0.8), 0.95, 0.0)) == 0.0
        smooth = float(d_psmin(jnp.asarray(0.8), 0.95, 0.02))
        assert np.isfinite(smooth) and smooth < 0.0

    def test_on_ramp_derivative_is_the_reference(self):
        # psa = 0.85 is mid-ramp, five widths from both the switch and the cap.
        d_psmin = jax.grad(self._rain)
        hard = float(d_psmin(jnp.asarray(0.8), 0.85, 0.0))
        smooth = float(d_psmin(jnp.asarray(0.8), 0.85, 0.01))
        assert hard < 0.0
        np.testing.assert_allclose(smooth, hard, rtol=0.05)

    def test_switch_tangent_closes_the_column_moisture_budget(self):
        """Below the switch the derivative is a whole plume switching on.

        As for the humidity trigger: convection converts column moisture
        into rain at a fixed ratio, and the switch's surrogate derivative
        has to respect it, or the tangent would be a moisture sink the
        scheme does not have.
        """
        def budget(psmin, psa_value, width):
            _, conv, moisture = self._run(jnp.asarray(psa_value), width, psmin=psmin)
            return moisture, conv.precnv[0, 0]

        # The ratio carries the 1/psa of the flux-to-tendency conversion, so
        # the reference is taken at the SAME surface pressure, made active by
        # a lower psmin (0.75 is mid-ramp for psmin = 0.6).
        moisture, rain = budget(jnp.asarray(0.6), 0.75, 0.0)
        ratio = float(moisture / rain)
        (_, _), (d_moisture, d_rain) = jax.jvp(
            lambda p: budget(p, 0.75, 0.02), (jnp.asarray(0.8),), (jnp.asarray(1.0),))
        assert float(d_rain) < 0.0
        np.testing.assert_allclose(float(d_moisture), ratio * float(d_rain), rtol=1e-4)
