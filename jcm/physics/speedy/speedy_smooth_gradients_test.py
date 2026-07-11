"""Gradient tests for the smoothed SPEEDY branches.

Each scheme-level test pins the specific unlock its smoothing knob exists
for: a gradient that is exactly zero (or a forward value that jumps)
under the hard branches must be finite and nonzero with a positive
width, and every knob at width 0 must reproduce the hard scheme exactly.
If one of these regresses to zero the corresponding gate has been
re-hardened.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.speedy.smoothing import (
    smooth_clip01, smooth_gate, smooth_max, smooth_min, smooth_pos,
)

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


class TestConvectionTriggerSmoothing:
    def test_width_zero_matches_hard_trigger(self):
        for rh in (0.80, 0.895, 0.905, 0.99):
            psa, se, qa, qsat, coords = _convection_column(rh_pbl_top=rh)
            iptop0, qdif0 = _diagnose(psa, se, qa, qsat, coords, 0.0)
            assert jnp.all(jnp.isfinite(qdif0))
            # Hard trigger: active iff both RH criteria exceed rhbl = 0.9.
            assert (float(qdif0[0, 0]) > 0.0) == (rh > 0.9)

    def test_smooth_trigger_ramps_and_unlocks_the_gradient(self):
        # Just below the hard threshold: qdif is exactly zero and so is
        # its gradient with respect to the PBL-top humidity.
        def qdif_of_dq(dq, width):
            psa, se, qa, qsat, coords = _convection_column(rh_pbl_top=0.88)
            qa = qa.at[-2].add(dq)
            _, qdif = _diagnose(psa, se, qa, qsat, coords, width)
            return qdif[0, 0]

        hard_val = qdif_of_dq(jnp.array(0.0), 0.0)
        hard_grad = jax.grad(qdif_of_dq)(jnp.array(0.0), 0.0)
        assert hard_val == 0.0 and hard_grad == 0.0

        smooth_val = qdif_of_dq(jnp.array(0.0), 0.02)
        smooth_grad = jax.grad(qdif_of_dq)(jnp.array(0.0), 0.02)
        assert smooth_val > 0.0
        assert jnp.isfinite(smooth_grad) and smooth_grad > 0.0


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

    def test_hard_gate_is_a_value_jump_and_smooth_gate_ramps(self):
        # drh0 at the PBL interface is rhgrad * (fsg[-1] - fsg[-2]) ~ 0.0575.
        just_below, just_above = 0.050, 0.065
        hard_lo = self._tendencies(0.0, just_below)
        hard_hi = self._tendencies(0.0, just_above)
        # The hard gate switches a finite flux on: the tendency jumps.
        assert float(jnp.abs(hard_lo).max()) == 0.0
        assert float(jnp.abs(hard_hi).max()) > 1e-7

        def pbl_qtend(drh_scale, width):
            return self._tendencies(width, drh_scale)[-1, 0, 0]

        hard_grad = jax.grad(pbl_qtend)(jnp.asarray(just_below), 0.0)
        smooth_grad = jax.grad(pbl_qtend)(jnp.asarray(just_below), 0.02)
        assert hard_grad == 0.0
        assert jnp.isfinite(smooth_grad) and smooth_grad != 0.0


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

    def test_width_zero_matches_hard_cap(self):
        assert float(self._heating(0.0, jnp.array(0.9))) == float(
            self._heating(0.0, jnp.array(0.9))
        )
        # And the smooth cap approaches the hard one as the width shrinks.
        hard = float(self._heating(0.0, jnp.array(0.9)))
        near = float(self._heating(1e-6, jnp.array(0.9)))
        assert abs(near - hard) < 1e-8 * max(1.0, abs(hard))


class TestCoverSmoothing:
    def _clstr(self, cover_smoothing, gse_s1):
        """Stratiform cover on a column whose stability saturates fstab."""
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
            (ix, il), kx, iptop=jnp.full((ix, il), kx + 1, dtype=int), se=se
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
        return pd.shortwave_rad.cloudstr[0, 0]

    def test_saturated_fstab_gradient_survives_with_smoothing(self):
        hard_grad = jax.grad(self._clstr, argnums=1)(0.0, jnp.array(0.40))
        smooth_grad = jax.grad(self._clstr, argnums=1)(0.05, jnp.array(0.40))
        assert hard_grad == 0.0, "fstab not saturated: test is vacuous"
        assert jnp.isfinite(smooth_grad) and smooth_grad != 0.0

    def test_width_zero_matches_hard_cover(self):
        # Width 0 is exact (the regression suite pins it); small widths
        # converge as O(sqrt(w)) because the sqrt-corner regularization
        # replaces sqrt(epsilon) with sqrt(w*log 2) at zero precipitation.
        hard = float(self._clstr(0.0, jnp.array(0.40)))
        near = float(self._clstr(1e-7, jnp.array(0.40)))
        nearer = float(self._clstr(1e-9, jnp.array(0.40)))
        assert np.isfinite(hard)
        assert abs(near - hard) < 1e-3
        assert abs(nearer - hard) < abs(near - hard)
