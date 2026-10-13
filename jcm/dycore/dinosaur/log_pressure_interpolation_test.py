"""Tests for the semi-Lagrangian vertical interpolation coordinate (#1060).

The transport tests drive the SL primitive's own ``semi_lagrangian_transport``
with prescribed departure points — no horizontal motion and a uniform upward
velocity, the regime of the summer mesosphere — so they measure exactly the
vertical interpolation the dycore applies, on the ECHAM L47 grid whose top
levels are spaced geometrically.
"""

import unittest

import numpy as np

#: Upwelling used by the transport tests, as a log-pressure displacement per
#: step: w = 2 cm/s over a 7 km scale height for a 12-minute step.
_DLNP = 0.02 * 720.0 / 7000.0


def _l47_dycore(**sl_options):
    from jcm.dycore.dinosaur.dycore import DinosaurDycore
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.terrain import TerrainData
    from jcm.utils import get_coords

    coords = get_coords(get_echam_levels(47), spectral_truncation=21)
    return DinosaurDycore(
        coords=coords, terrain=TerrainData.aquaplanet(coords),
        dt_seconds=720.0, advection="semi_lagrangian",
        sl_options=sl_options,
    )


def _trajectory_nodes(dycore):
    """Return the SL primitive's trajectory nodes (hybrid ``s`` or sigma ``σ``)."""
    primitive = dycore.primitive
    if hasattr(primitive, "_reference_vertical_nodes"):
        return primitive._reference_vertical_nodes
    return primitive._vertical_nodes


def _upwelling_departure(dycore, dlnp):
    """Departure points displaced down by ``dlnp`` in ln s, horizontally fixed."""
    import jax.numpy as jnp
    from dinosaur import primitive_equations, semi_lagrangian

    grid = dycore.coords.horizontal
    nodes = _trajectory_nodes(dycore)
    nlev = len(nodes.centers)
    lon, sin_lat = grid.nodal_mesh
    shape = (nlev,) + grid.nodal_shape
    arrival = semi_lagrangian.lon_lat_to_cartesian(
        jnp.broadcast_to(lon, shape), jnp.broadcast_to(sin_lat, shape))
    centers = jnp.asarray(nodes.centers)[:, None, None]
    sigma = jnp.clip(centers * jnp.exp(dlnp), nodes.centers[0],
                     nodes.centers[-1]) * jnp.ones(shape)
    full = semi_lagrangian.DeparturePoints(cartesian=arrival, sigma=sigma)
    horizontal = semi_lagrangian.DeparturePoints(
        cartesian=semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat)[:, None])
    return primitive_equations.PrimitiveDeparturePoints(
        full=full, horizontal=horizontal)


def _state_with_temperature_profile(dycore, profile):
    """Build a resting state whose T′ is ``profile`` (per level), horizontally uniform."""
    import jax.numpy as jnp

    grid = dycore.coords.horizontal
    state = dycore.initial_state(None, random_seed=0)
    t = jnp.asarray(profile, dtype=jnp.float32)[:, None, None] * jnp.ones(
        (len(profile),) + grid.nodal_shape)
    return state.replace(
        vorticity=jnp.zeros_like(state.vorticity),
        divergence=jnp.zeros_like(state.divergence),
        temperature_variation=grid.to_modal(t),
    )


def _transported_temperature(dycore, profile, steps, dlnp=_DLNP):
    """Column-mean T′ after ``steps`` SL transports along a fixed upwelling."""
    import jax

    departure = _upwelling_departure(dycore, dlnp)
    state = _state_with_temperature_profile(dycore, profile)
    transport = dycore.primitive.semi_lagrangian_transport

    @jax.jit
    def run(s):
        return jax.lax.fori_loop(0, steps, lambda _, x: transport(x, departure), s)

    out = dycore.coords.horizontal.to_nodal(run(state).temperature_variation)
    return np.asarray(out).mean(axis=(1, 2))


class DefaultsTest(unittest.TestCase):
    """Hybrid grids interpolate in log-pressure; sigma grids keep sigma."""

    def test_hybrid_default_is_log_pressure(self):
        from jcm.dycore.dinosaur.log_pressure_interpolation import (
            LogPressureSemiLagrangianHybrid)

        dycore = _l47_dycore()
        self.assertEqual(dycore.sl_vertical_coordinate, "log_pressure")
        self.assertIsInstance(dycore.primitive, LogPressureSemiLagrangianHybrid)

    def test_sigma_override_restores_dinosaurs_native_rule(self):
        from dinosaur import primitive_equations

        from jcm.dycore.dinosaur.log_pressure_interpolation import (
            LogPressureSemiLagrangianHybrid)

        dycore = _l47_dycore(vertical_coordinate="sigma")
        self.assertEqual(dycore.sl_vertical_coordinate, "sigma")
        self.assertIs(type(dycore.primitive),
                      primitive_equations.SemiLagrangianPrimitiveEquationsHybrid)
        self.assertNotIsInstance(dycore.primitive,
                                 LogPressureSemiLagrangianHybrid)

    def test_sigma_grids_keep_sigma_by_default(self):
        from dinosaur import primitive_equations

        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.dycore.dinosaur.log_pressure_interpolation import (
            LogPressureSemiLagrangianSigma)
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.terrain import TerrainData

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        kw = dict(terrain=TerrainData.aquaplanet(coords), dt_seconds=2400.0,
                  advection="semi_lagrangian")
        dycore = DinosaurDycore(coords=coords, **kw)
        self.assertEqual(dycore.sl_vertical_coordinate, "sigma")
        self.assertIs(type(dycore.primitive),
                      primitive_equations.SemiLagrangianPrimitiveEquations)
        logp = DinosaurDycore(coords=coords, **kw,
                              sl_options={"vertical_coordinate": "log_pressure"})
        self.assertIsInstance(logp.primitive, LogPressureSemiLagrangianSigma)

    def test_an_unknown_coordinate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vertical_coordinate"):
            _l47_dycore(vertical_coordinate="height")

    def test_log_nodes_are_finite_and_increasing(self):
        from jcm.dycore.dinosaur.log_pressure_interpolation import (
            log_interpolation_nodes)

        nodes = _l47_dycore().primitive._reference_vertical_nodes
        mapped = log_interpolation_nodes(nodes)
        np.testing.assert_allclose(mapped.centers, np.log(nodes.centers),
                                   rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(mapped.boundaries)))
        self.assertTrue(np.all(np.diff(mapped.boundaries) > 0))
        self.assertTrue(np.all(np.diff(mapped.centers) > 0))


class TopCellAdvectionTest(unittest.TestCase):
    """A profile linear in height is advected exactly through the top cell.

    Isothermal and constant-lapse-rate layers are linear in ``ln p``. Under a
    small upward displacement the top level must take the value at its
    departure point, which lies inside the linear top cell. In ``s`` that cell
    credits the level with 0.44 of the advection from below (the ratio
    ``s₁ ln(s₂/s₁)/(s₂ − s₁)`` on L47) — the numerical cold bias of the
    summer-mesosphere lid under upwelling.
    """

    def _advected_fraction(self, coordinate):
        dycore = _l47_dycore(vertical_coordinate=coordinate)
        s = np.asarray(dycore.primitive._reference_vertical_nodes.centers)
        lapse = 10.0                     # K per unit ln p: T falls upward
        # Anchored at the top so T′ is small where the 0.02 K signal is
        # measured (float32 round-off of a large T′ would dominate it).
        profile = lapse * np.log(s / s[0])
        out = _transported_temperature(dycore, profile, steps=1)
        exact = lapse * _DLNP           # change of T at a departure dlnp below
        return (out[:3] - profile[:3]) / exact

    def test_log_pressure_advects_the_top_levels_exactly(self):
        # Measured 1.000 / 1.000 / 0.998; the ``s`` rule differs by 0.56.
        np.testing.assert_allclose(self._advected_fraction("log_pressure"),
                                   1.0, atol=1e-2)

    def test_sigma_misreads_the_top_cells(self):
        # Measured 0.44 / 1.37 / 1.16 for the top three levels: the top
        # level under-advected, the next two over-advected, which is the
        # cold-lid / warm-second-level split of the ``s`` rule.
        fraction = self._advected_fraction("sigma")
        self.assertLess(fraction[0], 0.5)
        self.assertGreater(fraction[0], 0.4)
        self.assertGreater(fraction[1], 1.2)


class SigmaGridTransportTest(unittest.TestCase):
    """``log_pressure`` on a sigma grid transports through ``ln σ``.

    The SPEEDY L8 grid keeps ``σ`` by default; asked for ``ln σ`` its
    transport (whose σ boundaries include 0 at the lid) must stay finite and
    advect a profile linear in ``ln σ`` exactly in the cubic cells.
    """

    def test_profile_linear_in_log_sigma_is_advected_exactly(self):
        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.terrain import TerrainData

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        dycore = DinosaurDycore(
            coords=coords, terrain=TerrainData.aquaplanet(coords),
            dt_seconds=720.0, advection="semi_lagrangian",
            sl_options={"vertical_coordinate": "log_pressure"})
        s = np.asarray(_trajectory_nodes(dycore).centers)
        profile = 10.0 * np.log(s / s[0])
        out = _transported_temperature(dycore, profile, steps=1)
        self.assertTrue(np.all(np.isfinite(out)))
        fraction = (out - profile) / (10.0 * _DLNP)
        # the cubic cells (the first and last cells are linear in ln σ too,
        # so every level but the clipped bottom one is exact)
        np.testing.assert_allclose(fraction[:-1], 1.0, atol=1e-2)


class TwoDeltaZStabilityTest(unittest.TestCase):
    """Under steady upwelling, 2Δz structure at the L47 top grows in ``s``.

    Four-point Lagrange weights on the geometrically spaced top levels
    overshoot a 2Δz pattern for departure points just below a node; with the
    departure points held there by upwelling the cubic step is
    anti-diffusive in ``s`` and damping in ``ln s``.
    """

    def _amplitude_ratio(self, coordinate, steps=600):
        dycore = _l47_dycore(vertical_coordinate=coordinate)
        nlev = dycore.coords.vertical.layers
        pattern = np.zeros(nlev)
        pattern[:8] = 5.0 * (-1.0) ** np.arange(8)
        out = _transported_temperature(dycore, pattern, steps=steps)
        # the 2Δz component of the top eight levels
        alternating = (-1.0) ** np.arange(8)
        return abs(np.dot(out[:8], alternating)) / abs(
            np.dot(pattern[:8], alternating))

    def test_sigma_amplifies_two_delta_z(self):
        # Measured 1.50 after five days at 2 cm/s.
        self.assertGreater(self._amplitude_ratio("sigma"), 1.25)

    def test_log_pressure_damps_two_delta_z(self):
        # Measured 0.06.
        self.assertLess(self._amplitude_ratio("log_pressure"), 0.3)


class RunnerDoorTest(unittest.TestCase):
    """``dycore.sl_vertical_coordinate`` reaches the dycore."""

    def _dycore(self, overrides):
        import os

        from hydra import compose, initialize_config_dir

        import jcm
        from jcm.runners import build_model

        with initialize_config_dir(
                config_dir=f"{os.path.dirname(jcm.__file__)}/config",
                version_base=None):
            cfg = compose(config_name="config", overrides=[
                "physics=held_suarez", "grid=echam_t63_l47_hybrid",
                "grid.spectral_truncation=21", "dycore.advection=semi_lagrangian",
                *overrides])
        return build_model(cfg).dycore

    def test_null_is_the_family_default(self):
        self.assertEqual(self._dycore([]).sl_vertical_coordinate,
                         "log_pressure")

    def test_override_reaches_the_dycore(self):
        self.assertEqual(
            self._dycore(["dycore.sl_vertical_coordinate=sigma"])
            .sl_vertical_coordinate, "sigma")


if __name__ == "__main__":
    unittest.main()
