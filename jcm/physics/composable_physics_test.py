"""Tests for PhysicsTerm and ComposablePhysics.

Phase 2b differentiability gate: verify nnx.grad flows through composable
physics terms, and that composition operators work correctly.

Date: 2026-04-12
"""

import types
import unittest
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from flax import nnx

from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData


# ---------------------------------------------------------------------------
# Toy terms for testing
# ---------------------------------------------------------------------------

class LinearHeating(PhysicsTerm):
    """Toy term: temperature tendency = alpha * mean(temperature)."""

    name: ClassVar[str] = "linear_heating"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("heating_rate",)

    def __init__(self, alpha: float = 1.0):
        """Initialize LinearHeating."""
        self.alpha = nnx.Param(jnp.array(alpha))

    def __call__(self, state, diagnostics, forcing, terrain):
        heating = self.alpha[...] * state.temperature
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            temperature=heating,
        )
        new_diag = {**diagnostics, "heating_rate": heating}
        return tendency, new_diag


class QuadraticMoistening(PhysicsTerm):
    """Toy term: specific_humidity tendency = beta * temperature^2."""

    name: ClassVar[str] = "quadratic_moistening"
    category: ClassVar[str] = "convection"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("moisture_source",)

    def __init__(self, beta: float = 0.5):
        """Initialize QuadraticMoistening."""
        self.beta = nnx.Param(jnp.array(beta))

    def __call__(self, state, diagnostics, forcing, terrain):
        source = self.beta[...] * state.temperature ** 2
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            specific_humidity=source,
        )
        new_diag = {**diagnostics, "moisture_source": source}
        return tendency, new_diag


class DiagnosticConsumer(PhysicsTerm):
    """Toy term that reads upstream diagnostics and uses them."""

    name: ClassVar[str] = "diagnostic_consumer"
    category: ClassVar[str] = "surface"
    requires: ClassVar[tuple[str, ...]] = ("heating_rate",)
    provides: ClassVar[tuple[str, ...]] = ("surface_flux",)

    def __init__(self, gamma: float = 0.1):
        """Initialize DiagnosticConsumer."""
        self.gamma = nnx.Param(jnp.array(gamma))

    def __call__(self, state, diagnostics, forcing, terrain):
        # Use upstream heating_rate to compute a wind tendency
        heating = diagnostics["heating_rate"]
        wind_tend = self.gamma[...] * heating
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            u_wind=wind_tend,
        )
        new_diag = {**diagnostics, "surface_flux": wind_tend}
        return tendency, new_diag


class ToyRadiation(PhysicsTerm):
    """Minimal radiation term exposing readable ``RadiationParameters``.

    Stands in for RRTMGP/grey/NN-emulator in tests of
    ``_rederive_radiation_dependents`` (jax-gcm#926) that don't need a real
    radiation solve — only the ``self.params`` an ``nnx.Param`` holding
    ``RadiationParameters`` the gated sibling below reads its cadence from.
    """

    name: ClassVar[str] = "toy_radiation"
    category: ClassVar[str] = "radiation"

    def __init__(self, radiation_interval: float = 7200.0):
        """Initialize with the given radiation cadence [s]."""
        self.params = nnx.Param(
            RadiationParameters.default(radiation_interval=radiation_interval))

    def __call__(self, state, diagnostics, forcing, terrain):
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


class RadiationWithNoParams(PhysicsTerm):
    """A radiation-category term that exposes no ``RadiationParameters``.

    Stands in for a user-supplied radiation term that never set
    ``self.params`` — valid on its own (nothing requires it), an error only
    where a gated sibling needs to read a cadence from it.
    """

    name: ClassVar[str] = "radiation_with_no_params"
    category: ClassVar[str] = "radiation"

    def __call__(self, state, diagnostics, forcing, terrain):
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


class GatedSibling(PhysicsTerm):
    """Toy sibling exposing ``configure_radiation_gate``, like ``JamOpticsTerm``.

    Records every interval it is configured with (rather than just the
    latest) so a test can tell "reconfigured" from "never touched".
    """

    name: ClassVar[str] = "gated_sibling"
    category: ClassVar[str] = "aerosol_optics"

    def __init__(self):
        """Start unconfigured, like a freshly constructed gated term."""
        self._radiation_interval_s = None
        self.configured_with: list[float] = []

    def configure_radiation_gate(self, interval_s) -> None:
        self._radiation_interval_s = float(interval_s)
        self.configured_with.append(float(interval_s))

    def __call__(self, state, diagnostics, forcing, terrain):
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


class TiedParamTermA(PhysicsTerm):
    """One half of a toy tied-parameter pair — see :class:`TiedParamTermB`."""

    name: ClassVar[str] = "tied_param_a"
    category: ClassVar[str] = "tied_diagnostics_a"

    def __init__(self, shared_param):
        """Hold a (possibly SHARED) ``nnx.Param`` -- never read, just carried."""
        self.shared_param = shared_param

    def __call__(self, state, diagnostics, forcing, terrain):
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


class TiedParamTermB(PhysicsTerm):
    """Other half of a toy tied-parameter pair — see :class:`TiedParamTermA`.

    Two terms built from the SAME ``nnx.Param`` object (one
    ``TiedParamTermA`` and one ``TiedParamTermB``, each handed the
    identical instance) stand in for a differentiable tied-parameter
    model: an optimizer stepping what it believes is ONE shared parameter
    must see that update reflected through BOTH terms. A composition-
    editing operator that clones terms ONE AT A TIME — ``[nnx.clone(t) for
    t in terms]`` — treats each as its own independent graph root, so the
    tie is silently broken even by an edit that never touches either term
    (GitHub Codex review of PR #1096, inline thread on
    ``composable_physics.py``'s old per-term clone loops). Two DIFFERENT
    classes/categories (rather than two instances of one class) so a test
    can pick either one back out of a result list by ``isinstance``
    without needing an identity link to the pre-clone originals.
    """

    name: ClassVar[str] = "tied_param_b"
    category: ClassVar[str] = "tied_diagnostics_b"

    def __init__(self, shared_param):
        """Hold a (possibly SHARED) ``nnx.Param`` -- never read, just carried."""
        self.shared_param = shared_param

    def __call__(self, state, diagnostics, forcing, terrain):
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


# ---------------------------------------------------------------------------
# Helper: build a small test state
# ---------------------------------------------------------------------------

def _make_test_state(shape=(2, 4, 8)):
    """Create a simple PhysicsState with non-trivial values."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    return PhysicsState(
        u_wind=jax.random.normal(keys[0], shape),
        v_wind=jax.random.normal(keys[1], shape),
        temperature=jax.random.normal(keys[2], shape) + 250.0,
        specific_humidity=jnp.abs(jax.random.normal(keys[3], shape)),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones(shape[1:]),
    )


def _make_test_forcing(shape=(4, 8)):
    return ForcingData.zeros(shape)


def _make_test_terrain(shape=(4, 8)):
    zero = jnp.zeros(shape)
    return TerrainData(
        orog=zero, phis0=zero, fmask=zero, lfluxland=jnp.bool_(False),
        orostd=zero, orosig=zero, orogam=zero,
        orothe=zero, oropic=zero, oroval=zero,
    )


def _get_test_coords():
    """Build a cheap, real ``CoordinateSystem`` for ``cache_coords()`` calls.

    jax-gcm#926's final design resolves/validates ``band_config`` and
    configures gated siblings lazily, in ``cache_coords()`` (via
    ``ComposablePhysics._finalize``) rather than eagerly at every
    composition-editing operator call — so a test pinning that behaviour
    needs a real (if minimal) coordinate system to call it with, not just
    the toy state/forcing/terrain helpers above.
    """
    from jcm.utils import get_coords
    return get_coords(np.linspace(0, 1, 9), spectral_truncation=21)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPhysicsTermBasics(unittest.TestCase):
    """Basic construction and call tests for PhysicsTerm subclasses."""

    def test_single_term_call(self):
        term = LinearHeating(alpha=2.0)
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()

        tend, diag = term(state, {}, forcing, terrain)

        self.assertEqual(tend.temperature.shape, state.temperature.shape)
        self.assertIn("heating_rate", diag)
        # tendency should be 2.0 * temperature
        npt.assert_allclose(tend.temperature, 2.0 * state.temperature)

    def test_term_addition_creates_composable(self):
        a = LinearHeating()
        b = QuadraticMoistening()
        physics = a + b
        self.assertEqual(type(physics).__name__, "ComposablePhysics")
        self.assertEqual(len(physics.terms), 2)

    def test_sum_of_terms(self):
        terms = [LinearHeating(), QuadraticMoistening()]
        physics = sum(terms)
        self.assertEqual(type(physics).__name__, "ComposablePhysics")
        self.assertEqual(len(physics.terms), 2)


class TestComposablePhysics(unittest.TestCase):
    """Tests for ComposablePhysics composition operators and validation."""

    def _make_physics(self):
        return ComposablePhysics(terms=[
            LinearHeating(alpha=1.0),
            QuadraticMoistening(beta=0.5),
            DiagnosticConsumer(gamma=0.1),
        ])

    def test_compute_tendencies(self):
        physics = self._make_physics()
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()

        tend, diag = physics.compute_tendencies(state, forcing, terrain)

        # Check all diagnostic keys are present
        self.assertIn("heating_rate", diag)
        self.assertIn("moisture_source", diag)
        self.assertIn("surface_flux", diag)

        # Temperature tendency comes from LinearHeating
        npt.assert_allclose(
            tend.temperature, 1.0 * state.temperature, rtol=1e-5
        )
        # specific_humidity tendency from QuadraticMoistening
        npt.assert_allclose(
            tend.specific_humidity, 0.5 * state.temperature ** 2, rtol=1e-5
        )
        # u_wind tendency from DiagnosticConsumer using upstream heating_rate
        npt.assert_allclose(
            tend.u_wind, 0.1 * state.temperature, rtol=1e-5
        )

    def test_preferred_advection_aggregation(self):
        """No vote -> None; any Eulerian -> Eulerian; any SL wins outright.

        SL wins because a term asking for it may need it for correctness
        (nodal, monotone transport), whereas an Eulerian preference is a
        fidelity/cost one (SPEEDY).
        """
        class Eulerian(LinearHeating):
            name: ClassVar[str] = "eulerian_vote"
            # A category OTHER than "radiation": _make_physics() already has
            # a "radiation"-category term (LinearHeating), and two terms of
            # that category in one composition is a jax-gcm#926 conflict
            # (nothing says which one is the active radiation scheme) --
            # this test is about preferred_advection aggregation, which has
            # nothing to do with that, so the vote term just needs SOME
            # category, not this specific one.
            category: ClassVar[str] = "transport_vote"

            def preferred_advection(self):
                return "eulerian"

        class SemiLagrangian(LinearHeating):
            name: ClassVar[str] = "sl_vote"
            category: ClassVar[str] = "transport_vote"

            def preferred_advection(self):
                return "semi_lagrangian"

        self.assertIsNone(self._make_physics().preferred_advection())
        self.assertEqual(
            (self._make_physics() + Eulerian()).preferred_advection(), "eulerian")
        self.assertEqual(
            ComposablePhysics(terms=[Eulerian(), SemiLagrangian()]).preferred_advection(),
            "semi_lagrangian")

    def test_speedy_prefers_eulerian(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics

        self.assertEqual(speedy_physics().preferred_advection(), "eulerian")

    def test_replace(self):
        physics = self._make_physics()
        new_rad = LinearHeating(alpha=5.0)
        replaced = physics.replace("radiation", new_rad)
        self.assertEqual(len(replaced.terms), 3)
        # A CLONE of `new_rad` joins, not `new_rad` itself (jax-gcm#926's
        # final design: ownership is the only invariant `replace` itself
        # enforces, and the incoming term may be one the caller reuses
        # elsewhere or that already belongs to another live composition —
        # see `replace`'s docstring).
        self.assertIsInstance(replaced.terms[0], LinearHeating)
        self.assertIsNot(replaced.terms[0], new_rad)
        self.assertEqual(float(replaced.terms[0].alpha[...]), 5.0)

    def test_remove(self):
        physics = self._make_physics()
        removed = physics.remove("convection")
        self.assertEqual(len(removed.terms), 2)
        categories = [t.category for t in removed.terms]
        self.assertNotIn("convection", categories)

    def test_add_two_composables(self):
        a = ComposablePhysics(terms=[LinearHeating()])
        b = ComposablePhysics(terms=[QuadraticMoistening()])
        combined = a + b
        self.assertEqual(len(combined.terms), 2)

    def test_validation_catches_missing_requires(self):
        with self.assertRaises(ValueError) as ctx:
            # DiagnosticConsumer requires "heating_rate" but no upstream provides it
            ComposablePhysics(terms=[DiagnosticConsumer()])
        self.assertIn("heating_rate", str(ctx.exception))

    def test_validation_passes_valid_ordering(self):
        # LinearHeating provides "heating_rate", DiagnosticConsumer requires it
        physics = ComposablePhysics(terms=[
            LinearHeating(),
            DiagnosticConsumer(),
        ])
        self.assertEqual(len(physics.terms), 2)

    def test_replace_hands_the_displaced_term_to_its_replacement(self):
        """Post-compose configuration must survive a swap (jax-gcm#835).

        Settings applied by a factory AFTER composition live on the instance,
        so without this handover a replacement silently reverts to constructor
        defaults — which is how a swapped optics term loses its radiation
        cadence and recomputes every band on every step.
        """
        seen = []

        class Configurable(LinearHeating):
            def adopt_runtime_configuration(self, previous):
                seen.append(previous)

        physics = self._make_physics()
        original = physics.terms[0]
        new_rad = Configurable(alpha=5.0)
        physics.replace("radiation", new_rad)
        self.assertEqual(seen, [original])

    def test_adopt_runtime_configuration_defaults_to_noop(self):
        """Most terms carry nothing, so the base hook must be harmless."""
        a, b = LinearHeating(), LinearHeating(alpha=2.0)
        self.assertIsNone(a.adopt_runtime_configuration(b))

    def test_replace_nonexistent_category_raises(self):
        physics = self._make_physics()
        with self.assertRaises(ValueError):
            physics.replace("nonexistent", LinearHeating())

    # -- jax-gcm#926: every composition-building path must be
    # -- indistinguishable from the same composition built from scratch with
    # -- the resulting radiation term. Five separate Codex review rounds each
    # -- found a DIFFERENT path missing this (replace, remove,
    # -- ComposablePhysics + ComposablePhysics, PhysicsTerm + ComposablePhysics
    # -- /ComposablePhysics + PhysicsTerm, PhysicsTerm + PhysicsTerm with
    # -- either side holding the radiation term) -- a design signal, not five
    # -- bugs, that led to restructuring the fix around ONE choke point
    # -- (_finalize_radiation_config) every one of these operators shares.
    # -- test_matches_from_scratch_across_every_composition_path below is
    # -- that one parametrised test, replacing what had become a growing,
    # -- per-round pile of near-duplicate tests; a handful of named tests
    # -- after it cover properties the matrix doesn't naturally express
    # -- (a true no-op, the ValueError, non-gated cache_band_config
    # -- consumers, a reused term aliasing two DIFFERENT compositions).

    def test_matches_from_scratch_across_every_composition_path(self):
        """The ``ComposablePhysics``/``PhysicsTerm`` matrix for jax-gcm#926's
        FINAL design.

        Nine Codex review rounds on this issue each found another way a
        composition-editing operator's OWN decision about ``band_config``/
        gated-sibling cadence could disagree by operand order or by which
        operator ran. The fix is no longer "make every operator decide
        correctly" — it is "no operator decides at all": ``replace``,
        ``remove`` and both ``__add__`` implementations are now pure term-
        list (plus ownership-cloning) operations, and ``band_config`` is
        resolved/validated exactly once, lazily, by
        :meth:`ComposablePhysics._finalize` (called from
        :meth:`~ComposablePhysics.cache_coords`) — see
        ``docs/source/design/composable_physics.md``.

        This matrix therefore no longer needs a "does this shape re-derive
        or preserve" distinction per cell: EVERY ``replace``/``remove`` call
        carries ``band_config`` through VERBATIM (trivially, since neither
        makes any radiation-relevant decision), and ``__add__`` picks
        between the two operands' CURRENT ``(band_config, explicit)`` pairs
        via ``_add_band_config`` — an EXPLICIT override on either side wins,
        both explicit and different raises, neither explicit defers to
        ``_finalize`` — a rule that does not look at which operand (if
        either) owns the active radiation term at all (see
        ``_add_band_config``'s own docstring for why that used to be part
        of the rule, and the bug that kept causing). The dedicated
        ``test_add_band_config_merge_rule_matrix`` below exercises that
        rule directly and exhaustively; the cells here instead pin that
        the FULL pipeline — merge rule, then ``_finalize`` — reaches the
        identical final picture a from-scratch build would.
        What's actually being pinned per shape is that ``cache_coords()`` —
        called on both ``result`` and a from-scratch ``expected`` — resolves
        to the IDENTICAL final picture: same ``band_config``, same gated
        ("JAM optics"-like) sibling cadence, same one-step tendencies (a
        real ``compute_tendencies`` call, not just inspected attributes —
        these toy terms return a zero tendency regardless of configuration,
        so this also pins that the comparison ACTUALLY runs the composition
        rather than only checking config side-channels).
        """
        coords = _get_test_coords()
        explicit_override = RadiationBandConfig(
            lw_band_centers_nm=(1000.0, 2000.0),
            sw_band_centers_nm=(400.0, 500.0, 600.0))

        def one_step(physics):
            state = _make_test_state()
            tend, _ = physics.compute_tendencies(
                state, _make_test_forcing(), _make_test_terrain())
            return tend

        def assert_matches(result, expected):
            result.cache_coords(coords)
            expected.cache_coords(coords)
            self.assertEqual(result.band_config, expected.band_config)
            result_gates = [t._radiation_interval_s for t in result.terms
                            if isinstance(t, GatedSibling)]
            expected_gates = [t._radiation_interval_s for t in expected.terms
                               if isinstance(t, GatedSibling)]
            self.assertEqual(result_gates, expected_gates)
            np.testing.assert_array_equal(
                one_step(result).temperature, one_step(expected).temperature)

        for band_config in (None, explicit_override):
            with self.subTest(band_config="explicit" if band_config else "derived"):

                with self.subTest(shape="unrelated_edit_preserves"):
                    # A category OTHER than radiation -- band_config is
                    # carried through verbatim by `replace` regardless of
                    # what it is, so this is really the SAME claim as every
                    # other `replace`/`remove` cell below; kept as its own
                    # cell because it is the shape the pre-redesign code
                    # had the OPPOSITE failure mode on (forced re-derivation
                    # even though nothing radiation-relevant changed).
                    physics = ComposablePhysics(
                        terms=[ToyRadiation(radiation_interval=7200.0),
                               GatedSibling(), QuadraticMoistening()],
                        band_config=band_config)
                    result = physics.replace(
                        "convection", QuadraticMoistening(beta=2.0))
                    expected = ComposablePhysics(
                        terms=[ToyRadiation(radiation_interval=7200.0),
                               GatedSibling(), QuadraticMoistening(beta=2.0)],
                        band_config=band_config)
                    assert_matches(result, expected)

                with self.subTest(shape="replace"):
                    # Swapping the radiation term ITSELF no longer forces
                    # re-derivation either -- `band_config` is carried
                    # through verbatim here too (the toy `ToyRadiation` is
                    # never RECOGNIZED by `for_terms`, so an explicit value
                    # is never validated/rejected against it either; see
                    # `_finalize`).
                    physics = ComposablePhysics(
                        terms=[ToyRadiation(radiation_interval=7200.0),
                               GatedSibling(), QuadraticMoistening()],
                        band_config=band_config)
                    new_rad = ToyRadiation(radiation_interval=1800.0)
                    result = physics.replace("radiation", new_rad)
                    expected = ComposablePhysics(
                        terms=[ToyRadiation(radiation_interval=1800.0),
                               GatedSibling(), QuadraticMoistening()],
                        band_config=band_config)
                    assert_matches(result, expected)

                with self.subTest(shape="remove"):
                    physics = ComposablePhysics(
                        terms=[ToyRadiation(radiation_interval=1800.0),
                               QuadraticMoistening()],
                        band_config=band_config)
                    result = physics.remove("radiation")
                    expected = ComposablePhysics(
                        terms=[QuadraticMoistening()], band_config=band_config)
                    assert_matches(result, expected)

                # The three `+` shapes below pair a composition that owns
                # NO radiation term of its own (carrying the outer-loop
                # `band_config`) with one that DOES (a bare `ToyRadiation`,
                # which -- unlike a `ComposablePhysics` -- never carries a
                # `band_config`/explicit flag of its own: `_unpack_add_operand`
                # always reports `(None, False)` for a bare term). Per
                # `_add_band_config`'s FINAL, radiation-position-independent
                # rule (jax-gcm#926, GitHub review of PR #1096's third
                # case-by-case finding on this exact merge), which side owns
                # the radiation term decides NOTHING here: an EXPLICIT
                # `band_config` (the ``"explicit"`` outer-loop iteration)
                # always survives regardless of which operand it sits on,
                # because the bare `ToyRadiation` operand is never explicit
                # either way. Only the ``"derived"`` iteration (`band_config`
                # is `None` on both sides) actually re-derives, to broadband
                # (`ToyRadiation` is never a recognized backend) -- which is
                # also what a from-scratch build with no override gets, so
                # `expected` below passes the SAME outer-loop `band_config`
                # as `physics`/`composable` rather than hard-coding "no
                # override" for every iteration.
                with self.subTest(shape="composable_plus_composable"):
                    physics = ComposablePhysics(
                        terms=[GatedSibling(), QuadraticMoistening()],
                        band_config=band_config)
                    added = ComposablePhysics(
                        terms=[ToyRadiation(radiation_interval=1800.0)])
                    result = physics + added
                    expected = ComposablePhysics(
                        terms=[GatedSibling(), QuadraticMoistening(),
                               ToyRadiation(radiation_interval=1800.0)],
                        band_config=band_config)
                    assert_matches(result, expected)

                with self.subTest(shape="term_plus_composable"):
                    composable = ComposablePhysics(
                        terms=[GatedSibling(), QuadraticMoistening()],
                        band_config=band_config)
                    result = ToyRadiation(radiation_interval=1800.0) + composable
                    expected = ComposablePhysics(
                        terms=[ToyRadiation(radiation_interval=1800.0),
                               GatedSibling(), QuadraticMoistening()],
                        band_config=band_config)
                    assert_matches(result, expected)

                with self.subTest(shape="composable_plus_term"):
                    composable = ComposablePhysics(
                        terms=[GatedSibling(), QuadraticMoistening()],
                        band_config=band_config)
                    result = composable + ToyRadiation(radiation_interval=1800.0)
                    expected = ComposablePhysics(
                        terms=[GatedSibling(), QuadraticMoistening(),
                               ToyRadiation(radiation_interval=1800.0)],
                        band_config=band_config)
                    assert_matches(result, expected)

                # Bare term + bare term never has an existing composition's
                # band_config to preserve on either side, so there is no
                # "explicit override survives" case here -- only the
                # "derived" subTest iteration (band_config is None) applies.
                if band_config is None:
                    with self.subTest(shape="term_plus_term_radiation_left"):
                        result = (ToyRadiation(radiation_interval=1800.0)
                                  + GatedSibling())
                        expected = ComposablePhysics(terms=[
                            ToyRadiation(radiation_interval=1800.0), GatedSibling()])
                        assert_matches(result, expected)

                    with self.subTest(shape="term_plus_term_radiation_right"):
                        result = (GatedSibling()
                                  + ToyRadiation(radiation_interval=1800.0))
                        expected = ComposablePhysics(terms=[
                            GatedSibling(), ToyRadiation(radiation_interval=1800.0)])
                        assert_matches(result, expected)

        # The explicit override lives on the operand that OWNS the
        # radiation term this time (the opposite of the three `+` cells
        # above) -- `_add_band_config` picks that operand's value in both
        # orders, so it survives unperturbed.
        def make_custom_radiation_composable():
            return ComposablePhysics(
                terms=[ToyRadiation(radiation_interval=1800.0), GatedSibling()],
                band_config=explicit_override)

        with self.subTest(shape="non_radiation_plus_custom_radiation"):
            no_rad = ComposablePhysics(terms=[QuadraticMoistening()])
            custom_rad = make_custom_radiation_composable()
            result = no_rad + custom_rad
            expected = ComposablePhysics(
                terms=[QuadraticMoistening(),
                       ToyRadiation(radiation_interval=1800.0), GatedSibling()],
                band_config=explicit_override)
            assert_matches(result, expected)

        with self.subTest(shape="custom_radiation_plus_non_radiation"):
            no_rad = ComposablePhysics(terms=[QuadraticMoistening()])
            custom_rad = make_custom_radiation_composable()
            result = custom_rad + no_rad
            expected = ComposablePhysics(
                terms=[ToyRadiation(radiation_interval=1800.0), GatedSibling(),
                       QuadraticMoistening()],
                band_config=explicit_override)
            assert_matches(result, expected)

    def test_remove_radiation_noop_preserves_explicit_band_config_and_gated_sibling(self):
        """``remove("radiation")`` on a composition with NO radiation term is
        a true no-op filter, not a re-derivation (jax-gcm#926 Codex review):
        ``category == "radiation"`` alone used to trigger re-derivation even
        when nothing of that category was present, silently discarding an
        explicit, non-default ``band_config`` on a call that changed
        nothing, and could spuriously raise on a gated sibling that was
        never reading a cadence from this (always-absent) radiation term in
        the first place.
        """
        custom = RadiationBandConfig(
            lw_band_centers_nm=(1.0, 2.0), sw_band_centers_nm=(3.0,))
        gated = GatedSibling()
        gated.configure_radiation_gate(1234.0)  # configured some other way
        physics = ComposablePhysics(
            terms=[gated, QuadraticMoistening()], band_config=custom)
        removed = physics.remove("radiation")  # must not raise
        self.assertEqual(removed.band_config, custom)
        (removed_gated,) = (t for t in removed.terms if isinstance(t, GatedSibling))
        self.assertEqual(removed_gated._radiation_interval_s, 1234.0)

    def test_gated_sibling_with_no_radiation_term_anywhere_raises(self):
        """A gated sibling that cannot find ANY radiation term to read a
        cadence from errors loudly -- on every path that could introduce
        one, at ``cache_coords()`` time (jax-gcm#926's final design: no
        composition-editing operator validates anything any more; see
        :meth:`ComposablePhysics._finalize`, called from ``cache_coords``).

        Per CLAUDE.md ("Finish the Job"): the per-band carry shapes a gated
        term's ``cache_coords``/``initial_carry_state`` already fixed are a
        trace-time choice this call cannot safely second-guess, so there is
        no safe "last known cadence" to fall back to -- this must be a
        clear error, not a silent stale or compute-every-step default.
        """
        coords = _get_test_coords()
        with self.subTest(path="replace_with_unreadable_params"):
            physics = ComposablePhysics(terms=[ToyRadiation(), GatedSibling()])
            replaced = physics.replace("radiation", RadiationWithNoParams())
            with self.assertRaisesRegex(ValueError, "radiation cadence"):
                replaced.cache_coords(coords)
        with self.subTest(path="add_to_composable_with_no_radiation"):
            physics = ComposablePhysics(terms=[QuadraticMoistening()])
            added = physics + GatedSibling()
            with self.assertRaisesRegex(ValueError, "radiation cadence"):
                added.cache_coords(coords)
        with self.subTest(path="term_plus_term_neither_side_radiation"):
            combined = GatedSibling() + QuadraticMoistening()
            with self.assertRaisesRegex(ValueError, "radiation cadence"):
                combined.cache_coords(coords)

    def test_add_two_radiation_terms_raises_on_every_shape(self):
        """``+`` combining two operands that EACH already carry a
        ``"radiation"``-category term must raise -- at ``cache_coords()``
        time (jax-gcm#926's final design): the combined composition would
        have two of them, and nothing says which is active. None of the
        three ``+`` shapes that can reach this validated it before this
        Codex review round (PR #1096).
        """
        coords = _get_test_coords()
        with self.subTest(shape="composable_plus_composable"):
            a = ComposablePhysics(terms=[ToyRadiation(radiation_interval=1800.0)])
            b = ComposablePhysics(terms=[ToyRadiation(radiation_interval=3600.0)])
            with self.assertRaisesRegex(ValueError, "radiation"):
                (a + b).cache_coords(coords)
        with self.subTest(shape="term_plus_composable"):
            b = ComposablePhysics(terms=[ToyRadiation(radiation_interval=3600.0)])
            with self.assertRaisesRegex(ValueError, "radiation"):
                (ToyRadiation(radiation_interval=1800.0) + b).cache_coords(coords)
        with self.subTest(shape="composable_plus_term"):
            a = ComposablePhysics(terms=[ToyRadiation(radiation_interval=1800.0)])
            with self.assertRaisesRegex(ValueError, "radiation"):
                (a + ToyRadiation(radiation_interval=3600.0)).cache_coords(coords)
        with self.subTest(shape="term_plus_term"):
            combined = (ToyRadiation(radiation_interval=1800.0)
                        + ToyRadiation(radiation_interval=3600.0))
            with self.assertRaisesRegex(ValueError, "radiation"):
                combined.cache_coords(coords)

    def test_add_with_no_radiation_anywhere_resolves_explicit_band_config(self):
        """``_add_band_config``'s "neither operand owns radiation" branch:
        there is nothing to derive ``band_config`` from either way, so an
        explicit override on just one side survives, matching explicit
        overrides on both sides survive, and conflicting explicit overrides
        (nothing to arbitrate by) raise.
        """
        custom_a = RadiationBandConfig(
            lw_band_centers_nm=(1.0,), sw_band_centers_nm=(2.0,))
        custom_b = RadiationBandConfig(
            lw_band_centers_nm=(9.0,), sw_band_centers_nm=(8.0,))

        with self.subTest(case="one_side_explicit_left"):
            a = ComposablePhysics(terms=[QuadraticMoistening()],
                                   band_config=custom_a)
            b = ComposablePhysics(terms=[QuadraticMoistening(beta=0.0)])
            self.assertEqual((a + b).band_config, custom_a)

        with self.subTest(case="one_side_explicit_right"):
            a = ComposablePhysics(terms=[QuadraticMoistening()])
            b = ComposablePhysics(terms=[QuadraticMoistening(beta=0.0)],
                                   band_config=custom_a)
            self.assertEqual((a + b).band_config, custom_a)

        with self.subTest(case="both_sides_explicit_and_equal"):
            a = ComposablePhysics(terms=[QuadraticMoistening()],
                                   band_config=custom_a)
            b = ComposablePhysics(terms=[QuadraticMoistening(beta=0.0)],
                                   band_config=custom_a)
            result = a + b
            self.assertEqual(result.band_config, custom_a)

        with self.subTest(case="both_sides_explicit_and_conflicting"):
            a = ComposablePhysics(terms=[QuadraticMoistening()],
                                   band_config=custom_a)
            b = ComposablePhysics(terms=[QuadraticMoistening(beta=0.0)],
                                   band_config=custom_b)
            with self.assertRaisesRegex(ValueError, "DIFFERENT explicit"):
                a + b

    def test_add_band_config_merge_rule_matrix(self):
        """``_add_band_config``'s FINAL, radiation-position-independent
        merge rule (jax-gcm#926, GitHub review of PR #1096's THIRD
        case-by-case finding on this exact merge — see that function's
        docstring for the first two and why a fourth patch was refused in
        favour of removing the radiation-term check entirely): an
        explicit override on either side survives, REGARDLESS of which
        operand (if either) owns the active ``"radiation"``-category
        term, and regardless of whether that term is a bare
        ``PhysicsTerm`` or sits inside a ``ComposablePhysics``; both sides
        explicit and DIFFERENT raises; neither side explicit defers
        entirely to :meth:`~ComposablePhysics._finalize`.

        Parametrised over ``(left_state, right_state)`` — each one of
        ``"none"`` (no ``band_config`` ever passed), ``"derived"`` (no
        ``band_config`` passed, but :meth:`~ComposablePhysics.cache_coords`
        already ran once, so it now holds a NON-explicit, previously-
        derived value), ``"explicit_a"``/``"explicit_b"`` (two DIFFERENT
        genuine overrides) — CROSSED with ``placement``: WHERE the active
        radiation term sits — the left operand's own terms, the right
        operand's own terms, a bare ``PhysicsTerm`` substituted for the
        entire left or right operand, or nowhere at all. ``ToyRadiation``
        is never a RECOGNIZED backend (see
        :meth:`RadiationBandConfig.is_recognized`), so the non-explicit
        outcome is :meth:`RadiationBandConfig.broadband` regardless of
        whether a radiation term is present at all, which is exactly what
        makes ``placement`` a pure confound here: every cell's EXPECTED
        ``(band_config, explicit)`` pair depends ONLY on
        ``(left_state, right_state)``, never on ``placement`` — this test
        asserts exactly that by running every state pair across every
        valid placement and checking the identical expected outcome each
        time (``bare_left``/``bare_right`` are only valid, i.e. meaningful
        to test, when that side's own state is ``"none"`` — a bare term
        can never itself be ``"derived"`` or ``"explicit"``).

        ``test_add_with_no_radiation_anywhere_resolves_explicit_band_config``
        pins the same claim restricted to ``placement="neither"``; this
        matrix is the superset that proves ``placement`` never matters.
        """
        coords = _get_test_coords()
        custom_a = RadiationBandConfig(
            lw_band_centers_nm=(1.0,), sw_band_centers_nm=(2.0,))
        custom_b = RadiationBandConfig(
            lw_band_centers_nm=(9.0,), sw_band_centers_nm=(8.0,))
        value_of = {"explicit_a": custom_a, "explicit_b": custom_b}

        def make_operand(state: str, radiation_here: bool) -> ComposablePhysics:
            """One ``+`` operand in the given provenance ``state``,
            optionally with a ``ToyRadiation`` term among its OWN terms
            (``radiation_here``) -- a bare-term operand is substituted by
            the caller afterward, never built here.
            """
            terms = [QuadraticMoistening()]
            if radiation_here:
                terms = [ToyRadiation(radiation_interval=1800.0), *terms]
            if state in value_of:
                physics = ComposablePhysics(terms=terms, band_config=value_of[state])
            else:
                physics = ComposablePhysics(terms=terms)
                if state == "derived":
                    physics.cache_coords(coords)
            return physics

        def valid_placements(left_state, right_state):
            placements = ["left_composable", "right_composable", "neither"]
            if left_state == "none":
                placements.append("bare_left")
            if right_state == "none":
                placements.append("bare_right")
            return placements

        def build_pair(left_state, right_state, placement):
            left = make_operand(
                left_state, radiation_here=(placement == "left_composable"))
            right = make_operand(
                right_state, radiation_here=(placement == "right_composable"))
            left_operand = (ToyRadiation(radiation_interval=1800.0)
                             if placement == "bare_left" else left)
            right_operand = (ToyRadiation(radiation_interval=1800.0)
                              if placement == "bare_right" else right)
            return left_operand, right_operand

        # Three explicit lists of (left_state, right_state) pairs, grouped
        # by their expected outcome AFTER cache_coords() -- spelled out
        # rather than computed from a rule, so a typo in the MERGE RULE
        # itself can't also corrupt the EXPECTATION the same way.
        non_explicit_pairs = [
            ("none", "none"), ("none", "derived"),
            ("derived", "none"), ("derived", "derived"),
        ]
        explicit_a_pairs = [
            ("explicit_a", "none"), ("explicit_a", "derived"),
            ("none", "explicit_a"), ("derived", "explicit_a"),
            ("explicit_a", "explicit_a"),
        ]
        conflicting_pairs = [("explicit_a", "explicit_b"),
                             ("explicit_b", "explicit_a")]

        for left_state, right_state in non_explicit_pairs:
            for placement in valid_placements(left_state, right_state):
                with self.subTest(left=left_state, right=right_state,
                                   placement=placement):
                    left_operand, right_operand = build_pair(
                        left_state, right_state, placement)
                    result = left_operand + right_operand
                    result.cache_coords(coords)
                    self.assertEqual(
                        result.band_config, RadiationBandConfig.broadband())
                    self.assertFalse(result._band_config_explicit)

        for left_state, right_state in explicit_a_pairs:
            for placement in valid_placements(left_state, right_state):
                with self.subTest(left=left_state, right=right_state,
                                   placement=placement):
                    left_operand, right_operand = build_pair(
                        left_state, right_state, placement)
                    result = left_operand + right_operand
                    result.cache_coords(coords)
                    self.assertEqual(result.band_config, custom_a)
                    self.assertTrue(result._band_config_explicit)

        for left_state, right_state in conflicting_pairs:
            # Both sides are already explicit, so a bare-term placement
            # (always non-explicit) is not a meaningful variant here --
            # only the three placements that keep both sides as an
            # explicit ComposablePhysics apply.
            for placement in ("left_composable", "right_composable", "neither"):
                with self.subTest(left=left_state, right=right_state,
                                   placement=placement):
                    left_operand, right_operand = build_pair(
                        left_state, right_state, placement)
                    with self.assertRaisesRegex(ValueError, "DIFFERENT explicit"):
                        left_operand + right_operand

    def test_add_conflicting_composition_settings_raises(self):
        """``checkpoint_terms``/``vectorize_columns``/``dt_seconds`` go
        through the same "operand order cannot matter" discipline as
        ``band_config`` (jax-gcm#926 Codex review of PR #1096):
        ``ComposablePhysics + ComposablePhysics`` with two DIFFERENT
        settings has nothing to arbitrate by, so it must raise rather than
        silently keep whichever operand happens to be ``self``. Matching
        settings on both sides, or a setting on only one side, are both
        fine -- covered by ``test_matches_from_scratch_across_every_composition_path``
        passing with its default (matching, unset-on-the-bare-term) settings.
        """
        a = ComposablePhysics(terms=[QuadraticMoistening()],
                               dt_seconds=900.0, vectorize_columns=True)
        b = ComposablePhysics(terms=[QuadraticMoistening(beta=0.0)],
                               dt_seconds=1800.0, vectorize_columns=False)
        with self.assertRaisesRegex(
                ValueError, "checkpoint_terms/vectorize_columns/dt_seconds"):
            a + b

    def test_add_forwards_the_compositions_settings_regardless_of_operand_order(self):
        """A bare term added to a composition must carry the COMPOSITION's
        ``checkpoint_terms``/``vectorize_columns``/``dt_seconds`` through,
        not silently fall back to ``ComposablePhysics.__init__``'s own
        defaults -- the exact regression a Codex review of the jax-gcm#926
        fix (PR #1096) found in ``PhysicsTerm.__add__`` (``term + composable``
        discarded ``composable.dt_seconds``) and in
        ``ComposablePhysics.__add__`` (``composable + composable`` always
        kept ``self``'s, even when ``other``'s differed).
        """
        composition = ComposablePhysics(
            terms=[ToyRadiation(radiation_interval=1800.0)],
            dt_seconds=900.0, vectorize_columns=True, checkpoint_terms=False)

        for result, shape in (
            (composition + QuadraticMoistening(), "composable_plus_term"),
            (QuadraticMoistening() + composition, "term_plus_composable"),
        ):
            with self.subTest(shape=shape):
                self.assertEqual(result.dt_seconds, 900.0)
                self.assertTrue(result.vectorize_columns)
                self.assertFalse(result.checkpoint_terms)

    def test_replace_radiation_untouched_when_no_gated_sibling(self):
        """No gated sibling -> no ``RadiationParameters`` requirement at all:
        a plain radiation swap with nothing reading its cadence must not
        start demanding ``.params`` just because the category is
        "radiation" — only a present gated sibling creates that requirement.
        """
        physics = ComposablePhysics(terms=[ToyRadiation(), QuadraticMoistening()])
        replaced = physics.replace("radiation", RadiationWithNoParams())  # no raise
        self.assertIsInstance(
            next(t for t in replaced.terms if t.category == "radiation"),
            RadiationWithNoParams,
        )

    def test_replace_radiation_does_not_mutate_the_original_composition(self):
        """Building a replaced variant must leave the SOURCE composition's
        gated sibling's cadence untouched: it is carried over BY REFERENCE
        into the new term list (``replace`` only rebuilds the radiation
        slot), so reconfiguring it in place would silently change the
        cadence ``physics`` itself — still live, still usable — sees.
        """
        gated = GatedSibling()
        gated.configure_radiation_gate(7200.0)  # as echam_physics() would at build time
        physics = ComposablePhysics(
            terms=[ToyRadiation(radiation_interval=7200.0), gated])

        physics.replace("radiation", ToyRadiation(radiation_interval=3600.0))

        (original_gated,) = (t for t in physics.terms if isinstance(t, GatedSibling))
        self.assertIs(original_gated, gated)
        self.assertEqual(gated._radiation_interval_s, 7200.0)

    def test_clone_does_not_deep_copy_plain_mutable_attributes(self):
        """Pin the EXACT scope of the ownership-cloning guarantee (local
        /code-review high finding on jax-gcm#926's final design, PR #1096).

        ``nnx.clone`` gives each clone its own ``nnx.Variable`` wrapper
        objects (verified: ``clone.w is not orig.w`` for an ``nnx.Param``),
        which is sufficient for every current ``cache_band_config``/
        ``configure_radiation_gate`` implementation, since all of them
        only ever REASSIGN plain scalar/array attributes
        (``self._n_bnd_sw = ...``, ``self._radiation_interval_s = ...``).
        It does NOT deep-copy a plain Python list/dict attribute that is
        mutated IN PLACE (``.append()``, ``[key] = value``) rather than
        reassigned: ``GatedSibling.configured_with`` (a plain list, not an
        ``nnx.Variable``) stays the SAME object on both clones, so
        appending through one clone is visible through the other -- a
        future term that accumulates state this way in
        ``cache_band_config``/``configure_radiation_gate`` would
        reintroduce the exact cross-composition aliasing this PR's
        ownership invariant otherwise rules out. No in-tree term does
        this today (verified by inspection of every production
        ``cache_band_config``/``configure_radiation_gate`` override); this
        test exists so a future one that does gets a specific, named
        behaviour to check itself against rather than rediscovering it the
        hard way.
        """
        original = GatedSibling()
        clone = nnx.clone(original)
        self.assertIsNot(clone, original)

        # Reassignment (what every current production implementation
        # does) is correctly isolated...
        clone.configure_radiation_gate(1800.0)
        self.assertEqual(clone._radiation_interval_s, 1800.0)
        self.assertIsNone(original._radiation_interval_s)

        # ...but the plain list `configure_radiation_gate` ALSO appends to
        # (never read by any current consumer -- it exists in this toy
        # purely as the "was this reconfigured" test instrumentation
        # described in `GatedSibling`'s own docstring) is the SAME object
        # on both sides, so the append above is visible on `original` too.
        self.assertIs(clone.configured_with, original.configured_with)
        self.assertEqual(original.configured_with, [1800.0])

    def test_add_gated_only_clones_a_term_reused_across_two_compositions(self):
        """Adding the SAME gated-sibling object to two DIFFERENT compositions
        must not let the second composition's ``cache_coords()`` corrupt the
        first's cadence.

        Nothing stops a caller from holding one ``JamOpticsTerm``-like
        instance and adding it to more than one composition. jax-gcm#926's
        final design clones ``other``'s terms UNCONDITIONALLY in ``+``
        (ownership is the only invariant an operator itself enforces; see
        ``ComposablePhysics.__add__``'s docstring) — gated or not — so
        ``optics`` ends up in neither ``p1`` nor ``p2`` by reference, and
        each composition's OWN ``cache_coords()`` call configures its own
        clone from its own radiation term without touching the other's.
        """
        coords = _get_test_coords()
        optics = GatedSibling()
        p_a = ComposablePhysics(
            terms=[ToyRadiation(radiation_interval=7200.0), QuadraticMoistening()])
        p_b = ComposablePhysics(
            terms=[ToyRadiation(radiation_interval=900.0), QuadraticMoistening()])

        p1 = p_a + optics
        p2 = p_b + optics  # reuses the SAME `optics` object
        p1.cache_coords(coords)
        p2.cache_coords(coords)

        p1_optics = next(t for t in p1.terms if isinstance(t, GatedSibling))
        p2_optics = next(t for t in p2.terms if isinstance(t, GatedSibling))
        self.assertEqual(p1_optics._radiation_interval_s, 7200.0)
        self.assertEqual(p2_optics._radiation_interval_s, 900.0)
        self.assertIsNot(p1_optics, p2_optics)
        self.assertIsNot(p1_optics, optics)
        self.assertIsNot(p2_optics, optics)
        # The original, caller-held object is untouched by either `+` or
        # either composition's `cache_coords()`.
        self.assertIsNone(optics._radiation_interval_s)

    def test_replace_radiation_does_not_corrupt_a_shared_band_consumers_cache(self):
        """A non-gated ``cache_band_config`` consumer must also be cloned.

        ``Macv2SpAerosol`` caches its SW/LW band counts via
        ``cache_band_config`` (called by ``cache_coords``), not via
        ``configure_radiation_gate`` — it has no gate at all. jax-gcm#926's
        final design clones EVERY term ``replace`` carries over, not just
        the incoming one (``self`` remains an independently live, usable
        composition — see ``replace``'s docstring), so ``physics`` and
        ``replaced`` never share the SAME ``Macv2SpAerosol`` instance, no
        matter what either composition's ``cache_coords()`` later does to
        it.

        ``ToyRadiation`` is never a RECOGNIZED backend (see
        :meth:`RadiationBandConfig.is_recognized`), so swapping one
        ``ToyRadiation`` for another does not even touch the explicit
        ``band_config`` — it is kept VERBATIM by ``replace`` and never
        validated against an unrecognized term by ``_finalize`` either
        (that check only fires for a RECOGNIZED backend, where there is a
        real band structure to be wrong about — see
        ``test_add_never_aliases_a_band_consumer_with_the_operand_it_came_from``
        and ``echam_terms_test.py``'s ``test_replace_radiation_matches_the_factory_route*``
        for the RRTMGP-recognized case). So BOTH compositions end up
        wanting the SAME band_config here — the point of this test is that
        they never corrupt each other even so, because neither one's
        ``Macv2SpAerosol`` is the other's object.
        """
        from jcm.physics.aerosol.macv2_sp import Macv2SpAerosol

        class DiagnosticsProvider(ToyRadiation):
            """Stands in for the real terms that provide what Macv2SpAerosol needs."""

            name: ClassVar[str] = "diagnostics_provider"
            category: ClassVar[str] = "diagnostics"
            provides: ClassVar[tuple[str, ...]] = (
                "height_full", "layer_thickness")

        coords = _get_test_coords()
        custom_band_config = RadiationBandConfig(
            lw_band_centers_nm=(1000.0, 2000.0),
            sw_band_centers_nm=(400.0, 500.0, 600.0))
        physics = ComposablePhysics(
            terms=[DiagnosticsProvider(), Macv2SpAerosol(), ToyRadiation()],
            band_config=custom_band_config,
        )
        physics.cache_coords(coords)
        original_aerosol = physics.terms[1]
        self.assertEqual(original_aerosol._n_bnd_sw, 3)
        self.assertEqual(original_aerosol._n_bnd_lw, 2)

        replaced = physics.replace("radiation", ToyRadiation())
        replaced.cache_coords(coords)

        # The ORIGINAL composition's aerosol term is unaffected by anything
        # `replaced`'s own `cache_coords()` just did.
        self.assertEqual(original_aerosol._n_bnd_sw, 3)
        self.assertEqual(original_aerosol._n_bnd_lw, 2)
        self.assertEqual(len(physics.band_config.sw_band_centers_nm), 3)
        self.assertEqual(len(physics.band_config.lw_band_centers_nm), 2)

        replaced_aerosol = next(
            t for t in replaced.terms if isinstance(t, Macv2SpAerosol))
        self.assertIsNot(replaced_aerosol, original_aerosol)
        # The explicit override survives the swap verbatim (ToyRadiation is
        # unrecognized), so BOTH compositions want the same band counts --
        # on two DIFFERENT objects.
        self.assertEqual(replaced.band_config, custom_band_config)
        self.assertEqual(replaced_aerosol._n_bnd_sw, 3)
        self.assertEqual(replaced_aerosol._n_bnd_lw, 2)

    def test_raw_construction_with_a_recognized_backend_derives_correctly_first_try(self):
        """Raw construction with a REAL radiation term and no explicit
        override must get the CORRECT band structure on the FIRST
        ``cache_coords()`` call — no second operator call needed to "fix"
        a stale default (jax-gcm#926's final design eliminates the round-6
        failure mode entirely, rather than papering over it): ``band_config``
        defaults to ``None`` (pending) now, not an eager ``broadband()``
        ``ComposablePhysics.__init__`` used to assume regardless of what
        radiation term was actually present, so there is no stale value
        for a LATER operator call to have to correct in the first place.
        """
        from jcm.physics.aerosol.macv2_sp import Macv2SpAerosol
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

        class RRTMGPDiagnosticsProvider(ToyRadiation):
            """Supplies what both Macv2SpAerosol and RRTMGPRadiation need."""

            name: ClassVar[str] = "rrtmgp_diagnostics_provider"
            category: ClassVar[str] = "diagnostics"
            provides: ClassVar[tuple[str, ...]] = (
                "height_full", "layer_thickness", *RRTMGPRadiation.requires)

        coords = _get_test_coords()
        physics = ComposablePhysics(
            terms=[RRTMGPDiagnosticsProvider(), Macv2SpAerosol(), RRTMGPRadiation()])
        self.assertIsNone(physics.band_config)  # pending, not a stale default
        physics.cache_coords(coords)

        aerosol = physics.terms[1]
        self.assertGreater(len(physics.band_config.sw_band_centers_nm), 1)
        self.assertEqual(
            aerosol._n_bnd_sw, len(physics.band_config.sw_band_centers_nm))
        self.assertEqual(
            aerosol._n_bnd_lw, len(physics.band_config.lw_band_centers_nm))

    def test_band_config_rederives_after_an_edit_separated_by_cache_coords(self):
        """Extends ``test_matches_from_scratch_across_every_composition_path``
        to a dimension that matrix does not cover at all: every cell there
        calls ``cache_coords()`` exactly ONCE, after the editing operator,
        on a composition that has never resolved ``band_config`` before.
        That matrix also only ever swaps between TWO ``ToyRadiation``
        instances, which is never a RECOGNIZED backend (see
        ``RadiationBandConfig.is_recognized``) either before or after the
        edit — so it cannot distinguish "the new value is correctly
        re-derived" from "the old value just happened to still be right",
        since an unrecognized-to-unrecognized swap always re-derives to the
        SAME broadband default either way.

        This test instead calls ``cache_coords()`` on the SOURCE
        composition BEFORE a ``replace``/``remove``/``__add__`` edit, using
        a REAL recognized backend (``RRTMGPRadiation``) on (at least) one
        side of the swap, so a stale carried-over value is actually
        DIFFERENT from the correct one and a silent failure to re-derive is
        observable. This is exactly the GitHub Codex review finding against
        PR #1096: ``echam_physics().cache_coords(...)`` then
        ``.replace("radiation", GreyTwoStreamRadiation())`` then
        ``.cache_coords(...)`` carried RRTMGP's 14 SW/16 LW bands into the
        grey composition, because ``_finalize`` only re-derived an UNSET
        (``None``) ``band_config`` or validated an EXPLICIT one against a
        *recognized* new backend -- a composition's own, already-resolved,
        non-explicit value satisfied neither condition, so it was never
        touched (``ComposablePhysics.__init__``/``_finalize``'s docstrings
        cover the fix: a separate ``_band_config_explicit`` provenance flag,
        decoupled from whether the value happens to be ``None``).

        ``GreyTwoStreamRadiation`` itself is not constructed here (it needs
        a real coordinate-dependent setup beyond this toy harness) --
        ``ToyRadiation`` already stands in for it throughout this file
        (never a recognized backend, see its docstring), which is all
        ``RadiationBandConfig.is_recognized`` cares about.
        """
        from jcm.physics.aerosol.macv2_sp import Macv2SpAerosol
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

        class RRTMGPDiagnosticsProvider(ToyRadiation):
            """Supplies what both Macv2SpAerosol and RRTMGPRadiation need."""

            name: ClassVar[str] = "rrtmgp_diagnostics_provider"
            category: ClassVar[str] = "diagnostics"
            provides: ClassVar[tuple[str, ...]] = (
                "height_full", "layer_thickness", *RRTMGPRadiation.requires)

        coords = _get_test_coords()

        def rrtmgp_terms():
            return [RRTMGPDiagnosticsProvider(), Macv2SpAerosol(), RRTMGPRadiation()]

        def grey_terms():
            return [RRTMGPDiagnosticsProvider(), Macv2SpAerosol(),
                    ToyRadiation(radiation_interval=3600.0)]

        def band_shape(physics):
            return (len(physics.band_config.sw_band_centers_nm),
                    len(physics.band_config.lw_band_centers_nm))

        # -- replace(), both directions, each separated by a prior
        # -- cache_coords() on the SOURCE composition (the GitHub review's
        # -- exact repro shape).
        with self.subTest(shape="replace", direction="rrtmgp_to_grey"):
            physics = ComposablePhysics(
                terms=[*rrtmgp_terms(), GatedSibling()])
            physics.cache_coords(coords)  # resolves RRTMGP's real bands
            self.assertGreater(band_shape(physics)[0], 1)  # sanity: not broadband

            edited = physics.replace(
                "radiation", ToyRadiation(radiation_interval=1800.0))
            edited.cache_coords(coords)
            expected = ComposablePhysics(terms=[*grey_terms(), GatedSibling()])
            expected.cache_coords(coords)
            self.assertEqual(band_shape(edited), (1, 0))  # broadband, not 14/16
            self.assertEqual(edited.band_config, expected.band_config)
            edited_gate = next(
                t for t in edited.terms if isinstance(t, GatedSibling))
            self.assertEqual(edited_gate._radiation_interval_s, 1800.0)

        with self.subTest(shape="replace", direction="grey_to_rrtmgp"):
            physics = ComposablePhysics(
                terms=[*grey_terms(), GatedSibling()])
            physics.cache_coords(coords)  # resolves to broadband
            self.assertEqual(band_shape(physics), (1, 0))

            edited = physics.replace("radiation", RRTMGPRadiation())
            edited.cache_coords(coords)  # must NOT raise: not a real override
            expected = ComposablePhysics(terms=[*rrtmgp_terms(), GatedSibling()])
            expected.cache_coords(coords)
            self.assertGreater(band_shape(edited)[0], 1)  # real bands, not broadband
            self.assertEqual(edited.band_config, expected.band_config)
            edited_gate = next(
                t for t in edited.terms if isinstance(t, GatedSibling))
            default_rrtmgp_interval = next(
                t for t in expected.terms if isinstance(t, GatedSibling)
            )._radiation_interval_s
            self.assertEqual(edited_gate._radiation_interval_s,
                              default_rrtmgp_interval)

        # -- remove(): band_config must fall back to broadband, not stay
        # -- pinned at RRTMGP's bands (no gated sibling here -- `_finalize`
        # -- would raise with no radiation term left to drive its cadence).
        with self.subTest(shape="remove"):
            physics = ComposablePhysics(terms=rrtmgp_terms())
            physics.cache_coords(coords)
            self.assertGreater(band_shape(physics)[0], 1)

            edited = physics.remove("radiation")
            edited.cache_coords(coords)
            expected = ComposablePhysics(
                terms=[RRTMGPDiagnosticsProvider(), Macv2SpAerosol()])
            expected.cache_coords(coords)
            self.assertEqual(band_shape(edited), (1, 0))
            self.assertEqual(edited.band_config, expected.band_config)

        # -- __add__(): a no-radiation composition that has ALREADY
        # -- resolved (broadband) gains a real radiation term; the stale
        # -- broadband value must not survive the `+`.
        with self.subTest(shape="add"):
            physics = ComposablePhysics(
                terms=[RRTMGPDiagnosticsProvider(), Macv2SpAerosol()])
            physics.cache_coords(coords)
            self.assertEqual(band_shape(physics), (1, 0))

            edited = physics + RRTMGPRadiation()
            edited.cache_coords(coords)
            expected = ComposablePhysics(terms=rrtmgp_terms())
            expected.cache_coords(coords)
            self.assertGreater(band_shape(edited)[0], 1)
            self.assertEqual(edited.band_config, expected.band_config)

        # -- A genuinely EXPLICIT override must still survive an edit
        # -- unchanged, even across a prior cache_coords() call -- the
        # -- provenance flag must not over-correct and start re-deriving
        # -- real overrides too.
        with self.subTest(shape="replace", direction="explicit_override_survives"):
            custom = RadiationBandConfig(
                lw_band_centers_nm=(1000.0, 2000.0),
                sw_band_centers_nm=(400.0, 500.0, 600.0))
            physics = ComposablePhysics(
                terms=[ToyRadiation(radiation_interval=7200.0), GatedSibling()],
                band_config=custom)
            physics.cache_coords(coords)
            self.assertEqual(physics.band_config, custom)

            edited = physics.replace(
                "radiation", ToyRadiation(radiation_interval=1800.0))
            edited.cache_coords(coords)
            self.assertEqual(edited.band_config, custom)  # untouched
            edited_gate = next(
                t for t in edited.terms if isinstance(t, GatedSibling))
            self.assertEqual(edited_gate._radiation_interval_s, 1800.0)

    def test_operators_preserve_tied_parameters_across_terms(self):
        """Extends the jax-gcm#926 matrix
        (``test_matches_from_scratch_across_every_composition_path``) to an
        invariant it never exercised: two terms sharing ONE ``nnx.Variable``
        (a tied-parameter model) must still share ONE variable in the
        RESULT of every composition-editing operator, not two
        independently-cloned copies.

        Every operator used to clone the terms joining its result ONE AT A
        TIME — ``[nnx.clone(t) for t in terms]`` — which treats each term
        as its OWN graph root. ``nnx.clone`` only preserves cross-object
        aliasing WITHIN a single call's graph (verified directly by
        :func:`_clone_composition_terms`'s own docstring reasoning), so
        two terms that intentionally shared a variable each got their own,
        now-DIFFERENT copy of it — silently untying them, even on a
        complete no-op edit (``remove`` of a category nothing has). Fixed
        by routing every operator through
        :func:`~jcm.physics.composable_physics._clone_composition_terms`,
        which clones the WHOLE combined/surviving collection in ONE
        ``nnx.clone`` call (GitHub Codex review of PR #1096, inline
        thread on the old per-term clone loops in ``replace``/``remove``/
        both ``__add__`` implementations).
        """
        shared = nnx.Param(jnp.array(3.0))

        def assert_tied_and_isolated(result, original_a, original_b):
            term_a = next(
                t for t in result.terms if isinstance(t, TiedParamTermA))
            term_b = next(
                t for t in result.terms if isinstance(t, TiedParamTermB))
            # Still tied to EACH OTHER in the result...
            self.assertIs(term_a.shared_param, term_b.shared_param)
            # ...but isolated from the pre-edit objects' own variable (the
            # ownership invariant every operator already enforced -- this
            # test adds the "and the tie survives" half, not a replacement
            # for it).
            self.assertIsNot(term_a.shared_param, original_a.shared_param)
            self.assertIsNot(term_b.shared_param, original_b.shared_param)

        with self.subTest(shape="replace"):
            term_a, term_b = TiedParamTermA(shared), TiedParamTermB(shared)
            physics = ComposablePhysics(
                terms=[ToyRadiation(radiation_interval=7200.0), term_a, term_b])
            result = physics.replace(
                "radiation", ToyRadiation(radiation_interval=1800.0))
            assert_tied_and_isolated(result, term_a, term_b)

        with self.subTest(shape="remove"):
            term_a, term_b = TiedParamTermA(shared), TiedParamTermB(shared)
            physics = ComposablePhysics(
                terms=[ToyRadiation(radiation_interval=7200.0), term_a, term_b])
            # A TRUE no-op filter -- nothing has this category -- which is
            # exactly the shape the GitHub review's example used ("even a
            # no-op physics.remove('missing') untied a tied-parameter
            # model").
            result = physics.remove("category_nothing_has")
            assert_tied_and_isolated(result, term_a, term_b)

        with self.subTest(shape="composable_plus_composable"):
            term_a, term_b = TiedParamTermA(shared), TiedParamTermB(shared)
            left = ComposablePhysics(terms=[term_a])
            right = ComposablePhysics(terms=[term_b])
            result = left + right
            assert_tied_and_isolated(result, term_a, term_b)

        with self.subTest(shape="term_plus_composable"):
            term_a, term_b = TiedParamTermA(shared), TiedParamTermB(shared)
            right = ComposablePhysics(terms=[term_b])
            result = term_a + right
            assert_tied_and_isolated(result, term_a, term_b)

        with self.subTest(shape="composable_plus_term"):
            term_a, term_b = TiedParamTermA(shared), TiedParamTermB(shared)
            left = ComposablePhysics(terms=[term_a])
            result = left + term_b
            assert_tied_and_isolated(result, term_a, term_b)

    def test_term_plus_composable_preserves_operand_order(self):
        """``term + composable`` keeps THIS operator's left-to-right order
        (``self`` first) — unlike delegating to ``composable.__add__``,
        which would put ``self`` last.
        """
        composable = ComposablePhysics(terms=[ToyRadiation(), QuadraticMoistening()])
        combined = QuadraticMoistening(beta=9.0) + composable
        self.assertEqual(len(combined.terms), 3)
        self.assertIsInstance(combined.terms[0], QuadraticMoistening)
        self.assertEqual(float(combined.terms[0].beta[...]), 9.0)

    def test_add_never_aliases_a_band_consumer_with_the_operand_it_came_from(self):
        """A composition must OWN every ``cache_band_config`` consumer it
        holds, never alias one with another still-live composition
        (jax-gcm#926 Codex review, round 8; gated-sibling aliasing across
        two LIVE compositions is pinned separately by
        ``test_add_gated_only_clones_a_term_reused_across_two_compositions``).

        Unlike ``replace``/``remove`` (which now clone EVERY surviving term
        anyway — see ``test_replace_radiation_does_not_corrupt_a_shared_band_consumers_cache``),
        ``composable + composable``/``term + composable``/``composable +
        term`` combine TWO operands that both remain live and usable after
        the ``+`` — so a consumer term joining from either one, if not
        cloned, ends up owned by THREE compositions simultaneously (both
        operands AND the result), and ``cache_coords()`` on any one of them
        would silently rewrite what the others still expect. This is the
        round-8 trigger specifically: the radiation-owning side's EXPLICIT
        ``band_config`` survives the `+` unchanged (same VALUE as before),
        which is exactly the case the round-6 fix's ``band_config !=
        prior_band_config`` guard alone did not catch under the (now
        retired) choke-point design — jax-gcm#926's final design makes it
        moot anyway, since ``+`` clones EVERY term from ``other``
        unconditionally (see ``ComposablePhysics.__add__``'s docstring),
        never deciding based on whether a VALUE changed at all.

        (`remove` and bare `term + term` are not exercised here: `remove`
        only subtracts, so nothing can join from elsewhere, and
        `term_a + term_b` starts with no prior composition on either side,
        so there is no still-live operand for a consumer to be aliased
        with in the first place. The bare-radiation-term shapes --
        `ToyRadiation() + source`/`source + ToyRadiation()` -- used to be
        pinned HERE too, asserting the VALUE `band_config` re-derives to
        when the radiation-owning operand is a bare term with no
        `band_config` of its own; that assertion went stale under
        `_add_band_config`'s FINAL rule (jax-gcm#926, GitHub review of PR
        #1096: an explicit override now survives regardless of which side
        owns radiation, bare term or not) and is superseded by
        `test_add_band_config_merge_rule_matrix`'s dedicated coverage of
        that rule. Ownership for those two shapes is covered generically,
        for ANY pair of terms including a tied one, by
        `test_operators_preserve_tied_parameters_across_terms` -- this
        test keeps only the two shapes where BOTH operands are
        compositions, which is the shape unique to this test.)
        """
        from jcm.physics.aerosol.macv2_sp import Macv2SpAerosol

        class DiagnosticsProvider(ToyRadiation):
            """Supplies what Macv2SpAerosol needs to cache_coords cleanly."""

            name: ClassVar[str] = "diagnostics_provider"
            category: ClassVar[str] = "diagnostics"
            provides: ClassVar[tuple[str, ...]] = (
                "height_full", "layer_thickness")

        coords = _get_test_coords()
        # EXPLICIT, so it survives a `+` that does not touch the radiation
        # term unchanged -- the scenario the round-8 fix is specifically
        # about (band_config's VALUE does not change, but a consumer still
        # joins from the other operand).
        shared_band_config = RadiationBandConfig(
            lw_band_centers_nm=(1000.0, 2000.0),
            sw_band_centers_nm=(400.0, 500.0, 600.0))

        def make_source():
            """Build a live, already-``cache_coords()``-ed non-radiation
            composition carrying a cached ``Macv2SpAerosol`` -- the "other,
            still-live composition" a consumer term must never end up
            aliased with. No gated sibling here: a gated sibling needs a
            radiation term to validate against (see
            ``ComposablePhysics._finalize``), and `source` deliberately has
            none of its own.
            """
            source = ComposablePhysics(
                terms=[DiagnosticsProvider(), Macv2SpAerosol()],
                band_config=shared_band_config)
            source.cache_coords(coords)
            return source

        def make_radiation_side():
            return ComposablePhysics(
                terms=[ToyRadiation(radiation_interval=1800.0)],
                band_config=shared_band_config)

        def assert_ownership(result, source, result_n_bnd_sw, result_n_bnd_lw):
            result.cache_coords(coords)
            source_aerosol = next(
                t for t in source.terms if isinstance(t, Macv2SpAerosol))
            result_aerosol = next(
                t for t in result.terms if isinstance(t, Macv2SpAerosol))
            # No shared identity between the result and the still-live source.
            self.assertIsNot(result_aerosol, source_aerosol)
            # The source's own cached carry dimensions are bit-identical to
            # what they were BEFORE the `+` -- `result`'s own
            # `cache_coords()` call just above must not have reached them,
            # regardless of what `result`'s own band_config re-derives to
            # (see the two subTest groups below: `band_config`'s VALUE
            # staying the same is the round-8 trigger, re-deriving to a
            # DIFFERENT value is the round-6 one -- both must leave `source`
            # alone).
            self.assertEqual(source_aerosol._n_bnd_sw, 3)
            self.assertEqual(source_aerosol._n_bnd_lw, 2)
            # And the result's OWN clone is correctly configured from ITS
            # OWN band_config.
            self.assertEqual(result_aerosol._n_bnd_sw, result_n_bnd_sw)
            self.assertEqual(result_aerosol._n_bnd_lw, result_n_bnd_lw)

        # `source` and `make_radiation_side()` are BOTH explicit, with the
        # SAME value, so `_add_band_config`'s rule keeps it (matching
        # explicit overrides survive on both orders) -- the round-8
        # trigger (band_config's VALUE does not change, but the aerosol
        # term still joins from `source`, a still-live composition, and
        # must be cloned).
        with self.subTest(shape="composable_plus_composable"):
            source = make_source()
            result = source + make_radiation_side()
            assert_ownership(result, source, result_n_bnd_sw=3, result_n_bnd_lw=2)

        with self.subTest(shape="composable_plus_composable_reversed"):
            source = make_source()
            result = make_radiation_side() + source
            assert_ownership(result, source, result_n_bnd_sw=3, result_n_bnd_lw=2)


class TestOutputAttrs(unittest.TestCase):
    """``ComposablePhysics.output_attrs`` merges per-term declarations (#740)."""

    def test_merges_declarations_first_wins(self):
        """Both terms' vars appear; the first term wins a duplicated key."""

        class DeclaringA(PhysicsTerm):
            name: ClassVar[str] = "declaring_a"
            category: ClassVar[str] = "radiation"
            provides: ClassVar[tuple[str, ...]] = ("heating_rate",)
            output_attrs: ClassVar[dict[str, dict[str, str]]] = {
                "radiation.lw_flux_up": {"units": "W m-2"},
                "shared": {"units": "from_a"},
            }

            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), {
                    **diagnostics, "heating_rate": state.temperature}

        class DeclaringB(PhysicsTerm):
            name: ClassVar[str] = "declaring_b"
            category: ClassVar[str] = "convection"
            requires: ClassVar[tuple[str, ...]] = ("heating_rate",)
            output_attrs: ClassVar[dict[str, dict[str, str]]] = {
                "convection.precip": {"units": "kg m-2 s-1"},
                # Collides with DeclaringA — the FIRST term must win.
                "shared": {"units": "from_b"},
            }

            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = ComposablePhysics(terms=[DeclaringA(), DeclaringB()])
        merged = physics.output_attrs()
        self.assertEqual(merged["radiation.lw_flux_up"]["units"], "W m-2")
        self.assertEqual(merged["convection.precip"]["units"], "kg m-2 s-1")
        # First term (DeclaringA) wins the duplicated "shared" key.
        self.assertEqual(merged["shared"]["units"], "from_a")

    def test_terms_without_output_attrs_are_tolerated(self):
        """A term predating the attribute contributes no term-owned metadata."""
        # ``LinearHeating`` declares no ``output_attrs``; the base default {}
        # plus the ``getattr`` guard must keep it out of the merge cleanly.
        # Container-owned positivity diagnostics still carry their metadata.
        physics = ComposablePhysics(
            terms=[LinearHeating(), DiagnosticConsumer()])
        attrs = physics.output_attrs()
        self.assertEqual(
            attrs["water_positivity_correction.total_water_tendency"]["units"],
            "kg kg-1 s-1",
        )
        self.assertEqual(
            attrs["water_positivity_correction.column_water_source"]["units"],
            "kg m-2 s-1",
        )


class TestDifferentiabilityGate(unittest.TestCase):
    """Phase 2b gating tests: verify gradients flow through ComposablePhysics.

    These are the critical tests that must pass before any term-wrapping work
    begins. They verify that nnx.grad produces non-zero, correct gradients
    for parameters stored as nnx.Param on PhysicsTerm subclasses.
    """

    def _make_physics(self):
        return ComposablePhysics(
            terms=[
                LinearHeating(alpha=1.0),
                QuadraticMoistening(beta=0.5),
                DiagnosticConsumer(gamma=0.1),
            ],
            checkpoint_terms=False,  # disable checkpointing for grad tests
        )

    def test_nnx_grad_produces_nonzero_gradients(self):
        """nnx.grad through composable physics gives non-zero grads."""
        physics = self._make_physics()
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()

        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain)
            return (
                jnp.sum(tend.temperature ** 2)
                + jnp.sum(tend.specific_humidity ** 2)
                + jnp.sum(tend.u_wind ** 2)
            )

        grads = nnx.grad(loss_fn)(physics)

        # Check that each term's parameter has a non-zero gradient
        alpha_grad = grads.terms[0].alpha[...]
        beta_grad = grads.terms[1].beta[...]
        gamma_grad = grads.terms[2].gamma[...]

        self.assertFalse(jnp.isnan(alpha_grad), "alpha gradient is NaN")
        self.assertFalse(jnp.isnan(beta_grad), "beta gradient is NaN")
        self.assertFalse(jnp.isnan(gamma_grad), "gamma gradient is NaN")

        self.assertNotEqual(float(alpha_grad), 0.0, "alpha gradient is zero")
        self.assertNotEqual(float(beta_grad), 0.0, "beta gradient is zero")
        self.assertNotEqual(float(gamma_grad), 0.0, "gamma gradient is zero")

    def test_nnx_grad_correct_values(self):
        """Verify gradient values match hand-computed expectations for a simple case."""
        physics = ComposablePhysics(
            terms=[LinearHeating(alpha=1.0)],
            checkpoint_terms=False,
        )
        # Use a very simple state for easy hand computation
        shape = (1, 1, 1)
        temp_val = 3.0
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, temp_val),
        )
        forcing = ForcingData.zeros(shape[1:])
        terrain = _make_test_terrain(shape[1:])

        # loss = sum((alpha * T)^2) = alpha^2 * T^2 * n_elements
        # d(loss)/d(alpha) = 2 * alpha * T^2 * n_elements
        # With alpha=1.0, T=3.0, n=1: d(loss)/d(alpha) = 2 * 1 * 9 * 1 = 18.0
        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain)
            return jnp.sum(tend.temperature ** 2)

        grads = nnx.grad(loss_fn)(physics)
        expected_grad = 2.0 * 1.0 * temp_val ** 2
        npt.assert_allclose(
            grads.terms[0].alpha[...], expected_grad, rtol=1e-5
        )

    def test_jax_grad_via_split_merge(self):
        """Verify jax.grad works through nnx.split/merge (Pattern 2 from design doc)."""
        physics = ComposablePhysics(
            terms=[LinearHeating(alpha=2.0), QuadraticMoistening(beta=0.3)],
            checkpoint_terms=False,
        )
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()

        graphdef, param_state = nnx.split(physics)

        def loss_fn(param_state):
            physics_restored = nnx.merge(graphdef, param_state)
            tend, _ = physics_restored.compute_tendencies(
                state, forcing, terrain,
            )
            return jnp.sum(tend.temperature ** 2) + jnp.sum(tend.specific_humidity ** 2)

        grads = jax.grad(loss_fn)(param_state)

        # Verify non-zero gradients exist
        grad_leaves = jax.tree_util.tree_leaves(grads)
        for i, leaf in enumerate(grad_leaves):
            self.assertFalse(
                jnp.all(leaf == 0.0),
                f"Gradient leaf {i} is all zeros via split/merge path",
            )

    def test_per_term_gradient_addressing(self):
        """Verify per-term parameter addressing for optimization."""
        physics = self._make_physics()

        # Access individual term parameters directly
        self.assertIsInstance(physics.terms[0].alpha, nnx.Param)
        self.assertIsInstance(physics.terms[1].beta, nnx.Param)
        self.assertIsInstance(physics.terms[2].gamma, nnx.Param)

        # Verify we can get parameter values
        self.assertAlmostEqual(float(physics.terms[0].alpha[...]), 1.0, places=5)
        self.assertAlmostEqual(float(physics.terms[1].beta[...]), 0.5, places=5)
        self.assertAlmostEqual(float(physics.terms[2].gamma[...]), 0.1, places=5)

    def test_gradient_through_diagnostic_chain(self):
        """Gradients must flow through the diagnostics dict."""
        # DiagnosticConsumer reads heating_rate from LinearHeating.
        # A change in LinearHeating's alpha should affect DiagnosticConsumer's
        # output, so the gradient of the combined loss w.r.t. alpha should
        # include the contribution through the diagnostic chain.
        physics = ComposablePhysics(
            terms=[LinearHeating(alpha=1.0), DiagnosticConsumer(gamma=1.0)],
            checkpoint_terms=False,
        )

        shape = (1, 1, 1)
        temp_val = 2.0
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, temp_val),
        )
        forcing = ForcingData.zeros(shape[1:])
        terrain = _make_test_terrain(shape[1:])

        # Only look at u_wind loss (produced by DiagnosticConsumer)
        # u_wind_tend = gamma * heating_rate = gamma * alpha * T
        # loss = sum(u_wind_tend^2) = (gamma * alpha * T)^2
        # d(loss)/d(alpha) = 2 * gamma^2 * alpha * T^2
        # With gamma=1, alpha=1, T=2: d(loss)/d(alpha) = 2 * 1 * 1 * 4 = 8
        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain)
            return jnp.sum(tend.u_wind ** 2)

        grads = nnx.grad(loss_fn)(physics)
        expected_alpha_grad = 2.0 * 1.0 ** 2 * 1.0 * temp_val ** 2
        npt.assert_allclose(
            grads.terms[0].alpha[...], expected_alpha_grad, rtol=1e-5,
            err_msg="Gradient through diagnostic chain is incorrect",
        )

    def test_cached_coords_as_nnx_variable(self):
        """nnx.Variable for cached coords should be traced but not differentiated."""

        class CoordsAwareTerm(PhysicsTerm):
            name: ClassVar[str] = "coords_aware"
            category: ClassVar[str] = "test"

            def __init__(self, scale: float = 1.0):
                """Initialize CoordsAwareTerm."""
                self.scale = nnx.Param(jnp.array(scale))
                # Initialize with a placeholder; cache_coords will overwrite
                self.cached_value = nnx.Variable(jnp.array(0.0))

            def cache_coords(self, coords):
                # Overwrite with coordinate-derived value
                self.cached_value[...] = jnp.array(42.0)

            def __call__(self, state, diagnostics, forcing, terrain):
                # Use both the param and the cached value
                out = self.scale[...] * state.temperature + self.cached_value[...]
                tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
                    temperature=out,
                )
                return tendency, diagnostics

        term = CoordsAwareTerm(scale=1.0)
        term.cache_coords(None)  # populate cached_value

        physics = ComposablePhysics(terms=[term], checkpoint_terms=False)

        shape = (1, 1, 1)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 5.0),
        )
        forcing = ForcingData.zeros(shape[1:])
        terrain = _make_test_terrain(shape[1:])

        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain)
            return jnp.sum(tend.temperature ** 2)

        # This should work — nnx.grad differentiates Param but not Variable
        grads = nnx.grad(loss_fn)(physics)
        scale_grad = grads.terms[0].scale[...]
        self.assertFalse(jnp.isnan(scale_grad))
        self.assertNotEqual(float(scale_grad), 0.0)


class TestTendencyRunHostParity(unittest.TestCase):
    """``_tendency_run`` must carry the same contract on both hosts.

    A term that reconstructs the operator-split state from it (JAM's
    removal chain) would otherwise silently revert to step-start reads on
    whichever host does not publish it.
    """

    class _Recorder(PhysicsTerm):
        name = "recorder"
        category = "diagnostics"
        requires = ()
        provides = ()

        def __call__(self, state, diagnostics, forcing, terrain):
            run = diagnostics.get("_tendency_run")
            seen = (jnp.zeros_like(state.temperature) if run is None
                    else run["temperature"])
            return (PhysicsTendency.zeros(state.temperature.shape),
                    {**diagnostics, "seen_temperature_tendency": seen})

    def _run(self, vectorize):
        physics = ComposablePhysics(
            terms=[LinearHeating(alpha=2.0), self._Recorder()],
            checkpoint_terms=False,
            vectorize_columns=vectorize,
        )
        shape = (2, 4, 8)
        state = _make_test_state(shape)
        tend, diag = physics.compute_tendencies(
            state, _make_test_forcing(shape[1:]), _make_test_terrain(shape[1:]),
        )
        return np.asarray(diag["seen_temperature_tendency"]).reshape(-1)

    def test_both_hosts_publish_the_running_tendency(self):
        grid = self._run(vectorize=False)
        columns = self._run(vectorize=True)
        self.assertNotEqual(float(np.abs(grid).max()), 0.0)
        np.testing.assert_allclose(np.sort(grid), np.sort(columns), rtol=1e-6)


class TestColumnVectorization(unittest.TestCase):
    """Test ComposablePhysics with vectorize_columns=True."""

    def test_column_vectorization_produces_correct_shapes(self):
        """Column-vectorized physics reshapes 3D → columns → 3D correctly."""
        # A simple term that works on column format
        term = LinearHeating(alpha=2.0)
        physics = ComposablePhysics(
            terms=[term],
            checkpoint_terms=False,
            vectorize_columns=True,
        )

        shape = (2, 4, 8)
        state = _make_test_state(shape)
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])

        tend, diag = physics.compute_tendencies(state, forcing, terrain)

        # Output should be 3D again
        self.assertEqual(tend.temperature.shape, shape)
        self.assertEqual(tend.u_wind.shape, shape)

    def test_column_vectorization_with_tracers(self):
        """Column vectorization handles tracers correctly."""
        term = LinearHeating(alpha=1.0)
        physics = ComposablePhysics(
            terms=[term],
            checkpoint_terms=False,
            vectorize_columns=True,
        )

        shape = (2, 4, 8)
        key = jax.random.PRNGKey(0)
        state = PhysicsState(
            u_wind=jnp.zeros(shape),
            v_wind=jnp.zeros(shape),
            temperature=jnp.ones(shape) * 250.0,
            specific_humidity=jnp.zeros(shape),
            geopotential=jnp.zeros(shape),
            normalized_surface_pressure=jnp.ones(shape[1:]),
            tracers={"qc": jax.random.normal(key, shape)},
        )
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])

        tend, _ = physics.compute_tendencies(state, forcing, terrain)
        self.assertEqual(tend.temperature.shape, shape)

    def test_column_vectorization_with_prev_data(self):
        """Column vectorization carries forward prev_physics_data."""
        term = LinearHeating(alpha=1.0)
        physics = ComposablePhysics(
            terms=[term],
            checkpoint_terms=False,
            vectorize_columns=True,
        )

        shape = (2, 4, 8)
        state = _make_test_state(shape)
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])

        prev_data = {"_cached_value": jnp.array(42.0)}
        tend, diag = physics.compute_tendencies(
            state, forcing, terrain, prev_physics_data=prev_data,
        )
        # prev_data should be carried forward
        self.assertIn("_cached_value", diag)

    def test_column_vs_3d_numerically_equivalent(self):
        """Column-vectorized and 3D paths should produce same results for simple terms."""
        term_3d = LinearHeating(alpha=2.0)
        term_col = LinearHeating(alpha=2.0)

        physics_3d = ComposablePhysics(
            terms=[term_3d], checkpoint_terms=False, vectorize_columns=False,
        )
        physics_col = ComposablePhysics(
            terms=[term_col], checkpoint_terms=False, vectorize_columns=True,
        )

        shape = (2, 4, 8)
        state = _make_test_state(shape)
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])

        tend_3d, _ = physics_3d.compute_tendencies(state, forcing, terrain)
        tend_col, _ = physics_col.compute_tendencies(state, forcing, terrain)

        npt.assert_allclose(tend_3d.temperature, tend_col.temperature, rtol=1e-6)
        npt.assert_allclose(tend_3d.u_wind, tend_col.u_wind, rtol=1e-6)


class TestComposablePhysicsUtilities(unittest.TestCase):
    """Test get_empty_data and data_struct_to_dict."""

    def test_get_empty_data(self):
        """get_empty_data returns a zeroed diagnostics dict."""
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        physics = ComposablePhysics(
            terms=[LinearHeating(), QuadraticMoistening()],
        )
        physics.cache_coords(coords)
        empty = physics.get_empty_data(coords)
        self.assertIsInstance(empty, dict)
        self.assertIn("heating_rate", empty)
        self.assertIn("water_positivity_correction", empty)
        self.assertIn(
            "total_water_tendency", empty["water_positivity_correction"],
        )
        self.assertNotIn(
            "column_water_source", empty["water_positivity_correction"],
        )
        # Array values should be zeros
        for v in empty.values():
            if isinstance(v, jax.Array) and v.shape:
                self.assertTrue(jnp.all(v == 0.0))

    def test_data_struct_to_dict_multichannel_5d(self):
        """data_struct_to_dict expands 5D multi-channel fields."""
        physics = ComposablePhysics(terms=[LinearHeating()])
        nodal_shape = (4, 8)
        # 5D: s[1:-1] == nodal_shape → should expand on trailing dim
        struct = {
            "flux": jnp.zeros((2, 4, 8, 3)),  # 4D, doesn't match
            "flux5d": jnp.zeros((1, 2, 4, 8, 3)),  # 5D, s[1:-1]=(2,4,8)!=nodal
        }
        result = physics.data_struct_to_dict(struct, nodal_shape=nodal_shape)
        # Neither should expand since shapes don't match the pattern
        self.assertIn("flux", result)

    def test_data_struct_to_dict_unflattens_column_axis(self):
        """A flattened ``ncols`` axis is reshaped back to ``(lon, lat)``.

        ``vectorize_columns=True`` produces diagnostics with a single merged
        ``ncols`` axis; ``data_struct_to_dict`` must restore the spatial pair
        before xarray sees them, and leave arrays without an ncols axis (or
        with an ambiguous double match) untouched.
        """
        physics = ComposablePhysics(terms=[LinearHeating()], vectorize_columns=True)
        nodal_shape = (4, 4)          # ncols = 16
        struct = {
            "colvar": jnp.arange(47 * 16, dtype=jnp.float32).reshape(47, 16),
            "noncol": jnp.ones((47, 5)),          # no ncols axis -> untouched
            "ambiguous": jnp.ones((16, 16)),      # two ncols axes -> untouched
            "chan": jnp.ones((4, 4, 3)),          # 3-D multichannel -> expanded
        }
        result = physics.data_struct_to_dict(struct, nodal_shape=nodal_shape)
        self.assertEqual(result["colvar"].shape, (47, 4, 4))
        self.assertEqual(result["noncol"].shape, (47, 5))
        self.assertEqual(result["ambiguous"].shape, (16, 16))
        # 3-D field whose leading dims match nodal_shape expands per channel.
        self.assertIn("chan.0", result)
        self.assertIn("chan.2", result)
        self.assertNotIn("chan", result)

    def test_data_struct_to_dict_none_returns_empty(self):
        physics = ComposablePhysics(terms=[LinearHeating()])
        self.assertEqual(physics.data_struct_to_dict(None), {})

    def test_data_struct_to_dict_nondict_and_substruct(self):
        """Non-dict input and ``_``-prefixed typed sub-structs.

        A non-dict struct routes to the base recursive flattener; a typed
        sub-struct stashed under ``_<name>`` is flattened into ``name.field``
        user keys; a bare ``_`` key (empty user name) is dropped.
        """
        physics = ComposablePhysics(terms=[LinearHeating()])
        nodal_shape = (4, 8)

        # Non-dict struct -> base recursive flattener (fields must be arrays).
        ns = types.SimpleNamespace(temperature=jnp.zeros((2, 4, 8)))
        flat = physics.data_struct_to_dict(ns, nodal_shape=nodal_shape)
        self.assertIn("temperature", flat)

        struct = {
            "_radiation": types.SimpleNamespace(flux=jnp.zeros((2, 4, 8))),
            "_": jnp.zeros((2, 4, 8)),   # empty user key -> dropped
            "plain": jnp.zeros((2, 4, 8)),
        }
        out = physics.data_struct_to_dict(struct, nodal_shape=nodal_shape)
        self.assertIn("radiation.flux", out)   # sub-struct flattened
        self.assertIn("plain", out)
        self.assertNotIn("_", out)
        self.assertNotIn("", out)

    def test_composition_operators(self):
        """``+``, ``sum()`` and the unsupported-operand paths."""
        a = ComposablePhysics(terms=[LinearHeating()])
        b = ComposablePhysics(terms=[QuadraticMoistening()])
        # __add__ of two composables concatenates terms.
        self.assertEqual(len(a.__add__(b).terms), 2)
        # __add__ of a composable and a bare term.
        self.assertEqual(len(a.__add__(QuadraticMoistening()).terms), 2)
        # sum([...]) starts from 0, exercising __radd__'s ``0 + self`` branch.
        self.assertEqual(len(sum([a, b]).terms), 2)
        # Unsupported operands return NotImplemented (Python raises TypeError).
        with self.assertRaises(TypeError):
            _ = a + 5

    def test_data_struct_to_dict_non_array_values(self):
        """Plain Python values (ints, strings) drop out of the user dict.

        Only jax.Array leaves and flattenable sub-structs make it into the
        xarray output — everything else is silently skipped.
        """
        physics = ComposablePhysics(terms=[LinearHeating()])
        struct = {"count": 42, "name": "test", "field": jnp.array([1.0, 2.0])}
        result = physics.data_struct_to_dict(struct, nodal_shape=(4, 8))
        self.assertNotIn("count", result)
        self.assertNotIn("name", result)
        self.assertIn("field", result)

    def test_data_struct_to_dict_filters_internal_keys(self):
        """Underscore-prefixed array keys are exposed without the underscore;
        underscore-prefixed plumbing keys (`_dt_seconds`, etc.) stay hidden.
        """
        physics = ComposablePhysics(terms=[LinearHeating()])
        struct = {
            "_internal": jnp.array(1.0),
            "public_key": jnp.array(2.0),
            "_dt_seconds": 1800.0,
        }
        result = physics.data_struct_to_dict(struct)
        self.assertIn("public_key", result)
        # Underscore-prefixed array key surfaces as the key without underscore.
        self.assertIn("internal", result)
        # `_dt_seconds` is plumbing — stays hidden.
        self.assertNotIn("_dt_seconds", result)
        self.assertNotIn("dt_seconds", result)
        # Original underscore key is not preserved.
        self.assertNotIn("_internal", result)

    def test_data_struct_to_dict_none(self):
        physics = ComposablePhysics(terms=[LinearHeating()])
        result = physics.data_struct_to_dict(None)
        self.assertEqual(result, {})

    def test_data_struct_to_dict_with_nodal_shape(self):
        physics = ComposablePhysics(terms=[LinearHeating()])
        nodal_shape = (4, 8)
        struct = {"temperature": jnp.zeros((1, 4, 8))}
        result = physics.data_struct_to_dict(struct, nodal_shape=nodal_shape)
        self.assertIn("temperature", result)

    def test_vectorize_columns_preserved_by_replace(self):
        physics = ComposablePhysics(
            terms=[LinearHeating(), QuadraticMoistening()],
            vectorize_columns=True,
        )
        replaced = physics.replace("radiation", LinearHeating(alpha=5.0))
        self.assertTrue(replaced.vectorize_columns)

    def test_vectorize_columns_preserved_by_remove(self):
        physics = ComposablePhysics(
            terms=[LinearHeating(), QuadraticMoistening()],
            vectorize_columns=True,
        )
        removed = physics.remove("convection")
        self.assertTrue(removed.vectorize_columns)

    def test_vectorize_columns_preserved_by_add(self):
        # Both sides agree (`vectorize_columns=True`), so there is nothing
        # for `_resolve_add_settings` to arbitrate -- this just pins that a
        # non-conflicting setting actually survives `+`. A genuine conflict
        # between two ComposablePhysics operands (jax-gcm#926 Codex review of
        # PR #1096: silently preferring `self`'s was the bug) now raises --
        # see `TestComposablePhysics.test_add_conflicting_composition_settings_raises` —
        # and a bare term contributing nothing to forward/preserve is
        # covered by `test_add_forwards_the_compositions_settings_regardless_of_operand_order`.
        a = ComposablePhysics(terms=[LinearHeating()], vectorize_columns=True)
        b = ComposablePhysics(terms=[QuadraticMoistening()], vectorize_columns=True)
        combined = a + b
        self.assertTrue(combined.vectorize_columns)


class TestInitialCarryState(unittest.TestCase):
    """Phase 0 scaffolding for operator-split physics (issue #471).

    ``PhysicsTerm.initial_carry_state(coords)`` and
    ``ComposablePhysics.initial_carry_state(coords)`` build the cross-step
    ``physics_state`` carry deterministically — no zero-state probing
    (which was the root-cause failure mode of ``get_empty_data`` on
    radiation, see PR #469 / issue #470).
    """

    def test_default_term_carry_is_empty(self):
        """Terms that don't override contribute nothing to the carry."""
        term = LinearHeating()
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.assertEqual(term.initial_carry_state(coords), {})

    def test_composable_carry_aggregates(self):
        """Carry-state slots from each term merge into a single dict."""
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)

        class CarryingTerm(PhysicsTerm):
            name: ClassVar[str] = "carrier"
            category: ClassVar[str] = "diagnostic"
            requires: ClassVar[tuple[str, ...]] = ()
            provides: ClassVar[tuple[str, ...]] = ()

            def __init__(self, slot_name: str, fill: float = 0.0):
                self._slot_name = slot_name
                self._fill = fill

            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

            def initial_carry_state(self, coords):
                return {self._slot_name: jnp.full((2, 3), self._fill)}

        physics = ComposablePhysics(
            terms=[CarryingTerm("_alpha", 1.0), CarryingTerm("_beta", 2.0)],
        )
        carry = physics.initial_carry_state(coords)
        self.assertIn("_alpha", carry)
        self.assertIn("_beta", carry)
        npt.assert_array_equal(carry["_alpha"], jnp.ones((2, 3)))
        npt.assert_array_equal(carry["_beta"], jnp.full((2, 3), 2.0))

    def test_composable_carry_conflict_raises(self):
        """Two terms writing the same slot key is a composition bug."""
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)

        class DuplicateCarrier(PhysicsTerm):
            name: ClassVar[str] = "dup"
            category: ClassVar[str] = "diagnostic"
            requires: ClassVar[tuple[str, ...]] = ()
            provides: ClassVar[tuple[str, ...]] = ()

            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

            def initial_carry_state(self, coords):
                return {"_collision": jnp.zeros(2)}

        physics = ComposablePhysics(
            terms=[DuplicateCarrier(), DuplicateCarrier()],
        )
        with self.assertRaises(ValueError):
            physics.initial_carry_state(coords)

    def test_initial_carry_state_no_zero_probe(self):
        """``initial_carry_state`` must NOT call ``__call__`` to build the carry.

        Probing the term loop with a zero ``PhysicsState`` is the
        regression path that produced 0/0 = NaN in radiation
        (`#470 <https://github.com/climate-analytics-lab/jax-gcm/issues/470>`_).
        Build the carry from ``coords`` directly instead.
        """
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        called: list[int] = []

        class TracerTerm(PhysicsTerm):
            name: ClassVar[str] = "tracer"
            category: ClassVar[str] = "diagnostic"
            requires: ClassVar[tuple[str, ...]] = ()
            provides: ClassVar[tuple[str, ...]] = ()

            def __call__(self, state, diagnostics, forcing, terrain):
                called.append(1)
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

            def initial_carry_state(self, coords):
                return {"_tracer_slot": jnp.zeros((coords.nodal_shape[0],))}

        physics = ComposablePhysics(terms=[TracerTerm()])
        _ = physics.initial_carry_state(coords)
        self.assertEqual(
            called, [],
            "initial_carry_state must not call term.__call__ — see #470",
        )

    def test_initial_carry_state_echam_keys_subset_of_get_empty_data(self):
        """For an ECHAM composition, every key produced by
        ``initial_carry_state`` matches a key in ``get_empty_data``
        with the same shape.

        This is the Phase 0 acceptance criterion (#471): the
        deterministic carry seed and the legacy zero-probe diagnostics
        seed agree on shape for keys that both produce. Keys only in
        ``get_empty_data`` are user-facing per-step diagnostics that
        the carry doesn't need to cover.
        """
        from jcm.utils import get_coords
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.testing import idealized_echam_physics

        coords = get_coords(get_echam_levels(47), spectral_truncation=31)
        physics = idealized_echam_physics(
            checkpoint_terms=False,
        )
        physics.cache_coords(coords)

        empty = physics.get_empty_data(coords)
        carry = physics.initial_carry_state(coords)
        for key in carry:
            self.assertIn(
                key, empty,
                f"initial_carry_state has slot {key!r} not in get_empty_data",
            )
            shapes_carry = jax.tree_util.tree_map(jnp.shape, carry[key])
            shapes_empty = jax.tree_util.tree_map(jnp.shape, empty[key])
            self.assertEqual(
                shapes_carry, shapes_empty,
                f"shape mismatch for {key!r}: "
                f"initial_carry_state {shapes_carry} vs "
                f"get_empty_data {shapes_empty}",
            )


class TestPackagesImport(unittest.TestCase):
    """Test packages/ factory re-exports."""

    def test_packages_speedy_import(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        self.assertTrue(callable(speedy_physics))

    def test_packages_echam_import(self):
        from jcm.physics.echam.echam_terms import echam_physics
        self.assertTrue(callable(echam_physics))


class TestTracerSpec(unittest.TestCase):
    """TracerSpec declaration + ComposablePhysics aggregation."""

    def test_default_required_tracers_is_empty(self):
        self.assertEqual(LinearHeating().required_tracers(), ())

    def test_aggregation_dedups_identical_specs(self):
        class A(PhysicsTerm):
            name: ClassVar[str] = "a"
            category: ClassVar[str] = "x"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qc"),)
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        class B(PhysicsTerm):
            name: ClassVar[str] = "b"
            category: ClassVar[str] = "y"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qc"), TracerSpec("qi"))
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = ComposablePhysics(terms=[A(), B()])
        names = {spec.name for spec in physics.required_tracers()}
        self.assertEqual(names, {"qc", "qi"})

    def test_conflicting_specs_raise(self):
        class A(PhysicsTerm):
            name: ClassVar[str] = "a"
            category: ClassVar[str] = "x"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qnc", nondimensionalize=False),)
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        class B(PhysicsTerm):
            name: ClassVar[str] = "b"
            category: ClassVar[str] = "y"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qnc", nondimensionalize=True),)
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = ComposablePhysics(terms=[A(), B()])
        with self.assertRaisesRegex(ValueError, "qnc"):
            physics.required_tracers()

    def test_nondimensionalize_flag_round_trip(self):
        """Tracers with nondimensionalize=False must round-trip unchanged through the converters."""
        from dinosaur import primitive_equations
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.dycore.dinosaur.state_bridge import (
            dynamics_state_to_physics_state,
            physics_state_to_dynamics_state,
        )

        coords = get_speedy_coords()
        # Constants-derived specs so coords.radius (sourced from PhysicalConstants
        # via get_coords) matches physics_specs.radius.
        from jcm.dycore.dinosaur.dycore import PHYSICS_SPECS as specs
        primitive = primitive_equations.PrimitiveEquations(
            reference_temperature=jnp.ones(coords.nodal_shape[0]) * 288.0,
            orography=jnp.zeros(coords.modal_shape[1:]),
            coords=coords,
            physics_specs=specs,
        )
        nodal_shape = coords.nodal_shape

        physics_state = PhysicsState.zeros(nodal_shape)
        physics_state = physics_state.copy(
            temperature=jnp.ones(nodal_shape) * 288.0,
            normalized_surface_pressure=jnp.ones(nodal_shape[1:]),
            tracers={"qnc": jnp.ones(nodal_shape) * 1e8},  # number per kg
        )

        tracer_specs = {"qnc": TracerSpec("qnc", nondimensionalize=False)}

        modal = physics_state_to_dynamics_state(
            physics_state, primitive, tracer_specs=tracer_specs,
        )
        back = dynamics_state_to_physics_state(
            modal, primitive, tracer_specs=tracer_specs,
        )

        # qnc should round-trip within spectral transform tolerance
        npt.assert_allclose(back.tracers["qnc"], physics_state.tracers["qnc"], rtol=1e-3)


class TestModelSeedsTracers(unittest.TestCase):
    """Model seeds the initial tracer dict from physics.required_tracers()."""

    def test_model_seeds_declared_tracers(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        class NeedsQC(PhysicsTerm):
            name: ClassVar[str] = "needs_qc"
            category: ClassVar[str] = "clouds"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qc"), TracerSpec("qnc", nondimensionalize=False))
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = ComposablePhysics(terms=[NeedsQC()])
        model = Model(coords=get_speedy_coords(), physics=physics, time_step=720)
        # _prepare_initial_dycore_state runs inside .run(), but we can call the
        # underlying prep directly to verify the tracer dict.
        state = model._prepare_initial_dycore_state()
        self.assertIn("specific_humidity", state.tracers)
        self.assertIn("qc", state.tracers)
        self.assertIn("qnc", state.tracers)
        # Semi-Lagrangian transport (what the physics-decided mode picks for
        # this tracer-carrying composition) splits the
        # tracer representations, so equal shapes are NOT the invariant:
        # ``specific_humidity`` stays MODAL because it participates in the
        # implicit q<->Tv coupling, while every declared extra tracer is
        # carried NODAL so it never round-trips through the spectral basis
        # (that round-trip is what rang sharp emission sources negative and
        # NaN'd the aerosol microphysics, #521). Pin both sides.
        coords = model.dycore.coords
        nodal = tuple(coords.horizontal.nodal_shape)
        modal = tuple(coords.horizontal.modal_shape)
        nlev = coords.vertical.layers
        for name in ("qc", "qnc"):
            with self.subTest(tracer=name):
                self.assertEqual(state.tracers[name].shape, (nlev,) + nodal)
        self.assertEqual(
            state.tracers["specific_humidity"].shape, (nlev,) + modal)


if __name__ == "__main__":
    unittest.main()


class TestWithheldOutputKeys:
    """A term can keep unpopulated struct fields out of the output.

    Regression for jax-gcm#647: ``RadiationData`` carries the four
    aerosol-free slots in every configuration, so with the diagnostic off
    they were published as zeros — and a downstream ERFari of
    ``rsut - rsutnoa`` then equals the whole all-sky flux, ~240 W/m2
    instead of ~-1, in a file that looks valid.
    """

    def test_rrtmgp_withholds_noa_only_when_the_diagnostic_is_off(self):
        from jcm.physics.radiation.aerosol_free import NOA_KEYS
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

        # Both the fluxes and the persisted noa_frac_* ratios: a published
        # zero fraction reads as "aerosol removes the entire flux" through
        # the fraction-based ERFari path (PR #730 review).
        expected = tuple(f"radiation.{k}_noa" for k in NOA_KEYS) + tuple(
            f"radiation.noa_frac_{k}" for k in NOA_KEYS
        )
        assert RRTMGPRadiation(
            aerosol_free_interval=None).withheld_output_keys() == expected
        for on in (1, 2, 4):
            assert RRTMGPRadiation(
                aerosol_free_interval=on).withheld_output_keys() == ()

    def test_rrtmgp_withholds_clear_sky_only_without_compute_cre(self):
        """Clear-sky slots are the same trap as the ``*noa`` ones.

        Without the second cloud-free solve they stay zero, which reads
        as a CRE equal to the entire all-sky flux.
        """
        from jcm.physics.radiation.radiation_types import CLEAR_SKY_KEYS
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

        # The clouds sub-struct mirrors of the clear-sky TOA fluxes are the
        # same zero placeholders without the companion solve (PR #730
        # review); the *_all mirrors are real and stay.
        clear = tuple(f"radiation.{k}" for k in CLEAR_SKY_KEYS) + (
            "clouds.toa_sw_up_clear", "clouds.toa_lw_up_clear",
        )
        off = RRTMGPRadiation(
            aerosol_free_interval=1, compute_cre=False,
        ).withheld_output_keys()
        assert off == clear
        assert RRTMGPRadiation(
            aerosol_free_interval=1, compute_cre=True,
        ).withheld_output_keys() == ()

    def test_composable_physics_drops_the_declared_keys(self):
        """The aggregated keys must actually be filtered from the output."""
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.radiation.radiation_types import RadiationData

        class _Withholder(PhysicsTerm):
            name = "withholder"
            category = "diagnostic"

            def withheld_output_keys(self):
                return ("radiation.toa_sw_up_noa",)

            def compute(self, state, diagnostics, forcing, dt):
                raise NotImplementedError

        diagnostics = {"radiation": RadiationData.zeros((4, 3), 5)}
        kept = ComposablePhysics(terms=[]).data_struct_to_dict(diagnostics)
        dropped = ComposablePhysics(
            terms=[_Withholder()]).data_struct_to_dict(diagnostics)

        assert "radiation.toa_sw_up_noa" in kept
        assert "radiation.toa_sw_up_noa" not in dropped
        # Only the declared key goes; its neighbours stay.
        assert "radiation.toa_lw_up_noa" in dropped
        assert "radiation.toa_sw_up" in dropped

    def test_held_fraction_is_never_published(self):
        """Internal carry state, not a diagnostic — hidden at every N."""
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.radiation.aerosol_free import NOA_KEYS
        from jcm.physics.radiation.radiation_types import RadiationData

        out = ComposablePhysics(terms=[]).data_struct_to_dict(
            {"radiation": RadiationData.zeros((4, 3), 5)})
        for key in NOA_KEYS:
            assert f"radiation.noa_frac_{key}" not in out, (
                "internal held fraction published")
