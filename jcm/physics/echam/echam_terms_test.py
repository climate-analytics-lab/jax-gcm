"""Tests for composable ECHAM physics (echam_terms.py).

Tests mixed-package composition, replacement of individual terms, and
roundtripping through nnx.split / nnx.merge.
"""

import unittest

import numpy as np
import jax
import pytest
import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.utils import get_coords
from jcm.physics.physics_term import PhysicsTerm


class DummyRadiationTerm(PhysicsTerm):
    """Minimal radiation term used to test ECHAM factory dispatch."""

    name = "dummy_radiation"
    category = "radiation"


class DummyConvectionTerm(PhysicsTerm):
    """Minimal non-radiation term used to test factory validation."""

    name = "dummy_convection"
    category = "convection"


def _make_echam_test_setup(nlev=8, nlat=64, nlon=32):
    """Create test setup matching ECHAM conventions."""
    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(nlat, nlon))
    terrain = TerrainData.aquaplanet(coords)
    forcing = ForcingData.zeros((nlat, nlon))

    shape_3d = (nlev, nlat, nlon)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    state = PhysicsState(
        temperature=250.0 + 20.0 * jax.random.normal(
            keys[0], shape_3d,
        ),
        specific_humidity=jnp.abs(
            3.0 * jax.random.normal(keys[1], shape_3d),
        ),
        u_wind=5.0 * jax.random.normal(keys[2], shape_3d),
        v_wind=5.0 * jax.random.normal(keys[3], shape_3d),
        geopotential=jnp.broadcast_to(
            jnp.linspace(50000, 0, nlev)[:, None, None],
            shape_3d,
        ),
        normalized_surface_pressure=(
            1.0
            + 0.01 * jax.random.normal(keys[4], (nlat, nlon))
        ),
        tracers={
            "qc": jnp.abs(
                1e-4 * jax.random.normal(keys[5], shape_3d),
            ),
            "qi": jnp.zeros(shape_3d),
        },
    )

    return coords, state, forcing, terrain


class TestEchamComposablePhysics(unittest.TestCase):
    """Test composable ECHAM physics wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.coords, self.state, self.forcing, self.terrain = (
            _make_echam_test_setup()
        )

    def test_echam_physics_factory(self):
        """echam_physics() creates composable physics with correct terms."""
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(checkpoint_terms=False)
        # Cloud fraction and microphysics are separate terms; the GWD
        # category split adds Hines + SSO (the simple-GWD scheme is kept
        # available but excluded from the default factory); the terminal
        # EchamSurfaceExchange publishes the #754 coupling struct.
        self.assertEqual(len(physics.terms), 13)
        categories = [t.category for t in physics.terms]
        self.assertIn("radiation", categories)
        self.assertIn("convection", categories)
        self.assertIn("surface", categories)
        self.assertIn("cloud_fraction", categories)
        self.assertIn("clouds", categories)
        self.assertIn("hines", categories)
        self.assertIn("sso", categories)
        self.assertNotIn("simple_gwd", categories)
        # Cloud fraction must precede microphysics so the microphysics term
        # can read the post-condensation qc/qi/cloud_fraction diagnostics.
        self.assertLess(
            categories.index("cloud_fraction"),
            categories.index("clouds"),
        )
        # Cloud fraction must also precede radiation so radiation sees the
        # current step's cloud field (matches ECHAM6's cov→rad ordering).
        self.assertLess(
            categories.index("cloud_fraction"),
            categories.index("radiation"),
        )
        # ECHAM physc ordering (radheat → vdiff → cucall → cloud): vertical
        # diffusion and the surface term that republishes its delivered
        # fluxes must precede convection, so the Tiedtke zdqpbl closure
        # reads the SAME-STEP vdiff moisture tendency and evaporation. A
        # convection-first ordering forces a one-step-lagged supply, which
        # compounds the convergence→convection feedback (onset7 NaN).
        self.assertLess(
            categories.index("radiation"),
            categories.index("vertical_diffusion"),
        )
        self.assertLess(
            categories.index("vertical_diffusion"),
            categories.index("surface"),
        )
        self.assertLess(
            categories.index("surface"),
            categories.index("convection"),
        )
        self.assertLess(
            categories.index("convection"),
            categories.index("clouds"),
        )

    def test_cu_lmfmid_flag_toggles_the_omega_requirement(self):
        """The scalar cu_lmfmid knob controls the dycore omega contract.

        With the mid-level trigger on (the default) TiedtkeConvection
        declares an ``omega`` dycore requirement, which fails Model
        construction on a backend that cannot supply omega (pySES, #698).
        Setting cu_lmfmid=False drops that requirement so the ne30
        experiments compose (#715).
        """
        from jcm.physics.echam.echam_terms import echam_physics

        on = echam_physics(checkpoint_terms=False)
        self.assertIn("omega", on.required_dycore_fields())

        off = echam_physics(checkpoint_terms=False, cu_lmfmid=False)
        self.assertNotIn("omega", off.required_dycore_fields())

    def test_updraft_precip_cover_follows_the_ham_submodel(self):
        """The sub-cloud rain-evaporation cover tracks the JAM chain (#812).

        ECHAM keys ``zcucov`` on ``lham`` (mo_cufluxdts.f90:414-420); jcm's
        ``lham`` is "the JAM aerosol chain is composed", so the convection
        term takes the updraft-area cover for a JAM run and the constant 0.05
        otherwise. An explicit ``convective_updraft_precip_cover`` pins it.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            TiedtkeConvection,
        )

        def cover_flag(physics):
            (conv,) = (t for t in physics.terms
                       if isinstance(t, TiedtkeConvection))
            return conv._updraft_precip_cover

        # Non-JAM ECHAM: the constant 0.05 (flag off).
        self.assertFalse(cover_flag(echam_physics(checkpoint_terms=False)))
        # Pinned on without JAM (the A/B escape hatch).
        self.assertTrue(cover_flag(echam_physics(
            checkpoint_terms=False, convective_updraft_precip_cover=True)))
        # A JAM run defaults the flag on; pinning it off recovers 0.05. Build
        # only the convection term list cheaply via the placeholder core.
        jam_on = echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            jam_microphysics="placeholder")
        self.assertTrue(cover_flag(jam_on))
        jam_off = echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            jam_microphysics="placeholder",
            convective_updraft_precip_cover=False)
        self.assertFalse(cover_flag(jam_off))

    def test_jam_wetdep_scheme_wires_precip_cover_and_the_term_flag(self):
        """``jam_wetdep_scheme="ham_below_cloud"`` (#1017) turns on the 2M
        scheme's ``precip_cover``/``pfrain``/``pfsnow`` publication and the
        wetdep term's own selector together; the default leaves both off/"jcm".
        """
        from jcm.physics.aerosol.jam.wetdep.wetdep_term import WetScavenging
        from jcm.physics.clouds.lohmann_2m import Lohmann2MMicrophysics
        from jcm.physics.echam.echam_terms import echam_physics

        def parts(physics):
            micro = next(t for t in physics.terms
                        if isinstance(t, Lohmann2MMicrophysics))
            wetdep = next(t for t in physics.terms
                          if isinstance(t, WetScavenging))
            return micro, wetdep

        default = echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            jam_microphysics="placeholder")
        micro, wetdep = parts(default)
        self.assertFalse(micro._publish_wetdep_hydro)
        for key in ("precip_cover", "pfrain", "pfsnow"):
            self.assertNotIn(key, micro.provides)
        self.assertEqual(wetdep.scheme, "jcm")
        self.assertNotIn("precip_cover", wetdep.requires)

        ham_bc = echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            jam_microphysics="placeholder", jam_wetdep_scheme="ham_below_cloud")
        micro, wetdep = parts(ham_bc)
        self.assertTrue(micro._publish_wetdep_hydro)
        for key in ("precip_cover", "pfrain", "pfsnow"):
            self.assertIn(key, micro.provides)
        self.assertEqual(wetdep.scheme, "ham_below_cloud")
        self.assertIn("precip_cover", wetdep.requires)

    def test_jam_wetdep_scheme_ham_below_cloud_requires_2m(self):
        """A clear, scheme-specific error, not the generic ordering-validator
        one that would otherwise fire deep inside ComposablePhysics.
        """
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaises(ValueError):
            echam_physics(
                checkpoint_terms=False, aerosol_module="jam", cloud_scheme="1m",
                jam_microphysics="placeholder", jam_wetdep_scheme="ham_below_cloud")

    def test_jam_wetdep_scheme_ham_wires_through(self):
        """``jam_wetdep_scheme="ham"`` (#1017, all three slices) turns on
        the 2M hydro diagnostics -- "ham_below_cloud"'s three PLUS
        "reffl"/"reffi" for follow-up B's in-cloud impaction -- and also
        sets the wetdep term's scheme + nucleation_activation.
        """
        from jcm.physics.aerosol.jam.wetdep.wetdep_term import WetScavenging
        from jcm.physics.clouds.lohmann_2m import Lohmann2MMicrophysics
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            jam_microphysics="placeholder", jam_wetdep_scheme="ham",
            jam_nucleation_activation="ham_lin_leaitch")
        micro = next(t for t in physics.terms if isinstance(t, Lohmann2MMicrophysics))
        wetdep = next(t for t in physics.terms if isinstance(t, WetScavenging))
        self.assertTrue(micro._publish_wetdep_hydro)
        self.assertEqual(wetdep.scheme, "ham")
        self.assertEqual(wetdep._nucleation_activation, "ham_lin_leaitch")
        for key in ("precip_cover", "pfrain", "pfsnow", "reffl", "reffi"):
            self.assertIn(key, wetdep.requires)
            self.assertIn(key, micro.provides)

    def test_jam_wetdep_scheme_ham_requires_2m(self):
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaises(ValueError):
            echam_physics(
                checkpoint_terms=False, aerosol_module="jam", cloud_scheme="1m",
                jam_microphysics="placeholder", jam_wetdep_scheme="ham")

    def test_jam_takes_ham_ice_inhomogeneity(self):
        """JAM (2M + ARG) defaults to ECHAM-HAM's ``zinhomi = 0.7``; every
        other stack keeps ECHAM6's 0.8, and an explicit override wins.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.radiation_types import RadiationParameters

        def zinhomi(physics):
            rad = next(t for t in physics.terms if t.category == "radiation")
            return float(rad.params.get_value().cloud_inhomogeneity_ice)

        jam = dict(checkpoint_terms=False, aerosol_module="jam",
                   cloud_scheme="2m", jam_microphysics="placeholder")
        self.assertAlmostEqual(zinhomi(echam_physics(**jam)), 0.7, places=6)
        self.assertAlmostEqual(
            zinhomi(echam_physics(checkpoint_terms=False)), 0.8, places=6)
        self.assertAlmostEqual(zinhomi(echam_physics(
            checkpoint_terms=False, cloud_scheme="2m")), 0.8, places=6)
        explicit = RadiationParameters.default(cloud_inhomogeneity_ice=0.9)
        self.assertAlmostEqual(
            zinhomi(echam_physics(**jam, radiation=explicit)), 0.9, places=6)

    def test_cu_lmfmid_rejects_a_simultaneous_convection_override(self):
        """cu_lmfmid and an explicit convection Parameters are exclusive."""
        from jcm.physics.convection.tiedtke_nordeng import ConvectionParameters
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            echam_physics(
                checkpoint_terms=False,
                cu_lmfmid=False,
                convection=ConvectionParameters.default(),
            )

    def test_field_override_mapping_for_a_scheme_not_composed_is_rejected(self):
        """A mapping the composition would ignore is an error (#933)."""
        from jcm.physics.echam.echam_terms import echam_physics

        for kwargs, name in (
                (dict(radiation_scheme=DummyRadiationTerm(),
                      radiation={"solar_constant": 1360.0}), "radiation"),
                (dict(gw_scheme="none", hines={"rmscon": 1.0}), "hines"),
                (dict(cloud_scheme="1m", microphysics_2m={"ccraut": 5.0}),
                 "microphysics_2m"),
                (dict(aerosol_module="jam", cloud_scheme="2m",
                      jam_microphysics="placeholder",
                      aerosol={"spa_exponent": 0.4}), "aerosol")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                        ValueError, rf"\['{name}'\] would be ignored"):
                    echam_physics(checkpoint_terms=False, **kwargs)

    def test_jam_natural_emission_parameters_reach_their_terms(self):
        """``seasalt`` / ``dms`` set the JAM emission terms' parameters.

        A mapping is applied on top of the class default and keeps the
        fields it does not name; a Parameters object is used as given; with
        neither, the terms keep the defaults.
        """
        from jcm.physics.aerosol.jam.emissions.dms import DmsParameters
        from jcm.physics.aerosol.jam.emissions.seasalt import SeaSaltParameters
        from jcm.physics.echam.echam_terms import echam_physics

        jam = dict(checkpoint_terms=False, aerosol_module="jam",
                   cloud_scheme="2m", jam_microphysics="placeholder")

        def emission_params(physics):
            by_name = {t.name: t.params.get_value() for t in physics.terms
                       if t.name in ("jam_seasalt_emissions",
                                     "jam_dms_emissions")}
            return by_name["jam_seasalt_emissions"], by_name["jam_dms_emissions"]

        default_ss, default_dms = SeaSaltParameters.default(), DmsParameters.default()
        ss, dms = emission_params(echam_physics(**jam))
        self.assertEqual(float(ss.scale), float(default_ss.scale))
        self.assertEqual(float(dms.flux_scale), float(default_dms.flux_scale))

        # Mappings: only the named field moves.
        ss, dms = emission_params(echam_physics(
            **jam, seasalt={"scale": 1.7}, dms={"flux_scale": 0.6}))
        self.assertAlmostEqual(float(ss.scale), 1.7)
        self.assertEqual(float(ss.wind_exponent),
                         float(default_ss.wind_exponent))
        self.assertAlmostEqual(float(dms.flux_scale), 0.6)
        # Each mapping reaches its own term only.
        ss, dms = emission_params(echam_physics(**jam, dms={"flux_scale": 0.6}))
        self.assertEqual(float(ss.scale), float(default_ss.scale))
        ss, dms = emission_params(echam_physics(**jam, seasalt={"scale": 1.7}))
        self.assertEqual(float(dms.flux_scale), float(default_dms.flux_scale))

        # Objects are used as given (the Python-API form of the same door).
        ss, dms = emission_params(echam_physics(
            **jam, seasalt=SeaSaltParameters(
                scale=2.5, wind_exponent=3.0),
            dms=DmsParameters(flux_scale=0.4)))
        self.assertAlmostEqual(float(ss.scale), 2.5)
        self.assertAlmostEqual(float(ss.wind_exponent), 3.0)
        self.assertAlmostEqual(float(dms.flux_scale), 0.4)

    def test_jam_natural_emission_override_rejects_unknown_fields(self):
        """A typo'd field is an error naming the scheme and the valid fields."""
        from jcm.physics.echam.echam_terms import echam_physics

        jam = dict(checkpoint_terms=False, aerosol_module="jam",
                   cloud_scheme="2m", jam_microphysics="placeholder")
        with self.assertRaisesRegex(
                ValueError, r"seasalt: unknown SeaSaltParameters field\(s\) "
                r"\['scal'\].*Valid fields: .*'scale'"):
            echam_physics(**jam, seasalt={"scal": 1.7})
        with self.assertRaisesRegex(
                ValueError, r"dms: unknown DmsParameters field\(s\) "
                r"\['flux_scal'\].*Valid fields: \['flux_scale'\]"):
            echam_physics(**jam, dms={"flux_scal": 0.6})

    def test_jam_natural_emission_parameters_without_jam_are_rejected(self):
        """The emission parameters need the JAM chain, mapping or object.

        MACv2-SP composes no sea-salt or DMS emission, so either form would
        be dropped without a trace.
        """
        from jcm.physics.aerosol.jam.emissions.dms import DmsParameters
        from jcm.physics.aerosol.jam.emissions.seasalt import SeaSaltParameters
        from jcm.physics.echam.echam_terms import echam_physics

        for kwargs, names in (
                (dict(seasalt={"scale": 1.7}), "seasalt"),
                (dict(dms={"flux_scale": 0.6}), "dms"),
                (dict(seasalt=SeaSaltParameters.default()), "seasalt"),
                (dict(dms=DmsParameters.default()), "dms"),
                (dict(seasalt={"scale": 1.7}, dms={"flux_scale": 0.6}),
                 "seasalt', 'dms")):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                        ValueError, rf"\['{names}'\] would be ignored"):
                    echam_physics(checkpoint_terms=False, **kwargs)

    #: The JAM process schemes behind ``echam_physics``'s mapping door beyond
    #: sea salt and DMS: argument -> (term name, one field, a value for it).
    _JAM_DOORS = {
        "anthropogenic_params": ("jam_anthropogenic_emissions", "scale", 1.2),
        "oxidants": ("jam_prescribed_oxidants", "oh_ref", 3.0e6),
        "sulfur_gas": ("jam_sulfur_gas_chemistry", "soag_production", 1.0e-16),
        "aqueous": ("jam_aqueous_sulfur", "rate_scale", 0.8),
        "activation": ("arg_activation", "w_min", 0.2),
        "cloud_borne_exchange": ("jam_cloud_borne_exchange",
                                 "activation_timescale", 600.0),
        "sedimentation": ("jam_sedimentation", "velocity_scale", 0.9),
        "drydep": ("jam_dry_deposition", "z0", 2.0e-4),
        "wetdep": ("jam_wet_deposition", "incloud_scale", 0.5),
        "tracer_diffusion": ("tracer_vertical_diffusion", "diffusion_scale",
                             0.7),
        "conv_transport": ("convective_tracer_transport", "conv_scav_scale",
                           0.5),
    }
    #: Doors whose parameters are held by more than one term: argument ->
    #: the other terms that must carry the same value.
    _JAM_ALSO = {"tracer_diffusion": ("jam_cloud_borne_store",)}
    _JAM_KWARGS = dict(checkpoint_terms=False, aerosol_module="jam",
                       cloud_scheme="2m", jam_microphysics="placeholder",
                       jam_anthropogenic=True)

    @staticmethod
    def _jam_term_params(physics, term):
        return next(t for t in physics.terms
                    if t.name == term).params.get_value()

    def test_jam_door_table_covers_the_factory_arguments(self):
        """The door lists every JAM scheme the JAM factory takes ``Parameters`` for.

        A scheme added to ``jam_aerosol_physics`` must either join
        ``JAM_PARAMETER_CLASSES`` (and so gain the door) or be named here as
        one without a mapping door yet (#995); it cannot be left out silently.
        """
        import inspect
        import typing

        from jcm.physics.aerosol.jam.jam_terms import (
            JAM_PARAMETER_CLASSES, jam_aerosol_physics)
        from jcm.physics.echam.echam_terms import echam_physics

        hints = typing.get_type_hints(jam_aerosol_physics)
        parameter_args = {
            name for name in inspect.signature(jam_aerosol_physics).parameters
            if name in hints and any(
                getattr(a, "__name__", "").endswith("Parameters")
                for a in typing.get_args(hints[name]))}
        self.assertEqual(parameter_args, set(JAM_PARAMETER_CLASSES) | {"dust"})
        self.assertLessEqual(set(JAM_PARAMETER_CLASSES),
                             set(inspect.signature(echam_physics).parameters))
        for name, cls in JAM_PARAMETER_CLASSES.items():
            self.assertIs(typing.get_args(hints[name])[0], cls, name)
        # This test file exercises every door but sea salt and DMS (above).
        self.assertEqual(set(self._JAM_DOORS),
                         set(JAM_PARAMETER_CLASSES) - {"seasalt", "dms"})

    def test_jam_scheme_mappings_reach_their_terms_only(self):
        """Each mapping sets its own scheme's field and nothing else.

        Every other field of that scheme keeps its default, and no other
        term's parameters move.
        """
        import dataclasses

        from jcm.physics.aerosol.jam.jam_terms import JAM_PARAMETER_CLASSES
        from jcm.physics.echam.echam_terms import echam_physics

        baseline = echam_physics(**self._JAM_KWARGS)
        for arg, (term, field, value) in self._JAM_DOORS.items():
            with self.subTest(door=arg):
                tuned = echam_physics(**self._JAM_KWARGS, **{arg: {field: value}})
                got = self._jam_term_params(tuned, term)
                self.assertAlmostEqual(float(getattr(got, field)) / value, 1.0,
                                       places=5)
                # Every other field is what the composition holds without the
                # override (the class default, but for the per-tracer
                # fractions of ``conv_transport``, which the term takes from
                # its mode layout).
                untuned = self._jam_term_params(baseline, term)
                for f in dataclasses.fields(untuned):
                    if f.name != field:
                        np.testing.assert_array_equal(
                            np.asarray(getattr(got, f.name)),
                            np.asarray(getattr(untuned, f.name)),
                            err_msg=f.name)
                    if f.name != field and arg != "conv_transport":
                        np.testing.assert_array_equal(
                            np.asarray(getattr(got, f.name)),
                            np.asarray(getattr(
                                JAM_PARAMETER_CLASSES[arg].default(), f.name)),
                            err_msg=f.name)
                self.assertEqual([t.name for t in baseline.terms],
                                 [t.name for t in tuned.terms])
                for other in self._JAM_ALSO.get(arg, ()):
                    shared = self._jam_term_params(tuned, other)
                    self.assertAlmostEqual(
                        float(getattr(shared, field)) / value, 1.0, places=5,
                        msg=other)
                for before, after in zip(baseline.terms, tuned.terms):
                    if (after.name == term
                            or after.name in self._JAM_ALSO.get(arg, ())
                            or not hasattr(after, "params")):
                        continue
                    for a, b in zip(
                            jax.tree_util.tree_leaves(before.params.get_value()),
                            jax.tree_util.tree_leaves(after.params.get_value())):
                        np.testing.assert_array_equal(
                            np.asarray(a), np.asarray(b), err_msg=after.name)

    def test_tracer_diffusion_without_the_cloud_borne_phase(self):
        """With ``jam_cloud_borne=False`` there is no carry store to set.

        The advected tracers' mixing term takes the override alone.
        """
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(**{**self._JAM_KWARGS, "jam_cloud_borne": False},
                                tracer_diffusion={"diffusion_scale": 0.7})
        self.assertNotIn("jam_cloud_borne_store",
                         [t.name for t in physics.terms])
        mixing = self._jam_term_params(physics, "tracer_vertical_diffusion")
        self.assertAlmostEqual(float(mixing.diffusion_scale), 0.7)

    def test_conv_transport_mapping_keeps_the_layout_fractions(self):
        """The mapping sets scalars; the per-tracer fractions stay the layout's.

        ``csr_conv`` is one fraction per transported tracer, built from the
        population's mode layout, so a mapping is applied on a default that
        leaves it to the term. It is not itself overridable.
        """
        from jcm.physics.echam.echam_terms import echam_physics

        base = self._jam_term_params(
            echam_physics(**self._JAM_KWARGS), "convective_tracer_transport")
        tuned = self._jam_term_params(
            echam_physics(**self._JAM_KWARGS, conv_transport={
                "conv_scav_scale": 0.4, "transport_scale": 0.8}),
            "convective_tracer_transport")
        self.assertGreater(base.csr_conv.shape[0], 1)
        self.assertGreater(float(base.csr_conv.max()), 0.0)
        np.testing.assert_array_equal(np.asarray(tuned.csr_conv),
                                      np.asarray(base.csr_conv))
        self.assertAlmostEqual(float(tuned.conv_scav_scale), 0.4)
        self.assertAlmostEqual(float(tuned.transport_scale), 0.8)
        self.assertEqual(float(base.conv_scav_scale), 1.0)
        with self.assertRaisesRegex(ValueError, "conv_transport.csr_conv"):
            echam_physics(**self._JAM_KWARGS,
                          conv_transport={"csr_conv": [0.5]})

    def test_wetdep_mapping_and_object_forms(self):
        """The wet-removal levers: mapping on the default, object as given."""
        from jcm.physics.aerosol.jam.wetdep.wetdep_term import WetDepParameters
        from jcm.physics.echam.echam_terms import echam_physics

        default = WetDepParameters.default()
        wet = self._jam_term_params(
            echam_physics(**self._JAM_KWARGS,
                          wetdep={"incloud_scale": 0.5, "impact_scale": 0.25}),
            "jam_wet_deposition")
        self.assertAlmostEqual(float(wet.incloud_scale), 0.5)
        self.assertAlmostEqual(float(wet.impact_scale), 0.25)
        self.assertEqual(float(wet.sol_factb), float(default.sol_factb))
        # Not given: the scheme's default, scales of one.
        wet = self._jam_term_params(
            echam_physics(**self._JAM_KWARGS), "jam_wet_deposition")
        self.assertEqual(float(wet.incloud_scale), 1.0)
        self.assertEqual(float(wet.impact_scale), 1.0)
        # An object is used as given, unscaled fields included.
        obj = WetDepParameters(
            incloud_scale=jnp.asarray(0.3), sol_factb=jnp.asarray(0.2),
            mu_water_air=default.mu_water_air,
            impact_scale=jnp.asarray(0.4),
            conv_scav_ratio=default.conv_scav_ratio,
            conv_updraft_velocity=default.conv_updraft_velocity,
            cdroprad_um=default.cdroprad_um)
        wet = self._jam_term_params(
            echam_physics(**self._JAM_KWARGS, wetdep=obj), "jam_wet_deposition")
        self.assertAlmostEqual(float(wet.incloud_scale), 0.3)
        self.assertAlmostEqual(float(wet.sol_factb), 0.2)
        self.assertAlmostEqual(float(wet.impact_scale), 0.4)

    def test_override_of_a_field_no_run_reads_warns(self):
        """A valid override that cannot move the run is flagged.

        ``conv_scav_ratio`` is read only without convective tracer transport
        (the in-plume scavenging then belongs to the transport term); the
        oxidant ozone and solar-geometry fallbacks, ARG's default updraft and
        the dry-deposition default friction velocity are never read here
        because the boundary conditions and the vertical-diffusion carry
        supply what they stand in for. A sweep over any of them would run
        identical arms. The live fields never warn.
        """
        import warnings

        from jcm.physics.echam.echam_terms import echam_physics

        for kwargs in (dict(wetdep={"conv_scav_ratio": 0.5}),
                       dict(oxidants={"o3_fallback_vmr": 5e-8}),
                       dict(oxidants={"cos_zenith_fallback": 0.5}),
                       dict(activation={"updraft_default": 0.5}),
                       dict(drydep={"u_star_default": 0.5})):
            with self.subTest(kwargs=kwargs):
                (scheme, fields), = kwargs.items()
                with self.assertWarnsRegex(
                        UserWarning, rf"{scheme}\.{next(iter(fields))} has no effect"):
                    echam_physics(**self._JAM_KWARGS, **kwargs)
        for kwargs in (
                dict(wetdep={"conv_scav_ratio": 0.5},
                     jam_convective_transport=False),
                dict(wetdep={"incloud_scale": 0.5, "impact_scale": 0.5}),
                dict(oxidants={"oh_ref": 3.0e6, "h2o2_ref_vmr": 4e-10,
                               "no3_ref_vmr": 2e-12}),
                dict(activation={"tke_factor": 0.5, "w_min": 0.2},
                     drydep={"z_ref": 12.0, "z0": 2e-4})):
            with self.subTest(kwargs=kwargs):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    echam_physics(**self._JAM_KWARGS, **kwargs)

    def test_jam_scheme_mappings_reject_unknown_fields(self):
        """A typo'd field names the scheme and lists the valid fields."""
        from jcm.physics.aerosol.jam.jam_terms import JAM_PARAMETER_CLASSES
        from jcm.physics.echam.echam_terms import echam_physics

        for arg, (_, field, value) in self._JAM_DOORS.items():
            with self.subTest(door=arg):
                cls = JAM_PARAMETER_CLASSES[arg].__name__
                with self.assertRaisesRegex(
                        ValueError,
                        rf"{arg}: unknown {cls} field\(s\) \['{field}_typo'\]"
                        rf".*Valid fields: .*'{field}'"):
                    echam_physics(**self._JAM_KWARGS,
                                  **{arg: {f"{field}_typo": value}})

    def test_jam_scheme_parameters_without_their_scheme_are_rejected(self):
        """A scheme the composition lacks would drop its parameters silently.

        MACv2-SP composes none of the JAM schemes; the anthropogenic emission
        and the cloud-borne exchange are composed only with their flag. A
        mapping and an object are refused alike.
        """
        from jcm.physics.aerosol.jam.jam_terms import JAM_PARAMETER_CLASSES
        from jcm.physics.echam.echam_terms import echam_physics

        for arg, (_, field, value) in self._JAM_DOORS.items():
            for form, given in (
                    ("mapping", {field: value}),
                    ("object", JAM_PARAMETER_CLASSES[arg].default())):
                with self.subTest(door=arg, form=form):
                    with self.assertRaisesRegex(
                            ValueError, rf"\['{arg}'\] would be ignored"):
                        echam_physics(checkpoint_terms=False, **{arg: given})
        jam = dict(self._JAM_KWARGS)
        for flag, arg in (("jam_anthropogenic", "anthropogenic_params"),
                          ("jam_cloud_borne", "cloud_borne_exchange"),
                          ("jam_convective_transport", "conv_transport")):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(
                        ValueError, rf"\['{arg}'\] would be ignored"):
                    echam_physics(**{**jam, flag: False},
                                  **{arg: {self._JAM_DOORS[arg][1]: 1.5}})

    def test_echam_physics_accepts_custom_radiation_term(self):
        """A radiation PhysicsTerm can be passed directly."""
        from jcm.physics.echam.echam_terms import echam_physics

        custom_rad = DummyRadiationTerm()
        physics = echam_physics(
            checkpoint_terms=False,
            radiation_scheme=custom_rad,
        )

        rad_term = next(t for t in physics.terms if t.category == "radiation")
        self.assertIs(rad_term, custom_rad)

    def test_default_radiation_is_rrtmgp(self):
        """The ECHAM factory composes RRTMGP unless told otherwise."""
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

        physics = echam_physics(checkpoint_terms=False)
        (rad,) = (t for t in physics.terms if t.category == "radiation")
        self.assertIsInstance(rad, RRTMGPRadiation)

    def test_grey_is_rejected_with_the_composition_route(self):
        """``radiation_scheme="grey"`` is not an ECHAM option (#918).

        The message must name the scheme as idealized and show the explicit
        composition route, and it must win over any other validation the
        call would also trip.
        """
        from jcm.physics.echam.echam_terms import echam_physics

        for extra in ({}, {"aerosol_free_interval": 1},
                      {"emulator_weights_file": "x.nc"}):
            with self.assertRaises(ValueError) as cm:
                echam_physics(radiation_scheme="grey", **extra)
            msg = str(cm.exception)
            self.assertIn("idealized scheme, not ECHAM physics", msg)
            self.assertIn(
                "from jcm.physics.radiation.grey_two_stream import "
                "GreyTwoStreamRadiation", msg)
            self.assertIn(
                "echam_physics(radiation_scheme=GreyTwoStreamRadiation())",
                msg)
            # One route only: ``replace`` keeps the displaced term's band
            # config and JAM optics cadence, so it is not offered (#926).
            self.assertNotIn("replace", msg)

    def test_unknown_radiation_string_lists_only_echam_options(self):
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaises(ValueError) as cm:
            echam_physics(radiation_scheme="bogus")
        msg = str(cm.exception)
        self.assertIn("'rrtmgp'", msg)
        self.assertIn("'emulated'", msg)
        self.assertNotIn("grey", msg)

    def test_grey_instance_route_composes_the_idealized_stack(self):
        """The explicit grey composition the rejection message shows works.

        The instance is composed as given, the composition carries the
        broadband band config the grey scheme needs, and the rest of the
        stack keys on the instance's own radiation parameters (the JAM
        optics gate follows its ``radiation_interval``).
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.band_config import RadiationBandConfig
        from jcm.physics.radiation.grey_two_stream import (
            GreyTwoStreamRadiation,
        )
        from jcm.physics.radiation.radiation_types import RadiationParameters

        grey = GreyTwoStreamRadiation(params=RadiationParameters.default(
            radiation_interval=3600.0))
        physics = echam_physics(
            checkpoint_terms=False, radiation_scheme=grey,
            aerosol_module="jam", cloud_scheme="2m",
            jam_microphysics="placeholder")
        (rad,) = (t for t in physics.terms if t.category == "radiation")
        self.assertIs(rad, grey)
        self.assertEqual(physics.band_config, RadiationBandConfig.broadband())
        (optics,) = (t for t in physics.terms
                     if hasattr(t, "configure_radiation_gate"))
        self.assertEqual(optics._radiation_interval_s, 3600.0)

    def test_jam_rejects_a_radiation_term_without_readable_params(self):
        """JAM's optics follow the radiation cadence, read from ``.params``.

        A term that exposes none is rejected where the JAM optics are
        composed rather than silently given the default cadence, and accepted
        where no sibling needs it: without JAM, and with JAM but no optics.
        """
        from jcm.physics.echam.echam_terms import echam_physics

        jam = dict(checkpoint_terms=False, aerosol_module="jam",
                   cloud_scheme="2m", jam_microphysics="placeholder")
        with self.assertRaisesRegex(ValueError, r"\.params"):
            echam_physics(radiation_scheme=DummyRadiationTerm(), **jam)
        echam_physics(checkpoint_terms=False,
                      radiation_scheme=DummyRadiationTerm())  # no raise
        echam_physics(radiation_scheme=DummyRadiationTerm(),
                      jam_optics=False, **jam)  # no raise

    def test_radiation_params_rejected_alongside_an_instance(self):
        """``radiation=`` would be silently ignored next to a term instance."""
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.grey_two_stream import (
            GreyTwoStreamRadiation,
        )
        from jcm.physics.radiation.radiation_types import RadiationParameters

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            echam_physics(radiation_scheme=GreyTwoStreamRadiation(),
                          radiation=RadiationParameters.default())

    def test_echam_physics_rejects_non_radiation_custom_term(self):
        """Custom radiation terms must declare the radiation category."""
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, "category 'radiation'"):
            echam_physics(
                checkpoint_terms=False,
                radiation_scheme=DummyConvectionTerm(),
            )

    def test_column_vector_handles_vmap_scalar_shapes(self):
        """Radiation scalar diagnostics are normalized to [ncols]."""
        from jcm.physics.radiation.grey_two_stream.radiation_scheme import (
            _column_vector,
        )

        self.assertEqual(_column_vector(jnp.arange(3), 3).shape, (3,))
        self.assertEqual(
            _column_vector(jnp.arange(3).reshape(3, 1), 3).shape,
            (3,),
        )

    def test_composable_with_model(self):
        """The composed ECHAM term stack runs inside Model.

        Machinery only, so the idealized composition keeps it cheap.
        """
        from jcm.model import Model
        from jcm.physics.echam.testing import idealized_echam_physics

        composable = idealized_echam_physics()
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=composable,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    def test_replace_radiation(self):
        """Can replace radiation with a different scheme."""
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.physics.radiation.grey_two_stream import (
            GreyTwoStreamRadiation,
        )

        # ``replace`` puts the new term in the radiation slot in place.
        real = echam_physics(checkpoint_terms=False)
        new_rad = GreyTwoStreamRadiation()
        swapped = real.replace("radiation", new_rad)
        self.assertEqual([t.category for t in swapped.terms],
                         [t.category for t in real.terms])
        self.assertIs(next(t for t in swapped.terms
                           if t.category == "radiation"), new_rad)

        # And a replaced composition still computes (cheap idealized stack).
        composable = idealized_echam_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)
        replaced = composable.replace("radiation", GreyTwoStreamRadiation())
        replaced.cache_coords(self.coords)

        tend, _ = replaced.compute_tendencies(self.state, self.forcing, self.terrain,
        )
        # Check shape is correct (NaNs expected with random state)
        self.assertEqual(
            tend.temperature.shape, self.state.temperature.shape,
        )

    def test_nnx_split_merge(self):
        """nnx.split/merge works for ECHAM composable physics."""
        from jcm.physics.echam.echam_terms import echam_physics

        composable = echam_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)

        # Verify split/merge roundtrip works
        graphdef, state = nnx.split(composable)
        restored = nnx.merge(graphdef, state)
        self.assertEqual(len(restored.terms), 13)

class TestAerosolFreeValidation(unittest.TestCase):
    """The mode/interval contract must hold for every radiation scheme.

    The guard cannot live only in ``RRTMGPRadiation.__init__``: the emulated
    branch and a custom radiation term never construct it, so a nonsensical
    interval would be accepted in silence on exactly the paths that cannot
    produce *noa fluxes at all.
    """

    def setUp(self):
        from jcm.physics.echam.echam_terms import echam_physics
        self.echam_physics = echam_physics

    def test_interval_is_rejected_on_non_rrtmgp_schemes(self):
        from jcm.physics.radiation.grey_two_stream import (
            GreyTwoStreamRadiation,
        )
        for scheme in ("emulated", GreyTwoStreamRadiation()):
            with self.assertRaises(ValueError) as cm:
                self.echam_physics(radiation_scheme=scheme,
                                   aerosol_free_interval=1)
            self.assertIn("radiation_scheme='rrtmgp'", str(cm.exception))

    def test_emulated_composition_carries_the_rrtmgp_bands(self):
        # The per-band emulator expects the RRTMGP band structure
        # (14 SW / 16 LW); the broadband 1-SW/0-LW layout passes
        # composition and then fails the emulator's band-count check at
        # first compute (PR #730 review). Only the Hydra runner path had
        # the emulator in its band selection; the Python factory must too.
        physics = self.echam_physics(radiation_scheme="emulated")
        bc = physics.band_config
        self.assertEqual(len(bc.sw_band_centers_nm), 14)
        self.assertEqual(len(bc.lw_band_centers_nm), 16)

    def test_nonsensical_interval_is_rejected_before_the_scheme_check(self):
        # A meaningless spacing must name the real problem rather than
        # complain about the radiation scheme, which would send the reader
        # down a blind alley.
        with self.assertRaises(ValueError) as cm:
            self.echam_physics(radiation_scheme="emulated",
                               aerosol_free_interval=0)
        self.assertIn("must be >= 1", str(cm.exception))


class TestEmulatorWeightsFile(unittest.TestCase):
    """The factory must load TRAINED emulator weights by default (#640 trap).

    ``echam_physics(radiation_scheme="emulated")`` used to build the term with
    random untrained weights, which the scheme's own docs say NaN within a step;
    it now defaults to the packaged trained checkpoint.
    """

    def setUp(self):
        from jcm.physics.echam.echam_terms import echam_physics
        self.echam_physics = echam_physics

    def _rad_term(self, physics):
        return next(t for t in physics.terms
                    if getattr(t, "name", "") == "nn_emulator_radiation")

    def test_default_loads_packaged_trained_weights(self):
        term = self._rad_term(self.echam_physics(radiation_scheme="emulated"))
        # ``_weights_file`` is set only on the load-from-file path (the random
        # init leaves it None), and the packaged default is the u64 checkpoint.
        self.assertIsNotNone(term._weights_file)
        self.assertTrue(
            str(term._weights_file).endswith(
                "emulator_weights_per_band_u64.nc"))

    def test_random_reaches_the_random_init_path(self):
        # The explicit "random" sentinel is the train-from-scratch value:
        # weights_file=None on the term (``_weights_file`` stays None).
        term = self._rad_term(
            self.echam_physics(radiation_scheme="emulated",
                               emulator_weights_file="random"))
        self.assertIsNone(term._weights_file)

    def test_none_falls_back_to_auto(self):
        # None is treated as the "auto" default (an omitted/null config key,
        # which the Hydra builder strips, must NOT reach random init) — it
        # loads the packaged trained checkpoint, exactly like the default.
        term = self._rad_term(
            self.echam_physics(radiation_scheme="emulated",
                               emulator_weights_file=None))
        self.assertIsNotNone(term._weights_file)
        self.assertTrue(
            str(term._weights_file).endswith(
                "emulator_weights_per_band_u64.nc"))

    def test_rejected_on_non_emulated_scheme(self):
        # An explicit value with rrtmgp is a silently-ignored argument —
        # the factory rejects it (same contract as aerosol_free_interval).
        with self.assertRaises(ValueError) as cm:
            self.echam_physics(radiation_scheme="rrtmgp",
                               emulator_weights_file="some_ckpt.nc")
        self.assertIn("radiation_scheme='emulated'", str(cm.exception))

    def test_auto_default_does_not_trip_non_emulated_schemes(self):
        # The "auto" default must stay silent for other schemes (it is the
        # unset state, not a user choice).
        self.echam_physics(radiation_scheme="rrtmgp")  # no raise


@pytest.mark.slow
class TestRadiationReadsLaggedConvectionType(unittest.TestCase):
    """End to end through the composed ECHAM stack, RRTMGP radiation (#870).

    Slow: two full RRTMGP solves over a T21 grid (~2 min on CPU). The
    scheme-level ``liquid_inhomogeneity`` / ktype checks in
    ``rrtmgp_test.py`` stay in the fast suite.

    Step 1 publishes a ``convection`` carry; the radiation term, run on that
    carry with ``ktype`` forced per column to 0 / 2 / 4, must thin the liquid
    cloud only in the ktype-4 columns — the carry → term → scheme path the
    scheme-level tests do not exercise.
    """

    def test_only_shallow_liquid_columns_change(self):
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        physics = echam_physics(checkpoint_terms=False)
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        physics.cache_coords(coords)
        nlev = 8
        nlon, nlat = coords.horizontal.nodal_shape
        ncols = nlon * nlat
        qc = jnp.zeros((nlev, nlon, nlat)).at[5:7].set(2e-4)
        state = PhysicsState.zeros(
            (nlev, nlon, nlat),
            temperature=jnp.full((nlev, nlon, nlat), 285.0),
            specific_humidity=jnp.full((nlev, nlon, nlat), 8e-3),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={**{spec.name: jnp.zeros((nlev, nlon, nlat))
                        for spec in physics.required_tracers()}, "qc": qc},
        )
        forcing = ForcingData.zeros((nlon, nlat))
        _, diag = physics.compute_tendencies(
            state, forcing, TerrainData.aquaplanet(coords))

        # Column view of the state and a fixed half-cover liquid cloud where
        # qc sits, so every column carries the same optically thick liquid.
        cols = lambda a: jnp.reshape(a, a.shape[:1] + (ncols,))  # noqa: E731
        state_cols = PhysicsState.zeros(
            (nlev, ncols),
            temperature=cols(state.temperature),
            specific_humidity=cols(state.specific_humidity),
            normalized_surface_pressure=jnp.ones(ncols),
            tracers={"qc": cols(qc), "qi": jnp.zeros((nlev, ncols))},
        )
        clouds = diag["clouds"].copy(
            cloud_fraction=jnp.where(cols(qc) > 0, 0.5, 0.0))
        pattern = jnp.tile(jnp.array([0, 2, 4], jnp.int32),
                           ncols // 3 + 1)[:ncols]
        rad = next(t for t in physics.terms if t.category == "radiation")
        params = rad.params.get_value()

        def solve(ktype):
            d = {**diag, "clouds": clouds,
                 "convection": diag["convection"].replace(ktype=ktype)}
            _, out, _radii = rad._compute_full(state_cols, d, forcing, None, params)
            return (np.asarray(out.sw_heating_rate),        # (nlev, ncols)
                    np.asarray(out.toa_sw_up), np.asarray(out.cos_zenith))

        base, base_up, mu0 = solve(jnp.zeros(ncols, jnp.int32))
        mixed, mixed_up, _ = solve(pattern)
        pattern = np.asarray(pattern)
        self.assertTrue(np.all(np.isfinite(mixed)))
        # 0 and 2 share the 0.8 liquid factor: bit-identical columns.
        np.testing.assert_array_equal(mixed[:, pattern != 4],
                                      base[:, pattern != 4])
        # ktype 4 in daylight: the 0.4 liquid factor thins the cloud, so the
        # column's SW heating changes and less SW is reflected on average.
        # (The LW is saturated by this much liquid, hence the SW check; a
        # few columns are float32-insensitive to the change.)
        lit4 = (pattern == 4) & (mu0 > 0.1)
        self.assertGreater(int(lit4.sum()), 0)
        changed = np.abs(mixed[:, lit4] - base[:, lit4]).max(axis=0) > 0.0
        self.assertGreater(float(changed.mean()), 0.9)
        self.assertLess(float(mixed_up[lit4].mean()),
                        float(base_up[lit4].mean()))


if __name__ == "__main__":
    unittest.main()


def _integer_leaves(tree):
    """``{path: dtype}`` of every non-float array leaf of ``tree``."""
    return {
        jax.tree_util.keystr(path): leaf.dtype
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        if hasattr(leaf, "dtype") and not jnp.issubdtype(leaf.dtype, jnp.floating)
    }


class TestSixtyFourBitMode(unittest.TestCase):
    """The composed ECHAM package traces and steps with x64 on (#945).

    ``jax_enable_x64`` is on whenever ``mam4_jax`` has been imported, and
    pySES runs float32 physics under it. Every ``lax.cond`` / ``lax.switch``
    / ``lax.scan`` in the package must then return identical dtypes from its
    branches, which untyped literals (int64 / float64 under x64) break. The
    flag is flipped only through the ``jax.enable_x64`` context manager, so
    the process-global setting is left as found; ``mam4_jax`` is not
    imported, so this runs in CI without the optional extra.
    """

    def _model(self, **physics_kwargs):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        coords = get_coords(np.linspace(0, 1, 9), spectral_truncation=21)
        return Model(coords=coords, time_step=30,
                     terrain=TerrainData.aquaplanet(coords),
                     physics=echam_physics(**physics_kwargs))

    def test_default_package_steps_with_int32_carry_indices(self):
        with jax.enable_x64(True):
            model = self._model()
            pred = model.run(save_interval=1 / 48, total_time=1 / 48)
            self.assertTrue(bool(jnp.all(jnp.isfinite(pred.dynamics.temperature))))
            carry = model._final_physics_state
            for leaf in jax.tree.leaves(carry):
                if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
                    self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
            ints = _integer_leaves(carry)
            # The convection ktype / cloud_base / cloud_top and the
            # radiation sub-cycle counter.
            self.assertGreaterEqual(len(ints), 4)
            for path, dtype in ints.items():
                self.assertEqual(dtype, jnp.int32, path)

    def _trace(self, state_dtype, **physics_kwargs):
        """Trace one physics step at ``state_dtype`` with x64 on."""
        with jax.enable_x64(True):
            model = self._model(**physics_kwargs)
            physics = model.physics
            coords = model.coords
            nodal = coords.horizontal.nodal_shape
            shape_3d = (coords.nodal_shape[0],) + nodal
            state = PhysicsState.zeros(shape_3d).copy(
                temperature=jnp.full(shape_3d, 288.0),
                normalized_surface_pressure=jnp.ones(nodal),
                tracers={spec.name: jnp.zeros(shape_3d)
                         for spec in physics.required_tracers()},
            )
            forcing = ForcingData.zeros(nodal)
            for term in physics.terms:
                forcing = term.augment_probe_forcing(forcing)

            def cast(tree):
                return jax.tree.map(
                    lambda x: x.astype(state_dtype)
                    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
                    else x, tree)

            return jax.eval_shape(physics.compute_tendencies, cast(state),
                                  cast(forcing), cast(model.terrain))

    def _assert_traced(self, out, state_dtype):
        for path, dtype in _integer_leaves(out).items():
            self.assertEqual(dtype, jnp.int32, path)
        # No float leaf escapes the working precision.
        for leaf in jax.tree.leaves(out):
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                self.assertEqual(leaf.dtype, state_dtype)

    def test_float32_physics_under_x64_traces(self):
        # pySES's physics_dtype=float32 mode.
        self._assert_traced(self._trace(jnp.float32), jnp.float32)

    def test_rrtmgp_aerosol_free_companion_traces_under_x64(self):
        # The held-vs-solved ``*noa`` cond (aerosol_free_interval > 1).
        for dtype in (jnp.float32, jnp.float64):
            with self.subTest(dtype=dtype):
                out = self._trace(dtype, radiation_scheme="rrtmgp",
                                  aerosol_free_interval=2)
                self._assert_traced(out, dtype)


class TestConvectiveDetrainmentCarry(unittest.TestCase):
    """``clouds.conv_detrainment_*`` hold THIS step's detrainment, composed.

    The ``clouds`` struct rides the cross-step carry, so a step starts from
    the previous step's detrainment fields. Through the whole ECHAM stack
    (grey radiation, for cost) a carry holding a stale value must come out
    as zero when no convection term is composed, and as exactly this step's
    applied detrainment when one is — under both cloud schemes, whose
    microphysics terms copy the struct through.
    """

    _STALE_QC, _STALE_QI = 1.0e-3, 2.0e-3

    def _run(self, physics):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        physics.cache_coords(coords)
        nlev = 8
        nlon, nlat = coords.horizontal.nodal_shape
        ncols = nlon * nlat
        shape = (nlev, nlon, nlat)
        state = PhysicsState.zeros(
            shape,
            temperature=jnp.full(shape, 280.0),
            specific_humidity=jnp.full(shape, 4e-3),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={spec.name: jnp.zeros(shape)
                     for spec in physics.required_tracers()},
        )
        carry = physics.initial_carry_state(coords)
        carry["clouds"] = carry["clouds"].copy(
            conv_detrainment_qc=jnp.full((nlev, ncols), self._STALE_QC),
            conv_detrainment_qi=jnp.full((nlev, ncols), self._STALE_QI),
        )
        _, diag = jax.jit(physics.compute_tendencies)(
            state, ForcingData.zeros((nlon, nlat)),
            TerrainData.aquaplanet(coords), carry,
        )
        return diag["clouds"]

    def test_zero_without_a_convection_term(self):
        from jcm.physics.echam.testing import idealized_echam_physics

        for scheme in ("1m", "2m"):
            with self.subTest(cloud_scheme=scheme):
                physics = idealized_echam_physics(
                    cloud_scheme=scheme).remove("convection")
                self.assertFalse(
                    [t for t in physics.terms if t.category == "convection"])
                clouds = self._run(physics)
                np.testing.assert_array_equal(
                    np.asarray(clouds.conv_detrainment_qc), 0.0)
                np.testing.assert_array_equal(
                    np.asarray(clouds.conv_detrainment_qi), 0.0)

    def test_this_steps_detrainment_with_convection(self):
        import jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng as tn
        from jcm.physics.convection.tiedtke_nordeng.types import (
            ConvectionTendencies)
        from jcm.physics.echam.testing import idealized_echam_physics

        det_qc, det_qi = 1.0e-8, 3.0e-8

        # A known detrainment with no heating (cap inactive), so the value
        # the composed stack must publish is exact.
        def fake_convection(temperature, *args, **kwargs):
            zeros = jnp.zeros_like(temperature)
            return ConvectionTendencies(
                dtedt=zeros, dqdt=zeros, dudt=zeros, dvdt=zeros,
                qc_conv=zeros, qi_conv=zeros, precip_formation=zeros,
                precip_conv=jnp.zeros((), temperature.dtype),
                precip_flux=zeros,
                precip_floor_source=jnp.zeros((), temperature.dtype),
                dqc_dt=jnp.full_like(temperature, det_qc),
                dqi_dt=jnp.full_like(temperature, det_qi),
            ), None

        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(tn, "tiedtke_nordeng_convection", fake_convection)
            for scheme in ("1m", "2m"):
                with self.subTest(cloud_scheme=scheme):
                    clouds = self._run(
                        idealized_echam_physics(cloud_scheme=scheme))
                    np.testing.assert_allclose(
                        np.asarray(clouds.conv_detrainment_qc), det_qc,
                        rtol=1e-6)
                    np.testing.assert_allclose(
                        np.asarray(clouds.conv_detrainment_qi), det_qi,
                        rtol=1e-6)
        finally:
            monkey.undo()


class TestSharedCloudConstants(unittest.TestCase):
    """ECHAM's one csecfrl and cthomi: jcm's two copies warn when they differ."""

    @staticmethod
    def _shared_warnings(**kwargs):
        import warnings
        from jcm.physics.echam.echam_terms import echam_physics
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            echam_physics(**kwargs)
        return [str(w.message) for w in caught if "ECHAM has one" in str(w.message)]

    def test_defaults_do_not_warn(self):
        for scheme in ("1m", "2m"):
            self.assertEqual(self._shared_warnings(cloud_scheme=scheme), [])

    def test_differing_copies_warn(self):
        for kwargs, name in (({"clouds": {"csecfrl": 1e-5}}, "csecfrl"),
                             ({"microphysics": {"cthomi": 240.0}}, "cthomi"),
                             ({"cloud_scheme": "2m", "clouds": {"t_ice": 240.0}},
                              "cthomi")):
            found = self._shared_warnings(**kwargs)
            self.assertEqual(len(found), 1, kwargs)
            self.assertIn(name, found[0])
