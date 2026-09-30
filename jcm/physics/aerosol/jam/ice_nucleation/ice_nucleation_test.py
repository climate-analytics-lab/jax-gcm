"""Tests for the JAM aerosol inputs to mixed-phase freezing (#953).

The partition against the compiled HAM routine is in
``ham_freezing_reference_test.py``; here: a hand-computed MAM4 cell, the MAM4 ->
HAM class mapping, the term's contract, gradients and the model wiring.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
from jcm.physics.aerosol.jam.activation.arg_term import JamActivationData
from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import (
    MAM4_FREEZING_CLASSES,
    HamFreezingClasses,
    ham_freezing_aerosol,
)
from jcm.physics.aerosol.jam.ice_nucleation.ice_term import IceNucleation
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.tracer_layout import number_name
from jcm.physics_interface import PhysicsState
from jcm.testing import check_gradients

_SHAPE = (3, 2)
_RHO = 0.7


def _masses(du_acc=2e-10, so4_acc=1e-9, bc_acc=1e-11, du_cor=3e-9, ss_cor=1e-9,
            bc_pcm=5e-11, poa_pcm=2e-10):
    """Per-(species, class) masses [kg/kg] for every species of the MAM4 classes HAM uses."""
    vals = {("du", "acc"): du_acc, ("so4", "acc"): so4_acc, ("bc", "acc"): bc_acc,
            ("du", "cor"): du_cor, ("ss", "cor"): ss_cor, ("bc", "pcm"): bc_pcm,
            ("poa", "pcm"): poa_pcm}
    out = {}
    for short in ("acc", "cor", "pcm"):
        for sp in MAM4_SPEC.mode(short).species:
            out[(sp, short)] = jnp.full(_SHAPE, vals.get((sp, short), 0.0))
    return out


def _partition(masses, nact_acc=3e7, nact_cor=5e4, n_pcm=5e8, cdncact=5e7, r_pcm=3e-8):
    full = lambda v: jnp.full(_SHAPE, v)  # noqa: E731
    return ham_freezing_aerosol(
        MAM4_SPEC, MAM4_FREEZING_CLASSES, masses, {"pcm": full(n_pcm)},
        {"acc": full(nact_acc), "cor": full(nact_cor), "ait": full(0.0)},
        {"acc": full(1e-7), "ait": full(2e-8), "cor": full(1e-6), "pcm": full(r_pcm)},
        full(_RHO), full(cdncact))


class HamFreezingPartitionTest(unittest.TestCase):
    def test_hand_computed_mam4_cell(self):
        """HAM's surface weighting by hand for one MAM4 cell (mo_ham_freezing.f90:198-465)."""
        fa = _partition(_masses())
        rho_d = {s.name: s.density for s in MAM4_SPEC.species}
        # soluble classes: volume ratio of dust over the class's dry species (with HAM's
        # 1000/density weights, which cancel in the ratio), to the 2/3, times the activated
        # number of the class, over the activated CDNC
        v = lambda m, sp: m / rho_d[sp]  # noqa: E731
        acc_vol = v(1e-9, "so4") + v(2e-10, "du") + v(1e-11, "bc")
        cor_vol = v(3e-9, "du") + v(1e-9, "ss")
        n_du = (v(2e-10, "du") / acc_vol) ** (2 / 3) * 3e7 + (v(3e-9, "du") / cor_vol) ** (2 / 3) * 5e4
        n_bc = (v(1e-11, "bc") / acc_vol) ** (2 / 3) * 3e7
        np.testing.assert_allclose(np.asarray(fa.dust_soluble), n_du / 5e7, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(fa.bc_soluble), n_bc / 5e7, rtol=2e-6)
        # the insoluble class is primary carbon: BC mass ratio to the 2/3, over its own number
        np.testing.assert_allclose(np.asarray(fa.bc_insoluble), (5e-11 / 2.5e-10) ** (2 / 3), rtol=2e-6)
        # The same by hand on paper (MAM4 densities so4 1770, du 2600, bc 1700, ss 1900):
        # accumulation dust volume ratio 7.692e-14/6.478e-13 = 0.11875, to the 2/3 = 0.2416,
        # x 3e7; coarse 1.1538e-12/1.6802e-12 = 0.6867 -> 0.7784 x 5e4; sum 7.287e6 / 5e7.
        np.testing.assert_allclose(float(fa.dust_soluble[0, 0]), 0.145735, rtol=1e-5)

    def test_mam4_has_no_insoluble_dust(self):
        """MAM4 carries all its dust in the soluble accumulation and coarse modes, so HAM's
        contact inputs for dust are zero and only the primary-carbon radius is set.
        """
        fa = _partition(_masses(du_acc=1e-8, du_cor=1e-7))
        for f in ("dust_insoluble_accumulation", "dust_insoluble_coarse",
                  "wet_radius_insoluble_accumulation", "wet_radius_insoluble_coarse"):
            self.assertTrue(np.all(np.asarray(getattr(fa, f)) == 0.0), f)
        np.testing.assert_array_equal(np.asarray(fa.wet_radius_insoluble_aitken),
                                      np.asarray(jnp.full(_SHAPE, 3e-8)))

    def test_no_aerosol_no_activation(self):
        fa = _partition(_masses(0, 0, 0, 0, 0, 0, 0), nact_acc=0.0, nact_cor=0.0, n_pcm=0.0,
                        cdncact=0.0)
        for f in ("dust_soluble", "bc_soluble", "bc_insoluble"):
            self.assertTrue(np.all(np.asarray(getattr(fa, f)) == 0.0), f)

    def test_fraction_clipped_at_one(self):
        fa = _partition(_masses(du_acc=1e-8, so4_acc=0.0, bc_acc=0.0), nact_acc=1e8, cdncact=1e7)
        np.testing.assert_array_equal(np.asarray(fa.dust_soluble), 1.0)

    def test_class_roles_are_validated(self):
        with self.assertRaises(ValueError):
            HamFreezingClasses(soluble=("pcm",)).validate(MAM4_SPEC)
        with self.assertRaises(ValueError):
            HamFreezingClasses(soluble=("acc",), insoluble_aitken="acc").validate(MAM4_SPEC)


class HamFreezingGradientTest(unittest.TestCase):
    """The fractions are differentiable in the aerosol, the activation and the CDNC, with a
    finite derivative at zero aerosol and where the MIN(., 1) clip binds.
    """

    def _f(self, du_acc, du_cor, bc_acc, nact_acc, cdncact):
        m = _masses()
        m[("du", "acc")], m[("du", "cor")], m[("bc", "acc")] = du_acc, du_cor, bc_acc
        full = lambda v: jnp.full(_SHAPE, 1.0) * v  # noqa: E731
        fa = ham_freezing_aerosol(
            MAM4_SPEC, MAM4_FREEZING_CLASSES, m, {"pcm": full(5e8)},
            {"acc": nact_acc, "cor": full(5e4), "ait": full(0.0)},
            {s: full(1e-7) for s in ("acc", "ait", "cor", "pcm")}, full(_RHO), cdncact)
        return fa.dust_soluble, fa.bc_soluble

    def test_gradients_typical(self):
        with jax.enable_x64(True):
            full = lambda v: jnp.full(_SHAPE, v, jnp.float64)  # noqa: E731
            check_gradients(self._f, (full(2e-10), full(3e-9), full(1e-11), full(3e7), full(5e7)),
                            rtol=1e-4, live_inputs=("[0]", "[1]", "[2]", "[3]", "[4]"))

    def test_gradients_finite_at_zero_aerosol_and_clip(self):
        full = lambda v: jnp.full(_SHAPE, v)  # noqa: E731
        for args in ((full(0.0), full(0.0), full(0.0), full(0.0), full(0.0)),
                     (full(1e-8), full(1e-7), full(1e-11), full(1e8), full(1e5))):
            g = jax.grad(lambda *a: sum(jnp.sum(x) for x in self._f(*a)),
                         argnums=(0, 1, 2, 3, 4))(*args)
            for x in g:
                self.assertTrue(bool(jnp.all(jnp.isfinite(x))))


def _term_inputs(du_cb=0.0):
    tracers = {}
    for short, sp, v in (("acc", "du", 2e-10), ("acc", "so4", 1e-9), ("acc", "bc", 1e-11),
                         ("cor", "du", 3e-9), ("cor", "ss", 1e-9), ("pcm", "bc", 5e-11),
                         ("pcm", "poa", 2e-10)):
        tracers[mass_name(sp, short)] = jnp.full(_SHAPE, v)
    tracers[number_name("pcm")] = jnp.full(_SHAPE, 5e8)
    state = PhysicsState.zeros(_SHAPE).copy(temperature=jnp.full(_SHAPE, 255.0), tracers=tracers)
    n_modes = len(MAM4_SPEC.modes)
    number = jnp.stack([jnp.full(_SHAPE, v) for v in (1e8, 1e9, 1e5, 5e8)])
    frac = jnp.stack([jnp.full(_SHAPE, v) for v in (0.3, 0.0, 0.5, 0.0)])
    aer = JamAerosolState.zeros((_SHAPE[1],), _SHAPE[0], n_modes).copy(
        number=number, r_wet=jnp.stack([jnp.full(_SHAPE, v) for v in (1e-7, 2e-8, 1e-6, 3e-8)]))
    act = JamActivationData(number_frac=frac, mass_frac=frac)
    diagnostics = {"air_density": jnp.full(_SHAPE, _RHO), "_jam_state": aer,
                   "_jam_activation": act,
                   "activated_cdnc": jnp.sum(frac * number, axis=0) * _RHO}
    if du_cb:
        diagnostics["_jam_cloud_borne"] = {
            mass_name("du", "acc", cloud_borne=True): jnp.full(_SHAPE, du_cb)}
    return state, diagnostics


class IceNucleationTermTest(unittest.TestCase):
    def test_publishes_freezing_aerosol(self):
        state, diagnostics = _term_inputs()
        _, diags = IceNucleation()(state, diagnostics, None, None)
        fa = diags["freezing_aerosol"]
        for f in ("dust_soluble", "bc_soluble", "bc_insoluble", "dust_insoluble_accumulation"):
            x = np.asarray(getattr(fa, f))
            self.assertEqual(x.shape, _SHAPE)
            self.assertTrue(np.all((x >= 0.0) & (x <= 1.0)), f)
        self.assertGreater(float(fa.dust_soluble[0, 0]), 0.0)
        # HAM's nact_strat is ARG's per-class activated number (0.3 x 1e8 and 0.5 x 1e5 per
        # kg, times rho) and pcdncact their sum
        expect = _partition(_masses(), nact_acc=0.3 * 1e8 * _RHO, nact_cor=0.5 * 1e5 * _RHO,
                            cdncact=(0.3 * 1e8 + 0.5 * 1e5) * _RHO)
        np.testing.assert_allclose(np.asarray(fa.dust_soluble), np.asarray(expect.dust_soluble),
                                   rtol=1e-5)
        np.testing.assert_allclose(np.asarray(fa.bc_soluble), np.asarray(expect.bc_soluble),
                                   rtol=1e-5)
        np.testing.assert_allclose(np.asarray(fa.bc_insoluble), np.asarray(expect.bc_insoluble),
                                   rtol=1e-5)
        # the insoluble-Aitken radius is the primary-carbon mode's wet radius
        np.testing.assert_array_equal(np.asarray(fa.wet_radius_insoluble_aitken),
                                      np.asarray(jnp.full(_SHAPE, 3e-8)))

    def test_population_without_the_mam4_classes_must_name_its_own(self):
        import dataclasses
        spec = dataclasses.replace(MAM4_SPEC, modes=tuple(
            dataclasses.replace(m, short=m.short + "x") for m in MAM4_SPEC.modes))
        with self.assertRaisesRegex(ValueError, "pass classes="):
            IceNucleation(spec=spec)
        classes = HamFreezingClasses(soluble=("accx", "corx"), insoluble_aitken="pcmx")
        self.assertIs(IceNucleation(spec=spec, classes=classes)._classes, classes)

    def test_cloud_borne_mass_is_part_of_the_class(self):
        """HAM has one phase per class; jcm's cloud-borne dust belongs to the composition."""
        s0, d0 = _term_inputs()
        s1, d1 = _term_inputs(du_cb=5e-10)
        f0 = IceNucleation()(s0, d0, None, None)[1]["freezing_aerosol"]
        f1 = IceNucleation()(s1, d1, None, None)[1]["freezing_aerosol"]
        self.assertGreater(float(f1.dust_soluble[0, 0]), float(f0.dust_soluble[0, 0]))

    def test_factory_places_it_after_activation(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics
        terms = jam_aerosol_physics()
        cats = [t.category for t in terms]
        self.assertLess(cats.index("aerosol_activation"), cats.index("aerosol_ice_nucleation"))
        term = terms[cats.index("aerosol_ice_nucleation")]
        self.assertEqual(term.provides, ("freezing_aerosol",))


@pytest.mark.slow
class IceNucleationModelTest(unittest.TestCase):
    """End-to-end wiring: the ECHAM+JAM+2M model runs with ECHAM-HAM's freezing and
    publishes its inputs. On a 3-step cold-start T21 aquaplanet the dust is still ~0, so
    this guards the trace/compile/coupling path; the physics is pinned by the unit and
    Fortran-reference tests.
    """

    def test_runs_finite_and_publishes_inputs(self):
        import numpy as onp

        from jcm.model import Model
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(onp.linspace(0, 1, 21), spectral_truncation=21)
        model = Model(coords=coords, time_step=30, terrain=TerrainData.aquaplanet(coords),
                      physics=idealized_echam_physics(aerosol_module="jam", cloud_scheme="2m"))
        preds = model.run(save_interval=0.0625, total_time=0.0625)
        dyn = preds.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        for key in ("qi", "qni"):
            self.assertFalse(bool(jnp.any(jnp.isnan(dyn.tracers[key]))))
        fa = preds.physics["freezing_aerosol"]
        for f in ("dust_soluble", "bc_soluble", "bc_insoluble"):
            x = np.asarray(getattr(fa, f))
            self.assertTrue(np.all(np.isfinite(x)) and np.all((x >= 0) & (x <= 1)), f)


if __name__ == "__main__":
    unittest.main()
