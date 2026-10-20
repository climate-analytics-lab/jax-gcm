"""AER-12: the in-tree modal core against the MAM4-JAX reference, process by process.

Needs the optional GPL-3.0 ``mam4-jax`` extra (``pip install jcm[mam4]``,
marked ``requires_extra("mam4")``): skipped without it, run by the
``extras-tests`` CI job. ``mam4_jax`` and the adapter are imported only
inside the tests, never at collection, and the x64 flag the adapter sets is
restored in ``tearDown`` (#729). The comparison reads outputs only (D-A3).

**How the comparison is built.** Both cores run on the same ``PhysicsState``
and diagnostics. The adapter always runs MAM's ``calcsize`` and
``wateruptake`` before ``amicphys``, so its tendency with every ``mdo_*``
process switched off is subtracted as a baseline; the per-process cells
toggle the ``mdo_*`` keyword arguments of ``amicphys`` (through the
adapter's module-level binding) against the matching subset of
``PROCESS_ORDER``. Renaming and new-particle formation are isolated as
``(exchange + X) − exchange``, because MAM evaluates both on the exchange
step's time-mean vapour, which only an exchange step fills.

**Both dry and humid populations.** Until D-A6 the harness's κ-Köhler state
had neither MAM's deliquescence/crystallization hysteresis nor its Kelvin
term, and at mid humidity every uptake-dependent process differed by the
ratio of the wet diameters (the Aitken mode's by up to 2×). The state is
now MAM's water uptake, so the per-process tolerances hold at mixed
humidity as well as dry, and the growth factors themselves are compared
mode by mode. Measured values behind every tolerance are in the design
doc, §9 (AER-12 and D-A6).
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.requires_extra("mam4")

SHAPE = (6, 8)
DT = 1800.0
OFF = dict(mdo_gasaerexch=0, mdo_rename=0, mdo_newnuc=0, mdo_coag=0, mdo_pcarbonaging=0)
ALL = {k: 1 for k in OFF}
EXCHANGE = {**OFF, "mdo_gasaerexch": 1}


def _consistent_tracers(rng, shape, *, gas_h2so4=(1e-13, 3e-10), empty=False):
    """Build populations whose number and mass imply a diameter inside the mode bounds.

    MAM's ``calcsize`` renumbers a mode whose implied diameter is out of
    bounds; the in-tree core only clips the diagnostic. Keeping every mode
    in bounds makes the all-off baseline inert and the comparison about
    the processes.
    """
    from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
    from jcm.physics.aerosol.jam.microphysics import moments
    from jcm.physics.aerosol.jam.tracer_layout import gas_name

    tracers = {}
    for mode in MAM4_SPEC.modes:
        n = jnp.asarray(10 ** rng.uniform(6.0, 9.0, shape)) * (0.0 if empty else 1.0)
        dg = jnp.asarray(np.exp(rng.uniform(np.log(mode.dgnum_lo * 1.2),
                                            np.log(mode.dgnum_hi / 1.2), shape)))
        volume = moments.volume_from_diameter(n, dg, mode)
        weights = rng.dirichlet(np.ones(len(mode.species)), size=shape)
        tracers[number_name(mode.short)] = n
        for j, sp in enumerate(mode.species):
            tracers[mass_name(sp, mode.short)] = (
                volume * jnp.asarray(weights[..., j]) * MAM4_SPEC.species_props(sp).density)
    lo, hi = np.log10(gas_h2so4)
    tracers[gas_name("h2so4")] = jnp.asarray(10 ** rng.uniform(lo, hi, shape))
    tracers[gas_name("soag")] = jnp.asarray(10 ** rng.uniform(-12, -9, shape))
    tracers[gas_name("so2")] = jnp.full(shape, 1e-10)
    tracers[gas_name("dms")] = jnp.full(shape, 1e-11)
    return tracers


def _state(rng, *, humidity=None, **kw):
    from jcm.physics_interface import PhysicsState

    tracers = _consistent_tracers(rng, SHAPE, **kw)
    temperature = jnp.asarray(rng.uniform(235.0, 300.0, SHAPE))
    q = (jnp.full(SHAPE, humidity) if humidity is not None
         else jnp.asarray(rng.uniform(1e-4, 8e-3, SHAPE)))
    state = PhysicsState.zeros(SHAPE).copy(temperature=temperature, specific_humidity=q,
                                          tracers=tracers)
    diagnostics = {"pressure_full": jnp.asarray(rng.uniform(3e4, 1e5, SHAPE)),
                   "height_full": jnp.full(SHAPE, 1.5e3),
                   "height_half": jnp.zeros((SHAPE[0] + 1, SHAPE[1])),
                   "_dt_seconds": DT}
    return state, diagnostics


def _rel_rms(reference, ours):
    """RMS of the difference over the RMS of the reference, finite cells only."""
    a, b = np.asarray(reference, float), np.asarray(ours, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    scale = np.sqrt(np.mean(a ** 2))
    return np.sqrt(np.mean((a - b) ** 2)) / max(scale, 1e-300), int((~ok).sum())


class AbTest(unittest.TestCase):

    def setUp(self):
        self._x64 = jax.config.read("jax_enable_x64")

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._x64)

    # --- the two cores ----------------------------------------------------

    def _reference(self, state, diagnostics, flags, *, monolayers=8.0):
        """Run the adapter with the given ``mdo_*`` switches; return its tendency."""
        from jcm.physics.aerosol.jam.microphysics import mam4_jax as adapter

        original = adapter.amicphys
        adapter.amicphys = functools.partial(original, **flags)
        try:
            tendency, _ = adapter.Mam4JaxMicrophysics(n_so4_monolayers=monolayers)(
                state, diagnostics, None, None)
        finally:
            adapter.amicphys = original
        return {k: np.asarray(v, float) for k, v in tendency.tracers.items()}

    def _ours(self, state, diagnostics, processes):
        from jcm.physics.aerosol.jam.microphysics.modal import (
            JamModalMicrophysics,
            PROCESS_ORDER,
        )

        core = JamModalMicrophysics(
            processes=tuple(p for p in PROCESS_ORDER if p[0] in processes))
        tendency, _ = core(state, diagnostics, None, None)
        return {k: np.asarray(v, float) for k, v in tendency.tracers.items()}

    def _cells(self, state, diagnostics):
        """Baseline-subtracted reference and our tendencies, per process cell."""
        base = self._reference(state, diagnostics, OFF)

        def ref(flags):
            t = self._reference(state, diagnostics, flags)
            return {k: t[k] - base[k] for k in t}

        ex_r, ex_o = ref(EXCHANGE), self._ours(state, diagnostics, ("gas_aerosol_exchange",))
        rn_r = ref({**EXCHANGE, "mdo_rename": 1})
        rn_o = self._ours(state, diagnostics, ("gas_aerosol_exchange", "rename"))
        np_r = ref({**EXCHANGE, "mdo_newnuc": 1})
        np_o = self._ours(state, diagnostics, ("gas_aerosol_exchange", "nucleation"))
        return {
            "exchange": (ex_r, ex_o),
            "rename": ({k: rn_r[k] - ex_r[k] for k in rn_r}, {k: rn_o[k] - ex_o[k] for k in rn_o}),
            "npf": ({k: np_r[k] - ex_r[k] for k in np_r}, {k: np_o[k] - ex_o[k] for k in np_o}),
            "coagulation": (ref({**OFF, "mdo_coag": 1}),
                            self._ours(state, diagnostics, ("coagulation",))),
            "ageing": (ref({**EXCHANGE, "mdo_coag": 1, "mdo_pcarbonaging": 1}),
                       self._ours(state, diagnostics,
                                  ("gas_aerosol_exchange", "coagulation", "carbonaceous_ageing"))),
            "full": (ref(ALL), self._ours(state, diagnostics,
                                          ("gas_aerosol_exchange", "rename", "nucleation",
                                           "coagulation", "carbonaceous_ageing"))),
        }

    def _check(self, cells, cell, tolerances):
        reference, ours = cells[cell]
        for key, tol in tolerances.items():
            rel, bad = _rel_rms(reference[key], ours[key])
            self.assertEqual(bad, 0, f"{cell}/{key}: {bad} non-finite cells")
            self.assertLess(rel, tol, f"{cell}/{key}: relative RMS difference {rel:.3f} "
                                      f"exceeds {tol}")

    # --- dry populations: the microphysics itself -------------------------

    def _check_every_process(self, humidity):
        rng = np.random.default_rng(11)
        state, diagnostics = _state(rng, humidity=humidity)
        cells = self._cells(state, diagnostics)
        # SOAG: the reference runs CAM6's SOA configuration (fine-mode uptake
        # only, equilibrium at 250 g/mol; D-A8 mirrors both) and holds the
        # SOAG it condenses onto primary carbon as a transient coating that
        # only ``mdo_pcarbonaging`` books into a tracer, while this core
        # routes that condensate to accumulation within the exchange itself
        # (D-A1). In the exchange-only cell that is a ~5 % gas difference and
        # a large accumulation-SOA one; the full step closes the SOA mass to
        # 2 %, so the gas tolerance here is the measured 0.06 and the
        # exchange-only ``m_soa_acc`` is bookkeeping, not physics. Measured
        # 2026-10-20 on mam4-jax 0.5.1 (0.4.0, without the CAM6 path, gave
        # 0.004 on the gas).
        self._check(cells, "exchange", {"g_h2so4": 0.02, "g_soag": 0.08,
                                        "m_so4_ait": 0.15, "m_so4_acc": 0.25,
                                        "m_soa_ait": 0.20, "m_soa_acc": 0.40})
        self._check(cells, "rename", {"n_ait": 0.15, "n_acc": 0.15,
                                      "m_so4_ait": 0.15, "m_soa_acc": 0.25})
        self._check(cells, "npf", {"n_ait": 0.05, "m_so4_ait": 0.05, "g_h2so4": 0.05})
        self._check(cells, "coagulation", {"n_ait": 0.03, "n_acc": 0.03, "n_pcm": 0.03,
                                           "m_so4_ait": 0.03, "m_soa_ait": 0.03,
                                           "m_bc_pcm": 0.03, "m_poa_pcm": 0.03})
        self._check(cells, "ageing", {"n_acc": 0.10, "n_pcm": 0.10, "m_so4_acc": 0.05,
                                      "m_soa_acc": 0.10, "m_bc_pcm": 0.30, "m_poa_pcm": 0.30})
        self._check(cells, "full", {"g_h2so4": 0.02, "g_soag": 0.08,
                                    "n_ait": 0.05, "n_acc": 0.10, "n_pcm": 0.10,
                                    "m_so4_ait": 0.05, "m_so4_acc": 0.05,
                                    "m_soa_acc": 0.10, "m_bc_pcm": 0.30})

    def test_every_process_matches_the_reference_when_both_states_are_dry(self):
        """Measured 2026-10-02 (design doc §9, AER-12); tolerances ~2× the measurement."""
        self._check_every_process(humidity=1.5e-4)

    def test_every_process_matches_the_reference_at_mixed_humidity(self):
        """The same cells at the fixture's mixed humidity (D-A6)."""
        self._check_every_process(humidity=None)

    def test_npf_in_the_vapour_limited_regime_books_the_same_particles(self):
        """Empty modes (survival 1), enough vapour that formation consumes it all:
        the number booked per kg of sulfate is MAM's (pure-sulfate molar
        mass over the particle's mass at the Aitken lower bound).
        """
        from jcm.physics.aerosol.jam.tracer_layout import gas_name
        from jcm.physics_interface import PhysicsState

        shape = (3, 3)
        rng = np.random.default_rng(3)
        tracers = _consistent_tracers(rng, shape, empty=True)
        # Vapour well past the limit: at 1e-11 kg/kg and 265 K the reference
        # sits in the transition (its formation self-limits progressively
        # where ours and E3SM's one-shot take the whole vapour; design doc).
        tracers[gas_name("h2so4")] = jnp.asarray(
            np.repeat(np.array([[1e-10, 3e-10, 1e-9]]), 3, axis=0))
        temperature = jnp.asarray(np.repeat(np.array([[235.0], [250.0], [265.0]]), 3, axis=1))
        state = PhysicsState.zeros(shape).copy(
            temperature=temperature, specific_humidity=jnp.full(shape, 2e-3), tracers=tracers)
        diagnostics = {"pressure_full": jnp.full(shape, 8e4), "height_full": jnp.full(shape, 1.5e3),
                       "height_half": jnp.zeros((4, 3)), "_dt_seconds": DT}
        ref_ex = self._reference(state, diagnostics, EXCHANGE)
        ref_np = self._reference(state, diagnostics, {**EXCHANGE, "mdo_newnuc": 1})
        our_ex = self._ours(state, diagnostics, ("gas_aerosol_exchange",))
        our_np = self._ours(state, diagnostics, ("gas_aerosol_exchange", "nucleation"))
        reference = ref_np["n_ait"] - ref_ex["n_ait"]
        ours = our_np["n_ait"] - our_ex["n_ait"]
        self.assertTrue(np.all(reference > 0.0))
        np.testing.assert_allclose(ours / reference, 1.0, atol=0.02)

    def test_binary_formation_rate_is_the_same_function(self):
        """Vehkamäki et al. (2002) as both cores evaluate it, on a (T, RH, c) grid."""
        from mam4_jax.physics import newnuc as reference

        from jcm.physics.aerosol.jam.microphysics import new_particle_formation as npf

        # The reference's polynomial fit underflows to an exact zero in
        # float32 on some CPUs (CI's runners; not every machine), so the
        # function comparison runs in float64. ``tearDown`` restores the flag.
        jax.config.update("jax_enable_x64", True)
        for temperature in (236.0, 250.0, 265.0, 280.0, 295.0):
            for rh in (0.3, 0.6):
                for conc in (1e6, 1e7, 1e8, 1e9):
                    rate, n_h2so4, radius = npf.binary_formation(
                        jnp.asarray(temperature), jnp.asarray(rh), jnp.asarray(conc))
                    ref = reference.binary_nuc_vehk2002(
                        jnp.asarray(temperature), jnp.asarray(rh), jnp.asarray(conc))
                    ref_rate, ref_n, ref_radius_nm = float(ref[0]), float(ref[2]), float(ref[-1])
                    self.assertGreater(ref_rate, 0.0, (temperature, rh, conc))
                    self.assertAlmostEqual(float(rate) / ref_rate, 1.0, places=3)
                    # One trailing digit of one coefficient differs between
                    # the Fortran and the documented fit (−1.4177e-11 vs
                    # −1.42177e-11 on the n_tot cubic term): ≤0.6 % on size.
                    self.assertAlmostEqual(float(n_h2so4) / ref_n, 1.0, delta=1e-2)
                    self.assertAlmostEqual(float(radius) * 1e9 / ref_radius_nm, 1.0, delta=1e-2)

    # --- water uptake: the state itself -----------------------------------

    def test_kohler_solve_is_mams(self):
        """``modal_aero_kohler`` and ours on a (r_dry, κ, s) grid."""
        from mam4_jax.physics import kohler as reference

        from jcm.physics.aerosol.jam.microphysics import koehler

        # The reference's Ferrari solve needs float64: in float32 it finds no
        # admissible root at the Kelvin-dominated corner and falls back to the
        # dry radius. ``tearDown`` restores the flag.
        jax.config.update("jax_enable_x64", True)
        for r_dry in (5e-9, 30e-9, 200e-9, 2e-6):
            for kappa in (0.05, 0.5, 1.16):
                for s in (0.4, 0.8, 0.95, 0.98):
                    ours = float(koehler.kohler_growth_factor(
                        jnp.asarray(r_dry), jnp.asarray(kappa), jnp.asarray(s))) * r_dry
                    ref = float(np.asarray(reference.modal_aero_kohler(
                        jnp.asarray([r_dry]), jnp.asarray([kappa]), jnp.asarray([s])))[0])
                    # MAM's rounded constants (R = 8.3e7, M_w = 18) put its Kelvin
                    # coefficient 0.2 % above jcm's; at 5 nm that is visible.
                    self.assertAlmostEqual(ours / ref, 1.0, delta=3e-3, msg=(r_dry, kappa, s))

    def test_wet_radius_and_density_agree_mode_by_mode(self):
        """The state the harness terms read, against the adapter's, at mixed RH.

        Both cores publish the END-of-step state (the adapter since
        mam4-jax 0.5.1 / jax-gcm#1117, this core from AER-01), so the
        comparison switches every reference process off: what is left is
        MAM's ``calcsize`` + ``wateruptake`` on the input population against
        our κ-Köhler state on the same population.
        """
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.microphysics import mam4_jax as adapter
        from jcm.physics.aerosol.jam.microphysics.modal import JamModalMicrophysics

        rng = np.random.default_rng(11)
        state, diagnostics = _state(rng)
        original = adapter.amicphys
        adapter.amicphys = functools.partial(original, **OFF)
        try:
            _, ref = adapter.Mam4JaxMicrophysics()(state, diagnostics, None, None)
        finally:
            adapter.amicphys = original
        _, our = JamModalMicrophysics(processes=())(state, diagnostics, None, None)
        for i, mode in enumerate(MAM4_SPEC.modes):
            for field, tol in (("r_wet", 0.01), ("rho", 0.01)):
                ratio = np.asarray(getattr(our["_jam_state"], field)[i]
                                   / getattr(ref["_jam_state"], field)[i])
                self.assertTrue(np.all(np.abs(ratio - 1.0) < tol),
                                f"{mode.short}/{field}: ours/ref in {ratio.min():.4f}..{ratio.max():.4f}")


if __name__ == "__main__":
    unittest.main()
