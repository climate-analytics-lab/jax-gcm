"""Tests for refractive indices and the JamOpticsTerm."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.optics.refractive_index import refractive_index_at
from jcm.physics.radiation.band_config import RadiationBandConfig


class RefractiveIndexTest(unittest.TestCase):
    def test_black_carbon_strongly_absorbing(self):
        _, k = refractive_index_at("bc", jnp.array([550.0]))
        self.assertGreater(float(k[0]), 0.5)

    def test_sulfate_transparent_in_sw_absorbing_in_lw(self):
        _, k_sw = refractive_index_at("so4", jnp.array([550.0]))
        _, k_lw = refractive_index_at("so4", jnp.array([10000.0]))
        self.assertLess(float(k_sw[0]), 1e-6)
        self.assertGreater(float(k_lw[0]), 1e-2)

    def test_interp_shape(self):
        n, k = refractive_index_at("du", jnp.array([300.0, 550.0, 10000.0]))
        self.assertEqual(n.shape, (3,))
        self.assertTrue(np.all(np.isfinite(np.asarray(n))))


def _setup(nlev=4, ncols=3, n_sw=3, n_lw=2):
    from jcm.physics.aerosol.aerosol_types import AerosolData
    from jcm.physics_interface import PhysicsState

    n_modes = MAM4_SPEC.n_modes()
    shape = (n_modes, nlev, ncols)
    aer = JamAerosolState(
        r_dry=jnp.full(shape, 0.1e-6),
        r_wet=jnp.full(shape, 0.2e-6),
        rho=jnp.full(shape, 1800.0),
        kappa=jnp.full(shape, 0.5),
        mass=jnp.full(shape, 1e-9),
        number=jnp.full(shape, 1.0e8),
    )
    tracers = {}
    for mode in MAM4_SPEC.modes:
        tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
        for sp in mode.species:
            tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1e-9)
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 285.0), tracers=tracers,
    )
    sw = tuple(float(x) for x in np.linspace(400.0, 1000.0, n_sw))
    lw = tuple(float(x) for x in np.linspace(8000.0, 20000.0, n_lw))
    band = RadiationBandConfig(lw_band_centers_nm=lw, sw_band_centers_nm=sw)
    aerosol = AerosolData.zeros((ncols,), nlev, n_bnd_sw=n_sw, n_bnd_lw=n_lw)
    diagnostics = {
        "_jam_state": aer,
        "aerosol": aerosol,
        "air_density": jnp.full((nlev, ncols), 1.0),
        "layer_thickness": jnp.full((nlev, ncols), 500.0),
        "_band_config": band,
    }
    return state, diagnostics, band, n_sw, n_lw


def _consistent_state(state, diagnostics, saturation=0.8, center_modes=False):
    """Replace ``_jam_state`` with the κ-Köhler state implied by the tracers.

    ``_setup`` sets ``mass``, ``number``, ``r_dry`` and ``r_wet``
    independently of the tracers it writes, which is fine for plumbing tests
    but not for any test about the modal GEOMETRY: the optics read the dry
    volume from the mass tracers and the radii/number from ``_jam_state``,
    and only a state the microphysics core actually produced makes the two
    agree. This runs the placeholder core's diagnosis on the fixture's
    tracers, so ``r_wet = r_dry·g`` at the κ-Köhler growth factor for
    ``saturation`` — which is all the water volume depends on.

    ``r_dry`` is a different matter. With ``_setup``'s own numbers the core
    CLIPS three of the four modes (Aitken and primary carbon at ``dgnum_hi``,
    coarse at ``dgnum_lo``), so ``r_dry`` is the clipped radius, not the
    solution of ``V = N (π/6) Dg³ exp(4.5 ln²σ)``. Pass ``center_modes=True``
    to pick, per mode, the number that puts ``dg`` exactly on ``mode.dgnum``
    — strictly inside every bound, so no clip is active and that identity
    holds for all four modes. Only tests that assert something about the
    third moment itself need it; the water volume is a pure ratio and does
    not care either way.
    """
    from jcm.physics.aerosol.jam.microphysics.placeholder import (
        equilibrium_modal_state,
    )
    masses = {k: v for k, v in state.tracers.items() if k.startswith("m_")}
    numbers = {k: v for k, v in state.tracers.items() if k.startswith("n_")}
    if center_modes:
        numbers = {}
        for mode in MAM4_SPEC.modes:
            vol = sum(masses[mass_name(sp, mode.short)]
                      / MAM4_SPEC.species_props(sp).density
                      for sp in mode.species)
            k = (np.pi / 6.0) * np.exp(4.5 * np.log(mode.geom_std_dev) ** 2)
            numbers[number_name(mode.short)] = vol / (k * mode.dgnum ** 3)
    sat = jnp.full(state.temperature.shape, saturation)
    aer = equilibrium_modal_state(masses, numbers, MAM4_SPEC, sat)
    return {**diagnostics, "_jam_state": aer}


class JamOpticsTermTest(unittest.TestCase):
    def _term(self, band):
        term = JamOpticsTerm()
        term.cache_band_config(band)
        return term

    def test_no_aerosol_radiative_effect_above_pmin(self):
        """Levels above _AER_RAD_PMIN must carry exactly zero tau — the thin
        lid otherwise turns any absorbed flux into unbounded heating once
        real absorbers mix up there (day-207 winter blow-up).
        """
        state, diagnostics, band, n_sw, n_lw = _setup()
        nlev = state.temperature.shape[0]
        # Top level above the cutoff, the rest well below it.
        p = np.full((nlev,) + state.temperature.shape[1:], 5.0e4)
        p[0] = 100.0    # < _AER_RAD_PMIN
        term = self._term(band)
        _, out = term(state, {**diagnostics,
                              "pressure_full": jnp.asarray(p)}, None, None)
        a = out["aerosol"]
        np.testing.assert_array_equal(np.asarray(a.aod_sw_per_band[:, 0]), 0.0)
        np.testing.assert_array_equal(np.asarray(a.aod_lw_per_band[:, 0]), 0.0)
        self.assertGreater(float(jnp.sum(a.aod_sw_per_band[:, 1:])), 0.0)

    def test_writes_grey_profile_fields(self):
        """The 550 nm profile fields grey radiation reads are populated (#640).

        With MACv2-SP gone from the JAM path, ``JamOpticsTerm`` is the only
        writer of ``aod_profile``/``ssa_profile``/``asy_profile``/``angstrom``
        — the fields the grey two-stream scheme band-scales for its direct
        effect. They must equal the SW band nearest 550 nm (band-centre
        approximation) and carry a finite Angstrom exponent.
        """
        state, diagnostics, band, n_sw, n_lw = _setup()
        term = self._term(band)
        _, out = term(state, diagnostics, None, None)
        a = out["aerosol"]
        idx = term._cache.aod_band_idx
        np.testing.assert_allclose(
            np.asarray(a.aod_profile), np.asarray(a.aod_sw_per_band[idx]),
            rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            np.asarray(a.ssa_profile), np.asarray(a.ssa_sw_per_band[idx]))
        np.testing.assert_allclose(
            np.asarray(a.asy_profile), np.asarray(a.asy_sw_per_band[idx]))
        # A real burden was set up, so the profile carries optical depth.
        self.assertGreater(float(jnp.sum(a.aod_profile)), 0.0)
        # Angstrom is finite everywhere (band-ratio where AOD exists, the 1.5
        # fine-mode default elsewhere); ang_band differs from the 550 band so
        # the ratio path is exercised.
        self.assertNotEqual(term._cache.ang_band_idx, term._cache.aod_band_idx)
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(a.angstrom)))))

    def test_single_broadband_sw_uses_default_angstrom(self):
        """With a single broadband SW band (grey's own config) there is no
        550/865 ratio, so Angstrom falls back to the fine-mode 1.5 default
        while the profile still comes from the sole SW band (#640).
        """
        state, diagnostics, band, n_sw, n_lw = _setup(n_sw=1)
        term = self._term(band)
        self.assertEqual(term._cache.ang_band_idx, term._cache.aod_band_idx)
        _, out = term(state, diagnostics, None, None)
        a = out["aerosol"]
        np.testing.assert_array_equal(
            np.asarray(a.angstrom), np.full_like(np.asarray(a.angstrom), 1.5))
        np.testing.assert_allclose(
            np.asarray(a.aod_profile), np.asarray(a.aod_sw_per_band[0]))

    def test_radiation_gate_replays_cache_between_compute_steps(self):
        """With the gate configured, non-radiation steps must reuse the
        cached per-band fields (the radiation term can't see fresh optics
        until its next compute step anyway), and radiation-compute steps
        must recompute fresh.
        """
        import dataclasses

        state, diagnostics, band, n_sw, n_lw = _setup()
        term = self._term(band)
        term.configure_radiation_gate(7200.0)   # 8 steps at dt=900

        @dataclasses.dataclass
        class _Rad:
            step: jnp.ndarray

        base = {**diagnostics, "_dt_seconds": jnp.asarray(900.0)}
        # Step 0 (compute): no cache in the carry yet -> unconditional
        # compute that seeds the ``_jam_optics`` slot.
        _, d0 = term(state, {**base, "radiation": _Rad(jnp.int32(0))},
                     None, None)
        self.assertIn("_jam_optics", d0)

        # Step 1 (cached): perturb the aerosol state; output must equal the
        # step-0 fields, not fresh ones.
        aer2 = d0["_jam_state"].copy(number=d0["_jam_state"].number * 3.0)
        d1_in = {**base, "radiation": _Rad(jnp.int32(1)),
                 "_jam_state": aer2,
                 "_jam_optics": d0["_jam_optics"]}
        _, d1 = term(state, d1_in, None, None)
        np.testing.assert_array_equal(
            np.asarray(d1["aerosol"].aod_sw_per_band),
            np.asarray(d0["aerosol"].aod_sw_per_band),
        )

        # Step 8 (compute): the tripled number must now show (larger AOD).
        d8_in = {**d1_in, "radiation": _Rad(jnp.int32(8))}
        _, d8 = term(state, d8_in, None, None)
        self.assertGreater(
            float(jnp.sum(d8["aerosol"].aod_sw_per_band)),
            float(jnp.sum(d0["aerosol"].aod_sw_per_band)),
        )

    def test_writes_finite_bounded_optics(self):
        state, diagnostics, band, n_sw, n_lw = _setup()
        term = self._term(band)
        _, diag = term(state, diagnostics, None, None)
        a = diag["aerosol"]
        self.assertEqual(a.aod_sw_per_band.shape[0], n_sw)
        self.assertEqual(a.aod_lw_per_band.shape[0], n_lw)
        for arr in (a.aod_sw_per_band, a.aod_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            self.assertTrue(bool(jnp.all(arr >= 0.0)))
        for arr in (a.ssa_sw_per_band, a.ssa_lw_per_band):
            self.assertTrue(bool(jnp.all((arr >= 0.0) & (arr <= 1.0 + 1e-5))))
        for arr in (a.asy_sw_per_band, a.asy_lw_per_band):
            self.assertTrue(bool(jnp.all((arr >= -1.0 - 1e-5) & (arr <= 1.0 + 1e-5))))

    def test_negative_number_ringing_stays_bounded(self):
        """Negative modal number (spectral Gibbs ringing on the growing aerosol
        field) must not blow the extinction-weighted SSA/asymmetry out of
        range. Without flooring the number at 0 the band AOD goes ≤ 0, the SSA
        (= scat / AOD) and asymmetry diverge to ±huge, and RRTMGP's two-stream
        solver NaNs — this is the step-10 echam-jam blow-up. Drive some cells
        negative and assert every per-band optic stays finite and physical.
        """
        state, diagnostics, band, n_sw, n_lw = _setup()
        aer = diagnostics["_jam_state"]
        # Flip the sign of the modal number in a subset of cells (ringing).
        num = aer.number
        num = num.at[:, 0, :].set(-jnp.abs(num[:, 0, :]))
        num = num.at[:, :, 1].set(-1.0e7)
        diagnostics = {**diagnostics, "_jam_state": aer.copy(number=num)}
        # And the corresponding number tracers: the ringing field this models
        # drives both negative together. (The optics read the number only
        # through ``num_per_area`` — the water volume is number-free, #790 —
        # so this pairing is realism, not a load-bearing input.)
        tracers = dict(state.tracers)
        for mode in MAM4_SPEC.modes:
            nm = number_name(mode.short)
            t = tracers[nm].at[0, :].set(-1.0e7)
            tracers[nm] = t.at[:, 1].set(-1.0e7)
        state = state.copy(tracers=tracers)

        term = self._term(band)
        _, diag = term(state, diagnostics, None, None)
        a = diag["aerosol"]
        for arr in (a.aod_sw_per_band, a.aod_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            self.assertTrue(bool(jnp.all(arr >= 0.0)))           # AOD floored
        for arr in (a.ssa_sw_per_band, a.ssa_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            self.assertTrue(bool(jnp.all((arr >= 0.0) & (arr <= 1.0 + 1e-5))))
        for arr in (a.asy_sw_per_band, a.asy_lw_per_band):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
            # Asymmetry is physically [-1, 1] (negative g = back-scattering).
            self.assertTrue(bool(jnp.all((arr >= -1.0 - 1e-5) & (arr <= 1.0 + 1e-5))))
        self.assertTrue(np.all(np.isfinite(np.asarray(diag["_jam_optics"]["aod_550"]))))

    def test_empty_levels_carry_exactly_zero_tau(self):
        """POSITIVE-side ringing: tiny +ve number with a garbage wet radius
        must give EXACTLY zero tau where the mode has no mass. n*q_ext*
        pi*r^2 was finite at empty levels, and the 1/dp amplification at
        the 1 Pa model top produced 13,000 K/day of spurious SW
        absorption — +90 K in 6 h and a global NaN by day 10 of the
        first coupled JAM year. The mass gate (vol_tot > 1e-24 m3/kg)
        pins tau to zero there.
        """
        state, diagnostics, band, n_sw, n_lw = _setup()
        aer = diagnostics["_jam_state"]
        # Ringing-like state: tiny positive number, inflated wet radius,
        # zero species mass everywhere.
        diagnostics = {**diagnostics, "_jam_state": aer.copy(
            number=jnp.full_like(aer.number, 1.0e-6),
            r_wet=jnp.full_like(aer.r_wet, 5.0e-5),
        )}
        tracers = {k: (jnp.zeros_like(v) if k.startswith("m_") else v)
                   for k, v in state.tracers.items()}
        state = state.copy(tracers=tracers)

        term = self._term(band)
        _, diag = term(state, diagnostics, None, None)
        a = diag["aerosol"]
        self.assertEqual(float(jnp.max(jnp.abs(a.aod_sw_per_band))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(a.aod_lw_per_band))), 0.0)

    def test_more_aerosol_more_aod(self):
        """Doubling the aerosol mass at fixed number grows every particle and
        so must raise the extinction.

        The modal state must be re-diagnosed from the doubled tracers, not
        held fixed: the per-particle cross-section comes from ``_jam_state``
        (number and wet radius), so doubling the mass tracers ALONE leaves
        the extinction exactly unchanged — mass enters only through the
        volume-mixed refractive index, whose species ratios and water
        fraction are both scale-free (#790). Pairing doubled tracers with a
        fixed state therefore asserts nothing about "more aerosol".
        """
        state, diagnostics, band, *_ = _setup()
        term = self._term(band)
        d_in1 = _consistent_state(state, diagnostics)
        _, d1 = term(state, d_in1, None, None)
        mass2 = {k: (2.0 * v if k.startswith("m_") else v)
                 for k, v in state.tracers.items()}
        state2 = state.copy(tracers=mass2)
        d_in2 = _consistent_state(state2, diagnostics)
        _, d2 = term(state2, d_in2, None, None)
        self.assertGreater(
            float(jnp.sum(d2["aerosol"].aod_sw_per_band)),
            float(jnp.sum(d1["aerosol"].aod_sw_per_band)),
        )

    def test_refractive_index_independent_of_number(self):
        """At fixed radii and masses, the modal number scales the extinction
        but must not touch the volume-mixed refractive index.

        The dry volume in the mixing rule comes from the mass tracers, so a
        water volume carrying an ``N`` of its own — as ``N·(4/3)π·(r_wet³ −
        r_dry³)`` does — would make the water FRACTION, and with it the SSA,
        move with a quantity the dry volume never saw (#790). With the
        number-free form the SSA and asymmetry are invariant and the AOD is
        exactly linear in the number.
        """
        state, diagnostics, band, *_ = _setup()
        term = self._term(band)
        aer = diagnostics["_jam_state"]
        _, d1 = term(state, diagnostics, None, None)
        # The fixture peaks near tau = 0.13 per layer and band, so tripling
        # the number stays clear of the ``_MAX_LAYER_TAU`` = 1 clamp and the
        # scaling below is exact rather than clipped.
        d_in3 = {**diagnostics, "_jam_state": aer.copy(number=3.0 * aer.number)}
        _, d3 = term(state, d_in3, None, None)
        a1, a3 = d1["aerosol"], d3["aerosol"]
        np.testing.assert_allclose(
            np.asarray(a3.aod_sw_per_band), 3.0 * np.asarray(a1.aod_sw_per_band),
            rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(a3.ssa_sw_per_band), np.asarray(a1.ssa_sw_per_band),
            rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(a3.asy_sw_per_band), np.asarray(a1.asy_sw_per_band),
            rtol=1e-6)

    def test_water_depends_on_growth_ratio_not_on_number(self):
        """The water volume is set by ``r_wet/r_dry`` alone.

        Two cases with the number/radius pairing made grossly inconsistent on
        purpose (1000x the number the radii imply):

        * ``r_wet == r_dry`` — no growth, so exactly zero water, while the
          dry species still extinguish;
        * ``r_wet == 1.05·r_dry`` — one uniform growth factor across modes,
          so water's share of EVERY mode's volume, and hence of the column
          extinction it is apportioned by, is exactly ``(g³ − 1)/g³``.

        The second case is the discriminating one: ``N·(4/3)π·(r_wet³ −
        r_dry³)`` would put the inflated ``N`` into the water while the dry
        volume it mixes with comes from the mass tracers, so it misses that
        share by orders of magnitude (#790). The first case alone does not
        discriminate — both forms vanish at zero growth.
        """
        state, diagnostics, band, *_ = _setup()
        aer = diagnostics["_jam_state"]
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)

        dry = aer.copy(r_wet=aer.r_dry, number=1.0e3 * aer.number)
        _, out = term(state, {**diagnostics, "_jam_state": dry}, None, None)
        np.testing.assert_array_equal(np.asarray(out["od550_wat"]), 0.0)
        np.testing.assert_array_equal(np.asarray(out["abs550_wat"]), 0.0)
        self.assertTrue(bool(np.all(np.asarray(out["od550aer"]) > 0.0)))

        g = 1.05
        grown = aer.copy(r_wet=g * aer.r_dry, number=1.0e3 * aer.number)
        _, out = term(state, {**diagnostics, "_jam_state": grown}, None, None)
        share = np.asarray(out["od550_wat"]) / np.asarray(out["od550aer"])
        np.testing.assert_allclose(share, (g ** 3 - 1.0) / g ** 3, rtol=1e-5)

    def test_degenerate_dry_radius_gives_zero_water_and_zero_gradient(self):
        """A core reporting ``r_dry = 0`` must give 0 water, not NaN, and must
        not leave a gradient on the degenerate branch.

        Defensive: no in-repo core produces ``r_dry = 0`` (the placeholder
        falls back to ``mode.dgnum``, MAM4-JAX takes ``0.5·dgncur_a``), so
        this pins the behaviour for an external or future core. The growth
        ratio ``(r_wet/r_dry)³`` is the one division the water volume makes,
        and it multiplies a ``vol_dry`` that is zero on exactly those modes.
        The ``jnp.where`` selects no growth there — value and gradient both
        zero — and ``_MIN_DRY_RADIUS`` keeps the ratio's cube finite in the
        arm the select discards, since both arms are evaluated and an ``inf``
        there would come back as a NaN in the reverse pass. float64 has the
        range to absorb a smaller floor, so only the default float32 physics
        exercises the overflow half of this.
        """
        if jax.config.read("jax_enable_x64"):
            self.skipTest("the float32 overflow guarded here cannot occur in x64")
        state, diagnostics, band, *_ = _setup()
        aer = diagnostics["_jam_state"]
        empty = aer.copy(r_dry=jnp.zeros_like(aer.r_dry))
        no_mass = state.copy(
            tracers={k: (jnp.zeros_like(v) if k.startswith("m_") else v)
                     for k, v in state.tracers.items()})
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        _, out = term(no_mass, {**diagnostics, "_jam_state": empty}, None, None)
        self.assertEqual(out["aerosol"].aod_sw_per_band.dtype, jnp.float32)
        for key in ("od550aer", "od550_wat", "abs550_wat"):
            np.testing.assert_array_equal(np.asarray(out[key]), 0.0)
        self.assertTrue(
            bool(np.all(np.isfinite(np.asarray(out["aerosol"].aod_sw_per_band)))))

        def water_from_mass(jam_state, scale):
            tr = {k: (scale * v if k.startswith("m_") else v)
                  for k, v in state.tracers.items()}
            _, o = term(state.copy(tracers=tr),
                        {**diagnostics, "_jam_state": jam_state}, None, None)
            return jnp.sum(o["od550_wat"])

        # A degenerate r_dry carrying REAL mass must still add no water. This
        # is the discriminating half: with a bare floor instead of the select
        # the growth ratio would be (r_wet/1e-12)³ and the mode would be
        # essentially all water.
        wet = lambda s: water_from_mass(empty, s)
        self.assertEqual(float(wet(jnp.float32(1.0))), 0.0)
        self.assertEqual(float(jax.grad(wet)(jnp.float32(1.0))), 0.0)

        # And an all-empty cell must leave a FINITE gradient. Making the water
        # proportional to vol_dry means vol_tot is exactly zero there, which a
        # floored denominator would turn into a NaN cotangent (0 x inf) and
        # poison the whole reverse pass; clean columns are most of a cold-start
        # aerosol field, so this is a hard requirement, not a corner case.
        clean = lambda s: water_from_mass(aer, s)
        self.assertEqual(float(jax.grad(clean)(jnp.float32(0.0))), 0.0)

    def test_column_aod_550_diagnostic(self):
        from jcm.physics.aerosol.aerosol_types import AerosolData
        from jcm.physics_interface import PhysicsState

        # Bands chosen so the 550 nm pick is unambiguous (500 nm is closest).
        nlev, ncols = 4, 3
        sw = (350.0, 500.0, 900.0)
        lw = (8000.0, 20000.0)
        band = RadiationBandConfig(lw_band_centers_nm=lw, sw_band_centers_nm=sw)
        term = self._term(band)
        self.assertEqual(term._cache.aod_band_idx, 1)   # 500 nm is closest to 550
        self.assertEqual(term._cache.aod_band_nm, 500.0)

        n_modes = MAM4_SPEC.n_modes()
        shape = (n_modes, nlev, ncols)
        aer = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6), r_wet=jnp.full(shape, 0.2e-6),
            rho=jnp.full(shape, 1800.0), kappa=jnp.full(shape, 0.5),
            mass=jnp.full(shape, 1e-9), number=jnp.full(shape, 1.0e8),
        )
        tracers = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1e-9)
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 285.0), tracers=tracers,
        )
        diagnostics = {
            "_jam_state": aer,
            "aerosol": AerosolData.zeros((ncols,), nlev, n_bnd_sw=3, n_bnd_lw=2),
            "air_density": jnp.full((nlev, ncols), 1.0),
            "layer_thickness": jnp.full((nlev, ncols), 500.0),
            "_band_config": band,
        }
        _, diag = term(state, diagnostics, None, None)

        aod = diag["_jam_optics"]["aod_550"]
        # Column field, one value per column; finite and physical.
        self.assertEqual(aod.shape, (ncols,))
        self.assertTrue(np.all(np.isfinite(np.asarray(aod))))
        self.assertTrue(bool(jnp.all(aod > 0.0)))
        # It is exactly the column sum of the 550 nm-band per-layer AOD.
        per_layer_550 = diag["aerosol"].aod_sw_per_band[1]
        np.testing.assert_allclose(
            np.asarray(aod), np.asarray(jnp.sum(per_layer_550, axis=0)),
            rtol=1e-6,
        )

    def test_lognormal_exceeds_monodisperse_extinction(self):
        # ``r_wet`` is the number-median radius: the lognormal quadrature
        # must beat a monodisperse cross-section by at least the r^2-moment
        # factor exp(2 ln^2 sigma) (1.55 at sigma=1.6); Qext weighting only
        # adds to it for sub-micron sizes at 550 nm.
        from jcm.physics.aerosol.jam.optics.mie_lut import (
            default_mie_lut,
            interp_mie,
        )

        state, diagnostics, band, *_ = _setup()
        term = self._term(band)
        _, diag = term(state, diagnostics, None, None)
        aod_lognormal = float(jnp.max(diag["_jam_optics"]["aod_550"]))

        # Monodisperse expectation: same LUT, same geometry, single radius.
        lut = default_mie_lut()
        lam = term._cache.aod_band_nm * 1e-9
        aer = diagnostics["_jam_state"]
        rho_dz = float(diagnostics["air_density"][0, 0]
                       * diagnostics["layer_thickness"][0, 0])
        nlev = state.temperature.shape[0]
        mono = 0.0
        for i in range(MAM4_SPEC.n_modes()):
            r = float(aer.r_wet[i, 0, 0])
            n_col = float(aer.number[i, 0, 0]) * rho_dz * nlev
            q = float(interp_mie(
                lut, jnp.asarray(2.0 * np.pi * r / lam),
                jnp.asarray(1.45), jnp.asarray(1e-2))[0])
            mono += n_col * q * np.pi * r ** 2
        self.assertGreater(aod_lognormal, 1.5 * mono)
        self.assertLess(aod_lognormal, 20.0 * mono)

    def test_grad_through_mass(self):
        state, diagnostics, band, *_ = _setup()
        term = self._term(band)
        key = mass_name("bc", "acc")

        def loss(scale):
            tr = {k: (v * scale if k == key else v) for k, v in state.tracers.items()}
            s = state.copy(tracers=tr)
            _, d = term(s, diagnostics, None, None)
            return jnp.sum(d["aerosol"].aod_sw_per_band)

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))


class OpticsDiagnosticsTest(unittest.TestCase):
    """AeroCom per-species / per-mode / spectral optics (jax-gcm#584)."""

    def _run(self, **kw):
        state, diagnostics, band, _, _ = _setup(**kw)
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        _, out = term(state, diagnostics, None, None)
        return term, out

    def test_off_by_default(self):
        """The diagnostic pass costs a second Mie sweep, so it must not run
        unless explicitly enabled.
        """
        state, diagnostics, band, _, _ = _setup()
        term = JamOpticsTerm()
        term.cache_band_config(band)
        _, out = term(state, diagnostics, None, None)
        self.assertEqual(term.optics_diagnostic_keys(), ())
        self.assertNotIn("od550aer", out)

    def test_publishes_declared_keys(self):
        """``optics_diagnostic_keys`` is what the output layer registers, so
        it must match what ``__call__`` actually emits, exactly.
        """
        term, out = self._run()
        declared = set(term.optics_diagnostic_keys())
        self.assertTrue(declared)
        self.assertTrue(declared.issubset(set(out)),
                        f"missing: {sorted(declared - set(out))}")

    def test_key_set_is_data_independent(self):
        """The diagnostics dict is part of the ``lax.scan`` carry: a key set
        that varies with the data changes the carry pytree and the scan
        rejects it.
        """
        _, out_a = self._run()
        state, diagnostics, band, _, _ = _setup()
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        # Same structure, an aerosol-free atmosphere.
        empty = {k: jnp.zeros_like(v) for k, v in state.tracers.items()}
        _, out_b = term(state.copy(tracers=empty), diagnostics, None, None)
        self.assertEqual(sorted(out_a), sorted(out_b))

    def test_species_apportionment_closes(self):
        """Species are volume-mixed into one effective refractive index
        BEFORE the Mie call, so the per-species fields are an apportionment,
        not a decomposition. The one property that must hold exactly is that
        they add back up to the total — otherwise extinction is being
        dropped or double counted.
        """
        term, out = self._run()
        species = sorted({sp for m in MAM4_SPEC.modes for sp in m.species}) + ["wat"]
        tau_sum = sum(out[f"od550_{sp}"] for sp in species)
        abs_sum = sum(out[f"abs550_{sp}"] for sp in species)
        np.testing.assert_allclose(np.asarray(tau_sum),
                                   np.asarray(out["od550aer"]), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(abs_sum),
                                   np.asarray(out["abs550aer"]), rtol=1e-5)

    def test_mode_decomposition_closes(self):
        """Per-mode tau IS a true decomposition (each mode gets its own Mie
        call), so it must close to the total as well.
        """
        term, out = self._run()
        tau_sum = sum(out[f"od550_mode_{m.short}"] for m in MAM4_SPEC.modes)
        np.testing.assert_allclose(np.asarray(tau_sum),
                                   np.asarray(out["od550aer"]), rtol=1e-5)

    def test_absorption_bounded_by_extinction(self):
        term, out = self._run()
        self.assertTrue(np.all(np.asarray(out["abs550aer"])
                               <= np.asarray(out["od550aer"]) + 1e-12))
        ssa = np.asarray(out["ssa440aer"])
        self.assertTrue(np.all((ssa >= 0.0) & (ssa <= 1.0)))

    def test_angstrom_positive_for_fine_aerosol(self):
        """The fixture is 0.2 um wet radius — well inside the fine mode, so
        extinction must fall with wavelength (positive Angstrom exponent).
        This is the check that the diagnostic wavelengths are being applied
        in the right order and not silently collapsing to one value.
        """
        term, out = self._run()
        self.assertTrue(np.all(np.asarray(out["od440aer"])
                               > np.asarray(out["od865aer"])))
        self.assertTrue(np.all(np.asarray(out["ang4487aer"]) > 0.0))
        np.testing.assert_allclose(
            np.asarray(out["aerindex"]),
            np.asarray(out["od550aer"]) * np.asarray(out["ang4487aer"]), rtol=1e-6)

    def test_clean_column_is_finite_not_nan(self):
        """Aerosol-free columns hit 0/0 in SSA and log(0/0) in the Angstrom
        exponent; both must fall back to defined values rather than NaN.
        """
        state, diagnostics, band, _, _ = _setup()
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        empty = {k: jnp.zeros_like(v) for k, v in state.tracers.items()}
        aer0 = diagnostics["_jam_state"]
        d = {**diagnostics, "_jam_state": aer0.copy(
            mass=jnp.zeros_like(aer0.mass), number=jnp.zeros_like(aer0.number))}
        _, out = term(state.copy(tracers=empty), d, None, None)
        for k in term.optics_diagnostic_keys():
            self.assertTrue(np.all(np.isfinite(np.asarray(out[k]))), k)
        np.testing.assert_allclose(np.asarray(out["ssa440aer"]), 1.0)
        np.testing.assert_allclose(np.asarray(out["ang4487aer"]), 0.0)

    def test_extinction_profile_shape_and_units(self):
        """ec355aer is a 3-D extinction COEFFICIENT [m-1] = layer tau / dz,
        so integrating it back over dz must return the column AOD at 355.
        """
        state, diagnostics, band, _, _ = _setup()
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        _, out = term(state, diagnostics, None, None)
        ec = np.asarray(out["ec355aer"])
        self.assertEqual(ec.shape, state.temperature.shape)
        dz = np.asarray(diagnostics["layer_thickness"])
        np.testing.assert_allclose((ec * dz).sum(axis=0),
                                   np.asarray(out["od355aer"]), rtol=1e-5)

    def test_bc_dominates_absorption(self):
        """Absorption is apportioned by the species' contribution to the
        IMAGINARY index, so soot must carry the absorption even though it is
        a small volume fraction — the check that the apportionment is
        k-weighted and not accidentally volume-weighted like extinction.
        """
        term, out = self._run()
        abs_bc = float(np.sum(np.asarray(out["abs550_bc"])))
        abs_so4 = float(np.sum(np.asarray(out["abs550_so4"])))
        self.assertGreater(abs_bc, abs_so4)
        # ... while extinction, apportioned by volume, is comparable between
        # two species carried at equal mass and similar density.
        self.assertGreater(float(np.sum(np.asarray(out["od550_so4"]))), 0.0)

    def test_survives_the_radiation_gate(self):
        """The diagnostics ride in the same carry as the band optics, so both
        branches of the gate's ``lax.cond`` must produce the identical pytree
        — a nested diagnostic dict on one side only would fail to trace.
        """
        import dataclasses

        state, diagnostics, band, _, _ = _setup()
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        term.configure_radiation_gate(7200.0)

        @dataclasses.dataclass
        class _Rad:
            step: jnp.ndarray

        base = {**diagnostics, "_dt_seconds": jnp.asarray(900.0)}
        # Prime the carry (no cache yet -> unconditional compute).
        _, out0 = term(state, {**base, "radiation": _Rad(jnp.int32(0))}, None, None)
        # A replay step must go through lax.cond with the diagnostics present.
        _, out1 = term(state, {**base, "radiation": _Rad(jnp.int32(3)),
                               "_jam_optics": out0["_jam_optics"]},
                       None, None)
        for k in term.optics_diagnostic_keys():
            np.testing.assert_allclose(np.asarray(out1[k]), np.asarray(out0[k]),
                                       rtol=1e-6, err_msg=k)

    def test_closure_holds_without_strong_absorbers(self):
        """Absorption is apportioned by V_s*k_s / sum(V_s*k_s). Strip the
        strong absorbers and that denominator gets very small — a purely
        scattering sulfate/sea-salt population is the realistic stratospheric
        case, and the components must still close on the total there rather
        than falling off the 1/sum guard.
        """
        state, diagnostics, band, _, _ = _setup()
        tr = dict(state.tracers)
        for name in tr:
            if "_bc_" in name or "_du_" in name:
                tr[name] = jnp.zeros_like(tr[name])
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        _, out = term(state.copy(tracers=tr), diagnostics, None, None)
        species = sorted({sp for m in MAM4_SPEC.modes for sp in m.species}) + ["wat"]
        np.testing.assert_allclose(
            np.asarray(sum(out[f"od550_{sp}"] for sp in species)),
            np.asarray(out["od550aer"]), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(sum(out[f"abs550_{sp}"] for sp in species)),
            np.asarray(out["abs550aer"]), rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(np.asarray(out["abs550_bc"]), 0.0, atol=1e-20)

    def test_water_fraction_is_the_third_moment_growth(self):
        """The water share of a mode's extinction equals its VOLUME share,
        ``(g³ − 1)/g³`` with ``g = r_wet/r_dry`` — exact and Mie-independent.

        Extinction is apportioned by volume fraction and every particle of
        a κ-Köhler mode grows by the same ``g``, so the wet third moment is
        ``g³`` times the dry one and the water fraction of the mode's total
        volume is ``(g³ − 1)/g³`` regardless of ``σ_g``. The modes carry
        different κ, so the column total is the per-mode-AOD-weighted mean of
        the shares.

        ``center_modes=True`` keeps every mode strictly inside its ``dg``
        bounds, which is what makes the comparison against issue #790's table
        exact here: with no clip active, ``N·(4/3)π·(r_wet³ − r_dry³)`` on
        this state gives a water volume smaller by precisely
        ``exp(4.5 ln²σ_g)`` — 2.70 at σ_g = 1.6, 4.73 at σ_g = 1.8. (On a
        clipped mode the factor is ``(dg_clip/dg_true)³/exp(4.5 ln²σ_g)``
        instead, which can land either side of 1.) The assertion itself does
        not depend on that: it is a pure ratio and holds clipped or not.
        """
        state, diagnostics, band, _, _ = _setup()
        d_in = _consistent_state(state, diagnostics, saturation=0.8,
                                 center_modes=True)
        aer = d_in["_jam_state"]
        term = JamOpticsTerm(optics_diagnostics=True)
        term.cache_band_config(band)
        _, out = term(state, d_in, None, None)
        expected = np.zeros_like(np.asarray(out["od550aer"]))
        for i, mode in enumerate(MAM4_SPEC.modes):
            # The docstring's appeal to #790's table holds only off the clip
            # bounds, so assert that rather than trusting ``center_modes``.
            dg = 2.0 * np.asarray(aer.r_dry[i])
            self.assertTrue(bool(np.all(dg > mode.dgnum_lo)), mode.short)
            self.assertTrue(bool(np.all(dg < mode.dgnum_hi)), mode.short)
            g3 = np.asarray(aer.r_wet[i] / aer.r_dry[i]) ** 3      # (nlev, ncols)
            # Uniform saturation and species mix -> one g per mode.
            np.testing.assert_allclose(g3, g3.flat[0], rtol=1e-6)
            share_i = (g3.flat[0] - 1.0) / g3.flat[0]
            self.assertGreater(share_i, 0.0)
            expected += share_i * np.asarray(out[f"od550_mode_{mode.short}"])
        np.testing.assert_allclose(np.asarray(out["od550_wat"]), expected,
                                   rtol=1e-5)
        # And the share is substantial for the soluble modes at RH 0.8, so on
        # this unclipped state the number-median-radius form would show as the
        # 2.70x/4.73x deficit of #790's table.
        share = np.asarray(out["od550_wat"]) / np.asarray(out["od550aer"])
        self.assertGreater(float(share.min()), 0.3)

    def test_angstrom_discriminates_fine_from_coarse(self):
        """Physical validation of the spectral pass, not just its plumbing.

        The Angstrom exponent is ~2-3 for accumulation-mode aerosol and
        falls towards ~0 for coarse particles, whose extinction is already
        in the geometric limit and so nearly wavelength-independent. Getting
        that ordering right requires the wavelengths, the Mie size parameter
        and the lognormal quadrature all to be consistent — a sign error or
        a swapped wavelength pair would still pass the closure tests.
        """
        def ang_for(r_wet_m):
            state, diagnostics, band, _, _ = _setup()
            aer = diagnostics["_jam_state"]
            d = {**diagnostics,
                 "_jam_state": aer.copy(r_wet=jnp.full_like(aer.r_wet, r_wet_m),
                                        r_dry=jnp.full_like(aer.r_dry, r_wet_m))}
            term = JamOpticsTerm(optics_diagnostics=True)
            term.cache_band_config(band)
            _, out = term(state, d, None, None)
            return float(np.asarray(out["ang4487aer"]).mean())

        fine = ang_for(0.05e-6)      # accumulation mode
        coarse = ang_for(2.0e-6)     # coarse mode
        self.assertGreater(fine, 1.5, f"fine-mode Angstrom too low: {fine}")
        self.assertLess(coarse, 0.5, f"coarse-mode Angstrom too high: {coarse}")
        self.assertGreater(fine, coarse)


class _VolumeExtinctionOptics(JamOpticsTerm):
    """A minimal per-VOLUME backend, to exercise the ``_mode_optics`` seam.

    Deliberately shaped like an emulator rather than like the default
    quadrature: it predicts an extinction per unit particle VOLUME, so it
    consumes ``vol_total``/``col_factor`` instead of ``num_per_area`` and never
    touches the Mie LUT. That is the normalisation the seam exists to support
    — anything that already speaks in per-particle cross-sections would fit
    the default path without a hook.

    The refractive index enters through ``m_k`` only so the test can tell that
    the mixed index really reached the backend.
    """

    KE_RHO = 2.0e6      # [1/m]
    SSA = 0.9
    ASY = 0.7

    def _build_mie_lut(self):
        return None

    def _mode_optics(self, inputs):
        ke_rho = self.KE_RHO * (1.0 + inputs.m_k)
        tau = ke_rho * inputs.vol_total * inputs.col_factor
        scat = self.SSA * tau
        return tau, scat, self.ASY * scat

    def _map_bands(self, one_band, lam_all, ri_j):
        return jax.lax.map(lambda xs: one_band(xs[0], xs[1]), (lam_all, ri_j))


class _VmapVolumeExtinctionOptics(_VolumeExtinctionOptics):
    """The same backend on the default (``vmap``) band map, for the A/B."""

    def _map_bands(self, one_band, lam_all, ri_j):
        return jax.vmap(one_band)(lam_all, ri_j)


class ModeOpticsSeamTest(unittest.TestCase):
    """The ``_mode_optics``/``_map_bands`` extension point (jax-gcm#791).

    An external package supplies an alternative Mie pathway by subclassing
    ``JamOpticsTerm`` and overriding these hooks; the base class keeps the
    modal geometry, the mass gate, the SSA/asymmetry weighting and the
    AeroCom apportionment, so none of that is duplicated. These tests pin the
    parts of the contract an out-of-tree subclass depends on.
    """

    def test_backend_optics_reach_the_aerosol_struct(self):
        """A per-volume backend's SSA/asymmetry survive to the band arrays."""
        state, diagnostics, band, n_sw, n_lw = _setup()
        diagnostics = _consistent_state(state, diagnostics)
        term = _VolumeExtinctionOptics()
        term.cache_band_config(band)
        fields = term._compute_fields(state, diagnostics)

        aod = np.asarray(fields["aod_sw_per_band"])
        self.assertEqual(aod.shape[0], n_sw)
        self.assertTrue(np.all(np.isfinite(aod)))
        self.assertGreater(float(aod.max()), 0.0)
        # Every populated cell is a mix of modes that all report the same
        # SSA/asymmetry, so the extinction-weighted averages are exactly them.
        lit = aod > 0.0
        np.testing.assert_allclose(
            np.asarray(fields["ssa_sw_per_band"])[lit],
            _VolumeExtinctionOptics.SSA, rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(fields["asy_sw_per_band"])[lit],
            _VolumeExtinctionOptics.ASY, rtol=1e-5)
        self.assertEqual(
            np.asarray(fields["aod_lw_per_band"]).shape[0], n_lw)

    def test_backend_skips_the_lut_build(self):
        """``_build_mie_lut`` is what saves a backend the ~4 s table build."""
        self.assertIsNone(_VolumeExtinctionOptics()._lut)
        self.assertIsNotNone(JamOpticsTerm()._lut)

    def test_band_map_hook_is_memory_strategy_only(self):
        """``lax.map`` and ``vmap`` over bands must agree to rounding.

        Not bit-for-bit: XLA lowers a batched and a scanned band axis
        differently, and the two associate the float32 arithmetic differently,
        which shows up at ~2 ulp. The point of the test is that the choice of
        band map is a memory strategy and nothing else — any difference beyond
        rounding would mean it had changed the physics.
        """
        state, diagnostics, band, _, _ = _setup()
        diagnostics = _consistent_state(state, diagnostics)
        out = {}
        for name, cls in (("map", _VolumeExtinctionOptics),
                          ("vmap", _VmapVolumeExtinctionOptics)):
            term = cls()
            term.cache_band_config(band)
            out[name] = term._compute_fields(state, diagnostics)
        for key in ("aod_sw_per_band", "ssa_sw_per_band", "asy_sw_per_band",
                    "aod_lw_per_band", "aod_profile"):
            np.testing.assert_allclose(
                np.asarray(out["map"][key]), np.asarray(out["vmap"][key]),
                rtol=1e-5, atol=0.0,
                err_msg=f"{key} differs between the two band maps")

    def test_base_class_still_owns_the_aerocom_decomposition(self):
        """Per-species apportionment works for a backend that never sees it."""
        state, diagnostics, band, _, _ = _setup()
        diagnostics = _consistent_state(state, diagnostics)
        term = _VolumeExtinctionOptics(optics_diagnostics=True)
        term.cache_band_config(band)
        _, out = term(state, diagnostics, None, None)

        total = np.asarray(out["od550aer"])
        self.assertGreater(float(total.mean()), 0.0)
        species = sorted({sp for m in MAM4_SPEC.modes for sp in m.species})
        parts = sum(np.asarray(out[f"od550_{sp}"]) for sp in species)
        parts = parts + np.asarray(out["od550_wat"])
        np.testing.assert_allclose(parts, total, rtol=1e-5)

    def test_backend_gradients_are_finite(self):
        """The seam's contract: finite derivatives through an aerosol mass.

        The dry-mass gate and the layer-tau cap downstream select and bound
        VALUES — they cannot repair a non-finite derivative produced inside
        ``_mode_optics``, because reverse mode multiplies a zero cotangent
        into whatever came back. This pins that a conforming backend's
        gradients survive the surrounding machinery.
        """
        state, diagnostics, band, _, _ = _setup()
        diagnostics = _consistent_state(state, diagnostics)
        term = _VolumeExtinctionOptics()
        term.cache_band_config(band)
        key = mass_name("so4", "acc")

        def loss(scale):
            tracers = dict(state.tracers)
            tracers[key] = tracers[key] * scale
            fields = term._compute_fields(state.copy(tracers=tracers),
                                          diagnostics)
            return jnp.sum(fields["aod_sw_per_band"])

        g = float(jax.grad(loss)(1.0))
        self.assertTrue(np.isfinite(g))
        self.assertGreater(g, 0.0)

    def test_replace_attaches_a_subclass_by_category(self):
        """The documented attachment route for an out-of-tree backend.

        A backend ships as a ``JamOpticsTerm`` subclass and is attached in
        Python, by category, on an assembled package — there is no in-repo
        registry or config key to add a backend to, deliberately, because the
        implementations live outside this repository.
        """
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(aerosol_module="jam", cloud_scheme="2m")
        swapped = physics.replace("aerosol_optics", _VolumeExtinctionOptics())

        default = [t for t in physics.terms if t.category == "aerosol_optics"]
        replaced = [t for t in swapped.terms if t.category == "aerosol_optics"]
        self.assertEqual(len(default), 1)
        self.assertEqual(len(replaced), 1)
        self.assertIsInstance(replaced[0], _VolumeExtinctionOptics)
        # Position preserved, so the JAM chain's validated ordering — the
        # optics sit after the microphysics core that writes ``_jam_state``
        # — survives the swap. ``ComposablePhysics`` would raise here if not.
        self.assertEqual([t.category for t in physics.terms],
                         [t.category for t in swapped.terms])


if __name__ == "__main__":
    unittest.main()
