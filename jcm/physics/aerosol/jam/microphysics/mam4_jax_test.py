"""Tests for the MAM4-JAX microphysics core wrapper (issue #490).

Needs the optional GPL-3.0 ``mam4-jax`` dependency (``pip install jcm[mam4]``,
marked ``requires_extra("mam4")``): skipped without it, and run by the
``extras-tests`` CI job, which installs it. The wrapper enables
``jax_enable_x64`` on construction, so each test restores the prior flag in
``tearDown`` to keep sibling tests on jcm's default float32. ``mam4_jax`` is
imported only inside the tests, never at collection, so collecting this
module cannot flip the flag for its neighbours (#729).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.requires_extra("mam4")


def _column_state(nlev=4, ncols=2):
    """Build a PhysicsState + diagnostics with seeded aerosol tracers."""
    from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
    from jcm.physics_interface import PhysicsState

    tracers = {}
    for mode in MAM4_SPEC.modes:
        tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
        tracers[number_name(mode.short, cloud_borne=True)] = jnp.full(
            (nlev, ncols), 1.0e6
        )
        for sp in mode.species:
            tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1.0e-10)
            tracers[mass_name(sp, mode.short, cloud_borne=True)] = jnp.full(
                (nlev, ncols), 1.0e-12
            )
    # Gas-phase precursors fed to the core.
    tracers["g_h2so4"] = jnp.full((nlev, ncols), 1.0e-12)
    tracers["g_soag"] = jnp.full((nlev, ncols), 1.0e-12)
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 280.0),
        specific_humidity=jnp.full((nlev, ncols), 5.0e-3),
        tracers=tracers,
    )
    diagnostics = {
        "pressure_full": jnp.full((nlev, ncols), 9.0e4),
        "height_full": jnp.full((nlev, ncols), 3.0e3),
        # Needed only for the jam_organic_closure_residual diagnostic's
        # column integration (jax-gcm#1064 PR #1170 review); round,
        # synthetic values matching the rest of this fixture, not a real
        # sounding.
        "air_density": jnp.full((nlev, ncols), 1.0),
        "layer_thickness": jnp.full((nlev, ncols), 500.0),
        "_dt_seconds": 1800.0,
    }
    return state, diagnostics


def _column_state_with_negatives(nlev=4, ncols=3, h2so4=1.0e-9):
    """Like ``_column_state`` but with some NEGATIVE interstitial and
    cloud-borne mass AND NUMBER cells (spectral-advection Gibbs ringing,
    the issue #778 candidate mechanism (a): ``fetch()`` floors these at 0
    before packing -- a negative NUMBER cell exercises a different path
    than a negative mass cell, since ``calcsize`` derives modal diameters
    and the Aitken<->accum transfer trigger from ``num``, so flooring it
    changes the modal-size bookkeeping, not just one species' total) and
    a healthy H2SO4 reservoir so pcarbon ageing (the candidate mechanism
    (c): the pcm->accum repack) is actually active. Deterministic (no
    RNG) so the test is exact and reviewable.
    """
    from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
    from jcm.physics_interface import PhysicsState

    # A checkerboard of +-1 across the column so roughly half the cells of
    # every number/mass field go negative, the same "ringing" shape named
    # in the issue -- not a uniformly-signed field, which would only ever
    # probe one branch of ``jnp.maximum(..., 0.0)``.
    checker = np.where(
        (np.arange(nlev)[:, None] + np.arange(ncols)[None, :]) % 2 == 0,
        1.0, -1.0,
    )

    def field_checker(name):
        # A per-FIELD phase shift of the checkerboard (deterministic, from
        # the field name) so every mass field has BOTH positive and
        # negative cells -- a uniform scalar sign per field (the previous
        # version) left whole fields entirely negative, so fetch() floored
        # them to zero everywhere and e.g. the pcarbon ageing donors
        # m_poa_pcm/m_moa_pcm never had any positive mass to age (only BC
        # did), never exercising ageing's positive-cell path for them.
        return np.roll(checker, sum(map(ord, name)) % nlev, axis=0)

    tracers = {}
    for mode in MAM4_SPEC.modes:
        tracers[number_name(mode.short)] = jnp.asarray(checker * 1.0e8)
        tracers[number_name(mode.short, cloud_borne=True)] = jnp.asarray(
            -checker * 1.0e6)
        for sp in mode.species:
            m_name = mass_name(sp, mode.short)
            mc_name = mass_name(sp, mode.short, cloud_borne=True)
            tracers[m_name] = jnp.asarray(field_checker(m_name) * 1.0e-10)
            tracers[mc_name] = jnp.asarray(field_checker(mc_name) * 1.0e-12)
    tracers["g_h2so4"] = jnp.full((nlev, ncols), h2so4)
    tracers["g_soag"] = jnp.full((nlev, ncols), 1.0e-12)

    # Every pcarbon-mode species is an ageing donor (mam_pcarbon_aging_1
    # subarea moves its sulfate-coated fraction to accum): assert each one
    # has at least one POSITIVE cell, or ageing would have nothing to
    # move and the ageing-stress part of this fixture would be vacuous.
    pcm_mode = next(m for m in MAM4_SPEC.modes if m.short == "pcm")
    for sp in pcm_mode.species:
        donor = tracers[mass_name(sp, pcm_mode.short)]
        assert bool(jnp.any(donor > 0.0)), (
            f"pcarbon ageing donor {sp!r} has no positive cell -- fetch()"
            " would floor it to zero everywhere and ageing could never "
            "move any of its mass"
        )
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 280.0),
        specific_humidity=jnp.full((nlev, ncols), 5.0e-3),
        tracers=tracers,
    )
    diagnostics = {
        "pressure_full": jnp.full((nlev, ncols), 9.0e4),
        "height_full": jnp.full((nlev, ncols), 3.0e3),
        # Needed only for the jam_organic_closure_residual diagnostic's
        # column integration (jax-gcm#1064 PR #1170 review); round,
        # synthetic values matching the rest of this fixture, not a real
        # sounding.
        "air_density": jnp.full((nlev, ncols), 1.0),
        "layer_thickness": jnp.full((nlev, ncols), 500.0),
        "_dt_seconds": 1800.0,
    }
    return state, diagnostics


#: Species the MAM4 core has no source or sink for (only sulfate and SOA
#: do, via gas exchange/nucleation): their column-integrated mass must
#: be exactly invariant across one adapter call, per the module
#: docstring's conservation invariant (issue #778).
_NO_SOURCE_SPECIES = ("du", "ss", "bc", "poa", "moa")

#: float32 machine epsilon, the unit used to scale the f32 conservation
#: test's PER-CELL tolerance to that cell's own packed burden (see
#: ``test_no_source_species_conserve_mass_bounded_f32``).
_EPS32 = float(np.finfo(np.float32).eps)

#: The f32 conservation test's per-cell tolerance multiplier (k in
#: ``k * eps(float32) * burden_cell``) -- see
#: ``test_no_source_species_conserve_mass_bounded_f32`` for the
#: measurement that justifies it.
_F32_BURDEN_REL_K = 2.0


def _species_packed_burden(spec, tracers, carry, sp):
    """Per-cell POSITIVE packed burden of ``sp``: ``max(., 0)`` summed over
    its modes (interstitial + cloud-borne) -- the magnitude ``fetch()``
    actually hands the core at that cell, which is what bounds the
    core's float32 round-off there (a negative input is floored to 0
    before any arithmetic touches it, so it contributes no round-off).
    """
    from jcm.physics.aerosol.jam import mass_name

    total = None
    for mode in spec.modes:
        if sp not in mode.species:
            continue
        contribution = (
            np.maximum(
                np.asarray(tracers[mass_name(sp, mode.short)], np.float64),
                0.0)
            + np.maximum(
                np.asarray(
                    carry[mass_name(sp, mode.short, cloud_borne=True)],
                    np.float64),
                0.0)
        )
        total = contribution if total is None else total + contribution
    return total


def _species_column_mass(spec, tracers, carry, sp):
    """Per-cell mass of ``sp``: summed over ITS MODES (interstitial +
    cloud-borne) only -- shape ``(nlev, ncols)``, NOT reduced over level
    or column. The core's amicphys step is a per-cell box model (each
    (level, column) point is vmapped independently, no cross-cell mass
    exchange -- see the module docstring's "per-cell box model"
    description), so conservation must hold cell-by-cell; summing over
    level/column here would let an equal-and-opposite residual between
    two DIFFERENT cells cancel in the total while still failing the real
    per-cell invariant.
    """
    from jcm.physics.aerosol.jam import mass_name

    total = None
    for mode in spec.modes:
        if sp not in mode.species:
            continue
        contribution = (
            np.asarray(tracers[mass_name(sp, mode.short)], np.float64)
            + np.asarray(
                carry[mass_name(sp, mode.short, cloud_borne=True)],
                np.float64,
            )
        )
        total = contribution if total is None else total + contribution
    return total


class Mam4JaxAdapterTest(unittest.TestCase):
    def setUp(self):
        self._x64 = jax.config.read("jax_enable_x64")

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._x64)

    def test_factory_resolves_mam4_jax(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics(microphysics="mam4_jax")
        core = next(t for t in terms if t.category == "aerosol_microphysics")
        self.assertEqual(core.name, "jam_mam4_jax_microphysics")

    def test_gas_tracers_map_to_their_pcnst_slots(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        # Gases share the single q_pack with the aerosol tracers.
        packed = dict(Mam4JaxMicrophysics()._q_pack)
        self.assertEqual(packed["g_h2so4"], 6)
        self.assertEqual(packed["g_soag"], 9)

    def test_cam6_preset_selects_astem(self):
        from jcm.physics.aerosol.jam.jam_terms import _resolve_microphysics
        term = _resolve_microphysics("mam4_jax_astem")
        self.assertEqual(term._condensation_backend, "astem")

    def test_upstream_sources_are_processed_once(self):
        """Sequential chemistry/emissions equal explicitly updated input.

        SOAG made this timestep must condense this timestep. Returning a
        tendency against the step-start input would count the source twice.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import Mam4JaxMicrophysics

        state, diagnostics = _column_state(nlev=1, ncols=1)
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        diagnostics[CARRY_KEY] = {
            k: q for k, q in state.tracers.items() if k.startswith(("mc_", "nc_"))}
        dt = diagnostics["_dt_seconds"]
        sources = {
            "g_soag": jnp.full((1, 1), 1e-9 / dt),
            "m_du_cor": jnp.full((1, 1), 1e-8 / dt),
            "n_cor": jnp.full((1, 1), 1e7 / dt),
        }
        working = state.copy(tracers={
            k: q + dt * sources.get(k, 0.0) for k, q in state.tracers.items()})
        term = Mam4JaxMicrophysics()
        expected, expected_diag = term(working, diagnostics, None, None)
        actual, actual_diag = term(
            state, {**diagnostics, "_tendency_run": {"tracers": sources}},
            None, None)
        for name in expected.tracers:
            np.testing.assert_allclose(actual.tracers[name], expected.tracers[name],
                                       rtol=1e-6, atol=1e-25)
        np.testing.assert_allclose(actual_diag["_jam_state"].r_wet,
                                   expected_diag["_jam_state"].r_wet)
        # CAM's captured conversion factors use 12.011/150 for SOAG and
        # 12/150 for aerosol SOA. Close the common molecular basis, rather
        # than treating their 0.09% convention difference as mass creation.
        from mam4_jax.core import data
        gas_index = dict(term._q_pack)["g_soag"]
        soa_index = dict(term._q_pack)["m_soa_acc"]
        gas_to_soa = (data.MMR_TO_VMR[gas_index] * data.FCVT_GAS[0] /
                      (data.MMR_TO_VMR[soa_index] * data.FCVT_AER[0]))
        soa_delta = gas_to_soa * np.asarray(actual.tracers["g_soag"], np.float64)
        soa_delta += sum(np.asarray(v, np.float64) for k, v in actual.tracers.items()
                         if k.startswith("m_soa_"))
        soa_delta += sum(
            np.asarray(actual_diag[CARRY_KEY][k] - q, np.float64) / dt
            for k, q in diagnostics[CARRY_KEY].items() if k.startswith("mc_soa_"))
        np.testing.assert_allclose(soa_delta, 0.0, atol=1e-19)

    def test_removal_geometry_matches_post_microphysics_population(self):
        """The radius's third moment must describe the mass/number returned.

        Coagulation changes number and condensation changes volume; carrying
        their pre-step radii into settling would use a different population.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import Mam4JaxMicrophysics
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name

        state, diagnostics = _column_state(nlev=1, ncols=1)
        term = Mam4JaxMicrophysics()
        tendency, result = term(state, diagnostics, None, None)
        dt = diagnostics["_dt_seconds"]
        aer = result["_jam_state"]
        for i, mode in enumerate(MAM4_SPEC.modes):
            def updated(name):
                return state.tracers[name] + dt * tendency.tracers[name]
            volume = sum(updated(mass_name(sp, mode.short)) /
                         MAM4_SPEC.species_props(sp).density for sp in mode.species)
            number = updated(number_name(mode.short))
            diameter = (6 * volume / (np.pi * number)) ** (1 / 3) * np.exp(
                -1.5 * np.log(mode.geom_std_dev) ** 2)
            np.testing.assert_allclose(2 * aer.r_dry[i],
                np.clip(diameter, mode.dgnum_lo, mode.dgnum_hi), rtol=2e-5)

    def test_cam6_soag_produces_fine_mode_soa(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import Mam4JaxMicrophysics
        state, diagnostics = _column_state(nlev=1, ncols=1)
        tracers = {k: (jnp.zeros_like(q) if "soa_" in k else q)
                   for k, q in state.tracers.items()}
        tracers["g_soag"] = jnp.full((1, 1), 1e-8)
        state = state.copy(tracers=tracers)
        tendency, _ = Mam4JaxMicrophysics()(state, diagnostics, None, None)
        self.assertLess(float(tendency.tracers["g_soag"][0, 0]), 0.)
        self.assertGreater(float(tendency.tracers["m_soa_acc"][0, 0]), 0.)
        np.testing.assert_array_equal(tendency.tracers["m_soa_cor"], 0.)

    def _assert_organic_budget_closes(self, backend):
        """Shared body for the cold/evaporating-cell organic-closure check.

        Eight cells: four temperatures (260/280/298/310 K) crossed with
        aerosol-poor+gas-rich (condensing) and aerosol-rich+gas-poor
        (evaporating) SOA/SOAG. The warm, aerosol-rich, gas-poor cell (310 K)
        is exactly the "warm, aerosol-rich cell" #1064 describes: the
        substep backend's frozen-flux SOA exchange can evaporate past a
        mode's reservoir and, pre-fix, clip the aerosol without returning
        the shortfall to the gas. ASTEM's own semi-implicit corrector never
        had this defect (its step 2 backs out the new gas from the
        conserved total), so it is the reference bar: 2e-6 relative.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import Mam4JaxMicrophysics
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        state, diagnostics = _column_state(nlev=1, ncols=8)
        tracers = {k: (jnp.zeros_like(q) if "soa_" in k else q)
                   for k, q in state.tracers.items()}
        tracers["m_soa_acc"] = jnp.asarray([[1e-10] * 4 + [1e-8] * 4])
        tracers["g_soag"] = jnp.asarray([[1e-8] * 4 + [1e-12] * 4])
        state = state.copy(temperature=jnp.asarray([[260., 280., 298., 310.] * 2]),
                           tracers=tracers)
        dt = diagnostics["_dt_seconds"] = 720.
        carry = {k: q for k, q in tracers.items() if k.startswith(("mc_", "nc_"))}
        diagnostics[CARRY_KEY] = carry
        tendency, result = Mam4JaxMicrophysics(condensation_backend=backend)(
            state, diagnostics, None, None)
        delta = (12.011 / 12 * np.asarray(tendency.tracers["g_soag"], np.float64)
                 + sum(np.asarray(q, np.float64) for k, q in tendency.tracers.items()
                       if k.startswith("m_soa_"))) * dt
        delta += sum(np.asarray(result[CARRY_KEY][k] - q, np.float64)
                     for k, q in carry.items() if k.startswith("mc_soa_"))
        total = 12.011 / 12 * np.asarray(tracers["g_soag"], np.float64)
        total += sum(np.asarray(q, np.float64) for k, q in tracers.items()
                     if "soa_" in k)
        np.testing.assert_allclose(delta / total, 0., atol=2e-6)

    def test_organic_budget_closes_in_cold_and_evaporating_cells(self):
        self._assert_organic_budget_closes("astem")

    def test_organic_budget_closes_in_cold_and_evaporating_cells_substep(self):
        """jax-gcm#1064: the default, differentiable fixed-substep backend
        must close the same organic budget ASTEM does, at the same
        tolerance.

        The real fix is a bounded corrector inside the upstream kernel
        itself (``mam4_jax.coupling.amicphys``, reflective-org/MAM4-JAX#85,
        branch ``fix/substep-condensation-organic-closure``), not yet
        released. Until a mam4-jax release carrying it is pinned, jcm
        closes the same budget at step level
        (``Mam4JaxMicrophysics._close_organic_budget``, gated by
        ``_organic_closure_active`` on the installed ``mam4_jax.__version__``
        — see its docstring and ``docs/source/design/jam_soa_dust.md``), so
        this now asserts unconditionally like the ASTEM test above.
        """
        self._assert_organic_budget_closes("substep")

    def test_organic_budget_closure_is_noop_when_budget_already_closes(self):
        """The step-level closure (jax-gcm#1064) must not perturb a scenario
        that already conserves organic mass: calling
        ``_close_organic_budget`` directly on a before/after pair where the
        organic total is unchanged (the common case — e.g. a cold cell
        where no mode's reservoir is ever exhausted, or any state produced
        by the ASTEM backend, which never has this defect since its step-2
        solve is already self-consistent) must return ``q_new`` untouched.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        term = Mam4JaxMicrophysics(condensation_backend="substep")
        self.assertTrue(term._organic_closure_active)

        q_old = jnp.zeros((8, term._pcnst), dtype=jnp.float64)
        q_old = q_old.at[:, term._soag_pcnst_idx].set(1e-8)
        for idx in term._soa_mode_pcnst_idxs:
            q_old = q_old.at[:, idx].set(1e-10)
        # A budget-neutral change: move mass from the gas into ONE mode's
        # SOA, in the gas-to-aerosol-equivalent units the closure itself
        # uses, so the organic total is unchanged -- exactly what a
        # genuinely-conserving condensation/evaporation step produces.
        moved = 2.0e-9
        soa0 = term._soa_mode_pcnst_idxs[0]
        q_new = q_old.at[:, term._soag_pcnst_idx].add(
            -moved / term._organic_gas_to_aer_factor)
        q_new = q_new.at[:, soa0].add(moved)

        closed, residual = term._close_organic_budget(q_old, q_new)
        # float64 round-off only (the budget-neutral construction above is
        # exact only up to the ULPs of the intervening +/-/* on q_new
        # itself) -- not a correction: ``excess`` must be exactly clipped
        # to 0 by the ``jnp.maximum`` floor, so any discrepancy here can
        # only be that round-off, never a fraction of ``moved``.
        np.testing.assert_allclose(closed, q_new, rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(residual, 0.0, atol=1e-20)

    def test_organic_budget_closure_tolerates_float32_roundoff(self):
        """PR #1170 review round 3 (Codex P2): ``total_before``/
        ``total_after`` are each an independently-rounded floating-point
        sum, so a genuinely budget-neutral exchange (no floor hit
        anywhere) can leave them a few ULPs apart even with nothing
        physically wrong -- negligible at float64 (~2e-16/ULP), not at
        ``core_dtype="float32"`` (~1e-7/ULP, a forward-only production
        run may select it for speed). Without a dtype-scaled tolerance,
        that roundoff reads as a tiny "excess" and gets removed from
        SOAG; because roundoff is as likely to be negative (silently
        ignored by the ``max(0, ...)`` floor already) as positive, only
        ever correcting the positive side turns it into a one-sided
        organic-mass sink repeated every timestep, with no reservoir
        floor ever actually hit.

        Runs several successive budget-neutral "steps" in float32 (each
        step's ``q_new`` becomes the next step's ``q_old``, so a
        one-sided bias would accumulate across them) and asserts the
        closure leaves SOAG BIT-IDENTICAL to the uncorrected value and
        the residual EXACTLY 0 at every step.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        term = Mam4JaxMicrophysics(
            condensation_backend="substep", enable_x64=False,
            core_dtype="float32",
        )
        soa_idxs = term._soa_mode_pcnst_idxs

        q = jnp.zeros((8, term._pcnst), dtype=jnp.float32)
        q = q.at[:, term._soag_pcnst_idx].set(jnp.float32(1e-8))
        for idx in soa_idxs:
            q = q.at[:, idx].set(jnp.float32(1e-10))

        rng = np.random.default_rng(0)
        gas_to_aer = jnp.float32(term._organic_gas_to_aer_factor)
        for step in range(5):
            # A budget-neutral change each step, in float32 throughout:
            # move mass from the gas into one mode's SOA, in the
            # gas-to-aerosol-equivalent units the closure itself uses.
            moved = jnp.float32(1e-9 * rng.uniform(0.5, 2.0))
            soa_i = soa_idxs[step % len(soa_idxs)]
            q_new = q.at[:, term._soag_pcnst_idx].add(-moved / gas_to_aer)
            q_new = q_new.at[:, soa_i].add(moved)
            self.assertEqual(q_new.dtype, jnp.float32)

            closed, residual = term._close_organic_budget(q, q_new)
            # Bit-identical, not merely close: any correction at all here
            # -- even a tiny one -- is exactly the one-sided sink this
            # test guards against.
            np.testing.assert_array_equal(closed, q_new)
            np.testing.assert_array_equal(residual, jnp.zeros_like(residual))
            q = q_new

    def test_organic_budget_closure_leaves_a_residual_when_gas_is_exhausted(self):
        """PR #1170 review round 2 (Codex P2): the closure must NOT guess
        a recipient on the aerosol side when the gas has nothing left to
        give -- an earlier version took the remainder from whichever
        mode(s) showed a positive mass increment over the whole
        ``calcsize -> amicphys`` call, which is not specific to SOA
        exchange (rename/coagulation/ageing can also give a mode a
        positive increment) and could tax mass a mode received
        legitimately. The redesign: repay only what the gas can give,
        leave every aerosol slot untouched, and return the leftover as
        a residual for the caller to publish as a diagnostic instead.

        Direct, deterministic construction: gas already at its floor (0)
        in both ``q_old`` and ``q_new`` -- i.e. genuinely nothing left to
        give -- while mode 0's SOA mass increased by an "unrealizable"
        amount with no corresponding gas decrease to balance it.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        term = Mam4JaxMicrophysics(condensation_backend="substep")
        soa_idxs = term._soa_mode_pcnst_idxs

        q_old = jnp.zeros((1, term._pcnst), dtype=jnp.float64)
        q_old = q_old.at[:, term._soag_pcnst_idx].set(0.0)  # gas starts at its floor
        for idx in soa_idxs:
            q_old = q_old.at[:, idx].set(1e-10)

        q_new = q_old.at[:, term._soag_pcnst_idx].set(0.0)  # and ends there too
        excess = 5e-11
        q_new = q_new.at[:, soa_idxs[0]].add(excess)  # mode 0 over-received it

        def organic_total(q):
            g = term._organic_gas_to_aer_factor * q[:, term._soag_pcnst_idx]
            a = sum(q[:, idx] for idx in soa_idxs)
            return g + a

        total_before = organic_total(q_old)
        uncorrected_rel = (organic_total(q_new) - total_before) / total_before
        # Sanity check the scenario is non-vacuous: without the fix this
        # genuinely fails to close (mode 0's gain is real, uncorrected).
        self.assertGreater(float(uncorrected_rel[0]), 1e-3)

        closed, residual = term._close_organic_budget(q_old, q_new)
        self.assertEqual(float(closed[0, term._soag_pcnst_idx]), 0.0)
        # Every aerosol slot is untouched -- bit-identical to the input
        # q_new, including the over-received mode 0 (NOT walked back).
        for idx in soa_idxs:
            self.assertEqual(float(closed[0, idx]), float(q_new[0, idx]))
        # The gas had nothing to give (it started and ended at 0), so the
        # ENTIRE excess is reported as the residual, exactly.
        np.testing.assert_allclose(residual, excess, rtol=1e-12)

    def test_organic_budget_closes_when_strong_condensation_exhausts_the_gas(self):
        """End-to-end repro of the same Codex P2 gap through the real
        adapter and kernel, not just the closure helper in isolation: a
        cell whose small accumulation-mode SOA reservoir evaporates past
        itself (the original jax-gcm#1064 mechanism) while the gas ends
        the step pinned at its own floor (no headroom left for a
        gas-only correction), because its own, much larger, aggregate
        equilibrium target (driven by the hot accumulation mode alone;
        see below) leaves it fully consumed.

        A heavily POA-loaded, high-number Aitken mode is included to
        stress a second, strongly-absorbing partner mode in the same
        cell -- not load-bearing for the specific numbers asserted here
        (confirmed separately: the mechanism that actually pins the gas
        at its floor in this fixture is accumulation's own large
        equilibrium target at 310 K exceeding what the unconstrained
        closed form can supply), but representative of the kind of
        multi-mode cell a real run visits, and kept because it is the
        configuration that was found to reproduce the gap.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
            _ORGANIC_RESIDUAL_KEY,
        )
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY

        state, diagnostics = _column_state(nlev=1, ncols=1)
        tracers = dict(state.tracers)
        tracers["n_ait"] = jnp.full((1, 1), 1e11)
        tracers["m_poa_ait"] = jnp.full((1, 1), 1e-7)
        tracers["m_soa_ait"] = jnp.zeros((1, 1))
        tracers["m_soa_cor"] = jnp.zeros((1, 1))
        tracers["m_soa_acc"] = jnp.full((1, 1), 1e-10)
        tracers["g_soag"] = jnp.full((1, 1), 1e-12)
        state = state.copy(temperature=jnp.full((1, 1), 310.0), tracers=tracers)
        dt = diagnostics["_dt_seconds"] = 1800.0
        carry = {k: q for k, q in tracers.items() if k.startswith(("mc_", "nc_"))}
        diagnostics[CARRY_KEY] = carry

        factor = 12.011 / 12  # the same gas<->aerosol basis the module derives

        def organic_total(tracer_dict):
            g = factor * np.asarray(tracer_dict["g_soag"], np.float64)
            a = sum(np.asarray(v, np.float64) for k, v in tracer_dict.items()
                    if k.startswith("m_soa_"))
            return g + a

        total_before = organic_total(tracers)

        # Sanity check: the raw kernel (closure disabled) really does not
        # close here -- this fixture is not a vacuous pass.
        term_raw = Mam4JaxMicrophysics(condensation_backend="substep")
        term_raw._organic_closure_active = False
        tendency_raw, result_raw = term_raw(state, diagnostics, None, None)
        soa_raw = {k: tracers[k] + dt * tendency_raw.tracers[k]
                   for k in tracers if k.startswith("m_soa_")}
        updated_raw = {**tracers, **soa_raw,
                       "g_soag": tracers["g_soag"] + dt * tendency_raw.tracers["g_soag"]}
        rel_raw = (organic_total(updated_raw) - total_before) / total_before
        self.assertGreater(float(rel_raw[0, 0]), 1.0)  # wildly open, not 2e-6

        term = Mam4JaxMicrophysics(condensation_backend="substep")
        self.assertTrue(term._organic_closure_active)
        tendency, result = term(state, diagnostics, None, None)
        soa_fixed = {k: tracers[k] + dt * tendency.tracers[k]
                     for k in tracers if k.startswith("m_soa_")}
        g_soag_fixed = tracers["g_soag"] + dt * tendency.tracers["g_soag"]
        updated = {**tracers, **soa_fixed, "g_soag": g_soag_fixed}

        # The closure repays ONLY the gas: SOAG ends at exactly its floor,
        # and every modal SOA mass is bit-identical to the raw (uncorrected)
        # kernel output -- the redesign's whole point (Codex round 2).
        self.assertEqual(float(g_soag_fixed[0, 0]), 0.0)
        for name, raw_value in soa_raw.items():
            np.testing.assert_array_equal(soa_fixed[name], raw_value)

        # The organic budget is therefore NOT closed by tendencies alone
        # any more -- the gap is exactly the published residual diagnostic.
        rel_after_gas_only = (organic_total(updated) - total_before) / total_before
        measured_excess = float(rel_after_gas_only[0, 0]) * float(total_before[0, 0])
        self.assertGreater(measured_excess, 0.0)

        dm = float(diagnostics["air_density"][0, 0]
                   * diagnostics["layer_thickness"][0, 0])
        # Column-integrated already (no level axis -- the burden
        # convention every other JAM budget diagnostic uses): shape
        # (ncols,), not (nlev, ncols).
        residual_column = float(result[_ORGANIC_RESIDUAL_KEY][0])
        np.testing.assert_allclose(
            residual_column, measured_excess * dm, rtol=1e-6)
        # The raw (closure-disabled) run never computes this diagnostic's
        # correction path at all, so it reports zero -- the key is still
        # published (fixed key set for the scan carry), just with nothing
        # in it.
        self.assertEqual(float(result_raw[_ORGANIC_RESIDUAL_KEY][0]), 0.0)

    def test_organic_budget_closure_retires_on_fixed_mam4_jax_version(self):
        """Once the installed ``mam4_jax`` reports a version carrying the
        upstream per-substep fix (reflective-org/MAM4-JAX#85), jcm's
        step-level closure must gate itself off rather than double-correct
        an already-fixed kernel. Simulated here with a monkeypatched
        ``__version__`` rather than a real future release, since this is a
        property of the version comparison in ``__init__``, not of what the
        installed kernel actually does.
        """
        import mam4_jax

        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        original_version = mam4_jax.__version__
        try:
            mam4_jax.__version__ = "0.5.2"
            term = Mam4JaxMicrophysics(condensation_backend="substep")
            self.assertFalse(term._organic_closure_active)

            mam4_jax.__version__ = "0.5.1"
            term = Mam4JaxMicrophysics(condensation_backend="substep")
            self.assertTrue(term._organic_closure_active)

            mam4_jax.__version__ = "0.5.2"
            term = Mam4JaxMicrophysics(condensation_backend="astem")
            self.assertFalse(term._organic_closure_active)
        finally:
            mam4_jax.__version__ = original_version

    def test_packing_covers_every_interstitial_tracer(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        term = Mam4JaxMicrophysics()
        packed = {name for name, _ in term._q_pack}
        expected = {"g_h2so4", "g_soag"}
        for mode in MAM4_SPEC.modes:
            expected.add(number_name(mode.short))
            for sp in mode.species:
                expected.add(mass_name(sp, mode.short))
        self.assertEqual(packed, expected)
        # pcnst indices are unique and inside the aerosol/gas band [6, 34].
        idxs = [idx for _, idx in term._q_pack]
        self.assertEqual(len(idxs), len(set(idxs)))
        self.assertTrue(all(6 <= i <= 34 for i in idxs))

    def test_forward_finite_and_physical(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        state, diagnostics = _column_state()
        term = Mam4JaxMicrophysics()
        tend, diags = term(state, diagnostics, None, None)

        key = mass_name(MAM4_SPEC.modes[0].species[0], MAM4_SPEC.modes[0].short)
        self.assertIn(key, tend.tracers)
        for v in tend.tracers.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))))
        # Tendencies are returned at the model dtype, not float64.
        self.assertEqual(tend.tracers[key].dtype, state.temperature.dtype)

        aer = diags["_jam_state"]
        self.assertEqual(aer.r_dry.shape, (MAM4_SPEC.n_modes(), 4, 2))
        for f in (aer.r_dry, aer.r_wet, aer.rho, aer.kappa):
            self.assertTrue(np.all(np.isfinite(np.asarray(f))))
        self.assertTrue(np.all(np.asarray(aer.r_wet) >= np.asarray(aer.r_dry)))
        self.assertTrue(np.all(np.asarray(aer.r_dry) > 0.0))

    def _assert_backend_runs_finite(self, backend):
        from mam4_jax.coupling import amicphys as _amicphys

        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        if not hasattr(_amicphys, "configure_condensation"):
            self.skipTest("mam4_jax lacks configure_condensation (PR #59)")

        # A condensation backend must produce finite, physical output through
        # the full wrapper. Restore the process-global default afterwards so it
        # can't leak into other tests sharing this process.
        state, diagnostics = _column_state()
        try:
            term = Mam4JaxMicrophysics(
                condensation_backend=backend, n_substeps=4,
            )
            self.assertEqual(_amicphys._COND["backend"], backend)
            self.assertEqual(term._condensation_backend, backend)
            tend, diags = term(state, diagnostics, None, None)
            for v in tend.tracers.values():
                self.assertTrue(np.all(np.isfinite(np.asarray(v))))
            aer = diags["_jam_state"]
            for f in (aer.r_dry, aer.r_wet, aer.rho, aer.kappa):
                self.assertTrue(np.all(np.isfinite(np.asarray(f))))
            self.assertTrue(np.all(np.asarray(aer.r_dry) > 0.0))
            key = mass_name(
                MAM4_SPEC.modes[0].species[0], MAM4_SPEC.modes[0].short
            )
            self.assertIn(key, tend.tracers)
        finally:
            _amicphys.configure_condensation(backend="substep")

    def test_substep_backend_runs_finite_and_physical(self):
        self._assert_backend_runs_finite("substep")

    def test_astem_backend_runs_finite_and_physical(self):
        self._assert_backend_runs_finite("astem")

    def test_default_backend_is_substep(self):
        from mam4_jax.coupling import amicphys as _amicphys

        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        if not hasattr(_amicphys, "configure_condensation"):
            self.skipTest("mam4_jax lacks configure_condensation (PR #59)")
        try:
            term = Mam4JaxMicrophysics()
            self.assertEqual(term._condensation_backend, "substep")
        finally:
            _amicphys.configure_condensation(backend="substep")

    def test_enable_x64_control(self):
        from mam4_jax.coupling import amicphys as _amicphys

        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        if not hasattr(_amicphys, "configure_condensation"):
            self.skipTest("mam4_jax lacks configure_condensation (PR #59)")
        # setUp/tearDown restore the process-global x64 flag around this test.
        try:
            # Default (None) keeps float64 (current behaviour).
            self.assertTrue(Mam4JaxMicrophysics()._enable_x64)
            self.assertTrue(jax.config.read("jax_enable_x64"))
            # Opt into float32 with a float32-safe backend.
            term = Mam4JaxMicrophysics(
                condensation_backend="substep", enable_x64=False,
            )
            self.assertFalse(term._enable_x64)
            self.assertFalse(jax.config.read("jax_enable_x64"))
            # Backend validation is delegated to mam4_jax: an unknown backend
            # raises the library's own ValueError (jcm adds no guard of its own).
            with self.assertRaises(ValueError):
                Mam4JaxMicrophysics(condensation_backend="not-a-backend")
        finally:
            _amicphys.configure_condensation(backend="substep")

    def test_carbon_aging_moves_bc_from_pcm_to_accum(self):
        """Ageing (jax-gcm#721): condensed H2SO4 coats the pcm mode and the
        core transfers the coated fraction of BC (and pcm number) to accum.
        Attribution is by monolayer-threshold sensitivity: an absurdly thick
        required coating (1e9 monolayers) makes ageing inert, leaving only
        the pcm→acc coagulation pathway, so the default-vs-inert difference
        isolates the ageing transfer.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        state, diagnostics = _column_state()
        # A healthy H2SO4 reservoir so within-step condensation builds a
        # real shell on pcm.
        state = state.copy(tracers={**state.tracers,
                                    "g_h2so4": jnp.full((4, 2), 1.0e-9)})
        # No try/finally needed: the constructor holds the threshold as
        # an nnx.Param and passes it per call — it never mutates the
        # core's process-global config, so instances cannot interfere.
        aged, _ = Mam4JaxMicrophysics()(state, diagnostics, None, None)
        inert_term = Mam4JaxMicrophysics(n_so4_monolayers=1.0e9)
        inert, _ = inert_term(state, diagnostics, None, None)

        d_bc_pcm = np.asarray(aged.tracers["m_bc_pcm"]
                              - inert.tracers["m_bc_pcm"], np.float64)
        d_bc_acc = np.asarray(aged.tracers["m_bc_acc"]
                              - inert.tracers["m_bc_acc"], np.float64)
        d_n_pcm = np.asarray(aged.tracers["n_pcm"]
                             - inert.tracers["n_pcm"], np.float64)
        self.assertTrue(np.all(d_bc_pcm < 0.0),
                        "ageing must remove BC from the pcm mode")
        self.assertTrue(np.all(d_bc_acc > 0.0),
                        "ageing must deliver BC to the accum mode")
        self.assertTrue(np.all(d_n_pcm < 0.0),
                        "ageing must move pcm number out")
        # BC has no other source/sink in the core: the ageing difference
        # must conserve BC between the two modes.
        np.testing.assert_allclose(
            d_bc_pcm + d_bc_acc, np.zeros_like(d_bc_pcm),
            atol=1e-6 * float(np.abs(d_bc_pcm).max()),
            err_msg="ageing must conserve BC across pcm+acc")

    def _assert_no_source_species_conserved(
        self, core_dtype, backend, *, atol=None, burden_rel_k=None,
    ):
        """Shared body for the f64/f32 conservation tests below.

        ``atol`` is a fixed absolute tolerance (the f64 round-off case).
        ``burden_rel_k`` instead builds a PER-CELL tolerance scaled to
        that cell's own packed burden, ``k * eps(float32) *
        burden_cell`` -- see ``test_no_source_species_conserve_mass_
        bounded_f32`` for why a fixed absolute tolerance is the wrong
        shape for this check.
        """
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        # Build the host state under float64 containers regardless of
        # ``core_dtype``: as in ``test_core_dtype_scoped_float32``, the
        # float32 CORE still round-trips tendencies through the host
        # dtype, so a float64 host isolates the core's OWN precision
        # (scoped ``jax.enable_x64(False)`` inside ``__call__``) rather
        # than conflating it with a float32 *host* container.
        jax.config.update("jax_enable_x64", True)
        state, diagnostics = _column_state_with_negatives()
        carry_before = {
            k: q for k, q in state.tracers.items() if k.startswith("mc_")}
        diagnostics[CARRY_KEY] = carry_before
        dt = diagnostics["_dt_seconds"]

        term = Mam4JaxMicrophysics(
            core_dtype=core_dtype, condensation_backend=backend,
            enable_x64=True)
        tendency, result = term(state, diagnostics, None, None)
        # Cloud-borne names stay in ``carry_before``/``result[CARRY_KEY]``
        # (the carry-mode path: ``tendency.tracers`` carries only the
        # interstitial tendencies the stepper would apply to dycore
        # tracers, see ``apply_updates``'s empty-passthrough contract).
        tracers_after = {
            name: state.tracers[name] + dt * tendency.tracers[name]
            for name in tendency.tracers
        }
        carry_after = result[CARRY_KEY]

        for sp in _NO_SOURCE_SPECIES:
            before = np.asarray(_species_column_mass(
                MAM4_SPEC, state.tracers, carry_before, sp))
            after = np.asarray(_species_column_mass(
                MAM4_SPEC, tracers_after, carry_after, sp))
            resid = np.abs(after - before)
            if burden_rel_k is not None:
                # A PER-CELL array, not a scalar: ``np.testing.
                # assert_allclose`` builds its (unused-on-pass, but
                # unconditionally evaluated) header message with
                # ``f"atol={atol:g}"``, which raises ``TypeError`` for a
                # non-scalar ``atol`` on numpy >= 2 regardless of
                # whether the comparison itself would pass -- so this
                # check is done by hand instead of through that helper.
                burden = _species_packed_burden(
                    MAM4_SPEC, state.tracers, carry_before, sp)
                cell_atol = burden_rel_k * _EPS32 * burden
            else:
                cell_atol = np.full_like(resid, atol)
            # ``resid > cell_atol`` is False for a NaN residual (NaN
            # compares False to everything), so a non-finite output
            # would otherwise pass silently -- reject it explicitly.
            bad = ~np.isfinite(resid) | (resid > cell_atol)
            self.assertFalse(
                bool(np.any(bad)),
                f"species {sp!r} is not conserved by the core (issue "
                f"#778): max |resid|={resid.max():.3e} vs atol max="
                f"{cell_atol.max():.3e} at {int(np.sum(bad))} cell(s)")

    def test_no_source_species_conserve_mass_to_roundoff_f64(self):
        """#778: per the module docstring's conservation invariant, only
        sulfate/SOA have a source or sink in the core, so dust/sea-salt/
        BC/POA/MOA's column total must be EXACTLY invariant (round-off
        only) across one adapter call -- including with negative packed
        mass AND number inputs (mechanism (a): the ``fetch()`` floor)
        and active pcarbon ageing (mechanism (c): the pcm->accum
        repack). Re-measured on the current mam4-jax 0.5.1 pin (the
        issue's +0.16 %/yr dust drift was measured on 0.4.0): the leak
        does not reproduce here, for either condensation backend.
        """
        for backend in ("substep", "astem"):
            self._assert_no_source_species_conserved(
                "float64", backend, atol=1e-18)

    def test_no_source_species_conserve_mass_bounded_f32(self):
        """Same as the f64 test, bounded to float32 round-off (the core
        dtype is the ~1M-cell amicphys vmap's production setting).

        The tolerance is PER CELL and scaled to that cell's own packed
        burden (``_F32_BURDEN_REL_K * eps(float32) * burden_cell``), not
        a single fixed absolute number: with this fixture's per-cell
        masses (~1e-10 to ~1e-12), a fixed ``atol=1e-15`` is ~1e-5 to
        ~1e-3 RELATIVE to a cell's own mass -- far too loose to mean
        anything (Codex #1117 thread 4274279050).

        Measured on this fixture (both backends): the actual per-cell
        round-off never exceeds ~0.80 * eps(float32) relative to its
        cell's packed burden, so ``_F32_BURDEN_REL_K = 2`` gives > 2x
        margin above that measured maximum (chosen over a tighter k
        specifically to keep this margin: the alternative, named below,
        is worse).

        This tolerance is NOT "10x tighter than the historical drift".
        It can't be, and that is itself the finding worth recording: the
        issue's own historical defect rate (+0.16 %/yr dust, ~9e-8
        relative PER CALL at this fixture's implied timestep) sits at
        ~0.84 * eps(float32) in the same units as the measured round-off
        floor above (~0.80 * eps(float32)) -- a ~5 % gap. Verified
        locally (not committed): at ``_F32_BURDEN_REL_K`` anywhere from
        0.80 up (including 2, the value used here), injecting an exact
        1e-7 RELATIVE loss into one cell is NOT caught, because 1e-7 is
        itself only ~0.84 eps(float32) relative to that cell's burden --
        essentially the same size as real round-off. Only dropping
        ``_F32_BURDEN_REL_K`` to ~0.79 catches the injection, but that
        also starts failing on real (non-injected) round-off -- i.e.
        there is NO k that is simultaneously robust against false
        positives and sensitive to a defect at the historical scale, in
        float32, on this exact pathway: the historical defect size and
        the float32 noise floor are too close together here. The f64
        test above (atol=1e-18, ~1e8x smaller than any physical mass in
        this fixture) is what actually catches a regression at the
        issue's historical scale; this f32 test's narrower job is to
        catch a GROSSLY larger float32-core defect (multiple ULPs, e.g.
        an index-mapping or sign error) without false-positiving on the
        core's own legitimate round-off.
        """
        for backend in ("substep", "astem"):
            self._assert_no_source_species_conserved(
                "float32", backend, burden_rel_k=_F32_BURDEN_REL_K)

    def test_aging_threshold_is_a_differentiable_param_leaf(self):
        """The threshold must be an nnx.Param LEAF (visible to jax.grad /
        optimizers per the repo's differentiable-parameters rule), and
        the gradient of a tendency through the term w.r.t. it must be
        finite and non-zero.
        """
        from flax import nnx

        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        term = Mam4JaxMicrophysics()
        leaves = nnx.state(term, nnx.Param)
        flat = jax.tree.leaves(leaves)
        assert any(
            np.asarray(v).shape == () and float(np.asarray(v)) == 3.0
            for v in flat
        ), "n_so4_monolayers must appear as a Param leaf (default 3.0)"

        # Moderate H2SO4 so the pcm shell stays SUB-saturated at n=3:
        # in the saturated regime the criterion clamps and d/dn is
        # (correctly) exactly zero, which would make this test vacuous.
        state, diagnostics = _column_state()
        state = state.copy(tracers={**state.tracers,
                                    "g_h2so4": jnp.full((4, 2), 1.0e-11)})

        def bc_acc_tendency(n):
            t = Mam4JaxMicrophysics(n_so4_monolayers=1.0)
            t.n_so4_monolayers.set_value(n)
            tend, _ = t(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_bc_acc"])

        g = jax.grad(bc_acc_tendency)(jnp.asarray(3.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertLess(float(g), 0.0,
                        "thicker required coating must age less BC")

    def test_negative_threshold_rejected_at_construction(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        with self.assertRaises(ValueError):
            Mam4JaxMicrophysics(n_so4_monolayers=-1.0)

    def test_core_dtype_scoped_float32(self):
        # The float32 core runs under a SCOPED x64-off context: the global
        # flag (float64 host, e.g. pySES dynamics) must stay untouched, and
        # the float32-core forward tendencies must closely track the float64
        # core (forward is float32-safe per MAM4-JAX #60; the DEFAULT stays
        # float64 because the float32 REVERSE pass is non-finite — see the
        # constructor docstring).
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY

        jax.config.update("jax_enable_x64", True)
        t64 = Mam4JaxMicrophysics()          # default: float64 core
        self.assertFalse(t64._core_f32)
        state, diagnostics = _column_state()
        self.assertEqual(state.temperature.dtype, jnp.dtype("float64"))
        diagnostics[CARRY_KEY] = {
            k: q for k, q in state.tracers.items()
            if k.startswith(("mc_", "nc_"))
        }
        tend64, _ = t64(state, diagnostics, None, None)

        t32 = Mam4JaxMicrophysics(core_dtype="float32")
        self.assertTrue(t32._core_f32)
        tend32, result32 = t32(state, diagnostics, None, None)
        self.assertTrue(jax.config.read("jax_enable_x64"),
                        "scoped f32 core must not clear the global x64 flag")

        # Inspect actual JAX dtypes before NumPy conversion can hide a
        # float32 result from a host cast inside the x64-off scope (#1033).
        for field in ("u_wind", "v_wind", "temperature", "specific_humidity"):
            self.assertEqual(getattr(tend32, field).dtype,
                             getattr(state, field).dtype, field)
        for name, value in tend32.tracers.items():
            self.assertEqual(value.dtype, state.temperature.dtype, name)
        for field in ("r_dry", "r_wet", "rho", "kappa", "mass", "number"):
            self.assertEqual(getattr(result32["_jam_state"], field).dtype,
                             state.temperature.dtype, field)
        self.assertEqual(set(result32[CARRY_KEY]), set(diagnostics[CARRY_KEY]))
        for name, value in result32[CARRY_KEY].items():
            self.assertEqual(value.dtype, state.temperature.dtype, name)

        for k in tend64.tracers:
            a = np.asarray(tend32.tracers[k], np.float64)
            b = np.asarray(tend64.tracers[k], np.float64)
            scale = max(float(np.abs(b).max()), 1e-30)
            # The 1e-16 floor absorbs tendencies that are numerically zero in
            # float64 (O(1e-17) round-off) and flush to exact zero in float32.
            np.testing.assert_allclose(
                a, b, atol=1e-3 * scale + 1e-16, err_msg=f"tracer {k}",
            )

        with self.assertRaises(ValueError):
            Mam4JaxMicrophysics(core_dtype="bf16")

    def test_grad_through_a_tracer_is_finite(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        state, diagnostics = _column_state()
        term = Mam4JaxMicrophysics()
        key = mass_name(MAM4_SPEC.modes[0].species[0], MAM4_SPEC.modes[0].short)
        base = state.tracers[key]

        def loss(scale):
            tr = {**state.tracers, key: base * scale}
            s = state.copy(tracers=tr)
            tend, _ = term(s, diagnostics, None, None)
            return sum(jnp.sum(v.astype(jnp.float64) ** 2)
                       for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0, jnp.float64))
        self.assertTrue(np.isfinite(float(g)))


@pytest.mark.slow
class Mam4JaxModelTest(unittest.TestCase):
    """End-to-end: the MAM4-JAX core runs inside the full ECHAM GCM.

    Guards the per-cell ``jax.vmap`` of the box-model core: the upstream MAM4
    box model runs a single cell, so each (level, column) point is integrated
    independently. A regression that batches the whole grid through the core at
    once would surface here as a blow-up rather than a silent slowdown.

    T21 is the lowest spectral truncation ``get_coords`` accepts, so the
    horizontal grid is already minimal. The vmap this test guards
    (``mam4_jax.py``'s ``jax.vmap(one_step, ...)`` over the flattened
    (level, column) axis) runs identically on every physics call, so it does
    not need repeated steps to expose a batching regression -- a single
    30-minute step (one GCM timestep) already packs the whole grid through
    it once. 3 sigma levels (vs. the original 20) is the fewest that still
    gives the per-cell vmap more than one level to mix across if it were
    wrongly batched; the compile cost of the full ECHAM term graph, not grid
    size, dominates this test's runtime, so trimming levels/steps further
    would not meaningfully shrink it.
    """

    def setUp(self):
        self._x64 = jax.config.read("jax_enable_x64")

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._x64)

    def test_t21_runs_finite_with_mam4_core(self):
        from jcm.model import Model
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, 4), spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        model = Model(
            coords=coords, time_step=30, terrain=terrain,
            physics=idealized_echam_physics(
                aerosol_module="jam", cloud_scheme="2m",
                jam_microphysics="mam4_jax",
            ),
        )
        one_step_days = 30 / 1440  # a single 30-minute (time_step) GCM step
        pred = model.run(save_interval=one_step_days, total_time=one_step_days)
        dyn = pred.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        self.assertTrue(bool(jnp.all(dyn.temperature > 150.0)))
        self.assertTrue(bool(jnp.all(dyn.temperature < 360.0)))


if __name__ == "__main__":
    unittest.main()


@pytest.mark.requires_extra("mam4")
class DefaultCoreIsReverseDifferentiableTest(unittest.TestCase):
    """The ``echam-jam`` physics group's default core supports ``jax.grad``.

    #1052 made the adaptive ASTEM solve the default; its ``lax.while_loop``
    with data-dependent bounds raises under reverse mode, so every
    ``jax.grad`` through an ECHAM+JAM model raised. The default is the
    fixed-substep core; ASTEM stays an explicit, forward-only choice.
    """

    def setUp(self):
        self._x64 = jax.config.read("jax_enable_x64")

    def tearDown(self):
        jax.config.update("jax_enable_x64", self._x64)

    def test_echam_jam_config_defaults_to_the_substep_core(self):
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(Path(__file__).resolve().parents[4].joinpath(
            "config/physics/echam-jam.yaml").read_text())
        self.assertEqual(cfg["jam_microphysics"], "mam4_jax")

    def test_default_core_takes_a_reverse_mode_gradient(self):
        from mam4_jax.coupling import amicphys as _amicphys

        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )

        state, diagnostics = _column_state()
        key = mass_name(MAM4_SPEC.modes[1].species[0], MAM4_SPEC.modes[1].short)
        try:
            term = Mam4JaxMicrophysics(n_substeps=4)   # the default backend

            def loss(x):
                tend, _ = term(state.copy(tracers={**state.tracers, key: x}),
                               diagnostics, None, None)
                return jnp.sum(tend.tracers[key])

            g = jax.grad(loss)(state.tracers[key])   # raises on the ASTEM backend
            self.assertEqual(np.shape(g), np.shape(state.tracers[key]))
        finally:
            _amicphys.configure_condensation(backend="substep")


class RelativeHumidityGradientTest(unittest.TestCase):
    """``relative_humidity``'s float32 reverse-mode gradient is finite (#971),
    and its *value* is bit-identical to the original ``q/max(qs,1e-30))``
    reference form everywhere, including the ultra-cold short-circuit
    (PR #1113 review: an earlier fix that raised the ``qs`` floor to 1e-12
    silently changed RH from 1 to 0.1 for a genuinely tiny but positive
    ``q`` at T=100 K, and an unconditional "saturated" short-circuit would
    separately have turned a bone-dry cell's correct RH=0 into RH=1).

    ``q_sat`` decays continuously toward 0 as T falls and is already below
    float32's ~1.08e-19 derivative-safe bound at T=100 K, 900 hPa -- an
    ordinary polar-night/model-top state, not an edge case. The 1e-30 floor
    did nothing to guard against that (it is itself far below the bound).
    """

    def _reference(self, t, q, p):
        """Evaluate the original (pre-#971) ``q / max(qs, 1e-30)`` form at
        float64, so the reference itself has no float32 range issue.
        """
        from jcm.physics.convection.saturation import saturation_specific_humidity
        with jax.enable_x64(True):
            qs = saturation_specific_humidity(
                jnp.asarray(t, jnp.float64), jnp.asarray(p, jnp.float64))
            ref = jnp.clip(jnp.asarray(q, jnp.float64)
                           / jnp.maximum(qs, 1.0e-30), 0.0, 1.0)
        return float(ref)

    def test_value_matches_the_original_reference_everywhere(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            relative_humidity,
        )

        for t in (320.0, 290.0, 200.0, 150.0, 120.0, 100.0, 70.0, 50.0, 30.0, 10.0):
            for q in (0.0, 1.0e-20, 1.0e-17, 1.0e-16, 1.0e-13, 1.0e-9, 1.0e-6, 1.0e-3):
                with self.subTest(t=t, q=q):
                    rh = float(relative_humidity(
                        jnp.asarray(t, jnp.float32), jnp.asarray(q, jnp.float32),
                        jnp.asarray(9.0e4, jnp.float32)))
                    self.assertAlmostEqual(rh, self._reference(t, q, 9.0e4),
                                           delta=1e-5, msg=(t, q))

    def test_genuinely_unsaturated_ultracold_cell_is_not_forced_to_saturated(self):
        """A SECOND review round caught a second forward-value mistake: a
        fix that short-circuited to RH=1 whenever q_sat fell below a
        threshold (regardless of q) forced this genuinely 10%-saturated
        cell to read as fully saturated. T=120 K, p=900 hPa gives
        q_sat ~= 4.72e-16; q = 4.72e-17 is 10% of that, not 100%.
        """
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            relative_humidity,
        )

        rh = float(relative_humidity(
            jnp.asarray(120.0, jnp.float32), jnp.asarray(4.72e-17, jnp.float32),
            jnp.asarray(9.0e4, jnp.float32)))
        self.assertAlmostEqual(rh, 0.1, delta=1e-3)

    def test_gradient_finite_at_ordinary_cold_temperature(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            relative_humidity,
        )

        def loss(t, q, p):
            return jnp.sum(relative_humidity(t, q, p))

        for t in (320.0, 290.0, 200.0, 100.0, 70.0, 50.0, 30.0, 10.0):
            for q in (0.0, 1.0e-13, 1.0e-6):
                with self.subTest(t=t, q=q):
                    args = (jnp.asarray(t, jnp.float32), jnp.asarray(q, jnp.float32),
                            jnp.asarray(9.0e4, jnp.float32))
                    value = loss(*args)
                    self.assertTrue(np.isfinite(float(value)))
                    g = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*args)
                    for leaf in g:
                        self.assertTrue(np.isfinite(float(leaf)), (t, q))
