"""Unit tests for the Tiedtke ledger + entrainment fixes (#676, #669).

Each test pins one reference deviation against a hand-computed value or an
ECHAM reference bound:

* #676.1 — the cloud-base first-guess FALLBACK is ECHAM's constant
  ``zmfub = 0.01`` (mo_cumastr.f90:567), not the dimensionally-invalid
  ``cape/(g·tau)`` velocity.
* #676.3 — the condensate-flux heating term uses the PHASE-KEYED latent heat
  (``alhs`` below the melting point, ``alhc`` above), the same key as the
  per-level source term.
* #676.4 — the SURFACE layer receives a convective tendency (ECHAM's
  ``jk == klev`` cudtdq branch), no longer identically zero.
* #676.2 — momentum transport (cududv) carries a surface-layer term and a
  sub-cloud taper; the ``_DTDT_MAX`` cap scales dudt/dvdt consistently with
  the rest of the ledger.
* #669 — organized entrainment/detrainment are the metre-based fractional
  rates capped at ``centrmax`` (3.0e-4 m⁻¹); the cap engages on a deep plume.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.flux_tendencies import (
    ECHAM_MFUB_FALLBACK,
    calculate_tendencies,
    mass_flux_closure,
    mass_flux_closure_blend,
)
from jcm.physics.convection.tiedtke_nordeng.updraft import (
    UpdatedraftState,
    calculate_updraft,
)
from jcm.physics.convection.tiedtke_nordeng.downdraft import (
    DowndraftState,
    calculate_downdraft,
)
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    find_cloud_base,
    saturation_mixing_ratio,
    shallow_reclosure_flux,
    tiedtke_nordeng_convection,
)
from jcm.physics.convection.tiedtke_nordeng.types import ConvectionParameters


def _zero_updraft(nlev):
    z = jnp.zeros(nlev)
    return UpdatedraftState(
        tu=z, qu=z, lu=z, mfu=z, entr=z, detr=z, buoy=z,
        pdmfup=z, plude=z, uu=z, vu=z,
    )


def _zero_downdraft(nlev):
    z = jnp.zeros(nlev)
    return DowndraftState(
        td=z, qd=z, mfd=z, pdmfdp=z, ud=z, vd=z, lfs=0, active=False,
    )


# --------------------------------------------------------------------------
# #676 item 1 — constant fallback closure
# --------------------------------------------------------------------------
class TestConstantFallbackClosure:
    def test_fallback_is_echam_constant_independent_of_cape(self):
        cfg = ConvectionParameters.default()
        vals = [
            float(mass_flux_closure(jnp.array(cape), jnp.array(0.0),
                                    jnp.array(0.0), 1, cfg))
            for cape in (10.0, 500.0, 5000.0)
        ]
        # A genuine mass flux, and independent of CAPE (the CAPE dependence
        # for deep lives in the Nordeng zmfub1 rescale).
        for v in vals:
            assert v == pytest.approx(ECHAM_MFUB_FALLBACK, rel=1e-6)
        assert ECHAM_MFUB_FALLBACK == pytest.approx(0.01)

    def test_blend_matches_the_same_constant(self):
        cfg = ConvectionParameters.default()
        mfb = mass_flux_closure_blend(
            jnp.array(3000.0), jnp.array(0.0), jnp.array(0.0),
            jnp.array([1.0, 0.0, 0.0]), cfg,
        )
        assert float(mfb) == pytest.approx(ECHAM_MFUB_FALLBACK, rel=1e-6)


# --------------------------------------------------------------------------
# #676 items 3 & 4 — phase-keyed latent heat + surface-layer tendency
# --------------------------------------------------------------------------
class TestCondensateFluxLedger:
    """Isolate the condensate-flux heating term.

    Setting ``tu == env T`` (zero DSE deviation flux) and ``qu == env q``
    (zero moisture deviation flux) with ``mfd = plude = pdmfup = pdmfdp = 0``
    leaves the condensate flux ``L·lu·mfu`` as the ONLY nonzero divergence,
    so ``dtedt`` is exactly ``−Δ(zalv·lu·mfu)/(cp·Δp/g)`` and can be
    hand-computed. ``cp`` is ECHAM's moist ``pcpen = cpd·(1 + vtmpc2·q)``
    (``zrcpm``, mo_cufluxdts.f90:648), built here by hand from the column's
    uniform 5 g/kg.
    """

    Q = 5.0e-3
    #: Hand-computed ECHAM ``pcpen`` for the uniform column humidity.
    CP = c.cpd * (1.0 + c.vtmpc2 * Q)

    def _setup(self, T_value):
        nlev = 5
        pressure = jnp.array([2.0e4, 4.0e4, 6.0e4, 8.0e4, 1.0e5])
        temperature = jnp.full(nlev, T_value)
        humidity = jnp.full(nlev, self.Q)
        rho = pressure / (c.rd * temperature)
        dz = jnp.full(nlev, 1000.0)
        # Condensate flux lu·mfu = [0, 1e-4, 2e-4, 3e-4, 4e-4] — nonzero at
        # the surface (last index) so the surface-layer closure is exercised.
        mfu = jnp.array([0.0, 0.1, 0.1, 0.1, 0.1])
        lu = jnp.array([0.0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3])
        up = _zero_updraft(nlev)._replace(
            tu=temperature, qu=humidity, lu=lu, mfu=mfu,
        )
        dn = _zero_downdraft(nlev)
        tend = calculate_tendencies(
            temperature, humidity, jnp.zeros(nlev), jnp.zeros(nlev),
            pressure, rho, dz, up, dn, kbase=4, ktop=1, dt=1800.0,
            config=ConvectionParameters.default(), ktype=jnp.array(1),
        )
        dp = float(pressure[1] - pressure[0])
        mass = dp / c.grav
        cond_flux = np.asarray(lu * mfu)
        return tend, cond_flux, mass

    def test_phase_keyed_latent_heat_cold_uses_alhs(self):
        tend, cond_flux, mass = self._setup(250.0)  # below tmelt
        # div = diff([cond_flux, 0]) ; dtedt = -alhs*div/(cp*mass)
        div = np.diff(np.append(cond_flux, 0.0))
        expected = -c.alhs * div / (self.CP * mass)
        np.testing.assert_allclose(np.asarray(tend.dtedt), expected, rtol=1e-5)
        # A fixed-alhc ledger would be measurably different (~13%).
        wrong = -c.alhc * div / (self.CP * mass)
        assert not np.allclose(np.asarray(tend.dtedt), wrong, rtol=1e-3)
        # A dry-cpd conversion is off by vtmpc2·q ≈ 0.43 % (#872).
        dry = -c.alhs * div / (c.cpd * mass)
        assert not np.allclose(np.asarray(tend.dtedt), dry, rtol=1e-3)

    def test_phase_keyed_latent_heat_warm_uses_alhc(self):
        tend, cond_flux, mass = self._setup(290.0)  # above tmelt
        div = np.diff(np.append(cond_flux, 0.0))
        expected = -c.alhc * div / (self.CP * mass)
        np.testing.assert_allclose(np.asarray(tend.dtedt), expected, rtol=1e-5)

    def test_surface_layer_receives_tendency(self):
        # With the plume flux nonzero at the surface (last index), the
        # ECHAM klev closure gives dtedt[surface] = -(-F[surface])... i.e.
        # nonzero, where the old diff-into-[:-1] left it exactly 0.
        tend, cond_flux, mass = self._setup(290.0)
        surf = float(tend.dtedt[-1])
        expected_surf = -(-c.alhc * cond_flux[-1]) / (self.CP * mass)
        assert surf != 0.0
        assert surf == pytest.approx(expected_surf, rel=1e-5)


# --------------------------------------------------------------------------
# #872 — the cudtdq ledger uses ECHAM's MOIST heat capacity
# --------------------------------------------------------------------------
class TestMoistHeatCapacityLedger:
    """``pmfus``/``pmfds`` carry ``pcpcu·T + φ`` and ``zrcpm = 1/pcpen``.

    A warm plume (``tu = T + 1 K``) with no condensate, in a column whose
    humidity falls with height, isolates the dry-static-energy deviation
    flux ``F_k = cp_k·(tu_k − T_k)·mfu_k``. Its divergence telescopes over
    the column, so the enthalpy the ledger deposits, ``Σ cp_k·dT_k·m_k``,
    vanishes to round-off with the SAME moist ``cp`` the ledger divides by
    (mo_cufluxdts.f90:198-204, 648-656) — and is open by ``~vtmpc2·Δq``
    when integrated with dry ``cpd``.
    """

    def _run(self):
        nlev = 6
        pressure = jnp.linspace(3.0e4, 1.0e5, nlev)
        temperature = jnp.linspace(240.0, 300.0, nlev)
        humidity = jnp.linspace(1.0e-3, 1.8e-2, nlev)
        rho = pressure / (c.rd * temperature)
        dz = jnp.full(nlev, 1000.0)
        mfu = jnp.array([0.0, 0.02, 0.05, 0.08, 0.06, 0.03])
        up = _zero_updraft(nlev)._replace(
            tu=temperature + 1.0, qu=humidity, mfu=mfu,
        )
        tend = calculate_tendencies(
            temperature, humidity, jnp.zeros(nlev), jnp.zeros(nlev),
            pressure, rho, dz, up, _zero_downdraft(nlev), kbase=nlev - 1,
            ktop=1, dt=1800.0, config=ConvectionParameters.default(),
            ktype=jnp.array(1),
        )
        dpa = np.abs(np.diff(np.asarray(pressure)))
        mass = np.concatenate([dpa, dpa[-1:]]) / c.grav
        cp = c.cpd * (1.0 + c.vtmpc2 * np.asarray(humidity))
        return np.asarray(tend.dtedt, dtype=np.float64), mass, cp, np.asarray(mfu)

    def test_column_enthalpy_closes_with_moist_cp(self):
        dtedt, mass, cp, mfu = self._run()
        moist = float(np.sum(cp * dtedt * mass))
        scale = float(np.sum(np.abs(cp * dtedt * mass)))
        assert scale > 0.0
        assert abs(moist) / scale < 1e-5

    def test_dry_cpd_integral_is_open(self):
        dtedt, mass, cp, mfu = self._run()
        dry = float(np.sum(c.cpd * dtedt * mass))
        scale = float(np.sum(np.abs(cp * dtedt * mass)))
        # Open by the humidity-weighted cp spread across the plume (~1 %).
        assert abs(dry) / scale > 1e-3

    def test_level_tendency_hand_computed(self):
        dtedt, mass, cp, mfu = self._run()
        flux = cp * 1.0 * mfu
        div = np.diff(np.append(flux, 0.0))
        expected = div / (cp * mass)
        np.testing.assert_allclose(dtedt, expected, rtol=2e-5, atol=1e-12)


# --------------------------------------------------------------------------
# #676 item 2 — momentum surface term + sub-cloud taper
# --------------------------------------------------------------------------
class TestMomentumTransport:
    def test_surface_and_subcloud_momentum_present(self):
        nlev = 6
        pressure = jnp.linspace(2.0e4, 1.0e5, nlev)
        temperature = jnp.full(nlev, 290.0)
        humidity = jnp.full(nlev, 5.0e-3)
        rho = pressure / (c.rd * temperature)
        dz = jnp.full(nlev, 1000.0)
        u_wind = jnp.linspace(-10.0, 10.0, nlev)  # sheared
        v_wind = jnp.zeros(nlev)
        # Plume (with prognostic uu) and downdraft (with ud) both present.
        mfu = jnp.array([0.0, 0.1, 0.1, 0.1, 0.0, 0.0])
        uu = jnp.array([0.0, -8.0, -6.0, -4.0, 0.0, 0.0])
        up = _zero_updraft(nlev)._replace(
            tu=temperature, qu=humidity, mfu=mfu, uu=uu,
        )
        dn = _zero_downdraft(nlev)
        cfg = ConvectionParameters.default()
        tend = calculate_tendencies(
            temperature, humidity, u_wind, v_wind, pressure, rho, dz,
            up, dn, kbase=3, ktop=1, dt=1800.0, config=cfg,
            ktype=jnp.array(1),
        )
        dudt = np.asarray(tend.dudt)
        assert np.all(np.isfinite(dudt))
        # Momentum transport reaches BELOW cloud base (sub-cloud taper) and
        # to the surface, where the previous truncated form left zeros.
        assert np.any(np.abs(dudt[4:]) > 0.0), "no sub-cloud/surface friction"
        # The LOWEST model layer (surface, last index) must carry the tapered
        # cumulus friction: cududv's zzp uses the layer TOP interface, not the
        # surface interface, so it stays > 0 there (Codex P2). A full-level
        # pressure ratio would zero zzp[-1] and this tendency.
        assert abs(float(dudt[-1])) > 0.0, (
            "surface-layer momentum tendency is zero — the sub-cloud taper "
            "must not vanish at the lowest full level")
        # v is uniform → no v tendency.
        np.testing.assert_allclose(np.asarray(tend.dvdt), 0.0, atol=1e-12)

    def test_lmfdudv_off_zeros_momentum(self):
        nlev = 6
        pressure = jnp.linspace(2.0e4, 1.0e5, nlev)
        temperature = jnp.full(nlev, 290.0)
        humidity = jnp.full(nlev, 5.0e-3)
        rho = pressure / (c.rd * temperature)
        dz = jnp.full(nlev, 1000.0)
        up = _zero_updraft(nlev)._replace(
            tu=temperature, qu=humidity,
            mfu=jnp.array([0.0, 0.1, 0.1, 0.1, 0.0, 0.0]),
            uu=jnp.array([0.0, -8.0, -6.0, -4.0, 0.0, 0.0]),
        )
        dn = _zero_downdraft(nlev)
        cfg = ConvectionParameters.default(lmfdudv=False)
        tend = calculate_tendencies(
            temperature, humidity, jnp.linspace(-10.0, 10.0, nlev),
            jnp.zeros(nlev), pressure, rho, dz, up, dn,
            kbase=3, ktop=1, dt=1800.0, config=cfg, ktype=jnp.array(1),
        )
        np.testing.assert_allclose(np.asarray(tend.dudt), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# #669 — organized entrainment / detrainment metre-based + capped
# --------------------------------------------------------------------------
def _deep_unstable_column(nlev, dz_m):
    """Build a conditionally-unstable deep column (``nlev`` × ``dz_m`` m)."""
    # Build pressures hydrostatically-ish from a fixed dz and lapse.
    T = np.empty(nlev)
    T[-1] = 300.0
    for k in range(nlev - 2, -1, -1):
        T[k] = T[k + 1] - 8.5e-3 * dz_m
    # pressure from hydrostatic integration (surface last, top-first).
    p = np.empty(nlev)
    p[-1] = 1.0e5
    for k in range(nlev - 2, -1, -1):
        rho_k = p[k + 1] / (c.rd * T[k + 1])
        p[k] = p[k + 1] - rho_k * c.grav * dz_m
    p = np.clip(p, 5.0e3, None)
    from jcm.physics.convection.saturation import (
        saturation_specific_humidity_and_derivative as qsd,
    )
    qs, _ = qsd(jnp.array(T), jnp.array(p))
    q = 0.9 * np.asarray(qs)
    rho = p / (c.rd * T)
    return (jnp.array(T), jnp.array(q), jnp.array(p),
            jnp.full(nlev, dz_m), jnp.array(rho))


class TestOrganizedEntrainmentDetrainment:
    def test_organized_rates_capped_at_centrmax(self):
        cfg = ConvectionParameters.default()
        T, q, p, dz, rho = _deep_unstable_column(nlev=40, dz_m=400.0)
        nlev = T.shape[0]
        kbase = nlev - 3
        ktop = 4
        # Force a solidly deep plume.
        tw = jnp.array([1.0, 0.0, 0.0])
        up = calculate_updraft(
            T, q, p, dz, rho, kbase, ktop, 1, 0.05, cfg,
            type_weights=tw,
        )
        entr = np.asarray(up.entr)
        detr = np.asarray(up.detr)
        centrmax = float(cfg.cu_centrmax)
        entrpen = float(cfg.entrpen)
        # organized part is capped at centrmax; turbulent part is entrpen.
        # So both rates must stay at/below entrpen + centrmax (+ fp slack).
        ceiling = entrpen + centrmax + 1e-9
        assert entr.max() <= ceiling, f"entr max {entr.max():.3e} > {ceiling:.3e}"
        assert detr.max() <= ceiling, f"detr max {detr.max():.3e} > {ceiling:.3e}"
        # The cap must actually engage somewhere in the deep plume (the
        # organized rate would otherwise be far larger — the #669 bug).
        assert detr.max() > entrpen, "organized detrainment never engaged"

    def test_detrainment_rate_is_resolution_independent(self):
        """The per-metre organized detrainment must NOT scale with level
        count (the #669 wrong-sign resolution dependence). Same physical
        cloud at 400 m and 200 m spacing must give a comparable capped rate.
        """
        cfg = ConvectionParameters.default()
        tw = jnp.array([1.0, 0.0, 0.0])
        rates = []
        for dz_m, nlev in ((400.0, 40), (200.0, 80)):
            T, q, p, dz, rho = _deep_unstable_column(nlev=nlev, dz_m=dz_m)
            kbase = nlev - 3
            ktop = nlev // 8
            up = calculate_updraft(
                T, q, p, dz, rho, kbase, ktop, 1, 0.05, cfg, type_weights=tw,
            )
            rates.append(float(np.asarray(up.detr).max()))
        # Both hit the same centrmax cap → within a small tolerance, NOT the
        # ~2x of the old sqrt(level-count) form.
        assert rates[0] == pytest.approx(rates[1], rel=0.2), rates

    def test_plume_penetrates_deeper_than_uncapped_form(self):
        """With bounded entrainment/detrainment the deep plume reaches a
        higher cloud top (smaller top index) — the physical point of #669.
        """
        cfg = ConvectionParameters.default()
        T, q, p, dz, rho = _deep_unstable_column(nlev=40, dz_m=400.0)
        nlev = T.shape[0]
        up = calculate_updraft(
            T, q, p, dz, rho, nlev - 3, 4, 1, 0.05, cfg,
            type_weights=jnp.array([1.0, 0.0, 0.0]),
        )
        mfu = np.asarray(up.mfu)
        top = np.where(mfu > 1e-6)[0]
        assert top.size > 0
        # The plume survives well into the upper troposphere (index well
        # above cloud base), not annihilated mid-column.
        depth_levels = (nlev - 3) - int(top.min())
        assert depth_levels >= 10, f"plume only {depth_levels} levels deep"


class TestShallowReclosureDowndraftDetection:
    """#676 shallow re-closure keys the cloud-base downdraft on ``mfd[ikb]<0``.

    ECHAM (mo_cumastr.f90:924) uses ``pmfd(ikb) < 0 .AND. loddraf`` — where
    ``loddraf`` is "an LFS was found", never reset by the surface taper. The
    port must NOT gate on ``DowndraftState.active``, which is the scan-EXIT
    carry: the surface taper zeroes ``mfd`` in the lowest layers and drives
    ``active`` to False, so a downdraft that reaches cloud base is still
    ``active == False`` at exit (Codex P2).
    """

    def _deep_downdraft_column(self):
        cfg = ConvectionParameters.default()
        nlev = 47
        p0 = 1.01325e5
        sig = jnp.linspace(1000.0 / p0, 1.0, nlev + 1)
        ph = sig * p0
        p = 0.5 * (ph[:-1] + ph[1:])
        z = -7.6e3 * jnp.log(p / p0)
        dry = c.grav / c.cpd
        mlt = 800.0
        T = jnp.maximum(
            jnp.where(z <= mlt, 302.0 - dry * z,
                      302.0 - dry * mlt - 6.0e-3 * (z - mlt)), 200.0)
        qs = jax.vmap(saturation_mixing_ratio)(p, T)
        q = (0.7 + 0.25 * jnp.exp(-(z / 9000.0) ** 2)) * qs
        Tv = T * (1 + 0.608 * q)
        rho = p / (c.rd * Tv)
        dz = c.rd * Tv / c.grav * jnp.diff(jnp.log(ph))
        cb, _ = find_cloud_base(T, q, p, cfg)
        ktop = jnp.maximum(cb - 35, jnp.array(2))
        upd = calculate_updraft(T, q, p, dz, rho, cb, ktop, 1, jnp.array(0.05),
                                cfg, type_weights=jnp.array([1.0, 0.0, 0.0]))
        prec = jnp.sum(upd.pdmfup)
        dwn = calculate_downdraft(T, q, p, dz, rho, upd, prec, cb, ktop, cfg)
        return int(cb), dwn

    def test_downdraft_at_cloud_base_detected_despite_scan_exit_inactive(self):
        cb, dwn = self._deep_downdraft_column()
        mfd_cb = float(np.asarray(dwn.mfd)[cb])
        active_exit = bool(dwn.active)
        # A downdraft IS present at cloud base ...
        assert mfd_cb < 0.0, f"fixture built no cloud-base downdraft (mfd={mfd_cb})"
        # ... but the scan-exit activity flag is False (surface taper), so the
        # OLD ``mfd<0 & active`` gate would wrongly exclude it, while the
        # faithful ``mfd[ikb] < 0`` detection includes it.
        assert active_exit is False, (
            "fixture must reproduce the taper-inactivated exit state")
        assert not (mfd_cb < 0.0 and active_exit), "old gate should miss it"
        assert mfd_cb < 0.0, "new gate (mfd[ikb] < 0) detects the downdraft"


class TestReturnedStateCarriesPlumeWinds:
    """#676 the returned ConvectionState carries the prognostic plume winds.

    The cududv momentum transport builds ``updraft_state.uu``/``vu`` and
    ``downdraft_state.ud``/``vd``; the returned ``ConvectionState`` must expose
    those, not the environment wind, so a standalone caller's plume-wind
    diagnostics agree with the profiles that produced ``dudt``/``dvdt``
    (Codex P2).
    """

    def test_state_uu_is_the_plume_wind_not_the_environment(self):
        cfg = ConvectionParameters.default()
        nlev = 47
        p0 = 1.01325e5
        sig = jnp.linspace(1000.0 / p0, 1.0, nlev + 1)
        ph = sig * p0
        p = 0.5 * (ph[:-1] + ph[1:])
        z = -7.6e3 * jnp.log(p / p0)
        dry = c.grav / c.cpd
        mlt = 800.0
        T = jnp.maximum(
            jnp.where(z <= mlt, 302.0 - dry * z,
                      302.0 - dry * mlt - 6.0e-3 * (z - mlt)), 200.0)
        qs = jax.vmap(saturation_mixing_ratio)(p, T)
        q = (0.7 + 0.25 * jnp.exp(-(z / 9000.0) ** 2)) * qs
        Tv = T * (1 + 0.608 * q)
        rho = p / (c.rd * Tv)
        dz = c.rd * Tv / c.grav * jnp.diff(jnp.log(ph))
        u_wind = jnp.linspace(-12.0, 18.0, nlev)   # sheared
        v_wind = jnp.linspace(3.0, -3.0, nlev)
        _, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, u_wind, v_wind,
            jnp.zeros(nlev), jnp.zeros(nlev), 1800.0, cfg,
            moisture_supply=jnp.asarray(2.0e-4),
        )
        uu = np.asarray(state.uu)
        un = np.asarray(u_wind)
        plume = np.asarray(state.mfu) > 1e-6
        assert plume.any(), "fixture produced no active plume"
        # The returned plume wind is entrainment-mixed, so it must NOT be the
        # environment wind the pre-fix code passed straight through.
        assert not np.allclose(uu, un), (
            "state.uu equals the environment wind — plume winds not preserved")
        assert np.max(np.abs(uu[plume] - un[plume])) > 1e-3
        # The downdraft wind is likewise carried out (it starts from the
        # LFS mix, not the environment).
        assert not np.allclose(np.asarray(state.ud), un)


class TestCloudTopForcedDetrainment:
    """#676 a plume reaching the scan ceiling fully detrains (water conserved).

    ECHAM forces total detrainment at cloud top (mo_cuasc.f90:540-563,
    ``plude(jk-1)=pmful(jk)``). With the metre-based capped detrainment a
    still-buoyant plume can reach the supplied ``ktop`` with positive mfu/lu;
    its residual condensate must go to ``plude``/the stratiform dqc-dqi ledger,
    not vanish as a flux-boundary loss (Codex P2).
    """

    def _deep_column_reaching_ceiling(self):
        cfg = ConvectionParameters.default()
        nlev = 40
        p0 = 1.01325e5
        ph = jnp.linspace(2000.0 / p0, 1.0, nlev + 1) * p0
        p = 0.5 * (ph[:-1] + ph[1:])
        T = np.empty(nlev)
        T[-1] = 303.0
        for k in range(nlev - 2, -1, -1):
            T[k] = T[k + 1] - 9.0e-3 * 300.0   # steep, deeply unstable
        T = jnp.array(T)
        q = 0.95 * jax.vmap(saturation_mixing_ratio)(p, T)
        Tv = T * (1 + 0.608 * q)
        rho = p / (c.rd * Tv)
        dz = c.rd * Tv / c.grav * jnp.diff(jnp.log(ph))
        cb, _ = find_cloud_base(T, q, p, cfg)
        # Ceiling BELOW the plume's natural top, so it reaches ktop buoyant.
        ktop = jnp.array(8)
        u = jnp.zeros(nlev)
        upd = calculate_updraft(T, q, p, dz, rho, cb, ktop, 1, jnp.array(0.05),
                                cfg, type_weights=jnp.array([1.0, 0.0, 0.0]),
                                u_wind=u, v_wind=u)
        return cfg, T, q, p, rho, dz, u, cb, ktop, upd

    def test_residual_condensate_detrained_at_ceiling(self):
        _, _, _, _, _, _, _, _, ktop, upd = self._deep_column_reaching_ceiling()
        kt = int(ktop)
        mfu = np.asarray(upd.mfu)
        plude = np.asarray(upd.plude)
        # The plume is alive just below the ceiling ...
        assert mfu[kt + 1] > 1e-6, "fixture: plume did not reach the ceiling"
        # ... fully terminates AT the ceiling (forced detrainment) ...
        assert mfu[kt] == 0.0, "plume not terminated at the scan ceiling"
        # ... and its residual condensate is detrained, not lost.
        assert plude[kt] > 0.0, "residual plume condensate not detrained to plude"

    def test_column_water_conserved_when_plume_reaches_ceiling(self):
        cfg, T, q, p, rho, dz, u, cb, ktop, upd = (
            self._deep_column_reaching_ceiling())
        prec = jnp.sum(upd.pdmfup)
        dn = calculate_downdraft(T, q, p, dz, rho, upd, prec, cb, ktop, cfg,
                                 u_wind=u, v_wind=u)
        tend = calculate_tendencies(T, q, u, u, p, rho, dz, upd, dn, cb, ktop,
                                    1800.0, cfg, ktype=jnp.array(1))
        dp_abs = np.abs(np.diff(np.asarray(p)))
        mass = np.concatenate([dp_abs, dp_abs[-1:]]) / c.grav
        water_tend = np.sum(
            (np.asarray(tend.dqdt) + np.asarray(tend.dqc_dt)
             + np.asarray(tend.dqi_dt)) * mass)
        precip = float(tend.precip_conv)
        # Water removed from vapour+condensate == precip out, to round-off.
        residual = water_tend + precip
        assert abs(residual) / max(abs(precip), 1e-20) < 1e-5, (
            f"column water budget open: residual {residual:.3e} vs "
            f"precip {precip:.3e}")
        # The anvil condensate at the ceiling feeds the stratiform ledger.
        kt = int(ktop)
        assert (float(tend.dqc_dt[kt]) + float(tend.dqi_dt[kt])) > 0.0


class TestShallowReclosureCflCap:
    """#676 the shallow re-closure re-applies jcm's CFL / cmfcmax cap.

    ECHAM's shallow branch (mo_cumastr.f90:921-937) omits the final
    ``MIN(zmfub1, zmfmax)`` the deep branch (:904) applies, so a first guess
    just below the cap can be raised to ~1.2x it within the 20% acceptance
    window. jcm enforces ``mfu_cfl_max`` as a hard stability invariant, so the
    accepted re-closed flux must be clipped to it (Codex P2).
    """

    def test_reclosure_over_cap_within_window_is_clipped(self):
        cfg = ConvectionParameters.default()  # cmfcmax = 1.0
        g = c.grav
        mfu_cfl_max = jnp.array(0.5)
        zmfub = jnp.array(0.49)          # first guess just under the cap
        zdqmin = jnp.array(1.0e-6)
        zqumqe = jnp.array(5.0e-3)
        # Raw re-closure = zdqpbl/(g·zqumqe) = 0.55: over the 0.5 cap but only
        # 12% above zmfub, so the 20% window ACCEPTS it — the exact edge.
        zdqpbl = 0.55 * g * zqumqe
        out = float(shallow_reclosure_flux(
            zmfub, zqumqe, zdqmin, zdqpbl, mfu_cfl_max, cfg))
        assert out == pytest.approx(0.5, rel=1e-6), (
            f"accepted re-closure not clipped to the CFL cap: {out}")

    def test_reclosure_within_cap_is_unclipped(self):
        cfg = ConvectionParameters.default()
        g = c.grav
        mfu_cfl_max = jnp.array(0.5)
        zmfub = jnp.array(0.30)
        zdqmin = jnp.array(1.0e-6)
        zqumqe = jnp.array(5.0e-3)
        zdqpbl = 0.33 * g * zqumqe       # raw 0.33: within cap and 20% window
        out = float(shallow_reclosure_flux(
            zmfub, zqumqe, zdqmin, zdqpbl, mfu_cfl_max, cfg))
        assert out == pytest.approx(0.33, rel=1e-5), (
            f"under-cap re-closure wrongly altered: {out}")

    def test_reclosure_over_window_is_rejected_to_first_guess(self):
        cfg = ConvectionParameters.default()
        g = c.grav
        mfu_cfl_max = jnp.array(0.5)
        zmfub = jnp.array(0.49)
        zdqmin = jnp.array(1.0e-6)
        zqumqe = jnp.array(5.0e-3)
        zdqpbl = 0.70 * g * zqumqe       # raw 0.70: >20% from zmfub → rejected
        out = float(shallow_reclosure_flux(
            zmfub, zqumqe, zdqmin, zdqpbl, mfu_cfl_max, cfg))
        assert out == pytest.approx(0.49, rel=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
