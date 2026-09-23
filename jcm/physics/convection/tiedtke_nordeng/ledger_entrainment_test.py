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
    column_environment,
)
from jcm.physics.convection.tiedtke_nordeng.half_levels import (
    reconstruct_pressure_half,
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

    Setting the plume to the HALF-level environment, ``tu == ptenh`` (zero
    DSE deviation flux) and ``qu == pqenh`` (zero moisture deviation flux),
    with ``mfd = plude = pdmfup = pdmfdp = 0`` leaves the condensate flux
    ``L·lu·mfu`` as the ONLY nonzero divergence, so ``dtedt`` is exactly
    ``−Δ(palvsh·lu·mfu)/(cp·Δp/g)`` — the latent heat keyed to the
    half-level temperature ``ptenh`` the flux crosses — and can be
    hand-computed. ``cp`` is ECHAM's moist ``pcpen = cpd·(1 + vtmpc2·q)``
    (``zrcpm``, mo_cufluxdts.f90:654), built here by hand from the column's
    uniform 5 g/kg; ``Δp`` is the true layer thickness between the column's
    interfaces (the full-level midpoints, 200 hPa for every layer here).
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
        p_half = reconstruct_pressure_half(pressure)
        env = column_environment(temperature, humidity, pressure,
                                 pressure_half=p_half)
        # Condensate flux lu·mfu = [0, 1e-4, 2e-4, 3e-4, 4e-4] through each
        # layer's TOP interface — nonzero at the top of the surface layer
        # (last index) so the surface-layer closure is exercised.
        mfu = jnp.array([0.0, 0.1, 0.1, 0.1, 0.1])
        lu = jnp.array([0.0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3])
        up = _zero_updraft(nlev)._replace(
            tu=env.tenh, qu=env.qenh, lu=lu, mfu=mfu,
        )
        dn = _zero_downdraft(nlev)
        tend = calculate_tendencies(
            temperature, humidity, jnp.zeros(nlev), jnp.zeros(nlev),
            pressure, rho, dz, up, dn, kbase=4, ktop=1, dt=1800.0,
            config=ConvectionParameters.default(), ktype=jnp.array(1),
            pressure_half=p_half,
        )
        mass = np.diff(np.asarray(p_half)) / c.grav
        np.testing.assert_allclose(mass, 2.0e4 / c.grav, rtol=1e-6)
        cond_flux = np.asarray(lu * mfu)
        self.alvsh = np.asarray(env.alvsh)
        return tend, cond_flux, mass

    def test_phase_keyed_latent_heat_cold_uses_alhs(self):
        tend, cond_flux, mass = self._setup(250.0)  # below tmelt
        # Every interface is below the melting point, so palvsh = alhs.
        assert np.all(self.alvsh == c.alhs)
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
        assert np.all(self.alvsh == c.alhc)
        div = np.diff(np.append(cond_flux, 0.0))
        expected = -c.alhc * div / (self.CP * mass)
        np.testing.assert_allclose(np.asarray(tend.dtedt), expected, rtol=1e-5)

    def test_surface_layer_receives_tendency(self):
        # With the plume flux nonzero at the surface (last index), the
        # ECHAM klev closure gives dtedt[surface] = -(-F[surface])... i.e.
        # nonzero, where the old diff-into-[:-1] left it exactly 0.
        tend, cond_flux, mass = self._setup(290.0)
        surf = float(tend.dtedt[-1])
        expected_surf = -(-c.alhc * cond_flux[-1]) / (self.CP * mass[-1])
        assert surf != 0.0
        assert surf == pytest.approx(expected_surf, rel=1e-5)


# --------------------------------------------------------------------------
# #872 — the cudtdq ledger uses ECHAM's MOIST heat capacity
# --------------------------------------------------------------------------
class TestMoistHeatCapacityLedger:
    """``pmfus``/``pmfds`` carry ``pcpcu·T + φ`` and ``zrcpm = 1/pcpen``.

    A plume 1 K warmer than the HALF-level environment (``tu = ptenh + 1``)
    with no condensate, in a column whose humidity falls with height,
    isolates the dry-static-energy deviation flux through each layer's top
    interface, ``F_k = pcpcu_k·(tu_k − ptenh_k)·mfu_k``. Its divergence
    telescopes over the column, so the enthalpy the ledger deposits on the
    true layer mass, ``Σ pcpen_k·dT_k·m_k``, vanishes to round-off with the
    SAME moist ``cp`` the ledger divides by (mo_cufluxdts.f90:198-204,
    654-656) — and is open by ``~vtmpc2·Δq`` when integrated with dry
    ``cpd``.
    """

    def _run(self):
        nlev = 6
        pressure = jnp.linspace(3.0e4, 1.0e5, nlev)
        temperature = jnp.linspace(240.0, 300.0, nlev)
        humidity = jnp.linspace(1.0e-3, 1.8e-2, nlev)
        rho = pressure / (c.rd * temperature)
        dz = jnp.full(nlev, 1000.0)
        p_half = reconstruct_pressure_half(pressure)
        env = column_environment(temperature, humidity, pressure,
                                 pressure_half=p_half)
        mfu = jnp.array([0.0, 0.02, 0.05, 0.08, 0.06, 0.03])
        up = _zero_updraft(nlev)._replace(
            tu=env.tenh + 1.0, qu=env.qenh, mfu=mfu,
        )
        tend = calculate_tendencies(
            temperature, humidity, jnp.zeros(nlev), jnp.zeros(nlev),
            pressure, rho, dz, up, _zero_downdraft(nlev), kbase=nlev - 1,
            ktop=1, dt=1800.0, config=ConvectionParameters.default(),
            ktype=jnp.array(1), pressure_half=p_half,
        )
        mass = np.diff(np.asarray(p_half)) / c.grav
        cp = c.cpd * (1.0 + c.vtmpc2 * np.asarray(humidity))
        self.cpcu = np.asarray(env.cpcu)
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
        flux = self.cpcu * 1.0 * mfu
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
    ``loddraf`` is "an LFS was found", never reset by the surface taper,
    which drives the flux itself to zero at the surface. A downdraft that
    reaches the cloud-base interface must therefore be seen there, with
    ``DowndraftState.active`` carrying ``loddraf`` (Codex P2).
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
        # A moist boundary layer under a dry (50 % RH) free troposphere: the
        # wet-bulb depression of the dry air is what makes cudlfs' 50/50
        # plume/environment mixture negatively buoyant inside the cloud.
        q = jnp.where(z < 1500.0, 0.92, 0.5) * qs
        Tv = T * (1 + 0.608 * q)
        rho = p / (c.rd * Tv)
        dz = c.rd * Tv / c.grav * jnp.diff(jnp.log(ph))
        cb, _ = find_cloud_base(T, q, p, cfg)
        ktop = jnp.maximum(cb - 35, jnp.array(2))
        upd = calculate_updraft(T, q, p, dz, rho, cb, ktop, 1, jnp.array(0.05),
                                cfg, type_weights=jnp.array([1.0, 0.0, 0.0]))
        prec = jnp.sum(upd.pdmfup)
        # cudlfs searches inside the REALIZED cloud (kctop < jk < kcbot).
        kctop = int(np.min(np.where(np.asarray(upd.mfu) > 0.0)[0]))
        dwn = calculate_downdraft(T, q, p, dz, rho, upd, prec, cb, kctop, cfg)
        return int(cb), dwn

    def test_downdraft_at_cloud_base_detected_despite_surface_taper(self):
        cb, dwn = self._deep_downdraft_column()
        mfd = np.asarray(dwn.mfd)
        # A downdraft IS present at the cloud-base interface ...
        assert mfd[cb] < 0.0, f"fixture built no cloud-base downdraft (mfd={mfd[cb]})"
        # ... the surface taper has driven it to (near) zero at the lowest
        # interface, yet ``active`` still records that an LFS was found.
        assert abs(mfd[-1]) < abs(mfd[cb])
        assert bool(dwn.active), "active must carry ECHAM's loddraf"


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


class TestCloudTopOvershoot:
    """cuasc's cloud-top overshoot (mo_cuascent.f90:540-565).

    A plume still alive at its last passing interface ``kctop`` does not
    stop dead: a fraction ``cmfctop`` of the flux there carries on to the
    interface above, ``kctop − 1``, with the properties the ascent gave it
    there and no precipitation; the rest, ``(1 − cmfctop)·pmfu(kctop)``,
    detrains in layer ``kctop − 1`` with the plume's condensate, and the
    overshooting condensate ``pmful(kctop − 1)`` detrains in the layer above
    that. Here the cloud-top bound ``kctop0`` ends a still-buoyant plume, so
    every one of those is non-trivial.
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
        cb, _ = find_cloud_base(T, q, p, cfg, pressure_half=ph)
        # The bound BELOW the plume's natural top, so it reaches it buoyant.
        ktop = jnp.array(8)
        u = jnp.zeros(nlev)
        upd = calculate_updraft(T, q, p, dz, rho, cb, ktop, 1, jnp.array(0.05),
                                cfg, type_weights=jnp.array([1.0, 0.0, 0.0]),
                                u_wind=u, v_wind=u, pressure_half=ph)
        self.ph = ph
        return cfg, T, q, p, rho, dz, u, cb, ktop, upd

    def test_overshoot_flux_and_detrainment(self):
        cfg, _, _, _, _, _, _, _, ktop, upd = (
            self._deep_column_reaching_ceiling())
        kt = int(ktop)
        cmfctop = float(cfg.cu_cmfctop)
        mfu = np.asarray(upd.mfu, dtype=np.float64)
        lu = np.asarray(upd.lu, dtype=np.float64)
        plude = np.asarray(upd.plude, dtype=np.float64)
        # The plume passes the ascent test up to the bound ...
        assert int(upd.kctop) == kt
        assert mfu[kt] > 1e-6, "fixture: plume did not reach the bound"
        # ... a fraction cmfctop of its flux overshoots one interface ...
        np.testing.assert_allclose(mfu[kt - 1], cmfctop * mfu[kt], rtol=1e-3)
        # ... and nothing rises further.
        assert np.all(mfu[:kt - 1] == 0.0)
        # The rest detrains in the overshoot layer with the condensate of
        # the plume leaving kctop; the overshoot's own condensate detrains
        # in the layer above it.
        np.testing.assert_allclose(
            plude[kt - 1], (1.0 - cmfctop) * mfu[kt] * lu[kt], rtol=1e-3)
        np.testing.assert_allclose(
            plude[kt - 2], mfu[kt - 1] * lu[kt - 1], rtol=1e-3)
        assert plude[kt - 2] > 0.0
        # The overshoot does not precipitate.
        assert float(upd.pdmfup[kt - 1]) == pytest.approx(0.0, abs=1e-12)

    def test_column_water_conserved_when_plume_reaches_ceiling(self):
        cfg, T, q, p, rho, dz, u, cb, ktop, upd = (
            self._deep_column_reaching_ceiling())
        prec = jnp.sum(upd.pdmfup)
        dn = calculate_downdraft(T, q, p, dz, rho, upd, prec, cb, ktop, cfg,
                                 u_wind=u, v_wind=u, pressure_half=self.ph)
        tend = calculate_tendencies(T, q, u, u, p, rho, dz, upd, dn, cb, ktop,
                                    1800.0, cfg, ktype=jnp.array(1),
                                    pressure_half=self.ph)
        # The true layer mass between the column's interfaces.
        mass = np.diff(np.asarray(self.ph)) / c.grav
        water_tend = np.sum(
            (np.asarray(tend.dqdt) + np.asarray(tend.dqc_dt)
             + np.asarray(tend.dqi_dt)) * mass)
        precip = float(tend.precip_conv)
        # Water removed from vapour+condensate == precip out, to round-off.
        residual = water_tend + precip
        assert abs(residual) / max(abs(precip), 1e-20) < 1e-5, (
            f"column water budget open: residual {residual:.3e} vs "
            f"precip {precip:.3e}")
        # The overshoot's condensate, detrained above the cloud top, feeds
        # the stratiform ledger.
        kt = int(ktop)
        assert (float(tend.dqc_dt[kt - 2]) + float(tend.dqi_dt[kt - 2])) > 0.0


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


# --------------------------------------------------------------------------
# #872 — the plume DSE mixing pairs each level with the right moist cp
# --------------------------------------------------------------------------
class TestPlumeHeatCapacityIndexing:
    """cuasc/cuddraf carry heat across a layer with the half-level ``pcpcu`` of
    the interface the air leaves and convert back with that of the interface
    it reaches (mo_cuascent.f90:388-411, mo_cudescent.f90:222-233). An
    artificial ``cp`` gradient makes a wrong or dropped index shift
    measurable (it moves the plume temperature by tenths of a K).
    """

    NLEV = 8

    def _column(self):
        nlev = self.NLEV
        pressure = jnp.linspace(3.0e4, 1.0e5, nlev)
        temperature = jnp.linspace(250.0, 300.0, nlev)
        # Near-dry so no saturation adjustment interferes with the lift.
        humidity = jnp.full(nlev, 1.0e-7)
        dz = jnp.linspace(900.0, 400.0, nlev)
        rho = pressure / (c.rd * temperature)
        # Strongly level-dependent cp (a stand-in for a humidity lapse).
        cp = c.cpd * (1.0 + 0.03 * jnp.linspace(0.0, 1.0, nlev) ** 2)
        env = column_environment(temperature, humidity, pressure, cp)
        return pressure, temperature, humidity, dz, rho, cp, env

    def test_updraft_lift_uses_source_and_destination_cp(self):
        p, T, q, dz, rho, cp, env = self._column()
        cfg = ConvectionParameters.default(
            entrpen=0.0, entrscv=0.0, entrmid=0.0, cu_centrmax=0.0)
        kbase = self.NLEV - 2
        up = calculate_updraft(
            T, q, p, dz, rho, kbase, 1, 2, jnp.array(0.05), cfg,
            type_weights=jnp.array([0.0, 1.0, 0.0]), cp_moist=cp,
        )
        k = kbase - 1
        cpcu = np.asarray(env.cpcu)
        geoh = np.asarray(env.geoh)
        expected = (cpcu[kbase] * float(up.tu[kbase]) + geoh[kbase]
                    - geoh[k]) / cpcu[k]
        np.testing.assert_allclose(float(up.tu[k]), expected, rtol=1e-6)
        # A single (destination) cp for both ends is measurably different.
        unshifted = float(up.tu[kbase]) - (geoh[k] - geoh[kbase]) / cpcu[k]
        assert abs(float(up.tu[k]) - unshifted) > 0.1

    def test_downdraft_descent_uses_source_and_destination_cp(self):
        from jcm.physics.convection.tiedtke_nordeng.updraft import (
            UpdatedraftState,
        )
        p, T, q, dz, rho, cp, env = self._column()
        nlev = self.NLEV
        # No entrainment, so the descent below the LFS is a pure dry lift
        # down one layer followed by cuadjtq's evaporation into the parcel.
        cfg = ConvectionParameters.default(entrdd=0.0)
        kbase, ktop = nlev - 2, 1
        # A cold plume makes the LFS the first eligible interface (2).
        z = jnp.zeros(nlev)
        up = UpdatedraftState(
            tu=jnp.full(nlev, 200.0), qu=q, lu=z,
            mfu=jnp.zeros(nlev).at[kbase].set(0.05), entr=z, detr=z, buoy=z,
            pdmfup=z, plude=z, uu=z, vu=z,
        )
        dn = calculate_downdraft(T, q, p, dz, rho, up, jnp.array(1.0),
                                 kbase, ktop, cfg, cp_moist=cp, env=env)
        lfs = int(dn.lfs)
        assert lfs == 2 and bool(dn.active)
        j = lfs + 1
        td, qd = np.asarray(dn.td), np.asarray(dn.qd)
        # Undo cuadjtq(kcall=2)'s moistening, whose L/cp is DRY (the
        # reference table), to recover the dry descent.
        L = c.alhc if td[j] >= c.tmelt else c.alhs
        td_lift = td[j] + L / c.cpd * (qd[j] - qd[j - 1])
        cpcu = np.asarray(env.cpcu)
        geoh = np.asarray(env.geoh)
        expected = (cpcu[j - 1] * td[j - 1] + geoh[j - 1] - geoh[j]) / cpcu[j]
        np.testing.assert_allclose(td_lift, expected, rtol=1e-5)
        unshifted = td[j - 1] + (geoh[j - 1] - geoh[j]) / cpcu[j]
        assert abs(td_lift - unshifted) > 0.1
