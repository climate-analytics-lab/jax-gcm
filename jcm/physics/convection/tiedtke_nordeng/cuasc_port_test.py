"""The ECHAM ``cuasc``/``cuentr``/``cumastr`` ascent rules, one test per rule.

Each test pins one piece of the reference ascent against a value computed
from its Fortran definition (mo_cuascent.f90, mo_cumastr.f90,
mo_cuinitialize.f90):

* ``klwmin`` — the level of maximum resolved ascent;
* the cloud-top bounds ``kctop0``: the 400 hPa bound of a column without a
  surface plume, and ``cumastr``'s first-pass estimate ``ictop0``;
* ``khmin`` — where organized detrainment may start;
* the vertical gating of turbulent entrainment for deep, shallow and
  mid-level plumes, and the mid-level ``zentest`` enhancement;
* organized detrainment acting only from ``khmin`` up;
* the ascent stopping at the first interface where the plume does not
  condense, even when it is still buoyant;
* the ascent never passing ``kctop0``, and the deep scheme reaching the
  upper troposphere when nothing bounds it lower.

The cloud-top overshoot (``cmfctop``) is pinned in
``ledger_entrainment_test.TestCloudTopOvershoot``.
"""

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.half_level_ledger_test import (
    _l47_tropical_column,
)
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    find_cloud_base,
    tiedtke_nordeng_convection,
)
from jcm.physics.convection.tiedtke_nordeng.updraft import (
    calculate_updraft,
    cloud_base_mse,
    column_environment,
    cubase_parcel,
    estimate_cloud_top,
    max_ascent_level,
    mse_minimum_level,
    no_cubase_cloud_top_bound,
    saturated_mse_hat,
)

DEEP = jnp.array([1.0, 0.0, 0.0])
SHALLOW = jnp.array([0.0, 1.0, 0.0])
MID = jnp.array([0.0, 0.0, 1.0])
#: A mid-level cloud base on the L47 column (~830 hPa): a ``cubasmc`` plume
#: seeded there condenses and rises.
MID_BASE = 41


def _column(rh=0.95):
    """Build the L47 hybrid tropical column, its environment and cloud base."""
    T, q, p, p_half, dz, rho = _l47_tropical_column(rh=rh)
    cfg = ConvectionParameters.default()
    env = column_environment(T, q, p, pressure_half=p_half)
    kb, _ = find_cloud_base(T, q, p, cfg, pressure_half=p_half)
    return cfg, T, q, p, p_half, dz, rho, env, int(kb)


def _ascent(cfg, T, q, p, p_half, dz, rho, kb, ktop, ktype, weights, **kw):
    return calculate_updraft(
        T, q, p, dz, rho, kb, ktop, ktype, jnp.array(0.02), cfg,
        type_weights=weights, pressure_half=p_half, **kw)


def _dz_p(env, k):
    """Return cuentr's layer depth ``Δp·zrrho/g`` of layer ``k``."""
    paph = np.asarray(env.paph)
    tenh, qenh = np.asarray(env.tenh), np.asarray(env.qenh)
    zrrho = c.rd * tenh[k + 1] * (1 + c.vtmpc1 * qenh[k + 1]) / paph[k + 1]
    return (paph[k + 1] - paph[k]) * zrrho / c.grav


class TestMaxAscentLevel:
    """``klwmin`` (mo_cuinitialize.f90:189-196)."""

    def test_level_of_most_negative_omega(self):
        omega = jnp.zeros(10).at[6].set(-0.3).at[4].set(-0.1)
        assert int(max_ascent_level(omega)) == 6

    def test_ties_keep_the_lowest_level(self):
        # The walk runs upward with a strict ``<``: the lowest of equal
        # minima wins.
        omega = jnp.zeros(10).at[3].set(-0.2).at[7].set(-0.2)
        assert int(max_ascent_level(omega)) == 7

    def test_top_two_levels_are_not_searched(self):
        omega = jnp.zeros(10).at[1].set(-5.0).at[5].set(-0.1)
        assert int(max_ascent_level(omega)) == 5

    def test_no_ascent_gives_the_lowest_level(self):
        assert int(max_ascent_level(jnp.full(10, 0.2))) == 9


class TestCloudTopBounds:
    """The first-pass cloud-top bounds ``kctop0``."""

    def test_no_cubase_bound_is_lowest_interface_above_400hpa(self):
        p_half = jnp.array([0.0, 1.0e4, 2.5e4, 3.9e4, 4.1e4, 6.0e4, 8.0e4,
                            1.0e5])
        # Interface 3 (390 hPa) is the lowest with p < 400 hPa
        # (mo_cuascent.f90:191).
        assert int(no_cubase_cloud_top_bound(p_half)) == 3

    def test_ictop0_is_highest_interface_the_parcel_exceeds(self):
        nlev = 12
        kcbot = 10
        hhatt = jnp.full(nlev, 3.5e5)
        # The parcel (3.4e5) exceeds the reduced saturation MSE at 4 and 6;
        # 1 is above the search range (interfaces ≥ 3, 1-based) and 9 is
        # within two interfaces of cloud base.
        hhatt = hhatt.at[jnp.array([1, 4, 6, 9])].set(3.3e5)
        assert int(estimate_cloud_top(hhatt, 3.4e5, kcbot)) == 4

    def test_ictop0_without_a_crossing_is_just_above_cloud_base(self):
        hhatt = jnp.full(12, 3.5e5)
        assert int(estimate_cloud_top(hhatt, 3.4e5, 10)) == 9

    def test_khmin_lies_between_ictop0_and_cloud_base(self):
        cfg, T, q, p, p_half, dz, rho, env, kb = _column()
        tu, qu, _ = cubase_parcel(env, kb)
        ictop0 = int(estimate_cloud_top(
            saturated_mse_hat(env), cloud_base_mse(env, kb, tu, qu), kb))
        khmin = int(mse_minimum_level(env, T, q, env.cpcu, kb, ictop0))
        assert ictop0 < kb
        assert ictop0 <= khmin <= kb


class TestEntrainmentGating:
    """cuentr's vertical gating of turbulent entrainment (lines 719-765).

    Turbulent DETRAINMENT acts at every layer above cloud base; turbulent
    ENTRAINMENT only in each plume type's band. With the organized rates
    switched off (``cu_centrmax = 0``) and no step limiter, the diagnostic
    entrainment rate is exactly the plume type's rate inside its band and
    zero outside it.
    """

    def _rates(self, ktype, weights, ktop, kb=None, **kw):
        cfg, T, q, p, p_half, dz, rho, env, kb_surface = _column()
        kb = kb_surface if kb is None else kb
        cfg = cfg.replace(cu_centrmax=jnp.array(0.0))
        up = _ascent(cfg, T, q, p, p_half, dz, rho, kb, ktop, ktype,
                     weights, **kw)
        alive = np.asarray(up.mfu) > 0.0
        # Layers the continuing plume crossed: above cloud base, below the
        # last passing interface (the overshoot layer detrains only).
        layers = np.arange(T.shape[0])
        crossed = (layers < kb) & (layers >= int(up.kctop)) & alive
        return cfg, np.asarray(p_half), kb, up, crossed, env

    def test_shallow_entrains_within_200hpa_or_the_lower_half(self):
        ktop = 20
        cfg, p_half, kb, up, crossed, env = self._rates(2, SHALLOW, ktop)
        entr = np.asarray(up.entr)
        detr = np.asarray(up.detr)
        zpmid = 0.5 * (p_half[kb] + p_half[ktop])
        band = ((p_half[kb] - p_half[:-1]) <= 2.0e4) | (p_half[:-1] > zpmid)
        above = crossed & ~band
        assert above.any() and (crossed & band).any(), "fixture"
        np.testing.assert_allclose(entr[crossed & band], float(cfg.entrscv),
                                   rtol=1e-5)
        np.testing.assert_array_equal(entr[above], 0.0)
        # Detrainment is not gated: ``pentr·pmfu·Δz_p`` capped at 0.75 of
        # the entering flux (cuasc line 360).
        tenh, qenh = np.asarray(env.tenh), np.asarray(env.qenh)
        kp1 = np.minimum(np.arange(entr.shape[0]) + 1, entr.shape[0] - 1)
        zrrho = c.rd * tenh[kp1] * (1 + c.vtmpc1 * qenh[kp1]) / p_half[1:]
        dz_p = np.diff(p_half) * zrrho / c.grav
        expected = np.minimum(float(cfg.entrscv), 0.75 / dz_p)
        np.testing.assert_allclose(detr[above], expected[above], rtol=1e-4)

    def test_deep_entrains_below_max_ascent_or_the_lower_half(self):
        ktop = 12
        klwmin = 30
        cfg, p_half, kb, up, crossed, _ = self._rates(
            1, DEEP, ktop, klwmin=jnp.array(klwmin))
        entr = np.asarray(up.entr)
        zpmid = 0.5 * (p_half[kb] + p_half[ktop])
        layers = np.arange(entr.shape[0])
        band = (layers >= max(klwmin, ktop + 2)) | (p_half[:-1] > zpmid)
        above = crossed & ~band
        assert above.any() and (crossed & band).any(), "fixture"
        np.testing.assert_allclose(entr[crossed & band], float(cfg.entrpen),
                                   rtol=1e-5)
        np.testing.assert_array_equal(entr[above], 0.0)

    def test_mid_level_entrains_only_below_max_ascent(self):
        ktop = 12
        klwmin = 34
        cfg, p_half, kb, up, crossed, _ = self._rates(
            3, MID, ktop, kb=MID_BASE, klwmin=jnp.array(klwmin),
            lift=jnp.array(0.5))
        entr = np.asarray(up.entr)
        layers = np.arange(entr.shape[0])
        # The first step of a mid-level plume crosses layer kcbot unmixed.
        mixing = crossed & (layers < kb)
        band = layers >= klwmin
        assert (mixing & band).any() and (mixing & ~band).any(), "fixture"
        np.testing.assert_allclose(entr[mixing & band], float(cfg.entrmid),
                                   rtol=1e-5)
        np.testing.assert_array_equal(entr[mixing & ~band], 0.0)

    def test_mid_level_zentest_adds_the_moisture_convergence(self):
        """``zentest = min(centrmax, max(pqte,0)/pqenh(k+1)/(pmfu·zrrho))``
        (lines 756-760), added where the half-level humidity below the
        layer exceeds 1e-5 kg/kg.
        """
        ktop = 12
        klwmin = 34
        cfg, T, q, p, p_half, dz, rho, env, _ = _column()
        kb = MID_BASE
        cfg = cfg.replace(cu_centrmax=jnp.array(1.0))
        nlev = T.shape[0]
        qte = jnp.zeros(nlev).at[36:kb].set(2.0e-8)
        up = _ascent(cfg, T, q, p, p_half, dz, rho, kb, ktop, 3, MID,
                     klwmin=jnp.array(klwmin), moisture_tendency=qte,
                     lift=jnp.array(0.5))
        up0 = _ascent(cfg, T, q, p, p_half, dz, rho, kb, ktop, 3, MID,
                      klwmin=jnp.array(klwmin), lift=jnp.array(0.5))
        # zentest is formed with the flux entering the layer, which the
        # enhanced entrainment itself changes further up; compare the
        # lowest layer that mixes (the first step of a mid-level plume
        # crosses layer kcbot unmixed), where both ascents arrive with the
        # same flux.
        k = kb - 1
        assert k >= klwmin and float(qte[k]) > 0.0, "fixture"
        np.testing.assert_allclose(float(up.mfu[kb]), float(up0.mfu[kb]))
        mfu_b = float(up.mfu[kb])
        paph = np.asarray(env.paph)
        tenh, qenh = np.asarray(env.tenh), np.asarray(env.qenh)
        zrrho = c.rd * tenh[k + 1] * (1 + c.vtmpc1 * qenh[k + 1]) / paph[k + 1]
        zentest = min(1.0, 2.0e-8 / qenh[k + 1] / (mfu_b * zrrho))
        np.testing.assert_allclose(float(up0.entr[k]), float(cfg.entrmid),
                                   rtol=1e-5)
        np.testing.assert_allclose(
            float(up.entr[k]) - float(up0.entr[k]), zentest, rtol=1e-4)


class TestOrganizedDetrainmentOnset:
    """Organized detrainment acts only from ``khmin`` up to ``kctop0``
    (cuentr lines 767-788): below ``khmin`` a deep plume detrains at the
    turbulent rate alone.
    """

    def test_detrainment_is_turbulent_below_khmin(self):
        cfg, T, q, p, p_half, dz, rho, env, kb = _column()
        ktop = 12
        khmin = 30
        up = _ascent(cfg, T, q, p, p_half, dz, rho, kb, ktop, 1, DEEP,
                     khmin=jnp.array(khmin))
        detr = np.asarray(up.detr)
        layers = np.arange(detr.shape[0])
        crossed = (layers < kb) & (layers >= int(up.kctop))
        below = crossed & (layers > khmin)
        onset = crossed & (layers <= khmin)
        assert below.any() and onset.any(), "fixture"
        np.testing.assert_allclose(detr[below], float(cfg.entrpen),
                                   rtol=1e-5)
        assert detr[onset].max() > float(cfg.entrpen)


class TestAscentStops:
    """The ascent test (mo_cuascent.f90:442-466)."""

    def test_stops_at_first_non_condensing_interface_while_buoyant(self):
        """Above a saturated boundary layer the environment is warm-bottomed
        but very dry: the plume, diluted by the shallow entrainment, no
        longer condenses at the first interface above cloud base — and
        stops there, although it is still warmer than its surroundings.
        """
        cfg, T, q, p, p_half, dz, rho, env, kb = _column()
        dry = jnp.arange(T.shape[0]) < kb
        q_dry = jnp.where(dry, 0.05 * q, q)
        env_d = column_environment(T, q_dry, p, pressure_half=p_half)
        up = calculate_updraft(
            T, q_dry, p, dz, rho, kb, 2, 2, jnp.array(0.02), cfg,
            type_weights=SHALLOW, pressure_half=p_half)
        mfu = np.asarray(up.mfu)
        k = kb - 1
        # No interface above cloud base passed: the plume's top is its base
        # and only the overshoot reaches the next interface.
        assert int(up.kctop) == kb
        np.testing.assert_allclose(
            mfu[k], float(cfg.cu_cmfctop) * mfu[kb], rtol=1e-6)
        assert np.all(mfu[:k] == 0.0)
        # ... although the mixed parcel there is still buoyant.
        assert float(up.tu[k]) > float(env_d.tenh[k])
        # And it did not condense: no condensate gained above the base.
        assert float(up.pdmfup[k]) == 0.0
        # The overshoot carries the air mixed in that layer (cuasc forms the
        # plume's tracer mixture before the test), so the tracer ledger
        # sees the layer's whole entrainment, and none above it.
        dmfen = np.asarray(up.dmfen)
        np.testing.assert_allclose(
            dmfen[k], float(up.entr[k]) * mfu[kb] * _dz_p(env_d, k), rtol=1e-4)
        assert np.all(dmfen[:k] == 0.0)

    def test_ascent_never_passes_the_cloud_top_bound(self):
        cfg, T, q, p, p_half, dz, rho, env, kb = _column()
        for ktop in (38, 30, 20):
            up = _ascent(cfg, T, q, p, p_half, dz, rho, kb, ktop, 1, DEEP)
            mfu = np.asarray(up.mfu)
            assert int(up.kctop) >= ktop
            # Only the overshoot may reach the interface above the top.
            assert np.all(mfu[:int(up.kctop) - 1] == 0.0)


class TestDeepPlumeReachesUpperTroposphere:
    """End to end: on a moist tropical L47 column with resolved convergence
    the deep plume rises well into the upper troposphere — the ascent is
    bounded only by ECHAM's cloud-top estimates, not by a level count.
    """

    def test_l47_deep_cloud_top_above_300hpa(self):
        T, q, p, p_half, dz, rho = _l47_tropical_column(rh=0.8)
        nlev = T.shape[0]
        mass = jnp.diff(p_half) / c.grav
        supply = 1.5e-4
        sl = slice(nlev // 2, nlev - 4)
        conv = jnp.zeros(nlev).at[sl].set(1.5 * supply / jnp.sum(mass[sl]))
        z = jnp.zeros(nlev)
        _, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, z, z, z, z, 900.0,
            ConvectionParameters.default(), pressure_half=p_half,
            moisture_supply=jnp.asarray(supply), qte_dynamics=conv,
        )
        assert int(state.ktype) == 1
        mfu = np.asarray(state.mfu)
        top = int(np.nonzero(mfu > 0.0)[0].min())
        assert float(p_half[top]) < 3.0e4, float(p_half[top])
