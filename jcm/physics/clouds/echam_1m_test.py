"""Tests of the ECHAM6.3 1-moment cloud scheme (``echam_1m.py``).

The process tests drive one level of the sweep (``_sweep_level``) with chosen
incoming fluxes and compare each intermediate with the formula of
``mo_cloud.f90`` (ECHAM6.3-HAM2.3 r7492, lines cited as ``F:``) evaluated
independently here in NumPy float64. Each test therefore fails if its process
is removed and if its formula changes. The comparison with the Fortran routine
itself is ``echam_fortran_reference_test.py``.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.clouds.echam_1m import (
    LevelInputs,
    MicrophysicsParameters,
    _es_and_derivative,
    _sweep_level,
    autoconversion,
    autoconversion_beheng,
    autoconversion_kk2000,
    cloud_microphysics_column_sweep,
    contact_freezing_radius,
    ice_autoconversion,
    ice_fall_speed,
    ice_phase_weight,
    lo2_ice_phase,
    lonacc_levels,
    temperature_switch,
)

DT = 1800.0
ZXSEC = 1.0 - 1.0e-12


@pytest.fixture(autouse=True)
def _float64():
    """Every test here runs in float64 unless it asks for float32 itself."""
    with jax.enable_x64():
        yield


# ---------------------------------------------------------------------------
# NumPy references (ECHAM formulas, float64)
# ---------------------------------------------------------------------------

def _es(t, ice):
    """``(e_s, de_s/dT)`` from the sweep's one saturation formula, float64.

    The process formulas below are independent transcriptions; the
    saturation formula itself is the one choice they share with the sweep
    (it is checked against the Fortran by the reference comparison).
    """
    e, de = _es_and_derivative(jnp.asarray(t, jnp.float64), ice)
    return float(e), float(de)


def np_dlnes(t, ice):
    e, de = _es(t, ice)
    return de / e


def np_ua(t, water_only=False):
    """ECHAM ``ua``/``dua`` (``uaw``/``duaw`` if ``water_only``)."""
    ice = (t <= c.tmelt) and not water_only
    e, de = _es(t, ice)
    return e * c.rd / c.rv, de * c.rd / c.rv


def np_qs(u, p):
    z = min(u / p, 0.5)
    return z / (1.0 - c.vtmpc1 * z)


def qs_water(t, p):
    return np_qs(np_ua(t, water_only=True)[0], p)


def qs_ice(t, p):
    return np_qs(_es(t, True)[0] * c.rd / c.rv, p)


def cp_moist(q):
    return c.cpd + (c.cpv - c.cpd) * max(q, 0.0)


def beheng(zxlb, rho, n, dt, ccraut=15.0):
    """F:976-993."""
    rate = (ccraut * 1.2e27) / rho * (n * 1e-6) ** -3.3 * (rho * 1e-3) ** 4.7
    return zxlb * (1.0 - (1.0 + rate * dt * 3.7 * zxlb ** 3.7) ** (-1.0 / 3.7))


def levkov(zxib, rho, dt, ccsaut=95.0, ceffmin=10.0, ceffmax=150.0):
    """F:996-1001, 1029-1048."""
    zrieff = min(max(83.8 * (zxib * rho * 1000.0) ** 0.216, ceffmin), ceffmax)
    zrih = math.log10(math.sqrt(5113188.0 + 2809.0 * zrieff ** 3) - 2261.0)
    zc1 = 17.5 * rho / 500.0 * (1.3 / rho) ** 0.33
    zdt2 = -6.0 / zc1 * (zrih / 3.0 - 2.0)
    return zxib * (1.0 - 1.0 / (1.0 + ccsaut / zdt2 * dt * zxib))


def sweep_out_kernel(content, rho, cn0s=3.0e6, crhosno=100.0):
    """Marshall-Palmer sweep-out rate of F:1065-1066."""
    return (math.pi * cn0s * 3.078
            * (content / (math.pi * crhosno * cn0s)) ** 0.8125
            * math.sqrt(1.3 / rho))


# ---------------------------------------------------------------------------
# One level of the sweep
# ---------------------------------------------------------------------------

def run_level(carry=(0.0, 0.0, 0.0, 0.0), config=None, dt=DT, **kw):
    """Run ``_sweep_level`` on one level; keyword arguments are LevelInputs."""
    d = dict(tm1=280.0, qm1=None, dtemp=0.0, dq=0.0, xlp=0.0, xip=0.0,
             paclc=0.0, p=70000.0, dp=5000.0, rho=None, dz=500.0,
             cdnc=8.0e7, cdnc_aut=None, pcair=None, zauloc_off=False,
             top=False, bottom=False)
    d.update(kw)
    if d["qm1"] is None:
        d["qm1"] = 0.8 * qs_water(d["tm1"], d["p"])
    if d["rho"] is None:
        d["rho"] = d["p"] / (c.rd * d["tm1"])
    if d["pcair"] is None:
        d["pcair"] = cp_moist(d["qm1"])
    if d["cdnc_aut"] is None:
        d["cdnc_aut"] = d["cdnc"]
    inputs = LevelInputs(**{k: jnp.asarray(v) for k, v in d.items()})
    cfg = config or MicrophysicsParameters.default()
    new_carry, out = _sweep_level(tuple(jnp.asarray(x, jnp.float64) for x in carry),
                                  inputs, cfg, dt)
    return ([float(x) for x in new_carry], out, d)


def f(x):
    return float(np.asarray(x))


def zmass(d, dt=DT):
    return d["dp"] / (dt * c.grav)


def run_sweep(t, q, dtemp, dq, qc, qi, cf, p, dp, rho, dz, n, dt=DT, config=None,
              **kw):
    """Run the column sweep with the condensate as anchor and no condensate increment."""
    zeros = jnp.zeros_like(qc)
    return cloud_microphysics_column_sweep(t, q, qc, qi, dtemp, dq, zeros, zeros,
                                           cf, p, dp, rho, dz, n, dt, config, **kw)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestParameters:

    def test_defaults_are_echams_t63_values(self):
        p = MicrophysicsParameters.default()
        assert f(p.cvtfall) == 2.5
        assert f(p.csecfrl) == 5.0e-6
        assert f(p.clwprat) == 4.0
        assert f(p.cthomi) == pytest.approx(c.tmelt - 35.0)
        assert f(p.ccwmin) == 1e-7 and f(p.cqtmin) == 1e-12
        assert f(p.ccraut) == 15.0 and f(p.ccsaut) == 95.0
        assert f(p.ccracl) == 6.0 and f(p.ccsacl) == 0.1
        assert f(p.cauloc) == 0.0
        assert p.autoconversion_twomey is False

    def test_resolution_leaves_are_differentiable_and_widths_static(self):
        p = MicrophysicsParameters.default()
        leaves = jax.tree_util.tree_leaves(p)
        n_static = sum(1 for fld in dataclasses.fields(p)
                       if not fld.metadata.get("pytree_node", True))
        assert n_static == 8
        assert p.defaults_truncation == 63
        assert MicrophysicsParameters.default(
            autoconversion_scheme="kk2000").autoconversion_scheme == 1
        for name in ("cvtfall", "csecfrl", "clwprat", "cthomi"):
            assert any(leaf is getattr(p, name) for leaf in leaves), name

    def test_explicit_value_wins_over_resolution_default(self):
        p = MicrophysicsParameters.default(cvtfall=3.3, csecfrl=1e-6)
        assert f(p.cvtfall) == pytest.approx(3.3)
        assert f(p.csecfrl) == pytest.approx(1e-6)
        assert f(p.clwprat) == 4.0

    def test_truncation_defaults_and_override_precedence(self):
        pytest.importorskip("jcm.physics.clouds.echam_cloud_defaults")
        from jcm.physics.physics_term import with_field_overrides
        p127 = MicrophysicsParameters.default(truncation=127)
        assert f(p127.cvtfall) == 3.0 and f(p127.csecfrl) == 1e-5
        assert p127.defaults_truncation == 127
        p106 = MicrophysicsParameters.default(truncation=106)
        assert 2.5 < f(p106.cvtfall) < 3.0 and 5e-6 < f(p106.csecfrl) < 1e-5
        over = with_field_overrides(p127, {"cvtfall": 2.0}, scheme="test")
        assert f(over.cvtfall) == 2.0
        assert f(over.csecfrl) == 1e-5
        # Both the overridden and a defaulted leaf carry a live gradient.
        g = jax.grad(lambda prm: _precip_of(prm))(over)
        assert f(g.cvtfall) != 0.0 and np.isfinite(f(g.cvtfall))
        assert np.isfinite(f(g.csecfrl))

    def test_overridden_and_defaulted_leaves_are_live(self):
        from jcm.physics.physics_term import with_field_overrides
        over = with_field_overrides(MicrophysicsParameters.default(),
                                    {"cvtfall": 2.2}, scheme="test")
        g = jax.grad(_precip_of)(over)
        assert f(g.cvtfall) != 0.0 and np.isfinite(f(g.cvtfall))
        assert f(g.ccsaut) != 0.0 and np.isfinite(f(g.ccsaut))

    def test_term_records_whether_parameters_are_defaults(self):
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        assert Echam1MMicrophysics().params_are_defaults
        mine = MicrophysicsParameters.default(ccraut=12.0)
        assert not Echam1MMicrophysics(mine).params_are_defaults
        assert Echam1MMicrophysics(mine, params_are_defaults=True).params_are_defaults

    def test_scheme_aliases(self):
        assert MicrophysicsParameters.default(
            autoconversion_scheme="kk2000").autoconversion_scheme == 1
        assert MicrophysicsParameters.default(
            autoconversion_scheme="beheng").autoconversion_scheme == 0
        with pytest.raises(ValueError):
            MicrophysicsParameters.default(autoconversion_scheme="nope")

    def test_legacy_kk2000_ccraut_override_raises(self):
        with pytest.raises(ValueError, match="ccraut_kk_threshold"):
            MicrophysicsParameters.default(autoconversion_scheme="kk2000",
                                           ccraut=1e-3)
        MicrophysicsParameters.default(autoconversion_scheme="kk2000",
                                       ccraut_kk_threshold=2e-5)
        MicrophysicsParameters.default(ccraut=20.0)

    def test_with_field_overrides_round_trip(self):
        from jcm.physics.physics_term import with_field_overrides
        base = MicrophysicsParameters.default()
        out = with_field_overrides(base, {"ccsaut": 80.0, "phase_switch_width": 2.0},
                                   scheme="test")
        assert f(out.ccsaut) == 80.0 and out.phase_switch_width == 2.0
        with pytest.raises(ValueError):
            with_field_overrides(base, {"t_mix_min": 1.0}, scheme="test")


def _precip_of(params):
    """Surface precipitation of a mixed-phase column, for gradient checks."""
    col = mixed_column()
    _, st = run_sweep(*col, DT, params)
    return jnp.sum(st.precip_rain + st.precip_snow)


def mixed_column(nlev=12, cf_value=0.6):
    """Build a contiguous deck, ice aloft and liquid below, with increments."""
    p = np.linspace(25000.0, 95000.0, nlev)
    t = np.linspace(228.0, 288.0, nlev)
    q = np.array([0.97 * qs_water(ti, pi) if ti > c.tmelt else 0.97 * qs_ice(ti, pi)
                  for ti, pi in zip(t, p)])
    qc = np.where((t > 250.0), 1.5e-4, 0.0)
    qi = np.where(t < 265.0, 4e-5, 0.0)
    cf = np.where(qc + qi > 0.0, cf_value, 0.0)
    dp = np.full(nlev, 6000.0)
    rho = p / (c.rd * t)
    dz = dp / (rho * c.grav)
    dtemp = np.full(nlev, -0.4)
    dq = np.full(nlev, 3e-5)
    n = np.full(nlev, 8e7)
    return tuple(jnp.asarray(a) for a in
                 (t, q, dtemp, dq, qc, qi, cf, p, dp, rho, dz, n))


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

class TestSaturation:

    def test_lo2_is_strict(self):
        csec, cth = 5e-6, c.tmelt - 35.0

        def lo2(t, xi):
            return bool(lo2_ice_phase(jnp.asarray(t), jnp.asarray(xi), csec, cth))

        assert lo2(cth - 0.01, 0.0) and not lo2(cth, 0.0)
        assert lo2(260.0, 6e-6) and not lo2(260.0, 5e-6)
        assert not lo2(c.tmelt, 1e-3) and lo2(c.tmelt - 0.01, 1e-3)

    def test_mixed_table_switches_to_ice_at_and_below_tmelt(self):
        from jcm.physics.clouds.echam_1m import _ua, _uaw, _ub
        t = jnp.asarray([c.tmelt - 1.0, c.tmelt, c.tmelt + 1.0])
        ua, dua = _ua(t)
        uaw, duaw = _uaw(t)
        ei = np.array([_es(float(x), True) for x in t])
        ew = np.array([_es(float(x), False) for x in t])
        np.testing.assert_allclose(np.asarray(ua), np.r_[ei[:2, 0], ew[2, 0]] * c.rd / c.rv,
                                   rtol=1e-14)
        np.testing.assert_allclose(np.asarray(uaw), ew[:, 0] * c.rd / c.rv, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(duaw), ew[:, 1] * c.rd / c.rv, rtol=1e-14)
        ub = np.asarray(_ub(t))
        np.testing.assert_allclose(ub[:2], c.alhs / c.cpd * ei[:2, 1] / ei[:2, 0], rtol=1e-14)
        np.testing.assert_allclose(ub[2], c.alhc / c.cpd * ew[2, 1] / ew[2, 0], rtol=1e-14)
        assert np.all(np.asarray(dua) > 0)


# ---------------------------------------------------------------------------
# Section 3.1: melting
# ---------------------------------------------------------------------------

class TestMelting:

    def test_incoming_snow_melts_at_step_start_temperature(self):
        zsfl = 2e-4
        # Dry enough that the melt cooling cannot saturate the level.
        carry, out, d = run_level(carry=(0.0, zsfl, 0.0, 0.0), tm1=274.0,
                                  dtemp=-5.0, qm1=0.3 * qs_water(274.0, 70000.0))
        lfdcp = (c.alhs - c.alhc) / d["pcair"]
        zcons = 1.0 / (DT * c.grav) * (d["dp"] / lfdcp)
        want = min(ZXSEC * zsfl, zcons * (274.0 - c.tmelt))
        assert f(out.intermediates.zsmlt) == pytest.approx(want / zmass(d), rel=1e-12)
        assert carry[0] == pytest.approx(want, rel=1e-12)
        assert carry[1] == pytest.approx(zsfl - want, rel=1e-12)

    def test_melt_is_capped_below_the_whole_flux(self):
        zsfl = 1e-7
        _, out, d = run_level(carry=(0.0, zsfl, 0.0, 0.0), tm1=285.0)
        assert f(out.intermediates.zsmlt) * zmass(d) == pytest.approx(ZXSEC * zsfl,
                                                                     rel=1e-12)

    def test_no_melt_at_the_top_level(self):
        _, out, _ = run_level(carry=(0.0, 1e-4, 0.0, 0.0), tm1=280.0, top=True,
                              xip=1e-5)
        assert f(out.intermediates.zsmlt) == 0.0
        assert f(out.intermediates.zimlt) == 0.0

    def test_all_cloud_ice_melts_above_tmelt(self):
        _, warm, _ = run_level(tm1=273.5, dtemp=-3.0, xip=3e-5, paclc=0.5)
        _, cold, _ = run_level(tm1=272.9, dtemp=+3.0, xip=3e-5, paclc=0.5)
        assert f(warm.intermediates.zimlt) == 3e-5
        assert f(cold.intermediates.zimlt) == 0.0

    def test_ice_melt_moves_ice_to_liquid(self):
        _, out, _ = run_level(tm1=274.0, xip=3e-5, paclc=0.5,
                              qm1=0.5 * qs_water(274.0, 70000.0))
        # zimlt leaves the ice and enters the liquid (F:1244-1247).
        inter = out.intermediates
        assert f(inter.zimlt) == 3e-5
        assert f(out.zxite) * DT == pytest.approx(
            -3e-5 + f(inter.zqsed) + f(inter.zdep) - f(inter.zspr) + f(inter.zfrl)
            + f(inter.zdxicor) * DT, abs=1e-18)


# ---------------------------------------------------------------------------
# Sections 3.2 and 3.3: sublimation and evaporation of the incoming fluxes
# ---------------------------------------------------------------------------

class TestSnowSublimation:

    def _expected(self, d, zsfl, zclcpre):
        t, p, q, rho = d["tm1"], d["p"], d["qm1"], d["rho"]
        zlsdcp = c.alhs / d["pcair"]
        zqsi = np_qs(np_ua(t)[0], p)
        zsusati = min(q / zqsi - 1.0, 0.0)
        zb1 = zlsdcp ** 2 / (2.43e-2 * c.rv * t ** 2)
        zb2 = 1.0 / (rho * zqsi * 0.211e-4)
        zcoeff = 3.0e6 * 2.0 * math.pi * (zsusati / (rho * (zb1 + zb2)))
        t1 = math.sqrt(math.sqrt(1.3 / rho))
        t2 = math.sqrt((zsfl / zclcpre / 2.5) ** (1 / 1.16) / (math.pi * 100.0 * 3e6))
        t3 = t2 ** 1.3125
        zcfac4c = 0.78 * t2 + 232.19 * t1 * t3
        zdpg = d["dp"] / c.grav
        zzeps = max(-ZXSEC * zsfl / zclcpre, zcoeff * zcfac4c * zdpg)
        zsub = -(zzeps / zdpg) * DT * zclcpre
        zsub = min(zsub, max(ZXSEC * (zqsi - q), 0.0))
        return min(max(zsub, 0.0), zsfl / zmass(d))

    def test_lin_sublimation_formula(self):
        zsfl, zclcpre = 3e-5, 0.5
        carry, out, d = run_level(carry=(0.0, zsfl, zclcpre, 0.0), tm1=262.0,
                                  qm1=0.5 * qs_ice(262.0, 70000.0))
        want = self._expected(d, zsfl, zclcpre)
        assert want > 0.0
        assert f(out.intermediates.zsub) == pytest.approx(want, rel=1e-11)
        # The sublimated snow leaves the flux in 7.3 and moistens the air.
        assert f(out.snow_sub_flux) == pytest.approx(want * zmass(d), rel=1e-11)
        assert carry[1] == pytest.approx(zsfl - want * zmass(d), rel=1e-9)

    def test_needs_a_precipitating_level_above(self):
        _, out, _ = run_level(carry=(0.0, 3e-5, 0.0, 0.0), tm1=262.0,
                              qm1=0.5 * qs_ice(262.0, 70000.0))
        assert f(out.intermediates.zsub) == 0.0

    def test_capped_by_the_ice_saturation_deficit(self):
        zsfl, zclcpre = 5e-3, 1.0
        _, out, d = run_level(carry=(0.0, zsfl, zclcpre, 0.0), tm1=262.0,
                              qm1=0.999 * qs_ice(262.0, 70000.0))
        deficit = ZXSEC * (np_qs(np_ua(262.0)[0], 70000.0) - d["qm1"])
        assert f(out.intermediates.zsub) == pytest.approx(deficit, rel=1e-11)


class TestRainEvaporation:

    def _expected(self, d, zrfl, zclcpre):
        t, p, q, rho = d["tm1"], d["p"], d["qm1"], d["rho"]
        uaw = np_ua(t, water_only=True)[0]
        zesw = min(uaw / p, 0.5)
        zqsw = zesw / (1.0 - c.vtmpc1 * zesw)
        zsusatw = min(q / zqsw - 1.0, 0.0)
        zast = c.alhc * (c.alhc / t / c.rv - 1.0) / t / 0.024
        zbst = t / (2.21 / p * (uaw / c.rd))
        zzepr = (870.0 * zsusatw * (zrfl / zclcpre) ** 0.61 * math.sqrt(1.3 / rho)
                 / math.sqrt(1.3) / (zast + zbst))
        zdpg = d["dp"] / c.grav
        zzepr = max(-ZXSEC * zrfl / zclcpre, zzepr * zdpg)
        zevp = -(zzepr / zdpg) * DT * zclcpre
        zevp = min(zevp, max(ZXSEC * (zqsw - q), 0.0))
        return min(max(zevp, 0.0), zrfl / zmass(d))

    def test_rotstayn_formula_at_step_start_state(self):
        zrfl, zclcpre = 6e-5, 0.4
        carry, out, d = run_level(carry=(zrfl, 0.0, zclcpre, 0.0), tm1=286.0,
                                  dtemp=+2.0, dq=-1e-4,
                                  qm1=0.7 * qs_water(286.0, 70000.0))
        want = self._expected(d, zrfl, zclcpre)
        assert want > 0.0
        assert f(out.intermediates.zevp) == pytest.approx(want, rel=1e-11)
        assert carry[0] == pytest.approx(zrfl - want * zmass(d), rel=1e-9)

    def test_capped_by_the_water_saturation_deficit(self):
        # Rate and deficit both scale with the subsaturation; a large flux
        # makes the rate exceed the deficit.
        _, out, d = run_level(carry=(0.5, 0.0, 1.0, 0.0), tm1=286.0,
                              qm1=0.999 * qs_water(286.0, 70000.0))
        deficit = ZXSEC * (qs_water(286.0, 70000.0) - d["qm1"])
        assert f(out.intermediates.zevp) == pytest.approx(deficit, rel=1e-10)

    def test_evaporation_cooling_drives_condensation_in_cloud(self):
        """Rain evaporation runs before section 5, and its cooling enters zdtdt.

        No upstream increments and a cloudy level: ECHAM condenses
        ``zqcdif = zlvdcp·zevp·zdqsat1·paclc`` from the evaporative cooling
        alone (F:706-730).
        """
        zrfl, zclcpre, cf, t, p = 6e-5, 0.4, 0.5, 286.0, 70000.0
        _, out, d = run_level(carry=(zrfl, 0.0, zclcpre, 0.0), tm1=t, p=p,
                              qm1=0.8 * qs_water(t, p), paclc=cf, xlp=1e-4)
        zevp = f(out.intermediates.zevp)
        zlvdcp = c.alhc / d["pcair"]
        uaw, duaw = np_ua(t, water_only=True)
        z = min(uaw / p, 0.5)
        zcor = 1.0 / (1.0 - c.vtmpc1 * z)
        zdqsdt = zcor ** 2 * duaw / p
        zdqsat1 = zdqsdt / (1.0 + cf * zlvdcp * zdqsdt)
        want = (0.0 - (-zlvdcp * zevp) * zdqsat1) * cf
        assert zevp > 0.0 and want > 0.0
        assert f(out.intermediates.zcnd) == pytest.approx(want, rel=1e-11)


# ---------------------------------------------------------------------------
# Section 4: sedimentation, lo2, in-cloud values, clear cells
# ---------------------------------------------------------------------------

class TestSedimentation:

    def _expected(self, d, xip, zxitop, cvtfall=2.5):
        rho = d["rho"]
        zxip1 = max(xip, 2.220446049250313e-16)
        v = cvtfall * (rho * zxip1) ** 0.16
        zal1 = math.exp(-v * c.grav * rho * (DT / d["dp"]))
        zal2 = zxitop / (rho * v)
        zxised = max(0.0, zxip1 * zal1 + zal2 * (1.0 - zal1))
        zqsed = zxised - zxip1
        zxibot = max(0.0, zxitop - zqsed * zmass(d))
        return (zxitop - zxibot) / zmass(d), zxibot

    @pytest.mark.parametrize("xip,zxitop", [(2e-5, 0.0), (2e-5, 1e-6),
                                            (0.0, 2e-6), (1e-6, 5e-8)])
    def test_analytic_sedimentation(self, xip, zxitop):
        carry, out, d = run_level(carry=(0.0, 0.0, 0.0, zxitop), tm1=235.0,
                                  xip=xip, paclc=0.5)
        zqsed, zxibot = self._expected(d, xip, zxitop)
        assert f(out.intermediates.zqsed) == pytest.approx(zqsed, rel=1e-10, abs=1e-24)
        assert carry[3] == pytest.approx(zxibot, rel=1e-10, abs=1e-24)

    def test_bottom_level_ice_flux_joins_the_snow(self):
        zxitop = 2e-6
        carry, out, d = run_level(carry=(0.0, 0.0, 0.0, zxitop), tm1=250.0,
                                  xip=0.0, bottom=True)
        _, zxibot = self._expected(d, 0.0, zxitop)
        assert carry[3] == 0.0
        assert carry[1] == pytest.approx(zxibot, rel=1e-10)


class TestPhaseSwitch:

    @pytest.mark.parametrize("tm1,dtemp,xip,ice", [
        (230.0, 0.0, 0.0, True),      # below cthomi
        (230.0, 20.0, 0.0, False),    # provisional T decides: 250 K, no ice
        (255.0, 0.0, 3e-5, True),     # ice memory above csecfrl
        (255.0, 0.0, 0.0, False),     # no ice: water saturation
        (272.5, -1.5, 3e-5, True),    # provisional 271 K with ice
        (275.0, -4.0, 3e-5, False),   # ptm1 > tmelt: the ice melted first
        (270.0, 5.0, 3e-5, False),    # provisional above tmelt
    ])
    def test_lo2_on_provisional_temperature_and_sedimented_ice(self, tm1, dtemp,
                                                               xip, ice):
        _, out, _ = run_level(tm1=tm1, dtemp=dtemp, xip=xip, paclc=0.5, dp=40000.0)
        assert f(out.intermediates.zlo2) == (1.0 if ice else 0.0)

    def test_lo2_selects_latent_heat_and_saturation(self):
        """In ice phase the growth is deposition with Ls and ice saturation."""
        t, p, cf, dq = 255.0, 60000.0, 0.5, 1e-4
        _, out, d = run_level(tm1=t, p=p, xip=3e-5, paclc=cf, dq=dq, dp=40000.0,
                              qm1=0.9 * qs_ice(t, p))
        zlsdcp = c.alhs / d["pcair"]
        u, du = np_ua(t)
        z = min(u / p, 0.5)
        zcor = 1.0 / (1.0 - c.vtmpc1 * z)
        zdqsdt = zcor ** 2 * du / p
        zdqsat1 = zdqsdt / (1.0 + cf * zlsdcp * zdqsdt)
        want = (dq - (cf * zlsdcp * dq) * zdqsat1) * cf
        assert f(out.intermediates.zdep) == pytest.approx(want, rel=1e-11)
        assert f(out.intermediates.zcnd) == 0.0


class TestClearCell:

    def test_clear_cell_returns_all_condensate(self):
        xlp, xip = 3e-5, 2e-5
        _, out, d = run_level(tm1=268.0, xlp=xlp, xip=xip, paclc=0.0)
        inter = out.intermediates
        assert f(inter.zxlevap) == xlp
        assert f(inter.zxievap) == pytest.approx(xip + f(inter.zqsed), rel=1e-14)
        assert f(out.zxlte) * DT == pytest.approx(-xlp, rel=1e-12)

    def test_any_positive_cover_is_cloudy(self):
        _, out, _ = run_level(tm1=285.0, xlp=3e-5, paclc=1e-20)
        assert f(out.intermediates.zxlevap) == 0.0

    def test_no_evaporation_of_partial_cloud_without_increments(self):
        """#940 headline: cf 0.3, RH 0.9, no tendencies -> no condensate change.

        ECHAM's section 5 condenses only what the step's increments bring;
        a subsaturated partial cloud keeps its condensate (F:726-734).
        """
        t, p = 285.0, 70000.0
        _, out, _ = run_level(tm1=t, p=p, qm1=0.9 * qs_water(t, p), paclc=0.3,
                              xlp=1e-4)
        assert f(out.intermediates.zcnd) == 0.0
        assert f(out.intermediates.zdep) == 0.0


# ---------------------------------------------------------------------------
# Section 5: condensation
# ---------------------------------------------------------------------------

def _zdqsat(d, cf, dq, zdtdt, ice):
    t, p = d["tm1"], d["p"]
    zlc = (c.alhs if ice else c.alhc) / d["pcair"]
    u, du = np_ua(t, water_only=not ice)
    z = min(u / p, 0.5)
    zcor = 1.0 / (1.0 - c.vtmpc1 * z)
    zdqsdt = zcor ** 2 * du / p
    zdqsat1 = zdqsdt / (1.0 + cf * zlc * zdqsdt)
    return (zdtdt + cf * zlc * dq) * zdqsat1


class TestCondensation:

    def test_growth_from_increments(self):
        t, p, cf, dq, dtemp = 285.0, 70000.0, 0.4, 2e-4, -0.5
        _, out, d = run_level(tm1=t, p=p, qm1=0.9 * qs_water(t, p), paclc=cf,
                              xlp=1e-4, dq=dq, dtemp=dtemp)
        want = (dq - _zdqsat(d, cf, dq, dtemp, ice=False)) * cf
        assert want > 0.0
        assert f(out.intermediates.zcnd) == pytest.approx(want, rel=1e-11)
        assert f(out.intermediates.zdep) == 0.0

    def test_dissipation_splits_by_in_cloud_ice_fraction(self):
        t, p, cf, dq = 262.0, 70000.0, 0.5, -1e-5
        xlp, xip = 4e-5, 1e-5
        _, out, d = run_level(tm1=t, p=p, qm1=0.9 * qs_water(t, p), paclc=cf,
                              xlp=xlp, xip=xip, dq=dq, dp=60000.0)
        inter = out.intermediates
        zxib = (xip + f(inter.zqsed)) / cf
        zxlb = xlp / cf
        ice = f(inter.zlo2) == 1.0
        zqcdif = max((dq - _zdqsat(d, cf, dq, 0.0, ice)) * cf, -(zxib + zxlb) * cf)
        zifrac = zxib / (zxib + zxlb)
        assert zqcdif < 0.0
        assert f(inter.zcnd) == pytest.approx(zqcdif * (1 - zifrac), rel=1e-10)
        assert f(inter.zdep) == pytest.approx(zqcdif * zifrac, rel=1e-10)

    def test_dissipation_bounded_by_the_condensate(self):
        t, p, cf = 285.0, 70000.0, 0.5
        _, out, _ = run_level(tm1=t, p=p, qm1=0.5 * qs_water(t, p), paclc=cf,
                              xlp=1e-6, dq=-5e-4)
        assert f(out.intermediates.zcnd) == pytest.approx(-1e-6, rel=1e-12)

    def test_growth_bounded_by_the_vapour(self):
        # A strong cooling increment lowers the saturation humidity by more
        # than the vapour present, so zqcdif = qsec*zqp1.
        t, p, cf = 285.0, 70000.0, 1.0
        q, dq = 1e-3, 1e-4
        _, out, _ = run_level(tm1=t, p=p, qm1=q, paclc=cf, dq=dq, dtemp=-40.0,
                              xlp=1e-4)
        qp1 = q + dq
        # zqcdif = min(., qsec*zqp1); 5.4 then finds the box subsaturated.
        assert f(out.intermediates.zcnd) == pytest.approx((1 - 1e-12) * qp1, rel=1e-12)


class TestWholeBoxSupersaturation:

    def _zcor(self, t, p, q, pcair, ice=False):
        u, du = np_ua(t, water_only=not ice)
        zes = min(u / p, 0.5)
        zcor = 1.0 / (1.0 - c.vtmpc1 * zes)
        qsp = zes * zcor
        zlc = (c.alhs if ice else c.alhc) / pcair
        if zes >= 0.4:
            ub = (c.alhs if (ice or t <= c.tmelt) else c.alhc) / c.cpd * np_dlnes(
                t, t <= c.tmelt)
            zlcdqsdt = qsp * zcor * ub
        else:
            zlcdqsdt = zlc * zcor ** 2 * du / p
        return max((q - qsp - 0.01 * qsp) / (1.0 + zlcdqsdt), 0.0)

    def test_clear_supersaturated_cell_condenses_the_excess_over_one_percent(self):
        t, p = 285.0, 70000.0
        q = 1.05 * qs_water(t, p)
        _, out, d = run_level(tm1=t, p=p, qm1=q, paclc=0.0)
        want = self._zcor(t, p, q, d["pcair"])
        assert want > 0.0
        assert f(out.intermediates.zcnd) == pytest.approx(want, rel=1e-11)

    def test_low_pressure_branch_uses_ub(self):
        t, p = 300.0, 1500.0
        q = 0.9
        _, out, d = run_level(tm1=t, p=p, qm1=q, paclc=0.0, rho=0.02, dp=100.0)
        want = self._zcor(t, p, q, d["pcair"])
        assert want > 0.0
        assert f(out.intermediates.zcnd) == pytest.approx(want, rel=1e-10)

    def test_clear_cell_with_new_condensate_is_cloudy_for_the_microphysics(self):
        """F:800-810: zclcaux = 1, and the new condensate can rain out."""
        t, p = 290.0, 90000.0
        _, out, _ = run_level(tm1=t, p=p, qm1=1.2 * qs_water(t, p), paclc=0.0)
        inter = out.intermediates
        assert f(inter.zcnd) > 1e-4
        assert f(inter.zclcaux) == 1.0
        assert f(inter.zrpr) > 0.0
        # The cover itself stays clear.
        assert f(out.cloud_fraction) == 0.0


# ---------------------------------------------------------------------------
# Section 6: freezing
# ---------------------------------------------------------------------------

class TestFreezing:

    def test_all_liquid_freezes_at_or_below_cthomi(self):
        cth = c.tmelt - 35.0
        for t in (cth - 5.0, cth):
            _, out, _ = run_level(tm1=t, p=40000.0, xlp=5e-5, paclc=0.5,
                                  qm1=0.5 * qs_ice(t, 40000.0))
            assert f(out.intermediates.zfrl) == pytest.approx(5e-5, rel=1e-13)

    def _bigg_contact(self, zxlb, t, rho, n, cf):
        zfrho = rho / (1000.0 * n)
        zfrl = 100.0 * (math.exp(0.66 * (c.tmelt - t)) - 1.0) * zfrho
        zfrl = zxlb * (1.0 - 1.0 / (1.0 + zfrl * DT * zxlb))
        zradl = (0.75 * zxlb * zfrho / math.pi) ** (1.0 / 3.0)
        zf1 = max(0.0, 4.0 * math.pi * zradl * n * 2.0e5 * (c.tmelt - 3.0 - t) / rho)
        zfrl = max(0.0, min(zfrl + DT * 1.4e-20 * zf1, zxlb))
        return zfrl * cf

    @pytest.mark.parametrize("t", [245.0, 258.0, 271.0])
    def test_bigg_and_contact_freezing(self, t):
        cf, xlp, n = 0.5, 2e-4, 5e7
        _, out, d = run_level(tm1=t, p=60000.0, xlp=xlp, paclc=cf, cdnc=n,
                              cdnc_aut=3e8, qm1=0.8 * qs_ice(t, 60000.0))
        want = self._bigg_contact(xlp / cf, t, d["rho"], n, cf)
        assert want > 0.0
        assert f(out.intermediates.zfrl) == pytest.approx(want, rel=1e-11)

    def test_no_freezing_above_tmelt(self):
        _, out, _ = run_level(tm1=274.0, xlp=2e-4, paclc=0.5)
        assert f(out.intermediates.zfrl) == 0.0


# ---------------------------------------------------------------------------
# Section 7: precipitation formation and fluxes
# ---------------------------------------------------------------------------

class TestWarmRain:

    def test_beheng_autoconversion_uses_its_droplet_number(self):
        t, p, cf, xlp = 288.0, 85000.0, 0.6, 3e-4
        _, out, d = run_level(tm1=t, p=p, xlp=xlp, paclc=cf, cdnc=8e7, cdnc_aut=5e7,
                              qm1=0.8 * qs_water(t, p))
        want = cf * beheng(xlp / cf, d["rho"], 5e7, DT)
        assert f(out.intermediates.zrpr) == pytest.approx(want, rel=1e-11)
        assert f(out.autoconv_rate) == pytest.approx(want / DT, rel=1e-11)

    def test_accretion_by_incoming_rain(self):
        t, p, cf, xlp, zrfl, zclcpre = 288.0, 85000.0, 0.6, 3e-4, 2e-4, 0.3
        # Saturated at the step start: the incoming rain does not evaporate.
        _, out, d = run_level(carry=(zrfl, 0.0, zclcpre, 0.0), tm1=t, p=p, xlp=xlp,
                              paclc=cf, qm1=qs_water(t, p))
        zxlb = (xlp + f(out.intermediates.zcnd)) / cf
        zraut = beheng(zxlb, d["rho"], 8e7, DT)
        zxrp1 = (zrfl / zclcpre / (12.45 * math.sqrt(1.3 / d["rho"]))) ** (8 / 9)
        zrac1 = (zxlb - zraut) * (1.0 - math.exp(-6.0 * zxrp1 * DT))
        want = cf * zraut + min(cf, zclcpre) * zrac1
        assert f(out.intermediates.zrpr) == pytest.approx(want, rel=1e-11)
        assert f(out.accretion_rate) == pytest.approx(min(cf, zclcpre) * zrac1 / DT,
                                                      rel=1e-11)

    def test_local_rain_accretion_with_cauloc(self):
        t, p, cf, xlp = 288.0, 85000.0, 0.6, 3e-4
        cfg = MicrophysicsParameters.default(cauloc=10.0)
        _, out, d = run_level(tm1=t, p=p, xlp=xlp, paclc=cf, dz=400.0, config=cfg,
                              qm1=0.8 * qs_water(t, p))
        zxlb = xlp / cf
        zraut = beheng(zxlb, d["rho"], 8e7, DT)
        zauloc = max(min(10.0 * 400.0 / 5000.0, 0.5), 0.0)
        zrac2 = (zxlb - zraut) * (1.0 - math.exp(-6.0 * zauloc * d["rho"] * zraut * DT))
        assert f(out.intermediates.zrpr) == pytest.approx(cf * (zraut + zrac2), rel=1e-11)
        _, off, _ = run_level(tm1=t, p=p, xlp=xlp, paclc=cf, dz=400.0, config=cfg,
                              qm1=0.8 * qs_water(t, p), zauloc_off=True)
        assert f(off.intermediates.zrpr) == pytest.approx(cf * zraut, rel=1e-11)

    def test_kk2000_option(self):
        t, p, cf, xlp = 288.0, 85000.0, 0.6, 3e-4
        cfg = MicrophysicsParameters.default(autoconversion_scheme="kk2000")
        _, out, d = run_level(tm1=t, p=p, xlp=xlp, paclc=cf, config=cfg,
                              qm1=0.8 * qs_water(t, p))
        zxlb = xlp / cf
        rate = 1350.0 * zxlb ** 2.47 * (8e7 * 1e-6 + 1e-12) ** -1.79
        want = cf * min(rate * DT, zxlb)
        assert f(out.intermediates.zrpr) == pytest.approx(want, rel=1e-11)


class TestColdPrecipitation:

    def test_aggregation_and_accretion_of_ice_by_snow(self):
        t, p, cf, xip, zsfl, zclcpre = 245.0, 45000.0, 0.6, 5e-5, 3e-5, 0.5
        _, out, d = run_level(carry=(0.0, zsfl, zclcpre, 0.0), tm1=t, p=p, xip=xip,
                              paclc=cf, qm1=qs_ice(t, p), dp=8000.0)
        inter = out.intermediates
        rho = d["rho"]
        zxib = (xip + f(inter.zqsed)) / cf
        zsaut = levkov(zxib, rho, DT)
        zxsp1 = (zsfl / zclcpre / 2.5) ** (1 / 1.16)
        k1 = sweep_out_kernel(zxsp1, rho)
        zcolleffi = math.exp(0.025 * (t - c.tmelt))
        zsaci1 = (zxib - zsaut) * (1.0 - math.exp(-k1 * zcolleffi * DT))
        want = cf * zsaut + min(cf, zclcpre) * zsaci1
        assert f(inter.zsub) == 0.0 and f(inter.zdep) == 0.0
        assert f(inter.zspr) == pytest.approx(want, rel=1e-10)

    def test_riming_by_incoming_snow(self):
        t, p, cf, xlp, zsfl, zclcpre = 273.65, 80000.0, 0.6, 2e-4, 5e-3, 0.5
        carry, out, d = run_level(carry=(0.0, zsfl, zclcpre, 0.0), tm1=t, p=p,
                                  xlp=xlp, paclc=cf, qm1=qs_water(t, p))
        inter = out.intermediates
        rho = d["rho"]
        snow_in = zsfl - f(inter.zsmlt) * zmass(d)
        rain_in = f(inter.zsmlt) * zmass(d)
        # The melt cooling condenses in the cloud (section 5) before riming.
        zxlb = (xlp + f(inter.zcnd)) / cf
        zraut = beheng(zxlb, rho, 8e7, DT)
        zxrp1 = (rain_in / zclcpre / (12.45 * math.sqrt(1.3 / rho))) ** (8 / 9)
        zrac1 = (zxlb - zraut) * (1.0 - math.exp(-6.0 * zxrp1 * DT))
        zxsp1 = (snow_in / zclcpre / 2.5) ** (1 / 1.16)
        k1 = sweep_out_kernel(zxsp1, rho)
        zsacl1 = (zxlb - zraut - zrac1) * (1.0 - math.exp(-k1 * 0.1 * DT))
        assert f(inter.zsacl) == pytest.approx(min(cf, zclcpre) * zsacl1, rel=1e-10)
        # Riming feeds the snow flux and heats with the fusion heat.
        assert f(out.snow_source) == pytest.approx(
            zmass(d) * (f(inter.zspr) + f(inter.zsacl)), rel=1e-12)


class TestPrecipitatingFraction:

    def test_reset_to_local_cover_when_local_production_dominates(self):
        t, p, cf, xlp = 288.0, 85000.0, 0.3, 1e-3
        carry, out, _ = run_level(carry=(1e-9, 0.0, 0.9, 0.0), tm1=t, p=p, xlp=xlp,
                                  paclc=cf, qm1=qs_water(t, p))
        assert f(out.intermediates.zclcpre) == pytest.approx(cf)
        assert carry[2] == pytest.approx(cf)

    def test_weighted_mean_when_incoming_dominates(self):
        t, p, cf, xlp, zrfl, zclcpre = 288.0, 85000.0, 0.8, 2e-5, 1e-3, 0.3
        carry, out, d = run_level(carry=(zrfl, 0.0, zclcpre, 0.0), tm1=t, p=p,
                                  xlp=xlp, paclc=cf, qm1=qs_water(t, p))
        zpredel = f(out.rain_source)
        zpretot = zrfl
        assert zpredel < zpretot
        want = max(zclcpre, (cf * zpredel + zclcpre * zpretot) / (zpredel + zpretot))
        assert carry[2] == pytest.approx(want, rel=1e-12)

    def test_no_precipitation_no_fraction(self):
        carry, _, _ = run_level(carry=(1e-14, 0.0, 0.7, 0.0), tm1=285.0)
        assert carry[2] == 0.0

    def test_bottom_level_melts_its_own_snow_at_the_updated_temperature(self):
        """F:1119-1126: the ice flux and the level's snow melt at ztp1tmp."""
        zxitop = 3e-5
        carry, out, d = run_level(carry=(0.0, 0.0, 0.0, zxitop), tm1=272.0,
                                  dtemp=3.0, bottom=True, qm1=0.5 * qs_water(272.0, 70000.0))
        inter = out.intermediates
        zlfdcp = (c.alhs - c.alhc) / d["pcair"]
        # Clear, dry level: no condensation, ztp1tmp = tm1 + dtemp - latent terms.
        ztp1tmp = (272.0 + 3.0 - c.alhc / d["pcair"] * f(inter.zxlevap)
                   - c.alhs / d["pcair"] * f(inter.zxievap))
        zxibot = TestSedimentation()._expected(d, 0.0, zxitop)[1]
        zzdrs = zmass(d) * (f(inter.zspr) + f(inter.zsacl)) + zxibot
        melt = min(ZXSEC * zzdrs, zmass(d) / zlfdcp * max(0.0, ztp1tmp - c.tmelt))
        assert melt > 0.0
        assert f(inter.zsmlt) == pytest.approx(melt / zmass(d), rel=1e-10)
        assert carry[0] == pytest.approx(melt, rel=1e-10)


# ---------------------------------------------------------------------------
# Section 8.4: condensate below ccwmin, cover write-back
# ---------------------------------------------------------------------------

class TestSmallCondensateCorrection:

    def test_liquid_below_ccwmin_returns_to_vapour_and_cover_clears(self):
        t, p, cf, xlp = 285.0, 70000.0, 0.5, 5e-8
        _, out, d = run_level(tm1=t, p=p, xlp=xlp, paclc=cf, qm1=0.9 * qs_water(t, p))
        inter = out.intermediates
        # Before the correction the liquid is xlp minus the tiny autoconversion.
        end = xlp - f(inter.zrpr)
        assert 0.0 < end < 1e-7
        assert f(inter.zdxlcor) == pytest.approx(-end / DT, rel=1e-12)
        assert f(out.zxlte) * DT == pytest.approx(-xlp, rel=1e-12)
        assert f(out.cloud_fraction) == 0.0
        # Rain production moves liquid to rain at no heat; the correction
        # evaporates the rest with Lv.
        assert f(inter.zcnd) == 0.0
        assert f(out.ztte) == pytest.approx(c.alhc / d["pcair"] * f(inter.zdxlcor),
                                            rel=1e-10)

    def test_negative_condensate_is_filled_from_vapour(self):
        _, out, _ = run_level(tm1=285.0, xlp=-2e-8, paclc=0.5)
        assert f(out.intermediates.zdxlcor) * DT == pytest.approx(2e-8, rel=1e-10)

    def test_cover_kept_while_either_phase_holds_ccwmin(self):
        _, out, _ = run_level(tm1=262.0, xlp=5e-8, xip=5e-5, paclc=0.4, dp=40000.0,
                              qm1=qs_ice(262.0, 70000.0))
        assert f(out.cloud_fraction) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# The whole column: budgets, broadcasting, top level
# ---------------------------------------------------------------------------

def random_columns(seed, nlev=20, ncols=16):
    rng = np.random.default_rng(seed)
    p = np.linspace(8000.0, 100000.0, nlev)[:, None] * np.ones((1, ncols))
    t = (np.linspace(205.0, 298.0, nlev)[:, None]
         + rng.normal(0.0, 3.0, (nlev, ncols)))
    qsat = np.vectorize(lambda ti, pi: qs_water(ti, pi) if ti > c.tmelt
                        else qs_ice(ti, pi))(t, p)
    q = qsat * rng.uniform(0.6, 1.08, (nlev, ncols))
    qc = np.where(t > 240.0, rng.uniform(0, 4e-4, (nlev, ncols)), 0.0)
    qi = np.where(t < 270.0, rng.uniform(0, 8e-5, (nlev, ncols)), 0.0)
    clear = rng.uniform(size=(nlev, ncols)) < 0.3
    cf = np.where(clear, 0.0, rng.uniform(0.02, 1.0, (nlev, ncols)))
    dp = np.full((nlev, ncols), 92000.0 / nlev)
    rho = p / (c.rd * t)
    dz = dp / (rho * c.grav)
    dtemp = rng.normal(0.0, 0.8, (nlev, ncols))
    dq = rng.normal(0.0, 5e-5, (nlev, ncols))
    n = np.full((nlev, ncols), 8e7)
    return (t, q, dtemp, dq, qc, qi, cf, p, dp, rho, dz, n)


def budgets(cols, tend, st):
    t, q, dtemp, dq, qc, qi, cf, p, dp, rho, dz, n = (np.asarray(a, np.float64)
                                                      for a in cols)
    dm = dp / c.grav
    cp = c.cpd + (c.cpv - c.cpd) * np.maximum(q, 0.0)
    dqt = np.asarray(tend.dqdt) + np.asarray(tend.dqcdt) + np.asarray(tend.dqidt)
    rain, snow = np.asarray(st.precip_rain), np.asarray(st.precip_snow)
    water = np.sum(dqt * dm, axis=0) + rain + snow
    water_gross = np.sum(np.abs(dqt) * dm, axis=0) + rain + snow
    lf = c.alhs - c.alhc
    h = cp * np.asarray(tend.dtedt) + c.alhc * np.asarray(tend.dqdt) \
        - lf * np.asarray(tend.dqidt)
    energy = np.sum(h * dm, axis=0) - lf * snow
    energy_gross = np.sum((np.abs(cp * np.asarray(tend.dtedt))
                           + c.alhs * np.abs(np.asarray(tend.dqdt))) * dm, axis=0)
    return water, water_gross, energy, energy_gross


class TestColumnBudgets:
    """Water and energy of the sweep close on every column.

    Energy: ``Σ dm·(pcair·dT/dt + Lv·dq/dt − Lf·dqi/dt) = Lf·P_snow``, the
    identity ECHAM's section 8.3 ledger satisfies with its moist ``pcair``.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_float64_round_off(self, seed):
        cols = random_columns(seed)
        tend, st = run_sweep(
            *(jnp.asarray(a) for a in cols), DT)
        water, wg, energy, eg = budgets(cols, tend, st)
        assert np.all(wg > 0) and np.all(eg > 0)
        np.testing.assert_array_less(np.abs(water), 1e-13 * wg + 1e-20)
        np.testing.assert_array_less(np.abs(energy), 1e-12 * eg + 1e-12)

    def test_float32_tolerance(self):
        cols = random_columns(3)
        with jax.enable_x64(False):
            tend, st = run_sweep(
                *(jnp.asarray(a, jnp.float32) for a in cols), DT)
            assert tend.dtedt.dtype == jnp.float32
        water, wg, energy, eg = budgets(cols, tend, st)
        # float32 round-off through a 20-level scan: measured below 3e-6 of the
        # gross exchange for water and energy.
        np.testing.assert_array_less(np.abs(water), 2e-5 * wg)
        np.testing.assert_array_less(np.abs(energy), 2e-5 * eg)

    def test_float32_follows_float64(self):
        cols = random_columns(4)
        t64, _ = run_sweep(*(jnp.asarray(a) for a in cols), DT)
        with jax.enable_x64(False):
            t32, _ = run_sweep(
                *(jnp.asarray(a, jnp.float32) for a in cols), DT)
        scale = np.max(np.abs(np.asarray(t64.dtedt)))
        # Where a phase switch or the clear-cell criterion sits within float32
        # round-off of its threshold the two precisions may take different
        # branches; everywhere else they agree to float32 precision.
        diff = np.abs(np.asarray(t32.dtedt, np.float64) - np.asarray(t64.dtedt))
        assert np.quantile(diff, 0.98) < 1e-4 * scale


class TestBroadcasting:

    def test_column_and_block_agree(self):
        cols = [jnp.asarray(a) for a in random_columns(5, ncols=4)]
        tb, sb = run_sweep(*cols, DT)
        for j in range(4):
            tc, sc = run_sweep(*(a[:, j] for a in cols), DT)
            for got, want in zip(jax.tree.leaves((tb, sb)), jax.tree.leaves((tc, sc))):
                got = np.asarray(got)
                want = np.asarray(want)
                # XLA vectorises the block differently: round-off of the
                # field's own scale (fluxes that cancel to ~1e-20 carry it).
                scale = float(np.max(np.abs(want))) if want.size else 0.0
                np.testing.assert_allclose(got[..., j] if got.ndim else got, want,
                                           rtol=1e-13, atol=1e-13 * scale)

    def test_three_dimensional_grid(self):
        cols = [jnp.asarray(a) for a in random_columns(6, ncols=6)]
        grid = [a.reshape(a.shape[0], 2, 3) for a in cols]
        tb, _ = run_sweep(*cols, DT)
        tg, _ = run_sweep(*grid, DT)
        np.testing.assert_array_equal(np.asarray(tg.dtedt).reshape(tb.dtedt.shape),
                                      np.asarray(tb.dtedt))

    def test_bottom_flux_is_the_surface_precipitation(self):
        cols = [jnp.asarray(a) for a in random_columns(7)]
        _, st = run_sweep(*cols, DT)
        np.testing.assert_array_equal(np.asarray(st.rain_flux[-1]),
                                      np.asarray(st.precip_rain))
        np.testing.assert_array_equal(np.asarray(st.snow_flux[-1]),
                                      np.asarray(st.precip_snow))


# ---------------------------------------------------------------------------
# Surrogate derivatives
# ---------------------------------------------------------------------------

from jcm.physics.clouds.echam_1m import (  # noqa: E402
    contact_radius_pair, ice_fall_speed_pair, ice_phase_pair,
    temperature_switch_pair,
)
from jcm.physics.surrogate_gradient import with_surrogate_gradient  # noqa: E402
from jcm.testing import check_gradients, check_surrogate_gradient  # noqa: E402


class TestSurrogates:
    """Every surrogate of the sweep.

    The value is exact, the derivative is the surrogate's, the surrogate is
    smooth, and it is close to the reference.
    """

    def test_temperature_switch(self):
        width = 1.0
        exact, sur = temperature_switch_pair(width)
        wrapped = with_surrogate_gradient(exact, sur)
        d = jnp.linspace(-3.0, 3.0, 13) + 0.05
        check_surrogate_gradient(wrapped, exact, sur, (d,))
        check_gradients(sur, (d,), rtol=1e-6)
        grid = jnp.linspace(-50.0, 50.0, 2001)
        far = jnp.abs(grid) >= 5.0 * width
        dist = np.abs(np.asarray(exact(grid) - sur(grid)))
        assert np.max(dist[np.asarray(far)]) <= float(jax.nn.sigmoid(-5.0)) + 1e-15
        assert np.max(dist) <= 0.5 + 1e-15
        # Width zero: the reference step and its zero derivative.
        assert f(jax.grad(lambda x: temperature_switch(x, 0.0))(0.3)) == 0.0
        assert f(jax.grad(lambda x: temperature_switch(x, width))(0.3)) > 0.0

    def test_ice_phase(self):
        exact, sur = ice_phase_pair(1.0, 0.1)
        wrapped = with_surrogate_gradient(exact, sur)
        t = jnp.array([230.0, 237.9, 255.0, 255.0, 272.5, 280.0])
        xi = jnp.array([0.0, 1e-6, 4.9e-6, 6e-6, 1e-4, 1e-4])
        csec = jnp.full(6, 5e-6)
        cth = jnp.full(6, c.tmelt - 35.0)
        check_surrogate_gradient(wrapped, exact, sur, (t, xi, csec, cth))
        check_gradients(sur, (t, xi, csec, cth), rtol=1e-4)
        # Away from every threshold (5 widths in T, 5 widths in ice) the
        # surrogate is within 2·sigmoid(-5) of the switch.
        tt, xx = np.meshgrid(np.linspace(200.0, 300.0, 401),
                             np.concatenate([[0.0], np.geomspace(1e-9, 1e-3, 60)]))
        tt, xx = jnp.asarray(tt), jnp.asarray(xx)
        cs, ct = jnp.full_like(tt, 5e-6), jnp.full_like(tt, c.tmelt - 35.0)
        far = ((jnp.abs(tt - (c.tmelt - 35.0)) >= 5.0) & (jnp.abs(tt - c.tmelt) >= 5.0)
               & (jnp.abs(xx - 5e-6) >= 5 * 0.1 * 5e-6))
        dist = np.abs(np.asarray(exact(tt, xx, cs, ct) - sur(tt, xx, cs, ct)))
        assert np.max(dist[np.asarray(far)]) <= 2 * float(jax.nn.sigmoid(-5.0))

    def test_ice_fall_speed(self):
        cutoff = 1e-7
        exact, sur = ice_fall_speed_pair(cutoff)
        wrapped = with_surrogate_gradient(exact, sur)
        y = jnp.array([0.0, 1e-30, 3e-9, 5e-8, 2e-7, 1e-5, 1e-3])
        floor = jnp.full_like(y, 0.6 * 2.220446049250313e-16)
        check_surrogate_gradient(wrapped, exact, sur, (y, floor))
        check_gradients(sur, (y + 1e-9, floor), rtol=1e-5)
        # Distance: zero above the cutoff, below cutoff**0.16 under it, and
        # largest just below the cutoff.
        grid = jnp.concatenate([jnp.zeros(1), jnp.geomspace(1e-20, 1e-2, 400)])
        fl = jnp.full_like(grid, 0.6 * 2.220446049250313e-16)
        dist = np.abs(np.asarray(exact(grid, fl) - sur(grid, fl)))
        above = np.asarray(grid) >= cutoff
        assert np.all(dist[above] == 0.0)
        assert np.max(dist) < cutoff ** 0.16
        # The slope is bounded by (2 - a)·cutoff**(a - 1) everywhere.
        slope = jax.vmap(jax.grad(lambda x: wrapped(x, 1e-16)))(grid)
        assert np.all(np.isfinite(np.asarray(slope)))
        assert np.max(np.asarray(slope)) <= (2 - 0.16) * cutoff ** (-0.84) * (1 + 1e-12)
        # No ice and a trace of ice have nearly the same slope.
        g0 = f(jax.grad(lambda x: wrapped(x, 1e-16))(0.0))
        g1 = f(jax.grad(lambda x: wrapped(x, 1e-16))(1e-30))
        assert g0 == pytest.approx(g1, rel=1e-12) and g0 > 0.0
        # The floor is a guard, not a parameter: no derivative.
        assert f(jax.grad(lambda fl_: wrapped(1e-6, fl_))(1e-16)) == 0.0

    def test_contact_radius(self):
        rc = 1e-7
        exact, sur = contact_radius_pair(rc)
        wrapped = with_surrogate_gradient(exact, sur)
        v = jnp.array([0.0, 1e-24, 1e-21, 1e-18, 1e-15])
        check_surrogate_gradient(wrapped, exact, sur, (v,))
        # Smooth on each side of the cutoff volume 4π/3·rc³ (C1 across it).
        check_gradients(sur, (jnp.array([1e-23, 3e-22, 1e-21]),), rtol=1e-5)
        check_gradients(sur, (jnp.array([1e-19, 1e-18, 5e-18]),), rtol=1e-5)
        grid = jnp.concatenate([jnp.zeros(1), jnp.geomspace(1e-30, 1e-12, 300)])
        dist = np.abs(np.asarray(exact(grid) - sur(grid)))
        assert np.max(dist) < rc

    def test_kk2000_gate(self):
        cfg = MicrophysicsParameters.default(autoconversion_scheme="kk2000")
        qc = jnp.array([2e-5, 9e-6, 1.1e-5, 1e-3])
        rate = lambda thr: autoconversion_kk2000(  # noqa: E731
            qc, jnp.ones_like(qc), jnp.ones_like(qc), jnp.full_like(qc, 1e8),
            DT, cfg.replace(ccraut_kk_threshold=thr))
        # The value is the hard gate; the threshold keeps a derivative.
        np.testing.assert_array_equal(np.asarray(rate(1e-5)) > 0.0,
                                      np.asarray(qc) > 1e-5)
        assert f(jax.grad(lambda thr: jnp.sum(rate(thr)))(1e-5)) < 0.0

    def test_wrappers_select_the_reference_derivative_at_width_zero(self):
        t = jnp.asarray(255.0)
        g_on = jax.grad(lambda x: ice_phase_weight(x, 6e-6, 5e-6, 238.15, 1.0, 0.1))(t)
        g_off = jax.grad(lambda x: ice_phase_weight(x, 6e-6, 5e-6, 238.15, 0.0, 0.1))(t)
        assert f(g_on) < 0.0 and f(g_off) == 0.0
        # Fall speed: ECHAM's value with the EPSILON(1._wp) floor at no ice.
        v0 = f(ice_fall_speed(0.6, 0.0, 2.5, 1e-7))
        assert v0 == pytest.approx(2.5 * (0.6 * 2.220446049250313e-16) ** 0.16,
                                   rel=1e-14)
        assert f(ice_fall_speed(0.6, 2e-5, 2.5, 1e-7)) == pytest.approx(
            2.5 * (0.6 * 2e-5) ** 0.16, rel=1e-14)
        slope = jax.grad(lambda xi: ice_fall_speed(0.6, xi, 2.5, 1e-7))
        slope_ref = jax.grad(lambda xi: ice_fall_speed(0.6, xi, 2.5, 0.0))
        assert f(slope(0.0)) > 0.0 and f(slope_ref(0.0)) == 0.0
        r = contact_freezing_radius(jnp.asarray(4.0 / 3.0 * math.pi * 1e-18), 1e-7)
        assert f(r) == pytest.approx(1e-6, rel=1e-12)
        assert np.isfinite(f(jax.grad(lambda v: contact_freezing_radius(v, 1e-7))(0.0)))

    def test_sweep_value_does_not_depend_on_any_width(self):
        cols = [jnp.asarray(a) for a in random_columns(8)]
        ref, rs = run_sweep(*cols, DT)
        zero = MicrophysicsParameters.default(
            phase_switch_width=0.0, ice_fall_speed_gradient_cutoff=0.0,
            contact_radius_cutoff=0.0)
        other = MicrophysicsParameters.default(
            phase_switch_width=3.0, phase_switch_ice_width=0.5,
            ice_fall_speed_gradient_cutoff=1e-9, contact_radius_cutoff=1e-6)
        for cfg in (zero, other):
            got, gs = run_sweep(*cols, DT, cfg)
            for a, b in zip(jax.tree.leaves((got, gs)), jax.tree.leaves((ref, rs))):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# Gradients through the sweep
# ---------------------------------------------------------------------------

def _outputs(tend, st):
    return (tend.dtedt, tend.dqdt, tend.dqcdt, tend.dqidt,
            st.rain_flux, st.snow_flux, st.precip_rain, st.precip_snow)


class TestSweepGradients:

    def _args(self):
        t, q, dtemp, dq, qc, qi, cf, p, dp, rho, dz, n = mixed_column()
        return (t, q, dtemp, dq, qc, qi), (cf, p, dp, rho, dz, n)

    def test_reference_derivative_matches_a_difference_off_every_switch(self):
        """Widths zero: the derivative of the value itself, in float64.

        The deck has condensate and cover in every level and sits away from
        every threshold, so the value is differentiable there.
        """
        (t, q, dtemp, dq, qc, qi), (cf, p, dp, rho, dz, n) = self._args()
        qc = qc + 2e-5
        qi = qi + 3e-6
        cf = jnp.maximum(cf, 0.2) * jnp.linspace(0.8, 1.1, cf.shape[0])
        cfg = MicrophysicsParameters.default(
            phase_switch_width=0.0, ice_fall_speed_gradient_cutoff=0.0,
            contact_radius_cutoff=0.0)

        def fn(t_, q_, dtemp_, dq_, qc_, qi_):
            return _outputs(*run_sweep(
                t_, q_, dtemp_, dq_, qc_, qi_, cf, p, dp, rho, dz, n, DT, cfg))

        check_gradients(fn, (t, q, dtemp, dq, qc, qi), rtol=1e-4)

    @pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="a clear layer with exactly zero condensate sits on ECHAM's "
               "switches at once: the clear-cell criterion paclc > 0 (F:621) and "
               "the kink max(0, zxlp1) of the clear-cell evaporation at zero "
               "condensate (F:669-670), so no central difference converges. The "
               "same column holding condensate in its clear layers, or with no "
               "clear layer, matches a difference (test above). (#843)")
    def test_clear_layers_without_condensate_have_no_two_sided_derivative(self):
        """Record the zero-condensate clear layer as a defect, not a tolerance."""
        (t, q, dtemp, dq, qc, qi), (cf, p, dp, rho, dz, n) = self._args()
        cf = cf * jnp.linspace(0.8, 1.1, cf.shape[0])
        cfg = MicrophysicsParameters.default(
            phase_switch_width=0.0, ice_fall_speed_gradient_cutoff=0.0,
            contact_radius_cutoff=0.0)

        def fn(t_, q_, dtemp_, dq_, qc_, qi_, cf_):
            return _outputs(*run_sweep(
                t_, q_, dtemp_, dq_, qc_, qi_, cf_, p, dp, rho, dz, n, DT, cfg))

        check_gradients(fn, (t, q, dtemp, dq, qc, qi, cf), rtol=1e-4)

    def test_surrogate_derivative_is_finite_and_adjoint(self):
        """Check that jvp and vjp are adjoint and every input is live.

        The covers are untied level to level: where the cover and the carried
        precipitating fraction are exactly equal, ``min``/``max`` sit on a tie
        and the two AD modes may take different one-sided derivatives (a
        measure-zero point; with every cover exactly 0.6 the modes differ by
        5 %, untied by 1e-19).
        """
        (t, q, dtemp, dq, qc, qi), (cf, p, dp, rho, dz, n) = self._args()
        cf = cf * jnp.linspace(0.8, 1.1, cf.shape[0])

        # The surface snow of this deck is the 1e-12 remainder that ECHAM's
        # melt cap (zxsec = 1 - 1e-12, F:434) leaves above the warm levels:
        # round-off, whose AD projections are round-off too once normalised
        # by its own RMS. It is left out of the projection.
        def fn(t_, q_, dtemp_, dq_, qc_, qi_, cf_):
            return _outputs(*run_sweep(
                t_, q_, dtemp_, dq_, qc_, qi_, cf_, p, dp, rho, dz, n, DT))[:-1]

        check_gradients(fn, (t, q, dtemp, dq, qc, qi, cf), reference="adjoint",
                        live_inputs=["[0]", "[1]", "[2]", "[3]", "[4]", "[5]"])

    def test_ice_free_layers_under_an_ice_flux(self):
        """Cirrus above ice-free layers: gradients finite through the scan."""
        nlev = 10
        p = jnp.linspace(20000.0, 60000.0, nlev)
        t = jnp.linspace(215.0, 250.0, nlev)
        q = jnp.asarray([0.5 * qs_ice(float(a), float(b)) for a, b in zip(t, p)])
        qi = jnp.zeros(nlev).at[:2].set(8e-5)
        cf = jnp.zeros(nlev).at[:2].set(0.5)
        dp = jnp.full(nlev, 4000.0)
        rho = p / (c.rd * t)
        dz = dp / (rho * c.grav)
        zeros = jnp.zeros(nlev)
        n = jnp.full(nlev, 5e7)

        def surface_snow(qi_, t_):
            _, st = run_sweep(
                t_, q, zeros, zeros, zeros, qi_, cf, p, dp, rho, dz, n, DT)
            return jnp.sum(st.snow_flux) + jnp.sum(st.intermediates.zqsed)

        g_qi, g_t = jax.grad(surface_snow, argnums=(0, 1))(qi, t)
        assert np.all(np.isfinite(np.asarray(g_qi)))
        assert np.all(np.isfinite(np.asarray(g_t)))
        assert np.any(np.asarray(g_qi)[2:] != 0.0)
        # The fall-speed cutoff leaves the value bit for bit unchanged.
        cfg0 = MicrophysicsParameters.default(ice_fall_speed_gradient_cutoff=0.0)
        a = run_sweep(t, q, zeros, zeros, zeros, qi, cf, p, dp,
                                            rho, dz, n, DT)
        b = run_sweep(t, q, zeros, zeros, zeros, qi, cf, p, dp,
                                            rho, dz, n, DT, cfg0)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_parameter_gradients_live(self):
        base = MicrophysicsParameters.default()
        g = jax.grad(_precip_of)(base)
        for name in ("ccraut", "ccracl", "ccsaut", "cvtfall", "ccsacl"):
            val = f(getattr(g, name))
            assert np.isfinite(val) and val != 0.0, name
        for leaf in jax.tree.leaves(g):
            assert np.all(np.isfinite(np.asarray(leaf)))


# ---------------------------------------------------------------------------
# Rate helpers
# ---------------------------------------------------------------------------

class TestRateHelpers:

    def test_beheng_grid_mean_rate(self):
        cfg = MicrophysicsParameters.default()
        qc, cf, rho, n = 3e-4, 0.6, 1.1, 8e7
        got = f(autoconversion_beheng(jnp.asarray(qc), jnp.asarray(cf), jnp.asarray(rho),
                                      jnp.asarray(n / rho), DT, cfg))
        assert got == pytest.approx(cf * beheng(qc / cf, rho, n, DT) / DT, rel=1e-12)
        assert f(autoconversion(jnp.asarray(qc), jnp.asarray(cf), jnp.asarray(rho),
                                jnp.asarray(n / rho), DT, cfg)) == pytest.approx(got)

    def test_levkov_grid_mean_rate(self):
        cfg = MicrophysicsParameters.default()
        qi, cf, rho = 5e-5, 0.5, 0.5
        got = f(ice_autoconversion(jnp.asarray(qi), 240.0, jnp.asarray(cf), DT, cfg,
                                   air_density=jnp.asarray(rho)))
        assert got == pytest.approx(cf * levkov(qi / cf, rho, DT) / DT, rel=1e-12)
        assert f(ice_autoconversion(jnp.asarray(0.0), 240.0, jnp.asarray(cf), DT, cfg,
                                    air_density=jnp.asarray(rho))) == 0.0

    def test_lonacc_levels(self):
        omega = jnp.array([[0.1, 0.1], [0.1, -0.1], [0.1, 0.1], [0.1, 0.1]])
        mask = lonacc_levels(jnp.array([1, 1]), omega, 1, 2, 4)
        np.testing.assert_array_equal(np.asarray(mask),
                                      [[False, False], [True, False],
                                       [True, True], [False, False]])
        none = lonacc_levels(jnp.array([3, 0]), omega, 1, 2, 4)
        assert not np.any(np.asarray(none))


# ---------------------------------------------------------------------------
# The composable term
# ---------------------------------------------------------------------------

def _term_inputs(nlev=6, ncols=2, t_col=None, q_scale=1.0, qc_level=3, qc=3e-4, cf=0.6,
                 cdnc_factor=(1.0, 1.4), fmask=(0.0, 0.9), tendency_run=None,
                 dt=1200.0):
    from types import SimpleNamespace
    from jcm.physics.clouds.cloud_data import CloudData
    from jcm.physics.aerosol.aerosol_types import AerosolData
    from jcm.physics_interface import PhysicsState

    shape = (nlev, ncols)
    p_col = np.linspace(60000.0, 95000.0, nlev)
    t_col = np.linspace(270.0, 290.0, nlev) if t_col is None else np.asarray(t_col)
    q_col = q_scale * np.array([qs_water(a, b) for a, b in zip(t_col, p_col)])
    pressure = jnp.asarray(p_col[:, None] * np.ones((1, ncols)))
    temperature = jnp.asarray(t_col[:, None] * np.ones((1, ncols)))
    humidity = jnp.asarray(q_col[:, None] * np.ones((1, ncols)))
    qcf = jnp.zeros(shape).at[qc_level].set(qc)
    cover = jnp.where(qcf > 0.0, cf, 0.0)
    dp = jnp.full(shape, 5000.0)
    rho = pressure / (c.rd * temperature)
    state = PhysicsState.zeros(shape, temperature=temperature,
                               specific_humidity=humidity,
                               tracers={"qc": qcf, "qi": jnp.zeros(shape)})
    diagnostics = {
        "_dt_seconds": dt,
        "pressure_full": pressure,
        "pressure_thickness": dp,
        "air_density": rho,
        "layer_thickness": dp / (c.grav * rho),
        "clouds": CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=cover, qc=qcf, qi=jnp.zeros(shape)),
        "aerosol": AerosolData.zeros((ncols,), nlev).copy(
            cdnc_factor=jnp.asarray(cdnc_factor)),
    }
    if tendency_run is not None:
        diagnostics["_tendency_run"] = tendency_run
    terrain = SimpleNamespace(fmask=jnp.asarray(fmask))
    forcing = SimpleNamespace(glacier_fraction=None)
    return state, diagnostics, forcing, terrain


class TestTerm:

    def test_radiative_cooling_condenses_in_the_same_step(self):
        """An upstream cooling tendency condenses in the cloud by zqcdif.

        A saturated cloudy level with a radiative cooling tendency in
        ``_tendency_run`` and nothing else: ECHAM condenses
        ``zqcdif = −zdqsat·paclc`` in the same step (F:706-734). The input
        matters because the running ``thermo_run`` view holds only the terms
        that advance it (vertical diffusion, the surface, convection), not
        radiation, so the cooling would reach the cloud scheme only through
        the next step's state.
        """
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        from jcm.physics.clouds.cloud_utils import moist_isobaric_heat_capacity

        nlev, ncols, k, cf, dt = 6, 2, 3, 0.6, 1200.0
        cooling = -20.0 / 86400.0
        zeros = jnp.zeros((nlev, ncols))
        run = {"temperature": zeros.at[k].set(cooling), "specific_humidity": zeros,
               "tracers": {"qc": zeros, "qi": zeros}}
        state, diag, forcing, terrain = _term_inputs(tendency_run=run, cf=cf, dt=dt)
        tend, _ = Echam1MMicrophysics()(state, diag, forcing, terrain)
        t = float(state.temperature[k, 0])
        p = float(diag["pressure_full"][k, 0])
        pcair = float(moist_isobaric_heat_capacity(state.specific_humidity[k, 0]))
        zlvdcp = c.alhc / pcair
        uaw, duaw = np_ua(t, water_only=True)
        z = min(uaw / p, 0.5)
        zcor = 1.0 / (1.0 - c.vtmpc1 * z)
        zdqsdt = zcor ** 2 * duaw / p
        zqcdif = -(cooling * dt) * zdqsdt / (1.0 + cf * zlvdcp * zdqsdt) * cf
        cond = -float(tend.specific_humidity[k, 0]) * dt
        assert zqcdif > 0.0
        assert cond == pytest.approx(zqcdif, rel=1e-9)
        # Without the upstream tendency nothing condenses at this level (up to
        # the round-off of ECHAM's 1e-20 in-cloud floor, F:911-912).
        state, diag, forcing, terrain = _term_inputs(cf=cf, dt=dt)
        tend0, _ = Echam1MMicrophysics()(state, diag, forcing, terrain)
        assert abs(float(tend0.specific_humidity[k, 0])) * dt < 1e-17

    def test_twomey_factor_reaches_the_radiation_number_not_the_autoconversion(self):
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        from jcm.physics.clouds.cloud_utils import prescribed_droplet_number

        state, diag, forcing, terrain = _term_inputs(cdnc_factor=(1.4, 1.4))
        state1, diag1, _, _ = _term_inputs(cdnc_factor=(1.0, 1.0))
        _, out = Echam1MMicrophysics()(state, diag, forcing, terrain)
        _, out1 = Echam1MMicrophysics()(state1, diag1, forcing, terrain)
        np.testing.assert_allclose(
            np.asarray(out["clouds"].droplet_number),
            np.asarray(prescribed_droplet_number(diag["pressure_full"], terrain,
                                                 forcing, jnp.asarray([1.4, 1.4]))))
        np.testing.assert_array_equal(np.asarray(out["autoconv"]),
                                      np.asarray(out1["autoconv"]))
        on = Echam1MMicrophysics(MicrophysicsParameters.default(autoconversion_twomey=True))
        _, out_on = on(state, diag, forcing, terrain)
        assert np.all(np.asarray(out_on["autoconv"]) < np.asarray(out["autoconv"]))

    def test_carried_radius_is_left_untouched(self):
        """The radiation owns ``clouds.r_eff_*`` (#929); the 1M leaves them."""
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        state, diag, forcing, terrain = _term_inputs()
        diag["clouds"] = diag["clouds"].copy(r_eff_liq=jnp.full((6, 2), 7.5),
                                             r_eff_ice=jnp.full((6, 2), 7.5))
        _, out = Echam1MMicrophysics()(state, diag, forcing, terrain)
        np.testing.assert_array_equal(np.asarray(out["clouds"].r_eff_liq), 7.5)
        np.testing.assert_array_equal(np.asarray(out["clouds"].r_eff_ice), 7.5)

    def test_cover_write_back(self):
        """F:1280: a cell left below ccwmin in both phases loses its cover."""
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        state, diag, forcing, terrain = _term_inputs(qc=5e-8, q_scale=0.9)
        _, out = Echam1MMicrophysics()(state, diag, forcing, terrain)
        assert np.all(np.asarray(out["clouds"].cloud_fraction) == 0.0)
        state, diag, forcing, terrain = _term_inputs(qc=1e-3, q_scale=0.9)
        _, out = Echam1MMicrophysics()(state, diag, forcing, terrain)
        assert np.all(np.asarray(out["clouds"].cloud_fraction)[3] == pytest.approx(0.6))

    def test_heat_capacity_is_the_anchor_humidity(self):
        """Check ``pcair`` is built from ``pqm1`` (ECHAM physc.f90).

        A clear cell holding liquid cools by exactly ``alv·qc/pcair(qm1)``.
        """
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        from jcm.physics.clouds.cloud_utils import moist_isobaric_heat_capacity
        nlev, ncols, k = 6, 2, 3
        zeros = jnp.zeros((nlev, ncols))
        # A moistening upstream increment must not change the heat capacity.
        run = {"temperature": zeros, "specific_humidity": zeros.at[k].set(-2e-6),
               "tracers": {"qc": zeros, "qi": zeros}}
        state, diag, forcing, terrain = _term_inputs(q_scale=0.7, cf=0.0,
                                                     tendency_run=run)
        diag["clouds"] = diag["clouds"].copy(cloud_fraction=zeros)
        tend, _ = Echam1MMicrophysics()(state, diag, forcing, terrain)
        want = -c.alhc * 3e-4 / float(moist_isobaric_heat_capacity(
            state.specific_humidity[k, 0])) / 1200.0
        assert float(tend.temperature[k, 0]) == pytest.approx(want, rel=1e-12)

    def test_tendency_run_condensate_is_part_of_the_provisional_condensate(self):
        """Check the condensate increments of vdiff and convection reach the scheme."""
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        nlev, ncols, k = 6, 2, 3
        zeros = jnp.zeros((nlev, ncols))
        run = {"temperature": zeros, "specific_humidity": zeros,
               "tracers": {"qc": zeros.at[k].set(1e-7), "qi": zeros}}
        state, diag, forcing, terrain = _term_inputs(q_scale=0.7, tendency_run=run)
        diag["clouds"] = diag["clouds"].copy(cloud_fraction=zeros)
        tend, _ = Echam1MMicrophysics()(state, diag, forcing, terrain)
        # Clear cell: anchor plus increment returns to vapour.
        assert float(tend.tracers["qc"][k, 0]) * 1200.0 == pytest.approx(
            -(3e-4 + 1e-7 * 1200.0), rel=1e-12)


class TestShallowLiquidConvectionType:
    """ECHAM ``mo_cloud.f90``'s radiation ``ktype = 4`` re-typing (F:1439-1455)."""

    NLEV = 6

    def _columns(self):
        from jcm.physics.clouds.echam_1m import shallow_liquid_convection_type

        nlev = self.NLEV
        p = jnp.linspace(2e4, 1e5, nlev)[:, None] * jnp.ones((1, 6))
        dp = jnp.full((nlev, 6), 1.0e4)
        qc = jnp.zeros((nlev, 6))
        qc = qc.at[4, 0].set(1e-4)
        qc = qc.at[1, 1].set(1e-4).at[4, 1].set(3e-4)
        qc = qc.at[1, 2].set(1e-4).at[4, 2].set(5e-4)
        qc = qc.at[4, 3].set(1e-4)
        qc = qc.at[4, 4].set(1e-4)
        ktype = jnp.array([2, 2, 2, 1, 0, 2], dtype=jnp.int32)
        top = jnp.full((6,), 3, dtype=jnp.int32)
        return shallow_liquid_convection_type, ktype, top, p, qc, dp

    def test_retypes_exactly_echam_cases(self):
        fn, ktype, top, p, qc, dp = self._columns()
        out = fn(ktype, top, p, qc, dp, 4.0)
        np.testing.assert_array_equal(np.asarray(out), [4, 2, 4, 1, 0, 2])
        assert out.dtype == ktype.dtype
        out0 = fn(ktype, top, p, qc, dp, 0.0)
        np.testing.assert_array_equal(np.asarray(out0), [4, 4, 4, 1, 0, 2])

    def test_negative_ringing_does_not_retype_a_dry_column(self):
        fn, ktype, top, p, _, dp = self._columns()
        qc = jnp.zeros_like(p).at[1].set(-1e-7)
        np.testing.assert_array_equal(np.asarray(fn(ktype, top, p, qc, dp, 4.0)),
                                      np.asarray(ktype))
        qc = qc.at[4].set(1e-4)
        np.testing.assert_array_equal(np.asarray(fn(ktype, top, p, qc, dp, 4.0)),
                                      [4, 4, 4, 1, 0, 4])

    def test_independent_of_level_orientation(self):
        fn, ktype, top, p, qc, dp = self._columns()
        out = fn(ktype, top, p, qc, dp, 4.0)
        flipped = fn(ktype, self.NLEV - 1 - top, p[::-1], qc[::-1], dp[::-1], 4.0)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(flipped))

    def test_term_amends_the_convection_carry_and_only_it(self):
        from types import SimpleNamespace
        from jcm.physics.clouds.echam_1m import Echam1MMicrophysics
        from jcm.physics.convection.tiedtke_nordeng.types import ConvectionData
        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics.aerosol.aerosol_types import AerosolData
        from jcm.physics_interface import PhysicsState

        _, ktype, top, p, qc, dp = self._columns()
        nlev, ncols = qc.shape
        t = jnp.full((nlev, ncols), 285.0)
        state = PhysicsState.zeros((nlev, ncols), temperature=t,
                                   specific_humidity=jnp.full((nlev, ncols), 1e-3),
                                   tracers={"qc": qc, "qi": jnp.zeros_like(qc)})
        diagnostics = {
            "_dt_seconds": 600.0, "pressure_full": p, "pressure_thickness": dp,
            "air_density": p / (c.rd * t), "layer_thickness": jnp.full((nlev, ncols), 500.0),
            "clouds": CloudData.zeros((ncols,), nlev).copy(
                qc=qc, qi=jnp.zeros_like(qc), cloud_fraction=jnp.where(qc > 0, 0.5, 0.0)),
            "aerosol": AerosolData.zeros((ncols,), nlev),
        }
        term = Echam1MMicrophysics()
        _, out = term(state, diagnostics, SimpleNamespace(glacier_fraction=None), None)
        assert "convection" not in out
        diagnostics["convection"] = ConvectionData.zeros((ncols,), nlev).replace(
            ktype=ktype, cloud_top=top)
        _, out = term(state, diagnostics, SimpleNamespace(glacier_fraction=None), None)
        np.testing.assert_array_equal(np.asarray(out["convection"].ktype),
                                      [4, 2, 4, 1, 0, 2])
