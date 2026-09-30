"""Ice sources and number bookkeeping of the Lohmann 2M scheme (#941).

The rules ECHAM6.3-HAM2.3 applies to convectively detrained condensate and
to the ice crystal number in ``mo_cloud_micro_2m.f90`` (F below), and jcm's
own closures around them. Kept apart from ``lohmann_2m_test.py`` (already
~3000 lines) so the #941 contract reads as one unit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics import thermodynamics
from jcm.testing import check_gradients

from .cloud_utils import (
    detrained_ice_crystal_number,
    ice_volume_mean_radius_from_temperature,
)
from .lohmann_2m import cloud_microphysics_2m, demott2010_inp
from .lohmann_2m import scheme as scheme_mod
from .lohmann_2m_params import CloudParams2M

_P = CloudParams2M.default()
DT = 900.0


def _qsat(T, p, phase):
    qs, _ = thermodynamics.saturation_specific_humidity_and_derivative(
        T, p, phase=phase)
    return qs


_OPTIONAL = dict(temperature_m1="T_m1", specific_humidity_m1="q_m1",
                 qc_m1="qc_m1", qi_m1="qi_m1",
                 detrained_qc="det_qc", detrained_qi="det_qi")


def _run(col, params=_P, dt=DT):
    """Call the column function on a dict of named inputs."""
    n = col["T"].shape[0]
    zeros = jnp.zeros(n)
    return cloud_microphysics_2m(
        col["T"], col["q"], col["p"], col["qc"], col["qi"],
        col.get("qnc", zeros), col.get("qni", zeros),
        col["cf"], col["rho"], col.get("dz", jnp.full(n, 500.0)),
        col.get("tke", zeros), col.get("act", jnp.full(n, 5e7)),
        col.get("inp", zeros), zeros, dt, params,
        **{arg: col[key] for arg, key in _OPTIONAL.items() if key in col},
    )


def _tendency_outputs(out):
    """Return the differentiable outputs a gradient check contracts against."""
    t = out[0]
    return (t.dtedt, t.dqdt, t.dqcdt, t.dqidt, t.dqncdt, t.dqnidt,
            out[1], out[2])


def _end_state(col, out, dt=DT):
    """End-of-step (T, q, qc, qi) the host would hold after this step."""
    t = out[0]
    return (col["T"] + dt * t.dtedt, col["q"] + dt * t.dqdt,
            col["qc"] + dt * t.dqcdt, col["qi"] + dt * t.dqidt)


def assert_column_budgets_close(name, col, out):
    """Column water and moist enthalpy close against the surface fluxes.

    Water: Σ ρ·dz·(dq + dqc + dqi)/dt + rain + snow = 0. Enthalpy:
    Σ ρ·dz·(cp·dT − Lv·dqc − Ls·dqi)/dt = Lv·rain + Ls·snow, with the moist
    cp the scheme converts latent heat with, evaluated at the step-start
    humidity (#706). Both bounds are relative to the gross movement.
    """
    t, rain, snow = out[0], float(out[1]), float(out[2])
    n = col["T"].shape[0]
    mass = np.asarray(col["rho"] * col.get("dz", jnp.full(n, 500.0)))
    q_anchor = np.asarray(col.get("q_m1", col["q"]))
    cp = c.cpd + (c.cpv - c.cpd) * np.maximum(q_anchor, 0.0)

    dw = np.asarray(t.dqdt + t.dqcdt + t.dqidt)
    gross_w = float(np.sum((np.abs(np.asarray(t.dqdt))
                            + np.abs(np.asarray(t.dqcdt))
                            + np.abs(np.asarray(t.dqidt))) * mass))
    gross_w += abs(rain + snow)
    resid_w = float(np.sum(dw * mass) + rain + snow)
    assert gross_w > 0.0, f"{name}: column did nothing"
    assert abs(resid_w) < max(1e-5 * gross_w, 1e-12), (
        f"{name}: water open by {resid_w:.3e} (gross {gross_w:.3e})")

    heating = mass * cp * np.asarray(t.dtedt)
    dE = float(np.sum(heating - mass * c.alhc * np.asarray(t.dqcdt)
                      - mass * c.alhs * np.asarray(t.dqidt)))
    boundary = c.alhc * rain + c.alhs * snow
    gross_e = float(np.sum(np.abs(heating))) + abs(boundary)
    assert gross_e > 1.0, f"{name}: no heating — fixture is vacuous"
    assert abs(dE - boundary) < 1e-5 * gross_e, (
        f"{name}: enthalpy open by {dE - boundary:+.3e} W/m² "
        f"(gross {gross_e:.3e})")


def _spy(monkeypatch, module, name, pick):
    """Wrap ``module.name`` and record ``pick(args, kwargs, out)`` per call."""
    records = []
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        jax.debug.callback(
            lambda *vals: records.append([np.asarray(v) for v in vals]),
            *pick(args, kwargs, out))
        return out

    monkeypatch.setattr(module, name, wrapper)
    return records


def _level_of(rho_column, rho):
    """Level index of a recorded air density (each level's is distinct)."""
    k = int(np.argmin(np.abs(np.asarray(rho_column) - rho)))
    np.testing.assert_allclose(rho_column[k], rho, rtol=1e-6)
    return k


# ---------------------------------------------------------------------------
# zrid and znidetr
# ---------------------------------------------------------------------------


class TestIceRadiusFromTemperature:
    """ECHAM ``zrid`` (F 945-956), in metres."""

    def test_values_against_the_formula(self):
        T = np.array([300.0, 273.15, 253.0, 233.0, 220.0, 200.0])
        got = np.asarray(ice_volume_mean_radius_from_temperature(
            jnp.asarray(T, dtype=jnp.float32), _P))
        r_eff = np.maximum(
            23.2 * np.exp(0.015 * np.minimum(T - 273.15, 0.0)), 1.0)
        expected = np.maximum(1e-6, 0.9e-6 * r_eff)
        np.testing.assert_allclose(got, expected, rtol=1e-6)
        # ECHAM's own values from the compiled reference (micrometres).
        np.testing.assert_allclose(
            got * 1e6, [20.88, 20.88, 15.43, 11.43, 9.41, 6.97], rtol=1e-3)
        assert np.all(np.diff(got[1:]) < 0.0)


class TestDetrainedIceCrystalNumber:
    """ECHAM ``znidetr`` (F 958-982) and its gates."""

    @staticmethod
    def _call(detr=1e-5, T=250.0, lo2_2d=True, cf=0.5, rho=0.6):
        arr = lambda v: jnp.array([v], dtype=jnp.float32)  # noqa: E731
        T = arr(T)
        zrid = ice_volume_mean_radius_from_temperature(T, _P)
        return float(detrained_ice_crystal_number(
            arr(detr), T, jnp.array([lo2_2d]), arr(cf), arr(rho), zrid,
            _P)[0]), float(zrid[0])

    def test_one_level_formula(self):
        got, zrid = self._call()
        expected = (0.9 * 0.5e-2 ** 2.475 * 1000.0 / 8.253e-3
                    * 0.6 * 1e-5 / (0.5 * zrid ** 2.475))
        np.testing.assert_allclose(got, expected, rtol=1e-5)
        # A physical number of detrained crystals, 1e5-1e7 /m^3.
        assert 1e5 < got < 1e7

    def test_echam_reference_values(self):
        """Crystals per kg of in-cloud detrained condensate, from ECHAM.

        ``znidetr / (ρ·zxtec/paclc)`` is a function of temperature alone;
        the compiled ECHAM routine gives 5.04e11 /kg at 225 K and 1.66e11 /kg
        at 255 K.
        """
        for T, per_kg in ((225.0, 5.04e11), (255.0, 1.66e11)):
            got, _ = self._call(detr=1e-5, T=T, lo2_2d=True, cf=0.5, rho=0.6)
            # The reference values carry three significant figures.
            np.testing.assert_allclose(got / (0.6 * 1e-5 / 0.5), per_kg,
                                       rtol=5e-3)

    def test_gates(self):
        cqtmin = float(_P.cqtmin)
        # Warm: no detrained ice number however much is detrained.
        assert self._call(T=275.0)[0] == cqtmin
        assert self._call(T=float(_P.tmelt))[0] == cqtmin
        # Mixed phase with lo2_2d false: none (the condensate goes liquid).
        assert self._call(T=250.0, lo2_2d=False)[0] == cqtmin
        # Mixed phase with lo2_2d true, and below cthomi regardless of it.
        assert self._call(T=250.0, lo2_2d=True)[0] > 1e3
        assert self._call(T=230.0, lo2_2d=False)[0] > 1e3
        # No cover beyond clc_min, or nothing detrained: the cqtmin floor.
        assert self._call(cf=float(_P.clc_min))[0] == cqtmin
        assert self._call(cf=0.0)[0] == cqtmin
        assert self._call(detr=0.0)[0] == cqtmin

    def test_number_joins_after_sedimentation_capped_at_icemax(
            self, monkeypatch):
        """``picnc = min(picnc_sedimented + znidetr, icemax)`` (F 1251-1252).

        Records the ICNC sedimentation returns and the number melting then
        receives, per level: the difference is exactly ``znidetr``, until
        the sum reaches ``icemax``.
        """
        n = 6
        T = jnp.full(n, 228.0)                       # below cthomi
        p = jnp.linspace(2.5e4, 4e4, n)
        rho = p / (287.0 * T)
        det = jnp.array([0.0, 2e-6, 2e-5, 2e-4, 2e-3, 0.0])
        col = dict(T=T, q=_qsat(T, p, "ice"), p=p, rho=rho,
                   cf=jnp.full(n, 0.5), qc=jnp.zeros(n), qi=det,
                   qi_m1=jnp.zeros(n), qc_m1=jnp.zeros(n),
                   det_qi=det, det_qc=jnp.zeros(n))
        sedi = _spy(monkeypatch, scheme_mod, "sedimentation_ice",
                    lambda a, k, out: (a[3], out[1]))
        melt = _spy(monkeypatch, scheme_mod, "melting_snow_and_ice",
                    lambda a, k, out: (a[4],))
        _run(col)
        zrid = ice_volume_mean_radius_from_temperature(T, _P)
        lo2_2d = jnp.zeros(n, dtype=bool)            # irrelevant below cthomi
        znidetr = np.asarray(detrained_ice_crystal_number(
            det, T, lo2_2d, col["cf"], rho, zrid, _P))
        assert len(sedi) == n and len(melt) == n
        seen = []
        for (rho_k, icnc_sed), (icnc_melt_in,) in zip(sedi, melt):
            k = _level_of(rho, float(rho_k))
            expected = min(float(icnc_sed) + znidetr[k], float(_P.icemax))
            np.testing.assert_allclose(float(icnc_melt_in), expected,
                                       rtol=1e-5)
            seen.append(float(icnc_melt_in))
        assert max(seen) == pytest.approx(float(_P.icemax))  # the cap binds
        assert min(seen) < 1e3                                # and not always


# ---------------------------------------------------------------------------
# ICE-1: detrained ice is not sedimented this step
# ---------------------------------------------------------------------------


class TestDetrainedIceIsNotSedimented:
    """Sedimentation sees ``pxim1 + ztmst·pxite`` only (F 1227-1248)."""

    N = 8
    K = 3

    def _column(self, where):
        n, k = self.N, self.K
        T = jnp.full(n, 225.0)
        p = jnp.linspace(2e4, 3.5e4, n)
        rho = p / (287.0 * T)
        mass = jnp.zeros(n).at[k].set(2e-4)
        col = dict(T=T, q=_qsat(T, p, "ice"), p=p, rho=rho,
                   cf=jnp.zeros(n).at[k].set(1.0),
                   qc=jnp.zeros(n), qc_m1=jnp.zeros(n),
                   det_qc=jnp.zeros(n), qi=mass)
        if where == "detrained":
            col.update(qi_m1=jnp.zeros(n), det_qi=mass, qni=jnp.zeros(n))
        else:
            # The same mass carried from the step start as a few large
            # crystals, which do fall.
            col.update(qi_m1=mass, det_qi=jnp.zeros(n),
                       qni=jnp.where(mass > 0, 2e3, 0.0))
        return col

    def _flux_out_of_level(self, monkeypatch, col):
        records = _spy(monkeypatch, scheme_mod, "sedimentation_ice",
                       lambda a, k, out: (a[3], out[2]))
        out = _run(col)
        for rho_k, flux in records:
            if _level_of(col["rho"], float(rho_k)) == self.K:
                return float(flux), out
        raise AssertionError("level not recorded")

    def test_detrained_ice_does_not_fall_this_step(self, monkeypatch):
        col = self._column("detrained")
        flux, out = self._flux_out_of_level(monkeypatch, col)
        assert flux == 0.0, f"detrained ice sedimented: flux {flux:.3e}"
        _, _, _, qi_end = _end_state(col, out)
        # Full cover (no clear-sky share), ice saturation (no deposition):
        # the level keeps the detrained mass up to the scheme's own sinks.
        assert float(qi_end[self.K]) > 0.5 * float(col["det_qi"][self.K])
        # Nothing reaches the levels below through sedimentation.
        assert float(jnp.max(jnp.abs(out[0].dqidt[self.K + 1:]))) < 1e-12
        assert_column_budgets_close("detrained ice", col, out)

    def test_carried_ice_does_fall(self, monkeypatch):
        col = self._column("carried")
        flux, out = self._flux_out_of_level(monkeypatch, col)
        assert flux > 1e-7, "carried ice did not sediment — fixture is off"
        _, _, _, qi_end = _end_state(col, out)
        detrained_end = _end_state(
            self._column("detrained"), _run(self._column("detrained")))[3]
        assert float(qi_end[self.K]) < float(detrained_end[self.K])


# ---------------------------------------------------------------------------
# ICE-2: zrid is the radius of the ICNC diagnosis
# ---------------------------------------------------------------------------


class TestIcncDiagnosisRadius:
    """``update_in_cloud_water`` inverts ice mass at ``zrid`` (F 1511, 2616)."""

    def test_echam_reference_value(self):
        """ECHAM diagnoses 2.98e5 /m³ at 200 K from 1.44e-6 kg/kg, ρ = 0.271."""
        from .lohmann_2m import update_in_cloud_water
        one = lambda v: jnp.array([v], dtype=jnp.float32)  # noqa: E731
        T = one(200.0)
        _, icnc, *_ = update_in_cloud_water(
            pressure=one(1.6e4), activated_cdnc=one(0.0),
            condensation_rate=one(0.0), deposition_rate=one(0.0),
            tompkins_genti=one(0.0), tompkins_gentl=one(0.0),
            newly_formed_ice=one(0.0), specific_humidity_tmp=one(1e-6),
            sat_spec_humidity_tmp=one(1e-6), air_density=one(0.271),
            ice_radius_mean=ice_volume_mean_radius_from_temperature(T, _P),
            temp_prev=T, cloud_flag=jnp.array([True]),
            ice_crystal_number=one(0.0), nucleation_rate=one(0.0),
            droplet_number=one(0.0), cloud_fraction=one(1.0),
            cloud_ice_in_cloud=one(1.44e-6), cloud_liquid_in_cloud=one(0.0),
            dt=jnp.float32(DT), params=_P)
        np.testing.assert_allclose(float(icnc[0]), 2.98e5, rtol=5e-3)

    def test_number_less_ice_is_diagnosed_at_zrid(self, monkeypatch):
        n = 6
        T = jnp.linspace(215.0, 250.0, n)
        p = jnp.linspace(2e4, 4.5e4, n)
        rho = p / (287.0 * T)
        qi = jnp.full(n, 5e-6)
        col = dict(T=T, q=_qsat(T, p, "ice"), p=p, rho=rho,
                   cf=jnp.full(n, 1.0), qc=jnp.zeros(n), qi=qi,
                   qni=jnp.zeros(n))                 # ice mass, no crystals
        records = _spy(
            monkeypatch, scheme_mod, "update_in_cloud_water",
            # (rho, prid, icnc in, in-cloud ice out, icnc out)
            lambda a, k, out: (a[9], a[10], a[13], out[5], out[1]))
        _run(col)
        zrid = np.asarray(ice_volume_mean_radius_from_temperature(T, _P),
                          dtype=np.float64)
        diagnosed = 0
        for rho_k, prid, icnc_in, qib, icnc_out in records:
            k = _level_of(rho, float(rho_k))
            np.testing.assert_allclose(float(prid), zrid[k], rtol=1e-6)
            if float(icnc_in) <= float(_P.icemin) and float(qib) > 1e-12:
                formula = (0.75 * float(rho_k) * float(qib)
                           / (np.pi * float(_P.rhoice) * zrid[k] ** 3))
                expected = max(min(formula, float(_P.icemax)),
                               float(_P.icemin))
                np.testing.assert_allclose(float(icnc_out), expected,
                                           rtol=1e-4)
                diagnosed += 1
        assert diagnosed >= n // 2, f"only {diagnosed} levels diagnosed"


# ---------------------------------------------------------------------------
# ICE-5: number tendencies against the raw tracers
# ---------------------------------------------------------------------------


class TestNumberTendencyAgainstRawTracer:
    """``pxtte = (n/ρ − pxtm1)/ztmst`` with the RAW ``pxtm1`` (F 1781, 3625).

    The entry floor at 0 shapes the working numbers only, so the end-of-step
    tracer ``raw + dt·tendency`` is the scheme's own number: a negative raw
    tracer ends exactly where a zero one does. There is no upper bound on
    entry (ECHAM has none, F 600-605): ICNC is capped at ``icemax`` once the
    detrained number has joined it (F 1252), and CDNC is not capped at all.
    """

    N = 8

    def _column(self, qni, qnc):
        n = self.N
        T = jnp.linspace(240.0, 275.0, n)
        p = jnp.linspace(4e4, 8e4, n)
        rho = p / (287.0 * T)
        qc = jnp.array([0, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 0, 0], jnp.float32)
        qi = jnp.array([1e-5, 1e-5, 1e-5, 1e-5, 0, 0, 0, 1e-5], jnp.float32)
        return dict(T=T, q=0.95 * _qsat(T, p, "water"), p=p, rho=rho,
                    qc=qc, qi=qi, cf=jnp.where(qc + qi > 0, 0.7, 0.0),
                    qni=jnp.asarray(qni, jnp.float32),
                    qnc=jnp.asarray(qnc, jnp.float32),
                    tke=jnp.full(n, 0.3))

    @staticmethod
    def _end(col, out, name):
        tend = out[0].dqnidt if name == "qni" else out[0].dqncdt
        return np.asarray(col[name] + DT * tend, dtype=np.float64)

    def test_negative_raw_tracer_ends_where_zero_does(self):
        qni_raw = [-5e3, 1e4, -1.0, 5e4, 2.0, -7.0, -3.0, 1e3]
        qnc_raw = [1e7, -3e6, 5e7, -1.0, 3e7, 2.0, -4e5, 1e6]
        raw = self._column(qni_raw, qnc_raw)
        floored = self._column(np.maximum(qni_raw, 0.0),
                               np.maximum(qnc_raw, 0.0))
        out_raw, out_floored = _run(raw), _run(floored)
        for name in ("qni", "qnc"):
            end_raw = self._end(raw, out_raw, name)
            end_floored = self._end(floored, out_floored, name)
            # Float32 round-off of raw + dt·(n/ρ − raw)/dt, relative to the
            # raw magnitude; the defect this pins leaves the whole floor
            # residual (the negative raw value) in the tracer.
            scale = (np.maximum(np.abs(np.asarray(raw[name])),
                                np.abs(end_floored)) + 1.0)
            assert np.all(np.abs(end_raw - end_floored) <= 1e-5 * scale), (
                name, end_raw, end_floored)
            assert np.all(end_raw >= -1e-5 * scale), (name, end_raw)
        # Where the ccwmin repair empties the ice (returned to vapour), the
        # number goes with it, whatever the raw tracer held.
        end_ni = self._end(raw, out_raw, "qni")
        end_qi = np.asarray(raw["qi"] + DT * out_raw[0].dqidt)
        repaired = np.abs(end_qi) < 1e-12
        assert repaired.sum() >= 2, "no repaired level — fixture is off"
        np.testing.assert_allclose(end_ni[repaired], 0.0,
                                   atol=1e-5 * float(np.max(np.abs(qni_raw))))

    def test_no_entry_cap_icnc_capped_after_detrained_number(self):
        rho = np.asarray(self._column(np.zeros(self.N), np.zeros(self.N))["rho"])
        qni_raw = np.full(self.N, 1e3)
        qnc_raw = np.full(self.N, 5e7)
        qni_raw[1] = 3.0 * float(_P.icemax) / rho[1]   # ice + liquid level
        qnc_raw[5] = 3.0e11 / rho[5]                   # liquid-only level
        col = self._column(qni_raw, qnc_raw)
        out = _run(col)
        icnc_end = self._end(col, out, "qni") * rho
        cdnc_end = self._end(col, out, "qnc") * rho
        # ICNC: from 3·icemax, capped at icemax once the detrained number
        # has joined (F 1252); the cold chain then only removes crystals.
        assert icnc_end[1] <= float(_P.icemax) * 1.0001
        # CDNC: ECHAM has no cap, so 3e11 /m³ survives the step (warm rain
        # barely touches so many droplets); an entry cap would end it at
        # or below 1e11.
        assert cdnc_end[5] > 1.5e11


# ---------------------------------------------------------------------------
# ICE-6, DeMott density, the freezing substitute's number cap
# ---------------------------------------------------------------------------


def _supercooled_liquid_column(n=8):
    """Mixed-phase deck, turbulent so WBF stays shut.

    The deck holds ice mass above ``ccwmin`` but only ~70 crystals/m³, fewer
    than DeMott's ~300 at 256 K: the heterogeneous-freezing substitute then
    creates crystals, and they survive the step (in an ice-free deck the few
    hundred frozen droplets weigh less than ``ccwmin`` and the negative-mass
    repair removes them again).
    """
    T = jnp.full(n, 256.0)
    p = jnp.linspace(4e4, 7e4, n)
    rho = p / (287.0 * T)
    qc = jnp.zeros(n).at[2:6].set(3e-4)
    deck = qc > 0
    return dict(T=T, q=_qsat(T, p, "water"), p=p, rho=rho, qc=qc,
                qi=jnp.where(deck, 3e-5, 0.0), cf=jnp.where(deck, 0.8, 0.0),
                qnc=jnp.where(deck, 5e7, 0.0), qni=jnp.where(deck, 100.0, 0.0),
                tke=jnp.full(n, 5.0))


def _outputs_equal(out_a, out_b):
    for x, y in zip(jax.tree.leaves(out_a[0]), jax.tree.leaves(out_b[0])):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    return float(out_a[1]) == float(out_b[1]) and float(out_a[2]) == float(out_b[2])


class TestInpFloor:
    """``n_inp = max(ice_nuclei, DeMott)`` — the #953 stopgap."""

    def test_tiny_online_inp_leaves_the_demott_floor(self):
        col = _supercooled_liquid_column()
        base = _run(col)
        tiny = _run(dict(col, inp=jnp.where(col["qc"] > 0, 1e-3, 0.0)))
        assert _outputs_equal(base, tiny), (
            "a tiny online INP switched the DeMott floor off")
        # ...and the floor is doing something here: fewer coarse aerosol
        # (a smaller DeMott INP) makes fewer crystals.
        cleaner = _run(col, params=_P.replace(n_aer_coarse=0.01))
        fewer = float(jnp.sum((base[0].dqnidt - cleaner[0].dqnidt)
                              * col["rho"]))
        assert fewer > 0.0, "the DeMott floor is inactive in this fixture"

    def test_large_online_inp_wins(self):
        col = _supercooled_liquid_column()
        base = _run(col)
        large = _run(dict(col, inp=jnp.where(col["qc"] > 0, 1e5, 0.0)))
        extra = float(jnp.sum((large[0].dqnidt - base[0].dqnidt) * col["rho"]))
        assert extra > 0.0, "a large online INP did not raise the ICNC"

    def test_floor_gradient_matches_a_central_difference(self):
        """d/d n_aer_coarse where the DeMott floor sets the new crystals."""
        col = _supercooled_liquid_column()

        def f(n_aer_coarse):
            return _tendency_outputs(
                _run(col, params=_P.replace(n_aer_coarse=n_aer_coarse)))

        grad = jax.grad(lambda x: jnp.sum(f(x)[5] ** 2))(_P.n_aer_coarse)
        assert bool(jnp.isfinite(grad)) and float(grad) != 0.0
        check_gradients(f, (_P.n_aer_coarse,), rtol=1e-2, seed=0,
                        adjoint_rtol=1e-3)


class TestDeMottAmbientDensity:
    """DeMott (2010) is per standard litre; the scheme needs ambient m^-3."""

    def test_density_scaling(self):
        T = jnp.array([260.0, 250.0, 240.0], dtype=jnp.float32)
        rho_stp = 101350.0 / (c.rd * 273.15)
        at_stp = np.asarray(demott2010_inp(T, 0.5, rho_stp))
        d_t = 273.16 - np.asarray(T, dtype=np.float64)
        per_std_litre = 5.94e-5 * d_t ** 3.33 * 0.5 ** (0.0264 * d_t + 0.0033)
        np.testing.assert_allclose(at_stp, 1e3 * per_std_litre, rtol=1e-5)
        half = np.asarray(demott2010_inp(T, 0.5, 0.5 * rho_stp))
        np.testing.assert_allclose(half, 0.5 * at_stp, rtol=1e-6)


class TestFreezingSubstituteNumberCap:
    """New crystals cannot exceed the droplets available."""

    def test_inp_beyond_the_droplet_number_changes_nothing(self):
        col = _supercooled_liquid_column()
        deck = col["qc"] > 0
        a = _run(dict(col, inp=jnp.where(deck, 1e9, 0.0)))
        b = _run(dict(col, inp=jnp.where(deck, 1e12, 0.0)))
        assert _outputs_equal(a, b), (
            "INP above CDNC kept adding crystals")
        # The deck froze, and it holds no more crystals than it had droplets.
        assert float(jnp.sum(a[0].dqcdt * col["rho"])) < 0.0
        # The working CDNC is floored at the fixed minimum (40 /cm³).
        icnc_end = np.asarray((col["qni"] + DT * a[0].dqnidt) * col["rho"])
        icnc_start = np.asarray(col["qni"] * col["rho"])
        cdnc_start = np.maximum(np.asarray(col["qnc"] * col["rho"]),
                                1e6 * float(_P.cdnc_min_fixed))
        deck = np.asarray(deck)
        assert np.all(icnc_end[deck]
                      <= 1.0001 * (cdnc_start + icnc_start)[deck])
