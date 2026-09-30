"""Ice sources and number bookkeeping of the Lohmann 2M scheme (#941).

The rules ECHAM6.3-HAM2.3 applies to convectively detrained condensate and
to the ice crystal number in ``mo_cloud_micro_2m.f90`` (F below), and jcm's
own closures around them. Kept apart from ``lohmann_2m_test.py`` (already
~3000 lines) so the #941 contract reads as one unit.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics import thermodynamics

from .cloud_utils import (
    detrained_ice_crystal_number,
    ice_volume_mean_radius_from_temperature,
)
from .lohmann_2m import cloud_microphysics_2m
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
