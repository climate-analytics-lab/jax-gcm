"""The scheme's derivatives through ECHAM's exact decisions.

The trigger, type, ascent-termination and precipitation-onset decisions keep
ECHAM's values exactly and carry the derivatives of logistic surrogates
(``switches.py``; ``docs/source/design/surrogate_gradients.md``). The
switch-level checks live in ``switches_test.py``; these are the scheme-level
ones:

* the forward value does not depend on the surrogate widths — setting every
  width to zero, which selects the reference derivatives, changes no output
  bit on the 600 ECHAM reference columns, among them the marginal ones where
  the first ascent test decides whether the column convects;
* the tunable parameters carry finite, live gradients on a convecting column,
  and the precipitation-onset depth, which enters only through a switch,
  carries one only through its surrogate;
* a column that ECHAM's first ascent test turns off returns exactly nothing,
  and its derivative with respect to the column state is the surrogate's
  continuation of the convection it would have had, which the reference
  derivative (all widths zero) does not see.
"""

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    saturation_mixing_ratio,
    tiedtke_nordeng_convection,
)
from jcm.physics.convection.tiedtke_nordeng.types import SURROGATE_WIDTH_FIELDS

_REFERENCE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "data",
    "test", "echam_cumastr_reference", "echam_cumastr.npz")
_INPUTS = ("temperature", "humidity", "pressure", "layer_thickness", "rho",
           "u_wind", "v_wind", "qc", "qi", "land_fraction", "moisture_supply",
           "moisture_tend_profile", "thvsig", "omega", "qte_dynamics",
           "layer_mass", "humidity_m1", "pressure_half")
_ZERO_WIDTHS = {name: 0.0 for name in SURROGATE_WIDTH_FIELDS}


@functools.lru_cache(maxsize=1)
def _reference():
    with np.load(_REFERENCE) as z:
        return {k: z[k] for k in z.files}


def _column_call(config, dt=900.0):
    def column(*a):
        (temperature, humidity, pressure, layer_thickness, rho, u_wind,
         v_wind, qc, qi, land_fraction, moisture_supply,
         moisture_tend_profile, thvsig, omega, qte_dynamics, layer_mass,
         humidity_m1, pressure_half) = a
        return tiedtke_nordeng_convection(
            temperature, humidity, pressure, layer_thickness, rho, u_wind,
            v_wind, qc, qi, dt, config, land_fraction, moisture_supply,
            moisture_tend_profile, thvsig, omega, qte_dynamics, layer_mass,
            humidity_m1, False, pressure_half)
    return column


def _reference_columns(dtype):
    ref = _reference()
    return [jnp.asarray(ref[f"input_{k}"], dtype) for k in _INPUTS]


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_forward_value_does_not_depend_on_the_widths(precision):
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    with jax.enable_x64(precision == "float64"):
        args = _reference_columns(dtype)
        default = jax.jit(jax.vmap(_column_call(
            ConvectionParameters.default())))(*args)
        reference = jax.jit(jax.vmap(_column_call(
            ConvectionParameters.default(**_ZERO_WIDTHS))))(*args)
    for got, want in zip(jax.tree.leaves(default), jax.tree.leaves(reference)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # The columns include ones that convect and ones that do not.
    ktype = np.asarray(default[1].ktype)
    assert np.any(ktype == 0) and np.any(ktype == 1) and np.any(ktype == 2)
    assert np.any(ktype == 3)


def _deep_column(nlev=20, t_sfc=300.0):
    """TOA-first moist conditionally-unstable column with a mixed layer."""
    p = jnp.linspace(10_000.0, 101_325.0, nlev)
    t = t_sfc - 6.5e-3 * 16_000.0 * (1.0 - p / p[-1]) ** 0.8
    t = jnp.clip(t, 200.0, t_sfc)
    q = 0.85 * jax.vmap(saturation_mixing_ratio)(p, t)
    rho = p / (287.0 * t)
    dz = jnp.abs(jnp.diff(p, prepend=p[:1] * 0.5)) / (rho * 9.81)
    return t, q, p, dz, rho


def _deep_precip(params, supply=2e-5, convergence=1.5):
    """Return the convective precipitation of the deep column.

    ECHAM's ``zdqcv`` test is deep where the column converges more moisture
    than 1.1 times the surface supplies, so it gets a resolved convergence on
    top of the supply.
    """
    t, q, p, dz, rho = _deep_column()
    nlev = t.shape[0]
    sl = slice(nlev // 2, nlev - 4)
    conv = jnp.zeros(nlev).at[sl].set(
        convergence * supply / jnp.sum(rho[sl] * dz[sl]))
    tend, _ = tiedtke_nordeng_convection(
        t, q, p, dz, rho, jnp.zeros_like(t), jnp.zeros_like(t),
        jnp.zeros_like(t), jnp.zeros_like(t), dt=900.0, config=params,
        moisture_supply=jnp.asarray(supply), land_fraction=jnp.asarray(0.0),
        qte_dynamics=conv)
    return tend.precip_conv


def _grad_wrt(field, base=None):
    base = ConvectionParameters.default() if base is None else base

    def loss(x):
        return _deep_precip(base.replace(**{field: x}))

    return jax.grad(loss)(jnp.asarray(getattr(base, field)))


class TestParameterGradients:
    @pytest.mark.parametrize("field", ["entrpen", "tau", "cprcon", "cmfdeps"])
    def test_tunable_parameter_gradient_is_live(self, field):
        g = _grad_wrt(field)
        assert np.isfinite(float(g)) and float(g) != 0.0, (field, g)

    def test_precip_onset_depth_is_live_only_through_its_surrogate(self):
        """``cu_dnoprc_ocean`` enters only ECHAM's precipitation-onset switch.

        Its reference derivative is zero; the surrogate's is not.
        """
        g = _grad_wrt("cu_dnoprc_ocean")
        assert np.isfinite(float(g)) and float(g) != 0.0, g
        g_ref = _grad_wrt("cu_dnoprc_ocean",
                          ConvectionParameters.default(**_ZERO_WIDTHS))
        assert float(g_ref) == 0.0, g_ref

    def test_all_parameter_gradients_finite(self):
        """No NaN through any differentiable leaf of the parameters."""
        base = ConvectionParameters.default()
        leaves = {
            k: getattr(base, k) for k in base.__dataclass_fields__
            if k not in SURROGATE_WIDTH_FIELDS
            and jnp.issubdtype(jnp.asarray(getattr(base, k)).dtype,
                               jnp.floating)}

        def loss(values):
            return _deep_precip(base.replace(**values))

        grads = jax.grad(loss)({k: jnp.asarray(v) for k, v in leaves.items()})
        for k, g in grads.items():
            assert bool(jnp.all(jnp.isfinite(g))), (k, g)


def test_stable_column_is_exactly_off():
    """No phantom convection in a statically stable column."""
    nlev = 20
    p = jnp.linspace(10_000.0, 101_325.0, nlev)
    t = jnp.full(nlev, 280.0)
    q = jnp.full(nlev, 1e-3)
    rho = p / (287.0 * t)
    dz = jnp.abs(jnp.diff(p, prepend=p[:1] * 0.5)) / (rho * 9.81)
    tend, state = tiedtke_nordeng_convection(
        t, q, p, dz, rho, jnp.zeros_like(t), jnp.zeros_like(t),
        jnp.zeros_like(t), jnp.zeros_like(t), dt=900.0,
        config=ConvectionParameters.default(),
        moisture_supply=jnp.asarray(2e-5), land_fraction=jnp.asarray(0.0))
    assert int(state.ktype) == 0
    assert float(tend.precip_conv) == 0.0
    assert float(jnp.max(jnp.abs(tend.dtedt))) == 0.0


class TestAcrossTheOnset:
    """Columns whose convection ECHAM's first ascent test decides.

    Fogged-RCE columns with a cloud base at ``klevm1``: in one ECHAM's plume
    passes the first interface above cloud base and the column convects, in
    the other it fails there and the column does not. The derivative of the
    column heating along a warming of the whole column is taken with the
    default widths and with every width zero (the reference derivative).
    """

    @staticmethod
    def _columns():
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            find_cloud_base,
        )
        ref = _reference()
        nlev = ref["input_temperature"].shape[1]
        fogged = np.where(ref["group"] == "rce_fogged")[0]
        on = [i for i in fogged if ref["echam_ktype"][i] > 0
              and ref["echam_kcbot"][i] == nlev - 1]
        # Off, although ``cubase`` finds a cloud base (at ``klevm1``): the
        # first ascent test failed. jcm's earlier soft ascent convected here.
        off = []
        for i in fogged[ref["sample_class"][fogged] == "previously_port_only"]:
            kb, found = find_cloud_base(
                jnp.asarray(ref["input_temperature"][i]),
                jnp.asarray(ref["input_humidity"][i]),
                jnp.asarray(ref["input_pressure"][i]),
                ConvectionParameters.default(),
                thvsig=jnp.asarray(ref["input_thvsig"][i]),
                cp_moist=None,
                pressure_half=jnp.asarray(ref["input_pressure_half"][i]))
            if bool(found) and int(kb) == nlev - 2:
                off.append(i)
                break
        return np.array([on[0], off[0]])

    @staticmethod
    def _heating_jvp(config):
        def one(*a):
            def heating(temperature):
                tend, state = _column_call(config)(temperature, *a[1:])
                return (jnp.sum(tend.dtedt * jnp.diff(a[-1])), state.ktype,
                        tend.precip_conv)

            primal, tangent = jax.jvp(
                lambda t: heating(t)[0], (a[0],), (jnp.ones_like(a[0]),))
            _, ktype, precip = heating(a[0])
            return primal, tangent, ktype, precip

        return jax.jit(jax.vmap(one))

    def test_value_is_exact_and_derivative_crosses_the_onset(self):
        ref = _reference()
        rows = self._columns()
        with jax.enable_x64(True):
            args = [jnp.asarray(ref[f"input_{k}"][rows], jnp.float64)
                    for k in _INPUTS]
            heat, dheat, ktype, precip = self._heating_jvp(
                ConvectionParameters.default())(*args)
            heat_r, dheat_r, ktype_r, precip_r = self._heating_jvp(
                ConvectionParameters.default(**_ZERO_WIDTHS))(*args)
        heat, dheat = np.asarray(heat), np.asarray(dheat)
        dheat_r = np.asarray(dheat_r)
        # Values: ECHAM's decisions, identical under both widths.
        np.testing.assert_array_equal(np.asarray(ktype), [2, 0])
        np.testing.assert_array_equal(np.asarray(ktype), np.asarray(ktype_r))
        np.testing.assert_array_equal(heat, np.asarray(heat_r))
        assert heat[1] == 0.0 and float(np.asarray(precip)[1]) == 0.0
        assert heat[0] > 0.0
        # Derivatives: finite everywhere; the column that is off responds to
        # a warming only through the surrogates, the one that is on through
        # its plume and, differently, through the surrogates as well.
        assert np.all(np.isfinite(dheat)) and np.all(np.isfinite(dheat_r))
        assert dheat[1] != 0.0 and dheat_r[1] == 0.0
        assert dheat[0] != dheat_r[0]
