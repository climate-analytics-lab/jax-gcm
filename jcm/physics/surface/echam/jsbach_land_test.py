"""The JSBACH land tile against the compiled ECHAM6.3 / JSBACH Fortran (#979).

``jcm/data/test/echam_land_reference/land_T63L47.npz`` holds inputs and outputs
of the unmodified Fortran (``README.md``, ``provenance.json`` there): the
relative-humidity, stress and canopy functions and the humidity-factor block
of ``mo_soil.f90``, ``update_soiltemp``, ``richtmyer_land``,
``update_surfacetemp`` and ``update_land``, on 324 land columns of the #979
diagnosis control run and on scans through every switch. Each test feeds the
Fortran's inputs to the jcm function and compares the outputs in float64 at
round-off tolerance; jcm's constants are set to ECHAM's for the duration
(the functions read ``jcm.constants``).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.surface.echam import jsbach_land as jl
from jcm.testing import check_gradients, check_surrogate_gradient

REF = (Path(__file__).resolve().parents[3] / "data" / "test" / "echam_land_reference"
       / "land_T63L47.npz")

ECHAM_CONSTANTS = dict(grav=9.80665, rv=461.51, alhc=2.5008e6, alhs=2.8345e6,
                       cpd=1004.64, cpv=1869.46, tmelt=273.15, rhow=1000.0, sbc=5.6704e-8)
#: Round-off: the jcm expressions are the Fortran's, in a different order of
#: operations in places (a division where ECHAM multiplies by a reciprocal).
RTOL = 1e-12


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF))


@contextlib.contextmanager
def echam_float64():
    saved = c.physical_constants
    c.set_constants(**ECHAM_CONSTANTS)
    try:
        with jax.enable_x64(True):
            yield
    finally:
        c.set_constants(saved)


def _close(actual, expected, rtol=RTOL, atol=0.0):
    np.testing.assert_allclose(np.asarray(actual, np.float64), expected, rtol=rtol, atol=atol)


def _f64(*arrays):
    return [jnp.asarray(a, jnp.float64) for a in arrays]


class TestAgainstFortran:

    def test_bare_soil_relative_humidity_is_calc_relative_humidity_upper(self, ref):
        """h(w) on a 401-point fill grid, seven field capacities, incl. w = 0 and w = 1."""
        with echam_float64():
            w, = _f64(ref["rh_scan_in/w_upper"])
            _close(jl.bare_soil_relative_humidity(w), ref["rh_scan/rh_upper"], atol=1e-15)

    def test_water_stress_factor_is_calc_water_stress_factor(self, ref):
        with echam_float64():
            w, = _f64(ref["rh_scan_in/w_root"])
            got = jl.water_stress_factor(w, 0.35, 0.75)
            _close(got, ref["rh_scan/upper/water_stress_factor"], atol=1e-15)

    def test_canopy_conductance_is_unstressed_canopy_cond_par(self, ref):
        """ECHAM3's formula on LAI 0-8 x PAR 0-600 W/m2, the night floor included."""
        with echam_float64():
            lai, par = _f64(ref["canopy_scan_in/lai"], ref["canopy_scan_in/par"])
            got = jl.unstressed_canopy_conductance(lai, par, jl.JsbachLandParameters())
            _close(got, ref["canopy_scan/conductance"], rtol=1e-11)

    @pytest.mark.parametrize("group", ["factor_scan", "factors"])
    def test_humidity_factors_are_update_soils(self, ref, group):
        """cair, csat and the stress factor of update_soil's two blocks.

        ``factor_scan`` crosses every switch (bare-soil hinge, dew, wilting,
        critical, snow 0/0.4/1, glacier 0/0.5/1, vegetation 0/0.6/1);
        ``factors`` is the 324 control-run columns. ``h`` is the Fortran's own
        ``calc_relative_humidity_upper`` value (the nsoil = 5 path jcm runs).
        """
        inp = {k.split("/", 1)[1]: ref[k] for k in ref if k.startswith(f"{group}_in/")}
        out = {k.split("/", 1)[1]: ref[k] for k in ref if k.startswith(f"{group}/")}
        n = len(out["rh_upper"])
        b = lambda k: np.broadcast_to(inp[k], (n,))  # noqa: E731
        chu = b("zchl") * np.maximum(1.0, b("wind"))
        with echam_float64():
            cair, csat, stress = jl.humidity_factors(
                *_f64(out["rh_upper"], b("w_root"), b("snow"), b("glac"), b("veg"), b("gc"),
                      chu, b("qa"), b("qs")),
                jl.JsbachLandParameters())
            _close(cair, out["upper/cair"], atol=1e-15)
            _close(csat, out["upper/csat"], atol=1e-15)
            _close(stress, out["upper/water_stress_factor"], atol=1e-15)
        # Not vacuous: every branch is exercised.
        assert np.any((out["upper/qair_fact"] == 1.0) & (out["upper/qsat_fact"] < 1.0))  # RH form
        assert np.any(out["upper/csat"] == 0.0) or group == "factors"     # a dry column
        assert np.any(b("qa") > b("qs")) or group == "factors"            # dew

    def test_top_layer_capacity_and_conductance_are_update_soiltemps(self, ref):
        """C_s = zdz2(1)Δt and Λ = zdz1(1), extracted from update_soiltemp's outputs.

        Six FAO soil rows, glacier, and snow from none through graded to a full
        snow layer. The bundle zeroes snow on glaciers, so glacier cases with
        snow are not a configuration jcm can reach and are left out.
        """
        t1 = ref["soiltemp_scan_in/t1"]
        c1, d1 = ref["soiltemp_scan/pgrndc1"], ref["soiltemp_scan/pgrndd1"]
        lam_ref = ref["soiltemp_scan/pgrndhflx"] / (c1 + (d1 - 1.0) * t1)
        cap_ref = ref["soiltemp_scan/pgrndcapc"] - 720.0 * (1.0 - d1) * lam_ref
        glac = ref["soiltemp_scan_in/glacflag"]
        psn = ref["soiltemp_scan_in/psn"]
        keep = (glac == 0.0) | (psn == 0.0)
        with echam_float64():
            for i in np.flatnonzero(keep):
                p = jl.JsbachLandParameters(
                    soil_heat_capacity=float(ref["soiltemp_scan_in/prgcgn"][i]),
                    soil_thermal_diffusivity=float(ref["soiltemp_scan_in/psodif"][i]))
                snow = psn[i] / float(p.full_cover_snow_water_equivalent)
                cap, lam = jl.top_layer_thermal_properties(
                    jnp.float64(snow), jnp.float64(glac[i]), p)
                _close(cap, cap_ref[i], rtol=1e-10)
                _close(lam, lam_ref[i], rtol=1e-10)
        assert keep.sum() == 36   # 6 soil rows x (5 snow depths + bare glacier)

    def test_richtmyer_morton_is_richtmyer_land(self, ref):
        s = {k.split("/", 1)[1]: ref[k] for k in ref if k.startswith("seb_in/")}
        o = {k.split("/", 1)[1]: ref[k] for k in ref if k.startswith("seb/")}
        zqdp = 1.0 / (s["paph_s"] - s["paph_k"])
        zfac = s["zcfh_km1"] * zqdp
        with echam_float64():
            den = 1.0 + zfac * (1.0 - s["zebsh_km1"])
            got = jl.richtmyer_morton(
                *_f64(den, den, s["ztdif_k"] + zfac * s["ztdif_km1"],
                      s["zqdif_k"] + zfac * s["zqdif_km1"], s["zcfhl"] * zqdp, s["zcfhl"] * zqdp,
                      s["zcair"], s["zcsat"]), 1.5)
            for g, k in zip(got, ("zetnl", "zftnl", "zeqnl", "zfqnl")):
                _close(g, o[k], rtol=1e-13)

    def test_surface_energy_balance_is_update_surfacetemp(self, ref):
        """ŝ on 324 columns, a quarter under a midday (x3) shortwave."""
        s = {k.split("/", 1)[1]: ref[k] for k in ref if k.startswith("seb_in/")}
        o = {k.split("/", 1)[1]: ref[k] for k in ref if k.startswith("seb/")}
        with echam_float64():
            got = jl.update_surfacetemp(
                *_f64(s["zcpq"], o["zetnl"], o["zftnl"], o["zeqnl"], o["zfqnl"], s["psold"],
                      s["pqsold"], s["pdqsold"], s["pnetrad"], s["pgrdfl"], o["pcfh"],
                      s["zcair"], s["zcsat"], s["pfracsu"], s["pgrdcap"]),
                720.0, 1.5, 0.996)
            _close(got, o["psnew"], rtol=1e-13)
            # update_land: the new lowest-level values from the same relations
            _close(o["zetnl"] * got + o["zftnl"], o["ztklevl"], rtol=1e-13)
        # Not vacuous: the balance moves the skin by up to several kelvin.
        dt_skin = (o["psnew"] - s["psold"]) / s["zcpq"]
        assert np.max(np.abs(dt_skin)) > 2.0


class TestSmoothness:
    """Exact forward values; derivatives from the named surrogates."""

    def _args(self, n=6, dry=False):
        rng = np.random.default_rng(3)
        qs = 0.03 * np.ones(n)
        h = np.full(n, 0.05) if dry else rng.uniform(0.2, 0.9, n)
        qa = qs * rng.uniform(0.2, 0.6, n)
        return tuple(jnp.asarray(a, jnp.float32) for a in (
            h, rng.uniform(0.1, 0.9, n), rng.uniform(0.0, 0.3, n), np.zeros(n),
            rng.uniform(0.2, 0.8, n), np.full(n, 0.015), np.full(n, 0.02), qa, qs))

    def test_humidity_factors_carry_the_surrogate_derivative(self):
        p = jl.JsbachLandParameters()
        f = lambda *a: jl.humidity_factors(*a, p)  # noqa: E731
        exact = lambda *a: jl.humidity_factors(  # noqa: E731
            *a, p.replace(hinge_width=0.0, stress_width=0.0))
        check_surrogate_gradient(f, exact, f, self._args())

    def test_dry_column_still_has_a_gradient_through_the_hinge(self):
        """A dry, hot bare soil (h·q_s < q_a): the reference derivative of csat is
        zero in h; the surrogate's is not.
        """
        p = jl.JsbachLandParameters()
        args = self._args(dry=True)
        csat = lambda h: jnp.sum(jl.humidity_factors(h, *args[1:], p)[1])  # noqa: E731
        g = jax.grad(csat)(args[0])
        assert np.all(np.isfinite(g)) and np.all(np.asarray(g) > 0.0)
        p0 = p.replace(hinge_width=0.0, stress_width=0.0)
        csat0 = lambda h: jnp.sum(jl.humidity_factors(h, *args[1:], p0)[1])  # noqa: E731
        assert np.all(np.asarray(jax.grad(csat0)(args[0])) == 0.0)

    def test_cold_start_exchange_has_a_finite_gradient(self):
        """A cold-start carry has no exchange coefficient and night has no canopy
        conductance: the canopy factor's slope stays finite, even where the land
        fraction that would weight it is 0.
        """
        p = jl.JsbachLandParameters()
        args = list(self._args())
        args[5] = jnp.zeros_like(args[5])   # canopy conductance (PAR = 0)
        args[6] = jnp.zeros_like(args[6])   # C_h|U| before the first vdiff step
        total = lambda *a: sum(jnp.sum(x) for x in jl.humidity_factors(*a, p))  # noqa: E731
        grads = jax.grad(total, argnums=tuple(range(9)))(*args)
        assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)

    def test_smooth_parts_have_converging_differences(self):
        p = jl.JsbachLandParameters()
        lai, par = jnp.asarray([2.0, 4.0, 6.0]), jnp.asarray([30.0, 150.0, 400.0])
        check_gradients(lambda a, b: jl.unstressed_canopy_conductance(a, b, p), (lai, par),
                        rtol=1e-2)
        snow, glac = jnp.asarray([0.0, 0.3, 0.9]), jnp.asarray([0.0, 0.2, 0.5])
        check_gradients(lambda s, g: jl.top_layer_thermal_properties(s, g, p), (snow, glac),
                        rtol=1e-2)
        w = jnp.asarray([0.1, 0.5, 0.9])
        check_gradients(jl.bare_soil_relative_humidity, (w,), rtol=1e-2)

    def test_melt_cap_is_exact_with_a_live_slope(self):
        p = jl.JsbachLandParameters()
        t = jnp.asarray([270.0, 273.0, 274.0, 280.0])
        cap = jnp.asarray([True, True, True, False])
        np.testing.assert_array_equal(np.asarray(jl.melt_cap(t, cap, p)),
                                      np.float32([270.0, 273.0, 273.15, 280.0]))
        g = jax.grad(lambda x: jnp.sum(jl.melt_cap(x, cap, p)))(t)
        assert np.all(np.asarray(g) > 0.0) and float(g[2]) < 0.5
