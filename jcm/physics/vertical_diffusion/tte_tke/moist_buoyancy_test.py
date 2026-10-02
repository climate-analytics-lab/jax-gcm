"""Tests for the moist, cloud-weighted buoyancy of the TTE-TKE vertical diffusion.

The reference is ECHAM itself: ``jcm/data/test/echam_vdiff_reference/`` holds
what the compiled, unmodified ``vdiff.f90`` r7492 statements (saturation,
half-level averaging, buoyancy, shear and Richardson number, l.658-700 and
777-799) and ``mo_surface_{land,ocean,ice}.f90::precalc_*`` return for 384
columns (320 from a T63L47 ``t63-echam-1m`` run in 8 stratified classes and 64
designed ones) and 96 surface cells (provenance in the README there). jcm runs
with ECHAM's constants for the comparison, as the ``cuadjtq`` test does.

Tolerances (all measured, then given a margin of several times):

* float64 against ECHAM as it runs, spline tables included: the fields agree
  to 6.5e-14 of their largest value (``zqss``, the table's interpolation
  error); the buoyancy, shear and Richardson number to 2e-15. The bounds are
  1e-12 and 1e-13 of the largest value.
* float64 against the build with the tables replaced by the Sonntag fit they
  tabulate: the same arithmetic, so round-off only (``zqss`` 1.5e-15, the
  Richardson number 2e-15 of the largest value). The bound is 1e-13.
* float32: the inputs round to float32 (the geopotential, ~1e5 m²/s², to 8e-3
  of a 1e3 m²/s² thickness), and the measured error is 1.3e-6 of the largest
  value. The bound is 1e-5.
"""

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.testing import check_gradients
from jcm.physics.thermodynamics import saturation_specific_humidity
from .moist_buoyancy import (
    SHEAR_FLOOR,
    cloud_weighted_buoyancy_multipliers,
    interfaces_to_levels,
    interior_buoyancy_and_shear,
    interior_buoyancy_terms,
    richardson_number,
)
from .surface_layer import surface_bulk_richardson
from .turbulence_coefficients import compute_richardson_number
from .vertical_diffusion_types import VDiffParameters, VDiffState

_REFERENCE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "data", "test",
    "echam_vdiff_reference", "vdiff_T63L47.npz")

# (atol as a fraction of the field's largest |value|, rtol) per comparison.
_TOL = {
    ("float64", ""): (1.0e-12, 1.0e-10),
    ("float64", "analytic_"): (1.0e-13, 1.0e-10),
    ("float32", ""): (1.0e-5, 1.0e-3),
}

# The tile order of the vdiff state: water, ice, land.
_TILE = {"ocean": 0, "ice": 1, "land": 2}


def _reference():
    with np.load(_REFERENCE) as z:
        return {k: z[k] for k in z.files}


@contextlib.contextmanager
def _echam_constants(ref):
    """ECHAM's ``grav``, ``alv``, ``als``, ``rv``, ``cpd``, ``tmelt`` for the
    duration (``rd`` and the derived coefficients follow), then jcm's own again.
    """
    saved = c.physical_constants
    c.set_constants(grav=float(ref["echam_grav"]), alhc=float(ref["echam_alv"]),
                    alhs=float(ref["echam_als"]), rv=float(ref["echam_rv"]),
                    cpd=float(ref["echam_cpd"]), tmelt=float(ref["echam_tmelt"]))
    try:
        yield
    finally:
        c.set_constants(saved)


def _state(**fields) -> VDiffState:
    """Build a state for the buoyancy.

    Only the thermodynamic, wind, pressure and geopotential fields matter; the
    rest are zeros of the right shape.
    """
    t = fields["temperature"]
    ncol, nlev = t.shape
    dtype = t.dtype
    zero = jnp.zeros((ncol, nlev), dtype)
    tiles = jnp.zeros((ncol, 3), dtype)
    base = dict(
        u=zero, v=zero, qv=zero, qc=zero, qi=zero, cloud_fraction=zero,
        air_mass=zero, surface_temperature=tiles, surface_fraction=tiles,
        roughness_length=tiles, roughness_heat=tiles, surface_wetness=tiles,
        height_full=zero, height_half=jnp.zeros((ncol, nlev + 1), dtype),
        tke=zero, thv_variance=zero, ocean_u=jnp.zeros(ncol, dtype),
        ocean_v=jnp.zeros(ncol, dtype))
    base.update(fields)
    return VDiffState(**base)


def _reference_state(ref, dtype) -> VDiffState:
    """Return the reference columns as a (ncol, nlev) state, top level first."""
    g = lambda k: jnp.asarray(ref["in_" + k].T, dtype)  # noqa: E731
    return _state(
        u=g("pum1"), v=g("pvm1"), temperature=g("ptm1"), qv=g("pqm1"),
        qc=g("pxlm1"), qi=g("pxim1"), cloud_fraction=g("paclc"),
        pressure_full=g("papm1"), pressure_half=g("paphm1"),
        geopotential=g("pgeom1"))


def _hydrostatic_state(temperature, pressure_half, **fields) -> VDiffState:
    """Build a top-first column whose geopotential is hydrostatic in ``temperature``."""
    ph = pressure_half
    pf = 0.5 * (ph[:, :-1] + ph[:, 1:])
    dphi = c.rd * temperature * jnp.log(ph[:, 1:] / ph[:, :-1])   # layer thickness
    phi_half = jnp.concatenate(
        [jnp.sum(dphi, axis=1, keepdims=True),
         jnp.sum(dphi, axis=1, keepdims=True) - jnp.cumsum(dphi, axis=1)], axis=1)
    return _state(temperature=temperature, pressure_full=pf, pressure_half=ph,
                  geopotential=0.5 * (phi_half[:, :-1] + phi_half[:, 1:]), **fields)


class TestAgainstEchamVdiff:
    """jcm's interior stability against ECHAM's compiled ``vdiff.f90``."""

    def test_reference_covers_the_regimes(self):
        """The set is not vacuous: it contains the cases the formulation turns on."""
        ref = _reference()
        t, cc, ri = ref["in_ptm1"], ref["in_paclc"], ref["echam_zri"]
        assert t.shape[1] >= 384
        assert (cc > 0).mean() > 0.05 and ((cc > 0) & (cc < 1)).any()
        assert (ri < 0).mean() > 0.01 and (ri > 1.0).any()
        # liquid and ice levels on the two sides of an interface
        crossing = (t[1:] - c.tmelt) * (t[:-1] - c.tmelt) <= 0
        assert crossing.sum() > 100
        # ECHAM's shear floor is the denominator somewhere, and is not everywhere
        floor = ref["echam_zshear"] < SHEAR_FLOOR
        assert floor.any() and not floor.all()
        # the cloud weighting matters: the answer differs from the clear one
        assert np.abs(ref["echam_zdus1"] - (1.0 + 0.6078 * 0.01)).max() > 0.01

    def test_constants_match_echam(self):
        """Under the override jcm's derived rd and vtmpc1 are ECHAM's."""
        ref = _reference()
        with _echam_constants(ref):
            np.testing.assert_allclose(c.rd, ref["echam_rd"], rtol=1e-15)
            np.testing.assert_allclose(c.vtmpc1, ref["echam_vtmpc1"], rtol=1e-14)

    @pytest.mark.parametrize("variant", ["", "analytic_"],
                             ids=["echam_tables", "analytic_tables"])
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    def test_interior_matches_echam(self, precision, variant):
        if precision == "float32" and variant:
            pytest.skip("float32 is bounded against ECHAM as it runs")
        ref = _reference()
        atol_frac, rtol = _TOL[(precision, variant)]
        dtype = jnp.float64 if precision == "float64" else jnp.float32
        with _echam_constants(ref), jax.enable_x64(precision == "float64"):
            state = _reference_state(ref, dtype)
            terms = interior_buoyancy_terms(state)
            ri = compute_richardson_number(state)
            qss = saturation_specific_humidity(state.temperature, state.pressure_full)
        got = {"zqss": qss, "zbuoy": terms.buoyancy, "zri": ri}
        if not variant:
            got.update({"zshear": terms.shear, "zqssm": terms.saturation_humidity,
                        "zdus1": terms.dus1, "zdus2": terms.dus2,
                        "zteldif": terms.teldif, "zqddif": terms.qddif})
        assert terms.buoyancy.dtype == dtype
        for name, value in got.items():
            want = ref[f"echam_{variant}{name}"].T
            scale = np.abs(want).max()
            # float32: the table error (ECHAM's spline vs the fit) is far below
            # the rounding, so the same bound serves both.
            np.testing.assert_allclose(
                np.asarray(value, np.float64), want, rtol=rtol, atol=atol_frac * scale,
                err_msg=f"{name} ({precision}, {variant or 'echam_tables'})")

    def test_richardson_number_is_buoyancy_over_floored_shear(self):
        ref = _reference()
        with _echam_constants(ref), jax.enable_x64(True):
            state = _reference_state(ref, jnp.float64)
            buoyancy, shear = interior_buoyancy_and_shear(state)
            ri = compute_richardson_number(state)
            expected = buoyancy / jnp.maximum(shear, SHEAR_FLOOR)
        np.testing.assert_allclose(np.asarray(ri), np.asarray(expected), rtol=1e-12)
        assert SHEAR_FLOOR == 1.0e-5

    def test_thv_gradient_is_echams_zthvirdif(self):
        """The variance budget's ∂θ_v/∂z is the same ECHAM statement (l.791)."""
        from .vertical_diffusion import _column_thv_gradient
        ref = _reference()
        with _echam_constants(ref), jax.enable_x64(True):
            s = _reference_state(ref, jnp.float64)
            height = s.geopotential / c.grav
            grad = _column_thv_gradient(s.temperature, s.pressure_full, s.qv,
                                        s.qc, s.qi, height)
        # ECHAM's zthvirdif sits on the interface below the full level it is
        # padded onto; the model's gradient is the same difference.
        want = ref["echam_zthvirdif"].T                           # (ncol, nlev-1)
        scale = np.abs(want).max()
        np.testing.assert_allclose(np.asarray(grad[:, 1:]), want, rtol=1e-9,
                                   atol=1e-12 * scale)

    @pytest.mark.parametrize("tile", ["ocean", "ice", "land"])
    def test_surface_bulk_richardson_matches_echam(self, tile):
        """Match the surface layer's Ri to ``precalc_{ocean,ice,land}``, cover included."""
        ref = _reference()
        ncell = len(ref["sfc_in_ptm1"])
        prefix = "" if tile == "land" else f"{tile}_"
        t_sfc = ref["sfc_in_ptslm1"] if tile == "land" else ref[f"sfc_in_{tile}_ptslm1"]
        want = ref[f"sfc_echam_{prefix}zril"]
        cover = ref["sfc_in_paclc"]
        assert (cover > 0).sum() >= 40 and (cover == 0).sum() >= 40
        with _echam_constants(ref), jax.enable_x64(True):
            def column(name):
                # the lowest level is the last: two identical levels suffice
                return jnp.asarray(np.stack([ref[f"sfc_in_{name}"]] * 2, axis=1))

            tiles = jnp.stack([jnp.asarray(ref["sfc_in_ptslm1"] * 0 + 280.0)] * 3, axis=1)
            tiles = tiles.at[:, _TILE[tile]].set(jnp.asarray(t_sfc))
            height = jnp.asarray(ref["sfc_in_pgeom1"] / c.grav)[:, None] * jnp.ones((1, 2))
            state = _state(
                u=column("pum1"), v=column("pvm1"), temperature=column("ptm1"),
                qv=column("pqm1"), qc=column("pxlm1"), qi=column("pxim1"),
                cloud_fraction=column("paclc"), pressure_full=column("papm1"),
                pressure_half=jnp.stack([jnp.asarray(ref["sfc_in_paphm1"])] * 2, axis=1),
                geopotential=height * c.grav,
                height_full=height, height_half=jnp.zeros((ncell, 3)),
                surface_wetness=jnp.ones((ncell, 3)))
            params = VDiffParameters.default(surface_layer_fsl=0.5)
            speed = jnp.sqrt(state.u[:, -1] ** 2 + state.v[:, -1] ** 2)
            ri = surface_bulk_richardson(state, params, speed, tiles,
                                         state.temperature[:, -1])[:, _TILE[tile]]
            clear = surface_bulk_richardson(
                state._replace(cloud_fraction=jnp.zeros_like(state.cloud_fraction)),
                params, speed, tiles, state.temperature[:, -1])[:, _TILE[tile]]
        np.testing.assert_allclose(np.asarray(ri), want, rtol=1e-9,
                                   atol=1e-11 * np.abs(want).max(),
                                   err_msg=f"{tile} bulk Richardson number")
        # the lowest-level cover is what ECHAM passes: dropping it must matter
        cloudy = cover > 0
        assert np.abs(np.asarray(ri - clear))[cloudy].max() > 1e-4 * np.abs(want).max()
        np.testing.assert_array_equal(np.asarray(ri)[~cloudy], np.asarray(clear)[~cloudy])


class TestMultipliers:
    """The shared multiplier function (the interior and the surface layer call it)."""

    def test_clear_sky_is_the_virtual_correction(self):
        q_t = jnp.array([0.0, 0.004, 0.02])
        d1, d2 = cloud_weighted_buoyancy_multipliers(
            c.alhc, jnp.array(285.0), q_t, jnp.array(0.01), jnp.array(0.0))
        np.testing.assert_allclose(d1, 1.0 + c.vtmpc1 * q_t, rtol=1e-6)
        np.testing.assert_allclose(d2, c.vtmpc1, rtol=1e-6)

    def test_saturated_value_is_the_moist_adiabatic_correction(self):
        """Overcast: ``zdus1 = zmult5``, ``zdus2 = zmult4`` from the issue's formulas."""
        with jax.enable_x64(True):
            L, T, qt, qs = c.alhc, 280.0, 0.012, 0.0095
            d1, d2 = cloud_weighted_buoyancy_multipliers(
                L, jnp.array(T), jnp.array(qt), jnp.array(qs), jnp.array(1.0))
            fux, fox = L / (c.cpd * T), L / (c.rd * T)
            m1 = 1.0 + c.vtmpc1 * qt
            m3 = (c.rd / c.rv) * fox * qs / (1.0 + (c.rd / c.rv) * fux * fox * qs)
            m5 = m1 - (fux * m1 - c.rv / c.rd) * m3
            np.testing.assert_allclose(float(d1), m5, rtol=1e-12)
            np.testing.assert_allclose(float(d2), fux * m5 - 1.0, rtol=1e-12)

    def test_cover_blends_linearly(self):
        args = (c.alhc, jnp.array(285.0), jnp.array(0.01), jnp.array(0.011))
        clear = cloud_weighted_buoyancy_multipliers(*args, jnp.array(0.0))
        full = cloud_weighted_buoyancy_multipliers(*args, jnp.array(1.0))
        half = cloud_weighted_buoyancy_multipliers(*args, jnp.array(0.25))
        for a, b, h in zip(clear, full, half):
            np.testing.assert_allclose(h, 0.75 * a + 0.25 * b, rtol=1e-6)


class TestPhysicalLimits:
    """What the moist buoyancy must do, independent of the reference."""

    @staticmethod
    def _saturated_column(ncol=2, nlev=8, lapse=None, theta_l=296.0, q_total=0.013,
                          cover=1.0):
        """Saturated column; ``lapse`` None: well mixed (uniform θ_l, q_t)."""
        ph = jnp.linspace(75000.0, 92000.0, nlev + 1)[None, :] * jnp.ones((ncol, 1))
        pf = 0.5 * (ph[:, :-1] + ph[:, 1:])
        kappa = c.rd / c.cpd
        if lapse is None:
            # T solves  T − (L/cp)(q_t − q_s(T, p)) = θ_l (p/p0)^κ  (bisection)
            lo, hi = jnp.full_like(pf, 200.0), jnp.full_like(pf, 330.0)
            target = theta_l * (pf / c.p0) ** kappa
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                f = mid - (c.alhc / c.cpd) * (q_total - saturation_specific_humidity(mid, pf)) - target
                lo, hi = jnp.where(f < 0, mid, lo), jnp.where(f < 0, hi, mid)
            t = 0.5 * (lo + hi)
            qv = saturation_specific_humidity(t, pf)
            qc = q_total - qv
        else:
            z_full = (c.rd * 285.0 / c.grav) * jnp.log(92000.0 / pf)
            t = 290.0 - lapse * (z_full - z_full[:, -1:])
            qv = saturation_specific_humidity(t, pf)
            qc = jnp.zeros_like(t)
        return _hydrostatic_state(t, ph, qv=qv, qc=qc, cloud_fraction=jnp.full_like(t, cover),
                                  u=jnp.linspace(1.0, 4.0, nlev)[None, :] * jnp.ones((ncol, 1)))

    def test_well_mixed_saturated_layer_is_neutral_but_dry_stable(self):
        """Uniform θ_l and q_t in cloud is neutral to the moist buoyancy, while the dry
        static stability of the same layer is positive: the dry N² would damp it.
        """
        with jax.enable_x64(True):
            state = self._saturated_column()
            assert float(np.asarray(state.qc).min()) > 1e-5, "vacuous: the layer is not cloudy"
            buoyancy, _ = interior_buoyancy_and_shear(state)
            theta = state.temperature * (c.p0 / state.pressure_full) ** (c.rd / c.cpd)
            zhh = state.geopotential[:, :-1] - state.geopotential[:, 1:]
            dry = np.asarray((c.grav / theta[:, 1:]) * (theta[:, :-1] - theta[:, 1:]) / zhh * c.grav)
            buoyancy = np.asarray(buoyancy)
        assert dry.min() > 1e-5
        assert np.abs(buoyancy).max() < 1e-4 * dry.min()

    def test_conditionally_unstable_layer_flips_sign_with_cover(self):
        """A saturated lapse rate between the moist and dry adiabats is dry-stable and
        moist-unstable; the Richardson number falls monotonically with the cover.
        """
        with jax.enable_x64(True):
            ri = [np.asarray(compute_richardson_number(
                self._saturated_column(lapse=6.0e-3, cover=cc)))
                for cc in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert ri[0].min() > 0.0, "clear weighting: the dry-like stable answer"
        assert ri[-1].max() < 0.0, "overcast: conditionally unstable"
        for a, b in zip(ri[:-1], ri[1:]):
            assert np.all(b < a)

    def test_dry_clear_limit_is_the_potential_temperature_gradient(self):
        """Without water and cloud the buoyancy is ``(g/θ) ∂θ/∂z``.

        (A cover of 1 over air with no water is not the dry limit: the cover
        says the layer is saturated, and its multipliers carry ``q_s``.)
        """
        with jax.enable_x64(True):
            t = jnp.array([[225.0, 232.0, 241.0, 252.0, 262.0, 270.0]])
            ph = jnp.linspace(30000.0, 60000.0, 7)[None, :]
            state = _hydrostatic_state(t, ph)
            buoyancy, _ = interior_buoyancy_and_shear(state)
            theta = t * (c.p0 / state.pressure_full) ** (c.rd / c.cpd)
            zhh = state.geopotential[:, :-1] - state.geopotential[:, 1:]
            w_up = (ph[:, :-2] - ph[:, 1:-1]) / (ph[:, :-2] - ph[:, 2:])
            theta_mid = w_up * theta[:, :-1] + (1 - w_up) * theta[:, 1:]
            want = c.grav ** 2 / theta_mid * (theta[:, :-1] - theta[:, 1:]) / zhh
            np.testing.assert_allclose(buoyancy, want, rtol=1e-12)

    def test_mass_weighted_saturation_is_not_the_saturation_of_the_mean(self):
        state = self._saturated_column(lapse=6.0e-3, cover=0.5)
        terms = interior_buoyancy_terms(state)
        qs = saturation_specific_humidity(state.temperature, state.pressure_full)
        ph = state.pressure_half
        w_up = (ph[:, :-2] - ph[:, 1:-1]) / (ph[:, :-2] - ph[:, 2:])
        np.testing.assert_allclose(
            terms.saturation_humidity, w_up * qs[:, :-1] + (1 - w_up) * qs[:, 1:], rtol=1e-6)

    def test_floor_makes_calm_air_finite(self):
        """With no shear the Richardson number is the buoyancy over ECHAM's floor."""
        state = self._saturated_column(lapse=3.0e-3)._replace(u=jnp.zeros((2, 8)))
        buoyancy, shear = interior_buoyancy_and_shear(state)
        ri = richardson_number(buoyancy, shear)
        assert float(jnp.max(shear)) == 0.0
        np.testing.assert_allclose(ri, buoyancy / SHEAR_FLOOR, rtol=1e-6)

    def test_levels_take_the_interface_above(self):
        x = jnp.arange(6.0)[None, :]
        np.testing.assert_array_equal(interfaces_to_levels(x), [[0, 0, 1, 2, 3, 4, 5]])


class TestGradients:
    """Automatic differentiation of the interior stability."""

    @staticmethod
    def _columns(n=12):
        ref = _reference()
        t = ref["in_ptm1"]
        away = np.all(np.abs(t - c.tmelt) > 3.0, axis=0)           # off the phase jump
        cloudy = (ref["in_paclc"] > 0.1).any(axis=0) & (ref["in_pxlm1"] + ref["in_pxim1"] > 1e-6).any(axis=0)
        idx = np.flatnonzero(away & cloudy)[:n]
        assert len(idx) == n, "vacuous: no cloudy columns off the phase jump"
        return ref, idx

    def test_richardson_number_gradients(self):
        ref, idx = self._columns()
        with _echam_constants(ref), jax.enable_x64(True):
            full = _reference_state(ref, jnp.float64)
            sub = jax.tree.map(lambda a: a[idx] if a.ndim and a.shape[0] == ref["in_ptm1"].shape[1] else a, full)

            def ri(temperature, qv, qc, qi, cloud_fraction, u, v):
                return compute_richardson_number(sub._replace(
                    temperature=temperature, qv=qv, qc=qc, qi=qi,
                    cloud_fraction=cloud_fraction, u=u, v=v))

            args = (sub.temperature, sub.qv, sub.qc, sub.qi, sub.cloud_fraction, sub.u, sub.v)
            check_gradients(ri, args, rtol=1e-4,
                            live_inputs=("[0]", "[1]", "[2]", "[4]", "[5]"))

    def test_gradients_are_finite_at_a_degenerate_state(self):
        """Dry, calm, clear, isothermal at the melting point: no 0·inf in the reverse pass."""
        with jax.enable_x64(True):
            t = jnp.full((1, 6), c.tmelt)
            ph = jnp.linspace(50000.0, 60000.0, 7)[None, :]
            state = _hydrostatic_state(t, ph)

            def total(temperature, qv, qc, qi, cover, u):
                s = state._replace(temperature=temperature, qv=qv, qc=qc, qi=qi,
                                   cloud_fraction=cover, u=u)
                b, sh = interior_buoyancy_and_shear(s)
                return jnp.sum(richardson_number(b, sh)) + jnp.sum(b)

            zeros = jnp.zeros((1, 6))
            grads = jax.grad(total, argnums=(0, 1, 2, 3, 4, 5))(t, zeros, zeros, zeros, zeros, zeros)
            for g in grads:
                assert bool(jnp.all(jnp.isfinite(g)))
