"""Unit tests for the Lott-Miller (1997) SSO drag port.

The bit-exact Fortran-comparison harness lives in
``fortran_harness/compare_ssodrag.py`` and depends on a local Fortran
build that is intentionally not shipped with the repository. These tests
are sanity checks that run as part of the regular ``pytest`` suite.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

from jcm.constants import grav, rd
from jcm.physics.gravity_waves.sso import (
    SSOParameters, sso_drag,
)


def _make_alps_column(nlev: int = 47, **overrides):
    """Mid-latitude column with Alps-like sub-grid orography.

    Returns a dict suitable for ``sso_drag(**col, config=...)``.
    """
    pressure_half = np.logspace(np.log10(10.0), np.log10(101325.0), nlev + 1)
    pressure_full = 0.5 * (pressure_half[:-1] + pressure_half[1:])
    z = np.zeros(nlev)
    zh = np.zeros(nlev + 1)
    Tprof = np.zeros(nlev)
    for k in range(nlev - 1, -1, -1):
        z_g = zh[k + 1]
        if z_g < 11000.0:
            Tprof[k] = 288.15 - 0.0065 * z_g
        elif z_g < 50000.0:
            Tprof[k] = max(220.0 - 0.001 * (z_g - 20000.0), 200.0)
        else:
            Tprof[k] = max(270.0 - 0.0035 * (z_g - 50000.0), 180.0)
        dz = (rd * Tprof[k] / grav) * np.log(pressure_half[k + 1]
                                             / pressure_half[k])
        zh[k] = zh[k + 1] + dz
        z[k] = 0.5 * (zh[k] + zh[k + 1])
    layer_mass = (pressure_half[1:] - pressure_half[:-1]) / grav
    u = 30.0 * np.exp(-((z - 10000.0) / 6000.0) ** 2) + 5.0
    v = 5.0 * np.exp(-((z - 10000.0) / 8000.0) ** 2) + 1.0
    inputs = dict(
        dt=jnp.asarray(1800.0),
        coriolis=jnp.asarray(1.0e-4),
        height_full=jnp.asarray(z),
        surface_height=jnp.asarray(500.0),
        pressure_half=jnp.asarray(pressure_half),
        pressure_full=jnp.asarray(pressure_full),
        layer_mass=jnp.asarray(layer_mass),
        temperature=jnp.asarray(Tprof),
        u_wind=jnp.asarray(u),
        v_wind=jnp.asarray(v),
        mean_orography=jnp.asarray(1500.0),
        orography_std=jnp.asarray(400.0),
        orography_slope=jnp.asarray(0.07),
        orography_anisotropy=jnp.asarray(0.4),
        orography_orientation=jnp.asarray(30.0),
        peak_elevation=jnp.asarray(2500.0),
        valley_elevation=jnp.asarray(900.0),
        land_fraction=jnp.asarray(1.0),
    )
    inputs.update({k: jnp.asarray(v) for k, v in overrides.items()})
    return inputs


class TestSSOBasic:
    """Sanity properties of the SSO scheme."""

    def test_returns_finite_tendencies(self):
        col = _make_alps_column()
        tend, _ = sso_drag(**col, config=SSOParameters.default())
        assert jnp.all(jnp.isfinite(tend.dudt))
        assert jnp.all(jnp.isfinite(tend.dvdt))
        assert jnp.all(jnp.isfinite(tend.dissip))

    def test_inactive_when_orography_below_threshold(self):
        """Activation gate: setting std-dev below ``min_orog_std`` and
        peak below ``min_peak_minus_mean_elevation`` should disable the
        scheme entirely.
        """
        col = _make_alps_column(orography_std=0.5, peak_elevation=600.0)
        tend, _ = sso_drag(**col, config=SSOParameters.default())
        np.testing.assert_array_equal(np.asarray(tend.dudt), 0.0)
        np.testing.assert_array_equal(np.asarray(tend.dvdt), 0.0)
        np.testing.assert_array_equal(np.asarray(tend.dissip), 0.0)

    def test_drag_opposes_low_level_wind(self):
        """The column-integrated zonal stress should oppose the mean wind."""
        col = _make_alps_column()
        _, state = sso_drag(**col, config=SSOParameters.default())
        assert float(state.u_stress) < 0.0   # westerly column

    def test_dissipation_non_negative(self):
        """Energy dissipation should be non-negative (KE → heat). Tolerance
        is loose because the project default precision is f32.
        """
        col = _make_alps_column()
        tend, _ = sso_drag(**col, config=SSOParameters.default())
        peak_dissip = float(jnp.max(jnp.abs(tend.dissip)))
        assert jnp.all(tend.dissip >= -1e-4 * peak_dissip)

    def test_land_fraction_scaling(self):
        """Halving land_fraction halves the tendencies."""
        config = SSOParameters.default()
        tend_full, _ = sso_drag(**_make_alps_column(), config=config)
        col_half = _make_alps_column(land_fraction=0.5)
        tend_half, _ = sso_drag(**col_half, config=config)
        np.testing.assert_allclose(np.asarray(tend_half.dudt),
                                   0.5 * np.asarray(tend_full.dudt),
                                   rtol=1e-6, atol=1e-12)


class TestSSOJaxTransforms:
    def test_jit_runs(self):
        col = _make_alps_column()
        config = SSOParameters.default()
        jitted = jax.jit(lambda **kw: sso_drag(**kw, config=config))
        tend, _ = jitted(**col)
        assert jnp.all(jnp.isfinite(tend.dudt))

    def test_vmap_over_columns(self):
        col1 = _make_alps_column()
        col2 = _make_alps_column(peak_elevation=1500.0)   # smaller peak
        col3 = _make_alps_column(peak_elevation=4000.0)   # larger peak
        keys = list(col1.keys())
        batch = {k: jnp.stack([col1[k], col2[k], col3[k]]) for k in keys}
        config = SSOParameters.default()

        def one(*args):
            t, _ = sso_drag(*args, config=config)
            return t.dudt

        out = jax.vmap(one)(*[batch[k] for k in keys])
        assert out.shape == (3, 47)
        # Taller peaks → bigger surface stress (when not Froude-blocked).
        peak1 = float(jnp.max(jnp.abs(out[0])))
        peak2 = float(jnp.max(jnp.abs(out[1])))
        assert peak1 > peak2, (
            f"larger peak should give larger drag: small={peak2}, "
            f"medium={peak1}"
        )


class TestSSOParameters:
    def test_defaults_match_echam_namelist(self):
        """Tunable knobs match echam6 defaults. Static knobs (nktopg, ntop)
        live as :func:`sso_drag` kwargs.
        """
        p = SSOParameters.default()
        for name, expected in [
            ("min_peak_minus_mean_elevation", 1.0),
            ("min_orog_std", 1.0),
            ("wave_drag_coeff", 0.2),
            ("blocked_flow_drag_coeff", 1.0),
            ("mountain_lift_coeff", 0.0),
        ]:
            np.testing.assert_allclose(float(getattr(p, name)), expected,
                                       atol=1e-6, rtol=1e-6)
