"""Surface-coupled boundary-layer cases run through the *existing* SCM.

These are the companion to ``boundary_layer_cases_test.py``. That file tests the
TTE-TKE term as an isolated interior operator; this one closes the loop the
three classic single-column cases actually run on, by driving the real
:class:`jcm.single_column_model.SingleColumnModel` so the *surface-coupled*
signatures emerge:

  * **Andren et al. (1994)** neutral Ekman layer — a sub-geostrophic, backed
    surface wind, recovery to geostrophic aloft, and the supergeostrophic Ekman
    jet (https://doi.org/10.1002/qj.49712052003).
  * **GABLS1 / Cuxart et al. (2006)** stable layer — a surface inversion, a
    nocturnal low-level jet, and suppressed turbulence
    (https://doi.org/10.1007/s10546-005-3780-1).
  * **Wangara Day 33** convective layer — an upward surface heat flux that
    builds a deepening, well-mixed convective layer.

What is real vs supplied
------------------------
REAL, unmodified jcm code (the physics under test):
  * The existing ``SingleColumnModel`` provides the BCs, the relaxation/forcing
    hooks and the ``lax.scan`` run loop.
  * ``echam_physics()`` supplies the real ECHAM **surface** term
    (``EchamSurface``) and the **vdiff** term (``TteTkeVerticalDiffusion``) —
    the surface fluxes and turbulent mixing are computed exactly as in a model
    step. The winds/temperature/humidity free-run under those physics
    tendencies via the SCM's relaxation hook at a long timescale.

SUPPLIED here, because the SCM deliberately omits it:
  * A small ``GeostrophicCoriolisForcing`` term — ``du/dt = f (v - v_g)``,
    ``dv/dt = -f (u - u_g)``. The SCM has no dynamical core, so without this the
    wind-rotation cases (Ekman spiral, low-level jet) cannot form. It is pure
    large-scale dynamics, not a physics parameterization.
  * The stack is trimmed to surface + vdiff (radiation, convection,
    microphysics, ... removed). In the full stack convection + microphysics
    cool the free-running column ~12 K/10 h and bury the BL signal; isolating
    surface + vdiff is standard for BL LES intercomparisons (GABLS1/Wangara
    prescribe or omit radiation) and keeps the physics under test untouched.

Fidelity caveats: columns are dry; surface skin temperature is a *constant*
boundary condition per case (the SCM forcing is time-invariant, so GABLS1's
0.25 K/h cooling ramp is replaced by a fixed cold surface — the stable
signatures still form); assertions are qualitative regime signatures, not
LES-quantitative profiles. Marked ``slow`` (one ``lax.scan`` over the ECHAM
stack per case), matching the existing ECHAM-SCM test convention.

Array convention: ``SingleColumnModel`` column fields are 1-D with **index 0 =
model top** and **index -1 = lowest level above the surface**.
"""

from functools import lru_cache
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest
from dinosaur.sigma_coordinates import SigmaCoordinates

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.echam.echam_terms import echam_physics
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.single_column_model import SingleColumnModel
from jcm.terrain import TerrainData

pytestmark = pytest.mark.slow

P0 = float(c.p0)
RD = float(c.rd)
CPD = float(c.cpd)
G = float(c.grav)
DT = 60.0  # s


class GeostrophicCoriolisForcing(PhysicsTerm):
    """Large-scale Coriolis + geostrophic pressure-gradient forcing.

    ``du/dt = f (v - v_g)``, ``dv/dt = -f (u - u_g)``. This is the dynamics the
    ``SingleColumnModel`` deliberately omits; supplying it lets the wind-driven
    boundary-layer cases (Ekman spiral, low-level jet) form. ``f``/``u_g``/
    ``v_g`` are stored as plain floats (static config baked into the jit).
    """

    name: ClassVar[str] = "geostrophic_coriolis_forcing"
    category: ClassVar[str] = "large_scale_forcing"
    requires: ClassVar[tuple] = ()
    provides: ClassVar[tuple] = ()

    def __init__(self, f: float, u_g: float, v_g: float = 0.0):
        """Store the Coriolis parameter and geostrophic wind components."""
        self.f = float(f)
        self.u_g = float(u_g)
        self.v_g = float(v_g)

    def __call__(self, state, diagnostics, forcing, terrain):
        """Return the Coriolis/geostrophic wind tendency; pass diagnostics through."""
        du = self.f * (state.v_wind - self.v_g)
        dv = -self.f * (state.u_wind - self.u_g)
        tend = PhysicsTendency.zeros(state.temperature.shape).copy(u_wind=du, v_wind=dv)
        return tend, diagnostics


# ---------------------------------------------------------------------------
# Column / physics construction
# ---------------------------------------------------------------------------

def _stretched_sigma(n: int = 48, power: float = 1.5) -> SigmaCoordinates:
    """Sigma boundaries clustered near the surface for boundary-layer resolution.

    ``sigma = 1 - (1 - xi)**power`` maps a uniform ``xi`` to boundaries that bunch
    up toward ``sigma = 1`` (the surface), giving thin near-surface layers and
    coarse layers aloft.
    """
    xi = np.linspace(0.0, 1.0, n + 1)
    b = 1.0 - (1.0 - xi) ** power
    b[0], b[-1] = 0.0, 1.0
    return SigmaCoordinates(jnp.asarray(b))


def _build_state(sigma: SigmaCoordinates, theta, u_g: float, v_g: float):
    """1-D column ``PhysicsState`` from a potential-temperature profile.

    ``theta`` is either a scalar (neutral) or a callable ``theta(z_m)``. The
    column is dry. Geopotential is integrated hydrostatically from the surface.
    Returns ``(state, full_level_height_m)`` with index 0 = top, -1 = surface.
    """
    b = np.asarray(sigma.boundaries)
    sc = 0.5 * (b[:-1] + b[1:])
    n = sc.size
    p_full = sc * P0
    p_half = b * P0

    # First pass with an isothermal guess to get heights, then evaluate theta(z).
    t_guess = np.full(n, 270.0)
    z_half = np.zeros(n + 1)
    for k in range(n - 1, -1, -1):
        z_half[k] = z_half[k + 1] + (RD * t_guess[k] / G) * np.log(
            max(p_half[k + 1], 1.0) / max(p_half[k], 1.0))
    z_full = 0.5 * (z_half[:-1] + z_half[1:])

    if callable(theta):
        theta_prof = np.asarray(theta(z_full), float)
    else:
        theta_prof = np.full(n, float(theta))
    temperature = theta_prof * (p_full / P0) ** (RD / CPD)

    # Re-integrate geopotential with the real temperature.
    z_half = np.zeros(n + 1)
    for k in range(n - 1, -1, -1):
        z_half[k] = z_half[k + 1] + (RD * temperature[k] / G) * np.log(
            max(p_half[k + 1], 1.0) / max(p_half[k], 1.0))
    z_full = 0.5 * (z_half[:-1] + z_half[1:])

    state = PhysicsState(
        u_wind=jnp.full(n, u_g), v_wind=jnp.full(n, v_g),
        temperature=jnp.asarray(temperature), specific_humidity=jnp.zeros(n),
        geopotential=jnp.asarray(G * z_full),
        normalized_surface_pressure=jnp.asarray(1.0),
        tracers={"qc": jnp.zeros(n), "qi": jnp.zeros(n)},
    )
    return state, z_full


def _surface_plus_vdiff_physics(f: float, u_g: float, v_g: float):
    """``echam_physics()`` trimmed to surface + vdiff, plus the Coriolis forcing.

    Convection and microphysics are the terms that bury the BL signal, so they
    are removed (asserted); radiation/aerosol/etc. are removed best-effort so
    only the real ECHAM surface + vdiff act on the column.
    """
    phys = echam_physics(radiation_scheme="grey")
    for cat in ("convection", "clouds", "chemistry", "hines", "sso",
                "radiation", "aerosol", "cloud_fraction"):
        try:
            phys = phys.remove(cat)
        except Exception:
            pass  # benign if a dependency keeps it; convection/clouds checked below
    remaining = {t.category for t in phys.terms}
    assert "convection" not in remaining and "clouds" not in remaining, (
        f"convection/microphysics must be trimmed, got {sorted(remaining)}"
    )
    return phys + GeostrophicCoriolisForcing(f, u_g, v_g)


def _run(theta, sst_offset, f, u_g, v_g, nsteps, n=48):
    """Run one case through the existing ``SingleColumnModel`` (free-run column).

    ``sst_offset`` sets the constant ocean skin temperature relative to the
    lowest-level air temperature (>0 warm/convective, <0 cold/stable, 0 neutral).
    u/v/T/q free-run via the relaxation hook at a long (1e9 s) timescale.
    """
    sigma = _stretched_sigma(n=n)
    state, z_full = _build_state(sigma, theta, u_g, v_g)
    t_air = float(np.asarray(state.temperature)[-1])

    terrain = TerrainData.single_column(fmask=0.0)  # all ocean
    forcing = ForcingData.zeros((1, 1)).copy(
        sea_surface_temperature=jnp.full((1, 1), t_air + sst_offset))

    scm = SingleColumnModel(
        physics=_surface_plus_vdiff_physics(f, u_g, v_g),
        vertical=sigma, lat_deg=45.0, lon_deg=0.0,
        terrain=terrain, forcing=forcing, dt_seconds=DT,
        relaxation_timescales={"u_wind": 1e9, "v_wind": 1e9,
                               "temperature": 1e9, "specific_humidity": 1e9},
    )
    pred = scm.run([state] * nsteps)
    return pred, z_full


def _final(pred, name):
    """Final-step 1-D profile of a relaxed prognostic variable (index 0=top)."""
    return np.asarray(pred.relaxed_states[name])[-1]


def _potential_temperature(pred, step=-1):
    """Potential-temperature profile [K] at ``step`` (index 0=top, -1=surface)."""
    temps = np.asarray(pred.relaxed_states["temperature"])
    p = np.asarray(pred.physics_data["pressure_full"]).reshape(temps.shape)[step]
    return temps[step] * (P0 / p) ** (RD / CPD)


def _surface_flux(pred, field):
    """Final-step grid-box-mean surface flux scalar."""
    arr = np.asarray(getattr(pred.physics_data["surface"], field))
    return float(arr.reshape(arr.shape[0], -1)[-1, 0])


# ---------------------------------------------------------------------------
# Cached case rollouts (each runs once, shared across its test methods)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _ekman_result():
    """Andren neutral Ekman: 10 h, neutral surface (SST = air), Vg = 10 m/s."""
    u_g = 10.0
    pred, z = _run(theta=290.0, sst_offset=0.0, f=1.0e-4, u_g=u_g, v_g=0.0,
                   nsteps=600)
    u, v = _final(pred, "u_wind"), _final(pred, "v_wind")
    spd = np.hypot(u, v)
    return {
        "u_g": u_g, "z": z, "spd": spd,
        "u_surface": float(u[-1]), "v_surface": float(v[-1]),
        "spd_surface": float(spd[-1]),
        "spd_free": float(np.median(spd[: len(spd) // 2])),  # upper half (free air)
        "spd_max": float(spd.max()),
        "jet_height": float(z[int(np.argmax(spd))]),
        "angle_surface": float(np.degrees(np.arctan2(v[-1], u[-1]))),
    }


@lru_cache(maxsize=None)
def _gabls1_result():
    """GABLS1 stable: 9 h, cold surface (SST = air - 8 K), Vg = 8 m/s at 73 N."""
    u_g = 8.0
    pred, z = _run(
        theta=lambda zz: np.where(zz <= 100.0, 265.0, 265.0 + 0.01 * (zz - 100.0)),
        sst_offset=-8.0, f=1.39e-4, u_g=u_g, v_g=0.0, nsteps=540)
    u, v = _final(pred, "u_wind"), _final(pred, "v_wind")
    spd = np.hypot(u, v)
    temp = _final(pred, "temperature")
    tke = np.asarray(pred.physics_data["vertical_diffusion"].tke)[-1]
    return {
        "u_g": u_g, "z": z, "spd": spd,
        "spd_max": float(spd.max()),
        "jet_height": float(z[int(np.argmax(spd))]),
        "t_surface": float(temp[-1]), "t_above": float(temp[-4]),
        "max_tke": float(np.max(tke)),
        "sensible_heat_flux": _surface_flux(pred, "sensible_heat_flux"),
    }


@lru_cache(maxsize=None)
def _wangara_result():
    """Wangara convective: 6 h, warm surface (SST = air + 8 K), light wind."""
    pred, z = _run(
        theta=lambda zz: np.where(zz <= 100.0, 288.0, 288.0 + 0.004 * (zz - 100.0)),
        sst_offset=8.0, f=1.0e-4, u_g=3.0, v_g=0.0, nsteps=360)
    theta = _potential_temperature(pred)
    theta0 = _potential_temperature(pred, 0)
    tke = np.asarray(pred.physics_data["vertical_diffusion"].tke)[-1]
    # Mixed-layer depth: from just above the surface skin, the contiguous height
    # over which theta stays within 0.5 K of the layer value.
    i_ref = -3  # ~40-80 m, above the superadiabatic skin
    depth = 0.0
    for k in range(len(theta) - 1, -1, -1):
        if abs(theta[k] - theta[i_ref]) < 0.5:
            depth = z[k]
        else:
            break
    return {
        "z": z, "theta": theta,
        "theta_surface_initial": float(theta0[-1]),
        "theta_surface_final": float(theta[-1]),
        "mixed_layer_std": float(np.std(theta[i_ref - 6:i_ref + 1])),
        "mixed_layer_depth": depth,
        "max_tke": float(np.max(tke)),
        "sensible_heat_flux": _surface_flux(pred, "sensible_heat_flux"),
    }


# ---------------------------------------------------------------------------
# Andren (1994) — neutral Ekman spiral
# ---------------------------------------------------------------------------

class TestEkmanSpiralScm:
    """Neutral Ekman boundary layer via the existing SingleColumnModel."""

    def test_surface_wind_is_subgeostrophic_and_backed(self):
        """Surface drag leaves a sub-geostrophic wind backed toward low pressure.

        The defining Ekman signature: friction slows the near-surface wind below
        geostrophic and Coriolis turns it across the isobars (a positive ``v``
        develops from a purely zonal geostrophic wind).
        """
        r = _ekman_result()
        assert r["spd_surface"] < 0.9 * r["u_g"], (
            f"surface wind should be sub-geostrophic: |V|={r['spd_surface']:.2f} "
            f"vs Vg={r['u_g']}"
        )
        assert r["v_surface"] > 0.5, (
            f"surface wind should be backed (v>0), got v={r['v_surface']:.2f}"
        )
        assert 5.0 < r["angle_surface"] < 45.0, (
            f"cross-isobaric angle should be ~5-45 deg, got {r['angle_surface']:.1f}"
        )

    def test_wind_recovers_geostrophic_aloft(self):
        """Above the friction layer the wind returns to the geostrophic value."""
        r = _ekman_result()
        assert abs(r["spd_free"] - r["u_g"]) < 1.0, (
            f"free-atmosphere wind should approach Vg={r['u_g']}, got {r['spd_free']:.2f}"
        )

    def test_supergeostrophic_ekman_jet_forms(self):
        """A supergeostrophic jet sits in the lower boundary layer."""
        r = _ekman_result()
        assert r["spd_max"] > r["u_g"], (
            f"an Ekman jet should exceed geostrophic: max|V|={r['spd_max']:.2f} "
            f"vs Vg={r['u_g']}"
        )
        assert r["jet_height"] < 600.0, (
            f"the Ekman jet should sit low, got z={r['jet_height']:.0f} m"
        )


# ---------------------------------------------------------------------------
# GABLS1 / Cuxart (2006) — stable boundary layer
# ---------------------------------------------------------------------------

class TestStableLowLevelJetScm:
    """Stable boundary layer via the existing SingleColumnModel."""

    def test_surface_heat_flux_is_downward(self):
        """A cold surface drives a downward (negative) sensible heat flux."""
        r = _gabls1_result()
        assert r["sensible_heat_flux"] < 0.0, (
            f"cold surface should give downward heat flux, got "
            f"{r['sensible_heat_flux']:.2f} W/m^2"
        )

    def test_surface_inversion_develops(self):
        """Surface cooling builds a temperature inversion (T increases upward)."""
        r = _gabls1_result()
        assert r["t_above"] > r["t_surface"], (
            f"a surface inversion should form: T(surface)={r['t_surface']:.2f} "
            f"< T(aloft)={r['t_above']:.2f}"
        )

    def test_nocturnal_low_level_jet_forms(self):
        """Decoupling aloft + inertial turning build a supergeostrophic low jet."""
        r = _gabls1_result()
        assert r["spd_max"] > r["u_g"], (
            f"a supergeostrophic LLJ should form: max|V|={r['spd_max']:.2f} "
            f"vs Vg={r['u_g']}"
        )
        assert r["jet_height"] < 400.0, (
            f"the nocturnal jet should be low, got z={r['jet_height']:.0f} m"
        )

    def test_turbulence_is_weak(self):
        """Stable stratification keeps the TKE small."""
        r = _gabls1_result()
        assert r["max_tke"] < 1.0, (
            f"stable-BL TKE should stay small, got {r['max_tke']:.3f}"
        )


# ---------------------------------------------------------------------------
# Wangara Day 33 — convective boundary layer
# ---------------------------------------------------------------------------

class TestConvectiveMixedLayerScm:
    """Convective boundary layer via the existing SingleColumnModel."""

    def test_surface_heat_flux_is_upward(self):
        """A warm surface drives an upward (positive) sensible heat flux."""
        r = _wangara_result()
        assert r["sensible_heat_flux"] > 0.0, (
            f"warm surface should give upward heat flux, got "
            f"{r['sensible_heat_flux']:.2f} W/m^2"
        )

    def test_near_surface_warms(self):
        """Surface heating warms the near-surface potential temperature."""
        r = _wangara_result()
        assert r["theta_surface_final"] > r["theta_surface_initial"], (
            f"surface heating should warm theta: "
            f"{r['theta_surface_initial']:.2f} -> {r['theta_surface_final']:.2f} K"
        )

    def test_convective_layer_is_well_mixed(self):
        """A deep, near-uniform-theta convective mixed layer forms."""
        r = _wangara_result()
        assert r["mixed_layer_std"] < 0.5, (
            f"the convective layer should be well mixed in theta, "
            f"std={r['mixed_layer_std']:.3f} K"
        )
        assert r["mixed_layer_depth"] > 200.0, (
            f"a convective mixed layer should grow deep, "
            f"got {r['mixed_layer_depth']:.0f} m"
        )

    def test_convective_turbulence_present(self):
        """Buoyant production sustains appreciable TKE — clearly above the floor.

        The default surface-layer scheme is the faithful ECHAM-Louis form, whose
        unstable-regime heat exchange is more moderate (and more physical) than
        the old Businger-Dyer ``(1-16Ri)^0.5`` enhancement. With the surface
        exchange now solved as the bottom boundary row of the vdiff implicit
        column solve (reported flux == delivered flux, no separate imp_heat
        damping), this +8 K air-sea contrast delivers ~24 W/m^2 of sensible
        heat flux (the pre-coupling *reported* value was ~28 W/m^2, but that
        was the undamped bulk flux, of which only the imp_heat fraction ever
        reached the column) and the buoyantly-produced convective TKE peaks
        around 0.031 m^2/s^2. That is a clear convective signal — ~3x the
        0.01 m^2/s^2 quiescent floor — and the primary convective signatures
        asserted above (a well-mixed deepening layer, an upward surface heat
        flux, near-surface warming) all form. The threshold distinguishes
        appreciable, buoyancy-driven TKE from a collapse to the floor; it is
        deliberately not pinned to the Businger-Dyer magnitude the earlier
        0.05 value was tuned to.
        """
        r = _wangara_result()
        assert r["max_tke"] > 0.02, (
            f"convective TKE should be appreciable (clearly above the ~0.01 "
            f"floor), got {r['max_tke']:.3f}"
        )
