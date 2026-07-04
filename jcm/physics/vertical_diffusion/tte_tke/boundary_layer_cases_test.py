"""Qualitative boundary-layer physics tests for the TTE-TKE vdiff scheme.

These tests are distilled from the three classic single-column boundary-layer
validation cases used by the ``jax_scm`` project
(https://github.com/mpierzyna/jax_scm/tree/main/validation):

  * **Andren et al. (1994)** — a *neutral* Ekman boundary layer
    (https://doi.org/10.1002/qj.49712052003). Headline behaviours: TKE spins
    up to a steady state, the surface-layer length scale is ``l = kappa*z``
    (the ingredient behind ``Phi_M -> 1``), and the neutral exchange
    coefficient has the classic boundary-layer profile.
  * **GABLS1 / Cuxart et al. (2006)** — a *stable* nocturnal boundary layer
    (https://doi.org/10.1007/s10546-005-3780-1). Headline behaviours: stable
    stratification sharply suppresses turbulent mixing and confines it to a
    shallow near-surface layer; strong stratification collapses turbulence;
    buoyancy is a TKE sink.
  * **Wangara Day 33 / Yamada & Mellor** — a *convective* daytime boundary
    layer. Headline behaviours: buoyancy is the dominant TKE *source*, mixing
    is enhanced relative to neutral, and turbulence is sustained by buoyancy
    even without wind shear.

Scope — what these tests pin
----------------------------
The ``jax_scm`` validation runs are *full* single-column models: they solve
the momentum equations with Coriolis + geostrophic forcing and apply
Monin-Obukhov *surface fluxes*. The JCM ``TteTkeVerticalDiffusion`` term now
carries the surface exchange as the bottom-row Robin boundary condition of
its implicit solve (the ECHAM-faithful coupling): by default it applies real
surface drag and surface heat/moisture fluxes to the prognostic state, and
the delivered fluxes equal the column-integrated tendencies by construction
(see ``vertical_diffusion_test.TestSurfaceCoupledSolve``).

This file, however, tests the *interior diffusion operator in isolation*:
every run here passes ``couple_surface=False`` (via ``_vdc`` below), which
zeroes the surface exchange input of the solve and restores the zero-flux
(insulating, free-slip) boundaries. The boundary condition is now an INPUT
of the solve — zeroed here on purpose — so these classes pin the
*turbulence-closure* physics the term owns (mixing lengths, TKE budget,
stability functions, down-gradient conservative mixing), flavoured by the
three regimes above. That is exactly the part the canonical cases stress,
and where the previous tests were weakest (mostly bounds / finiteness
checks).

The companion **surface-coupled** cases — the neutral Ekman spiral + jet
(Andren), the nocturnal low-level jet + surface inversion (GABLS1), and the
growing convective mixed layer (Wangara) — live in
``scm_boundary_layer_cases_test.py``. Those drive the real
:class:`~jcm.single_column_model.SingleColumnModel` (ECHAM surface + vdiff terms
plus a Coriolis / geostrophic forcing), which is what it takes to make the
surface-coupled signatures emerge.

Grid / array convention
-----------------------
The scheme indexes columns as ``(ncol, nlev)`` with **level 0 = model top**
and **level nlev-1 = lowest level above the surface** (heights *decrease*
with index; the surface half-level is ``height_half[:, -1]``). The
``_make_column`` helper takes intuitive *surface-first* profiles and flips
them into this top-first convention, so the tests below can be written in
"ground up" terms.
"""

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from .vertical_diffusion_types import VDiffParameters, VDiffState
from .vertical_diffusion import vertical_diffusion_column
from .turbulence_coefficients import (
    compute_richardson_number,
    compute_mixing_length,
    compute_exchange_coefficients,
)
from .tke_budget import compute_tke_diagnostics

KAPPA = float(c.karman_const)
G = float(c.grav)
CPD = float(c.cpd)
RD = float(c.rd)
P0 = float(c.p0)


def _vdc(state, params, dt):
    """Interior-only vdiff column: surface Robin BC zeroed (see module doc)."""
    return vertical_diffusion_column(state, params, dt, couple_surface=False)


# ---------------------------------------------------------------------------
# Column construction helpers
# ---------------------------------------------------------------------------

def _stretched_grid(ztop: float = 3000.0, nlev: int = 40, dz0: float = 10.0):
    """Surface-first, ascending half-level heights with fine near-surface spacing.

    A geometric stretch (thin layers near the ground, thicker aloft) is the
    natural choice for boundary-layer columns. Returns an array of length
    ``nlev + 1`` running from 0 m at the surface to ~``ztop`` at the model top.
    """
    def total(ratio):
        if abs(ratio - 1.0) < 1e-9:
            return dz0 * nlev
        return dz0 * (ratio ** nlev - 1.0) / (ratio - 1.0)

    lo, hi = 1.0, 1.3
    for _ in range(100):  # bisection for the stretch ratio that hits ztop
        mid = 0.5 * (lo + hi)
        if total(mid) < ztop:
            lo = mid
        else:
            hi = mid
    ratio = 0.5 * (lo + hi)
    dz = dz0 * ratio ** np.arange(nlev)
    return np.concatenate([[0.0], np.cumsum(dz)])


def _make_column(
    theta,
    u,
    v,
    z_half=None,
    qv=None,
    tke0: float = 0.5,
    surface_temp_offset: float = 0.0,
) -> VDiffState:
    """Build a single-column ``VDiffState`` from surface-first profiles.

    Parameters
    ----------
    theta, u, v : array (nlev,)
        Potential temperature [K] and winds [m/s] on full levels, ordered
        *surface-first*. Potential temperature is the intuitive control for
        stability: constant ``theta`` is neutral, increasing-with-height is
        stable, decreasing-with-height is unstable.
    z_half : array (nlev+1,), optional
        Surface-first ascending half-level heights. Defaults to
        ``_stretched_grid()``.
    qv : array (nlev,), optional
        Water-vapour mixing ratio [kg/kg]. Defaults to dry.
    tke0 : float
        Uniform initial TKE [m^2/s^2].
    surface_temp_offset : float
        Offset added to the lowest-level air temperature to set the surface
        tile temperature. Only affects the (decoupled) surface diagnostics,
        not the prognostic evolution; left at 0 (neutral surface) by default.

    """
    if z_half is None:
        z_half = _stretched_grid()
    z_half = np.asarray(z_half, float)
    nlev = len(z_half) - 1
    z_full = 0.5 * (z_half[:-1] + z_half[1:])

    theta = np.asarray(theta, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    qv = np.zeros(nlev) if qv is None else np.asarray(qv, float)

    # Hydrostatic pressure, integrated upward from a 1000 hPa surface, then
    # convert potential temperature -> temperature via the Exner function.
    p_half = np.zeros(nlev + 1)
    p_half[0] = P0
    for k in range(nlev):
        dz = z_half[k + 1] - z_half[k]
        t_k = theta[k] * (p_half[k] / P0) ** (RD / CPD)
        rho = p_half[k] / (RD * t_k)
        p_half[k + 1] = p_half[k] - rho * G * dz
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    t_full = theta * (p_full / P0) ** (RD / CPD)

    flip = lambda a: np.asarray(a)[::-1].copy()  # surface-first -> top-first
    col = lambda a: jnp.asarray(a)[None, :]      # add the ncol axis

    z_half_tf = flip(z_half)
    z_full_tf = flip(z_full)
    p_half_tf = flip(p_half)
    p_full_tf = flip(p_full)
    t_tf = flip(t_full)

    ncol, nsfc = 1, 3
    air_mass = jnp.abs(jnp.diff(col(p_half_tf), axis=1)) / G

    sfc_t = float(t_full[0]) + surface_temp_offset
    surface_temperature = jnp.full((ncol, nsfc), sfc_t)
    surface_fraction = jnp.zeros((ncol, nsfc)).at[:, 2].set(1.0)  # all land
    roughness_length = jnp.full((ncol, nsfc), 0.1)

    return VDiffState(
        u=col(flip(u)), v=col(flip(v)), temperature=col(t_tf),
        qv=col(flip(qv)), qc=jnp.zeros((ncol, nlev)), qi=jnp.zeros((ncol, nlev)),
        pressure_full=col(p_full_tf), pressure_half=col(p_half_tf),
        geopotential=col(z_full_tf) * G,
        air_mass=air_mass,
        surface_temperature=surface_temperature,
        surface_fraction=surface_fraction,
        roughness_length=roughness_length,
        roughness_heat=0.1 * roughness_length,
        surface_wetness=jnp.ones((ncol, nsfc)),
        height_full=col(z_full_tf), height_half=col(z_half_tf),
        tke=jnp.full((ncol, nlev), tke0),
        thv_variance=jnp.zeros((ncol, nlev)),
        ocean_u=jnp.zeros(ncol), ocean_v=jnp.zeros(ncol),
    )


def _integrate(state, params, dt, nsteps, evolve_winds=False, evolve_temp=False,
               freeze_winds=False):
    """Advance a column ``nsteps`` times applying the vdiff tendencies.

    TKE is always evolved (and floored at the 0.01 m^2/s^2 ECHAM bound, as the
    composable term does). ``u``/``v``/``T`` are evolved only when requested,
    so diagnostics-only tests can let TKE adjust to the regime while holding
    the mean state fixed. ``freeze_winds`` re-imposes the initial winds each
    step to probe TKE equilibrium under *steady* shear forcing.
    """
    u0, v0 = state.u, state.v
    s = state
    for _ in range(nsteps):
        tend, _ = _vdc(s, params, dt)
        updates = {"tke": jnp.maximum(s.tke + dt * tend.tke_tendency, 0.01)}
        if evolve_winds:
            updates["u"] = s.u + dt * tend.u_tendency
            updates["v"] = s.v + dt * tend.v_tendency
        if evolve_temp:
            updates["temperature"] = s.temperature + dt * tend.temperature_tendency
        s = s._replace(**updates)
        if freeze_winds:
            s = s._replace(u=u0, v=v0)
    return s


def _closure_diagnostics(state, params):
    """Recompute Ri, mixing length, exchange coefficients and the TKE-budget
    terms exactly as ``vertical_diffusion_column`` does internally.

    Returns ``(ri, mixing_length, km, kh, kq, shear_prod, buoy_prod, diss)``,
    all as plain numpy arrays for the single column (top-first ordering).
    """
    ri = compute_richardson_number(
        state.u, state.v, state.temperature, state.height_full, state.height_half,
    )
    pbl_guess = jnp.full(state.u.shape[0], 1000.0)  # matches the scheme's guess
    ml = compute_mixing_length(state.height_full, state.height_half, ri, pbl_guess)
    km, kh, kq = compute_exchange_coefficients(state, params, ml, ri)
    shear_prod, buoy_prod, diss, _ = compute_tke_diagnostics(state, params, km, kh, ml)
    out = (ri, ml, km, kh, kq, shear_prod, buoy_prod, diss)
    return tuple(np.asarray(a[0]) for a in out)


# Reusable canonical profiles. A modest linear jet (0 -> 12 m/s over the
# domain) seeds shear-driven turbulence; theta sets the stability regime.
def _neutral_column(jet=12.0, tke0=0.5, nlev=40, **kw):
    z = _stretched_grid(nlev=nlev, **kw)
    n = len(z) - 1
    theta = np.full(n, 290.0)
    return _make_column(theta, np.linspace(0.0, jet, n), np.zeros(n), z, tke0=tke0)


def _stable_column(jet=12.0, dthdz=0.01, tke0=0.5, nlev=40, **kw):
    z = _stretched_grid(nlev=nlev, **kw)
    n = len(z) - 1
    theta = 290.0 + dthdz * 0.5 * (z[:-1] + z[1:])
    return _make_column(theta, np.linspace(0.0, jet, n), np.zeros(n), z, tke0=tke0)


def _unstable_column(jet=12.0, dthdz=-0.004, tke0=0.5, nlev=40, **kw):
    z = _stretched_grid(nlev=nlev, **kw)
    n = len(z) - 1
    theta = 290.0 + dthdz * 0.5 * (z[:-1] + z[1:])
    return _make_column(theta, np.linspace(0.0, jet, n), np.zeros(n), z, tke0=tke0)


DT = 60.0


# ---------------------------------------------------------------------------
# Andren (1994) — neutral Ekman boundary layer
# ---------------------------------------------------------------------------

class TestNeutralBoundaryLayerAndren:
    """Neutral-regime behaviours (Andren et al. 1994)."""

    def test_tke_spins_up_to_steady_state(self):
        """TKE grows from a tiny seed and plateaus under steady neutral shear.

        Andren Fig. 2 tracks the vertically-integrated TKE approaching a
        steady value. With surface forcing held fixed, a healthy closure must
        reach production = dissipation equilibrium rather than grow without
        bound or decay away.
        """
        params = VDiffParameters.default()
        state = _neutral_column(jet=12.0, tke0=0.01)

        mid = _integrate(state, params, DT, 80, freeze_winds=True)
        late = _integrate(mid, params, DT, 70, freeze_winds=True)

        tke_mid = float(jnp.max(mid.tke))
        tke_late = float(jnp.max(late.tke))

        assert tke_late > 0.2, (
            f"TKE failed to spin up from the 0.01 seed: max={tke_late:.4f}"
        )
        # Plateau: the last 70 steps should barely change the maximum.
        assert abs(tke_late - tke_mid) / tke_mid < 0.1, (
            f"TKE not at steady state: step80 max={tke_mid:.4f}, "
            f"step150 max={tke_late:.4f}"
        )

    def test_equilibrium_tke_scales_with_shear_squared(self):
        """Doubling the shear roughly quadruples the equilibrium TKE.

        At production = dissipation the closure gives ``e ~ (l*S)^2``, i.e.
        equilibrium TKE scales with shear squared. This is the signature of a
        properly coupled ``K = c*l*sqrt(e)`` closure (a decoupled scheme would
        not show this clean scaling).
        """
        params = VDiffParameters.default()
        weak = _integrate(_neutral_column(jet=8.0, tke0=0.01),
                          params, DT, 150, freeze_winds=True)
        strong = _integrate(_neutral_column(jet=16.0, tke0=0.01),
                            params, DT, 150, freeze_winds=True)

        ratio = float(jnp.max(strong.tke)) / float(jnp.max(weak.tke))
        assert 2.5 < ratio < 5.5, (
            f"Equilibrium TKE should scale ~quadratically with shear; "
            f"TKE(2S)/TKE(S) = {ratio:.2f} (expected ~4)"
        )

    def test_surface_mixing_length_is_kappa_z(self):
        """The lowest-level mixing length equals the von Karman length kappa*z.

        ``l = kappa*z`` in the surface layer is what makes the non-dimensional
        wind shear ``Phi_M -> 1`` (Andren Fig. 4a) and underpins the log-law.
        """
        params = VDiffParameters.default()
        state = _neutral_column(jet=10.0, tke0=0.5)
        _, diag = _vdc(state, params, DT)

        ml = np.asarray(diag.mixing_length[0])              # top-first
        z_above_sfc = float(state.height_full[0, -1] - state.height_half[0, -1])
        l_surface = float(ml[-1])                            # lowest level

        assert abs(l_surface / (KAPPA * z_above_sfc) - 1.0) < 0.05, (
            f"Surface mixing length should be kappa*z = {KAPPA * z_above_sfc:.3f} m, "
            f"got {l_surface:.3f} m"
        )

    def test_neutral_diffusivity_has_boundary_layer_profile(self):
        """Neutral K_m rises from the surface, peaks aloft, then decreases.

        Andren Figs 6a/6b show momentum-flux (and hence diffusivity) profiles
        that peak in the lower-mid boundary layer rather than at the surface or
        the top — the classic neutral-PBL shape.
        """
        params = VDiffParameters.default()
        state = _integrate(_neutral_column(jet=12.0, tke0=0.01),
                           params, DT, 60, freeze_winds=True)
        _, diag = _vdc(state, params, DT)

        km_sf = np.asarray(diag.exchange_coeff_momentum[0])[::-1]  # surface-first
        nlev = km_sf.size
        peak = int(np.argmax(km_sf))

        assert 2 <= peak <= nlev - 3, (
            f"K_m should peak in the interior, not at a boundary (peak index "
            f"{peak} of {nlev})"
        )
        assert km_sf[0] < km_sf[peak], "K_m at the surface should be below the peak"
        assert km_sf[-1] < km_sf[peak], "K_m at the model top should be below the peak"


# ---------------------------------------------------------------------------
# GABLS1 / Cuxart (2006) — stable boundary layer
# ---------------------------------------------------------------------------

class TestStableBoundaryLayerGabls1:
    """Stable-regime behaviours (GABLS1 / Cuxart et al. 2006)."""

    def test_stable_stratification_suppresses_mixing(self):
        """Stable stratification sharply reduces K and TKE versus neutral.

        With the same shear forcing, the stable column's mixing must be far
        weaker than the neutral one — the reason the GABLS1 boundary layer is
        shallow and weakly turbulent.
        """
        params = VDiffParameters.default()
        neutral = _integrate(_neutral_column(jet=12.0), params, DT, 30)
        stable = _integrate(_stable_column(jet=12.0, dthdz=0.01), params, DT, 30)

        km_neutral = float(jnp.max(_closure_diagnostics(neutral, params)[2]))
        km_stable = float(jnp.max(_closure_diagnostics(stable, params)[2]))
        tke_neutral = float(jnp.max(neutral.tke))
        tke_stable = float(jnp.max(stable.tke))

        assert km_neutral > 5.0 * km_stable, (
            f"Stable stratification should strongly suppress K_m: "
            f"neutral={km_neutral:.3f}, stable={km_stable:.3f}"
        )
        assert tke_neutral > 5.0 * tke_stable, (
            f"Stable stratification should suppress TKE: "
            f"neutral={tke_neutral:.4f}, stable={tke_stable:.4f}"
        )

    def test_turbulent_layer_is_shallow_when_stable(self):
        """Turbulence is confined to a shallow layer under strong stability.

        Counting levels with appreciable momentum diffusivity (K_m > 1 m^2/s),
        the neutral boundary layer fills much of the column while the stable
        one has essentially none — the qualitative GABLS1 "shallow stable BL".
        """
        params = VDiffParameters.default()
        neutral = _integrate(_neutral_column(jet=12.0), params, DT, 30)
        stable = _integrate(_stable_column(jet=12.0, dthdz=0.01), params, DT, 30)

        km_neutral = _closure_diagnostics(neutral, params)[2]
        km_stable = _closure_diagnostics(stable, params)[2]
        n_turb_neutral = int(np.sum(km_neutral > 1.0))
        n_turb_stable = int(np.sum(km_stable > 1.0))

        assert n_turb_neutral >= 10, (
            f"Neutral turbulent layer should be deep ({n_turb_neutral} levels)"
        )
        assert n_turb_stable <= 2, (
            f"Stable turbulent layer should be shallow ({n_turb_stable} levels)"
        )

    def test_strong_stability_collapses_turbulence(self):
        """Strong stratification + weak shear drives TKE back to the floor.

        Seeded with elevated TKE, a strongly stable column with little shear
        should see buoyancy destruction win and TKE decay to the 0.01 m^2/s^2
        floor — the nocturnal "turbulence collapse" GABLS1 probes.
        """
        params = VDiffParameters.default()
        z = _stretched_grid()
        zf = 0.5 * (z[:-1] + z[1:])
        n = len(z) - 1
        # Very stable (3 K / 100 m) with only weak shear (5 -> 7 m/s).
        state = _make_column(290.0 + 0.03 * zf, np.linspace(5.0, 7.0, n),
                             np.zeros(n), z, tke0=1.0)

        evolved = _integrate(state, params, DT, 40)
        assert float(jnp.max(evolved.tke)) < 0.1, (
            f"TKE should collapse toward the floor under strong stability, "
            f"got max={float(jnp.max(evolved.tke)):.4f}"
        )

    def test_buoyancy_is_a_tke_sink_when_stable(self):
        """Buoyancy production is negative (a sink) for stable stratification."""
        params = VDiffParameters.default()
        state = _stable_column(jet=10.0, dthdz=0.01, tke0=0.5)
        buoy_prod = _closure_diagnostics(state, params)[6]

        # Sign depends only on -K_h * N^2 with N^2 > 0 everywhere here.
        assert np.all(buoy_prod <= 1e-9), (
            f"Buoyancy production should be <= 0 when stable, "
            f"max={buoy_prod.max():.3e}"
        )


# ---------------------------------------------------------------------------
# Wangara Day 33 — convective boundary layer
# ---------------------------------------------------------------------------

class TestConvectiveBoundaryLayerWangara:
    """Convective-regime behaviours (Wangara Day 33)."""

    def test_buoyancy_is_a_tke_source_when_unstable(self):
        """Buoyancy production is positive (a source) for unstable stratification.

        Wangara Fig. 6 shows buoyancy as the dominant TKE production term in the
        daytime convective boundary layer.
        """
        params = VDiffParameters.default()
        state = _unstable_column(jet=10.0, dthdz=-0.004, tke0=0.5)
        buoy_prod = _closure_diagnostics(state, params)[6]

        assert np.all(buoy_prod >= -1e-9), (
            f"Buoyancy production should be >= 0 when unstable, "
            f"min={buoy_prod.min():.3e}"
        )

    def test_unstable_enhances_mixing_vs_neutral(self):
        """Unstable stratification gives mixing at least as vigorous as neutral."""
        params = VDiffParameters.default()
        neutral = _integrate(_neutral_column(jet=12.0), params, DT, 30)
        unstable = _integrate(_unstable_column(jet=12.0, dthdz=-0.004), params, DT, 30)

        km_neutral = float(jnp.max(_closure_diagnostics(neutral, params)[2]))
        km_unstable = float(jnp.max(_closure_diagnostics(unstable, params)[2]))

        assert km_unstable > km_neutral, (
            f"Unstable mixing should exceed neutral: "
            f"unstable K_m={km_unstable:.3f}, neutral K_m={km_neutral:.3f}"
        )
        assert float(jnp.max(unstable.tke)) > float(jnp.max(neutral.tke)), (
            "Unstable TKE should exceed neutral TKE"
        )

    def test_convective_turbulence_sustained_without_shear(self):
        """Buoyancy alone keeps a sheared-free unstable column turbulent.

        With zero wind shear, a neutral column's TKE decays to the floor, but
        an unstable one stays turbulent because buoyancy production sustains it
        — the defining feature of free convection.
        """
        params = VDiffParameters.default()
        z = _stretched_grid()
        zf = 0.5 * (z[:-1] + z[1:])
        n = len(z) - 1
        no_wind = (np.zeros(n), np.zeros(n))

        unstable = _make_column(290.0 - 0.004 * zf, *no_wind, z, tke0=0.5)
        neutral = _make_column(np.full(n, 290.0), *no_wind, z, tke0=0.5)

        unstable = _integrate(unstable, params, DT, 40)
        neutral = _integrate(neutral, params, DT, 40)

        tke_unstable = float(jnp.max(unstable.tke))
        tke_neutral = float(jnp.max(neutral.tke))
        assert tke_unstable > 0.05, (
            f"Free convection should sustain TKE without shear, got {tke_unstable:.4f}"
        )
        assert tke_unstable > 3.0 * tke_neutral, (
            f"Unstable (buoyancy-driven) TKE {tke_unstable:.4f} should far exceed "
            f"neutral no-shear TKE {tke_neutral:.4f}"
        )


# ---------------------------------------------------------------------------
# Cross-cutting closure invariants (what unifies the three cases)
# ---------------------------------------------------------------------------

class TestTurbulenceClosureInvariants:
    """Invariants of the Mellor-Yamada-2.5 closure across stability regimes."""

    def test_stability_ordering_of_diffusivity_and_tke(self):
        """Peak K_m and peak TKE order as unstable >= neutral > stable.

        This single ordering is the physics that distinguishes the three
        canonical cases (convective / neutral / stable) and is the heart of a
        stability-dependent closure.
        """
        params = VDiffParameters.default()
        cols = {
            "unstable": _integrate(_unstable_column(jet=12.0), params, DT, 30),
            "neutral": _integrate(_neutral_column(jet=12.0), params, DT, 30),
            "stable": _integrate(_stable_column(jet=12.0), params, DT, 30),
        }
        km = {k: float(jnp.max(_closure_diagnostics(s, params)[2]))
              for k, s in cols.items()}
        tke = {k: float(jnp.max(s.tke)) for k, s in cols.items()}

        assert km["unstable"] >= km["neutral"] > km["stable"], f"K_m ordering: {km}"
        assert tke["unstable"] >= tke["neutral"] > tke["stable"], f"TKE ordering: {tke}"
        assert km["neutral"] > 3.0 * km["stable"], f"K_m contrast too weak: {km}"

    def test_turbulent_prandtl_number_is_constant(self):
        """K_m / K_h = c_m / c_h = 0.8 wherever the closure is active.

        The scheme uses ``K_m = 0.4*l*sqrt(e)`` and ``K_h = 0.5*l*sqrt(e)``,
        so the turbulent Prandtl number is a constant 0.8 in the turbulent
        interior (away from the background-diffusivity clip).
        """
        params = VDiffParameters.default()
        state = _integrate(_neutral_column(jet=14.0, tke0=0.01),
                           params, DT, 40, freeze_winds=True)
        _, _, km, kh, _, _, _, _ = _closure_diagnostics(state, params)

        active = (km > 0.05) & (km < 999.0)  # exclude clipped background/ceiling
        ratio = km[active] / kh[active]
        assert np.allclose(ratio, 0.8, atol=0.01), (
            f"Turbulent Prandtl number K_m/K_h should be 0.8, got "
            f"range [{ratio.min():.3f}, {ratio.max():.3f}]"
        )

    def test_shear_and_dissipation_are_nonnegative(self):
        """Shear production and dissipation are sign-definite (>= 0).

        Shear production ``K_m * S^2`` and dissipation ``c_d * e^1.5 / l`` can
        never be negative; only buoyancy production changes sign with stability.
        """
        params = VDiffParameters.default()
        for col in (_neutral_column(), _stable_column(), _unstable_column()):
            s = _integrate(col, params, DT, 20)
            _, _, _, _, _, shear_prod, _, diss = _closure_diagnostics(s, params)
            assert np.all(shear_prod >= -1e-12), "shear production went negative"
            assert np.all(diss >= -1e-12), "dissipation went negative"


# ---------------------------------------------------------------------------
# Interior operator: down-gradient mixing + zero-flux conservation
# ---------------------------------------------------------------------------

class TestInteriorOperatorConservation:
    """The term is a conservative, down-gradient interior diffusion operator."""

    def test_uniform_field_is_left_unchanged(self):
        """A vertically uniform wind produces ~zero tendency (no spurious flux).

        Diffusion of a constant field is exactly zero, and with zero-flux
        boundaries there is no surface drag to spin a uniform wind down. The
        tolerance is set for float32 round-off in the implicit solve (residual
        ~1e-8 m/s^2 on an 8 m/s field), still ~1e5x below the real mixing
        tendencies (~1e-3 m/s^2) seen when a genuine shear is present.
        """
        params = VDiffParameters.default()
        z = _stretched_grid()
        n = len(z) - 1
        state = _make_column(np.full(n, 290.0), np.full(n, 8.0), np.zeros(n),
                             z, tke0=0.5)
        tend, _ = _vdc(state, params, DT)
        assert float(jnp.max(jnp.abs(tend.u_tendency))) < 1e-6, (
            "A uniform wind should not be mixed (no interior gradient, no drag)"
        )

    def test_momentum_is_conserved_and_mixes_down_gradient(self):
        """Column momentum is conserved while an internal jet mixes out.

        Down-gradient momentum transport reduces the shear over time, and the
        zero-flux boundaries conserve the air-mass-weighted column momentum.
        """
        params = VDiffParameters.default()
        state = _neutral_column(jet=20.0, tke0=0.5)
        air_mass = np.asarray(state.air_mass[0])
        mom0 = float(np.sum(air_mass * np.asarray(state.u[0])))
        shear0 = float(np.ptp(np.asarray(state.u[0])))

        evolved = _integrate(state, params, DT, 120, evolve_winds=True)
        mom1 = float(np.sum(air_mass * np.asarray(evolved.u[0])))
        shear1 = float(np.ptp(np.asarray(evolved.u[0])))

        assert abs(mom1 / mom0 - 1.0) < 1e-3, (
            f"Column momentum should be conserved, ratio={mom1 / mom0:.6f}"
        )
        assert shear1 < 0.9 * shear0, (
            f"Down-gradient mixing should reduce the shear: {shear0:.2f} -> {shear1:.2f}"
        )

    def test_heat_content_is_conserved(self):
        """Air-mass-weighted heat content is conserved by interior heat diffusion.

        With insulating boundaries the heat operator only redistributes energy,
        so ``sum(air_mass * T)`` is invariant (no surface heat flux into or out
        of the column).
        """
        params = VDiffParameters.default()
        z = _stretched_grid()
        n = len(z) - 1
        theta = np.full(n, 290.0)
        theta[n // 2] += 6.0  # a warm anomaly to be diffused
        state = _make_column(theta, np.linspace(2.0, 10.0, n), np.zeros(n),
                             z, tke0=1.0)
        air_mass = np.asarray(state.air_mass[0])
        heat0 = float(np.sum(air_mass * np.asarray(state.temperature[0])))

        evolved = _integrate(state, params, DT, 80, evolve_temp=True, evolve_winds=True)
        heat1 = float(np.sum(air_mass * np.asarray(evolved.temperature[0])))

        assert abs(heat1 / heat0 - 1.0) < 1e-3, (
            f"Heat content should be conserved, ratio={heat1 / heat0:.6f}"
        )


if __name__ == "__main__":
    import sys
    import inspect

    failures = 0
    for name, obj in list(globals().items()):
        if inspect.isclass(obj) and name.startswith("Test"):
            inst = obj()
            for meth in dir(inst):
                if meth.startswith("test_"):
                    try:
                        getattr(inst, meth)()
                        print(f"  PASS {name}.{meth}")
                    except AssertionError as e:
                        failures += 1
                        print(f"  FAIL {name}.{meth}: {e}")
    print("ALL PASSED" if not failures else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
