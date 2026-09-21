"""Simple single-wave mountain-wave drag (the fallback GWD scheme).

A cheap single monochromatic gravity-wave drag: a linear mountain-wave stress
launched at the surface opposite the low-level wind, propagated upward under the
McFarlane (1987) / Palmer et al. (1986) saturation hypothesis, and absorbed at
critical levels. It is a deliberately minimal stand-in for the full Hines +
Lott-Miller stack (``gravity_waves/{hines,sso}``), kept for aquaplanet tests;
the blocked-flow/form-drag physics is Lott-Miller's job, not this scheme's.

References:
- McFarlane, N. M. (1987), *J. Atmos. Sci.* 44, 1775-1800 (saturated orographic
  gravity-wave drag).
- Palmer, T. N., Shutts, G. J., Swinbank, R. (1986), *QJRMS* 112, 1001-1039.

"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple, Optional
import tree_math

import jcm.constants as c


@tree_math.struct
class SimpleGwdParameters:
    """Parameters for the simple monochromatic mountain-wave drag scheme.

    The scheme is a single-wave orographic drag with McFarlane (1987)
    saturation, so the tunables are the linear-wave stress efficiency, the
    representative horizontal wavenumber that turns the wave amplitude into a
    stress, the blocked-flow Froude taper, a launch multiplier and the height
    window over which drag is applied. (The Richardson-number/amplitude
    breaking knobs of the earlier placeholder are gone: saturation of the flux
    during upward propagation now decides where momentum is deposited, so those
    parameters had nothing to act on.)
    """

    gkdrag: float           # Linear-wave stress efficiency G (dimensionless)
    kwave: float            # Representative horizontal wavenumber k (m^-1)
    gwdrag_cd: float        # Launch-stress multiplier
    zmin: float             # No drag below this height (m)
    zmax: float             # No drag above this height (m)

    @classmethod
    def default(cls, gkdrag=0.1, kwave=1.0e-4,
                gwdrag_cd=1.0, zmin=1000.0, zmax=100000.0
                ) -> 'SimpleGwdParameters':
        """Return default gravity wave parameters.

        ``kwave = 1e-4 m^-1`` is a ~63 km representative sub-grid mountain-wave
        horizontal wavelength; with the surface density it is the ``rho_s k``
        factor the earlier stress formula was missing (which is why that formula
        produced ~10^2 Pa launch stresses). ``gkdrag = 0.1`` is the stress
        efficiency G; together they put the default launch stress for a 300 m
        sub-grid peak in a 10 m/s wind at ~0.1 Pa, the order of real sub-grid
        orographic drag.
        """
        return cls(
            gkdrag=jnp.array(gkdrag),
            kwave=jnp.array(kwave),
            gwdrag_cd=jnp.array(gwdrag_cd),
            zmin=jnp.array(zmin),
            zmax=jnp.array(zmax),
        )


class SimpleGwdState(NamedTuple):
    """State variables for gravity wave calculations"""
    
    tau_x: jnp.ndarray            # Zonal momentum flux (N/m²)
    tau_y: jnp.ndarray            # Meridional momentum flux (N/m²)
    wave_stress: jnp.ndarray      # Wave stress magnitude (N/m²)
    breaking_level: jnp.ndarray   # Level where waves break
    deposited_momentum: jnp.ndarray  # Momentum deposited (m/s²)


class SimpleGwdTendencies(NamedTuple):
    """Tendencies from gravity wave drag"""
    
    dudt: jnp.ndarray             # Zonal wind tendency (m/s²)
    dvdt: jnp.ndarray             # Meridional wind tendency (m/s²)
    dtedt: jnp.ndarray            # Temperature tendency from dissipation (K/s)


def _safe_hypot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """``sqrt(x**2 + y**2)`` whose derivative is finite at the origin.

    The Euclidean norm is a cone at ``(0, 0)``: ``d|v|/dx = x/|v|`` is ``0/0``
    there, so a bare ``jnp.sqrt`` returns the correct forward value 0 and a NaN
    derivative. Both places this is used sit *exactly* on the origin in ordinary
    operation — a calm surface column has ``u = v = 0``, and the momentum flux
    is identically zero at every level a wave never reaches — so the NaN is the
    common case, not an edge case, and one calm column poisons the gradient of
    the whole batch through the reduction that follows.

    The safe-denominator double-``where`` feeds ``jnp.sqrt`` a 1.0 in exactly
    the cells the outer ``where`` discards, so the differentiated branch is
    evaluated away from the cone tip and the forward result is bit-identical:
    ``where`` selects the literal 0 there, which is what ``sqrt(0)`` returns.
    The derivative reported at the origin is then 0 — no two-sided derivative
    exists at a cone tip, and 0 is the value the surrounding physics wants,
    since a vanishing flux exerts no drag however it is approached.
    """
    square = x**2 + y**2
    positive = square > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)


@jax.jit
def brunt_vaisala_frequency(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray
) -> jnp.ndarray:
    """Calculate Brunt-Väisälä frequency
    
    Args:
        temperature: Temperature profile (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        
    Returns:
        N²: Brunt-Väisälä frequency squared (s⁻²) [nlev]

    """
    theta = temperature * (c.p0 / pressure) ** (c.rd / c.cpd)
    
    # Calculate vertical gradient of potential temperature
    # Use one-sided differences at boundaries
    dtheta_dz = jnp.zeros_like(theta)
    
    # Interior points - central differences
    dtheta_dz = dtheta_dz.at[1:-1].set(
        (theta[2:] - theta[:-2]) / (height[2:] - height[:-2])
    )
    
    # Boundaries - one-sided
    dtheta_dz = dtheta_dz.at[0].set(
        (theta[1] - theta[0]) / (height[1] - height[0])
    )
    dtheta_dz = dtheta_dz.at[-1].set(
        (theta[-1] - theta[-2]) / (height[-1] - height[-2])
    )
    
    # Brunt-Väisälä frequency squared
    n2 = c.grav / theta * dtheta_dz
    
    # Apply minimum threshold for stability
    n2 = jnp.maximum(n2, 1e-8)
    
    return n2


@jax.jit
def orographic_source(
    u_sfc: jnp.ndarray,
    v_sfc: jnp.ndarray,
    n_sfc: jnp.ndarray,
    rho_sfc: jnp.ndarray,
    h_std: jnp.ndarray,
    config: SimpleGwdParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the orographic gravity-wave source stress.

    Linear (hydrostatic, non-rotating) mountain-wave surface stress,
    ``tau = rho_s * k * G * N * |U| * h^2`` (McFarlane 1987 eq. 3 / Palmer et
    al. 1986), directed *opposite* the low-level wind so that its convergence
    aloft decelerates the flow. The ``rho_s * k`` factor is what makes this a
    stress (Pa): dropping it — as the earlier code did — leaves ``N |U| h^2``,
    which has units m^2 s^-2 and is ~10^3-10^4x too large.

    There is deliberately no blocked-flow (low-Froude) reduction here: blocking
    and its form drag are the job of the Lott-Miller SSO scheme
    (``gravity_waves/sso``). This simple single-wave scheme is the pure
    McFarlane picture — full linear launch stress, then saturation aloft — so
    that the saturation cap in :func:`simple_gwd`, not a launch taper, sets the
    deposition. (The earlier ``min(1, grcrit/Fr)`` taper suppressed exactly the
    high-Froude *linear* regime it should have left alone, and shrank the launch
    so far below the saturation stress that the wave never broke.)

    Args:
        u_sfc: Surface zonal wind (m/s)
        v_sfc: Surface meridional wind (m/s)
        n_sfc: Surface Brunt-Väisälä frequency (s⁻¹)
        rho_sfc: Surface air density (kg/m³)
        h_std: Standard deviation of orography (m)
        config: GW parameters

    Returns:
        Tuple of (tau_x, tau_y): Surface momentum fluxes (N/m²)

    """
    # Surface wind speed. The 1 m/s floor keeps the division by ``wind_speed``
    # finite in both value and derivative; ``_safe_hypot`` handles the calm
    # cone-tip so a zero-wind column does not poison the batch gradient.
    wind_speed = _safe_hypot(u_sfc, v_sfc)
    wind_speed = jnp.maximum(wind_speed, 1.0)  # Minimum wind speed

    # Linear mountain-wave stress magnitude (Pa).
    flux_magnitude = (
        rho_sfc * config.kwave * config.gkdrag
        * n_sfc * wind_speed * h_std**2
    )

    # Project onto wind direction (stress opposes it).
    tau_x = -flux_magnitude * u_sfc / wind_speed
    tau_y = -flux_magnitude * v_sfc / wind_speed

    return tau_x, tau_y


@jax.jit
def simple_gwd(
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray,
    air_density: jnp.ndarray,
    h_std: jnp.ndarray,
    dt: float,
    config: Optional[SimpleGwdParameters] = None
) -> Tuple[SimpleGwdTendencies, SimpleGwdState]:
    """Calculate gravity-wave drag tendencies for a single column.

    A single monochromatic mountain wave is launched at the surface with the
    linear stress of :func:`orographic_source` (directed opposite the low-level
    wind) and propagated upward under two hypotheses:

    - **Critical-level absorption.** The wave carries momentum flux along the
      launch direction. Where the wind component *along that direction* reverses
      (``U · û_launch <= 0``) the intrinsic phase speed matches the flow, the
      wave is absorbed and its remaining flux drops to zero. The earlier code
      tested ``u · τ < 0``; because ``τ`` is *antiparallel* to the launch wind,
      that condition is satisfied at every level where the wind still blows in
      the launch direction — i.e. everywhere the wave is *not* at a critical
      level — so the whole flux was absorbed one level above the source and no
      momentum was ever deposited (issue #842).

    - **McFarlane (1987) saturation.** As the wave climbs into thinner air its
      amplitude grows; the flux cannot exceed the saturation stress
      ``τ_sat = ρ k G U_proj³ / N`` (marginal overturning, streamline slope 1).
      The propagated flux is the running minimum of the launch stress and
      ``τ_sat`` from the surface up, so it can only decrease with height. Its
      convergence ``-∂τ/∂z / ρ`` is the drag, deposited along the launch
      direction — decelerating the flow wherever the wave breaks or is absorbed.

    The vertical axis is the physics-internal top-first frame: index ``-1`` is
    the surface (the launch level), index ``0`` the model top.

    Args:
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        h_std: Standard deviation of sub-grid orography (m)
        dt: Time step (s). Consumed by the ECHAM overshoot guard, which caps
            the drag so a single physics step cannot reverse the wind.
        config: GW parameters

    Returns:
        Tuple of (tendencies, state)

    """
    if config is None:
        config = SimpleGwdParameters.default()

    # Brunt-Väisälä frequency (floored at N² = 1e-8, so N >= 1e-4 s⁻¹ and every
    # division by ``n_bv`` below is finite in value and derivative).
    n2 = brunt_vaisala_frequency(temperature, pressure, height)
    n_bv = jnp.sqrt(n2)
    rho = air_density

    # Launch stress at the surface (index -1), opposite the low-level wind.
    u_sfc, v_sfc = u_wind[-1], v_wind[-1]
    tau_x_oro, tau_y_oro = orographic_source(
        u_sfc, v_sfc, n_bv[-1], rho[-1], h_std, config
    )
    launch_mag = _safe_hypot(tau_x_oro, tau_y_oro) * config.gwdrag_cd

    # Unit launch direction = the low-level *wind* direction (τ points opposite
    # it). ``_safe_hypot`` + floor keep the calm cone-tip finite; on a calm
    # column ``launch_mag`` is zero, so the direction it multiplies is moot.
    launch_speed = jnp.maximum(_safe_hypot(u_sfc, v_sfc), 1e-10)
    dir_x = u_sfc / launch_speed
    dir_y = v_sfc / launch_speed

    # Wind projected onto the launch direction. A non-positive projection is a
    # critical level: the saturation stress there is zero, so the running
    # minimum below drives the flux to zero and holds it there above.
    proj = u_wind * dir_x + v_wind * dir_y
    up = jnp.maximum(proj, 0.0)
    tau_sat = rho * config.kwave * config.gkdrag * up ** 3 / n_bv

    # Propagate the flux magnitude upward (surface -> top = decreasing index):
    # a cumulative minimum of [launch_mag, τ_sat(surface), τ_sat(surface-1), …].
    # Seeding with ``launch_mag`` caps the profile at what was actually launched
    # (a weak wave that never reaches τ_sat propagates conserved, zero drag),
    # while ``τ_sat`` clamps it wherever the wave saturates or is absorbed.
    sat_surface_first = tau_sat[::-1]
    seq = jnp.concatenate([launch_mag[jnp.newaxis], sat_surface_first])
    mag_surface_first = lax.cummin(seq)[1:]
    tau_mag = mag_surface_first[::-1]  # back to top-first

    # Signed flux profile along the launch direction (τ = -|τ| û_launch).
    tau_x = -tau_mag * dir_x
    tau_y = -tau_mag * dir_y

    # Drag = flux convergence, du/dt = -(1/ρ) ∂τ/∂z, with ∂τ/∂z estimated
    # one-sidedly from each level and the level *above* it (lower index in the
    # top-first frame). This upwind assignment — the wave travels upward, so its
    # lost momentum is deposited on the source side of each interface — puts the
    # drag on the level where the wind is still aligned with the launch. A
    # centred difference instead smears a critical-level flux drop onto the
    # reversed-wind level above it, where a launch-opposing force *adds* kinetic
    # energy; the one-sided form keeps the column integral of U·(dU/dt) ≤ 0.
    dz_up = height[:-1] - height[1:]              # h[k-1] - h[k] > 0
    dtau_x_dz = jnp.concatenate(
        [jnp.zeros((1,)), (tau_x[:-1] - tau_x[1:]) / dz_up])
    dtau_y_dz = jnp.concatenate(
        [jnp.zeros((1,)), (tau_y[:-1] - tau_y[1:]) / dz_up])
    dudt = -dtau_x_dz / rho
    dvdt = -dtau_y_dz / rho

    # ECHAM overshoot guard (mo_ssodrag lines 423-429, as in the Lott-Miller
    # port ``sso/lott_miller.py``): cap the drag acceleration at
    # ``rover * |U| / dt`` so one physics step removes at most a quarter of the
    # local wind speed and can never overshoot zero or reverse the flow. The
    # saturated acceleration scales as ``k G U³ / (N H)`` — cubic in wind — so
    # at e.g. 100 m/s and dt = 1800 s the unbounded tendency (~0.13 m/s²) would
    # apply a −225 m/s increment; a critical-level flux drop over a thin layer
    # is sharper still. Floors sit *inside* the sqrt so the derivative is
    # finite at the (common) zero-drag / calm state (issue #558), and the
    # division lives inside ``where`` with a floored denominator so the
    # untaken branch stays poison-free.
    rover = 0.25
    zforc = jnp.sqrt(jnp.maximum(dudt ** 2 + dvdt ** 2, 1.0e-30))
    ztend = jnp.sqrt(jnp.maximum(u_wind ** 2 + v_wind ** 2, 1.0e-30)) / dt
    factor = jnp.where(zforc >= rover * ztend,
                       rover * ztend / jnp.maximum(zforc, 1e-30),
                       1.0)
    dudt = dudt * factor
    dvdt = dvdt * factor

    # Mechanical heating: KE lost to drag reappears as heat, dT/dt = -Ẋ·U / cp,
    # computed from the *limited* tendencies so heat matches the momentum
    # actually removed (mo_ssodrag computes dissipation from the final
    # increments the same way).
    dtedt = -(u_wind * dudt + v_wind * dvdt) / c.cpd

    # Apply drag only inside the [zmin, zmax] window.
    height_mask = (height >= config.zmin) & (height <= config.zmax)
    dudt = jnp.where(height_mask, dudt, 0.0)
    dvdt = jnp.where(height_mask, dvdt, 0.0)
    dtedt = jnp.where(height_mask, dtedt, 0.0)

    tendencies = SimpleGwdTendencies(dudt=dudt, dvdt=dvdt, dtedt=dtedt)

    # Diagnostic: levels where the flux has been reduced below the launch value
    # (saturation or critical-level absorption), i.e. where momentum is
    # deposited. Piecewise-constant, so it is excluded from the gradient checks.
    breaking_level = (tau_mag < launch_mag).astype(jnp.float32)

    state = SimpleGwdState(
        tau_x=tau_x,
        tau_y=tau_y,
        # Norms of fields that are exactly zero over most of a typical column
        # (above a critical level, or on a calm column); see ``_safe_hypot``.
        wave_stress=_safe_hypot(tau_x, tau_y),
        breaking_level=breaking_level,
        deposited_momentum=_safe_hypot(dudt, dvdt),
    )

    return tendencies, state

# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class SimpleGwd(PhysicsTerm):
    """Simple monochromatic gravity-wave drag as a composable PhysicsTerm.

    Cheap fallback for the full Hines + Lott-Miller stack; kept available
    for aquaplanet tests but excluded from the default ``echam_physics()``
    factory. Operates on column-vectorized state ``(nlev, ncols)``. Reads
    ``pressure_full``, ``height_full``, ``air_density`` from the moist-air
    diagnostics dict and the model timestep from
    ``diagnostics["_dt_seconds"]`` (injected by ``ComposablePhysics``).
    Writes only u/v/T tendencies; no Data sub-struct.
    """

    name: ClassVar[str] = "simple_gwd"
    category: ClassVar[str] = "simple_gwd"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "height_full", "air_density",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: SimpleGwdParameters | None = None):
        """Hold the scheme-native :class:`SimpleGwdParameters`."""
        self.params = nnx.Param(params or SimpleGwdParameters.default())

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute u/v/T tendencies from the simple GWD scheme."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        # Placeholder fixed std-dev of sub-grid orography. The real
        # Lott-Miller SSO scheme uses per-column ``terrain.orostd`` instead.
        h_std = jnp.ones(ncols) * 200.0

        tend, _state = jax.vmap(
            simple_gwd,
            in_axes=(1, 1, 1, 1, 1, 1, 0, None, None),
            out_axes=(0, 0),
        )(
            state.u_wind, state.v_wind, state.temperature,
            diagnostics["pressure_full"], diagnostics["height_full"],
            diagnostics["air_density"], h_std, dt, params,
        )

        return PhysicsTendency(
            u_wind=tend.dudt.T,
            v_wind=tend.dvdt.T,
            temperature=tend.dtedt.T,
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers={},
        ), diagnostics
