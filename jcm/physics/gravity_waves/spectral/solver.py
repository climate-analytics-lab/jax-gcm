r"""Spectral non-orographic gravity-wave propagation/deposition solver.

Faithful JAX transliteration of CAM's ``gw_common.F90`` (ESCOMP/CAM, ref
``cam_cesm2_2_rel``, ``src/physics/cam/gw_common.F90``):

- :class:`GWBand`             — ``type GWBand`` + ``new_GWBand``
- :func:`gw_prof`             — ``subroutine gw_prof``
- :func:`gw_drag_prof`        — ``subroutine gw_drag_prof``
- :func:`calc_taucd`          — ``function calc_taucd``
- :func:`momentum_flux`       — ``subroutine momentum_flux``
- :func:`momentum_fixer`      — ``subroutine momentum_fixer``
- :func:`energy_change`       — ``subroutine energy_change``
- :func:`energy_fixer`        — ``subroutine energy_fixer``
- :func:`newtonian_cooling_profile` — the Wehrbein & Leovy (1982) ``alpha0``
  table + interpolation from ``gw_drag.F90::gw_init``

Axis conventions (broadcasting-native, per CLAUDE.md)
-----------------------------------------------------
* The vertical is **axis 0** and **top-first** (index 0 = model top), the
  same orientation as CAM's ``k = 1 .. pver`` and as this repo's
  physics-internal frame. Midpoint fields are ``(nlev, *horiz)``,
  interface fields ``(nlev + 1, *horiz)`` — midpoint ``k`` sits between
  interfaces ``k`` (above) and ``k + 1`` (below).
* The phase-speed spectrum is a separate leading axis placed **in front
  of** the horizontal axes: spectral per-column fields (``c``,
  ``tau_src``) are ``(nspec, *horiz)`` and spectral level fields are
  ``(nlev[+1], nspec, *horiz)`` with ``nspec = 2 * ngwv + 1`` (CAM's
  ``l = -ngwv .. ngwv`` maps to index ``l + ngwv``). Because the horizontal
  axes trail, a plain ``(*horiz)`` field broadcasts cleanly against a
  ``(nspec, *horiz)`` one — no reshapes, and the identical code runs on a
  single column (``*horiz == ()``), a vectorized block ``(ncols,)`` or a
  full grid ``(ix, il)``.

Deviations from the Fortran (each is deliberate; see also
``docs/source/design/frontal_gravity_wave_drag.md``):

1. ``src_level``/``tend_level`` are a single **static Python int**, uniform
   across columns (the frontal source always sets them equal and derives
   them from reference pressures). CAM's per-column integer arrays are not
   supported; the ``where (src_level >= k)`` masks become masks on the
   static level index inside the scans.
2. The ``gw_diffusion`` module (``gw_ediff``/``gw_diff_tend`` — effective
   diffusivity ``egwdffi``, constituent tendencies ``qtgw`` and the
   dry-static-energy diffusion term ``dttdf``) is **not ported**. This is
   CAM's own ``lapply_vdiff=.false.`` code path; the returned heating is
   the kinetic-energy conversion term ``dttke`` only (``ttgw = dttke``).
3. Only the ``lapply_effgw = .true.`` branch is ported (CESM2.2 CAM6
   default ``gw_apply_tndmax = .true.``): efficiency multiplies the stress
   profile up front and the ``tndmax`` limiter caps the summed tendency.
   The WACCM legacy branch (efficiency applied after the limiter) is not.
4. The ``kwvrdg``/``ro_adjust``/``vramp`` optional arguments (ridge scheme,
   inertial-wave adjustment, top-of-model ramp) are not ported — none is
   exercised by the frontal (``use_gw_front``) path with CAM6 defaults.
5. Guarded degenerate states: every division/sqrt that CAM protects with
   loop-level ``if``/``where`` (Intel FPE workarounds) is protected here
   with ``jnp.where`` using *safe operands inside* the ``where`` so that
   reverse-mode gradients cannot see a 0·inf. Callers must supply finite
   ``piln`` (floor the top interface pressure before taking the log).
6. ``limit_tendency_sum`` (default True): the tndmax stability limiter
   caps ``sum_l |gwut_l|`` rather than CAM's net ``|sum_l gwut_l|``, so
   the frictional heating ``dttke`` is bounded by
   ``max|u - c| * tndmax`` too. Identical to CAM whenever the per-wave
   tendencies share one sign; pass ``False`` for the exact CAM limiter.
   Rationale: on grids whose lid layer is only a few Pa thick (ECHAM
   L47, p_top = 0) Lindzen saturation forces *every* wave to break in
   the top layer, the two signs cancel in the net, and CAM's limiter
   leaves the heating unbounded (123 K/day observed) — CAM never runs
   this scheme with such a lid and has no heating bound of its own.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax

import jcm.constants as c

# --- Fortran ``parameter`` constants from gw_common.F90 -------------------
#: Background diffusivity [m^2/s] (``dback``).
DBACK = 0.05
#: Minimum non-zero stress [Pa] (``taumin``).
TAUMIN = 1.0e-10
#: Minimum value of (u - c)^2 [m^2/s^2] (``ubmc2mn``).
UBMC2MN = 0.01
#: Minimum Brunt-Vaisala frequency squared [1/s^2] (gw_prof ``n2min``).
N2MIN = 5.0e-5
#: Small-tendency floor below which gwut is zeroed (gw_drag_prof, "Protection
#: on SMALL gwut to prevent floating point issues").
GWUT_TINY = 1.0e-15


@struct.dataclass
class GWBand:
    """A band of gravity-wave phase speeds (CAM ``type GWBand``).

    ``ngwv`` is static aux data (it sets array sizes at trace time); the
    numeric members are differentiable pytree leaves.

    Attributes:
        dc: Delta between adjacent reference phase speeds [m/s].
        fcrit2: Critical Froude number squared.
        wavelength: Horizontal wavelength [m] (CAM ``wavelength_mid`` =
            1e5 m for the frontal band).
        ngwv: Spectrum half-width — wavenumbers run ``-ngwv .. ngwv``
            (static; CESM2.2 CAM6 ``pgwv`` = 32).

    """

    dc: jnp.ndarray = 2.5
    fcrit2: jnp.ndarray = 1.0
    wavelength: jnp.ndarray = 1.0e5
    ngwv: int = struct.field(pytree_node=False, default=32)

    @property
    def nspec(self) -> int:
        """Number of spectral elements, ``2 * ngwv + 1`` (static)."""
        return 2 * self.ngwv + 1

    def cref(self) -> jnp.ndarray:
        """Return the reference phase speeds ``dc * l``, ``l = -ngwv..ngwv`` [m/s]."""
        return self.dc * jnp.arange(-self.ngwv, self.ngwv + 1)

    def kwv(self) -> jnp.ndarray:
        """Horizontal wavenumber ``2*pi / wavelength`` [1/m]."""
        return 2.0 * jnp.pi / self.wavelength

    def effkwv(self) -> jnp.ndarray:
        """Effective horizontal wavenumber ``fcrit2 * kwv`` [1/m]."""
        return self.fcrit2 * self.kwv()


class GWDragResult(NamedTuple):
    """Outputs of :func:`gw_drag_prof`.

    Attributes:
        utgw: Zonal wind tendency [m/s^2], ``(nlev, *horiz)``.
        vtgw: Meridional wind tendency [m/s^2], ``(nlev, *horiz)``.
        ttgw: Heating rate (dry-static-energy tendency) [J/kg/s],
            ``(nlev, *horiz)``. Kinetic-energy conversion (``dttke``) only —
            see module deviation (2).
        gwut: Per-wave tendency along the source wind [m/s^2],
            ``(nlev, nspec, *horiz)``.
        tau: Wave Reynolds stress at interfaces after the down-scan
            adjustment [Pa], ``(nlev + 1, nspec, *horiz)``.

    """

    utgw: jnp.ndarray
    vtgw: jnp.ndarray
    ttgw: jnp.ndarray
    gwut: jnp.ndarray
    tau: jnp.ndarray


class DirectionalTau(NamedTuple):
    """Reynolds stress split into cardinal directions (:func:`calc_taucd`).

    Each component is ``(nlev + 1, *horiz)`` [Pa].
    """

    east: jnp.ndarray
    west: jnp.ndarray
    north: jnp.ndarray
    south: jnp.ndarray


def midpoint_interp(arr: jnp.ndarray) -> jnp.ndarray:
    """Average adjacent entries along axis 0 (gw_utils ``midpoint_interp``).

    CAM interpolates along dim 2 (levels); here levels are axis 0.
    """
    return 0.5 * (arr[:-1] + arr[1:])


def get_unit_vector(u: jnp.ndarray, v: jnp.ndarray):
    """Unit-vector components and magnitude (gw_utils ``get_unit_vector``).

    The Fortran uses an explicit loop/if instead of a masked divide because
    masked divides still raise FPEs; the JAX analogue of that concern is the
    reverse-mode 0·inf poison, so the safe operand (1.0) goes *inside* the
    ``where`` — ``sqrt`` and the division never see 0 on the taken branch.

    Returns:
        ``(u_n, v_n, mag)`` with ``u_n = v_n = 0`` where ``mag == 0``.

    """
    mag2 = u * u + v * v
    positive = mag2 > 0.0
    mag_safe = jnp.sqrt(jnp.where(positive, mag2, 1.0))
    u_n = jnp.where(positive, u / mag_safe, 0.0)
    v_n = jnp.where(positive, v / mag_safe, 0.0)
    mag = jnp.where(positive, mag_safe, 0.0)
    return u_n, v_n, mag


def dot_2d(u1, v1, u2, v2):
    """Vectorized 2-D dot product (gw_utils ``dot_2d``)."""
    return u1 * u2 + v1 * v2


def gw_prof(t: jnp.ndarray, p_ifc: jnp.ndarray, p_mid: jnp.ndarray):
    """Background-state profiles for the GW solver (gw_common ``gw_prof``).

    The parameterization is assumed to operate only where water vapor
    concentrations are negligible in determining the density (CAM comment).

    Args:
        t: Midpoint temperature [K], ``(nlev, *horiz)``.
        p_ifc: Interface pressure [Pa], ``(nlev + 1, *horiz)``, top-first.
        p_mid: Midpoint pressure [Pa], ``(nlev, *horiz)``.

    Returns:
        ``(rhoi, nm, ni)``: interface density [kg/m^3] ``(nlev+1, *horiz)``,
        midpoint and interface Brunt-Vaisala frequencies [1/s]
        (``(nlev, *horiz)`` and ``(nlev+1, *horiz)``).

    """
    # Interface temperature: top assumes an isothermal atmosphere above the
    # top level; interior is a centered (midpoint) average; bottom copies the
    # bottom midpoint temperature.
    ti = jnp.concatenate([t[:1], midpoint_interp(t), t[-1:]], axis=0)
    rhoi = p_ifc / (c.rd * ti)

    # Top interface: N^2 = g^2 / (cp * T) (isothermal above the model top).
    ni_top = jnp.sqrt(c.grav * c.grav / (c.cpd * ti[:1]))

    # Interior interfaces: dT/dp between the flanking midpoints, floored N^2.
    # The N2MIN floor also keeps the sqrt gradient finite (its argument never
    # reaches 0).
    dtdp = (t[1:] - t[:-1]) / (p_mid[1:] - p_mid[:-1])       # (nlev-1, *h)
    n2 = c.grav * c.grav / ti[1:-1] * (1.0 / c.cpd - rhoi[1:-1] * dtdp)
    ni_int = jnp.sqrt(jnp.maximum(N2MIN, n2))

    # Bottom interface copies the interface Brunt-Vaisala frequency above it.
    ni = jnp.concatenate([ni_top, ni_int, ni_int[-1:]], axis=0)
    nm = midpoint_interp(ni)
    return rhoi, nm, ni


def newtonian_cooling_profile(pref_edge: np.ndarray) -> np.ndarray:
    """Interpolate the Wehrbein & Leovy (1982) Newtonian cooling to a grid.

    Port of the ``alpha0``/``palph`` table and its preprocessing in
    ``gw_drag.F90::gw_init``: convert 1/day to 1/s, floor at 1e-6 s^-1 and
    linearly interpolate in pressure onto the interface grid. Called with
    *reference* pressures at coordinate-cache time (NumPy, not traced),
    exactly as CAM interpolates onto ``pref_edge`` once at init.

    Args:
        pref_edge: Reference interface pressures [Pa], shape ``(nlev + 1,)``,
            top-first.

    Returns:
        ``alpha`` [1/s], shape ``(nlev + 1,)``.

    """
    alpha0 = np.array([
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.10133333, 0.104,
        0.108, 0.112, 0.116, 0.12066667,
        0.126, 0.132, 0.138, 0.144,
        0.15133333, 0.16, 0.17, 0.18,
        0.19, 0.19933333, 0.208, 0.216,
        0.224, 0.232, 0.23466667, 0.232,
        0.224, 0.216, 0.208, 0.20133333,
        0.196, 0.192, 0.188, 0.184,
        0.18266667, 0.184, 0.188, 0.192,
        0.196, 0.19333333, 0.184, 0.168,
        0.152, 0.136, 0.12133333, 0.108,
        0.096, 0.084, 0.072, 0.061,
        0.051, 0.042, 0.033, 0.024,
        0.017666667, 0.014, 0.013, 0.012,
        0.011, 0.010333333, 0.01, 0.01,
        0.01, 0.01, 0.01,
    ])
    palph = np.array([
        2.06115e-06, 2.74280e-06, 3.64988e-06, 4.85694e-06,
        6.46319e-06, 8.60065e-06, 1.14450e-05, 1.52300e-05,
        2.02667e-05, 2.69692e-05, 3.58882e-05, 4.77568e-05,
        6.35507e-05, 8.45676e-05, 0.000112535, 0.000149752,
        0.000199277, 0.000265180, 0.000352878, 0.000469579,
        0.000624875, 0.000831529, 0.00110653, 0.00147247,
        0.00195943, 0.00260744, 0.00346975, 0.00461724,
        0.00614421, 0.00817618, 0.0108801, 0.0144783,
        0.0192665, 0.0256382, 0.0341170, 0.0453999,
        0.0604142, 0.0803939, 0.106981, 0.142361,
        0.189442, 0.252093, 0.335463, 0.446404,
        0.594036, 0.790490, 1.05192, 1.39980,
        1.86273, 2.47875, 3.29851, 4.38936,
        5.84098, 7.77266, 10.3432, 13.7638,
        18.3156, 24.3728, 32.4332, 43.1593,
        57.4326, 76.4263, 101.701, 135.335,
        180.092, 239.651, 318.907, 424.373,
        564.718, 751.477, 1000.0,
    ])
    alpha0 = np.maximum(alpha0 / 86400.0, 1.0e-6)   # 1/day -> 1/s, floored
    palph = palph * 1.0e2                            # hPa -> Pa
    # np.interp clamps outside the table range, matching CAM's lininterp
    # extrapolation behaviour (extrapolation is flat at the table ends).
    return np.interp(pref_edge, palph, alpha0)


def gw_drag_prof(
    band: GWBand,
    src_level: int,
    dt,
    t: jnp.ndarray,
    p_ifc: jnp.ndarray,
    piln: jnp.ndarray,
    rhoi: jnp.ndarray,
    ni: jnp.ndarray,
    ubm: jnp.ndarray,
    ubi: jnp.ndarray,
    xv: jnp.ndarray,
    yv: jnp.ndarray,
    effgw,
    c_l: jnp.ndarray,
    tau_src: jnp.ndarray,
    alpha: jnp.ndarray,
    kvtt: jnp.ndarray | None = None,
    tndmax=400.0 / 86400.0,
    umcfac=0.5,
    satfac=2.0,
    tau_0_ubc: bool = False,
    limit_tendency_sum: bool = True,
) -> GWDragResult:
    """Solve for the drag profile (gw_common ``gw_drag_prof``).

    1. scan up from the wave source to determine the stress profile
       (Lindzen saturation vs molecular/Newtonian damping),
    2. scan down the stress profile to determine the tendencies, applying
       (a) the WKB ``|c - u| / dt`` bound and (b) the ``tndmax``
       computational-stability bound, and adjusting the stress on the
       interface below to reflect the actual bounded tendency (smoothing
       large stress divergences downward while conserving total stress).

    ``tend_level`` is taken equal to ``src_level`` (the frontal source
    always sets them equal — module deviation 1).

    Args:
        band: :class:`GWBand` of the launched spectrum.
        src_level: Static 0-based midpoint index of the source level;
            the launch interface is ``src_level + 1``.
        dt: Model time step [s] (traced scalar OK).
        t: Midpoint temperature [K], ``(nlev, *horiz)``.
        p_ifc: Interface pressure [Pa], ``(nlev + 1, *horiz)``, top-first
            (strictly increasing along axis 0).
        piln: ``log(p_ifc)`` — must be finite (floor a zero top pressure
            before the log).
        rhoi: Interface density [kg/m^3], ``(nlev + 1, *horiz)``.
        ni: Interface Brunt-Vaisala frequency [1/s], ``(nlev + 1, *horiz)``.
        ubm: Wind projected on the source direction, midpoints,
            ``(nlev, *horiz)``.
        ubi: Wind projected on the source direction, interfaces,
            ``(nlev + 1, *horiz)``.
        xv: Zonal component of the source-wind unit vector, ``(*horiz)``.
        yv: Meridional component of the source-wind unit vector, ``(*horiz)``.
        effgw: Tendency efficiency (scalar or ``(*horiz)``).
        c_l: Phase speeds [m/s], ``(nspec, *horiz)``.
        tau_src: Momentum-flux spectrum launched at interface
            ``src_level + 1`` [Pa], ``(nspec, *horiz)`` (non-negative).
        alpha: Newtonian cooling at interfaces [1/s] — ``(nlev + 1,)`` or
            ``(nlev + 1, *horiz)``.
        kvtt: Molecular thermal diffusivity at interfaces [m^2/s]; ``None``
            (default) means zero, CAM's value whenever ``do_molec_diff`` is
            off.
        tndmax: Maximum wind tendency [m/s^2] (CAM: 400 m/s/day).
        umcfac: Maximum allowed fractional change in ``u - c`` per step.
        satfac: Saturation factor (CAM default 2).
        tau_0_ubc: Force tau = 0 at the top interface (static flag; CAM6
            non-WACCM default is False).
        limit_tendency_sum: Static flag (default True). CAM's stability
            limiter caps only the **net** tendency ``|sum_l gwut_l|`` at
            ``tndmax``, so when waves on both sides of ``ubm`` break in
            the same layer (guaranteed at a lid where ``rho -> 0`` drives
            ``tausat -> 0`` for every wave), ``sum_l |gwut_l|`` — and with
            it the frictional heating ``dttke = sum_l |u-c||gwut_l|`` —
            is unbounded even though the wind tendency is capped (123
            K/day observed in 2-Pa-thick top layers of the ECHAM L47
            grid; CAM never runs this scheme with a lid layer thinner
            than O(100 Pa) and CESM2.2 has no heating bound at all).
            With this flag the same limiter instead caps
            ``sum_l |gwut_l|`` at ``tndmax``, which (a) is identical to
            CAM whenever the per-wave tendencies share one sign (the
            common single-critical-level case, where the sum equals the
            net), (b) still caps the net (``|sum| <= sum |.|``), and (c)
            bounds the heating by ``max_l |u - c| * tndmax`` — the
            tndmax-consistent heating bound. Set False for the exact CAM
            behaviour (documented deviation 13).

    Returns:
        :class:`GWDragResult`.

    """
    nlev = t.shape[0]
    kwv = band.kwv()
    effkwv = band.effkwv()
    rog = c.rd / c.grav
    if kvtt is None:
        kvtt = jnp.zeros(nlev + 1)

    # Broadcast alpha / kvtt profiles to at least (nlev+1, ...) so slicing
    # along axis 0 works whether they are 1-D profiles or full fields.
    alpha = jnp.asarray(alpha)
    kvtt = jnp.asarray(kvtt)

    dlnp = piln[1:] - piln[:-1]                    # (nlev, *h), > 0 top-first

    # ---------------------------------------------------------------------
    # Up-scan: stress profiles (Fortran loop ``do k = kbot_src, ktop, -1``).
    # The carry is tau at the interface below the current midpoint; at the
    # source midpoint the carry is replaced by the launched spectrum, which
    # transliterates gw_cm_src's presetting of ``tau(:,l,ksrc+1)``.
    # ---------------------------------------------------------------------
    def up_step(tau_below, xs):
        k, ubi_k, ubi_kp1, rhoi_k, ni_k, alpha_k, kvtt_k, t_k, dlnp_k = xs
        active = k <= src_level
        tau_below = jnp.where(k == src_level, tau_src, tau_below)

        ubmc = ubi_k - c_l                                   # (nspec, *h)
        # Critical level test: does u - c keep its sign across the layer?
        same_sign = (ubmc > 0.0) == (ubi_kp1 > c_l)
        # Lindzen saturation stress |effkwv * rho * (u-c)^3 / (satfac * N)|.
        tausat = jnp.where(
            same_sign & active,
            jnp.abs(effkwv * rhoi_k * ubmc**3 / (satfac * ni_k)),
            0.0,
        )

        # Wave damping (molecular diffusion + Newtonian cooling): imaginary
        # part of the vertical wavenumber. The (u-c)^2 floor UBMC2MN keeps
        # both divisions finite (and their gradients — no safe-operand
        # where needed because the floor itself is the guard).
        d = DBACK + kvtt_k
        ubmc2 = jnp.maximum(ubmc**2, UBMC2MN)
        mi = ni_k / (2.0 * kwv * ubmc2) * (alpha_k + ni_k**2 / ubmc2 * d)
        wrk = -2.0 * mi * rog * t_k * dlnp_k                # <= 0
        taudmp = tau_below * jnp.exp(wrk)

        # PGI bit-for-bit quirk preserved: limit the operands, not the min.
        tausat = jnp.where(tausat <= TAUMIN, 0.0, tausat)
        taudmp = jnp.where(taudmp <= TAUMIN, 0.0, taudmp)

        tau_k = jnp.where(active, jnp.minimum(taudmp, tausat), 0.0)
        return tau_k, tau_k

    up_xs = (
        jnp.arange(nlev),
        ubi[:-1], ubi[1:], rhoi[:-1], ni[:-1],
        alpha[:nlev], kvtt[:nlev], t, dlnp,
    )
    tau_top_carry_init = jnp.zeros_like(tau_src)
    _, tau_scan = lax.scan(up_step, tau_top_carry_init, up_xs, reverse=True)
    # tau_scan[k] is the stress at interface k for k = 0..nlev-1.

    tau = jnp.zeros((nlev + 1,) + tau_src.shape, dtype=tau_scan.dtype)
    tau = tau.at[:nlev].set(tau_scan)
    # Launch interface: gw_cm_src presets tau at ksrc+1; interfaces below
    # the source stay zero (tau = 0 initialization in gw_cm_src).
    tau = tau.at[src_level + 1].set(tau_src)

    if tau_0_ubc:  # static flag — Python branch is trace-safe
        tau = tau.at[0].set(0.0)

    # Apply efficiency to the completed stress profile (lapply_effgw branch).
    # Fortran multiplies interfaces ktop..tend_level+1; interfaces below the
    # source are identically zero here, so multiplying everything is exact.
    tau = tau * effgw

    # ---------------------------------------------------------------------
    # Down-scan: tendencies from stress divergence (``do k = ktop,
    # kbot_tend``). Iteration k reads tau at interface k as adjusted by the
    # previous iteration (the carry) and tau at interface k+1 from the
    # up-scan profile, then writes the adjusted interface k+1.
    # ---------------------------------------------------------------------
    delp = p_ifc[1:] - p_ifc[:-1]                  # (nlev, *h)
    rdel = 1.0 / delp

    def down_step(tau_k_mod, xs):
        k, tau_kp1_up, rdel_k, delp_k, ubm_k = xs
        active = k <= src_level

        # Wind tendency including excess stress carried down from above.
        ubtl = c.grav * (tau_kp1_up - tau_k_mod) * rdel_k     # (nspec, *h)
        # WKB bound: |du/dt| < umcfac * |c - u| / dt so u - c cannot flip.
        ubtl = jnp.minimum(ubtl, umcfac * jnp.abs(c_l - ubm_k) / dt)

        # Fortran sign(ubtl, c - ubm): magnitude of ubtl with the sign of
        # c - ubm (sign(x, 0) = +|x|).
        sgn = jnp.where(c_l - ubm_k >= 0.0, 1.0, -1.0)
        gwut_l = jnp.where(active, jnp.abs(ubtl) * sgn, 0.0)
        ubt = jnp.sum(gwut_l, axis=0)                          # (*h)

        # Second limiter (CAM: |sum over waves| <= tndmax). With
        # limit_tendency_sum the cap applies to sum_l |gwut_l| instead,
        # bounding the frictional heating as well (see the docstring) —
        # the two are identical whenever the per-wave tendencies share a
        # sign. The safe operand (1.0) sits inside the where so the
        # masked division can't emit inf into the gradient.
        if limit_tendency_sum:  # static flag — Python branch is trace-safe
            lim = jnp.sum(jnp.abs(gwut_l), axis=0)             # >= |ubt|
        else:
            lim = jnp.abs(ubt)
        over = lim > tndmax
        ratio = jnp.where(over, tndmax / jnp.where(over, lim, 1.0), 1.0)
        ubt = ratio * ubt
        gwut_l = ratio * gwut_l
        # Protection on SMALL gwut to prevent floating point issues.
        gwut_l = jnp.where(jnp.abs(gwut_l) < GWUT_TINY, 0.0, gwut_l)

        # Redetermine the effective stress on the interface below from the
        # bounded tendency; conserves total stress while smoothing large
        # divergences downward.
        tau_kp1_new = jnp.where(
            active, tau_k_mod + jnp.abs(gwut_l) * delp_k / c.grav, tau_kp1_up,
        )

        utgw_k = jnp.where(active, ubt * xv, 0.0)
        vtgw_k = jnp.where(active, ubt * yv, 0.0)
        # Kinetic -> thermal energy conversion (dttke).
        dttke_k = -jnp.sum((ubm_k - c_l) * gwut_l, axis=0)

        return tau_kp1_new, (tau_kp1_new, gwut_l, utgw_k, vtgw_k, dttke_k)

    down_xs = (jnp.arange(nlev), tau[1:], rdel, delp, ubm)
    _, (tau_below_adj, gwut, utgw, vtgw, dttke) = lax.scan(
        down_step, tau[0], down_xs,
    )

    tau_adj = jnp.concatenate([tau[:1], tau_below_adj], axis=0)

    # ttgw = dttke + dttdf; dttdf = 0 without the (unported) gw_diffusion.
    return GWDragResult(utgw=utgw, vtgw=vtgw, ttgw=dttke, gwut=gwut, tau=tau_adj)


def calc_taucd(
    tau: jnp.ndarray,
    c_l: jnp.ndarray,
    xv: jnp.ndarray,
    yv: jnp.ndarray,
    ubi: jnp.ndarray,
    tend_level: int,
) -> DirectionalTau:
    """Reynolds stress per cardinal direction (gw_common ``calc_taucd``).

    Args:
        tau: Stress at interfaces, ``(nlev + 1, nspec, *horiz)``.
        c_l: Phase speeds, ``(nspec, *horiz)``.
        xv: Zonal source-wind unit-vector component, ``(*horiz)``.
        yv: Meridional source-wind unit-vector component, ``(*horiz)``.
        ubi: Projected interface wind, ``(nlev + 1, *horiz)``.
        tend_level: Static lowest midpoint level with tendencies.

    Returns:
        :class:`DirectionalTau` (entries are zero below interface
        ``tend_level + 1``).

    """
    nlev_p1 = tau.shape[0]
    ubi_tend = ubi[tend_level + 1]                          # (*h)

    # Signed stress: |tau| with the sign of c - ubi (sign(x, 0) = +|x|).
    sgn = jnp.where(c_l[None] - ubi[:, None] >= 0.0, 1.0, -1.0)
    tausg = sgn * jnp.abs(tau)                              # (nlev+1, nspec, *h)

    # Interface mask k <= tend_level + 1 (Fortran ``k-1 <= tend_level``).
    k_idx = jnp.arange(nlev_p1).reshape((-1,) + (1,) * (tau.ndim - 1))
    tausg = jnp.where(k_idx <= tend_level + 1, tausg, 0.0)

    behind = c_l[None] < ubi_tend                           # (1?, nspec, *h)
    taub = jnp.sum(jnp.where(behind, tausg, 0.0), axis=1)   # (nlev+1, *h)
    tauf = jnp.sum(jnp.where(behind, 0.0, tausg), axis=1)

    east = jnp.where(xv > 0.0, tauf * xv, taub * xv)
    west = jnp.where(xv > 0.0, taub * xv, tauf * xv)
    north = jnp.where(yv > 0.0, tauf * yv, taub * yv)
    south = jnp.where(yv > 0.0, taub * yv, tauf * yv)
    return DirectionalTau(east=east, west=west, north=north, south=south)


def momentum_flux(taucd: DirectionalTau, tend_level: int):
    """Momentum flux through the bottom interface (``momentum_flux``).

    Returns ``(um_flux, vm_flux)``, each ``(*horiz)`` [Pa].
    """
    um_flux = taucd.east[tend_level + 1] + taucd.west[tend_level + 1]
    vm_flux = taucd.north[tend_level + 1] + taucd.south[tend_level + 1]
    return um_flux, vm_flux


def momentum_fixer(
    tend_level: int,
    p_ifc: jnp.ndarray,
    um_flux: jnp.ndarray,
    vm_flux: jnp.ndarray,
    utgw: jnp.ndarray,
    vtgw: jnp.ndarray,
):
    """Restore momentum conservation below the GW region (``momentum_fixer``).

    Spreads the momentum sourced from below ``tend_level`` back over the
    mass between the source interface and the surface.
    """
    nlev = utgw.shape[0]
    # Total mass from ground to source interface: dp / g.
    rdm = c.grav / (p_ifc[-1] - p_ifc[tend_level + 1])       # (*h)
    du = -um_flux * rdm
    dv = -vm_flux * rdm
    k_idx = jnp.arange(nlev).reshape((-1,) + (1,) * (utgw.ndim - 1))
    below = k_idx > tend_level
    utgw = utgw + jnp.where(below, du, 0.0)
    vtgw = vtgw + jnp.where(below, dv, 0.0)
    return utgw, vtgw


def energy_change(
    dt,
    p_ifc: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    dudt: jnp.ndarray,
    dvdt: jnp.ndarray,
    dsdt: jnp.ndarray,
):
    """Column total-energy change from the tendencies (``energy_change``).

    Returns ``de`` [W/m^2], ``(*horiz)``.
    """
    delp = p_ifc[1:] - p_ifc[:-1]
    return jnp.sum(
        delp / c.grav * (
            dsdt
            + dudt * (u + dudt * 0.5 * dt)
            + dvdt * (v + dvdt * 0.5 * dt)
        ),
        axis=0,
    )


def energy_fixer(
    tend_level: int,
    p_ifc: jnp.ndarray,
    de: jnp.ndarray,
    ttgw: jnp.ndarray,
):
    """Remove the energy change below the GW region (``energy_fixer``)."""
    nlev = ttgw.shape[0]
    de_dm = -de * c.grav / (p_ifc[-1] - p_ifc[tend_level + 1])
    k_idx = jnp.arange(nlev).reshape((-1,) + (1,) * (ttgw.ndim - 1))
    return ttgw + jnp.where(k_idx > tend_level, de_dm, 0.0)
