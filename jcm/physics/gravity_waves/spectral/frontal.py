r"""Frontogenesis-triggered gravity-wave source (CAM ``gw_front.F90``).

Faithful JAX transliteration of CAM's frontal source (ESCOMP/CAM, ref
``cam_cesm2_2_rel``, ``src/physics/cam/gw_front.F90``):

- :func:`flat_spectrum`      — ``flat_cm_desc`` (``src_tau`` part)
- :func:`gaussian_spectrum`  — ``gaussian_cm_desc`` (``src_tau`` part)
- :func:`gw_cm_src`          — ``gw_cm_src``

The ``CMSourceDesc`` bookkeeping type collapses to plain arguments here:
``ksrc``/``kfront`` are static level indices chosen by the caller from
reference pressures (see :mod:`.term`) and ``src_tau`` is the spectrum
array returned by the two spectrum builders.

Axis conventions match :mod:`.solver`: vertical axis 0 top-first,
spectrum axis in front of the trailing horizontal axes.

Deviation from the Fortran: :func:`gaussian_spectrum` accepts an optional
``center`` (CAM's Gaussian is hard-centered on c = 0); the default 0.0
reproduces CAM exactly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import erfc

from jcm.physics.gravity_waves.spectral.solver import (
    GWBand,
    dot_2d,
    get_unit_vector,
    midpoint_interp,
)


class CMSource(NamedTuple):
    """Outputs of :func:`gw_cm_src` (the ``intent(out)`` set of the Fortran).

    Attributes:
        tau_src: Launched stress spectrum at interface ``ksrc + 1`` [Pa],
            ``(nspec, *horiz)`` — zero in columns that do not launch.
        ubm: Midpoint wind projected on the source direction [m/s],
            ``(nlev, *horiz)``.
        ubi: Interface wind projected on the source direction [m/s],
            ``(nlev + 1, *horiz)``.
        xv: Zonal component of the source-wind unit vector, ``(*horiz)``.
        yv: Meridional component of the source-wind unit vector, ``(*horiz)``.
        c: Phase speeds ``cref + source wind`` [m/s], ``(nspec, *horiz)``.
        launch: Boolean launch mask ``frontgf > frontgfc``, ``(*horiz)``.

    """

    tau_src: jnp.ndarray
    ubm: jnp.ndarray
    ubi: jnp.ndarray
    xv: jnp.ndarray
    yv: jnp.ndarray
    c: jnp.ndarray
    launch: jnp.ndarray


def flat_spectrum(band: GWBand, taubgnd) -> jnp.ndarray:
    """Flat launch spectrum (``flat_cm_desc``): ``taubgnd`` per wavenumber.

    Wavenumber l = 0 (the stationary wave) is prohibited, as in CAM.

    Returns:
        ``src_tau`` [Pa], shape ``(nspec,)``.

    """
    src_tau = jnp.full((band.nspec,), taubgnd)
    return src_tau.at[band.ngwv].set(0.0)


def gaussian_spectrum(band: GWBand, height, width, center=0.0) -> jnp.ndarray:
    """Bin-averaged Gaussian launch spectrum (``gaussian_cm_desc``).

    Each spectral bin gets the *average* of ``height * exp(-((c - center) /
    width)^2)`` over the bin, computed exactly from the difference of the
    Gaussian integral (erfc) at the bin edges — transliterating the
    ``gaussian_bounds`` construction in the Fortran. Wavenumber l = 0 is
    prohibited.

    Args:
        band: The launched :class:`GWBand`.
        height: Peak stress [Pa] (CAM wires namelist ``taubgnd`` here).
        width: Gaussian width [m/s] (CAM ``front_gaussian_width`` = 30).
        center: Center phase speed [m/s]. CAM has no such parameter (its
            Gaussian is centered on c = 0); the default reproduces CAM.

    Returns:
        ``src_tau`` [Pa], shape ``(nspec,)``.

    """
    cref = band.cref()
    # Boundaries of each bin: cref - dc/2, plus the final right edge.
    bounds = jnp.concatenate([cref - 0.5 * band.dc,
                              cref[-1:] + 0.5 * band.dc])
    # Integral of the Gaussian from each bound to +infinity.
    integral = erfc((bounds - center) / width) * height * width * jnp.sqrt(jnp.pi) / 2.0
    # Average over each bin = (left integral - right integral) / dc.
    src_tau = (integral[:-1] - integral[1:]) / band.dc
    return src_tau.at[band.ngwv].set(0.0)


def gw_cm_src(
    band: GWBand,
    ksrc: int,
    u: jnp.ndarray,
    v: jnp.ndarray,
    frontgf_src: jnp.ndarray,
    frontgfc,
    src_tau: jnp.ndarray,
) -> CMSource:
    """Frontally-triggered wave source (``gw_cm_src``).

    Waves launch from interface ``ksrc + 1`` wherever the frontogenesis
    function at the trigger level exceeds ``frontgfc``. ``src_level`` and
    ``tend_level`` both equal ``ksrc`` (as in the Fortran) and are the
    caller's static int — they are not returned.

    Args:
        band: The launched :class:`GWBand`.
        ksrc: Static 0-based source midpoint index (CAM ``kbot_front``,
            the level whose lower interface is the last one above 500 hPa).
            Requires ``ksrc + 1 <= nlev - 1``.
        u: Midpoint zonal wind [m/s], ``(nlev, *horiz)``.
        v: Midpoint meridional wind [m/s], ``(nlev, *horiz)``.
        frontgf_src: Frontogenesis function at the trigger level ``kfront``
            [K^2/m^2/s], ``(*horiz)``.
        frontgfc: Critical frontogenesis threshold [K^2/m^2/s].
        src_tau: Launch spectrum [Pa], ``(nspec,)`` (from
            :func:`flat_spectrum` or :func:`gaussian_spectrum`).

    Returns:
        :class:`CMSource`.

    """
    # Source wind: average of the source midpoint and the one below — the
    # Fortran's "source level interface value" 0.5*(u(ksrc+1) + u(ksrc)).
    usrc = 0.5 * (u[ksrc + 1] + u[ksrc])
    vsrc = 0.5 * (v[ksrc + 1] + v[ksrc])
    xv, yv, mag = get_unit_vector(usrc, vsrc)

    # Project the midpoint winds onto the source direction. CAM only fills
    # k = 1..ksrc (the rest of ubm is never read by the solver, whose
    # tendencies are masked to k <= tend_level); computing every level is
    # equivalent and keeps the array well-defined.
    ubm = dot_2d(u, v, xv, yv)

    # Interface projection: top interface takes the top midpoint value,
    # interior interfaces average the flanking midpoints, and the source
    # interface ksrc + 1 carries the source-wind magnitude (which is the
    # source wind projected onto its own unit vector). The bottom interface
    # repeats the bottom midpoint — CAM leaves interfaces below ksrc + 1
    # unset; the solver never reads them.
    ubi = jnp.concatenate([ubm[:1], midpoint_interp(ubm), ubm[-1:]], axis=0)
    ubi = ubi.at[ksrc + 1].set(mag)

    # GW generation depends on frontogenesis at the trigger level (which may
    # be below the actual source level).
    launch = frontgf_src > frontgfc
    spec_shape = (band.nspec,) + (1,) * frontgf_src.ndim
    tau_src = jnp.where(launch, src_tau.reshape(spec_shape), 0.0)

    # Phase speeds: reference speeds plus the source-level wind.
    c = band.cref().reshape(spec_shape) + ubi[ksrc + 1]

    return CMSource(tau_src=tau_src, ubm=ubm, ubi=ubi, xv=xv, yv=yv, c=c,
                    launch=launch)
