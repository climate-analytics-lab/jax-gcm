r"""2-D frontogenesis function on a separable lat-lon grid.

Provider for the ``"frontogenesis"`` diagnostic consumed by
:class:`~jcm.physics.gravity_waves.spectral.term.FrontalGravityWaveDrag`.
Computes the function CAM's SE dycore supplies to the physics as
``frontgf`` (``src/dynamics/se/gravity_waves_sources.F90::
compute_frontogenesis``, ref ``cam_cesm2_2_rel``):

.. math::

    F = -\nabla_h\theta \cdot \big[(\nabla_h\theta \cdot \nabla_h)\,
        \mathbf{u}_h\big]
      = -\big[\theta_x (\theta_x u_x + \theta_y u_y)
            + \theta_y (\theta_x v_x + \theta_y v_y)\big]

with :math:`\theta_x = \frac{1}{a\cos\varphi}\partial_\lambda\theta`,
:math:`\theta_y = \frac{1}{a}\partial_\varphi\theta` (units K²/m²/s, the
same as CAM's ``FRONTGF`` history field).

Implementation notes / deviations:

- Scalar spherical gradients carry the metric factors
  ``1/(a cos(lat)) d/dlon`` and ``1/a d/dlat``. The *vector* term
  :math:`(\nabla\theta\cdot\nabla)\mathbf{u}` is evaluated component-wise
  on (u, v) as scalars — the spherical curvature (Christoffel) terms that
  HOMME's covariant ``ugradv_sphere`` includes are neglected. They are
  :math:`O(\tan\varphi\, |u| / a)` corrections, small against the
  deformation terms at the frontal scales this trigger cares about, and
  the trigger only compares F against a threshold.
- Centered finite differences: periodic (wrap-around) in longitude
  (uniform spacing required), centered non-uniform-spacing differences in
  latitude with one-sided differences at the first/last latitude rows
  (Gaussian grids do not reach the poles, so ``cos(lat) > 0`` everywhere).
- This provider is for **separable lat-lon grids only** (e.g. the
  dinosaur backend's Gaussian grid). Unstructured-grid stencils (pySES
  pg2 / SE-GLL) are explicitly out of scope and must supply their own
  ``"frontogenesis"`` diagnostic.
"""

from __future__ import annotations

import jax.numpy as jnp

import jcm.constants as c


def _ddlon_periodic(field: jnp.ndarray, dlon) -> jnp.ndarray:
    """Centered periodic derivative in longitude (axis -2), per radian."""
    return (jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-2)) / (2.0 * dlon)


def _ddlat(field: jnp.ndarray, lats: jnp.ndarray) -> jnp.ndarray:
    """Centered derivative in latitude (axis -1), per radian.

    Interior points use the non-uniform centered difference
    ``(f[j+1] - f[j-1]) / (lat[j+1] - lat[j-1])``; the first/last rows use
    one-sided differences.
    """
    interior = (field[..., 2:] - field[..., :-2]) / (lats[2:] - lats[:-2])
    first = (field[..., 1:2] - field[..., 0:1]) / (lats[1] - lats[0])
    last = (field[..., -1:] - field[..., -2:-1]) / (lats[-1] - lats[-2])
    return jnp.concatenate([first, interior, last], axis=-1)


def frontogenesis_function(
    u: jnp.ndarray,
    v: jnp.ndarray,
    theta: jnp.ndarray,
    lons: jnp.ndarray,
    lats: jnp.ndarray,
    radius: float | None = None,
) -> jnp.ndarray:
    """Compute the CAM frontogenesis function F on a lat-lon grid.

    Args:
        u: Zonal wind [m/s], ``(..., nlon, nlat)`` — e.g. ``(nlev, nlon,
            nlat)``. Longitude is axis -2 and latitude axis -1, matching
            the dinosaur nodal layout.
        v: Meridional wind [m/s], same shape as ``u``.
        theta: Potential temperature [K], same shape as ``u``.
        lons: 1-D longitudes [radians], uniformly spaced, periodic,
            ``(nlon,)``.
        lats: 1-D latitudes [radians], strictly monotonic, ``(nlat,)``;
            must satisfy ``|lat| < pi/2`` (no pole points).
        radius: Sphere radius [m]; defaults to ``jcm.constants.rearth``.

    Returns:
        Frontogenesis function [K^2/m^2/s], same shape as ``theta``.

    """
    a = c.rearth if radius is None else radius
    lats = jnp.asarray(lats)
    lons = jnp.asarray(lons)
    dlon = lons[1] - lons[0]

    # Metric factor for the zonal derivative; Gaussian latitudes never hit
    # the poles so cos(lat) stays strictly positive.
    inv_acos = 1.0 / (a * jnp.cos(lats))
    inv_a = 1.0 / a

    tx = _ddlon_periodic(theta, dlon) * inv_acos
    ty = _ddlat(theta, lats) * inv_a
    ux = _ddlon_periodic(u, dlon) * inv_acos
    uy = _ddlat(u, lats) * inv_a
    vx = _ddlon_periodic(v, dlon) * inv_acos
    vy = _ddlat(v, lats) * inv_a

    # C = (grad(theta) . grad) u_h, component-wise (see module notes).
    c1 = tx * ux + ty * uy
    c2 = tx * vx + ty * vy
    return -(tx * c1 + ty * c2)
