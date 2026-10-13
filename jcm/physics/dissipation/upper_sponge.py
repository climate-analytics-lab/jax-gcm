"""Upper sponge layer — ECHAM's ``uspnge`` (``mo_upper_sponge.f90``).

ECHAM damps the zonally *asymmetric* part of the flow at the top of the
model. After each dynamics step ``uspnge`` multiplies every spectral
coefficient of divergence, vorticity and temperature whose zonal wavenumber
``m`` is non-zero, at the sponge levels ``nlvspd1..nlvspd2``, by
``1 / (1 + zlf(k)·Δt)`` (``mo_upper_sponge.f90`` lines 91-113: the
``IF (mymsp(is) /= 0)`` test skips the m = 0 coefficients). The zonal mean is
never touched, so the sponge removes no angular momentum and leaves the
radiatively-set zonal-mean temperature alone; it absorbs the planetary and
gravity waves that would otherwise reflect off the lid.

This term is the exact grid-point equivalent. The maps from spectral
(vorticity, divergence) to grid-point (u, v), and from spectral to grid-point
temperature, are linear and conserve zonal wavenumber (they act on each
longitude Fourier mode separately), so scaling the m ≠ 0 spectral
coefficients is the same as scaling the zonal anomalies::

    x'  = x − [x]                  for x in (u, v, T), [·] the zonal mean
    x'  → x' / (1 + zlf(k)·Δt)      at each sponge level k

and the term returns the tendency that produces exactly that implicit step
when the host applies it over one physics step ``Δt``::

    dx/dt = −x' · zlf(k) / (1 + zlf(k)·Δt)

Level profile (ECHAM ``setdyn.f90`` / ``uspnge``): the coefficient at the
lowest sponge level is ``spdrag`` and it is multiplied by ``enspodi`` for each
level going up. Here the profile is parameterised from the top (the e-folding
time at the topmost level, ``sponge_timescale_s``, and the number of levels),
so ``zlf(top + i) = 1 / (sponge_timescale_s · enspodi**i)``; ECHAM's
``spdrag`` is the value at the lowest level,
``1 / (sponge_timescale_s · enspodi**(n_sponge_levels − 1))``. ECHAM's sponge
also always starts at the model top in production (``nlvspd1 = 1``), the only
case this term represents. The ECHAM defaults — ``spdrag = 0.926e-4 s⁻¹``
(3.0 h), ``enspodi = 1``, ``nlvspd1 = nlvspd2 = 1`` — are this term's
defaults.

The zonal mean needs a longitude axis, so the term runs on lon-lat grids only
(the dinosaur door); :meth:`UpperSponge.cache_coords` rejects any other
horizontal layout. The pySES backend has its own finite-lid sponge
(``dycore.lid_sponge``).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData

#: ECHAM ``setdyn.f90``: ``spdrag = 0.926E-04`` s⁻¹ (an e-folding time of
#: 3.0 h) at the sponge level, ``enspodi = 1``, ``nlvspd1 = nlvspd2 = 1``.
ECHAM_SPDRAG = 0.926e-4
ECHAM_SPONGE_TIMESCALE_S = 1.0 / ECHAM_SPDRAG
ECHAM_ENSPODI = 1.0
ECHAM_SPONGE_LEVELS = 1


class UpperSponge(PhysicsTerm):
    """Implicit damping of the zonal anomalies of u, v and T at the top levels."""

    name: ClassVar[str] = "upper_sponge"
    category: ClassVar[str] = "dissipation"

    def __init__(
        self,
        n_sponge_levels: int = ECHAM_SPONGE_LEVELS,
        sponge_timescale_s: float = ECHAM_SPONGE_TIMESCALE_S,
        enspodi: float = ECHAM_ENSPODI,
        damp_temperature: bool = True,
    ):
        """Configure the sponge.

        Args:
            n_sponge_levels: Number of levels, counted from the model top,
                over which the sponge acts (ECHAM ``nlvspd2`` with
                ``nlvspd1 = 1``). Default 1, ECHAM's.
            sponge_timescale_s: e-folding time of the damping at the topmost
                level (s). Default ``1 / 0.926e-4`` s = 3.0 h, ECHAM's
                ``spdrag``.
            enspodi: Factor by which the damping coefficient grows from one
                level to the next one up (ECHAM ``enspodi``); equivalently the
                factor by which the e-folding time grows per level going down.
                Default 1, ECHAM's.
            damp_temperature: Damp the zonal anomaly of temperature as well
                as of the wind, as ECHAM does (``stp`` is damped with the same
                factor). Default True.

        """
        if n_sponge_levels < 1:
            raise ValueError(
                f"n_sponge_levels must be at least 1, got {n_sponge_levels}")
        if sponge_timescale_s <= 0.0:
            raise ValueError(
                "sponge_timescale_s must be positive, got "
                f"{sponge_timescale_s}")
        if enspodi <= 0.0:
            raise ValueError(f"enspodi must be positive, got {enspodi}")
        self.n_sponge_levels = int(n_sponge_levels)
        self.sponge_timescale_s = float(sponge_timescale_s)
        self.enspodi = float(enspodi)
        self.damp_temperature = bool(damp_temperature)
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Precompute the damping coefficient profile and the grid shape.

        Raises:
            ValueError: if the horizontal grid is not a (lon, lat) grid, on
                which the zonal mean this sponge preserves is not defined.

        """
        nodal_shape = tuple(coords.horizontal.nodal_shape)
        if len(nodal_shape) != 2:
            raise ValueError(
                "UpperSponge needs a (longitude, latitude) grid to separate "
                f"the zonal mean; got horizontal nodal shape {nodal_shape}. "
                "On the pySES backend use dycore.lid_sponge instead.")
        nlev = coords.nodal_shape[0]
        # zlf(k) [1/s], top-first (index 0 = model top, the physics frame).
        zlf = jnp.zeros(nlev)
        for i in range(min(self.n_sponge_levels, nlev)):
            zlf = zlf.at[i].set(
                1.0 / (self.sponge_timescale_s * self.enspodi ** i))
        self._zlf = nnx.Variable(zlf)
        self._nlon, self._nlat = (int(n) for n in nodal_shape)
        self._coords_cached = True

    def _zonal_anomaly(self, x: jnp.ndarray) -> jnp.ndarray:
        """``x − [x]`` for a level-major field on the host's horizontal layout.

        The host hands either the whole grid ``(nlev, nlon, nlat)`` or the
        lon-major flattened columns ``(nlev, nlon·nlat)``; both reshape to the
        grid without copying, so one code path serves both.
        """
        grid = x.reshape((x.shape[0], self._nlon, self._nlat))
        anomaly = grid - jnp.mean(grid, axis=1, keepdims=True)
        return anomaly.reshape(x.shape)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Return the tendencies of ECHAM's implicit m ≠ 0 damping step."""
        dt = diagnostics["_dt_seconds"]
        zlf = self._zlf.get_value()
        # Rate that turns a forward-Euler step of length dt into ECHAM's
        # implicit factor 1 / (1 + zlf·dt).
        rate = zlf / (1.0 + zlf * dt)
        rate = rate.reshape((-1,) + (1,) * (state.u_wind.ndim - 1))

        du = -self._zonal_anomaly(state.u_wind) * rate
        dv = -self._zonal_anomaly(state.v_wind) * rate
        if self.damp_temperature:
            dT = -self._zonal_anomaly(state.temperature) * rate
        else:
            dT = jnp.zeros_like(state.temperature)

        tend = PhysicsTendency(
            u_wind=du,
            v_wind=dv,
            temperature=dT,
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers={name: jnp.zeros_like(x)
                     for name, x in state.tracers.items()},
        )
        return tend, diagnostics
