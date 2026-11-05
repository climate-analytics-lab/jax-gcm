"""Prognostic sea-ice surface temperature (ECHAM ``mo_surface_ice.f90::s_sicetemp``).

ECHAM6.3 with a prescribed ice fraction does not prescribe the ice surface
temperature: it integrates it. The ice surface is a thin layer of heat
capacity ``hcapice`` (a ``dice = 5 cm`` ice layer) that gains the net surface
flux, conducts heat through the ice to the freezing water below at ``ctfreez``,
and is held at ``tmelt`` when it would exceed it::

    T_new = (C/dt·T + F_net + (k_ice/h)·T_freeze) / (C/dt + k_ice/h)
    T_new = min(T_new, tmelt)

with ``h`` the ice thickness (``clsst.f90``: 2 m north of the equator, 1 m
south, wherever the ice fraction is positive), ``F_net`` the net flux into the
surface (net shortwave + longwave − sensible − latent, the sum ECHAM calls
``zsflx``) and the conduction written implicitly in ``T``. The fluxes are those
the step delivered to the atmosphere, evaluated at the step-start temperature:
ECHAM solves the lowest-level exchange with the ice at the old ``ptsi`` and
then calls ``s_sicetemp``, so the energy the surface gains is exactly what the
column lost.

Snow on ice is not modelled (``psni = 0``; the snow terms of ``s_sicetemp``
drop out), because no precipitation is routed to the ice tile yet.

Every function is broadcasting-native.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

import jcm.constants as c
from jcm.physics.surface.echam.albedo import CTFREEZ
from jcm.physics.surrogate_gradient import with_surrogate_gradient

__all__ = [
    "SeaIceTemperatureParameters",
    "ice_thickness",
    "seed_ice_temperature",
    "carried_ice_temperature",
    "update_ice_surface_temperature",
    "melt_cap",
]


@struct.dataclass
class SeaIceTemperatureParameters:
    """Constants of ``s_sicetemp`` (ECHAM6.3 ``mo_physical_constants.f90``).

    Every numeric value is a differentiable pytree leaf; ``melt_width`` only
    configures a derivative and is static (``docs/source/design/
    surrogate_gradients.md``).
    """

    #: ``alice``: thermal conductivity of ice [W/K/m].
    ice_conductivity: jnp.ndarray = 2.1656
    #: ``rhoice``: density of ice [kg/m3].
    ice_density: jnp.ndarray = 917.0
    #: ``cpice``: specific heat of ice [J/K/kg].
    ice_specific_heat: jnp.ndarray = 2106.0
    #: ``dice``: thickness of the upper ice layer whose heat capacity is
    #: ``hcapice = rhoice·cpice·dice`` [m]; also ECHAM's minimum ice thickness.
    heat_capacity_layer: jnp.ndarray = 0.05
    #: ``ctfreez``: saline freezing point of the water below the ice [K]
    #: (``iniphy.f90``).
    freezing_temperature: jnp.ndarray = CTFREEZ
    #: ``siced`` of ``clsst.f90``: ice thickness north / south of the equator [m].
    thickness_north: jnp.ndarray = 2.0
    thickness_south: jnp.ndarray = 1.0
    #: Width [K] of the softplus whose derivative stands in for the melt cap's
    #: (static; 0 keeps the exact ``min`` derivative).
    melt_width: float = struct.field(pytree_node=False, default=0.5)

    @classmethod
    def default(cls) -> "SeaIceTemperatureParameters":
        """ECHAM6.3 values."""
        return cls()


def ice_thickness(northern, params: SeaIceTemperatureParameters):
    """``siced``: 2 m where ``northern`` (latitude > 0), 1 m elsewhere (``clsst.f90:126-128``)."""
    return jnp.where(northern, params.thickness_north, params.thickness_south)


def seed_ice_temperature(sea_surface_temperature, params: SeaIceTemperatureParameters | None = None):
    """Return the ice surface temperature a cold start (or a fresh ice cell) begins at.

    ``min(SST, ctfreez)``: ECHAM's AMIP boundary fixes the water under the ice
    at ``ctfreez`` (``clsst.f90:122``), the temperature ``s_sicetemp`` relaxes
    the ice surface towards through its conduction term.
    """
    t_freeze = (CTFREEZ if params is None else params.freezing_temperature)
    return jnp.minimum(sea_surface_temperature, t_freeze)


def carried_ice_temperature(carried, sea_surface_temperature, ice_fraction,
                            params: SeaIceTemperatureParameters | None = None):
    """Return the ice-tile temperature a step starts from.

    The carried prognostic value where there is ice and it has been set
    (> 0); the seed :func:`seed_ice_temperature` for a cold start or a cell
    that has just gained ice; the SST where there is no ice (the value no
    tile reads, which keeps the tile-weighted radiative temperature smooth).
    """
    seed = seed_ice_temperature(sea_surface_temperature, params)
    if carried is None:
        started = seed
    else:
        started = jnp.where(carried > 0.0, carried, seed)
    return jnp.where(ice_fraction > 0.0, started, sea_surface_temperature)


def _melt_exact(t, tmelt):
    return jnp.minimum(t, tmelt)


def melt_cap(temperature, params: SeaIceTemperatureParameters):
    """``min(T, tmelt)`` (``s_sicetemp``: ``ptsi = tmelt`` when it would exceed it).

    Exact value; with ``params.melt_width > 0`` the derivative is that of the
    softplus minimum ``T − w·softplus((T − tmelt)/w)``, so a melting surface
    still responds to the energy it is given.
    """
    tmelt = jnp.asarray(c.tmelt, jnp.asarray(temperature).dtype)
    if params.melt_width == 0.0:
        return _melt_exact(temperature, tmelt)
    w = params.melt_width

    def smooth(t, tm):
        return t - w * jax.nn.softplus((t - tm) / w)

    return with_surrogate_gradient(_melt_exact, smooth)(temperature, tmelt)


def update_ice_surface_temperature(temperature, net_flux, thickness, dt,
                                   params: SeaIceTemperatureParameters):
    """One ``s_sicetemp`` step of the ice surface temperature.

    Args:
        temperature: step-start ice surface temperature ``ptsi`` [K].
        net_flux: net flux into the ice surface ``zsflx`` [W/m2], positive
            down (shortwave + longwave net − sensible − latent).
        thickness: ice thickness [m] (:func:`ice_thickness`).
        dt: physics step [s] (ECHAM's ``delta_time``).
        params: :class:`SeaIceTemperatureParameters`.

    Returns:
        ``(T_new, conductive_flux, melt_residual)``: the new temperature
        ``ptsi``, the conduction from the water below ``pahfice =
        alice·(T_new − ctfreez)/h`` [W/m2] (positive up), and the energy the
        melt cap removed ``pqres = (C/dt + alice/h)·(T_unc − tmelt)`` [W/m2]
        (0 when below the melting point).

    """
    p = params
    t_freeze = p.freezing_temperature
    heat_capacity_rate = (p.ice_density * p.ice_specific_heat * p.heat_capacity_layer) / dt
    conductance = p.ice_conductivity / thickness              # alice/zsniced
    t_unc = ((heat_capacity_rate * temperature + net_flux + conductance * t_freeze)
             / (heat_capacity_rate + conductance))
    t_new = melt_cap(t_unc, p)
    melt_residual = (heat_capacity_rate + conductance) * jnp.maximum(t_unc - c.tmelt, 0.0)
    conductive_flux = conductance * (t_new - t_freeze)
    return t_new, conductive_flux, melt_residual
