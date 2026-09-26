"""ECHAM publisher of the package-independent surface-exchange struct (#754).

Assembles :class:`~jcm.physics.surface.surface_exchange.SurfaceExchange`
from the ECHAM-family diagnostics that already exist — pure publication, no
physics change. It is a separate terminal term (unlike SPEEDY, whose
surface-flux term publishes inline) because the pieces of the contract are
finalized at different points of ECHAM's ``physc`` ordering: the turbulent
fluxes are delivered by the vdiff implicit solve (republished as
``"surface"``), but stratiform precipitation only exists after the cloud
microphysics, which runs *after* the surface term. Composing this publisher
after the microphysics is what guarantees every published field is the
value of the SAME step.
"""

from typing import ClassVar

import jax.numpy as jnp

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.surface.surface_exchange import (
    SURFACE_EXCHANGE_KEY,
    SurfaceExchange,
    surface_exchange_output_attrs,
)
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


class EchamSurfaceExchange(PhysicsTerm):
    """Publish the #754 coupling contract from the ECHAM diagnostics.

    Zero tendencies — this is a diagnostics-only terminal term. Every
    guaranteed field is filled from the fluxes the column actually
    received this step:

    - turbulent fluxes and evaporation from ``"surface"`` (the
      vdiff-delivered values ``EchamSurface`` republishes; in forced mode
      these carry the prescribed fluxes, so the published struct echoes
      what the coupler fed in — the closed loop);
    - the surface radiation balance from ``"radiation"``;
    - precipitation as stratiform (``clouds.precip_rain + precip_snow``)
      plus convective (``convection.precip_conv``);
    - the 10 m wind vector and speed from ``"vertical_diffusion"`` (ECHAM
      ``u10``/``v10``/``wind10``, ``wind_reference="10m"``), grid mean and
      per tile with the tile fractions it was weighted by. The wind is the
      atmosphere's own in forced mode too: vdiff diagnoses it before its
      surface-coupling branch.

    The optional rain/snow split stays ``None``: the Tiedtke port exposes
    only total convective precipitation, and a stratiform-only "rain"
    would be wrong as a total split (see the design doc). The per-tile
    FLUX fields stay ``None`` because ECHAM's per-tile explicit fluxes are
    not consistent with the delivered grid mean from the implicit solve;
    the per-tile wind fields are filled, since the grid-mean 10 m wind is
    by construction their fraction-weighted sum.

    Only the surface exchange itself is a hard dependency: ``"surface"`` and
    ``"vertical_diffusion"`` (the delivered fluxes + 10 m wind) and
    ``"pressure_full"``. Radiation and precipitation are read OPTIONALLY via
    ``diagnostics.get`` so a trimmed composition that drops them — e.g. the
    surface+vdiff single-column boundary-layer cases — still composes and
    runs; there the radiative term of ``net_heat_flux`` and the
    ``precipitation`` sum degrade to zero (the published struct is unused in
    those runs). A full ECHAM package always carries radiation and the cloud
    /convection precip, so the contract is complete in every real run.
    """

    name: ClassVar[str] = "echam_surface_exchange"
    category: ClassVar[str] = "surface_exchange"
    requires: ClassVar[tuple[str, ...]] = (
        "surface", "vertical_diffusion", "pressure_full",
    )
    # Literal string (== SURFACE_EXCHANGE_KEY) so the requires-audit's AST
    # walk can evaluate the tuple.
    provides: ClassVar[tuple[str, ...]] = ("surface_exchange",)
    output_attrs: ClassVar = surface_exchange_output_attrs("10m")

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Assemble and publish the SurfaceExchange struct; zero tendencies."""
        _nlev, ncols = state.temperature.shape

        surface = diagnostics["surface"]
        vdiff = diagnostics["vertical_diffusion"]
        # Optional in a trimmed (surface+vdiff) composition; present in every
        # full ECHAM package (see the class docstring).
        radiation = diagnostics.get("radiation")
        clouds = diagnostics.get("clouds")
        convection = diagnostics.get("convection")

        shf = surface.sensible_heat_flux.reshape(ncols)
        lhf = surface.latent_heat_flux.reshape(ncols)
        # Net downward surface radiation, both bands (zero when radiation is
        # trimmed out — the struct is unused in that case).
        if radiation is not None:
            rad_net_down = (
                (radiation.surface_sw_down - radiation.surface_sw_up)
                + (radiation.surface_lw_down - radiation.surface_lw_up)
            ).reshape(ncols)
        else:
            rad_net_down = jnp.zeros(ncols)

        # Total precipitation = stratiform (clouds) + convective, each read
        # only if its term is composed.
        precipitation = jnp.zeros(ncols)
        if clouds is not None:
            precipitation = (precipitation
                             + clouds.precip_rain.reshape(ncols)
                             + clouds.precip_snow.reshape(ncols))
        if convection is not None:
            precipitation = precipitation + convection.precip_conv.reshape(ncols)

        # Lowest-model-level thermodynamics for external bulk-flux
        # algorithms (#301 discussion). Physics-internal frame is
        # top-first: index -1 = lowest level.
        p_bot = diagnostics["pressure_full"][-1].reshape(ncols)
        t_bot = state.temperature[-1].reshape(ncols)
        q_bot = state.specific_humidity[-1].reshape(ncols)

        exchange = SurfaceExchange(
            # SHF/LHF are positive up, so the into-surface net flux
            # subtracts them from the absorbed radiation.
            net_heat_flux=rad_net_down - shf - lhf,
            sensible_heat_flux=shf,
            latent_heat_flux=lhf,
            evaporation=surface.evaporation.reshape(ncols),
            precipitation=precipitation,
            # momentum_flux_u/v are ALREADY the downward momentum flux into
            # the surface (positive with the wind; the delivered column
            # momentum change is their negative — verified against the
            # column-integrated vdiff tendency, and see the diagnosis in
            # tte_tke/matrix_solver.py::diagnose_surface_fluxes), which is
            # exactly the contract sign. No flip, unlike SPEEDY's
            # on-the-atmosphere ``ustr``.
            stress_u=surface.momentum_flux_u.reshape(ncols),
            stress_v=surface.momentum_flux_v.reshape(ncols),
            wind_speed=vdiff.wind_10m.reshape(ncols),
            wind_u=vdiff.wind_10m_u.reshape(ncols),
            wind_v=vdiff.wind_10m_v.reshape(ncols),
            air_density=p_bot / (c.rd * t_bot * (1.0 + c.vtmpc1 * q_bot)),
            air_potential_temperature=t_bot * (c.p0 / p_bot) ** c.akap,
            wind_reference="10m",
            # Tile axis 0 = water, 1 = sea ice, 2 = land (the vdiff tiles).
            tile_fraction=vdiff.surface_fraction.reshape(ncols, -1),
            wind_u_tile=vdiff.wind_10m_u_tile.reshape(ncols, -1),
            wind_v_tile=vdiff.wind_10m_v_tile.reshape(ncols, -1),
            wind_speed_tile=vdiff.wind_10m_tile.reshape(ncols, -1),
        )

        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, SURFACE_EXCHANGE_KEY: exchange}
