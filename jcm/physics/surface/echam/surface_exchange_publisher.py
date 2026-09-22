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

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.surface.surface_exchange import (
    SURFACE_EXCHANGE_KEY,
    SURFACE_EXCHANGE_OUTPUT_ATTRS,
    SurfaceExchange,
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
    - the 10 m wind from ``"vertical_diffusion"``.

    The optional rain/snow split stays ``None``: the Tiedtke port exposes
    only total convective precipitation, and a stratiform-only "rain"
    would be wrong as a total split (see the design doc). Per-tile fields
    stay ``None`` because ECHAM's per-tile explicit fluxes are not
    consistent with the delivered grid mean from the implicit solve.
    """

    name: ClassVar[str] = "echam_surface_exchange"
    category: ClassVar[str] = "surface_exchange"
    requires: ClassVar[tuple[str, ...]] = (
        "surface", "radiation", "convection", "clouds",
        "vertical_diffusion", "pressure_full",
    )
    # Literal string (== SURFACE_EXCHANGE_KEY) so the requires-audit's AST
    # walk can evaluate the tuple.
    provides: ClassVar[tuple[str, ...]] = ("surface_exchange",)
    output_attrs: ClassVar = SURFACE_EXCHANGE_OUTPUT_ATTRS

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
        radiation = diagnostics["radiation"]
        clouds = diagnostics["clouds"]
        convection = diagnostics["convection"]
        vdiff = diagnostics["vertical_diffusion"]

        shf = surface.sensible_heat_flux.reshape(ncols)
        lhf = surface.latent_heat_flux.reshape(ncols)
        # Net downward surface radiation, both bands.
        rad_net_down = (
            (radiation.surface_sw_down - radiation.surface_sw_up)
            + (radiation.surface_lw_down - radiation.surface_lw_up)
        ).reshape(ncols)

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
            precipitation=(
                clouds.precip_rain.reshape(ncols)
                + clouds.precip_snow.reshape(ncols)
                + convection.precip_conv.reshape(ncols)
            ),
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
            air_density=p_bot / (c.rd * t_bot * (1.0 + c.vtmpc1 * q_bot)),
            air_potential_temperature=t_bot * (c.p0 / p_bot) ** c.akap,
        )

        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, SURFACE_EXCHANGE_KEY: exchange}
