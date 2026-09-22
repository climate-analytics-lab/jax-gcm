"""Externally prescribed surface fluxes as a composable term (jax-gcm#301).

The forced-mode door for the ECHAM-family (column-vectorized) physics: an
external component (ocean/land/wave model, e.g. Veros through JAX-ESM)
computes the air-surface turbulent fluxes and jcm delivers them instead of
running its own surface exchange. The prescribed values arrive on the
``prescribed_*`` fields of :class:`jcm.forcing.ForcingData` in the
surface-exchange contract convention
(``docs/source/design/surface_exchange.md``): sensible heat and evaporation
positive up, stress positive down (momentum into the surface) — exactly
what the publishing side (:mod:`jcm.physics.surface.surface_exchange`)
emits, so a coupler can close the loop without any sign/unit shims.

Placement of the seam
---------------------
In the interactive ECHAM configuration the surface fluxes are delivered by
the TTE-TKE vertical diffusion's implicit solve (bottom-row Robin BC). The
forced configuration therefore composes
``TteTkeVerticalDiffusion(couple_surface=False)`` — interior-only mixing,
insulating/free-slip bottom — followed by this term, which adds the
prescribed fluxes as an explicit bottom-layer source:

    dT/dt|_bot = g·SHF / (cpd·Δp) ,  dq/dt|_bot = g·E / Δp ,
    du/dt|_bot = −g·τ_u / Δp      ,  dv/dt|_bot = −g·τ_v / Δp ,

with Δp the bottom-layer pressure thickness. A prescribed flux is
state-independent, so the explicit delivery is exact for the surface term
(no Richtmyer–Morton handshake to honour — the implicit coupling exists to
keep a *state-dependent* exchange stable). The same-step bookkeeping the
interactive path maintains is preserved: the delivered-flux fields of the
``vertical_diffusion`` diagnostics are overwritten (so ``EchamSurface``
republishes the prescribed values and the Tiedtke moisture-budget closure
anchors to the prescribed evaporation), the vdiff ``qv_tendency`` profile
gains the surface moistening (ECHAM's ``pqte`` at ``cucall`` time), and the
running ``thermo_run`` view is advanced.

SPEEDY's forced mode does NOT use this term: its surface fluxes are already
an explicit bottom-layer source, so the prescribed values replace the bulk
formulae inside the scheme itself — see
``jcm.physics.surface.speedy_surface_flux.get_surface_fluxes(prescribed=...)``.
"""

from typing import ClassVar

import jax.numpy as jnp

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

#: The ForcingData fields forced mode reads; absence of ANY of them is a
#: composition error, reported loudly rather than run as a zero flux.
PRESCRIBED_FLUX_FORCING_FIELDS = (
    "prescribed_sensible_heat_flux",
    "prescribed_evaporation",
    "prescribed_stress_u",
    "prescribed_stress_v",
)


def missing_prescribed_flux_fields(forcing: ForcingData) -> list[str]:
    """Names of the forced-mode forcing fields that are ``None``."""
    return [name for name in PRESCRIBED_FLUX_FORCING_FIELDS
            if getattr(forcing, name) is None]


class PrescribedSurfaceFlux(PhysicsTerm):
    """Deliver externally prescribed turbulent surface fluxes (#301).

    Composed immediately after ``TteTkeVerticalDiffusion(
    couple_surface=False)`` and before ``EchamSurface`` — see the module
    docstring for the seam rationale. Column-vectorized host only (the
    ECHAM family); the vertical axis is axis 0 and the surface is the last
    level of the physics-internal (top-first) frame.
    """

    name: ClassVar[str] = "prescribed_surface_flux"
    category: ClassVar[str] = "prescribed_surface_flux"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_half", "vertical_diffusion",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def augment_probe_forcing(self, forcing: ForcingData) -> ForcingData:
        """Seed the shape probe's ``prescribed_*`` fields with zeros.

        Lets ``get_empty_data`` trace the prescribed-flux path rather than a
        ``None``-guard; the real run's check lives in :meth:`validate_forcing`.
        """
        zeros = jnp.zeros_like(forcing.stl_am)
        return forcing.copy(
            prescribed_sensible_heat_flux=zeros,
            prescribed_evaporation=zeros,
            prescribed_stress_u=zeros,
            prescribed_stress_v=zeros,
        )

    def validate_forcing(self, forcing: ForcingData) -> None:
        """Fail loudly at run start if the prescribed fields are absent."""
        missing = missing_prescribed_flux_fields(forcing)
        if missing:
            raise ValueError(
                "PrescribedSurfaceFlux is composed but the forcing fields "
                f"{missing} are None. Supply them via "
                "forcing.prescribed_surface_flux (CLI) or set them on the "
                "ForcingData directly (coupler door); units/signs follow "
                "the surface-exchange contract, "
                "docs/source/design/surface_exchange.md."
            )

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Apply the prescribed fluxes as an explicit bottom-layer source.

        A missing prescribed field is caught before the run by
        :meth:`validate_forcing`; here (which also runs under the abstract
        shape probe) a ``None`` field falls back to a zero map so the trace
        stays well-defined.
        """
        _nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]

        # Contract convention: SHF/E positive up, stress positive down.
        zeros_col = jnp.zeros(ncols)

        def _prescribed(name):
            value = getattr(forcing, name)
            return zeros_col if value is None else value.reshape(ncols)

        shf = _prescribed("prescribed_sensible_heat_flux")
        evap = _prescribed("prescribed_evaporation")
        stress_u = _prescribed("prescribed_stress_u")
        stress_v = _prescribed("prescribed_stress_v")

        # Bottom-layer pressure thickness from the moist-air diagnostics
        # (physics-internal frame is top-first: index -1 = surface).
        pressure_half = diagnostics["pressure_half"]
        dp = pressure_half[-1] - pressure_half[-2]
        g_over_dp = c.grav / dp

        zeros = jnp.zeros_like(state.temperature)
        t_tend = zeros.at[-1].set(shf * g_over_dp / c.cpd)
        q_tend = zeros.at[-1].set(evap * g_over_dp)
        # Positive-down stress removes momentum from the atmosphere.
        u_tend = zeros.at[-1].set(-stress_u * g_over_dp)
        v_tend = zeros.at[-1].set(-stress_v * g_over_dp)

        tendency = PhysicsTendency(
            u_wind=u_tend,
            v_wind=v_tend,
            temperature=t_tend,
            specific_humidity=q_tend,
        )

        # Keep the interactive path's same-step bookkeeping intact:
        # - delivered-flux fields -> EchamSurface republishes the
        #   prescribed values as the public "surface" fluxes and the
        #   Tiedtke moisture-budget closure anchors to the prescribed
        #   evaporation (its ``effective_evaporation`` read);
        # - qv_tendency gains the surface moistening so the zdqpbl
        #   closure sees the ECHAM ``pqte``-at-cucall profile;
        # - the vdiff surface_stress_u/v fields are, like the contract,
        #   positive-down (the delivered column momentum change is their
        #   negative — see tte_tke/matrix_solver.py::
        #   diagnose_surface_fluxes), so the prescribed stress passes
        #   through unnegated.
        # The published latent heat is the vaporization value alhc*E; a
        # coupler whose evaporation includes sublimation accounts for the
        # ice enthalpy on its own side of the interface (documented in the
        # design doc).
        vdiff = diagnostics["vertical_diffusion"]
        vdiff = vdiff.copy(
            surface_evaporation=evap,
            surface_sensible_heat=shf,
            surface_latent_heat=c.alhc * evap,
            surface_stress_u=stress_u,
            surface_stress_v=stress_v,
            qv_tendency=vdiff.qv_tendency + q_tend,
        )

        from jcm.physics.diagnostics.moist_air_state import advance_thermo_run
        diagnostics = advance_thermo_run(
            diagnostics, dt,
            d_temperature=t_tend,
            d_specific_humidity=q_tend,
        )

        return tendency, {**diagnostics, "vertical_diffusion": vdiff}
