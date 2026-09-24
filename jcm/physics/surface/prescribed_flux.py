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


def missing_prescribed_flux_fields(forcing: ForcingData | None) -> list[str]:
    """Names of the forced-mode forcing fields that are ``None``.

    ``forcing=None`` — what the CLI assembly returns when the forcing config
    attaches nothing (``kind: default`` with every optional input off) —
    has every field missing, so a forced-mode physics gets the actionable
    "supply forcing.prescribed_surface_flux" error rather than an
    ``AttributeError``.
    """
    return [name for name in PRESCRIBED_FLUX_FORCING_FIELDS
            if getattr(forcing, name, None) is None]


def check_prescribed_flux_forcing(forcing: ForcingData, owner: str,
                                  run_window=None) -> None:
    """Fail loudly if forced mode cannot be served by ``forcing`` over the run.

    Shared by every forced-mode term's ``validate_forcing`` (the ECHAM
    :class:`PrescribedSurfaceFlux` and SPEEDY's
    ``SpeedySurfaceFlux(prescribed_fluxes=True)``) so the two stay identical.
    Two checks:

    1. every ``prescribed_*`` field is present (absence is a composition
       error, not a zero flux);
    2. when ``run_window = (start_s, end_s)`` (seconds since
       1970-01-01) is known, every date-aligned (``BY_DATE``/
       ``BY_DATE_INTERP``) ``TimeSeries`` field covers it — outside its axis
       the selection clamps and would silently hold the archive's end sample
       (see :func:`jcm.forcing.by_date_coverage_error`). Skipped when the
       window or the leaf is traced (``run`` inside a JAX transformation),
       where no concrete value exists to check.
    """
    import jax

    from jcm.forcing import TimeSeries, by_date_coverage_error

    missing = missing_prescribed_flux_fields(forcing)
    if missing:
        raise ValueError(
            f"{owner} needs the prescribed surface-flux forcing fields, but "
            f"{missing} are None. Supply them via "
            "forcing.prescribed_surface_flux (CLI) or set them on the "
            "ForcingData directly (coupler door); units/signs follow the "
            "surface-exchange contract, "
            "docs/source/design/surface_exchange.md."
        )
    if run_window is None:
        return
    start_s, end_s = run_window
    # The archive's declared coverage (CF ``time_bnds``) wins over the
    # end-sample cadence when the reader found one.
    bounds = getattr(forcing, "prescribed_flux_time_bounds", None)
    if any(isinstance(x, jax.core.Tracer)
           for x in jax.tree_util.tree_leaves(bounds)):
        bounds = None
    for name in PRESCRIBED_FLUX_FORCING_FIELDS:
        leaf = getattr(forcing, name)
        if not isinstance(leaf, TimeSeries) or any(
                isinstance(x, jax.core.Tracer)
                for x in jax.tree_util.tree_leaves(
                    (leaf.times, leaf.align_mode))):
            continue
        err = by_date_coverage_error(leaf, start_s, end_s,
                                     name=f"{owner}: forcing.{name}",
                                     bounds=bounds)
        if err is not None:
            raise ValueError(err)


def check_prescribed_flux_consumers(physics, forcing) -> None:
    """Reject prescribed surface fluxes that no composed term consumes.

    The converse of :func:`check_prescribed_flux_forcing`. The interactive
    surface schemes (``SpeedySurfaceFlux`` without ``prescribed_fluxes``, the
    surface-coupled ``TteTkeVerticalDiffusion``) compute their own fluxes and
    never read ``prescribed_*``, so fluxes supplied to such a composition
    would be silently ignored while the run looks forced. Consumers are found
    by the declared capability
    (:meth:`jcm.physics.physics_term.PhysicsTerm.consumed_forcing_fields`,
    aggregated by ``physics.consumed_forcing_fields()``), never by class name,
    so a user-replaced or removed term is judged by what it actually reads. A
    physics object without the hook (e.g. Held-Suarez) consumes nothing.

    Called where the physics and the concrete forcing first meet: by the CLI
    runners right after forcing assembly, and by ``Model.run`` (the choke
    point every Python entry point funnels through). Only presence is
    inspected (``is None``), so it is safe on traced forcing.
    """
    supplied = [name for name in PRESCRIBED_FLUX_FORCING_FIELDS
                if getattr(forcing, name, None) is not None]
    if not supplied:
        return
    consumed_fn = getattr(physics, "consumed_forcing_fields", None)
    consumed = set(consumed_fn()) if consumed_fn is not None else set()
    if consumed.intersection(supplied):
        return
    raise ValueError(
        "forcing.prescribed_surface_flux is set (ForcingData carries "
        f"{supplied}), but no term in the composed physics consumes "
        "prescribed surface fluxes: the interactive surface schemes "
        "(SpeedySurfaceFlux, surface-coupled TteTkeVerticalDiffusion) compute "
        "their own fluxes, so the run would silently ignore the prescribed "
        "ones. Enable forced mode (jax-gcm#301): on the CLI use "
        "physics=speedy-forced-flux or physics=echam-forced-flux; in Python "
        "compose SpeedySurfaceFlux(prescribed_fluxes=True) (SPEEDY) or "
        "TteTkeVerticalDiffusion(couple_surface=False) followed by "
        "PrescribedSurfaceFlux() (ECHAM). Otherwise drop the "
        "prescribed_surface_flux block / leave the prescribed_* fields None. "
        "See docs/source/design/surface_exchange.md.")


def validate_run_forcing(physics, forcing, run_window=None) -> None:
    """Enforce both directions of the forced-mode forcing contract (#301).

    The ONE check every run entry point applies to its concrete forcing
    before stepping, so no door can run a forced composition on a silent
    zero flux or an interactive one on silently ignored fluxes:

    (a) prescribed fluxes supplied → some composed term must consume them
        (:func:`check_prescribed_flux_consumers`);
    (b) a consumer composed → ``physics.validate_forcing(forcing,
        run_window)`` runs every term's own check: the forced-mode terms
        raise if the fluxes are absent or, given a concrete ``run_window``
        ``(start_s, end_s)`` in seconds since 1970-01-01, if a
        date-aligned archive does not cover it
        (:func:`check_prescribed_flux_forcing`).

    Callers: ``Model.run_from_state_with_carry`` (the choke point of
    ``run`` / ``resume`` / ``run_from_state``), ``SingleColumnModel.run``,
    ``PrescribedStateModel.run`` (window = its state times), and the CLI /
    recipe doors right after forcing assembly (window unknown there, so
    presence only; the model call then checks coverage). A physics object
    without ``validate_forcing`` is tolerated. Presence checks never read a
    tracer, so this is safe inside a JAX transformation.
    """
    check_prescribed_flux_consumers(physics, forcing)
    validate = getattr(physics, "validate_forcing", None)
    if validate is not None:
        validate(forcing, run_window=run_window)


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

    def consumed_forcing_fields(self) -> tuple[str, ...]:
        """Return the four ``prescribed_*`` fields, which this term always reads."""
        return PRESCRIBED_FLUX_FORCING_FIELDS

    def validate_forcing(self, forcing: ForcingData,
                         run_window=None) -> None:
        """Fail loudly at run start if the prescribed fields are absent or
        a date-aligned archive does not cover the run window.
        """
        check_prescribed_flux_forcing(forcing, "PrescribedSurfaceFlux",
                                      run_window)

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
