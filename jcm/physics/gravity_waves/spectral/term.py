r"""Composable :class:`PhysicsTerm` for CAM's frontal spectral GW drag.

Wires the frontal source (:mod:`.frontal`, CAM ``gw_front.F90``) into the
spectral solver (:mod:`.solver`, CAM ``gw_common.F90``) the same way
CAM's ``gw_drag.F90::gw_tend`` ``use_gw_front`` block does: source →
``gw_drag_prof`` → directional stresses → momentum fixer → energy fixer.

Frontogenesis-provider contract
-------------------------------
The trigger field is read from the diagnostics dict under the key
``"frontogenesis"`` — a midpoint-level field ``(nlev, *horiz)`` in
K²/m²/s (CAM's ``FRONTGF``), which an upstream provider term or the
dycore must supply (for the dinosaur lat-lon backend,
:func:`~jcm.physics.gravity_waves.spectral.frontogenesis.frontogenesis_function`
computes it from (u, v, theta); wiring that provider up is a follow-up).
If the key is absent, the trigger field falls back to the constant
``params.fallback_frontogenesis`` — **0.0 by default, so the term is
inert without a provider**: no waves launch and all tendencies are
exactly zero.

Deviations from CAM's driver block (see the solver module docstring for
solver-level deviations):

- No constituent/DSE vertical diffusion (``qtgw``/``dttdf``/``egwdffi``)
  — the heating is the kinetic-energy conversion term plus the energy
  fixer.
- No polar taper (namelist ``gw_polar_taper``; default off for the SE
  dycore this port follows).
- ``effgw_cm`` is spatially uniform (a differentiable scalar parameter).
- ``kvtt`` (molecular diffusivity, WACCM ``do_molec_diff``) is zero.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from dinosaur.hybrid_coordinates import HybridCoordinates
from flax import nnx

import jcm.constants as c
from jcm.physics_interface import PhysicsTendency
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.gravity_waves.spectral.frontal import (
    flat_spectrum,
    gaussian_spectrum,
    gw_cm_src,
)
from jcm.physics.gravity_waves.spectral.params import (
    FrontalGWParameters,
    SpectrumShape,
)
from jcm.physics.gravity_waves.spectral.solver import (
    GWBand,
    calc_taucd,
    energy_change,
    energy_fixer,
    gw_drag_prof,
    gw_prof,
    momentum_fixer,
    momentum_flux,
    newtonian_cooling_profile,
)

#: Floor applied to the top interface pressure before taking ``log(p)``.
#: Pure-sigma and some hybrid grids have p_top = 0 exactly; CAM grids do
#: not (hyai(1) > 0), so its ``piln`` is always finite. The floor keeps the
#: damping exponent ``wrk`` finite (a huge-but-finite dlnp at the top layer
#: just drives exp(wrk) -> 0, i.e. full damping) without poisoning
#: gradients with log(0).
_PTOP_FLOOR = 1.0e-4


class FrontalGravityWaveDrag(PhysicsTerm):
    """Frontogenesis-triggered spectral (Charron & Manzini) GW drag.

    The configuration is a :class:`FrontalGWParameters` held in an
    ``nnx.Param`` so its numeric tunables (``taubgnd``, ``frontgfc``,
    ``effgw``, ...) are differentiable leaves, while the static fields
    (``ngwv``, spectrum shape, level-selection pressures) ride along as
    aux data.

    Source/trigger levels are *static* midpoint indices derived from the
    reference pressures ``a + b * p0`` in :meth:`cache_coords`, following
    CAM's ``kbot_front`` (source just above 500 hPa) and ``kfront``
    (trigger check just above 600 hPa) selection from ``pref_edge``.
    """

    name: ClassVar[str] = "frontal_gravity_wave_drag"
    category: ClassVar[str] = "gravity_waves"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()
    # Composing this term into a Model requires a frontogenesis source:
    # either the dycore provider (DinosaurDycore(compute_frontogenesis=
    # True)) or an upstream term that ``provides`` the field. Model
    # validates this at construction (fail-loud beats silently-inert).
    requires_dycore_fields: ClassVar[tuple[str, ...]] = ("frontogenesis",)

    def __init__(self, params: FrontalGWParameters | None = None) -> None:
        """Initialize with a :class:`FrontalGWParameters` configuration."""
        self.params = nnx.Param(params or FrontalGWParameters.default())
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Cache hybrid/sigma coefficients, static levels and alpha.

        ``p_half = a + b * ps`` covers both ``HybridCoordinates`` and pure
        sigma (a = 0, b = sigma). Reference interface pressures use the
        standard surface pressure ``c.p0``, mirroring CAM's ``pref_edge``.
        """
        vertical = coords.vertical
        if isinstance(vertical, HybridCoordinates):
            a_half = np.asarray(vertical.a_boundaries)         # Pa
            b_half = np.asarray(vertical.b_boundaries)         # dimensionless
        else:  # SigmaCoordinates
            sigma_boundaries = np.asarray(vertical.boundaries)
            a_half = np.zeros_like(sigma_boundaries)
            b_half = sigma_boundaries
        self._a_half = nnx.Variable(jnp.asarray(a_half))
        self._b_half = nnx.Variable(jnp.asarray(b_half))

        params = self.params.get_value()
        nlev = a_half.shape[0] - 1
        pref_edge = a_half + b_half * c.p0                     # (nlev+1,)

        # CAM gw_drag_init level selection, translated to 0-based indices:
        # - kbot_front (source midpoint): the interface below it is the
        #   deepest interface above `source_pressure` (500 hPa) —
        #   Fortran ``maxloc(pref_edge, mask) - 1`` = count(<p) - 2 0-based.
        # - kfront (trigger midpoint): the deepest midpoint whose upper
        #   interface is above `front_pressure` (600 hPa) — count(<p) - 1.
        # Both are static Python ints (loop/mask bounds at trace time).
        ksrc = int(np.count_nonzero(pref_edge < params.source_pressure)) - 2
        kfront = int(np.count_nonzero(pref_edge < params.front_pressure)) - 1
        # Clamp for tiny test grids; gw_cm_src reads u[ksrc + 1].
        self._ksrc = int(np.clip(ksrc, 0, nlev - 2))
        self._kfront = int(np.clip(kfront, 0, nlev - 1))

        # Wehrbein & Leovy (1982) Newtonian cooling on reference interface
        # pressures (CAM interpolates once at init, onto pref_edge).
        self._alpha = nnx.Variable(
            jnp.asarray(newtonian_cooling_profile(pref_edge)))
        self._coords_cached = True

    def __call__(self, state, diagnostics, forcing, terrain):
        """Compute (u, v, T) tendencies from frontally-launched waves."""
        params = self.params.get_value()
        dt = diagnostics["_dt_seconds"]

        # Pressure fields from the cached hybrid coefficients.
        ps = state.normalized_surface_pressure * c.p0          # (*horiz)
        lev_shape = (-1,) + (1,) * ps.ndim
        a_half = self._a_half.get_value().reshape(lev_shape)
        b_half = self._b_half.get_value().reshape(lev_shape)
        p_half = a_half + b_half * ps[None]                    # (nlev+1, *h)
        p_full = 0.5 * (p_half[:-1] + p_half[1:])              # (nlev, *h)
        piln = jnp.log(jnp.maximum(p_half, _PTOP_FLOOR))

        rhoi, _nm, ni = gw_prof(state.temperature, p_half, p_full)

        band = GWBand(dc=params.dc, fcrit2=params.fcrit2,
                      wavelength=params.wavelength, ngwv=params.ngwv)
        if params.spectrum is SpectrumShape.GAUSSIAN:
            # CAM wires namelist taubgnd as the Gaussian height
            # (gw_drag.F90: gaussian_cm_desc(..., taubgnd, width)).
            src_tau = gaussian_spectrum(band, params.taubgnd,
                                        params.gaussian_width,
                                        params.gaussian_center)
        else:
            src_tau = flat_spectrum(band, params.taubgnd)

        # Frontogenesis trigger, in precedence order: the dycore-supplied
        # per-step field (Model injects DynamicalCore.physics_fields under
        # "_dycore_fields"), a physics-side provider's top-level
        # "frontogenesis" diagnostic, else the constant fallback
        # (default 0 -> inert; see module docstring).
        dycore_fields = diagnostics.get("_dycore_fields")
        frontgf = (dycore_fields.get("frontogenesis")
                   if isinstance(dycore_fields, dict) else None)
        if frontgf is None:
            frontgf = diagnostics.get("frontogenesis")
        if frontgf is None:
            frontgf_src = jnp.full(ps.shape, params.fallback_frontogenesis)
        else:
            frontgf_src = frontgf[self._kfront]

        ksrc = self._ksrc
        src = gw_cm_src(band, ksrc, state.u_wind, state.v_wind,
                        frontgf_src, params.frontgfc, src_tau)

        result = gw_drag_prof(
            band, ksrc, dt,
            state.temperature, p_half, piln, rhoi, ni,
            src.ubm, src.ubi, src.xv, src.yv,
            params.effgw, src.c, src.tau_src,
            self._alpha.get_value(),
            tndmax=params.tndmax,
            umcfac=params.umcfac,
            satfac=params.satfac,
            tau_0_ubc=params.tau_0_ubc,
        )
        utgw, vtgw, ttgw = result.utgw, result.vtgw, result.ttgw

        if params.apply_fixers:
            # CAM driver order: project stresses, fix momentum below the
            # source, then remove the residual column energy change.
            taucd = calc_taucd(result.tau, src.c, src.xv, src.yv,
                               src.ubi, ksrc)
            um_flux, vm_flux = momentum_flux(taucd, ksrc)
            utgw, vtgw = momentum_fixer(ksrc, p_half, um_flux, vm_flux,
                                        utgw, vtgw)
            de = energy_change(dt, p_half, state.u_wind, state.v_wind,
                               utgw, vtgw, ttgw)
            ttgw = energy_fixer(ksrc, p_half, de, ttgw)

        tendency = PhysicsTendency(
            u_wind=utgw,
            v_wind=vtgw,
            # ttgw is a dry-static-energy tendency [J/kg/s]; CAM converts
            # with 1/cpair for output, and our prognostic is temperature.
            temperature=ttgw / c.cpd,
            specific_humidity=jnp.zeros_like(state.specific_humidity),
        )
        return tendency, diagnostics
