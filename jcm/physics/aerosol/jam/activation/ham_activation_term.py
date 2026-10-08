"""``HamActivation`` -- ECHAM-HAM M7 activation as a composable PhysicsTerm.

Wraps :mod:`ham_activation`'s pure functions the way ``arg_term.py`` wraps
``arg.py``: reads the per-mode dry/wet radius and number from the
``_jam_state`` diagnostic (the M7 core's output, same convention
:class:`~jcm.physics.aerosol.jam.activation.arg_term.ArgActivation` uses),
and publishes exactly the keys that term publishes (``activated_cdnc``,
``activated_fraction``, ``_jam_activation``), so the two activation sources
stay interchangeable downstream (cloud-borne exchange, #602).

Not wired into ``jam_terms.py``/``echam_terms.py`` here (#1017 task split --
the lead integrates it into the ``echam-ham-m7`` preset).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.activation.arg_term import JamActivationData
from jcm.physics.aerosol.jam.activation.ham_activation import (
    LL_CRCUT_STRAT,
    PDF_DEFAULT_BINS,
    ham_arg,
    ham_logtail,
    ham_updraft,
    koehler_ab,
    lin_leaitch,
    mode_col,
)
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.removal_split import split_view
from jcm.physics.aerosol.jam.tracer_layout import mass_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.thermodynamics import saturation_vapor_pressure
from jcm.physics_interface import PhysicsTendency


@tree_math.struct
class HamActivationParameters:
    """Tunable knobs for HAM activation (differentiable)."""

    # mo_activ.f90:68: the module default, never overridden by the
    # reference template's namelist.
    w_min: jnp.ndarray
    # mo_cloud_utils.f90:77 (#345): the ARG turbulent-velocity prefactor.
    fact_tke: jnp.ndarray
    # mo_activ.f90:124 (#345): activ_updraft's OWN correction when
    # ncd_activ == 1 (Lin & Leaitch) -- a different prefactor, not a
    # fallback, see ham_activation.py's ham_updraft docstring.
    fact_tke_lin_leaitch: jnp.ndarray

    @classmethod
    def default(cls) -> "HamActivationParameters":
        return cls(
            w_min=jnp.asarray(0.0),
            fact_tke=jnp.asarray(0.7),
            fact_tke_lin_leaitch=jnp.asarray(1.33),
        )


class HamActivation(PhysicsTerm):
    """ECHAM-HAM M7 activation: ARG (default) or Lin & Leaitch."""

    name: ClassVar[str] = "ham_activation"
    category: ClassVar[str] = "aerosol_activation"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "pressure_full", "air_density",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "activated_cdnc", "activated_fraction", "_jam_activation",
    )
    # The large-scale omega the updraft needs (mo_activ.f90's
    # ``pvervel``) -- a genuine dycore requirement, unconditional (unlike
    # tiedtke_convection's omega, which only some configurations need).
    requires_dycore_fields: ClassVar[tuple[str, ...]] = ("omega",)

    def __init__(
        self,
        spec: ModalAerosolSpec,
        params: HamActivationParameters | None = None,
        *,
        scheme: str = "arg",
        nactivpdf: int = 0,
    ):
        """Hold the params, the M7 population, and the two static selectors.

        Args:
            spec: the M7 population (``ModalAerosolSpec``); ``_jam_state``'s
                mode axis is assumed to be in ``spec.modes`` order, the same
                assumption ``ArgActivation`` makes.
            params: differentiable tunables; defaults to HAM's own values.
            scheme: ``"arg"`` (HAM's Abdul-Razzak & Ghan, ``ncd_activ = 2``,
                the reference preset) or ``"lin_leaitch"`` (``ncd_activ =
                1``). A static Python string, resolved at compose time.
            nactivpdf: HAM's own switch (``setphys.f90:79``): ``0`` (the
                reference preset) runs the single characteristic updraft;
                any other int runs the West et al. (2013) PDF with that
                many bins (``abs(nactivpdf)``; ``1`` -- the only value the
                template or any known namelist sets when the PDF is on --
                means :data:`~ham_activation.PDF_DEFAULT_BINS` = 20, exactly
                as ``mo_activ.f90``'s ``activ_initialize`` resolves it).
                ``lin_leaitch`` ignores the PDF: HAM's ``setphys.f90``
                enforces ``nw = 1`` whenever ``ncd_activ = 1``, so this
                constructor does not even look at ``nactivpdf`` for that
                scheme.

        """
        self.params = nnx.Param(params or HamActivationParameters.default())
        self._spec = spec
        if scheme not in ("arg", "lin_leaitch"):
            raise ValueError(f"Unknown HAM activation scheme {scheme!r}.")
        self._scheme = scheme
        self._nactivpdf = int(nactivpdf)
        self._sigma_g = tuple(m.geom_std_dev for m in spec.modes)
        self._can_activate = tuple(bool(m.can_activate) for m in spec.modes)
        # mo_ham_m7ctl.f90:419: the count-to-mass-median-radius ratio, used
        # below only to give ARG a per-mode MASS-activated fraction (see
        # __call__'s docstring note) -- ham_activ_abdulrazzak_ghan itself
        # never calls ham_m7_logtail's mass branch.
        self._cmedr2mmedr = tuple(
            float(jnp.exp(3.0 * jnp.log(s) ** 2)) for s in self._sigma_g
        )

    def _n_pdf_bins(self):
        if self._nactivpdf == 0:
            return None
        if self._nactivpdf == 1:
            return PDF_DEFAULT_BINS
        return abs(self._nactivpdf)

    def _tke(self, diagnostics, shape):
        """Previous-step TKE, the same optional access ``ArgActivation``
        uses (no term provides ``updraft_velocity`` directly here: HAM's own
        ``activ_updraft`` always derives the updraft from TKE + omega).
        """
        if "vertical_diffusion" in diagnostics:
            return diagnostics["vertical_diffusion"].tke
        return jnp.zeros(shape)

    def _mass_view(self, state, diagnostics):
        """Per-(species, mode) mass mixing ratios from the operator-split
        tracer view (``split_view``).

        ``removal_split.split_view`` reconstructs the tracer values as every
        earlier term this step already left them (step-start state plus the
        running tendency ``_tendency_run`` times ``dt``) -- the same "working
        copy" ECHAM's own operator splitting hands each process. Köhler B
        depends on each mode's instantaneous electrolyte mass fractions, so
        using the split view (rather than the bare step-start state) means a
        new-this-step sulfate condensation event changes the activation
        Köhler coefficients within the same step it occurs in, as ECHAM's
        sequential call order does. (The dry radius / number instead come
        from ``_jam_state``, the M7 core's own per-step output -- the same
        source ``ArgActivation`` reads them from.)
        """
        view = split_view(self._spec, state, diagnostics)
        mass = {}
        for mode in self._spec.modes:
            for sp in mode.species:
                key = mass_name(sp, mode.short)
                if key in view:
                    mass[(sp, mode.short)] = view[key]
        return mass

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        spec = self._spec
        aer = diagnostics["_jam_state"]
        air_density = diagnostics["air_density"]
        pressure = diagnostics["pressure_full"]

        sigma_g = jnp.asarray(self._sigma_g)
        can_activate = jnp.asarray(self._can_activate)
        number_vol = aer.number * air_density[jnp.newaxis, ...]

        mass = self._mass_view(state, diagnostics)
        a_coef, b_coef = koehler_ab(spec, mass, state.temperature)

        tke = self._tke(diagnostics, state.temperature.shape)
        omega = diagnostics.get("_dycore_fields", {}).get(
            "omega", jnp.zeros_like(state.temperature))

        # Broadcasting-native: never assume a fixed horizontal/level rank.
        # ``_jam_state`` fields are (n_modes, *cell); read the cell rank off
        # ``number_vol`` (one less axis than it) so every per-mode static
        # array (can_activate, sigma_g, cmedr2mmedr, ...) reshapes to
        # whatever ``*cell`` actually is -- a single column, a vectorized
        # block of columns, or a whole (nlev, ix, il) grid alike.
        ndim_cell = number_vol.ndim - 1
        can_activate_col = mode_col(can_activate, ndim_cell)

        if self._scheme == "lin_leaitch":
            # setphys.f90 enforces nw = 1 for ncd_activ = 1 (Lin & Leaitch
            # never uses the updraft PDF): mo_activ.f90:120-125's OWN
            # turbulent-velocity correction (1.33 vs 0.7) applies here.
            w, _ = ham_updraft(tke, omega, air_density, params.w_min,
                                params.fact_tke_lin_leaitch, n_pdf_bins=None)
            # convective-cut outputs (na_cv, cdncact_cv) are not part of
            # this term's contract -- only the stratiform pair is used.
            na, _na_cv, cdncact, _cdncact_cv = lin_leaitch(
                number_vol, can_activate, aer.r_wet, sigma_g, w[0],
            )
            n_total = jnp.sum(jnp.where(can_activate_col, number_vol, 0.0), axis=0)
            ratio = jnp.where(na > 0.0, cdncact / jnp.where(na > 0.0, na, 1.0), 0.0)
            ln_sigma = mode_col(jnp.log(sigma_g), ndim_cell)
            cut_frac = ham_logtail(
                aer.r_wet, jnp.full_like(aer.r_wet, LL_CRCUT_STRAT), ln_sigma,
            )
            # Per-mode split that EXACTLY reproduces cdncact when summed
            # against number_vol (Sigma cut_frac*number_vol = na by
            # construction, so scaling by cdncact/na preserves the total):
            # Lin & Leaitch has no native per-mode output (mo_activ.f90's
            # formula acts on the POOLED available number), so this is this
            # term's own, documented decomposition -- not a Fortran output.
            # Flagged for lead review: there is no second, independent, more
            # "correct" partition to check it against.
            number_frac = jnp.where(can_activate_col, cut_frac * ratio, 0.0)
            # Lin & Leaitch has no mass-activation concept at all (purely
            # number-based); mass_frac mirrors number_frac as the simplest
            # defensible default for the downstream cloud-borne mass
            # exchange that both activation sources must feed -- also
            # flagged for lead review.
            mass_frac = number_frac
        else:
            n_pdf_bins = self._n_pdf_bins()
            w, pwpdf = ham_updraft(tke, omega, air_density, params.w_min,
                                    params.fact_tke, n_pdf_bins=n_pdf_bins)
            esw = saturation_vapor_pressure(state.temperature, phase="water")
            cdncact, number_frac, _activated_number, _sm, _smax, rc = ham_arg(
                aer.r_dry, number_vol, a_coef, b_coef, can_activate, sigma_g,
                w, pwpdf, state.temperature, pressure, state.specific_humidity, esw,
            )
            n_total = jnp.sum(jnp.where(can_activate_col, number_vol, 0.0), axis=0)
            # ham_activ_abdulrazzak_ghan never computes a mass-activated
            # fraction itself (ll_numb = .TRUE. throughout), but giving the
            # term the same MASS fraction ArgActivation publishes re-runs
            # ham_logtail's MASS branch (mass_factor = cmedr2mmedr) at the
            # SAME critical radius rc -- HAM's own technique for turning a
            # number-tail fraction into a mass-tail one at a critical
            # radius, exactly as ``ic_scav_nuc`` (mo_ham_wetdep.f90:684-795)
            # already does for in-cloud nucleation scavenging: it calls
            # ham_m7_logtail twice at the SAME rcritrad, once with
            # ll_trac_phase = .TRUE. (number) and once .FALSE. (mass).
            cmedr2mmedr = mode_col(jnp.asarray(self._cmedr2mmedr), ndim_cell)[
                :, jnp.newaxis, ...]
            ln_sigma = mode_col(jnp.log(sigma_g), ndim_cell)[:, jnp.newaxis, ...]
            r_dry_w = aer.r_dry[:, jnp.newaxis, ...]
            fracn_mass = ham_logtail(r_dry_w, rc, ln_sigma, mass_factor=cmedr2mmedr)
            top = jnp.sum(fracn_mass * pwpdf[jnp.newaxis, ...], axis=1)
            bot = jnp.sum(pwpdf, axis=0)
            bot_ok = bot > 0.0
            mass_frac = jnp.where(
                can_activate_col,
                jnp.where(bot_ok, top / jnp.where(bot_ok, bot, 1.0), 0.0),
                0.0,
            )

        activated_fraction = jnp.where(
            n_total > 0.0, cdncact / jnp.where(n_total > 0.0, n_total, 1.0), 0.0)

        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {
            **diagnostics,
            "activated_cdnc": cdncact,
            "activated_fraction": activated_fraction,
            "_jam_activation": JamActivationData(
                number_frac=number_frac, mass_frac=mass_frac,
            ),
        }
