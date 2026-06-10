"""``AqueousSulfur`` — in-cloud SO₂ oxidation to cloud-borne sulfate (#496).

Full port of ECHAM-HAM's ``mo_ham_chemistry.f90::ham_wet_chemistry`` (Feichter
et al. 1996 aqueous sulfur chemistry), with **no simplification** of the
kinetics: both the H₂O₂ and the O₃ pathways are integrated over ``niter=5``
sub-steps, and the cloud-droplet pH (H⁺) is solved each sub-step from the
sulfate/SO₂ charge balance (the quadratic in :func:`_aqueous_so4`), so the
pH-dependent SO₂+O₃ rate uses the evolving acidity rather than a fixed pH.

Rate/Henry constants are verbatim from HAM (``zhpbase``, ``ze1k/ze1h`` for O₃
Henry, ``ze3k/ze3h`` for the SO₂ first dissociation, the SO₂ Henry coefficient,
``za21/za22`` for the two SO₂+O₃ channels, the SO₂+H₂O₂ rate
``8e4·exp(-3650·(1/T−1/298))/(0.1+[H⁺])``). The in-cloud liquid water and the
``molec cm⁻³ ↔ mass-mixing-ratio`` conversions (``xtoc``/``ctox``) follow HAM.

Adaptations for jcm/MAM4:
* Oxidants (H₂O₂, O₃) arrive from the :mod:`oxidants` diagnostic already in
  molec cm⁻³, so HAM's mmr→concentration step is skipped.
* Product sulfate goes to the **cloud-borne** accumulation/coarse mass
  (``mc_so4_acc``/``mc_so4_cor``), split by their cloud-borne number fraction —
  HAM's ``ms4as``/``ms4cs`` distribution. SO₂ (``g_so2``) is consumed.
* Sulfate molar mass uses jcm's ``so4`` species value (MAM4-MOM ammonium
  bisulfate, 115 g/mol) consistently in both the pH and the produced mass.
* H₂O₂ is a prescribed oxidant. It is depleted *within* the ``niter``
  sub-stepping (so the within-step H₂O₂-limitation of SO₂ oxidation is
  respected) and reset to the prescribed field each step — this matches
  ECHAM-HAM's offline-oxidant behaviour: ``ham_wet_chemistry`` likewise
  discards the depleted H₂O₂ after its loop, and only a coupled MOZ-style
  prognostic H₂O₂ budget (out of scope) would persist it across steps.

Runs in the **post-cloud** block (needs the current step's cloud water /
fraction), alongside wet deposition.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.species import SPECIES
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

# --- HAM constants (mo_ham_chemistry.f90) ---------------------------------
_NITER = 5
_ZHPBASE = 2.5e-6          # background H+ [mol/l]
_ZE1K, _ZE1H = 1.1e-2, 2300.0   # O3 Henry
_ZE3K, _ZE3H = 1.2e-2, 2010.0   # SO2 first dissociation
_ZQ298 = 1.0 / 298.0
_ZRGAS = 8.2e-2           # R [l·atm/mol/K]
_ZLWCMIN = 1.0e-7         # in-cloud LWC threshold [kg/kg]
_AVO_XTOC = 6.022e20      # HAM's xtoc/ctox factor (avo with unit folding)
_AVOGADRO = 6.022e23      # molec/mol
# SO2 Henry's-law (H0 [mol/l/atm], activation [K]) — HAMMOZ speclist(id_so2).
_H_SO2_0, _H_SO2_ACT = 1.23, 3020.0

# Molar masses in g/mol (HAM works in grams).
_MW_SO2 = GAS_SPECIES["so2"].molar_mass * 1000.0          # 64.0648
_MW_SO4 = SPECIES["so4"].molar_mass * 1000.0             # 115.0 (jcm so4)
_CONV_SO2_SO4_MASS = _MW_SO4 / _MW_SO2

_TINY = 1.0e-30


def _xtoc(rho: jnp.ndarray, mw: float) -> jnp.ndarray:
    """Mass-mixing-ratio → molec cm⁻³ factor (HAM ``xtoc``: ``ρ·6.022e20/mw``)."""
    return rho * _AVO_XTOC / mw


def _aqueous_so4(so2, so4, h2o2, o3, lwc, rho, temperature, dt):
    """In-cloud SO₂ oxidised to sulfate over ``niter`` sub-steps [kg/kg].

    All faithful to ``ham_wet_chemistry``. ``so2``/``so4`` are in-cloud mass
    mixing ratios [kg/kg]; ``h2o2``/``o3`` number densities [molec cm⁻³];
    ``lwc`` the in-cloud liquid water [kg/kg]; ``rho`` air density [kg m⁻³].
    Returns the SO₄ mass produced (mmr); SO₂ consumed = that ·(M_SO2/M_SO4).
    """
    qtp1 = 1.0 / temperature - _ZQ298
    lwcl = jnp.maximum(lwc * rho * 1.0e-6, _TINY)   # [l-water/cm^3-air]
    lwcv = lwc * rho * 1.0e-3                        # liquid volume fraction
    # molec/cm^3(air) -> mol/l(water): HAM ``zfac1 = 1/(zlwcl·avo)``.
    fac1 = 1.0 / (lwcl * _AVOGADRO)

    # --- SO2 + H2O2 effective rate (pH from initial sulfate) ---
    hp0 = _ZHPBASE + so4 * 1000.0 / (jnp.maximum(lwc, _TINY) * _MW_SO4)
    rk = 8.0e4 * jnp.exp(-3650.0 * qtp1) / (0.1 + hp0)
    rke = rk / (lwcl * _AVOGADRO)
    h_so2 = _H_SO2_0 * jnp.exp(_H_SO2_ACT * qtp1)
    pfac = _ZRGAS * lwcv * temperature
    p_so2 = h_so2 * pfac
    f_so2 = p_so2 / (1.0 + p_so2)
    h_h2o2 = 9.7e4 * jnp.exp(6600.0 * qtp1)
    p_h2o2 = h_h2o2 * pfac
    f_h2o2 = p_h2o2 / (1.0 + p_h2o2)
    rkh2o2 = rke * f_so2 * f_h2o2

    # --- O3-path constants ---
    e1 = _ZE1K * jnp.exp(_ZE1H * qtp1)
    e3 = _ZE3K * jnp.exp(_ZE3H * qtp1)
    za = h_so2 * _ZRGAS * temperature * lwcv
    a21 = 4.39e11 * jnp.exp(-4131.0 / temperature)
    a22 = 2.56e3 * jnp.exp(-926.0 / temperature)
    ph_o3 = e1 * _ZRGAS * temperature * lwcv
    f_o3 = ph_o3 / (1.0 + ph_o3)

    so2m = so2 * _xtoc(rho, _MW_SO2)
    so4m = so4 * _xtoc(rho, _MW_SO4)
    h2o2m = h2o2
    zdt = dt / _NITER

    for _ in range(_NITER):
        # H2O2 oxidation.
        q = rkh2o2 * h2o2m
        so2mh = so2m * jnp.exp(-q * zdt)
        dso2h = so2m - so2mh
        h2o2m = jnp.maximum(h2o2m - dso2h, 0.0)
        so4m = so4m + dso2h
        # pH from the SO2/sulfate charge balance (quadratic in H+).
        so2l = so2mh * fac1
        so4l = so4m * fac1
        zb = _ZHPBASE + so4l
        zp = (za * e3 - zb - za * zb) / (1.0 + za)
        zq = -za * e3 * (zb + so2l) / (1.0 + za)
        zp = 0.5 * zp
        hp = -zp + jnp.sqrt(jnp.maximum(zp * zp - zq, 0.0))
        qhp = 1.0 / jnp.maximum(hp, _TINY)
        # SO2 + O3, pH-dependent.
        a2 = (a21 + a22 * qhp) * fac1
        heneff = 1.0 + e3 * qhp
        p_so2b = za * heneff
        f_so2b = p_so2b / (1.0 + p_so2b)
        rko3 = a2 * f_o3 * f_so2b
        q = o3 * rko3
        so2mo = so2mh * jnp.exp(-q * zdt)
        so4m = so4m + (so2mh - so2mo)
        so2m = so2mo

    # ctox: molec/cm3 -> mmr is mw/(6.022e20·rho); SO2 remaining as mmr.
    so2_rem = so2m * (_MW_SO2 / (_AVO_XTOC * rho))
    dso2tot = jnp.clip(so2 - so2_rem, 0.0, so2)
    return dso2tot * _CONV_SO2_SO4_MASS


@tree_math.struct
class AqueousSulfurParameters:
    """Tunable knob for the aqueous oxidation (differentiable)."""

    rate_scale: jnp.ndarray   # multiplies the in-cloud SO4 production

    @classmethod
    def default(cls) -> "AqueousSulfurParameters":
        return cls(rate_scale=jnp.asarray(1.0))


class AqueousSulfur(PhysicsTerm):
    """In-cloud SO₂ + H₂O₂/O₃ oxidation → cloud-borne sulfate."""

    name: ClassVar[str] = "jam_aqueous_sulfur"
    category: ClassVar[str] = "aerosol_aqueous_chemistry"
    requires: ClassVar[tuple[str, ...]] = (
        "oxidants", "clouds", "air_density",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: AqueousSulfurParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population."""
        self.params = nnx.Param(params or AqueousSulfurParameters.default())
        self._spec = spec or MAM4_SPEC
        # Modes carrying sulfate that can host cloud-borne SO4 (accum, coarse).
        self._so4_modes = tuple(
            m.short for m in self._spec.modes if "so4" in m.species
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        zeros = jnp.zeros_like(state.temperature)
        ox = diagnostics["oxidants"]
        clouds = diagnostics["clouds"]
        rho = diagnostics["air_density"]
        dt = jnp.asarray(diagnostics["_dt_seconds"], state.temperature.dtype)

        cloud_fraction = jnp.clip(clouds.cloud_fraction, 0.0, 1.0)
        # In-cloud liquid water (grid-mean qc divided by the cloudy fraction).
        cf_safe = jnp.maximum(cloud_fraction, 1.0e-3)
        lwc_incloud = jnp.maximum(clouds.qc, 0.0) / cf_safe
        active = (cloud_fraction > 0.0) & (lwc_incloud > _ZLWCMIN)

        so2 = state.tracers.get("g_so2", zeros)
        so4_total = sum(
            state.tracers.get(mass_name("so4", m), zeros)
            for m in self._so4_modes
        )

        dso4 = params.rate_scale * _aqueous_so4(
            so2=jnp.maximum(so2, 0.0),
            so4=jnp.maximum(so4_total, 0.0),
            h2o2=jnp.maximum(ox.h2o2, 0.0),
            o3=jnp.maximum(ox.o3, 0.0),
            lwc=lwc_incloud,
            rho=rho,
            temperature=state.temperature,
            dt=dt,
        )
        dso4 = jnp.where(active, dso4, 0.0)

        # Grid-mean rates: produced sulfate is in-cloud, so weight by the
        # cloudy area fraction.
        so4_rate = cloud_fraction * dso4 / dt
        so2_rate = -so4_rate * (_MW_SO2 / _MW_SO4)   # S-conserving SO2 sink

        # Distribute the cloud-borne sulfate over accum/coarse by their
        # cloud-borne number fraction (HAM ms4as/ms4cs split).
        nc = {
            m: jnp.maximum(
                state.tracers.get(number_name(m, cloud_borne=True), zeros), 0.0
            )
            for m in self._so4_modes
        }
        nc_tot = jnp.maximum(sum(nc.values()), _TINY)

        tracer_tends: dict[str, jnp.ndarray] = {"g_so2": so2_rate}
        for m in self._so4_modes:
            frac = nc[m] / nc_tot
            tracer_tends[mass_name("so4", m, cloud_borne=True)] = so4_rate * frac

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
