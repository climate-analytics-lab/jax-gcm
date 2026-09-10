"""``SulfurGasChemistry`` — gas-phase DMS/SO₂ oxidation feeding MAM4 (#496).

Port of ECHAM-HAM's ``mo_ham_chemistry.f90::ham_gas_chemistry`` (the
prescribed-oxidant HAM sulfur scheme), with one adaptation for the MAM4-JAX
coupling: where HAM converts SO₂+OH (and the DMS "MSA" branch) **directly** to
particulate SO₄, here those route to **gas-phase H₂SO₄** (``g_h2so4``) so the
MAM4 core does the gas→particle step (condensation + nucleation).

Reactions (rate constants verbatim from HAM, cm³ molec⁻¹ s⁻¹):

* DMS + OH (abstraction)  ``k1 = 9.6e-12·exp(-234/T)``
* DMS + OH (addition)     ``k2 = 1.7e-42·exp(7810/T)·[O₂] / (1 + 5.5e-31·exp(7460/T)·[O₂])``
  → branch ``f_so2 = (k1 + 0.75·k2)/(k1+k2)`` to SO₂; the rest to H₂SO₄ (MSA-as-sulfate).
* DMS + NO₃               ``k3 = 1.9e-13·exp(520/T)`` → SO₂  (night)
* SO₂ + OH + M            Troe ``k0=4.0e-31·(T/300)^-3.3``, ``k∞=2.0e-12``, ``Fc=0.45`` → H₂SO₄

Day/night is handled continuously through the oxidant fields themselves
(:mod:`oxidants` gives OH ∝ cos(zenith), NO₃ ∝ 1−cos(zenith)). Each precursor is
integrated over the physics step with a stable exponential decay. Sulfur is
conserved atom-for-atom (DMS→SO₂→H₂SO₄ each carry one S).

SOAG: jcm has no VOC precursor yet, so SOA gas is produced at a small
boundary-layer-weighted prescribed rate (tunable, interim) so the core sees a
nonzero condensable organic; replace with a real VOC+OH source later.
"""

from __future__ import annotations

import math
from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.chemistry.oxidants import air_number_density
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES, SULFUR_GASES
from jcm.physics.aerosol.jam.tracer_layout import gas_name, gas_tracer_specs
from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics_interface import PhysicsTendency

# Molar-mass conversion factors (mass mixing ratio), from the gas table.
_M_DMS = GAS_SPECIES["dms"].molar_mass
_M_SO2 = GAS_SPECIES["so2"].molar_mass
_M_H2SO4 = GAS_SPECIES["h2so4"].molar_mass
_CONV_DMS_SO2 = _M_SO2 / _M_DMS
_CONV_DMS_H2SO4 = _M_H2SO4 / _M_DMS
_CONV_SO2_H2SO4 = _M_H2SO4 / _M_SO2

_TINY = 1.0e-30
_P_REF = 1.0e5   # Pa — BL weighting for the interim SOAG source

# --- Gas-phase oxidation rate-constant coefficients -----------------------
# These are *empirical* Arrhenius / Troe parameters from the laboratory
# kinetics literature (JPL/DeMore & IUPAC/Atkinson evaluations), reproduced
# verbatim from ECHAM-HAM (``mo_ham_chemistry.f90``). They are measured
# reaction coefficients — **not** physical constants — so they cannot be
# derived from :mod:`jcm.constants`; they are the fixed published values.
# Arrhenius form k = A·exp(−Ea/RT), written here as A·exp(±C/T) with C = Ea/R.

# DMS + OH (Atkinson). Two channels:
#   H-abstraction:  k1 = 9.6e-12·exp(−234/T)
#   OH-addition:    k2 = 1.7e-42·exp(7810/T)·[O₂] / (1 + 5.5e-31·exp(7460/T)·[O₂])
_DMS_OH_ABS_A, _DMS_OH_ABS_C = 9.6e-12, 234.0
_DMS_OH_ADD_A, _DMS_OH_ADD_C = 1.7e-42, 7810.0
_DMS_OH_ADD_DEN_A, _DMS_OH_ADD_DEN_C = 5.5e-31, 7460.0
# 1.7e-42 underflows float32, so its log is folded into the exponent below.
_LN_DMS_OH_ADD_A = math.log(_DMS_OH_ADD_A)
_O2_FRAC = 0.21   # molar fraction of O₂ in air

# SO₂ + OH + M → HSO₃ (→ H₂SO₄), Troe termolecular (JPL):
#   k0 = k0_300·(T/300)^−n (low-P limit), k_inf (high-P limit), Fc broadening.
_SO2_OH_K0_300, _SO2_OH_K0_N = 4.0e-31, 3.3
_SO2_OH_KINF = 2.0e-12
_SO2_OH_FC = 0.45   # Troe broadening factor (measured, not a tuning knob)

# DMS + NO₃ → HNO₃ + products (night), Atkinson:  k3 = 1.9e-13·exp(520/T)
_DMS_NO3_A, _DMS_NO3_C = 1.9e-13, 520.0


def _k_dms_oh(t: jnp.ndarray, n_air: jnp.ndarray):
    """DMS+OH abstraction (k1) and addition (k2) rates [cm³ molec⁻¹ s⁻¹]."""
    o2 = _O2_FRAC * n_air
    k1 = _DMS_OH_ABS_A * jnp.exp(-_DMS_OH_ABS_C / t)
    numerator = jnp.exp(_DMS_OH_ADD_C / t + _LN_DMS_OH_ADD_A) * o2
    k2 = numerator / (
        1.0 + _DMS_OH_ADD_DEN_A * jnp.exp(_DMS_OH_ADD_DEN_C / t) * o2
    )
    return k1, k2


def _k_so2_oh(t: jnp.ndarray, n_air: jnp.ndarray) -> jnp.ndarray:
    """SO₂+OH+M termolecular (Troe) rate [cm³ molec⁻¹ s⁻¹]."""
    k0 = _SO2_OH_K0_300 * (t / 300.0) ** (-_SO2_OH_K0_N)
    hil = jnp.maximum(k0 * n_air / _SO2_OH_KINF, _TINY)
    expo = 1.0 / (1.0 + jnp.log10(hil) ** 2)
    return k0 * n_air / (1.0 + hil) * _SO2_OH_FC ** expo


def _k_dms_no3(t: jnp.ndarray) -> jnp.ndarray:
    """DMS+NO₃ rate [cm³ molec⁻¹ s⁻¹]."""
    return _DMS_NO3_A * jnp.exp(_DMS_NO3_C / t)


@tree_math.struct
class SulfurGasParameters:
    """Tunable knobs for the gas-phase sulfur chemistry (differentiable)."""

    soag_production: jnp.ndarray   # prescribed SOAG source [kg/kg/s], BL-weighted

    @classmethod
    def default(cls) -> "SulfurGasParameters":
        # 0 by default: jcm has no biogenic VOC chemistry yet, so any
        # nonzero value here is an unopposed global SOA source. The
        # previous 2e-15 "interim" default (~0.9 mg SOA/m²/day column-
        # integrated) was the sole SOA source in every JAM run and, being
        # ~5× the total sulfur source with only weak accumulation-mode
        # removal against it, drove multi-hundred-mg/m² SOA burdens that
        # in turn distorted sulfate removal (jax-gcm#654). Until real
        # biogenic emissions land, JAM carries zero SOA unless a run
        # opts in explicitly.
        return cls(soag_production=jnp.asarray(0.0))


def sulfur_gas_tendencies(
    dms: jnp.ndarray,
    so2: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    oh: jnp.ndarray,
    no3: jnp.ndarray,
    dt: jnp.ndarray,
    soag_production: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Per-tracer gas-phase tendencies [kg/kg/s] for the sulfur chain.

    ``dms``/``so2`` are mass mixing ratios [kg/kg]; ``oh``/``no3`` number
    densities [molec cm⁻³]. Returns tendencies keyed by gas tracer name.
    """
    n_air = air_number_density(temperature, pressure)
    k1, k2 = _k_dms_oh(temperature, n_air)
    k_oh = (k1 + k2) * oh            # DMS+OH first-order loss [1/s]
    k_no3 = _k_dms_no3(temperature) * no3
    k_dms = k_oh + k_no3
    f_so2 = (k1 + 0.75 * k2) / jnp.maximum(k1 + k2, _TINY)  # OH→SO2 branch

    # DMS loss (exponential over dt), split by channel.
    dms_lost = dms * (1.0 - jnp.exp(-k_dms * dt))
    frac_oh = k_oh / jnp.maximum(k_dms, _TINY)
    dms_lost_oh = dms_lost * frac_oh
    dms_lost_no3 = dms_lost - dms_lost_oh

    so2_from_dms = (dms_lost_oh * f_so2 + dms_lost_no3) * _CONV_DMS_SO2
    h2so4_from_dms = dms_lost_oh * (1.0 - f_so2) * _CONV_DMS_H2SO4

    # SO2 + OH → H2SO4 (acts on SO2 plus the fresh DMS-derived SO2).
    k_so2 = _k_so2_oh(temperature, n_air) * oh
    so2_avail = so2 + so2_from_dms
    so2_lost = so2_avail * (1.0 - jnp.exp(-k_so2 * dt))
    h2so4_from_so2 = so2_lost * _CONV_SO2_H2SO4

    # Interim SOAG source: boundary-layer-weighted prescribed production.
    soag_src = soag_production * jnp.clip(pressure / _P_REF, 0.0, 1.0)

    return {
        gas_name("dms"): -dms_lost / dt,
        gas_name("so2"): (so2_from_dms - so2_lost) / dt,
        gas_name("h2so4"): (h2so4_from_dms + h2so4_from_so2) / dt,
        gas_name("soag"): soag_src,
    }


class SulfurGasChemistry(PhysicsTerm):
    """Gas-phase DMS/SO₂ oxidation → SO₂/H₂SO₄/SOAG, fed to the MAM4 core."""

    name: ClassVar[str] = "jam_sulfur_gas_chemistry"
    category: ClassVar[str] = "aerosol_gas_chemistry"
    requires: ClassVar[tuple[str, ...]] = ("oxidants", "pressure_full")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: SulfurGasParameters | None = None):
        """Hold the (differentiable) chemistry knobs."""
        self.params = nnx.Param(params or SulfurGasParameters.default())

    def required_tracers(self) -> tuple[TracerSpec, ...]:  # type: ignore[override]
        """Declare the four prognostic gas-phase precursor tracers."""
        return gas_tracer_specs(SULFUR_GASES)

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        zeros = jnp.zeros_like(state.temperature)
        ox = diagnostics["oxidants"]
        dt = jnp.asarray(diagnostics["_dt_seconds"], state.temperature.dtype)

        tends = sulfur_gas_tendencies(
            dms=state.tracers.get(gas_name("dms"), zeros),
            so2=state.tracers.get(gas_name("so2"), zeros),
            temperature=state.temperature,
            pressure=diagnostics["pressure_full"],
            oh=ox.oh,
            no3=ox.no3,
            dt=dt,
            soag_production=params.soag_production,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tends,
        )
        return tendency, diagnostics
