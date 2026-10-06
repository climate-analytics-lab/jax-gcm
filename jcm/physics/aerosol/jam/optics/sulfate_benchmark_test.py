"""Physical sulfate-mode benchmarks for the complete JAM optics pathway."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.optics.mie import mie_efficiencies
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.optics.refractive_index import refractive_index_at
from jcm.physics.aerosol.jam.tracer_layout import mass_name
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics_interface import PhysicsState


def _sulfate_population(rh):
    """Build one sulfate mode with mass, number and size sharing one moment."""
    rh = np.asarray(rh)
    shape = (1, rh.size)
    sigma = MAM4_SPEC.modes[0].geom_std_dev
    density = MAM4_SPEC.species_props("so4").density
    kappa = MAM4_SPEC.species_props("so4").hygroscopicity
    # 55 nm dry median radius is MAM4's reference accumulation diameter/2.
    dry_radius = 55e-9
    number = 1e9
    mass = number * 4 * math.pi / 3 * dry_radius**3 * np.exp(4.5 * np.log(sigma)**2) * density
    growth = (1 + kappa * rh / (1 - rh))**(1 / 3)
    rd = jnp.asarray([m.dgnum / 2 for m in MAM4_SPEC.modes])[:, None, None]
    rd = jnp.broadcast_to(rd, (4,) + shape).at[0].set(dry_radius)
    rw = rd.at[0].set(dry_radius * growth[None, :])
    n = jnp.zeros_like(rd).at[0].set(number)
    masses = jnp.zeros_like(rd).at[0].set(mass)
    aer = JamAerosolState(r_dry=rd, r_wet=rw, rho=jnp.full_like(rd, density),
                          kappa=jnp.full_like(rd, kappa), mass=masses, number=n)
    state = PhysicsState.zeros(shape).copy(
        tracers={mass_name("so4", "acc"): jnp.full(shape, mass)})
    term = JamOpticsTerm(optics_diagnostics=True)
    term.cache_band_config(RadiationBandConfig(
        sw_band_centers_nm=(550.0,), lw_band_centers_nm=()))
    return term, state, aer, mass, growth


def test_sulfate_mode_matches_dense_mie_at_dry_and_ambient_rh():
    """Check mass normalisation, water dilution, LUT and size integration.

    The reference integrates direct Mie over 128 nodes, independently of
    the online eight-node LUT path. The dry index is the ammonium-bisulfate
    value n=1.473 from Li et al. (2001), rather than the
    value returned by the implementation under test.
    """
    rh = np.array([0.0, 0.5, 0.8, 0.95])
    term, state, aer, mass, growth = _sulfate_population(rh)
    factor = jnp.ones(state.temperature.shape)
    out = term._optics_diagnostics_fields(state, aer, aer.number, factor, factor)
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    reference = []
    for g in growth:
        wet_index = (1.473 + 1.33 * (g**3 - 1)) / g**3
        radius = 55e-9 * g * np.exp(np.sqrt(2) * np.log(1.8) * nodes)
        qe = [mie_efficiencies(2 * np.pi * r / 550e-9, wet_index, 1e-8)[0]
              for r in radius]
        reference.append(1e9 * np.sum(weights / np.sqrt(np.pi) * np.pi * radius**2 * qe))
    np.testing.assert_allclose(np.asarray(out["od550aer"]), reference, rtol=0.06)
    # Ambient extinction at 80% RH per DRY mass is of order 8–11 m²/g.
    assert 8 < float(out["od550aer"][2]) / mass / 1000 < 11
    np.testing.assert_allclose(out["od550dryaer"], reference[0], rtol=0.06)
    dry_modes = sum(out[f"od550dry_mode_{m.short}"] for m in MAM4_SPEC.modes)
    np.testing.assert_allclose(dry_modes, out["od550dryaer"], rtol=1e-6)
    assert float(out["od550aer"][2] / out["od550dryaer"][2]) > 2
    # Neither a species share nor its remainder measures humidity enhancement.
    assert not np.isclose(float(out["od550_wat"][2]),
                          float(out["od550aer"][2] - out["od550dryaer"][2]),
                          rtol=1e-3, atol=0)


@pytest.mark.parametrize("wavelength", [355.0, 550.0, 865.0])
def test_dry_sulfate_surrogate_is_not_an_aqueous_index(wavelength):
    n, _ = refractive_index_at("so4", jnp.asarray(wavelength))
    assert float(n) == pytest.approx(1.473, abs=1e-6)


def test_dry_diagnostic_equals_ambient_without_water():
    term, state, aer, _, _ = _sulfate_population([0.0])
    factor = jnp.ones(state.temperature.shape)
    out = term._optics_diagnostics_fields(state, aer, aer.number, factor, factor)
    np.testing.assert_array_equal(out["od550dryaer"], out["od550aer"])
    np.testing.assert_array_equal(out["od550_wat"], 0)


def test_mass_normalization_survives_clipped_or_lagged_modal_number():
    """Optical mass must not change when a bounded radius decouples from N."""
    term, state, aer, _, _ = _sulfate_population([0.0, 0.8])
    factor = jnp.ones(state.temperature.shape)
    reference = term._optics_diagnostics_fields(state, aer, aer.number, factor, factor)
    for multiplier in (0.1, 10.0):
        changed = aer.copy(number=aer.number * multiplier)
        out = term._optics_diagnostics_fields(state, changed, changed.number, factor, factor)
        np.testing.assert_allclose(out["od550aer"], reference["od550aer"], rtol=1e-6)
    doubled = state.copy(tracers={k: 2*v for k, v in state.tracers.items()})
    out = term._optics_diagnostics_fields(doubled, aer, aer.number, factor, factor)
    np.testing.assert_allclose(out["od550aer"], 2*reference["od550aer"], rtol=1e-6)
