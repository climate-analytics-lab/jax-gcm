"""Conservation tests for bounded ECHAM 1M ice sedimentation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from .echam_1m import MicrophysicsParameters, _ice_sedimentation_layer


_CUTOFF = 1.0e-10


@pytest.fixture(autouse=True, scope="module")
def _enable_x64_for_precision_checks():
    """Keep float64 references live without changing the rest of the suite."""
    with jax.enable_x64():
        yield


def _density_power_reference(ice_density, d_epsilon, cutoff):
    """Independent NumPy reference for the optional low-density join."""
    if cutoff == 0.0:
        return np.maximum(ice_density, d_epsilon) ** 0.16
    if ice_density < cutoff:
        t = ice_density / cutoff
        return cutoff**0.16 * (1.84 * t - 0.84 * t**2)
    return ice_density**0.16


def _sedimentation_reference(
    cloud_ice,
    incoming_ice_flux,
    air_density,
    layer_mass,
    dt,
    is_bottom,
    config,
    cutoff,
):
    """Evaluate the layer update independently in NumPy float64."""
    cloud_ice = max(float(cloud_ice), 0.0)
    ice_density = float(air_density) * cloud_ice
    if ice_density > 0.0 or cutoff > 0.0:
        power = _density_power_reference(
            ice_density, float(config.d_epsilon), cutoff)
        fall_speed = float(config.cvtfall) * power
    else:
        fall_speed = 0.0

    pressure_thickness = float(layer_mass) * float(c.grav)
    denominator = max(pressure_thickness, float(config.epsilon))
    sedimentation_number = (
        fall_speed * float(c.grav) * float(air_density) * float(dt)
        / denominator
    )
    if sedimentation_number > 1.0e-8:
        phi = -np.expm1(-sedimentation_number) / sedimentation_number
    else:
        phi = 1.0 - 0.5 * sedimentation_number
    provisional_ice = max(
        0.0,
        cloud_ice * np.exp(-sedimentation_number)
        + float(incoming_ice_flux) * float(c.grav) * float(dt)
        / denominator * phi,
    )
    mass_rate = float(layer_mass) / float(dt)
    through_flux = max(
        0.0,
        float(incoming_ice_flux)
        - (provisional_ice - cloud_ice) * mass_rate,
    )
    ice_change = (float(incoming_ice_flux) - through_flux) / mass_rate
    cloud_ice_out = cloud_ice + ice_change
    outgoing_flux = 0.0 if is_bottom else through_flux
    surface_flux = through_flux if is_bottom else 0.0
    return cloud_ice_out, ice_change / float(dt), outgoing_flux, surface_flux


@pytest.mark.parametrize("incoming_ice_flux", [0.0, 2.0e-6])
@pytest.mark.parametrize("cloud_ice", [0.0, 1.0e-12, 1.0e-5])
@pytest.mark.parametrize("cutoff", [0.0, _CUTOFF])
def test_layer_matches_reference_and_closes_ice_budget(
    incoming_ice_flux, cloud_ice, cutoff,
):
    """Resident ice and true incoming flux close one local mass ledger."""
    config = MicrophysicsParameters.default()
    air_density, layer_mass, dt = 0.7, 950.0, 900.0
    actual = _ice_sedimentation_layer(
        jnp.asarray(cloud_ice, dtype=jnp.float64),
        jnp.asarray(incoming_ice_flux, dtype=jnp.float64),
        jnp.asarray(air_density, dtype=jnp.float64),
        jnp.asarray(layer_mass, dtype=jnp.float64),
        dt,
        False,
        config,
        cutoff,
    )
    expected = _sedimentation_reference(
        cloud_ice, incoming_ice_flux, air_density, layer_mass, dt,
        False, config, cutoff,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), expected, rtol=2e-13, atol=1e-18)

    cloud_ice_out, dqidt_sed, outgoing_flux, surface_flux = actual
    np.testing.assert_allclose(surface_flux, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        layer_mass * dqidt_sed,
        incoming_ice_flux - outgoing_flux,
        rtol=2e-13,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        cloud_ice_out, cloud_ice + dt * dqidt_sed,
        rtol=2e-13, atol=1e-18)


def test_positive_influx_deposits_into_an_ice_free_layer():
    """The zero-speed removable limit retains, rather than drops, influx."""
    config = MicrophysicsParameters.default()
    incoming_flux = 2.0e-6
    cloud_ice_out, dqidt_sed, outgoing_flux, _ = _ice_sedimentation_layer(
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(incoming_flux, dtype=jnp.float64),
        jnp.asarray(0.7, dtype=jnp.float64),
        jnp.asarray(950.0, dtype=jnp.float64),
        900.0,
        False,
        config,
        _CUTOFF,
    )
    assert float(cloud_ice_out) > 0.0
    assert float(dqidt_sed) > 0.0
    np.testing.assert_allclose(outgoing_flux, 0.0, rtol=0.0, atol=1e-18)


@pytest.mark.parametrize("cutoff", [0.0, _CUTOFF])
def test_multilayer_ice_budget_telescopes_to_surface_flux(cutoff):
    """Internal through-fluxes cancel from the complete column ledger."""
    config = MicrophysicsParameters.default()
    cloud_ice = np.array([2.0e-7, 1.0e-12, 7.0e-6, 0.0])
    air_density = np.array([0.35, 0.5, 0.75, 1.0])
    layer_mass = np.array([220.0, 410.0, 730.0, 1050.0])
    dt = 900.0
    incoming_flux = jnp.asarray(0.0, dtype=jnp.float64)
    column_mass_tendency = 0.0
    surface_flux = 0.0

    for level in range(cloud_ice.size):
        is_bottom = level == cloud_ice.size - 1
        _, dqidt_sed, outgoing_flux, bottom_flux = _ice_sedimentation_layer(
            jnp.asarray(cloud_ice[level], dtype=jnp.float64),
            incoming_flux,
            jnp.asarray(air_density[level], dtype=jnp.float64),
            jnp.asarray(layer_mass[level], dtype=jnp.float64),
            dt,
            is_bottom,
            config,
            cutoff,
        )
        column_mass_tendency += layer_mass[level] * float(dqidt_sed)
        incoming_flux = outgoing_flux
        surface_flux += float(bottom_flux)

    np.testing.assert_allclose(
        column_mass_tendency + surface_flux, 0.0,
        rtol=0.0, atol=2e-18)


def test_default_layer_map_matches_the_legacy_density_power():
    """The default path retains the pre-continuation trace-ice formula."""
    config = MicrophysicsParameters.default()
    inputs = (3.0e-14, 1.3e-6, 0.62, 680.0, 900.0, False, config)
    actual = _ice_sedimentation_layer(*inputs, continuation_cutoff=0.0)
    expected = _sedimentation_reference(*inputs, cutoff=0.0)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), expected, rtol=2e-13, atol=1e-18)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("cloud_ice", [0.0, 1.0e-12, 1.0e-5])
def test_bounded_layer_map_has_finite_jvp_and_vjp(dtype, cloud_ice):
    """The actual transport map has finite forward and reverse sensitivities."""
    config = MicrophysicsParameters.default()

    def outputs(qi):
        return jnp.stack(_ice_sedimentation_layer(
            qi,
            jnp.asarray(2.0e-6, dtype=dtype),
            jnp.asarray(0.7, dtype=dtype),
            jnp.asarray(950.0, dtype=dtype),
            900.0,
            False,
            config,
            _CUTOFF,
        )[:3])

    qi = jnp.asarray(cloud_ice, dtype=dtype)
    primal, tangent = jax.jvp(outputs, (qi,), (jnp.ones_like(qi),))
    _, pullback = jax.vjp(outputs, qi)
    cotangent = pullback(jnp.ones_like(primal))[0]
    assert jnp.all(jnp.isfinite(primal))
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.isfinite(cotangent)
