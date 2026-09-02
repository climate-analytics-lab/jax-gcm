"""Library-level tests for the initial-state injectors.

The JW and balanced-isothermal *profile* injectors are exercised through
``jcm.runners`` (which re-exports them) in ``runners_test.py``; these tests
cover the two injectors that the runner adapters wrap rather than mirror:
the ERA5 seed (no network access) and the checkpoint warm-start's clock reset.
"""

import numpy as np
import pytest

from jcm.initial_states import inject_checkpoint_state, inject_era5_state


def _held_suarez_model():
    from jcm.model import Model
    from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
    from jcm.physics.held_suarez.utils import get_held_suarez_coords

    coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
    model = Model(coords=coords, physics=held_suarez_physics(), time_step=180)
    return model


def test_inject_era5_state_seeds_from_synthetic_slice(monkeypatch):
    """inject_era5_state populates the dycore state from era5.initial_state.

    The WeatherBench2 fetch is monkeypatched to a synthetic, network-free
    PhysicsState so the test asserts the injector wiring: it must build
    ``_final_dycore_state`` from the returned slice and round-trip the
    injected temperature back through the state bridge.
    """
    import jax.numpy as jnp

    import jcm.data.era5 as era5
    from jcm.physics_interface import PhysicsState

    model = _held_suarez_model()
    nlon, nlat = model.coords.horizontal.nodal_shape
    nlev = model.coords.vertical.centers.size

    # Horizontally-uniform, level-varying temperature: a constant field per
    # level round-trips through the spectral transform exactly, so the
    # bridge round-trip below is a clean equality check independent of the
    # dycore's vertical ordering.
    t_profile = 250.0 + 5.0 * np.arange(nlev, dtype=np.float64)
    temperature = jnp.asarray(
        np.broadcast_to(t_profile[:, None, None], (nlev, nlon, nlat)))
    zeros3d = jnp.zeros((nlev, nlon, nlat))

    synthetic = PhysicsState(
        u_wind=zeros3d,
        v_wind=zeros3d,
        temperature=temperature,
        specific_humidity=zeros3d,
        geopotential=zeros3d,
        normalized_surface_pressure=jnp.ones((nlon, nlat)),
    )

    captured = {}

    def fake_initial_state(coords, date, **kwargs):
        captured["date"] = date
        captured["coords"] = coords
        return synthetic

    monkeypatch.setattr(era5, "initial_state", fake_initial_state)

    inject_era5_state(model, "2001-05-05")

    assert model._final_dycore_state is not None
    assert captured["date"] == "2001-05-05"
    assert captured["coords"] is model.coords

    physics_state = model.dycore.to_physics_state(model._final_dycore_state)
    round_tripped = np.asarray(physics_state.temperature)
    np.testing.assert_allclose(
        round_tripped, np.asarray(temperature), atol=1e-3)


def test_inject_checkpoint_state_resets_clock(tmp_path):
    """inject_checkpoint_state loads a donor state but zeros its sim_time.

    A donor checkpoint written with a nonzero elapsed sim-time must load its
    fields yet reset the dycore clock to zero (dates, forcing interpolation
    and output timestamps all derive from sim_time). The returned donor-day
    count is reported for logging.
    """
    import jax.numpy as jnp

    from jcm.checkpoint import save_checkpoint

    donor = _held_suarez_model()
    donor.bootstrap_state()
    # Stamp the donor state with a nonzero clock so the reset is observable.
    donor_sim_time = donor.dycore.sim_time(donor._final_dycore_state)
    donor._final_dycore_state = donor.dycore.with_sim_time(
        donor._final_dycore_state,
        jnp.full_like(donor_sim_time, 5.0 * 86400.0),
    )
    ckpt = tmp_path / "donor.ckpt"
    save_checkpoint(donor, ckpt, elapsed_days=5.0)

    warm = _held_suarez_model()
    days = inject_checkpoint_state(warm, str(ckpt))

    assert days == pytest.approx(5.0)
    reset_sim_time = np.asarray(warm.dycore.sim_time(warm._final_dycore_state))
    np.testing.assert_allclose(reset_sim_time, 0.0)
