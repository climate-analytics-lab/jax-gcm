"""Library-level tests for the initial-state builders.

The JW and balanced-isothermal *profile* builders are exercised through
``jcm.runners`` (which re-exports them) in ``runners_test.py``; these tests
cover the two entry points that the runner adapters wrap rather than mirror:
the ERA5 re-export (no network access) and the checkpoint warm-start's
clock reset.
"""

import numpy as np
import pytest


def _held_suarez_model():
    from jcm.model import Model
    from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
    from jcm.physics.held_suarez.utils import get_held_suarez_coords

    coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
    model = Model(coords=coords, physics=held_suarez_physics(), time_step=180)
    return model


def test_era5_state_reexport_is_accepted_by_run(monkeypatch):
    """``era5_state`` re-exports ``jcm.data.era5.initial_state``, and the
    ``PhysicsState`` it returns is accepted by ``model.run``.

    The WeatherBench2 fetch is monkeypatched to a synthetic, network-free
    ``PhysicsState``. The test asserts (a) the lazy re-export resolves to
    ``era5.initial_state`` and (b) ``model.run(initial_state=...)`` integrates
    the returned gridpoint state without producing non-finite fields.
    """
    import jax.numpy as jnp

    import jcm.data.era5 as era5
    import jcm.initial_states as initial_states
    from jcm.physics_interface import PhysicsState

    model = _held_suarez_model()
    nlon, nlat = model.coords.horizontal.nodal_shape
    nlev = model.coords.vertical.centers.size

    # Horizontally-uniform, level-varying temperature: a physically-plausible
    # seed that projects cleanly onto the dycore.
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

    # The lazy re-export resolves to the (now-patched) era5.initial_state.
    assert initial_states.era5_state is era5.initial_state

    state = initial_states.era5_state(model.coords, "2001-05-05")
    assert captured["date"] == "2001-05-05"
    assert captured["coords"] is model.coords
    assert isinstance(state, PhysicsState)

    # model.run accepts the returned gridpoint state and integrates it.
    dt_days = 180.0 / 86400.0
    model.run(initial_state=state, save_interval=dt_days, total_time=dt_days)
    final = model.dycore.to_physics_state(model._final_dycore_state)
    assert np.all(np.isfinite(np.asarray(final.temperature)))


def test_checkpoint_state_returns_state_and_resets_clock(tmp_path):
    """``checkpoint_state`` returns ``(state, donor_days)`` with a zeroed clock.

    A donor checkpoint written with a nonzero elapsed sim-time must load its
    fields yet reset the *returned* dycore clock to zero (dates, forcing
    interpolation and output timestamps all derive from sim_time). The donor's
    elapsed-day count is returned for logging.
    """
    import jax.numpy as jnp

    from jcm.checkpoint import save_checkpoint
    from jcm.initial_states import checkpoint_state

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
    state, days = checkpoint_state(warm, str(ckpt))

    assert days == pytest.approx(5.0)
    # The RETURNED state carries the zeroed clock.
    reset_sim_time = np.asarray(warm.dycore.sim_time(state))
    np.testing.assert_allclose(reset_sim_time, 0.0)
