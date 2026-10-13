"""Initial-state helpers shared across dycores.

Analytic atmospheric profiles used to build starting states — the
U.S. Standard Atmosphere 1976 (:mod:`~jcm.initial_states.ussa1976`) — and
the state *builders* that return a starting dycore/physics state for a built
``Model``: a JW-style lapse-rate profile, a balanced-isothermal rest state,
an ERA5 slice, or a saved warm-start checkpoint
(:mod:`~jcm.initial_states.injectors`).

Hand the returned state to ``model.run(initial_state=...)``::

    predictions = model.run(initial_state=jw_state(model, rh=0.6), ...)

These are the library homes of the initial conditions the Hydra CLI exposes
as ``init.kind={jw,balanced_isothermal,era5,from_state}``; ``jcm.runners``
provides only the thin config adapters that call through here.

``era5_state`` is re-exported from :mod:`jcm.data.era5` for discoverability;
it is imported lazily so the (heavier) ERA5 dependencies aren't pulled in at
package import.
"""

from jcm.initial_states.injectors import (
    balanced_isothermal_state,
    checkpoint_state,
    jw_state,
)
from jcm.initial_states.ussa1976 import ussa_pressure, ussa_temperature

__all__ = [
    "balanced_isothermal_state",
    "checkpoint_state",
    "era5_state",
    "jw_state",
    "ussa_pressure",
    "ussa_temperature",
]


def __getattr__(name):
    # Lazy re-export: ``jcm.data.era5.initial_state`` already returns a
    # PhysicsState that ``model.run(initial_state=...)`` accepts; importing it
    # eagerly would pull ERA5/WeatherBench2 deps in at package import.
    if name == "era5_state":
        from jcm.data.era5 import initial_state as era5_state

        return era5_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
