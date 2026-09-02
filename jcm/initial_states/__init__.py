"""Initial-state helpers shared across dycores.

Analytic atmospheric profiles used to build starting states — the
U.S. Standard Atmosphere 1976 (:mod:`~jcm.initial_states.ussa1976`) — and
the *injectors* that replace a built ``Model``'s initial dycore state with a
JW-style lapse-rate profile, a balanced-isothermal rest state, an ERA5 slice,
or a saved warm-start checkpoint (:mod:`~jcm.initial_states.injectors`).

The injectors are the library homes of the initial conditions the Hydra CLI
exposes as ``init.kind={jw,balanced_isothermal,era5,from_state}``;
``jcm.runners`` provides only the thin config adapters that call through here.
"""

from jcm.initial_states.injectors import (
    inject_balanced_isothermal_profile,
    inject_checkpoint_state,
    inject_era5_state,
    inject_jw_profile,
)
from jcm.initial_states.ussa1976 import ussa_pressure, ussa_temperature

__all__ = [
    "inject_balanced_isothermal_profile",
    "inject_checkpoint_state",
    "inject_era5_state",
    "inject_jw_profile",
    "ussa_pressure",
    "ussa_temperature",
]
