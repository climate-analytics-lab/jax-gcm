"""Initial-state helpers shared across dycores.

Analytic atmospheric profiles used to build starting states: the
U.S. Standard Atmosphere 1976 (:mod:`~jcm.initial_states.ussa1976`) here;
the JW-style lapse-rate and balanced-isothermal *injectors* live in
``jcm.runners`` (they mutate a built ``Model``) and can migrate here if a
dycore ever needs their profile math directly.
"""

from jcm.initial_states.ussa1976 import ussa_pressure, ussa_temperature

__all__ = ["ussa_pressure", "ussa_temperature"]
