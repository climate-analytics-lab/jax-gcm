from __future__ import annotations

import jax.numpy as jnp
import tree_math

_DAYS_YEAR = 365.2425

@tree_math.struct
class DateData:
    tyear: jnp.float32 # Fractional time of year, should possibly be part of the model itself (i.e. not in physics_data)
    model_year: jnp.int32
    model_step: jnp.int32
    dt_seconds: jnp.float32 # Model timestep in seconds

    @classmethod
    def zeros(cls, model_time=None, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=fraction_of_year_elapsed(model_time) if model_time is not None else 0.0,
          model_year=model_year if model_year is not None else 1950,
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else jnp.float32(1800.0))

    @classmethod
    def set_date(cls, model_time, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=fraction_of_year_elapsed(model_time),
          model_year=model_year if model_year is not None else 1950,
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else jnp.float32(1800.0))

    @classmethod
    def ones(cls, model_time=None, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=fraction_of_year_elapsed(model_time) if model_time is not None else 1.0,
          model_year=model_year if model_year is not None else 1950,
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else jnp.float32(1800.0))

    def model_day(self):
        return jnp.round(self.tyear*_DAYS_YEAR).astype(jnp.int32)

    def copy(self, tyear=None, model_year=None, model_step=None, dt_seconds=None):
        return DateData(
          tyear=tyear if tyear is not None else self.tyear,
          model_year=model_year if model_year is not None else self.model_year,
          model_step=model_step if model_step is not None else self.model_step,
          dt_seconds=dt_seconds if dt_seconds is not None else self.dt_seconds)

def fraction_of_year_elapsed(dt):
    """
    Calculate the fraction of the year that has elapsed at the given datetime.

    This deals with leap years by just assuming that every year has 365.2425 days. This is a simplification, but it should be close
    enough for most purposes (especially just e.g. annually varying solar radiation calculations). Speedy does something similar.

    Args:
        dt: A Datetime JAX object
    """

    # Get days elapsed since start of year, without using non-traceable datetime64
    days_elapsed_in_year = dt.delta.days % _DAYS_YEAR
    
    # Add the seconds to the days elapsed
    days_elapsed_in_year += dt.delta.seconds / (24 * 60 * 60)
    
    # Calculate the fraction of the year elapsed
    return jnp.float32(days_elapsed_in_year / _DAYS_YEAR)
