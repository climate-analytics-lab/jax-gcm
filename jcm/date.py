from __future__ import annotations

import jax.numpy as jnp
import tree_math

### NOTE, the below code is taken verbatim from the NeuralGCM experimental branch and should be
### imported from there (or wherever it ends up) once it is stable
import jax
from typing import Any
import numpy as np

days_year = 365.25

# Generic types.
#
Dtype = jax.typing.DTypeLike | Any
Array = np.ndarray | jax.Array
Numeric = float | int | Array
Timestep = np.timedelta64 | float
PRNGKeyArray = jax.Array

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
        return jnp.round(self.tyear*days_year).astype(jnp.int32)

    def copy(self, tyear=None, model_year=None, model_step=None, dt_seconds=None):
        return DateData(
          tyear=tyear if tyear is not None else self.tyear,
          model_year=model_year if model_year is not None else self.model_year,
          model_step=model_step if model_step is not None else self.model_step,
          dt_seconds=dt_seconds if dt_seconds is not None else self.dt_seconds)


def fraction_of_year_elapsed(dt):
    """
    Calculate the fraction of the year that has elapsed at the given datetime.

    This deals with leap years by just assuming that every year has 365.25 days. This is a simplification, but it should be close
    enough for most purposes (especially just e.g. annually varying solar radiation calculations). Speedy does something similar.

    Args:
        dt: A Datetime JAX object
    """

    # Get the year without using the `to_datetime64` method to avoid the need for a JAX transformation
    # Convert the number of days since 1970 into a year, accounting for leap years
    year = 1970 + dt.delta.days // days_year

    # Calculate the number of days elapsed in the year without using numpy
    days_elapsed = dt.delta.days - (year - 1970) * days_year
    
    # Add the seconds to the days elapsed
    days_elapsed += dt.delta.seconds / (24 * 60 * 60)

    # Calculate the fraction of the year elapsed
    return jnp.float32(days_elapsed / days_year)
