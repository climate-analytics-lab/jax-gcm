from __future__ import annotations

import jax.numpy as jnp
import jax_datetime as jdt
import tree_math

### NOTE, the below code is taken verbatim from the NeuralGCM experimental branch and should be
### imported from there (or wherever it ends up) once it is stable
import dataclasses
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

Timedelta = jdt.Timedelta

_UNIX_EPOCH = np.datetime64('1970-01-01T00:00:00', 's')

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Timestamp:
  """JAX compatible timestamp, stored as a delta from the Unix epoch.

  The easiest way to create a Timestamp is to use `from_datetime64`, which
  supports `np.datetime64` objects and NumPy arrays with a datetime64 dtype:

    >>> Timestamp.from_datetime64(np.datetime64('1970-01-02'))
    Timestamp(delta=Timedelta(days=1, seconds=0))
  """

  delta: Timedelta

  @classmethod
  def from_datetime64(cls, values: np.datetime64 | np.ndarray) -> Timestamp:
    return cls(Timedelta.from_timedelta64(values - _UNIX_EPOCH))

  @classmethod
  def from_datetime(cls, datetime) -> Timestamp:
    return cls.from_datetime64(np.datetime64(datetime))

  def to_datetime64(self) -> np.timedelta64 | np.ndarray:
    return self.delta.to_timedelta64() + _UNIX_EPOCH

  def __add__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    return Timestamp(self.delta + other)

  __radd__ = __add__

  def __sub__(self, other):
    if isinstance(other, Timestamp):
      return self.delta - other.delta
    elif isinstance(other, Timedelta):
      return Timestamp(self.delta - other)
    else:
      return NotImplemented

  def tree_flatten(self):
    leaves = (self.delta,)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None
    return cls(*leaves)

### END OF CODE TAKEN FROM NeuralGCM experimental branch

@tree_math.struct
class DateData:
    tyear: jnp.float32 # Fractional time of year, should possibly be part of the model itself (i.e. not in physics_data)
    model_year: jnp.int32
    model_step: jnp.int32
    dt_seconds: jnp.int32 # Model timestep in seconds

    @classmethod
    def zeros(cls, model_time=None, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=fraction_of_year_elapsed(model_time) if model_time is not None else 0.0,
          model_year=model_year if model_year is not None else 1950,
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else jnp.int32(1800))

    @classmethod
    def set_date(cls, model_time, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=fraction_of_year_elapsed(model_time),
          model_year=model_year if model_year is not None else 1950,
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else jnp.int32(1800))

    @classmethod
    def ones(cls, model_time=None, model_year=None, model_step=None, dt_seconds=None):
        return cls(
          tyear=fraction_of_year_elapsed(model_time) if model_time is not None else 1.0,
          model_year=model_year if model_year is not None else 1950,
          model_step=model_step if model_step is not None else jnp.int32(0),
          dt_seconds=dt_seconds if dt_seconds is not None else jnp.int32(1800))

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
        dt: A Timestamp JAX object
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
