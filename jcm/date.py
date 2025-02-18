from __future__ import annotations

import jax.numpy as jnp
import tree_math
from datetime import datetime
from jax import tree_util

### NOTE, the below code is taken verbatim from the NeuralGCM experimental branch and should be 
### imported from there (or wherever it ends up) once it is stable
import dataclasses
import jax
from typing import Any, Callable, Generic, TypeVar
import numpy as np

days_year = 365.25

# Generic types.
#
Dtype = jax.typing.DTypeLike | Any
Array = np.ndarray | jax.Array
Numeric = float | int | Array
Timestep = np.timedelta64 | float
PRNGKeyArray = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Timedelta:
  """JAX compatible time duration, stored in days and seconds.

  Like datetime.timedelta, the Timedelta constructor and arithmetic operations
  normalize seconds to fall in the range [0, 24 * 60 * 60). Timedelta objects
  are pytrees, but normalization is skipped inside jax.tree operations because
  JAX uses pytrees with non-numeric types to implement JAX transformations.

  Using integer days and seconds is recommended to avoid loss of precision. With
  int32 days, Timedelta can exactly represent durations over 5 million years.

  The easiest way to create a Timedelta is to use `from_timedelta64`, which
  supports `np.timedelta64` objects and NumPy arrays with a timedelta64 dtype:

    >>> Timedelta.from_timedelta64(np.timedelta64(1, 's'))
    Timedelta(days=0, seconds=1)
  """

  days: Numeric = 0
  seconds: Numeric = 0

  # TODO(shoyer): can we rewrite this a custom JAX dtype, like jax.random.key?

  def __post_init__(self):
    days_delta, seconds = divmod(self.seconds, 24 * 60 * 60)
    self.days = self.days + days_delta
    self.seconds = seconds

  @classmethod
  def from_timedelta64(cls, values: np.timedelta64 | np.ndarray) -> Timedelta:
    seconds = values // np.timedelta64(1, 's')
    # no need to worry about overflow, because timedelta64 represents values
    # internally with int64 and normalization uses native array operations
    return Timedelta(0, seconds)

  def to_timedelta64(self) -> np.timedelta64 | np.ndarray:
    seconds = np.int64(self.days) * 24 * 60 * 60 + np.int64(self.seconds)
    return seconds * np.timedelta64(1, 's')

  def __add__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    days = self.days + other.days
    seconds = self.seconds + other.seconds
    return Timedelta(days, seconds)

  def __neg__(self):
    return Timedelta(-self.days, -self.seconds)

  def __sub__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    return self + (-other)

  def __mul__(self, other):
    if not isinstance(other, Numeric):
      return NotImplemented
    return Timedelta(self.days * other, self.seconds * other)

  __rmul__ = __mul__

  # TODO(shoyer): consider adding other methods supported by datetime.timedelta.

  def tree_flatten(self):
    leaves = (self.days, self.seconds)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None
    # JAX uses non-numeric values for pytree leaves inside transformations, so
    # we skip __post_init__ by constructing the object directly:
    # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
    result = object.__new__(cls)
    result.days, result.seconds = leaves
    return result


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

    @classmethod
    def zeros(self, model_time=None, model_year=None):        
        return DateData(
          tyear=fraction_of_year_elapsed(model_time) if model_time is not None else 0.0,
          model_year=model_year if model_year is not None else 1950)
    
    @classmethod
    def set_date(self, model_time, model_year=None):        
        return DateData(
          tyear=fraction_of_year_elapsed(model_time),
          model_year=model_year if model_year is not None else 1950)
    
    @classmethod
    def ones(self, model_time=None, model_year=None):        
        return DateData(
          tyear=fraction_of_year_elapsed(model_time) if model_time is not None else 1.0,
          model_year=model_year if model_year is not None else 1950)

    def model_day(self):
        return jnp.round(self.tyear*days_year).astype(jnp.int32)

    def copy(self, tyear=None, model_year=None):
        return DateData(
          tyear=tyear if tyear is not None else self.tyear,
          model_year=model_year if model_year is not None else self.model_year)


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
