import jax.numpy as jnp
import tree_math
from datetime import datetime

@tree_math.struct
class DateData:
    tyear: jnp.ndarray # Fractional time of year, should possibly be part of the model itself (i.e. not in physics_data)
    model_steps: jnp.ndarray

    def __init__(self, model_time=None, model_steps=None) -> None:        
        self.tyear = fraction_of_year_elapsed(model_time) if model_time is not None else jnp.zeros((1))
        self.model_steps = model_steps if model_steps is not None else jnp.zeros((1))

    def copy(self, tyear=None, model_steps=None):
        copy = DateData()
        copy.tyear = tyear if tyear is not None else self.tyear
        copy.model_steps = model_steps if model_steps is not None else self.model_steps
        return copy

def fraction_of_year_elapsed(dt):
    """
    Calculate the fraction of the year that has elapsed at the given datetime.

    This deals with leap years by calculating the total time in the year and the time elapsed since the start of the year.
    Note, that I don't think the corresponding function in Speedy handles leap years (though the model time tracks them), 
    potentially leading to a bug there.
    """    
    start_of_year = datetime(dt.year, 1, 1)
    end_of_year = datetime(dt.year + 1, 1, 1)
    
    elapsed_time = (dt - start_of_year).total_seconds()
    total_time = (end_of_year - start_of_year).total_seconds()
    
    fraction_elapsed = elapsed_time / total_time
    return fraction_elapsed
