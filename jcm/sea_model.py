import jax.numpy as jnp
from jcm.params import ix, il

tsea = 290. * jnp.ones((il, ix)) #ssts TODO: port sea_model.f90