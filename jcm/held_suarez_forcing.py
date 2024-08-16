from jcm.physics import PhysicsTendency, PhysicsState
from held_suarez import HeldSuarezForcing
'''
we need to return an instance of the Physics Tendency:

class PhysicsTendency:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray

Held-Saurez 1994

'''

def held_suarez_forcings(state: PhysicsState):
    hsf = HeldSuarezForcing()

    Teq = hsf.equilibrium_temperature(state.surface_pressure)
    d_temperature = -hsf.kt() * (state.surface_temperature - Teq)

    d_v_wind = -hsf.kv() * state.v_wind
    d_u_wind = -hsf.kv() * state.v_wind
    d_spec_humidity = 0 # just keep the same specific humidity?

    return PhysicsTendency(d_u_wind, d_v_wind, d_temperature, d_spec_humidity)