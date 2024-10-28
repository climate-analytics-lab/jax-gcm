import jax.numpy as jnp
# Radiation and cloud constants
# Albedo values
albsea = 0.07  # Albedo over sea
albice = 0.60  # Albedo over sea ice (for ice fraction = 1)
albsn  = 0.60  # Albedo over snow (for snow cover = 1)

# Longwave parameters
epslw  = 0.05  # Fraction of blackbody spectrum absorbed/emitted by PBL only
emisfc = 0.98  # Longwave surface emissivity

# Placeholder for CO2-related variable
ablco2_ref = None  # To be initialized elsewhere
fband = jnp.zeros((301, 4))
eps1 = 1.0 - epslw
t_min, t_max = 200, 320
jtemp = jnp.arange(t_min, t_max + 1)
fband_2 = (0.148 - 3.0e-6 * (jtemp - 247) ** 2) * eps1
fband_3 = (0.356 - 5.2e-6 * (jtemp - 282) ** 2) * eps1
fband_4 = (0.314 + 1.0e-5 * (jtemp - 315) ** 2) * eps1
fband_1 = eps1 - (fband_2 + fband_3 + fband_4)
fband = fband.at[jtemp - 100, :4].set(jnp.stack((fband_1, fband_2, fband_3, fband_4), axis=-1))

jb = jnp.arange(4)
fbandradset = fband.at[:(t_min - 100), jb].set(fband[t_min - 100, jb])
fbandradset = fbandradset.at[(t_max + 1 - 100):, jb].set(fband[t_max - 100, jb])



