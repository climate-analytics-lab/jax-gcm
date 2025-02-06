'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''
import tree_math
import jax.numpy as jnp
from jax import tree_util 

@tree_math.struct
class ConvectionParameters:
    psmin = jnp.array(0.8) # Minimum (normalised) surface pressure for the occurrence of convection
    trcnv = jnp.array(6.0) # Time of relaxation (in hours) towards reference state
    rhil = jnp.array(0.7) # Relative humidity threshold in intermeduate layers for secondary mass flux
    rhbl = jnp.array(0.9) # Relative humidity threshold in the boundary layer
    entmax = jnp.array(0.5) # Maximum entrainment as a fraction of cloud-base mass flux
    smf = jnp.array(0.8) # Ratio between secondary and primary mass flux at cloud-base

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
    
@tree_math.struct
class Parameters:
    convection: ConvectionParameters
    

# Time stepping parameters
nsteps = 36     # Number of time steps in one day
delt = 86400.0 / nsteps # Time step in seconds
rob = 0.05      # Damping factor in Robert time filter
wil = 0.53      # Parameter of Williams filter
alph = 0.5      # Coefficient for semi-implicit computations

# Physics parameters
iseasc = 1      # Seasonal cycle flag (0=no, 1=yes)
nstrad = 3      # Period (number of steps) for shortwave radiation
sppt_on = False # Turn on SPPT?
issty0 = 1979   # Starting year for SST anomaly file

# User-specified parameters
nstdia = 36 * 5 # Period (number of steps) for diagnostic print-out
nsteps_out = 1  # Number of time steps between outputs
