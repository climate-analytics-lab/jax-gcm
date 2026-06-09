"""SPEEDY physics constants.

This module holds ONLY the constants that are genuinely SPEEDY-specific — those
that differ in value/units from the shared set, plus scheme tunables with no
shared counterpart. General physical constants (radius, gravity, gas/heat
constants, Stefan-Boltzmann, ...) are NOT re-exported here: read them directly
from :mod:`jcm.constants` (``import jcm.constants as c; c.grav``) so there is a
single source of truth and runtime ``set_constants`` overrides are honoured.

Original module date: 1/25/2024.
"""
import jax.numpy as jnp

# --- SPEEDY-specific constants that intentionally DIFFER from jcm.constants ---
# These are not duplicates to be unified — the differing value/units are part of
# SPEEDY's formulation:
#   * Latent heats are in J/g (not J/kg) because SPEEDY carries specific humidity
#     in g/kg; the shared c.alhc / c.alhs are the SI J/kg values.
#   * solc is the area-averaged insolation (S0/4) used directly by SPEEDY's
#     shortwave scheme, not the TOA solar constant c.solc (≈1361 W/m²).
#   * epsilon is SPEEDY's gradient-safety floor (1e-9), looser than the shared
#     numerical epsilon (1e-12).
alhc = 2501.0       # Latent heat of condensation (J/g)
alhs = 2801.0       # Latent heat of sublimation (J/g)
solc = 342.0        # Area-averaged solar input (W/m²)
epssw = 0.020       # Fraction of incoming solar radiation absorbed by ozone
epsilon = 1e-9      # Gradient-safety floor for SPEEDY physics

# --- SPEEDY scheme tunables (no shared counterpart) --------------------------
gamma  = 6.0       # Reference temperature lapse rate (-dT/dz in deg/km)
hscale = 7.5       # Reference scale height for pressure (in km)
hshum  = 2.5       # Reference scale height for specific humidity (in km)
refrh1 = 0.7       # Reference relative humidity of near-surface air
thd    = 2.4       # Max damping time (in hours) for horizontal diffusion
                                             # (del^6) of temperature and vorticity
thdd   = 2.4       # Max damping time (in hours) for horizontal diffusion
                                             # (del^6) of divergence
thds   = 12.0      # Max damping time (in hours) for extra diffusion
                                             ## (del^2) in the stratosphere
tdrs   = 24.0*30.0 # Damping time (in hours) for drag on zonal-mean wind
                                             # in the stratosphere

# Land model parameters moved here since they are only used in boundaries preprocessing
sd2sc = 60.0 # Snow depth (mm water) corresponding to snow cover = 1
swcap = 0.30 # Soil wetness at field capacity (volume fraction)
swwil = 0.17 # Soil wetness at wilting point  (volume fraction)

nstrad = 3 # number of timesteps between shortwave evaluations

SIGMA_LAYER_BOUNDARIES = {
    # 5: jnp.array([0.0, 0.15, 0.35, 0.65, 0.9, 1.0]), # FIXME: not supported at the moment
    7: jnp.array([0.0, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
    8: jnp.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
}
