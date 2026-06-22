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

# Hand-tuned SPEEDY sigma half-level boundaries (length nlev+1, from TOA sigma=0
# to the surface sigma=1). These are the non-analytic tables shipped with the
# original SPEEDY for its standard 7- and 8-level configurations. They are kept
# as the EXACT special cases used whenever nlev is 7 or 8 so that all legacy
# behaviour and regression tests are bit-for-bit unchanged. For any other nlev,
# `compute_sigma_boundaries` falls back to the analytic Frierson (2006) stretch.
SIGMA_LAYER_BOUNDARIES = {
    # 5: jnp.array([0.0, 0.15, 0.35, 0.65, 0.9, 1.0]), # FIXME: not supported at the moment
    7: jnp.array([0.0, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
    8: jnp.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
}


def compute_sigma_boundaries(nlev: int) -> jnp.ndarray:
    """Return SPEEDY sigma half-level boundaries for an arbitrary level count.

    Returns an array of length ``nlev + 1`` running from the top of the
    atmosphere (sigma = 0) to the surface (sigma = 1).

    For ``nlev`` in {7, 8} this returns the hand-tuned SPEEDY tables verbatim
    (see :data:`SIGMA_LAYER_BOUNDARIES`), preserving the original model exactly.

    For any other ``nlev`` it uses the Frierson (2006) cubic stretch. This is
    the spacing SpeedyWeather.jl ships (commented out in
    ``src/dynamics/vertical_coordinates.jl``) explicitly for "higher resolution
    in surface boundary layer and in stratosphere", which reproduces the
    *qualitative* shape of SPEEDY's hand-tuned tables (fine near the surface and
    near the model top, coarse in the mid-troposphere). We prefer it over
    SpeedyWeather's equidistant default because SPEEDY's physics relies on a thin
    lowest layer (surface fluxes / PBL) and on the top two levels behaving like a
    stratosphere; an equidistant grid would put almost no resolution there.

        z      = linspace(1, 0, nlev + 1)          # 1 at surface end, 0 at top
        sigma  = exp(-5 * (0.05*z + 0.95*z**3))    # Frierson (2006) stretch
        sigma[0]  = 0.0                            # enforce exact TOA
        sigma[-1] = 1.0                            # enforce exact surface

    The result is strictly increasing for all ``nlev >= 1``. Note that at
    nlev = 7 or 8 this analytic profile does NOT equal the hand-tuned tables
    (they differ by tens of a sigma-unit mid-column), which is why the tables
    are retained as exact special cases rather than being replaced.

    Args:
        nlev: Number of vertical levels (>= 2).

    Returns:
        Array of sigma half-level boundaries, shape ``(nlev + 1,)``.

    """
    if nlev < 2:
        raise ValueError(f"SPEEDY physics requires nlev >= 2, got {nlev}")

    if nlev in SIGMA_LAYER_BOUNDARIES:
        return SIGMA_LAYER_BOUNDARIES[nlev]

    # Frierson (2006) cubic stretch. z runs 1 -> 0 so that exp(-5*...) maps the
    # surface end (z=1) toward sigma=1 and the top (z=0) toward sigma=0.
    z = jnp.linspace(1.0, 0.0, nlev + 1)
    sigma = jnp.exp(-5.0 * (0.05 * z + 0.95 * z**3))
    # Pin the endpoints exactly (the exp only reaches them approximately).
    sigma = sigma.at[0].set(0.0).at[-1].set(1.0)
    return sigma
