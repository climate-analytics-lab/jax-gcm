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


# --- Numerical-stability time step at high vertical resolution ---------------
#
# Background (see SPEEDY_VARIABLE_LEVELS.md "Numerical stability at high nlev"):
# the model goes non-finite within a few timesteps at high nlev / high
# truncation. The diagnosed cause is an *explicit (forward-Euler) surface-drag*
# instability in the thinnest bottom sigma layer, NOT a dynamics CFL problem
# (the dry dynamical core alone is stable at dt=30 min and nlev=32).
#
# SPEEDY's surface stress is applied as a tendency
#     du/dt|_sfc = ustr * rps * grdsig[-1],   grdsig[-1] = g / (dsigma_bot * p0)
# with ustr = -C_drag * |V| * u (see jcm/physics/surface/speedy_surface_flux.py).
# The Frierson stretch makes the bottom layer thin (dsigma_bot ~0.10 at nlev=8
# but ~0.008 at nlev=32), so grdsig[-1] grows ~12x and the explicit drag
# damping factor dt * C_drag*|V| * grdsig[-1] crosses the forward-Euler
# stability limit, producing a growing oscillation that blows up in the surface
# layer first. Higher truncation makes it worse because the resolved
# near-surface |V| (and the Gibbs ringing of the surface-drag tendency) grows
# with the number of horizontal modes.
#
# Rather than make every thin-layer surface tendency implicit (a large, invasive
# change to the operator-split coupling), we reduce the model time step when the
# configuration is severe. Empirically (T21/T31 x nlev=8..32 spinups) the
# stability boundary is ordered monotonically by the dimensionless severity
#     S = (trunc + 1)**1.3 / dsigma_bot
# normalised to the canonical-stable T21/nlev=8 baseline. All configurations
# that are stable at 30 min sit below S_norm ~= 9.4; every unstable one sits
# above it. We keep dt = 30 min on that plateau (so every currently-stable
# config -- including all 7/8-level runs -- is bit-for-bit unchanged) and shrink
# dt ~ 1/S above it, with a safety factor for margin below the measured blow-up
# boundary.
_DT_REFERENCE_MINUTES = 30.0   # validated dt at the T21/nlev=8 baseline
_DT_SEVERITY_EXPONENT = 1.3    # truncation weighting in the severity number
_DT_SEVERITY_PLATEAU  = 9.4    # S_norm below which dt stays at the reference
_DT_SEVERITY_SAFETY   = 0.85   # margin applied on the reduced branch

# Candidate time steps (minutes): integer divisors of a 1440-minute day, in
# descending order. Snapping the raw stability step DOWN to one of these keeps
# every saved frame aligned to an exact whole number of steps (the trajectory
# builder uses ``inner_steps = int(save_interval / dt)``), so daily/monthly
# save intervals don't drift over a 365-day run. Snapping down (never up) also
# means the snapped step is always <= the raw stability step, so it cannot
# re-introduce the instability.
_DT_DAY_DIVISORS_MINUTES = (30.0, 24.0, 20.0, 18.0, 16.0, 15.0, 12.0, 10.0,
                            9.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)


def _bottom_layer_thickness(nlev: int) -> float:
    boundaries = compute_sigma_boundaries(nlev)
    return float(boundaries[-1] - boundaries[-2])


# Reference severity for the canonical-stable T21 (trunc=21), nlev=8 run.
_S_REFERENCE = (21 + 1) ** _DT_SEVERITY_EXPONENT / _bottom_layer_thickness(8)


def stable_time_step_minutes(nlev: int, spectral_truncation: int) -> float:
    """Numerically-stable model time step (minutes) for a SPEEDY configuration.

    Returns a time step that keeps a realistic SPEEDY spin-up finite over long
    integrations. The binding constraint at high vertical resolution is the
    *explicit surface-drag* tendency in the thin bottom sigma layer (see the
    module-level note above), whose forward-Euler stability limit is crossed as
    ``dsigma_bot`` shrinks (high ``nlev``) and as the resolved near-surface wind
    grows (high ``spectral_truncation``).

    The criterion is a CFL-like rule on the dimensionless severity

        S = (spectral_truncation + 1) ** 1.3 / dsigma_bot

    normalised to the validated T21/nlev=8 baseline. On the stable plateau
    (``S_norm <= 9.4``) the reference 30-minute step is returned unchanged, so
    every configuration that is already stable at 30 min -- in particular all of
    SPEEDY's standard 7- and 8-level runs -- is bit-for-bit unaffected. Above the
    plateau the step is reduced as ``dt ~ 1/S`` with a 0.85 safety factor for
    margin below the empirically measured blow-up boundary.

    Args:
        nlev: Number of vertical levels.
        spectral_truncation: Triangular spectral truncation (e.g. 21 for T21).

    Returns:
        Time step in minutes.

    """
    s_norm = (
        (spectral_truncation + 1) ** _DT_SEVERITY_EXPONENT
        / _bottom_layer_thickness(nlev)
    ) / _S_REFERENCE
    if s_norm <= _DT_SEVERITY_PLATEAU:
        raw = _DT_REFERENCE_MINUTES
    else:
        raw = (
            _DT_REFERENCE_MINUTES * _DT_SEVERITY_SAFETY
            * _DT_SEVERITY_PLATEAU / s_norm
        )
    # Snap DOWN to a divisor of a 1440-minute day so saved frames stay aligned
    # (and the snapped step is never larger than the stability bound).
    for candidate in _DT_DAY_DIVISORS_MINUTES:
        if candidate <= raw:
            return candidate
    return _DT_DAY_DIVISORS_MINUTES[-1]
