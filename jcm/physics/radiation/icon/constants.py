"""Constants for ICON radiation scheme.

The band definitions here are the single source of truth used by both
``gas_optics``/``planck``/``cloud_optics`` (which need Python ints at
trace time for static shapes / loop unrolls) and by
``RadiationParameters.default()`` (which converts them to jnp arrays
for runtime use). Keep both in sync by editing only this file.
"""

# Shortwave bands (wavenumber in cm⁻¹)
SW_BAND_LIMITS = (
    (4000, 14500),   # UV + visible
    (14500, 50000),  # Near-IR
)

# Longwave bands (wavenumber in cm⁻¹)
LW_BAND_LIMITS = (
    (10, 350),     # Far-IR + H2O rotation
    (350, 500),    # CO2 + H2O window
    (500, 2500),   # H2O continuum + O3
)

N_SW_BANDS = len(SW_BAND_LIMITS)
N_LW_BANDS = len(LW_BAND_LIMITS)
N_BANDS_TOTAL = N_SW_BANDS + N_LW_BANDS
