"""Constants for ICON radiation scheme.

This module contains physical and numerical constants used throughout
the ICON radiation implementation.

The band counts here MUST match the defaults in RadiationParameters
(n_sw_bands=2, n_lw_bands=3) and the band limits used by the
gas-optics routines.  A previous version defined 6 SW and 8 LW
fine-resolution bands, but the absorption coefficients in gas_optics.py
were tuned for the coarser 2 SW / 3 LW structure, causing a shape
mismatch and dramatically underestimated greenhouse effect.
"""

# Spectral bands — must match RadiationParameters.default()
N_SW_BANDS = 2  # Shortwave bands
N_LW_BANDS = 3  # Longwave bands
N_BANDS_TOTAL = N_SW_BANDS + N_LW_BANDS

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