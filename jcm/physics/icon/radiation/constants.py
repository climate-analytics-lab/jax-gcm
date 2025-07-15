"""
Constants for ICON radiation scheme.

This module contains physical and numerical constants used throughout
the ICON radiation implementation.
"""

# Enhanced spectral resolution for improved accuracy
N_SW_BANDS = 6  # Shortwave bands (UV, visible, near-IR)
N_LW_BANDS = 8  # Longwave bands (far-IR, window, near-IR)
N_BANDS_TOTAL = N_SW_BANDS + N_LW_BANDS  # Total bands

# Band definitions for enhanced resolution
# Shortwave bands (wavelength in μm)
SW_BAND_LIMITS = (
    (0.20, 0.29),  # UV-C/B
    (0.29, 0.32),  # UV-A
    (0.32, 0.44),  # Blue
    (0.44, 0.69),  # Green-Red
    (0.69, 1.19),  # Near-IR 1
    (1.19, 4.00),  # Near-IR 2
)

# Longwave bands (wavenumber in cm⁻¹)
LW_BAND_LIMITS = (
    (10, 200),     # Far-IR window
    (200, 280),    # H2O rotation band
    (280, 400),    # CO2 bending + H2O
    (400, 540),    # CO2 v2 + H2O
    (540, 800),    # H2O continuum
    (800, 1000),   # H2O + O3
    (1000, 1200),  # O3 + H2O
    (1200, 2600),  # H2O bands
)