"""
Constants for ICON radiation scheme.

This module contains physical and numerical constants used throughout
the ICON radiation implementation.
"""

# Fixed number of spectral bands (simplified for JAX compatibility)
N_SW_BANDS = 2  # Shortwave bands (visible, near-IR)
N_LW_BANDS = 3  # Longwave bands (far-IR, window, near-IR)
N_BANDS_TOTAL = N_SW_BANDS + N_LW_BANDS  # Total bands