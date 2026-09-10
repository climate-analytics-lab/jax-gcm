"""Constants for ICON radiation scheme.

The band definitions here are the single source of truth used by both
``gas_optics``/``planck``/``cloud_optics`` (which need Python ints at
trace time for static shapes / loop unrolls) and by
``RadiationParameters.default()`` (which converts them to jnp arrays
for runtime use). Keep both in sync by editing only this file.
"""

# Shortwave bands (wavenumber in cm⁻¹). The two labels were transposed until
# #678: 4000-14500 cm-1 is 0.69-2.50 um (near-IR) and 14500-50000 cm-1 is
# 0.20-0.69 um (UV + visible). Only the comments were wrong -- the ORDER of
# the tuples is baked into downstream band indexing and must not move. The
# labels matter because visible and near-IR surface albedo differ by a factor
# of several over snow and vegetation, so anyone picking a band index from
# the label put the albedo in the wrong band.
SW_BAND_LIMITS = (
    (4000, 14500),   # Near-IR (0.69-2.50 um)
    (14500, 50000),  # UV + visible (0.20-0.69 um)
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
