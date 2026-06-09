"""Online aerosol optics for the JAM harness (#495).

A self-contained Mie pathway: a NumPy Bohren-Huffman kernel builds a lookup
table once at construction, and the per-step :class:`JamOpticsTerm` does a
differentiable interpolation to produce per-band SW+LW aerosol optical depth /
single-scattering albedo / asymmetry from the modal population.
"""

from jcm.physics.aerosol.jam.optics.mie import mie_efficiencies
from jcm.physics.aerosol.jam.optics.mie_lut import (
    MieLUT,
    build_mie_lut,
    interp_mie,
)
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.optics.refractive_index import refractive_index_at

__all__ = [
    "JamOpticsTerm",
    "mie_efficiencies",
    "build_mie_lut",
    "interp_mie",
    "MieLUT",
    "refractive_index_at",
]
