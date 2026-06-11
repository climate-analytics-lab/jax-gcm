"""Heterogeneous ice nucleation (dust/BC) for the JAM harness (#494).

Immersion + deposition freezing on the prognostic dust and black-carbon
populations, via a switchable parameterization (``"niemand"`` singular
active-site, or ``"lohmann_diehl"`` ECHAM-HAM number-based). The
:class:`IceNucleation` term writes an ``ice_nuclei`` diagnostic [m⁻³] that the
2-moment cloud scheme reads to set the heterogeneous ice-crystal number.
"""
