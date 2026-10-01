"""Aerosol inputs to heterogeneous ice formation for the JAM harness (#953).

:class:`~jcm.physics.aerosol.jam.ice_nucleation.ice_term.IceNucleation`
computes ECHAM-HAM's mixed-phase freezing inputs (``mo_ham_freezing.f90``:
the dust and black-carbon fractions of the activated droplets and of the
insoluble aerosol, and the insoluble-mode wet radii) from the prognostic
population and publishes them as ``freezing_aerosol``, which the 2-moment
cloud scheme's ``het_mxphase_freezing`` turns into contact and immersion
freezing rates.
"""
