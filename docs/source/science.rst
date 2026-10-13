Model description
=================

This is the living model-description document: a positive, by-process statement
of what JAX-GCM's physics and dynamics actually *do*, why each consequential
choice was made, and where it departs from the reference models it was ported
from. It is organised by physical process — not by configuration — because the
supported configurations share most of their description; each configuration is
instead a short :doc:`selection page <science/configurations>` naming which option
fills each slot. The shape follows Zhang et al. (2012, ECHAM-HAM2 §2) and
Danabasoglu et al. (2020, CESM2): the section list is intended to double as the
skeleton of a future model-description paper.

How to read a section
----------------------

Every process section — and every individual decision entry — follows the same
fixed template:

- **What we do** — a positive statement of what the implementation *is*.
- **What ECHAM / CAM does** — the reference formulation, with literature
  references (paper, year, and the Fortran ``file::routine`` actually followed).
- **Why we differ** — tagged ``science``, ``compute``, or ``differentiability``.
  The ``differentiability`` tag is first-class: several choices exist in no
  reference model because gradient flow forced them.
- **Status & known limitations** — biases and unported pieces stated openly, the
  way the HAM2 paper states its own biases.
- **Code pointers** — the modules and symbols that implement the section, cited by
  ``file::symbol`` so they survive line-number churn.
- **Validation evidence** — the tests, A/Bs and validation runs that back it up.

The reference trees this document is checked against are ECHAM 6.3-HAM2.3
(release r7492) and ESCOMP/CAM at tag ``cam6_4_196`` — with ``zm_conv``,
``wv_sat_methods`` and ``cloud_fraction`` read from the ``atmos_phys`` external
at its pinned tag ``atmos_phys0_27_000``, and the MAM4 lineage tracing to E3SM,
not CAM. Where a component
was found to carry an inherited reference label that its code does not follow, the
*true* reference and the deviation are both recorded here — that reconciliation is
the reason this document exists. The engineering "how" lives in the
:doc:`design references <design>`, which the sections below link down into.

.. toctree::
   :maxdepth: 1
   :caption: By process

   science/dynamical_core
   science/operator_splitting
   science/radiation
   science/convection
   science/clouds_microphysics
   science/vertical_diffusion
   science/gravity_waves
   science/surface
   science/aerosol
   science/chemistry
   science/boundary_conditions
   science/constants

.. toctree::
   :maxdepth: 1
   :caption: By configuration

   science/configurations
