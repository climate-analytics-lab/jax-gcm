Welcome to JAX-GCM's documentation!
====================================

JAX-GCM is a differentiable atmospheric general circulation model written
in JAX. Its pluggable dynamical-core interface currently ships with the
`Dinosaur <https://github.com/neuralgcm/dinosaur>`_ spectral backend and
couples it to modular SPEEDY, Held-Suarez, and ECHAM-style physics packages.

For the v2.0 release line, the main target configuration is ECHAM
physics on the T63L47 hybrid grid with RRTMGP radiation
(``physics=echam-rrtmgp grid=echam_t63_l47_hybrid``). The SPEEDY package
remains the lightweight default for quick tests, tutorials, and
optimization examples.

.. note::

   New development targets the ``dev`` branch. Tagged release candidates
   and clean releases are promoted through ``main``.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   speedy_physics
   echam_physics
   speedy_translation
   api
   design
   developer
