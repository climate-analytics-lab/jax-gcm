"""A non-lat/lon dycore used by the protocol-validation test.

The :class:`FakeCubedSphereDycore` lives here so it never leaks into the
shipped public API — only :mod:`jcm.dycore.protocol_test` imports it. Its
purpose is to prove that the :class:`DynamicalCore` protocol can host a
backend whose horizontal layout is *not* a regular ``(nlon, nlat)`` grid:
arrays are shaped ``(nlev, nelem, gll, gll)``, latitudes/longitudes are 2-D,
and there is no spherical-harmonic basis. If the rest of jax-gcm (Model,
physics_interface, the physics packages we care about for the design spike)
silently assumed a lat/lon shape, this fake would break.

When the real :class:`PyscesDycore` lands in Phase 2, this module stays as
the protocol-correctness guardrail: a non-trivial layout that exercises every
method on :class:`DynamicalCore` end-to-end.
"""

from jcm.dycore._fake_cubed_sphere.dycore import FakeCubedSphereDycore

__all__ = ["FakeCubedSphereDycore"]
