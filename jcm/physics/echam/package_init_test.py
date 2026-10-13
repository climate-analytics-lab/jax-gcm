"""Unit tests for the ``jcm.physics.echam`` package ``__init__``.

The package exposes ``echam_physics`` and ``wmo_tropopause`` through a
lazy ``__getattr__`` (to avoid circular imports after the physics
reorganisation); these tests pin that indirection to the real objects.
"""

import unittest


class TestEchamPackageLazyAttrs(unittest.TestCase):
    def test_echam_physics_lazy_import_is_factory(self):
        import jcm.physics.echam as echam_pkg
        from jcm.physics.echam.echam_terms import echam_physics

        # The lazily-resolved attribute must be the actual factory from
        # echam_terms, not a copy or a stale reference.
        self.assertIs(echam_pkg.echam_physics, echam_physics)

    def test_wmo_tropopause_lazy_import(self):
        import jcm.physics.echam as echam_pkg
        from jcm.physics.diagnostics.wmo_tropopause import wmo_tropopause

        self.assertIs(echam_pkg.wmo_tropopause, wmo_tropopause)

    def test_unknown_attribute_raises_attribute_error(self):
        import jcm.physics.echam as echam_pkg

        with self.assertRaises(AttributeError):
            echam_pkg.no_such_symbol  # noqa: B018

    def test_physical_constants_reexport(self):
        # ``physical_constants`` re-exports the live jcm.constants module.
        import jcm.constants as c
        import jcm.physics.echam as echam_pkg

        self.assertIs(echam_pkg.physical_constants, c)
        # Sanity: the constants module resolves attribute access (v2 API).
        self.assertGreater(float(echam_pkg.physical_constants.grav), 9.0)
