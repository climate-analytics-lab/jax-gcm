"""Guards for the ``jcm.constants`` override contract (#772).

The contract (``jcm/constants.py``, CLAUDE.md) is that consumers read
constants by *attribute access* on the module — ``import jcm.constants as c;
... c.grav`` — so ``set_constants`` reaches every consumer. Two ways to break
it, both silent:

* ``from jcm.constants import grav`` binds the float at import time;
* ``from jcm.constants import physical_constants`` binds the singleton
  *object*, and ``set_constants`` rebinds the module global rather than
  mutating it (``PhysicalConstants`` is a ``NamedTuple``), so the captured
  reference goes equally stale.

Either leaves a process computing with a mix of overridden and Earth values
and no error anywhere. The structural test below makes the contract
mechanical; the behavioural tests prove it end-to-end for the modules that
were converted, so a future regression fails on the physics, not just on a
lint-style rule.

Scope note: this covers jcm's *own* bindings. A JAM scheme that takes a value
from ``mam4_jax``'s internal constants is out of scope here — threading
``set_constants`` into that package is a separate concern.
"""

import ast
import pathlib
import unittest

import jax.numpy as jnp

import jcm.constants as c
from jcm.constants import PhysicalConstants


_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent

# The class is a type, not a value: importing it binds no constant, and the
# dycore uses it as an annotation while reading the live singleton separately.
_ALLOWED_NAMES = frozenset({"PhysicalConstants"})


def _module_name(path: pathlib.Path) -> str:
    """Dotted module name for a file inside the package (``jcm.physics.x``)."""
    return ".".join(path.relative_to(_REPO_ROOT).with_suffix("").parts)


def _resolve_import(node: ast.ImportFrom, module_name: str) -> str | None:
    """Absolute module a ``from ... import`` names, relative imports included."""
    if not node.level:
        return node.module
    parts = module_name.split(".")[: -node.level]
    return ".".join(parts + ([node.module] if node.module else []))


def _import_time_nodes(tree: ast.AST):
    """Yield every node evaluated when the module is imported.

    Function bodies are skipped — an import inside a function runs at *call*
    time and therefore reads the live singleton, which is a legitimate
    pattern the codebase uses (``jcm/utils.py``, ``state_bridge.py``). Class
    bodies are not skipped: they execute on import like any other statement.
    """
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield child
        yield from _import_time_nodes(child)


class ConstantsImportContractTest(unittest.TestCase):
    """No module may capture a constant's value at import time."""

    def test_no_import_time_constant_bindings(self):
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            if path.name.endswith("_test.py"):
                continue  # tests may pin a value on purpose
            module_name = _module_name(path)
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in _import_time_nodes(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if _resolve_import(node, module_name) != "jcm.constants":
                    continue
                for alias in node.names:
                    if alias.name not in _ALLOWED_NAMES:
                        offenders.append(
                            f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: "
                            f"from jcm.constants import {alias.name}"
                        )
        self.assertEqual(
            offenders, [],
            "These module-level imports bind a constant at import time, so "
            "jcm.constants.set_constants() will not reach them (#772). Use "
            "`import jcm.constants as c` and read `c.<name>` where it is "
            "used. Only PhysicalConstants (a type) may be imported by name:\n"
            + "\n".join(offenders),
        )

    def test_guard_would_catch_a_value_binding(self):
        # The guard is only worth having if it actually fires, and only for
        # import-time bindings: a function-local import is evaluated at call
        # time and must stay allowed.
        source = (
            "from jcm.constants import PhysicalConstants\n"
            "from jcm.constants import grav as _G\n"
            "from jcm.constants import physical_constants\n"
            "def f():\n"
            "    from jcm.constants import grav\n"
        )
        tree = ast.parse(source)
        flagged = [
            alias.name
            for node in _import_time_nodes(tree)
            if isinstance(node, ast.ImportFrom)
            and _resolve_import(node, "jcm.fake") == "jcm.constants"
            for alias in node.names
            if alias.name not in _ALLOWED_NAMES
        ]
        self.assertEqual(flagged, ["grav", "physical_constants"])


class _OverrideCase(unittest.TestCase):
    """Base class: run a callable under an overridden gravity, then restore."""

    def under_grav(self, factor, fn):
        """Return ``(baseline, overridden)`` results of ``fn()``.

        ``set_constants`` is process-global, so the original set is restored
        in a ``finally`` — an escaped override would silently corrupt every
        later test in this xdist worker.
        """
        original = c.physical_constants
        try:
            baseline = float(fn())
            c.set_constants(grav=original.grav * factor)
            overridden = float(fn())
        finally:
            c.set_constants(original)
        self.assertEqual(c.physical_constants.grav, original.grav)
        return baseline, overridden


class TropopauseHonoursOverrideTest(_OverrideCase):
    """``wmo_tropopause`` held a stale reference to the singleton object."""

    def test_geopotential_height_scales_with_gravity(self):
        from jcm.physics.diagnostics.wmo_tropopause import (
            compute_geopotential_height,
        )

        pressure = jnp.asarray([9.0e4, 7.0e4, 5.0e4])
        temperature = jnp.asarray([280.0, 270.0, 250.0])
        surface_pressure = jnp.asarray(1.0e5)

        def top_height():
            return compute_geopotential_height(
                pressure, temperature, surface_pressure
            )[-1]

        baseline, halved_g = self.under_grav(0.5, top_height)
        # z = (R T / g) ln(p_lo/p_hi): halving g doubles every thickness.
        self.assertAlmostEqual(halved_g / baseline, 2.0, places=5)


class ArgActivationHonoursOverrideTest(_OverrideCase):
    """``activation/arg.py`` bound eleven constants at import time."""

    def test_max_supersaturation_moves_with_gravity(self):
        from jcm.physics.aerosol.jam.activation.arg import arg_activation

        one = lambda v: jnp.asarray([v]).reshape(1, 1, 1)

        def s_max():
            _, _, smax, _, _ = arg_activation(
                r_dry=one(0.05e-6), kappa=one(0.6), number_vol=one(1.0e8),
                sigma_g=one(1.8), can_activate=one(1.0),
                updraft=jnp.full((1, 1), 0.5),
                temperature=jnp.full((1, 1), 283.0),
                pressure=jnp.full((1, 1), 9.0e4),
                sigma_acc=1.8, variant="arg2000",
            )
            return smax[0, 0]

        baseline, doubled_g = self.under_grav(2.0, s_max)
        # α carries g linearly, and s_max grows with α, so this must move.
        self.assertGreater(doubled_g, baseline * 1.05)


class SedimentationHonoursOverrideTest(_OverrideCase):
    """``sedimentation/sedi_term.py`` bound grav, m_air and R*."""

    def test_settling_velocity_is_proportional_to_gravity(self):
        from jcm.physics.aerosol.jam.sedimentation.sedi_term import (
            stokes_velocity,
        )

        def v():
            return stokes_velocity(
                r_wet=jnp.asarray(0.5e-6), rho_p=jnp.asarray(1800.0),
                temperature=jnp.asarray(285.0), pressure=jnp.asarray(9.0e4),
                geom_std_dev=1.6, moment=3,
            )

        baseline, doubled_g = self.under_grav(2.0, v)
        # Stokes settling is linear in g (the slip correction is not, but it
        # depends on the mean free path, which carries no g at all).
        self.assertAlmostEqual(doubled_g / baseline, 2.0, places=5)


class DryDepositionHonoursOverrideTest(_OverrideCase):
    """``drydep/resistances.py`` bound the Boltzmann constant, grav, M_a, R*."""

    def test_quasi_laminar_resistance_moves_with_gravity(self):
        from jcm.physics.aerosol.jam.drydep.resistances import (
            quasi_laminar_resistance,
        )

        def r_b():
            return quasi_laminar_resistance(
                r_wet=jnp.asarray(1.0e-6), v_grav=jnp.asarray(1.0e-4),
                u_star=jnp.asarray(0.4), temperature=jnp.asarray(285.0),
                pressure=jnp.asarray(9.0e4), air_density=jnp.asarray(1.2),
            )

        baseline, tenth_g = self.under_grav(0.1, r_b)
        # St = v_grav u*²/(g ν): a tenth of g is a ten-fold Stokes number, so
        # the inertial term rises and the resistance falls.
        self.assertLess(tenth_g, baseline)


class IceNucleationHonoursOverrideTest(_OverrideCase):
    """``ice_nucleation/ice_term.py`` bound grav and cpd."""

    def test_cooling_rate_is_proportional_to_gravity(self):
        from jcm.physics.aerosol.jam.ice_nucleation.ice_term import (
            IceNucleation,
        )

        term = IceNucleation()
        temperature = jnp.full((2,), 250.0)

        def cooling_rate():
            # No vertical-diffusion diagnostic, so the fallback updraft is
            # used and the rate is exactly w·g/cp.
            return term._cooling_rate({}, temperature)[0]

        baseline, doubled_g = self.under_grav(2.0, cooling_rate)
        self.assertAlmostEqual(doubled_g / baseline, 2.0, places=5)


class SetConstantsRestorationTest(unittest.TestCase):
    """The override helper itself must leave nothing behind."""

    def test_derived_quantities_track_a_base_override(self):
        original = c.physical_constants
        try:
            c.set_constants(cpd=2000.0)
            self.assertEqual(c.rd, PhysicalConstants().akap * 2000.0)
            self.assertEqual(c.cvd, c.cpd - c.rd)
        finally:
            c.set_constants(original)
        self.assertEqual(c.cpd, original.cpd)


if __name__ == "__main__":
    unittest.main()
