"""Guards for the ``jcm.constants`` override contract (#772).

The contract (``jcm/constants.py``, CLAUDE.md) is that consumers read
constants by *attribute access* on the module — ``import jcm.constants as c;
... c.grav`` — so ``set_constants`` reaches every consumer. Two ways to break
it, both silent:

* ``from jcm.constants import grav`` binds the float at import time;
* ``from jcm.constants import physical_constants`` binds the singleton
  *object*, and ``set_constants`` rebinds the module global rather than
  mutating it (``PhysicalConstants`` is a ``NamedTuple``), so the captured
  reference goes equally stale;
* and — the form that is easiest to miss, because the module *does* use the
  approved alias — evaluating ``c.<name>`` at import time anyway: a derived
  module constant (``_MW_AIR = c.m_air * 1000.0``), a default argument
  (``def f(..., gravity=c.grav)``, evaluated once when the ``def`` executes)
  or a class-body attribute. The alias is necessary but not sufficient; what
  matters is *when* the attribute is read.

Either leaves a process computing with a mix of overridden and Earth values
and no error anywhere. The structural test below makes the contract
mechanical; the behavioural tests prove it end-to-end for the modules that
were converted, so a future regression fails on the physics, not just on a
lint-style rule.

Scope note, stated because the guard is structural and therefore has a
boundary. It covers jcm's *own* bindings, read directly. Two things it
cannot see:

* a module-level *call* that reads constants inside itself
  (``_TABLE = _build_table()``). The one instance in the package is
  ``dycore.PHYSICS_SPECS``, which is deliberate and safe: it is built from
  ``PhysicalConstants.default()`` rather than the live singleton, and is
  referenced only by tests — production code builds its specs at
  construction from ``self.constants``.
* constants internal to ``mam4_jax``. Threading ``set_constants`` into that
  package is a separate concern and is not attempted here.
"""

import ast
import pathlib
import unittest

import jax
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

    A function *body* is skipped — code there runs at call time and therefore
    reads the live singleton, which is a legitimate pattern the codebase uses
    (``jcm/utils.py``, ``state_bridge.py``). Its decorators and default
    arguments are **not** skipped: those expressions are evaluated once, when
    the ``def`` statement executes at import, which is precisely how
    ``gravity: float = c.grav`` froze a constant while looking innocuous.
    Class bodies are not skipped either; they execute on import like any
    other statement.

    A ``lambda`` is treated exactly like a ``def`` and for the same reason:
    its *body* runs at call time, but ``lambda g=c.grav: g`` evaluates that
    default the moment the lambda is created. Skipping the whole node would
    leave the one expression that is not lazy unchecked.
    """
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            decorators = getattr(child, "decorator_list", [])
            for expr in (
                *decorators,
                *child.args.defaults,
                *(d for d in child.args.kw_defaults if d is not None),
            ):
                yield expr
                yield from _import_time_nodes(expr)
            continue
        yield child
        yield from _import_time_nodes(child)


def _constants_aliases(
    tree: ast.AST, module_name: str
) -> tuple[set[str], set[str]]:
    """Names referring to ``jcm.constants``, as ``(module, package)`` sets.

    ``module`` names hold the module itself (``import jcm.constants as c``,
    ``from jcm import constants``, and the package-relative ``from . import
    constants as c``), so ``<name>.grav`` is a constant read. ``package``
    names come from a bare ``import jcm.constants``, which binds ``jcm``;
    there the read is ``jcm.constants.grav`` and the intermediate
    ``jcm.constants`` is the module, not a constant.

    The ``from`` forms are resolved through :func:`_resolve_import` so a
    relative import counts exactly as the absolute one does — otherwise
    ``from . import constants as c`` inside the package would slip past.
    """
    module, package = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jcm.constants":
                    (module if alias.asname else package).add(alias.asname or "jcm")
        elif isinstance(node, ast.ImportFrom):
            if _resolve_import(node, module_name) == "jcm":
                for alias in node.names:
                    if alias.name == "constants":
                        module.add(alias.asname or "constants")
    return module, package


def _captured_attributes(tree: ast.AST, module_name: str = "jcm.fake"):
    """Yield ``(lineno, text)`` for each constant read at import time."""
    module_names, package_names = _constants_aliases(tree, module_name)
    if not module_names and not package_names:
        return
    for node in _import_time_nodes(tree):
        if not isinstance(node, ast.Attribute) or node.attr in _ALLOWED_NAMES:
            continue
        target = node.value
        # ``c.grav`` where ``c`` is the module itself.
        if isinstance(target, ast.Name) and target.id in module_names:
            yield node.lineno, f"{target.id}.{node.attr}"
        # ``jcm.constants.grav`` after a bare ``import jcm.constants``.
        elif (
            isinstance(target, ast.Attribute)
            and target.attr == "constants"
            and isinstance(target.value, ast.Name)
            and target.value.id in package_names
        ):
            yield node.lineno, f"jcm.constants.{node.attr}"


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
            for lineno, text in _captured_attributes(tree, module_name):
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{lineno}: {text} "
                    f"evaluated at import time"
                )
        self.assertEqual(
            offenders, [],
            "These read a constant at import time, so "
            "jcm.constants.set_constants() will not reach them (#772). Read "
            "`c.<name>` where the value is USED — inside the function, at "
            "construction, or at trace time — not at module level, not in a "
            "derived module constant, and not in a default argument (which is "
            "evaluated once when the def executes). Only PhysicalConstants, a "
            "type that binds no value, may be imported by name:\n"
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

    def test_guard_would_catch_an_import_time_attribute_read(self):
        # The alias is necessary but not sufficient: these all USE the
        # approved `import jcm.constants as c` and are still frozen at import.
        # Reads inside a function body are the legitimate case and must not
        # fire, which is the half that makes this test worth writing.
        source = (
            "import jcm.constants as c\n"
            "_MW_AIR = c.m_air * 1000.0\n"
            "class K:\n"
            "    g = c.grav\n"
            "def f(x, gravity=c.grav):\n"
            "    return x * c.cpd + c.rd\n"
        )
        flagged = sorted(text for _, text in _captured_attributes(ast.parse(source)))
        self.assertEqual(flagged, ["c.grav", "c.grav", "c.m_air"])

    def test_guard_catches_a_lambda_default_but_not_a_lambda_body(self):
        # A lambda's body is lazy; its defaults are not — they are evaluated
        # when the lambda object is created, which at module level is import.
        source = (
            "import jcm.constants as c\n"
            "frozen = lambda g=c.grav: g\n"
            "live = lambda x: x * c.cpd\n"
        )
        flagged = [text for _, text in _captured_attributes(ast.parse(source))]
        self.assertEqual(flagged, ["c.grav"])

    def test_guard_catches_a_relative_constants_alias(self):
        # ``from . import constants as c`` inside the package binds the same
        # module as the absolute form, so it must be resolved the same way —
        # the value-import path already did, the attribute path did not.
        source = (
            "from . import constants as c\n"
            "FROZEN = c.grav\n"
        )
        tree = ast.parse(source)
        self.assertEqual(
            [t for _, t in _captured_attributes(tree, "jcm.some_module")],
            ["c.grav"],
        )
        # ``from .. import constants`` from one level deeper resolves to the
        # same module; a same-named module in a *sub*package does not.
        deep = ast.parse("from .. import constants\nFROZEN = constants.grav\n")
        self.assertEqual(
            [t for _, t in _captured_attributes(deep, "jcm.physics.thing")],
            ["constants.grav"],
        )
        unrelated = ast.parse(
            "from . import constants as c\nFROZEN = c.grav\n"
        )
        self.assertEqual(
            [t for _, t in _captured_attributes(unrelated, "jcm.physics.thing")],
            [],
        )

    def test_guard_accepts_the_bare_module_import_form(self):
        source = (
            "import jcm.constants\n"
            "_G = jcm.constants.grav\n"
        )
        flagged = [text for _, text in _captured_attributes(ast.parse(source))]
        self.assertEqual(flagged, ["jcm.constants.grav"])


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


class HetMxphaseFreezingHonoursOverrideTest(_OverrideCase):
    """``lohmann_2m/deposition_freezing.py::het_mxphase_freezing`` follows grav.

    It reads ``params.grav``, which ``CloudParams2M.default()`` takes from the
    live constants, and ``c.cpd``.

    Its immersion rate carries ECHAM's cooling ``fact_tke*sqrt(TKE)*g/cpd``
    (mo_cloud_micro_2m.f90:2800-2802), linear in g while the frozen fraction is
    small (dust immersion only, no contact nuclei, a warm supercooled cell).
    """

    def test_immersion_freezing_is_proportional_to_gravity(self):
        from jcm.physics.clouds.lohmann_2m.deposition_freezing import (
            het_mxphase_freezing,
        )
        from jcm.physics.clouds.lohmann_2m_params import CloudParams2M

        one = jnp.ones((1,))
        zero = jnp.zeros((1,))

        def frozen():
            p = CloudParams2M.default()
            out = het_mxphase_freezing(
                one > 0, 6.0e4 * one, 0.1 * one, zero, 0.5 * one, zero, zero,
                0.3 * one, zero, zero, 0.8 * one, 1.25 * one, zero, zero, zero,
                252.0 * one, 4e7 * one, 1e3 * one, 8e7 * one, zero, zero,
                2e-4 * one, 720.0, p.cqtmin, p)
            return out[2][0]

        baseline, doubled_g = self.under_grav(2.0, frozen)
        # 1 - exp(-x) at x ~ 3e-3 and float32: 2.0 to within 0.1 %.
        self.assertAlmostEqual(doubled_g / baseline, 2.0, delta=5e-3)


class TurbulenceDefaultArgHonoursOverrideTest(_OverrideCase):
    """The turbulence coefficients read ``grav`` when traced (#772).

    A default argument would have been evaluated once when the ``def`` executed
    at import and frozen the pre-override value.
    """

    def test_richardson_number_moves_with_gravity(self):
        from jcm.physics.vertical_diffusion.tte_tke.turbulence_coefficients import (
            compute_richardson_number,
        )
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VDiffState,
        )

        nlev = 3
        # Weak shear, so ECHAM's shear floor is the denominator and, with the
        # geopotential given, the buoyancy (which carries g²) is what moves.
        # (Where the shear exceeds the floor, g cancels: Ri = Δθ·Δφ/(θ·Δu²).)
        one = lambda *v: jnp.asarray([list(v)])  # noqa: E731
        zero = jnp.zeros((1, nlev))
        pressure_half = one(80000.0, 85000.0, 92000.0, 100000.0)
        state = VDiffState(
            u=one(1.0, 1.5, 2.0), v=zero, temperature=one(290.0, 284.0, 276.0),
            qv=zero + 2e-3, qc=zero, qi=zero, cloud_fraction=zero,
            pressure_full=0.5 * (pressure_half[:, :-1] + pressure_half[:, 1:]),
            pressure_half=pressure_half,
            geopotential=one(20000.0, 12000.0, 5000.0),
            air_mass=zero, surface_temperature=zero, surface_fraction=zero,
            roughness_length=zero, roughness_heat=zero, surface_wetness=zero,
            height_full=zero, height_half=jnp.zeros((1, nlev + 1)),
            tke=zero, thv_variance=zero, ocean_u=jnp.zeros(1), ocean_v=jnp.zeros(1),
        )

        def ri():
            # The function is ``@jax.jit``-ed, so its trace — constants and
            # all — is cached. Clearing first is what makes this the real
            # contract: a model *built after* set_constants sees the new
            # value. An override after tracing does not propagate until
            # recompilation, which is why the docs say to override first.
            jax.clear_caches()
            return compute_richardson_number(state)[0, 0]

        baseline, doubled_g = self.under_grav(2.0, ri)
        self.assertNotAlmostEqual(baseline, doubled_g, places=6)


class AqueousChemistryHamLiteralsIgnoreOverrideTest(unittest.TestCase):
    """``chemistry/aqueous.py``'s HAM literals do NOT track ``set_constants``.

    jax-gcm#1031 moved ``_aqueous_so4``'s Avogadro's number, its xtoc/ctox
    factor, the gas constant and SO2's molar mass from values derived from
    jcm's own (live, overridable) constants/species tables to r7492's own
    hardcoded literals (``_AVO``/``_AVO_XTOC``/``_ZRGAS``/``_MW_SO2``): the
    fidelity target is the exact Fortran value HAM hardcodes, which a
    derived-and-so-recomputable quantity would not reproduce bit for bit
    even when numerically close (the same policy ``activation/
    ham_activation.py`` uses for its own CAM/HAM literals). ``_mw_air`` is
    the one exception: it backs the *simple* aqueous scheme, jcm's own
    reduced alternative rather than a literal HAM transcription, so it
    legitimately keeps tracking ``c.m_air``.
    """

    def test_literals_unaffected_by_a_constants_override(self):
        from jcm.physics.aerosol.jam.chemistry import aqueous

        original = c.physical_constants
        try:
            baseline_avo_xtoc = aqueous._AVO_XTOC
            baseline_avo = aqueous._AVO
            baseline_zrgas = aqueous._ZRGAS
            baseline_mw_so2 = aqueous._MW_SO2
            baseline_mw_air = aqueous._mw_air()
            # A large override that WOULD move every one of these if any
            # were still derived from the live singleton (m_air = R*/Rd,
            # so doubling R* doubles it with Rd held fixed).
            c.set_constants(ak=original.ak * 0.5,
                             r_universal=original.r_universal * 2.0)
            self.assertEqual(aqueous._AVO_XTOC, baseline_avo_xtoc)
            self.assertEqual(aqueous._AVO, baseline_avo)
            self.assertEqual(aqueous._ZRGAS, baseline_zrgas)
            self.assertEqual(aqueous._MW_SO2, baseline_mw_so2)
            # _mw_air is the one legitimate exception (see the class docstring).
            self.assertAlmostEqual(aqueous._mw_air() / baseline_mw_air, 2.0, places=6)
        finally:
            c.set_constants(original)
        self.assertEqual(c.physical_constants.ak, original.ak)


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
