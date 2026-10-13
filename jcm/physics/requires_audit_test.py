"""Static audit: every diagnostics read is declared.

Walks each ``PhysicsTerm`` subclass's ``__call__`` and asserts that:

- every bare ``diagnostics["key"]`` subscript (a HARD, unconditional read)
  is covered by ``requires`` | ``provides`` | ``carry_slots`` |
  ``requires_dycore_fields`` | the framework plumbing keys;
- every ``diagnostics.get("key")`` call or ``if "key" in diagnostics"``
  guard (a SOFT, conditional read) is covered by the same set, UNION
  ``optional_requires``.

A soft read used to be exempt outright (see the ``requires`` comment in
``physics_term.py`` before jax-gcm#673): a term could read
``diagnostics.get("key")`` for an accidentally-stale previous-step value and
nothing would catch it, because the one documented escape hatch for a
*legitimate* cross-step/optional read (``.get``/``in``) was indistinguishable
from an undeclared one. ``optional_requires`` is the declaration that closes
that hole: it is the same bar ``requires`` sets for hard reads, applied to
soft ones, so an undeclared soft read is now exactly as loud as an undeclared
hard one — not quietly correct by construction, as a runtime-only check would
be for everything except the first step.

**Resolving non-literal keys.** A key is not always a bare string literal: a
handful of terms read/declare a diagnostics key through a shared module-level
constant (``diagnostics.get(SURFACE_OPTICS_KEY)``) or build a ``provides``
tuple with a helper call (``emission_flux_keys() + (MODEL_LEVEL_WIND_KEY,)``).
``ast.literal_eval`` cannot evaluate either form, and a Codex review of the
first version of this audit (PR #1094) caught exactly this: the key was
silently dropped from both ``hard``/``soft`` and from ``requires``/
``provides``, so the audit said nothing about those reads at all -- the same
"undeclared read funnelled into the one form nothing checks" failure mode
#673 itself is about, recurring at the next form the naive AST reader can't
parse. Rather than hand-maintain a second exemption list for this (which
would itself bypass the point of an *audit*, and would silently stop working
the next time a term introduces a new constant), this module actually
IMPORTS each audited file as a Python module and evaluates an unresolved key
expression against that module's own namespace: the exact same resolution
Python performed when the class was defined, so it is correct for any
constant, any import alias, and any pure helper-function call a ``provides``/
``optional_requires`` declaration or a read site might use -- not just the
two forms found so far.
"""

import ast
import importlib
import pathlib
import unittest

_PLUMBING = frozenset({
    "_dt_seconds", "_band_config", "_dycore_fields", "_forcing_2d",
    "_echam_params", "_echam_coords", "_speedy_coords",
    # physics_interface.POST_PHYSICS_STATE_KEY: injected by ComposablePhysics
    # itself (``_ensure_post_physics_slot``/``record_post_physics_state``),
    # never a term's own ``provides`` — the framework-plumbing analogue of
    # ``_dt_seconds`` for the carried previous-step state, gated on
    # ``requires_post_physics_fields`` rather than a producer term.
    "_post_physics_state",
})

# jax-gcm repo root: .../jax-gcm/jcm/physics/requires_audit_test.py -> 3 parents up.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

_MODULE_CACHE: dict[str, object] = {}


def _dotted_module_name(py_path: pathlib.Path) -> str | None:
    """``jcm.physics.aerosol.jam.cloud_borne_store`` for that file, or ``None``."""
    try:
        rel = py_path.resolve().relative_to(_REPO_ROOT).with_suffix("")
    except ValueError:
        return None
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def import_module_for(py_path: pathlib.Path):
    """Import and return the live module for ``py_path``, caching the result.

    Returns ``None`` if the path isn't importable as a ``jcm`` module or the
    import raises -- callers treat that exactly like "can't resolve this
    expression" (fall back to flagging nothing extra, never to a false
    positive).
    """
    dotted = _dotted_module_name(py_path)
    if dotted is None:
        return None
    if dotted not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[dotted] = importlib.import_module(dotted)
        except Exception:
            _MODULE_CACHE[dotted] = None
    return _MODULE_CACHE[dotted]


def _eval_against_module(node: ast.AST, module):
    """Evaluate ``node`` against ``module``'s own namespace, or ``None``.

    Only reached once ``ast.literal_eval`` has already failed, i.e. the
    expression references a name (a constant, an import, a helper function)
    rather than being a plain literal. Evaluating it against the live,
    already-imported module's ``vars()`` is exactly the resolution Python
    itself performed when the class body executed, so it is correct for any
    such expression -- not a re-implementation of import/alias resolution
    that could drift from Python's own rules. ``__builtins__`` is blanked:
    every one of these expressions is tuple concatenation or a call to a
    module-level helper that only touches module globals, so nothing here
    legitimately needs a builtin, and blanking it is cheap insurance since
    the expression comes from source under audit, not user input.
    """
    if module is None:
        return None
    try:
        expr = ast.Expression(body=node)
        ast.fix_missing_locations(expr)
        code = compile(expr, "<requires_audit_test>", "eval")
        return eval(code, {"__builtins__": {}, **vars(module)})  # noqa: S307
    except Exception:
        return None


def _resolve_tuple(node, module) -> tuple[str, ...] | None:
    """``node`` evaluated as a tuple/list of strings, literally or via ``module``."""
    try:
        v = ast.literal_eval(node)
    except (ValueError, TypeError):
        v = _eval_against_module(node, module)
    if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
        return tuple(v)
    return None


def _resolve_str(node, module) -> str | None:
    """``node`` evaluated as a single string, literally or via ``module``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    v = _eval_against_module(node, module)
    return v if isinstance(v, str) else None


def audit_class(cls: ast.ClassDef, module=None):
    """Return ``{"hard": [...], "soft": [...]}`` undeclared-read lists, or ``None``.

    ``None`` means the class has no ``requires`` declaration AND no
    ``__call__`` — not a ``PhysicsTerm`` this audit has anything to say
    about (a helper class, a dataclass, ...). Exposed as a module-level
    function (rather than nested in the test) so the self-tests below can
    exercise it directly against synthetic source, independent of the
    real tree. ``module`` is the already-imported module the class lives
    in, used only to resolve a key expression ``ast.literal_eval`` can't
    (see the module docstring); ``None`` (the self-tests' case) just means
    every such expression is left unresolved, same as a key the resolver
    genuinely can't place (a loop variable, a function parameter) --
    never a false positive.
    """
    requires = provides = dycore = None
    carry = set()
    optional = set()
    call_fn = None
    for item in cls.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            tgt = item.targets[0] if isinstance(item, ast.Assign) else item.target
            name = getattr(tgt, "id", None)
            val = item.value
            if name == "requires" and val is not None:
                requires = _resolve_tuple(val, module)
            elif name == "provides" and val is not None:
                provides = _resolve_tuple(val, module)
            elif name == "requires_dycore_fields" and val is not None:
                dycore = _resolve_tuple(val, module)
            elif name == "carry_slots" and isinstance(val, ast.Dict):
                for k in val.keys:
                    resolved = _resolve_str(k, module)
                    if resolved is not None:
                        carry.add(resolved)
            elif name == "optional_requires" and val is not None:
                lit = _resolve_tuple(val, module)
                if lit is not None:
                    optional |= set(lit)
        elif isinstance(item, ast.FunctionDef) and item.name == "__call__":
            call_fn = item
    if requires is None or call_fn is None:
        return None

    hard, soft = set(), set()
    for n in ast.walk(call_fn):
        # bare diagnostics["key"] loads
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "diagnostics"
                and isinstance(n.ctx, ast.Load)):
            key = _resolve_str(n.slice, module)
            if key is not None:
                hard.add(key)
        # diagnostics.get("key", ...)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "diagnostics"
                and n.args):
            key = _resolve_str(n.args[0], module)
            if key is not None:
                soft.add(key)
        # "key" in diagnostics guards
        if (isinstance(n, ast.Compare) and len(n.ops) == 1
                and isinstance(n.ops[0], (ast.In, ast.NotIn))
                and isinstance(n.comparators[0], ast.Name)
                and n.comparators[0].id == "diagnostics"):
            key = _resolve_str(n.left, module)
            if key is not None:
                soft.add(key)

    allowed = (set(requires) | set(provides or ()) | set(dycore or ())
               | carry | _PLUMBING)
    return {
        "hard": sorted((hard - soft) - allowed),
        # A soft read is declared either via optional_requires, or via the
        # same allowed set as a hard read (a term that both `provides` a
        # key and reads it back with `.get()` as its own carry seed —
        # e.g. ``SundqvistCloudFraction``'s ``"clouds"`` — needs no extra
        # declaration; it already owns the key).
        "soft": sorted(soft - allowed - optional),
    }


class RequiresAuditTest(unittest.TestCase):
    def test_every_diagnostics_read_is_declared(self):
        root = pathlib.Path(__file__).parent
        violations = {}
        for py in sorted(root.rglob("*.py")):
            if py.name.endswith("_test.py"):
                continue
            tree = ast.parse(py.read_text())
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            if not classes:
                continue
            # Only pay the import cost for files that declare a class at
            # all; `import_module_for` is cached per dotted name regardless.
            module = import_module_for(py)
            for cls in classes:
                missing = audit_class(cls, module)
                if missing and (missing["hard"] or missing["soft"]):
                    violations[f"{py.relative_to(root)}:{cls.name}"] = missing
        self.assertFalse(
            violations,
            "PhysicsTerm(s) read diagnostics keys not declared: a 'hard' "
            "entry needs `requires` (or `provides`/`carry_slots`/"
            "`requires_dycore_fields`); a 'soft' entry (`.get()`/`in`) "
            f"needs `optional_requires`: {violations}",
        )

    # ------------------------------------------------------------------
    # Self-tests of the audit mechanism itself (jax-gcm#673): prove that
    # the class of bug the issue describes — an undeclared
    # ``diagnostics.get("key")`` silently returning a previous-step (or
    # simply unintended) value — is now REJECTED by this audit, and that
    # the two legitimate declaration forms still pass. These parse
    # synthetic source directly (not the real tree), so they pin the
    # audit's behaviour independent of whatever the repo currently
    # contains.
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_one_class(src: str) -> ast.ClassDef:
        tree = ast.parse(src)
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert len(classes) == 1, "fixture must define exactly one class"
        return classes[0]

    def test_undeclared_soft_read_is_rejected(self):
        """The bug jax-gcm#673 reports: a bare ``.get()`` with no declaration."""
        src = """
class Offender(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    provides: ClassVar[tuple[str, ...]] = ("clouds",)

    def __call__(self, state, diagnostics, forcing, terrain):
        conv = diagnostics.get("convection")
        no_convection = conv is None or conv.ktype == 0
        return tendency, {**diagnostics, "clouds": clouds}
"""
        result = audit_class(self._parse_one_class(src))
        self.assertEqual(result["soft"], ["convection"])
        self.assertEqual(result["hard"], [])

    def test_undeclared_hard_read_is_rejected(self):
        src = """
class Offender(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, state, diagnostics, forcing, terrain):
        conv = diagnostics["convection"]
        return tendency, diagnostics
"""
        result = audit_class(self._parse_one_class(src))
        self.assertEqual(result["hard"], ["convection"])

    def test_soft_read_declared_via_optional_requires_passes(self):
        """The deliberate cross-step/optional pattern (#673's 'fix')."""
        src = """
class CrossStepReader(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    provides: ClassVar[tuple[str, ...]] = ("clouds",)
    optional_requires: ClassVar[tuple[str, ...]] = ("convection",)

    def __call__(self, state, diagnostics, forcing, terrain):
        conv = diagnostics.get("convection")
        no_convection = conv is None or conv.ktype == 0
        return tendency, {**diagnostics, "clouds": clouds}
"""
        result = audit_class(self._parse_one_class(src))
        self.assertEqual(result["soft"], [])
        self.assertEqual(result["hard"], [])

    def test_soft_read_of_own_provided_key_needs_no_extra_declaration(self):
        """A term reading back its own carry seed (``surface_physics.py``-style)."""
        src = """
class OwnCarrySeed(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("surface",)

    def __call__(self, state, diagnostics, forcing, terrain):
        prev = diagnostics.get("surface", Defaults.zeros())
        return tendency, {**diagnostics, "surface": prev.copy()}
"""
        result = audit_class(self._parse_one_class(src))
        self.assertEqual(result["soft"], [])

    def test_in_guard_form_is_also_checked(self):
        src = """
class GuardReader(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, state, diagnostics, forcing, terrain):
        if "vertical_diffusion" in diagnostics:
            tke = diagnostics["vertical_diffusion"].tke
        return tendency, diagnostics
"""
        result = audit_class(self._parse_one_class(src))
        self.assertEqual(result["soft"], ["vertical_diffusion"])

    # ------------------------------------------------------------------
    # Self-tests of the module-constant resolver (Codex finding on #1094):
    # a key given as a module-level constant, an aliased import, or a
    # helper-function call must resolve exactly as Python itself would,
    # using a real throwaway module built for the test rather than a
    # hand-rolled import-graph walk.
    # ------------------------------------------------------------------

    @staticmethod
    def _throwaway_module(ns: dict):
        import types
        mod = types.ModuleType("requires_audit_test._throwaway")
        mod.__dict__.update(ns)
        return mod

    def test_module_constant_key_is_resolved(self):
        """``diagnostics.get(SOME_KEY)`` -- the exact shape Codex flagged."""
        module = self._throwaway_module({"SOME_KEY": "_surface_optics"})
        src = """
class ConstantKeyReader(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, state, diagnostics, forcing, terrain):
        optics = diagnostics.get(SOME_KEY)
        return tendency, diagnostics
"""
        result = audit_class(self._parse_one_class(src), module)
        self.assertEqual(result["soft"], ["_surface_optics"])

    def test_module_constant_key_declared_via_optional_requires_passes(self):
        module = self._throwaway_module({"SOME_KEY": "_surface_optics"})
        src = """
class ConstantKeyReader(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()
    optional_requires: ClassVar[tuple[str, ...]] = ("_surface_optics",)

    def __call__(self, state, diagnostics, forcing, terrain):
        optics = diagnostics.get(SOME_KEY)
        return tendency, diagnostics
"""
        result = audit_class(self._parse_one_class(src), module)
        self.assertEqual(result["soft"], [])

    def test_helper_call_provides_tuple_is_resolved(self):
        """``provides = helper() + (CONST,)`` -- the emissions-terms shape."""
        module = self._throwaway_module({
            "emission_flux_keys": lambda: ("emi_so2", "emi_bc"),
            "WIND_KEY": "_model_level_wind",
        })
        src = """
class HelperProvides(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = (emission_flux_keys() + (WIND_KEY,))

    def __call__(self, state, diagnostics, forcing, terrain):
        wind = diagnostics.get(WIND_KEY, 0.0)
        return tendency, {**diagnostics, "emi_so2": 0.0, "emi_bc": 0.0,
                           "_model_level_wind": wind}
"""
        result = audit_class(self._parse_one_class(src), module)
        # WIND_KEY resolves to "_model_level_wind", which the class's own
        # (resolved) `provides` already carries -- a self-referential
        # carry seed, exactly like the literal-string case.
        self.assertEqual(result["soft"], [])
        self.assertEqual(result["hard"], [])

    def test_unresolvable_key_is_left_unflagged_not_a_false_positive(self):
        """A genuinely dynamic key (a loop/parameter name) must never be flagged.

        ``_eval_against_module`` only has the MODULE's globals to work
        with; a local variable or function parameter is never in scope
        there, so it fails to resolve and the read is silently skipped --
        never misread as an undeclared key with a garbage name.
        """
        module = self._throwaway_module({})
        src = """
class DynamicKeyReader(PhysicsTerm):
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def __call__(self, state, diagnostics, forcing, terrain):
        out = {name: diagnostics.get(name, 0.0) for name in self._names}
        return tendency, {**diagnostics, **out}
"""
        result = audit_class(self._parse_one_class(src), module)
        self.assertEqual(result["soft"], [])
        self.assertEqual(result["hard"], [])


if __name__ == "__main__":
    unittest.main()
