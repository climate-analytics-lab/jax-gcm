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
"""

import ast
import pathlib
import unittest

_PLUMBING = frozenset({
    "_dt_seconds", "_band_config", "_dycore_fields", "_forcing_2d",
    "_echam_params", "_echam_coords", "_speedy_coords",
})


def _literal_tuple(node):
    try:
        v = ast.literal_eval(node)
        return tuple(v) if isinstance(v, (tuple, list)) else None
    except (ValueError, TypeError):
        return None


def audit_class(cls: ast.ClassDef):
    """Return ``{"hard": [...], "soft": [...]}`` undeclared-read lists, or ``None``.

    ``None`` means the class has no ``requires`` declaration AND no
    ``__call__`` — not a ``PhysicsTerm`` this audit has anything to say
    about (a helper class, a dataclass, ...). Exposed as a module-level
    function (rather than nested in the test) so the self-tests below can
    exercise it directly against synthetic source, independent of the
    real tree.
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
                requires = _literal_tuple(val)
            elif name == "provides" and val is not None:
                provides = _literal_tuple(val)
            elif name == "requires_dycore_fields" and val is not None:
                dycore = _literal_tuple(val)
            elif name == "carry_slots" and isinstance(val, ast.Dict):
                carry |= {k.value for k in val.keys if isinstance(k, ast.Constant)}
            elif name == "optional_requires" and val is not None:
                lit = _literal_tuple(val)
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
                and isinstance(n.slice, ast.Constant)
                and isinstance(n.ctx, ast.Load)):
            hard.add(n.slice.value)
        # diagnostics.get("key", ...)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "diagnostics"
                and n.args and isinstance(n.args[0], ast.Constant)):
            soft.add(n.args[0].value)
        # "key" in diagnostics guards
        if (isinstance(n, ast.Compare) and len(n.ops) == 1
                and isinstance(n.ops[0], (ast.In, ast.NotIn))
                and isinstance(n.comparators[0], ast.Name)
                and n.comparators[0].id == "diagnostics"
                and isinstance(n.left, ast.Constant)):
            soft.add(n.left.value)

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
            for cls in [n for n in ast.walk(tree)
                        if isinstance(n, ast.ClassDef)]:
                missing = audit_class(cls)
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


if __name__ == "__main__":
    unittest.main()
