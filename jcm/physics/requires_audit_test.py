"""Static audit: every unconditional diagnostics read is declared.

Walks each ``PhysicsTerm`` subclass's ``__call__`` and asserts that every
bare ``diagnostics["key"]`` subscript is covered by ``requires`` ∪
``provides`` ∪ ``carry_slots`` ∪ ``requires_dycore_fields`` ∪ the
framework plumbing keys. Reads via ``diagnostics.get("key")`` or behind an
``if "key" in diagnostics`` guard are optional-by-convention and exempt
(see the ``requires`` comment in ``physics_term.py``).
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


def _audit_class(cls):
    requires = provides = dycore = None
    carry = set()
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
    return sorted((hard - soft) - allowed)


class RequiresAuditTest(unittest.TestCase):
    def test_every_unconditional_read_is_declared(self):
        root = pathlib.Path(__file__).parent
        violations = {}
        for py in sorted(root.rglob("*.py")):
            if py.name.endswith("_test.py"):
                continue
            tree = ast.parse(py.read_text())
            for cls in [n for n in ast.walk(tree)
                        if isinstance(n, ast.ClassDef)]:
                missing = _audit_class(cls)
                if missing:
                    violations[f"{py.relative_to(root)}:{cls.name}"] = missing
        self.assertFalse(
            violations,
            "PhysicsTerm(s) read diagnostics keys not declared in "
            f"requires (or read them via .get/guard if optional): {violations}",
        )


if __name__ == "__main__":
    unittest.main()
