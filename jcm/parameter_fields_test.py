"""Every field of a ``*Parameters`` struct is read by some code.

A tunable that nothing reads is a documented, differentiable parameter with a
zero gradient: the struct stores and documents the field while the scheme
rebuilds the quantity from a constant, so a calibration moves nothing.
Finiteness checks do not see this, since zero is finite, and the per-term
gradient tests only ask that *some* parameter of a term is live. This module
asks the question of every field, statically.

A field counts as read when a non-test module under ``jcm/`` loads an
attribute of that name (``config.cevapcu``) or calls ``getattr`` with it as a
literal. The class's own body does not count (a ``validate`` that loops over
field names is not a consumer), nor do tests (a parameter only a test reads
does nothing in a model), nor does constructing the struct with the field as a
keyword. The scan matches by *name* across the whole package, so it cannot
tell two classes' same-named fields apart: it catches a field nobody reads, not
one that is read but multiplied by zero, and a field that shares its name with
a read attribute elsewhere is invisible to it.
"""

import ast
import collections
import pathlib

import pytest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

# Fields that no code reads, each a known gap rather than an oversight.
#
# The first two are documented as inert where the decision was made (their
# docstrings say so), so they have no issue. The rest are declared, defaulted
# and documented but have no consumer, the same defect ``cevapcu`` had, and
# are tracked in jax-gcm#999: each is to be wired to the formulation it names
# or removed. An entry that becomes read fails
# ``test_known_unread_fields_are_still_unread``, so the table only shrinks.
_UNREAD_FIELDS = {
    "HinesParameters": {
        # ``hines.py``: "currently never enabled in production; this field is
        # kept for API compatibility".
        "cutoff_altitude",
    },
    "SSOParameters": {
        # ``lott_miller.py``: the mountain-lift branch (``orolift``) is not
        # implemented, disabled by default in production ECHAM.
        "mountain_lift_coeff",
    },
    # jax-gcm#999
    "RadiationParameters": {"sw_band_limits"},
    "SurfaceParameters": {
        "nsfc_type", "ml_depth", "rho_water", "cp_water", "rho_ice",
        "cp_ice", "conduct_ice",
    },
    "VDiffParameters": {"totte_min", "cchar", "nsfc_type", "itop"},
}


def _is_getattr_with_literal(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    )


def _reads(tree):
    """Names loaded as attributes, or fetched by a literal ``getattr``."""
    names = collections.Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            names[node.attr] += 1
        elif _is_getattr_with_literal(node):
            names[node.args[1].value] += 1
    return names


def _parameter_classes(tree):
    """``(class name, class node)`` for each ``*Parameters``/``*Params`` class."""
    return [
        (node.name, node) for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name.endswith(("Parameters", "Params"))
    ]


def _fields(cls):
    return [stmt.target.id for stmt in cls.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)]


def _unread_fields(sources):
    """``({class: {unread fields}}, n_classes, n_fields)`` over ``sources``.

    ``sources`` maps a label to Python source. Reads are pooled across all of
    it; each class's own body is subtracted from its fields' counts.
    """
    trees = {label: ast.parse(src, filename=label)
             for label, src in sources.items()}
    reads = collections.Counter()
    for tree in trees.values():
        reads.update(_reads(tree))
    unread = collections.defaultdict(set)
    n_classes = n_fields = 0
    for tree in trees.values():
        for name, cls in _parameter_classes(tree):
            fields = _fields(cls)
            if not fields:
                continue
            n_classes += 1
            own = _reads(cls)
            for field in fields:
                n_fields += 1
                if reads[field] - own[field] <= 0:
                    unread[name].add(field)
    return dict(unread), n_classes, n_fields


@pytest.fixture(scope="module")
def package_scan():
    sources = {
        str(path): path.read_text()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
        if not path.name.endswith("_test.py") and path.name != "conftest.py"
    }
    return _unread_fields(sources)


def test_no_new_parameter_field_goes_unread(package_scan):
    unread, _, _ = package_scan
    new = {name: sorted(fields - _UNREAD_FIELDS.get(name, set()))
           for name, fields in unread.items()
           if fields - _UNREAD_FIELDS.get(name, set())}
    assert not new, (
        "These parameter fields are never read by any non-test code under "
        f"jcm/, so they are tunables with a zero gradient: {new}. Read each "
        "where its physics uses it (a hard-coded constant that ought to be "
        "the parameter is the usual cause), or delete the field.")


def test_known_unread_fields_are_still_unread(package_scan):
    unread, _, _ = package_scan
    resolved = {name: sorted(fields - unread.get(name, set()))
                for name, fields in _UNREAD_FIELDS.items()
                if fields - unread.get(name, set())}
    assert not resolved, (
        f"These fields are now read (or gone): {resolved}. Drop them from "
        "_UNREAD_FIELDS so the table only lists what is still unread.")


def test_the_scan_sees_the_package(package_scan):
    # A scan that silently matches nothing would pass the two tests above, so
    # pin that it finds the package's parameter structs and the known gaps.
    unread, n_classes, n_fields = package_scan
    assert n_classes >= 30 and n_fields >= 300
    assert unread, "the scan finds none of the known unread fields"


def test_the_scan_flags_a_field_nobody_reads():
    # The scenario this module exists for: a field stored on the struct and
    # a consumer that rebuilds the quantity from a constant instead.
    sources = {
        "types.py": (
            "class ToyParameters:\n"
            "    used: float\n"
            "    inert: float\n"
            "    via_getattr: float\n"
            "    def validate(self):\n"
            "        return self.inert\n"
        ),
        "scheme.py": (
            "def run(cfg):\n"
            "    return cfg.used * 1.93e-6 + getattr(cfg, 'via_getattr')\n"
        ),
    }
    unread, n_classes, n_fields = _unread_fields(sources)
    assert (n_classes, n_fields) == (1, 3)
    # ``inert`` is read only inside its own class body, which is no consumer.
    assert unread == {"ToyParameters": {"inert"}}
