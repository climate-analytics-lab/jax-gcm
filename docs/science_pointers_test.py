"""Guards for the living model description (``docs/source/science/``).

The document's value is that its code pointers and its tracked-gap references
are true. Both rot silently: a renamed symbol leaves prose that reads
authoritative and points nowhere, and a closed issue turns a documented
limitation into a claim the reader believes was fixed. These tests make the
CLAUDE.md maintenance rule enforceable rather than aspirational.
"""

import ast
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCIENCE = REPO / "docs" / "source" / "science"

# ``file.py`` or ``file.py::symbol`` inside RST/MyST double backticks. Fortran
# references (mo_*.f90) name the ECHAM/CAM trees, which are not in this repo.
_POINTER = re.compile(
    r"``([A-Za-z0-9_./-]+\.(?:py|yaml|json))(?:::([A-Za-z0-9_.]+))?``"
)
# A bare ``Symbol`` / ``dotted.Symbol`` literal, as used in Code-pointer
# bullets of the form ``file.py`` — ``ClassA``, ``func_b``.
_BARE_SYMBOL = re.compile(r"``([A-Za-z_][A-Za-z0-9_.]*)``")
# Backtick literals that are config values / knobs, not Python symbols.
_NON_SYMBOLS = frozenset({
    "auto", "null", "true", "false", "none", "default", "hybrid", "sigma",
})


def _pages():
    pages = sorted(SCIENCE.glob("*.md"))
    assert pages, f"no science pages found under {SCIENCE}"
    return pages


class Ambiguous(list):
    """More than one file matches a non-repo-relative pointer fragment."""


def _resolve(rel: str):
    """Locate a pointer path, which may be repo-relative or a trailing fragment.

    A fragment (``lohmann_2m/types.py``) must match exactly one file: accepting
    the first of several would let the guard validate a pointer against an
    unrelated same-named module and stay green after the real one moved.
    """
    direct = REPO / rel
    if direct.exists():
        return direct
    matches = [m for m in REPO.glob(f"**/{rel}") if ".git" not in m.parts]
    if len(matches) > 1:
        return Ambiguous(sorted(str(m.relative_to(REPO)) for m in matches))
    return matches[0] if matches else None


def _defined_names(path: Path) -> set[str]:
    """Every name a module defines, as dotted paths down to class members.

    Walking the AST (rather than grepping ``def name``) is what lets a dotted
    pointer like ``Model.resume`` fail when the *method* is renamed while the
    class survives. Module-level assignments count too — several pointers name
    constants (``_DTDT_MAX``) and re-export aliases.
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return set()

    names: set[str] = set()

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                names.add(prefix + child.name)
                if isinstance(child, ast.ClassDef):
                    visit(child, prefix + child.name + ".")
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) \
                    else [child.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        names.add(prefix + t.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add(prefix + (alias.asname or
                                        alias.name.split(".")[0]))

    visit(tree)
    return names


def _symbol_defined(path: Path, symbol: str) -> bool:
    if path.suffix != ".py":
        # yaml/json pointers carry key names; a text check is the right level.
        return symbol in path.read_text(errors="ignore")
    return symbol in _defined_names(path)


def _bullet_claims(text: str):
    """Yield (file, symbol) claims from **Code pointers** sections.

    Only those sections use the reliable ``file.py`` — ``X``, ``Y`` convention;
    elsewhere a bare literal after a file mention is usually a reference-model
    name (``physc``, ``micro_mg``) and would be a false claim. Within a
    bullet, each bare ``Symbol`` literal is checked against the nearest
    *preceding* file pointer; config-value literals are skipped.
    """
    sections = re.findall(
        r"\*\*Code pointers[^*\n]*\*\*(.*?)(?=\n\*\*|\Z)", text, re.S,
    )
    for bullet in (b for s in sections for b in re.split(r"\n(?=- )", s)):
        bullet = " ".join(bullet.splitlines())
        events = []
        for m in _POINTER.finditer(bullet):
            events.append((m.start(), "file", m.group(1)))
        for m in _BARE_SYMBOL.finditer(bullet):
            name = m.group(1)
            if ("." in name and name.rsplit(".", 1)[-1] in
                    ("py", "yaml", "json", "f90", "F90", "nc", "csv", "md")):
                continue
            if name.lower() in _NON_SYMBOLS or name.startswith("mo_"):
                continue
            events.append((m.start(), "symbol", name))
        events.sort()
        current = None
        for _, kind, value in events:
            if kind == "file":
                current = value
            elif current is not None:
                yield current, value


class TestSciencePointersResolve(unittest.TestCase):
    """Every ``file::symbol`` pointer names code that exists."""

    def test_pointer_files_exist(self):
        missing = [
            f"{page.name}: {rel}"
            for page in _pages()
            for rel, _ in _POINTER.findall(page.read_text())
            if _resolve(rel) is None
        ]
        self.assertEqual(missing, [], "science-doc pointers name missing files")

    def test_pointers_resolve_unambiguously(self):
        ambiguous = [
            f"{page.name}: {rel} matches {list(hit)}"
            for page in _pages()
            for rel, _ in _POINTER.findall(page.read_text())
            if isinstance(hit := _resolve(rel), Ambiguous)
        ]
        self.assertEqual(
            ambiguous, [],
            "science-doc pointers match several files — write them "
            "repo-relative so the guard checks the intended one",
        )

    def test_qualified_symbols_exist(self):
        """``file::symbol`` pointers, resolved on the full dotted path."""
        missing = []
        for page in _pages():
            for rel, symbol in _POINTER.findall(page.read_text()):
                if not symbol:
                    continue
                target = _resolve(rel)
                if target is None or isinstance(target, Ambiguous):
                    continue  # reported by the file / ambiguity tests
                if not _symbol_defined(target, symbol):
                    missing.append(f"{page.name}: {rel}::{symbol}")
        self.assertEqual(missing, [], "science-doc pointers name missing symbols")

    def test_bullet_symbols_exist(self):
        """Bare ``Symbol`` literals following a file pointer in one bullet."""
        missing = []
        for page in _pages():
            for rel, symbol in _bullet_claims(page.read_text()):
                target = _resolve(rel)
                if target is None or isinstance(target, Ambiguous):
                    continue
                if not _symbol_defined(target, symbol) and \
                        symbol not in target.read_text(errors="ignore"):
                    missing.append(f"{page.name}: {rel} — {symbol}")
        self.assertEqual(
            missing, [],
            "code-pointer bullets name symbols their file does not define",
        )


class TestSciencePagesAreWired(unittest.TestCase):
    """Every page is reachable from the ``science`` toctree."""

    def test_every_page_in_toctree(self):
        toctree = (REPO / "docs" / "source" / "science.rst").read_text()
        orphans = [p.name for p in _pages() if f"science/{p.stem}" not in toctree]
        self.assertEqual(orphans, [], "science pages missing from science.rst")


if __name__ == "__main__":
    unittest.main()
