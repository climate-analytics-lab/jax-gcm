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
# ``some/package/`` — a directory pointer (trailing slash is the convention).
_DIR_POINTER = re.compile(r"``([A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*/)``")
# ``dir/{a,b}.yaml`` — a brace-grouped pointer naming several sibling files.
_BRACE_POINTER = re.compile(
    r"``([A-Za-z0-9_./-]*)\{([A-Za-z0-9_,-]+)\}([A-Za-z0-9_.-]*)``")
# Tracked-gap references. The register's rule is that an issue number means an
# OPEN gap; a closed one silently converts a documented limitation into a
# claim the reader believes was fixed.
_ISSUE_REF = re.compile(r"#(\d{3,4})\b")
# A bare ``Symbol`` / ``dotted.Symbol`` literal, as used in Code-pointer
# bullets of the form ``file.py`` — ``ClassA``, ``func_b``.
_BARE_SYMBOL = re.compile(r"``([A-Za-z_][A-Za-z0-9_.]*)``")
# Backtick literals that are config values / knobs, not Python symbols.
_NON_SYMBOLS = frozenset({
    "auto", "null", "true", "false", "none", "default", "hybrid", "sigma",
})


def _brace_expansions(text: str):
    for head, group, tail in _BRACE_POINTER.findall(text):
        for variant in group.split(","):
            yield f"{head}{variant}{tail}"


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
                else:
                    # Function-local constants (``_DTDT_MAX``) are real named
                    # entities pointers cite; collect them unprefixed.
                    visit(child, "")
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
    names = _defined_names(path)
    if symbol in names:
        return True
    # A bare literal may cite a class member (a protocol method, a classmethod
    # constructor): accept it when it is a member of any class in the file.
    return any(n.endswith("." + symbol) for n in names)


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

    def test_brace_grouped_pointers_exist(self):
        """``dir/{a,b}.yaml`` claims every named sibling."""
        missing = []
        for page in _pages():
            for rel in _brace_expansions(page.read_text()):
                hit = _resolve(rel)
                if isinstance(hit, Ambiguous):
                    missing.append(
                        f"{page.name}: {rel} matches {list(hit)} — write it "
                        "repo-relative")
                elif hit is None:
                    missing.append(f"{page.name}: {rel}")
        self.assertEqual(
            missing, [], "brace-grouped pointers name missing files",
        )

    def test_directory_pointers_exist(self):
        missing = []
        for page in _pages():
            for rel in _DIR_POINTER.findall(page.read_text()):
                hit = _resolve(rel.rstrip("/"))
                if isinstance(hit, Ambiguous):
                    missing.append(
                        f"{page.name}: {rel} matches {list(hit)} — write it "
                        "repo-relative")
                elif hit is None or not hit.is_dir():
                    missing.append(f"{page.name}: {rel}")
        self.assertEqual(
            missing, [], "science-doc pointers name missing directories",
        )

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
                # AST-strict: a mention in a comment or docstring is exactly
                # the stale-reference scenario this guard exists to catch.
                if not _symbol_defined(target, symbol):
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


class TestTrackedGapsAreOpen(unittest.TestCase):
    """Every ``#NNN`` reference points at an OPEN issue.

    Queries the public GitHub API (unauthenticated; a handful of requests) and
    skips cleanly when the network or the API is unavailable, so offline runs
    and rate-limited CI are not broken by it.
    """

    def test_issue_refs_are_open(self):
        import json
        import urllib.error
        import urllib.request

        refs: dict[str, list[str]] = {}
        for page in _pages():
            for num in _ISSUE_REF.findall(page.read_text()):
                refs.setdefault(num, []).append(page.name)
        self.assertTrue(refs, "the register should carry tracked-gap refs")

        stale = []
        for num, pages in sorted(refs.items()):
            url = ("https://api.github.com/repos/"
                   f"climate-analytics-lab/jax-gcm/issues/{num}")
            req = urllib.request.Request(
                url, headers={"Accept": "application/vnd.github+json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    state = json.load(resp).get("state")
            except urllib.error.HTTPError as e:
                # Only a permanent 404/410 means the citation itself is wrong.
                # Rate limits and server-side 5xx are outages: skip, so an API
                # incident cannot fail otherwise-valid documentation CI.
                if e.code in (404, 410):
                    stale.append(f"#{num} does not exist (HTTP {e.code}; "
                                 f"cited in {sorted(set(pages))})")
                    continue
                self.skipTest(f"GitHub API unavailable (HTTP {e.code}); "
                              "issue-state check skipped")
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                self.skipTest(f"GitHub API unreachable ({e}); "
                              "issue-state check skipped")
            if state != "open":
                stale.append(f"#{num} is {state} (cited in {sorted(set(pages))})")
        self.assertEqual(
            stale, [],
            "closed issues cited as tracked gaps — either the gap is fixed "
            "(update the section) or it needs a new open issue",
        )


if __name__ == "__main__":
    unittest.main()
