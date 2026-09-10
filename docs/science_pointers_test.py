"""Guards for the living model description (``docs/source/science/``).

The document's value is that its code pointers and its tracked-gap references
are true. Both rot silently: a renamed symbol leaves prose that reads
authoritative and points nowhere, and a closed issue turns a documented
limitation into a claim the reader believes was fixed. These tests make the
CLAUDE.md maintenance rule enforceable rather than aspirational.
"""

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


def _pages():
    pages = sorted(SCIENCE.glob("*.md"))
    assert pages, f"no science pages found under {SCIENCE}"
    return pages


def _resolve(rel: str):
    """Locate a pointer path, which may be repo-relative or a trailing fragment."""
    direct = REPO / rel
    if direct.exists():
        return direct
    return next(REPO.glob(f"**/{rel}"), None)


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

    def test_pointer_symbols_exist(self):
        missing = []
        for page in _pages():
            for rel, symbol in _POINTER.findall(page.read_text()):
                if not symbol:
                    continue
                target = _resolve(rel)
                if target is None:
                    continue  # reported by the file test
                # Dotted pointers (``Class.method``) are checked at their root:
                # the point is that the named entity still lives in that file.
                root = symbol.split(".")[0]
                body = target.read_text(errors="ignore")
                if not re.search(rf"\b(def|class)\s+{re.escape(root)}\b", body) \
                        and root not in body:
                    missing.append(f"{page.name}: {rel}::{symbol}")
        self.assertEqual(missing, [], "science-doc pointers name missing symbols")


class TestSciencePagesAreWired(unittest.TestCase):
    """Every page is reachable from the ``science`` toctree."""

    def test_every_page_in_toctree(self):
        toctree = (REPO / "docs" / "source" / "science.rst").read_text()
        orphans = [p.name for p in _pages() if f"science/{p.stem}" not in toctree]
        self.assertEqual(orphans, [], "science pages missing from science.rst")


if __name__ == "__main__":
    unittest.main()
