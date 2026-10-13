"""Tracked-gap references in the living model description.

A ``#NNN`` in ``docs/source/science/**`` asserts that the gap it names is
still open; a closed one silently converts a documented limitation into a
claim the reader believes was fixed. A cross-repository ``repo#NNN`` /
``owner/repo#NNN`` (e.g. ``jax-rrtmgp#37``) makes the same claim about that
repository's issue, and is resolved there. Three consumers police that invariant and
must agree on what counts as a citation, so the parsing and the API lookup
live here rather than in any one of them:

* ``science_pointers_test.py`` — the rot guard's opt-in network check.
* ``.github/workflows/science_register.yaml``'s close hook — tells whoever
  just closed an issue that a science page still cites it.
* the same workflow's scheduled sweep — catches what the hook cannot see
  (transfers, deletions, bulk closes, and citations of issues that never
  existed).

Why the state check is NOT part of the pull-request gate: its verdict depends
on GitHub, not on the tree, so a green merge could turn red with no commit
touching the repo, and the failure landed on the next contributor rather than
on whoever closed the issue (#836). The checks that ARE pure functions of the
tree stay in the blocking suite, where they catch rot on the commit that
causes it.

Stdlib only, deliberately: the guard runs in a docs job with no jcm install.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCIENCE = REPO / "docs" / "source" / "science"

#: The repository a bare ``#NNN`` names, and the owner a bare ``repo#NNN``
#: resolves under (the register's cross-repository citations name sibling
#: repositories of this organisation, e.g. ``jax-rrtmgp#37``).
HOME_REPO = "climate-analytics-lab/jax-gcm"
HOME_OWNER = HOME_REPO.split("/")[0]

#: A tracked-gap reference: ``#NNN`` (this repository), ``repo#NNN`` (a
#: sibling repository of :data:`HOME_OWNER`) or ``owner/repo#NNN``. The
#: optional prefix is part of the SAME match, so a cross-repository reference
#: is never also read as a citation of this repository's ``#NNN`` — reading
#: ``jax-rrtmgp#37`` as jax-gcm#37 reported a live jax-rrtmgp issue as "a pull
#: request, not an issue" (#882). Seven digits still match nothing at all: the
#: ``\b`` refuses every truncation of the run. The look-behind anchors a match
#: at the start of a token, so a deeper path (``a/b/c#3``) is not misread as
#: owner ``b``, repository ``c``.
ISSUE_REF = re.compile(
    r"(?<![A-Za-z0-9._/-])"
    r"(?:(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/)?"
    r"(?P<repo>[A-Za-z0-9._-]*[A-Za-z0-9_])?"
    r"#(?P<number>\d{1,6})\b"
)

_API = "https://api.github.com/repos/{repo}/issues/{number}"


def parse_refs(text: str) -> list[tuple[str, str]]:
    """Every tracked-gap reference in ``text`` as ``(repo, number)`` pairs.

    ``repo`` is the fully qualified ``owner/name``: :data:`HOME_REPO` for a
    bare ``#NNN``, ``HOME_OWNER/name`` for ``name#NNN``, and as written for
    ``owner/name#NNN``.
    """
    refs = []
    for m in ISSUE_REF.finditer(text):
        owner, name = m.group("owner"), m.group("repo")
        if name is None:
            repo = HOME_REPO
        else:
            repo = f"{owner or HOME_OWNER}/{name}"
        refs.append((repo, m.group("number")))
    return refs


def ref_label(repo: str, number: str | int) -> str:
    """How a reference is printed: ``#N`` at home, ``owner/name#N`` abroad."""
    return f"#{number}" if repo == HOME_REPO else f"{repo}#{number}"


class ApiUnavailable(RuntimeError):
    """The Issues API could not be reached, or answered with an outage code.

    Distinct from a citation being wrong: rate limits, 5xx and network
    failures say nothing about the documentation, so every caller treats this
    as "no verdict" rather than as a failure. An underprivileged 403 is
    indistinguishable from rate limiting, and lands here too.
    """


def science_pages() -> list[Path]:
    """Every page of the living model description, sorted."""
    pages = sorted(SCIENCE.glob("*.md"))
    if not pages:
        raise AssertionError(f"no science pages found under {SCIENCE}")
    return pages


def all_citations() -> dict[tuple[str, str], list[str]]:
    """Map each cited ``(repo, number)`` to the page names citing it.

    Cross-repository references are included: a ``jax-rrtmgp#37`` asserts
    that jax-rrtmgp issue 37 is open exactly as ``#37`` asserts it of this
    repository's, so the sweep resolves each against its own repository.
    """
    refs: dict[tuple[str, str], list[str]] = {}
    for page in science_pages():
        for ref in parse_refs(page.read_text()):
            if page.name not in refs.setdefault(ref, []):
                refs[ref].append(page.name)
    return refs


def citations() -> dict[str, list[str]]:
    """Map each cited issue number OF THIS REPOSITORY to the pages citing it.

    What the close hook asks about: closing jax-gcm issue ``N`` concerns only
    the pages citing ``#N``, never a page citing another repository's ``N``.
    """
    return {num: pages for (repo, num), pages in all_citations().items()
            if repo == HOME_REPO}


def pages_citing(number: int | str) -> list[str]:
    """Page names citing ``number``; empty if the register does not."""
    return citations().get(str(int(number)), [])


def issue_state(number: int | str, *, repo: str = HOME_REPO,
                token: str | None = None, timeout: float = 10.0) -> str:
    """Return ``"open"``, ``"closed"``, ``"missing"`` or ``"pull_request"``.

    A pull request is reported separately because the issues endpoint returns
    PRs too, and a PR — even an open one — is not a durable tracked gap.
    ``repo`` is the ``owner/name`` the number belongs to (see
    :func:`parse_refs`).

    Raises:
        ApiUnavailable: on rate limiting, server errors or a network failure.

    """
    headers = {"Accept": "application/vnd.github+json"}
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        # Unauthenticated is 60 req/hr per IP, which a shared runner exhausts;
        # the Actions-provided token is 5000 req/hr.
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        _API.format(repo=repo, number=int(number)), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code in (404, 410):
            return "missing"
        raise ApiUnavailable(f"HTTP {e.code}") from e
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise ApiUnavailable(str(e)) from e
    if "pull_request" in payload:
        return "pull_request"
    return payload.get("state", "unknown")


def stale_citations(*, token: str | None = None) -> list[str]:
    """Human-readable lines for every citation that no longer holds.

    Raises:
        ApiUnavailable: if any lookup hits an outage, so a caller never reads
            a short list as "the register is clean".

    """
    stale = []
    # Home references first, then each other repository's, numerically.
    order = lambda kv: (kv[0][0] != HOME_REPO, kv[0][0], int(kv[0][1]))  # noqa: E731
    for (repo, num), pages in sorted(all_citations().items(), key=order):
        state = issue_state(num, repo=repo, token=token)
        label = ref_label(repo, num)
        where = f"(cited in {sorted(set(pages))})"
        if state == "missing":
            stale.append(f"{label} does not exist {where}")
        elif state == "pull_request":
            stale.append(f"{label} is a pull request, not an issue {where}")
        elif state != "open":
            stale.append(f"{label} is {state} {where}")
    return stale


#: ``stale`` exit codes. Three, not two: "the register is clean" and "I could
#: not find out" are different answers, and a caller that conflates them will
#: act on an outage as though it were good news — closing a still-valid report,
#: in the sweep's case.
CLEAN, STALE, NO_VERDICT, USAGE = 0, 1, 3, 2


def _main(argv: list[str]) -> int:
    """``cites <number>`` or ``stale`` — the entry points the workflows call.

    ``cites`` exits 0 when the register cites the number and 1 when it does
    not. ``stale`` exits :data:`CLEAN`, :data:`STALE` or :data:`NO_VERDICT`.
    """
    if len(argv) == 3 and argv[1] == "cites":
        pages = pages_citing(argv[2])
        print("\n".join(pages))
        return CLEAN if pages else 1
    if len(argv) == 2 and argv[1] == "stale":
        try:
            stale = stale_citations()
        except ApiUnavailable as e:
            # An outage is not a documentation defect. It is also not a clean
            # register: the caller must be able to tell the two apart, and do
            # nothing rather than act on either.
            print(f"Issues API unavailable ({e}); no verdict.")
            return NO_VERDICT
        print("\n".join(stale))
        return STALE if stale else CLEAN
    print(__doc__)
    return USAGE


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
