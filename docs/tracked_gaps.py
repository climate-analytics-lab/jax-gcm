"""Tracked-gap references in the living model description.

A ``#NNN`` in ``docs/source/science/**`` asserts that the gap it names is
still open; a closed one silently converts a documented limitation into a
claim the reader believes was fixed. Three consumers police that invariant and
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

#: A tracked-gap reference. Deliberately the same shape the rot guard has
#: always used, so this refactor cannot change which references are policed.
ISSUE_REF = re.compile(r"#(\d{1,6})\b")

_API = "https://api.github.com/repos/climate-analytics-lab/jax-gcm/issues/{}"


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


def citations() -> dict[str, list[str]]:
    """Map each cited issue number to the page names citing it."""
    refs: dict[str, list[str]] = {}
    for page in science_pages():
        for num in ISSUE_REF.findall(page.read_text()):
            if page.name not in refs.setdefault(num, []):
                refs[num].append(page.name)
    return refs


def pages_citing(number: int | str) -> list[str]:
    """Page names citing ``number``; empty if the register does not."""
    return citations().get(str(int(number)), [])


def issue_state(number: int | str, *, token: str | None = None,
                timeout: float = 10.0) -> str:
    """Return ``"open"``, ``"closed"``, ``"missing"`` or ``"pull_request"``.

    A pull request is reported separately because the issues endpoint returns
    PRs too, and a PR — even an open one — is not a durable tracked gap.

    Raises:
        ApiUnavailable: on rate limiting, server errors or a network failure.

    """
    headers = {"Accept": "application/vnd.github+json"}
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        # Unauthenticated is 60 req/hr per IP, which a shared runner exhausts;
        # the Actions-provided token is 5000 req/hr.
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(_API.format(int(number)), headers=headers)
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
    for num, pages in sorted(citations().items(), key=lambda kv: int(kv[0])):
        state = issue_state(num, token=token)
        where = f"(cited in {sorted(set(pages))})"
        if state == "missing":
            stale.append(f"#{num} does not exist {where}")
        elif state == "pull_request":
            stale.append(f"#{num} is a pull request, not an issue {where}")
        elif state != "open":
            stale.append(f"#{num} is {state} {where}")
    return stale


def _main(argv: list[str]) -> int:
    """``cites <number>`` or ``stale`` — the entry points the workflows call."""
    if len(argv) == 3 and argv[1] == "cites":
        pages = pages_citing(argv[2])
        print("\n".join(pages))
        return 0 if pages else 1
    if len(argv) == 2 and argv[1] == "stale":
        try:
            stale = stale_citations()
        except ApiUnavailable as e:
            # Exit 0: an outage is not a documentation defect, and the sweep
            # must not file an issue about GitHub being briefly unreachable.
            print(f"Issues API unavailable ({e}); no verdict.")
            return 0
        print("\n".join(stale))
        return 1 if stale else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
