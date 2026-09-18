"""Hermetic tests for the tracked-gap helpers.

Every network call is stubbed. The point of #836 was that a documentation
check should not depend on GitHub's current state, so the module that decides
what counts as stale must itself be testable without asking GitHub anything.
"""

import io
import json
import os
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import tracked_gaps as tg


def _response(payload):
    body = io.BytesIO(json.dumps(payload).encode())
    body.__enter__ = lambda self=body: self
    body.__exit__ = lambda *a: False
    return body


class CitationParsingTest(unittest.TestCase):
    """What counts as a citation, read off the real register."""

    def test_register_carries_refs(self):
        self.assertTrue(tg.citations(), "the register should carry gap refs")

    def test_numbers_map_to_the_pages_that_cite_them(self):
        for num, pages in tg.citations().items():
            with self.subTest(num=num):
                self.assertTrue(pages)
                for name in pages:
                    self.assertIn(f"#{num}", (tg.SCIENCE / name).read_text())

    def test_pages_are_deduplicated(self):
        for num, pages in tg.citations().items():
            self.assertEqual(len(pages), len(set(pages)), num)

    def test_uncited_number_has_no_pages(self):
        self.assertEqual(tg.pages_citing(999999), [])

    def test_accepts_int_or_str(self):
        num = next(iter(tg.citations()))
        self.assertEqual(tg.pages_citing(int(num)), tg.pages_citing(num))

    def test_ref_shape_is_unchanged(self):
        r"""The regex must keep matching exactly what the old guard matched.

        Including its quirks: a run of seven digits matches nothing at all
        (the ``\b`` refuses every truncation of it), while a ``#`` glued to a
        preceding word still counts. Both are pre-existing behaviour, pinned
        here so this refactor cannot quietly change which refs are policed.
        """
        text = "see #123 and #45678, not #1234567 or C#4 or a#9"
        self.assertEqual(tg.ISSUE_REF.findall(text), ["123", "45678", "4", "9"])

    def test_science_pages_are_found(self):
        self.assertTrue(all(p.suffix == ".md" for p in tg.science_pages()))


class IssueStateTest(unittest.TestCase):
    """Each API outcome maps to exactly one verdict."""

    def _state(self, payload=None, error=None, number=1):
        target = "tracked_gaps.urllib.request.urlopen"
        side = error if error is not None else None
        with mock.patch(target, side_effect=side,
                        return_value=_response(payload or {})):
            return tg.issue_state(number, token="t")

    def test_open_and_closed(self):
        self.assertEqual(self._state({"state": "open"}), "open")
        self.assertEqual(self._state({"state": "closed"}), "closed")

    def test_pull_request_is_not_a_gap(self):
        self.assertEqual(
            self._state({"state": "open", "pull_request": {}}), "pull_request")

    def test_404_and_410_mean_the_citation_is_wrong(self):
        for code in (404, 410):
            err = urllib.error.HTTPError("u", code, "m", {}, None)
            self.assertEqual(self._state(error=err), "missing")

    def test_outages_raise_rather_than_accuse_the_docs(self):
        """403/429/5xx say nothing about the register, so they get no verdict."""
        for code in (403, 429, 500, 502, 503):
            err = urllib.error.HTTPError("u", code, "m", {}, None)
            with self.subTest(code=code), self.assertRaises(tg.ApiUnavailable):
                self._state(error=err)

    def test_network_failure_raises_api_unavailable(self):
        # ``.__name__``, not the class: a subTest parameter travels to the
        # xdist controller in the sub-report, and execnet serialises only
        # simple values — a type object fails there, under -n but not under a
        # plain unittest run.
        for err in (urllib.error.URLError("down"), TimeoutError(), OSError()):
            with self.subTest(err=type(err).__name__), \
                    self.assertRaises(tg.ApiUnavailable):
                self._state(error=err)


class StaleCitationsTest(unittest.TestCase):
    def _stale(self, states):
        with mock.patch.object(tg, "citations",
                               return_value={k: ["p.md"] for k in states}), \
             mock.patch.object(tg, "issue_state",
                               side_effect=lambda n, **kw: states[str(n)]):
            return tg.stale_citations()

    def test_open_refs_are_not_stale(self):
        self.assertEqual(self._stale({"1": "open", "2": "open"}), [])

    def test_every_bad_state_is_reported(self):
        stale = self._stale({"1": "open", "2": "closed",
                             "3": "missing", "4": "pull_request"})
        self.assertEqual(len(stale), 3)
        self.assertTrue(any("#2 is closed" in s for s in stale))
        self.assertTrue(any("#3 does not exist" in s for s in stale))
        self.assertTrue(any("#4 is a pull request" in s for s in stale))

    def test_an_outage_propagates_rather_than_reading_as_clean(self):
        """A partial sweep must never look like a clean register."""
        def boom(n, **kw):
            raise tg.ApiUnavailable("429")

        with mock.patch.object(tg, "citations", return_value={"1": ["p.md"]}), \
             mock.patch.object(tg, "issue_state", side_effect=boom), \
             self.assertRaises(tg.ApiUnavailable):
            tg.stale_citations()


class CliTest(unittest.TestCase):
    """The two entry points the workflows call, and their exit codes."""

    def test_cites_exit_codes(self):
        num = next(iter(tg.citations()))
        self.assertEqual(tg._main(["p", "cites", num]), 0)
        self.assertEqual(tg._main(["p", "cites", "999999"]), 1)

    def test_stale_exit_codes(self):
        with mock.patch.object(tg, "stale_citations", return_value=[]):
            self.assertEqual(tg._main(["p", "stale"]), 0)
        with mock.patch.object(tg, "stale_citations", return_value=["#1 is closed"]):
            self.assertEqual(tg._main(["p", "stale"]), 1)

    def test_stale_reports_no_verdict_on_an_outage(self):
        """Exit 0, so a GitHub incident cannot file a documentation issue."""
        with mock.patch.object(tg, "stale_citations",
                               side_effect=tg.ApiUnavailable("429")):
            self.assertEqual(tg._main(["p", "stale"]), 0)

    def test_usage(self):
        self.assertEqual(tg._main(["p"]), 2)


class RepoWiringTest(unittest.TestCase):
    """The decoupling #836 bought must not be quietly undone.

    These are cheap text assertions on purpose: the thing worth protecting is
    a *policy* — issue state never decides a pull request — and the way it
    gets lost is somebody moving the check back for convenience.
    """

    def _workflow(self, name):
        return (Path(tg.REPO) / ".github" / "workflows" / name).read_text()

    def _parsed(self, name):
        """Parse a workflow, skipping where PyYAML is absent.

        Not stdlib, and the docs job deliberately installs almost nothing —
        it pip-installs PyYAML for exactly these assertions. The skip keeps a
        bare local run from erroring instead of reporting.
        """
        try:
            import yaml  # noqa: PLC0415 — optional, only for these assertions
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyYAML not installed; workflow policy unchecked")
        return yaml.safe_load(self._workflow(name))

    def test_workflow_invokes_both_subcommands(self):
        wf = self._workflow("science_register.yaml")
        self.assertIn("tracked_gaps.py cites", wf)
        self.assertIn("tracked_gaps.py stale", wf)

    def test_workflow_never_runs_on_a_pull_request(self):
        """The whole point: a PR cannot fail for something no commit did."""
        spec = self._parsed("science_register.yaml")
        # PyYAML reads a bare ``on:`` key as the boolean True.
        triggers = set(spec.get("on", spec.get(True, {})))
        self.assertNotIn("pull_request", triggers)
        self.assertNotIn("push", triggers)
        self.assertEqual(triggers, {"issues", "schedule", "workflow_dispatch"})

    def test_docs_job_does_not_grant_issue_access(self):
        """No token, no permission: the PR-path guard cannot reach the API.

        Structural rather than a text search, so the prose explaining why the
        permission is absent does not read as the permission being present.
        """
        spec = self._parsed("run_docs.yaml")
        self.assertNotIn("issues", spec.get("permissions", {}))
        for job, jspec in spec["jobs"].items():
            for step in jspec["steps"]:
                with self.subTest(job=job, step=step.get("name")):
                    self.assertNotIn("GITHUB_TOKEN", step.get("env", {}) or {})

    def test_issue_state_check_is_opt_in(self):
        """Without the env var the rot guard must skip, not query GitHub."""
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items()
               if k != "JCM_CHECK_TRACKED_GAPS"}
        out = subprocess.run(
            [sys.executable, "science_pointers_test.py",
             "TestTrackedGapsAreOpen", "-v"],
            cwd=Path(tg.REPO) / "docs", env=env,
            capture_output=True, text=True, timeout=120)
        self.assertIn("skipped", out.stderr.lower(), out.stderr)
        self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
    unittest.main()
