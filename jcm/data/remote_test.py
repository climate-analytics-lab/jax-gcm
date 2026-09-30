"""Unit tests for the HF mirror downloader (no network, no huggingface_hub)."""

import os
import sys
import types
import unittest
from unittest import mock

from jcm.data import remote

A = "a" * 40


class _HTTPError(OSError):
    """Stand-in for ``HfHubHTTPError`` (carries a status)."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = types.SimpleNamespace(status_code=status)


def _fake_hub(download):
    """sys.modules stand-in for huggingface_hub (works without it)."""

    class LocalEntryNotFoundError(Exception):
        pass

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    errors.HfHubHTTPError = _HTTPError
    hub.errors = errors
    patcher = mock.patch.dict(sys.modules, {
        "huggingface_hub": hub, "huggingface_hub.errors": errors})
    return patcher, LocalEntryNotFoundError


def _env(value=None):
    """Environment with the override set to ``value`` (or removed)."""
    env = {k: v for k, v in os.environ.items() if k != remote.REVISION_ENV}
    if value is not None:
        env[remote.REVISION_ENV] = value
    return mock.patch.dict(os.environ, env, clear=True)


def _online_raises(exc):
    """Return a hub patch with nothing cached whose download raises ``exc``."""
    def download(**kw):
        if kw.get("local_files_only"):
            raise miss("not cached")
        raise exc
    patcher, miss = _fake_hub(download)
    return patcher


class RevisionTest(unittest.TestCase):
    def test_pin_is_a_commit_sha(self):
        self.assertRegex(remote.MIRROR_REVISION, r"\A[0-9a-f]{40}\Z")

    def test_pin_and_override(self):
        with _env():
            self.assertEqual(remote.mirror_revision(), remote.MIRROR_REVISION)
            self.assertEqual(remote.revision_source(), "pinned")
        remote._FROZEN = None                        # a new process
        with _env(A):
            self.assertEqual(remote.mirror_revision(), A)
            self.assertEqual(remote.revision_source(), "env")

    def test_changing_the_override_within_a_process_raises(self):
        with _env(A):
            remote.mirror_revision()
        with _env("b" * 40), self.assertRaisesRegex(
                RuntimeError, f"changed from {A} to {'b' * 40}"):
            remote.mirror_revision()

    def test_branch_override_is_refused_with_the_resolving_command(self):
        with _env("main"), self.assertRaisesRegex(
                ValueError, r"dataset_info\(.*revision='main'\)\.sha"):
            remote.mirror_revision()


class FetchTest(unittest.TestCase):
    def test_cache_first_at_the_process_revision(self):
        calls = []

        def download(**kw):
            calls.append(kw)
            return "/cache/hit"

        patcher, _ = _fake_hub(download)
        with patcher, _env(A):
            self.assertEqual(remote.bundle_file("t63", "terrain.nc"),
                             "/cache/hit")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["local_files_only"])
        self.assertEqual(calls[0]["revision"], A)

    def test_cache_miss_downloads_at_the_same_revision(self):
        calls = []

        def download(**kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                raise miss("not cached")
            return "/downloaded"

        patcher, miss = _fake_hub(download)
        with patcher, _env():
            self.assertEqual(remote.fetch("x.nc"), "/downloaded")
        self.assertEqual({c["revision"] for c in calls},
                         {remote.MIRROR_REVISION})

    def test_unreachable_hub_names_the_prefetch(self):
        # Direct and wrapped (HF_HUB_OFFLINE=1 raises LocalEntryNotFoundError
        # caused by the offline error) transport failures, and a 5xx.
        wrapped = OSError("miss")
        wrapped.__cause__ = ConnectionError("offline mode")
        for exc in (ConnectionError("no internet"), wrapped, _HTTPError(503)):
            with _online_raises(exc), _env(A):
                with self.assertRaisesRegex(
                        FileNotFoundError,
                        f"prefetch.*\n  {remote.REVISION_ENV}={A} python"):
                    remote.fetch("x.nc")

    def test_hub_refusal_does_not_suggest_prefetch(self):
        for status in (401, 404):
            with _online_raises(_HTTPError(status)), _env():
                with self.assertRaises(FileNotFoundError) as cm:
                    remote.fetch("x.nc")
            self.assertNotIn("prefetch", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
