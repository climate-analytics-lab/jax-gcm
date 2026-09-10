"""Unit tests for the HF mirror downloader (no network, no huggingface_hub)."""

import sys
import types
import unittest
from unittest import mock

from jcm.data import remote


def _fake_hub(download):
    """sys.modules stand-in for huggingface_hub (works without it)."""

    class LocalEntryNotFoundError(Exception):
        pass

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    hub.errors = errors
    patcher = mock.patch.dict(sys.modules, {
        "huggingface_hub": hub, "huggingface_hub.errors": errors})
    return patcher, LocalEntryNotFoundError


class FetchTest(unittest.TestCase):
    def test_fetch_is_cache_first(self):
        # a warm cache must resolve with local_files_only (no network)
        calls = []

        def fake(**kw):
            calls.append(kw)
            return "/cache/hit"

        patcher, _ = _fake_hub(fake)
        with patcher:
            self.assertEqual(remote.fetch("bundles/x.nc"), "/cache/hit")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["local_files_only"])

    def test_cache_miss_goes_online_then_errors_helpfully(self):
        def cold_then_ok(**kw):
            if kw.get("local_files_only"):
                raise err("not cached")
            return "/downloaded"

        patcher, err = _fake_hub(cold_then_ok)
        with patcher:
            self.assertEqual(
                remote.bundle_file("t63", "terrain.nc"), "/downloaded")

        def always_fails(**kw):
            if kw.get("local_files_only"):
                raise err2("not cached")
            raise ConnectionError("no internet")

        patcher, err2 = _fake_hub(always_fails)
        with patcher:
            with self.assertRaisesRegex(FileNotFoundError,
                                        "prefetch on a login node"):
                remote.fetch("bundles/t63/terrain.nc")


if __name__ == "__main__":
    unittest.main()
