"""The network-outage gate in ``conftest.py`` actually gates (#1152)."""

import pytest

from jcm.data.mirror import conftest as mirror_conftest


class _Node:
    """Minimal stand-in for a pytest ``item``/``request.node``."""

    def __init__(self, marked: bool):
        self._marks = [pytest.mark.requires_network.mark] if marked else []

    def iter_markers(self, name):
        return iter([m for m in self._marks if m.name == name])


class TestSkipReason:
    """``_skip_reason`` is the whole gating decision, kept pure and testable
    without constructing a real pytest ``request`` (see module docstring).
    """

    def test_unmarked_test_is_never_skipped(self):
        # Reachability must not even be consulted for an unmarked test --
        # passing a callable that would raise if invoked proves it.
        def _boom():
            raise AssertionError("must not probe the network for an unmarked test")

        assert mirror_conftest._skip_reason(_Node(marked=False), _boom) is None

    def test_marked_test_is_skipped_with_a_reason_naming_the_host_when_unreachable(self):
        reason = mirror_conftest._skip_reason(_Node(marked=True), False)
        assert reason == f"{mirror_conftest._HOST} unreachable"

    def test_marked_test_runs_normally_when_reachable(self):
        assert mirror_conftest._skip_reason(_Node(marked=True), True) is None

    def test_reachable_argument_may_be_lazy(self):
        """The real fixture passes a zero-arg callable, not a bool, so the
        network probe fixture is only ever built for a marked test.
        """
        calls = []

        def probe():
            calls.append(1)
            return True

        assert mirror_conftest._skip_reason(_Node(marked=True), probe) is None
        assert calls == [1]
        # An unmarked test never calls the probe at all (checked above), and
        # a marked-but-reachable test calls it exactly once.


class TestHostReachable:
    """``_host_reachable`` turns a HEAD request outcome into a bool."""

    def test_reachable_when_head_succeeds(self, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "head", lambda *a, **k: None)
        assert mirror_conftest._host_reachable() is True

    def test_unreachable_on_a_request_exception(self, monkeypatch):
        import requests

        def _raise(*a, **k):
            raise requests.exceptions.SSLError("Max retries exceeded")

        monkeypatch.setattr(requests, "head", _raise)
        assert mirror_conftest._host_reachable() is False

    def test_unreachable_when_requests_is_not_importable(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_requests(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("no requests")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_requests)
        assert mirror_conftest._host_reachable() is False


class TestFixtureWiring:
    """The autouse fixture is actually wired in for tests in this package."""

    def test_gate_fixture_runs_around_every_test_here(self, request):
        assert "_skip_if_ncar_unreachable" in request.fixturenames

    def test_marker_itself_is_registered_in_pyproject(self, request):
        # Not applying the marker to this test itself: that would make this
        # unit test's own result depend on live network reachability, which
        # is exactly what the gate below exists to keep out of the fast
        # suite. Registration is checked directly against the ini markers
        # instead.
        markers = request.config.getini("markers")
        assert any(m.startswith("requires_network:") for m in markers)
