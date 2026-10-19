"""The network-outage gate in ``conftest.py`` actually gates (#1152)."""

import pytest

from jcm.data.mirror import conftest as mirror_conftest


class _Node:
    """Minimal stand-in for a pytest ``item``/``request.node``."""

    def __init__(self, marked: bool):
        self._marks = [pytest.mark.requires_network.mark] if marked else []

    def iter_markers(self, name):
        return iter([m for m in self._marks if m.name == name])


class _Request:
    """Minimal stand-in for a pytest ``request``, for :func:`_gate`.

    ``getfixturevalue`` raises if called for an unmarked node: :func:`_gate`
    must never reach it without first confirming the ``requires_network``
    marker, which is the whole laziness contract the real fixture relies on.
    """

    def __init__(self, node, probe_result=None):
        self.node = node
        self._probe_result = probe_result

    def getfixturevalue(self, name):
        assert name == "_ncar_probe"
        if self._probe_result is None:
            raise AssertionError(
                "must not fetch the network probe for an unmarked test")
        return self._probe_result


class _Response:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(self, status_code):
        self.status_code = status_code
        self.ok = status_code < 400


class TestSkipReason:
    """``_skip_reason`` is the whole gating decision, kept pure and testable
    without constructing a real pytest ``request`` (see module docstring).
    """

    def test_unmarked_test_is_never_skipped(self):
        assert mirror_conftest._skip_reason(
            _Node(marked=False), reachable=False, detail="HTTP 503") is None

    def test_marked_test_is_skipped_with_a_reason_naming_the_host_and_detail_when_unreachable(self):
        reason = mirror_conftest._skip_reason(
            _Node(marked=True), reachable=False, detail="HTTP 503")
        assert reason == f"{mirror_conftest._HOST} unreachable (HTTP 503)"

    def test_marked_test_runs_normally_when_reachable(self):
        assert mirror_conftest._skip_reason(
            _Node(marked=True), reachable=True, detail="HTTP 200") is None


class TestProbeHost:
    """``_probe_host`` turns a HEAD request outcome into ``(reachable, detail)``.

    A completed TLS handshake is not enough -- #1154's review (Codex) caught
    that the earlier version of this probe treated any response, including a
    non-2xx/3xx status, as reachable, which let the six gated tests fail
    inside pooch's own ``raise_for_status()`` on a server error instead of
    skipping. These cases pin both outcomes.
    """

    def test_reachable_on_200(self, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "head",
                            lambda *a, **k: _Response(200))
        assert mirror_conftest._probe_host() == (True, "HTTP 200")

    def test_unreachable_on_503_even_though_the_handshake_completed(self, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "head",
                            lambda *a, **k: _Response(503))
        assert mirror_conftest._probe_host() == (False, "HTTP 503")

    def test_unreachable_on_a_request_exception(self, monkeypatch):
        import requests

        def _raise(*a, **k):
            raise requests.exceptions.SSLError("Max retries exceeded")

        monkeypatch.setattr(requests, "head", _raise)
        reachable, detail = mirror_conftest._probe_host()
        assert reachable is False
        assert "Max retries exceeded" in detail

    def test_unreachable_when_requests_is_not_importable(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_requests(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("no requests")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_requests)
        reachable, detail = mirror_conftest._probe_host()
        assert reachable is False
        assert "requests not importable" in detail


class TestGate:
    """``_gate`` end-to-end: a 503 skips, a 200 runs -- Codex's two cases."""

    def test_unmarked_node_never_touches_the_probe(self):
        # _Request raises if getfixturevalue is ever called here.
        mirror_conftest._gate(_Request(_Node(marked=False)))

    def test_marked_node_skips_on_a_503(self):
        request = _Request(_Node(marked=True), probe_result=(False, "HTTP 503"))
        with pytest.raises(pytest.skip.Exception) as excinfo:
            mirror_conftest._gate(request)
        assert "HTTP 503" in str(excinfo.value)
        assert mirror_conftest._HOST in str(excinfo.value)

    def test_marked_node_runs_on_a_200(self):
        request = _Request(_Node(marked=True), probe_result=(True, "HTTP 200"))
        mirror_conftest._gate(request)  # must not raise/skip


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
