"""Skip this package's NCAR-download tests when their host is unreachable.

Stopgap half of #1152. Six tests under ``jcm/data/mirror/`` --
``build_mirror_test.py::test_stage_bundles_writes_emis_biogenic_oc``,
``bundles_test.py``'s three ``build_emissions_nc`` callers, and
``emissions_test.py::BuildEmissionsNcBiogenicTest``'s two tests -- each
reach ``build_emissions_nc`` without mocking ``add_cam6_soa``'s source
lookup (:func:`jcm.data.mirror.bundles.add_cam6_soa`), so they
unconditionally pull the real CAM6 SOAG CMIP6 files from
``svn-ccsm-inputdata.cgd.ucar.edu`` through
:func:`jcm.data.emissions.cam6_soa.prepare_cam6_soa` ->
:func:`jcm.data.emissions.downloader.fetch` (pooch -> requests). A remote
TLS hiccup or outage then fails every PR's dependency-parity fast suite
and, through its cancel step, the slow shards too (#1152: CI runs
39180061235 and 39188287495, both 2026-10-19, both
``requests.exceptions.SSLError: ... Max retries exceeded``).

This makes an outage read as a clearly-labelled skip instead of a red PR.
It is deliberately NOT the real fix: #1152's preferred fix (a) is a small
packaged/synthetic CMIP6-shaped fixture so these tests never touch the
network at all, which is NOT built here and remains open under #1152.

Every test above is tagged ``@pytest.mark.requires_network`` (registered
in ``pyproject.toml``). The autouse fixture below is the only thing that
reads that marker: an unmarked test in this package never probes the
network and pays nothing for this module's presence.
"""

from __future__ import annotations

import pytest

#: The one host every ``requires_network`` test in this package shares --
#: both jcm.data.emissions.cam6_soa.INPUTDATA and
#: jcm.data.bc.cam_landuse.SOURCE_URL resolve here. A single hardcoded host
#: (rather than a per-test argument to the marker) is deliberate: nothing
#: under jcm/data/mirror/ reaches any other external host at test time
#: today (verified by grepping jcm/ for svn-ccsm-inputdata/requests.get/
#: urllib/download when #1152 was filed). A future test against a
#: different host needs this module extended, not reused as-is.
_HOST = "svn-ccsm-inputdata.cgd.ucar.edu"
_PORT = 443
_TIMEOUT_S = 5.0


def _host_reachable(host: str = _HOST, port: int = _PORT,
                     timeout: float = _TIMEOUT_S) -> bool:
    """Whether ``host`` answers an HTTPS HEAD within ``timeout`` seconds.

    A HEAD request (not a bare TCP connect) deliberately completes the TLS
    handshake, because #1152's actual failures were
    ``requests.exceptions.SSLError`` at the handshake, not a connection
    refusal -- a plain ``socket.create_connection`` would read that outage
    as reachable. ``requests`` is not a direct jcm dependency, but every
    test this gates already needs it transitively through pooch
    (:mod:`jcm.data.emissions.downloader`) to do the real fetch, so an
    environment where it is missing could not run them anyway; treat that
    case as unreachable too rather than raising out of a fixture.
    """
    try:
        import requests
    except ImportError:
        return False
    try:
        requests.head(f"https://{host}:{port}", timeout=timeout)
    except requests.exceptions.RequestException:
        return False
    return True


def _skip_reason(node, reachable) -> str | None:
    """Return a skip reason for ``node`` if it is ``requires_network``-marked
    and ``reachable`` (a bool, or a zero-argument callable returning one --
    laziness lives here so the real fixture never calls the network probe
    for an unmarked test) says the host is down; ``None`` otherwise.

    Kept separate from the autouse fixture below so the gating logic is
    testable without constructing real pytest fixtures/requests (see
    ``conftest_test.py``), and reused unchanged by the fixture.
    """
    if not any(node.iter_markers("requires_network")):
        return None
    is_reachable = reachable() if callable(reachable) else reachable
    if is_reachable:
        return None
    return f"{_HOST} unreachable"


@pytest.fixture(scope="session")
def _ncar_host_reachable() -> bool:
    """``_host_reachable()``, memoised for the life of the session.

    Session-scoped so every ``requires_network`` test sharing a pytest
    process (or one xdist worker) pays for one probe, not one per test.
    Pytest only builds a fixture a collected, selected test actually
    requests -- via :func:`_skip_if_ncar_unreachable`'s lazy
    ``request.getfixturevalue`` below -- so a session that never runs one
    of these six tests never opens a socket either.
    """
    return _host_reachable()


@pytest.fixture(autouse=True)
def _skip_if_ncar_unreachable(request):
    """Skip a ``requires_network`` test when :data:`_HOST` is unreachable.

    Autouse so tagging a test with the marker is the only thing its author
    has to do. For every other test in this package the marker lookup is
    the whole cost: ``request.getfixturevalue`` (and so the network probe
    behind it) is only reached when ``_skip_reason`` has already confirmed
    the current test is marked.
    """
    reason = _skip_reason(
        request.node, lambda: request.getfixturevalue("_ncar_host_reachable"))
    if reason is not None:
        pytest.skip(reason)
    yield
