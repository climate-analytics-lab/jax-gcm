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
#: both jcm.data.emissions.cam6_soa.HISTORICAL_INPUTDATA and
#: jcm.data.bc.cam_landuse.SOURCE_URL resolve here. A single hardcoded host
#: (rather than a per-test argument to the marker) is deliberate: nothing
#: under jcm/data/mirror/ reaches any other external host at test time
#: today (verified by grepping jcm/ for svn-ccsm-inputdata/requests.get/
#: urllib/download when #1152 was filed). A future test against a
#: different host needs this module extended, not reused as-is.
_HOST = "svn-ccsm-inputdata.cgd.ucar.edu"
_TIMEOUT_S = 5.0


def _probe_url() -> str:
    """Return the exact resource the six gated tests end up fetching.

    All six reach ``add_cam6_soa`` with ``year_range`` set (never a bare
    ``sources=None`` with no era), which makes ``prepare_cam6_soa`` take its
    ``historical=True`` branch -- ``HISTORICAL_INPUTDATA``/
    ``HISTORICAL_SOURCES``, not the plain ``INPUTDATA``/``SOURCES`` climo
    pair. Probing that specific file (rather than the bare host) is what
    makes "reachable" mean "the download these tests are about to do will
    succeed," not merely "the server answers something." Imported lazily so
    importing ``jcm.data.emissions.cam6_soa`` (and its chain through
    ``jcm.data.emissions.prepare`` into ``jcm.physics.aerosol...``) is never
    paid by a test in this package that collects but never runs a
    ``requires_network`` one.
    """
    from jcm.data.emissions.cam6_soa import HISTORICAL_INPUTDATA, HISTORICAL_SOURCES

    return HISTORICAL_INPUTDATA + HISTORICAL_SOURCES["anthro"]


def _probe_host() -> tuple[bool, str]:
    """Probe :func:`_probe_url` and return ``(reachable, detail)``.

    ``detail`` is always set -- an HTTP status on either outcome, or the
    exception text on a connection/handshake failure -- so a caller can
    report *why* a host was judged unreachable, not just that it was.

    Checks ``response.ok`` (status < 400: 2xx and 3xx), not merely that the
    HEAD request completed. Codex caught this in #1154's review: a HEAD to
    the bare host can finish its TLS handshake and still come back with a
    non-2xx/3xx status (a 503, a maintenance redirect loop, ...), which an
    earlier version of this probe read as "reachable" -- so a server-side
    error surfaced as a test FAILURE inside pooch's own
    ``raise_for_status()`` instead of a skip, exactly the #1152 failure mode
    this module exists to turn into a clean skip. ``requests`` is not a
    direct jcm dependency, but every test this gates already needs it
    transitively through pooch (:mod:`jcm.data.emissions.downloader`) to do
    the real fetch, so an environment where it is missing could not run
    them anyway; treated as unreachable too rather than raising out of a
    fixture.
    """
    try:
        import requests
    except ImportError as exc:
        return False, f"requests not importable: {exc}"
    try:
        response = requests.head(_probe_url(), timeout=_TIMEOUT_S,
                                 allow_redirects=True)
    except requests.exceptions.RequestException as exc:
        return False, str(exc)
    return response.ok, f"HTTP {response.status_code}"


def _skip_reason(node, reachable: bool, detail: str) -> str | None:
    """Return a skip reason for ``node`` if it is ``requires_network``-marked
    and ``reachable`` is ``False``; ``None`` otherwise (unmarked, or marked
    and reachable).

    Kept separate from the autouse fixture below so the gating decision is
    testable directly, without constructing real pytest fixtures/requests
    (see ``conftest_test.py``), and reused unchanged by the fixture.
    """
    if not any(node.iter_markers("requires_network")):
        return None
    if reachable:
        return None
    return f"{_HOST} unreachable ({detail})"


@pytest.fixture(scope="session")
def _ncar_probe() -> tuple[bool, str]:
    """``_probe_host()``, memoised for the life of the session.

    Session-scoped so every ``requires_network`` test sharing a pytest
    process (or one xdist worker) pays for one probe, not one per test.
    Pytest only builds a fixture a collected, selected test actually
    requests -- via :func:`_gate`'s lazy ``request.getfixturevalue`` below
    -- so a session that never runs one of these six tests never opens a
    socket either.
    """
    return _probe_host()


def _gate(request) -> None:
    """Skip the running test if it is marked and the session probe found
    the shared resource unreachable; a no-op otherwise.

    A plain function, not the fixture body itself, so it can be driven
    directly in ``conftest_test.py`` with a fake ``request`` whose
    ``getfixturevalue`` would raise if an unmarked test ever reached it --
    which is the laziness contract this exists to prove.
    """
    if not any(request.node.iter_markers("requires_network")):
        return
    reachable, detail = request.getfixturevalue("_ncar_probe")
    reason = _skip_reason(request.node, reachable, detail)
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _skip_if_ncar_unreachable(request):
    """Skip a ``requires_network`` test when the shared resource is down.

    Autouse so tagging a test with the marker is the only thing its author
    has to do. For every other test in this package :func:`_gate`'s marker
    check is the whole cost: the session probe is only reached once a
    ``requires_network``-marked test has actually been selected to run.
    """
    _gate(request)
    yield
