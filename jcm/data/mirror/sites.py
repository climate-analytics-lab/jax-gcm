"""Where each mirror source lives on the machines the builder runs on.

The builders read the same upstream products on every site — the CMIP7
input4MIPs tree, CESM inputdata, the NCAR RDA ERA5 archive, the ECHAM-HAMMOZ
input pool — but each machine mounts a different subset under different roots.
This module is the one place those roots are written down; every stage
resolves its source paths through :func:`current` instead of hard-coding them.

* ``glade``   — NCAR Derecho/Casper. Everything except the HAMMOZ pool.
* ``levante`` — DKRZ Levante. input4MIPs and the HAMMOZ/ECHAM6 pools are on
  ``/pool/data``; there is no RDA ERA5 archive, so the ERA5-derived Tier A
  product is *pulled* from the published mirror (``--stage pull``) rather than
  rebuilt. The few CESM inputdata files are downloaded once into
  ``$JCM_MIRROR_ROOT/sources/cesm_inputdata`` (same tree layout as the CESM
  inputdata server).

A root is ``None`` where the site does not provide that source; a stage that
needs it then fails fast in :func:`jcm.data.mirror.build_mirror.check_sources`
naming the site and the missing source.

Selection: ``JCM_MIRROR_SITE`` if set, else auto-detection by mount point, else
``glade`` (so module-level path constants still form, e.g. for unit tests on a
laptop, and the source check reports what is missing).
Individual roots can be overridden with ``JCM_HAMMOZ_DIR`` (a directory laid out
like ``/pool/data/ECHAM6-HAMMOZ``) and ``JCM_CESM_INPUTDATA``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Site:
    """Source roots on one machine (``None`` = not available there)."""

    name: str
    #: Root holding ``CMIP7/CMIP/<institution>/...`` input4MIPs products.
    input4mips: str
    #: CESM inputdata tree (``atm/cam/...``): Lana DMS, CAM L26 oxidants.
    cesm_inputdata: str
    #: Directory of the WACCM CCMI REFC1 decade oxidant climatologies
    #: (``oxid_ozone_WACCM_CCMI_REFC1_f.e11.FWTREFC1.<decade>...nc``).
    waccm_oxidants: str
    #: NCAR RDA data root (``d633000``/``d633001`` ERA5 products).
    rda: str | None
    #: CESM ne30np4 topography file for the ne30pg3 terrain bundle.
    ne30_topo: str | None
    #: ECHAM-HAMMOZ input pool root (``v0007/hammoz/T63/...``).
    hammoz: str | None
    #: ECHAM6 input pool root (``T127/T127GR15_jan_surf.nc``) — terrain
    #: cross-checks only, never a bundle source.
    echam6: str | None
    #: Default ``JCM_MIRROR_ROOT`` (build + upload trees) on this site.
    default_root: str


def _user() -> str:
    return os.environ.get("USER", "")


def _glade() -> Site:
    inputdata = "/glade/campaign/cesm/cesmdata/inputdata"
    return Site(
        name="glade",
        input4mips=("/glade/campaign/cesm/cesmdata/input4MIPs_raw/"
                    "input4MIPs"),
        cesm_inputdata=inputdata,
        waccm_oxidants="/glade/p/cesmdata/cseg/inputdata/atm/cam/ozone",
        rda="/glade/campaign/collections/rda/data",
        ne30_topo=(f"{inputdata}/atm/cam/topo/se/"
                   "ne30np4_gmted2010_modis_bedmachine_nc3000_Laplace0100_"
                   "noleak_greenlndantarcsgh30fac2.50_20250825.nc"),
        hammoz=None,
        echam6=None,
        default_root=f"/glade/derecho/scratch/{_user()}/hf_mirror",
    )


def _levante() -> Site:
    user = _user()
    root = f"/scratch/{user[:1]}/{user}/hf_mirror"
    inputdata = f"{os.environ.get('JCM_MIRROR_ROOT', root)}/sources/cesm_inputdata"
    return Site(
        name="levante",
        input4mips="/pool/data/INPUT4MIP/data/input4MIPs",
        cesm_inputdata=inputdata,
        waccm_oxidants=f"{inputdata}/atm/cam/ozone",
        rda=None,
        ne30_topo=None,
        hammoz="/pool/data/ECHAM6-HAMMOZ",
        echam6="/pool/data/ECHAM6",
        default_root=root,
    )


SITES = {"glade": _glade, "levante": _levante}


def detect() -> str:
    """Return the name of the site this process runs on (see module docstring)."""
    name = os.environ.get("JCM_MIRROR_SITE")
    if name:
        if name not in SITES:
            raise ValueError(f"JCM_MIRROR_SITE={name!r}; known sites: "
                             f"{', '.join(SITES)}")
        return name
    if Path("/glade").is_dir():
        return "glade"
    if Path("/pool/data/ECHAM6-HAMMOZ").is_dir():
        return "levante"
    return "glade"


def current() -> Site:
    """Return the active :class:`Site`, with the per-root environment overrides."""
    site = SITES[detect()]()
    overrides = {}
    if os.environ.get("JCM_HAMMOZ_DIR"):
        overrides["hammoz"] = os.environ["JCM_HAMMOZ_DIR"]
    if os.environ.get("JCM_CESM_INPUTDATA"):
        overrides["cesm_inputdata"] = os.environ["JCM_CESM_INPUTDATA"]
    return replace(site, **overrides) if overrides else site


def input4mips(relative: str) -> str:
    """Absolute path of a product under the active site's input4MIPs root."""
    return f"{current().input4mips}/{relative}"
