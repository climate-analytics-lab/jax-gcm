"""Unit tests for ``build_mirror.py``'s staging drivers (jax-gcm#1017 F7).

``build_mirror.py`` is a Glade/Levante-only I/O driver (omitted from the
coverage gate, ``.coveragerc``) that otherwise cannot run in CI, but its
pure wiring -- which per-process builder it calls, with which arguments --
is still worth testing directly against synthetic Tier A sources, the same
pattern ``bundles_test.py``/``emissions_test.py`` use for the builders
themselves. This covers the one gap those two do not: that the
``biogenic`` stage stages HAM's biogenic-OC source as a Tier A product, and
that ``stage_bundles`` (and ``stage_amip``) load that product and forward it
into every ``build_emissions_nc``/``build_emissions_year`` call.
"""

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.mirror import build_mirror
from jcm.data.mirror.bundles import _ANTHRO_SECTORS, _ANTHRO_SUBSETS

_SPECIES = ("SO2", "BC", "OC")
_NLAT_SRC, _NLON_SRC = 36, 72  # 5-degree regular source grid
_TEST_GRID = "t63"  # a real PUBLISHED_GRIDS member so _truncation() resolves


def _synthetic_ceds_bb(build_dir: Path, seed=0) -> None:
    """Tiny ``ceds_anthro.zarr``/``bb4cmip7.zarr``, both eras, every channel

    ``build_emissions_nc`` reads -- the same shape as ``bundles_test.py``'s
    own fixture, just covering both ``_pd_clim``/``_pi_clim`` suffixes since
    ``stage_bundles`` builds both eras unconditionally.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-87.5, 87.5, _NLAT_SRC)
    lon = np.arange(_NLON_SRC) * 5.0
    month = np.arange(1, 13)

    def field():
        return rng.uniform(0.1, 1.0, (12, _NLAT_SRC, _NLON_SRC)).astype(np.float32)

    ceds = xr.Dataset(coords={"lat": lat, "lon": lon, "month": month})
    bb = xr.Dataset(coords={"lat": lat, "lon": lon, "month": month})
    for sp in _SPECIES:
        for era in ("pd", "pi"):
            for sector in _ANTHRO_SECTORS:
                ceds[f"{sp}_{sector}_{era}_clim"] = xr.DataArray(
                    field(), dims=("month", "lat", "lon"),
                    coords={"lat": lat, "lon": lon, "month": month})
            for subset in _ANTHRO_SUBSETS:
                ceds[f"{sp}_{subset}_{era}_clim"] = xr.DataArray(
                    field(), dims=("month", "lat", "lon"),
                    coords={"lat": lat, "lon": lon, "month": month})
            bb[f"{sp}_{era}_clim"] = xr.DataArray(
                field(), dims=("month", "lat", "lon"),
                coords={"lat": lat, "lon": lon, "month": month})
    ceds.to_zarr(str(build_dir / "ceds_anthro.zarr"))
    bb.to_zarr(str(build_dir / "bb4cmip7.zarr"))


def _synthetic_transient_ceds_bb(build_dir: Path, year: int, seed=0) -> None:
    """One year's worth of ``ceds_anthro.zarr``/``bb4cmip7.zarr``'s transient

    (non-``_clim``) variables, what ``build_emissions_year`` (via
    ``stage_amip``) reads -- species/sector-keyed with a real ``time`` axis,
    unlike ``_synthetic_ceds_bb``'s ``_<era>_clim``/``month`` convention.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-87.5, 87.5, _NLAT_SRC)
    lon = np.arange(_NLON_SRC) * 5.0
    time = xr.date_range(f"{year}-01-01", periods=12, freq="MS")

    def field():
        return rng.uniform(0.1, 1.0, (12, _NLAT_SRC, _NLON_SRC)).astype(np.float32)

    ceds = xr.Dataset(coords={"lat": lat, "lon": lon, "time": time})
    bb = xr.Dataset(coords={"lat": lat, "lon": lon, "time": time})
    for sp in _SPECIES:
        for sector in _ANTHRO_SECTORS:
            ceds[f"{sp}_{sector}"] = (("time", "lat", "lon"), field())
        for subset in _ANTHRO_SUBSETS:
            ceds[f"{sp}_{subset}"] = (("time", "lat", "lon"), field())
        bb[sp] = (("time", "lat", "lon"), field())
    ceds.to_zarr(str(build_dir / "ceds_anthro.zarr"))
    bb.to_zarr(str(build_dir / "bb4cmip7.zarr"))


def _synthetic_biogenic_oc(seed=1) -> xr.DataArray:
    """Build a ``load_biogenic_oc``-shaped stand-in: 12 monthly records, no file I/O."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-87.5, 87.5, _NLAT_SRC)
    lon = np.arange(_NLON_SRC) * 5.0
    data = rng.uniform(1.0e-13, 1.0e-12, (12, _NLAT_SRC, _NLON_SRC)).astype(np.float32)
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"lat": lat, "lon": lon}, name="oc_biogenic")


def _tier_a_biogenic(build_dir):
    """Stand in for ``load_biogenic_oc``, checking it reads the Tier A copy."""
    from jcm.data.mirror.emissions import HAM_BIOGENIC_OC_PRODUCT

    def load(path):
        assert Path(path) == build_dir / HAM_BIOGENIC_OC_PRODUCT, path
        return _synthetic_biogenic_oc()
    return load


def test_biogenic_oc_is_a_tier_a_product():
    """Published under ``products/``, pulled, and checked before bundles/amip."""
    from jcm.data.mirror.emissions import (HAM_BIOGENIC_OC_PRODUCT,
                                           HAM_BIOGENIC_OC_RELPATH)
    from jcm.data.mirror.sites import current

    assert HAM_BIOGENIC_OC_PRODUCT in build_mirror._TIER_A
    assert f"products/{HAM_BIOGENIC_OC_PRODUCT}" in build_mirror._TIER_A_PULL
    product = str(build_mirror.BUILD / HAM_BIOGENIC_OC_PRODUCT)
    without_tree = dataclasses.replace(current(), ham_emissions=None)
    table = build_mirror._stage_sources(without_tree)
    assert table["biogenic"] == ()
    for stage in ("bundles", "amip"):
        assert ("Tier A HAM biogenic OC", product) in table[stage]
    with_tree = dataclasses.replace(current(), ham_emissions="/ham")
    assert build_mirror._stage_sources(with_tree)["biogenic"] == (
        ("ECHAM-HAM emissions tree (AeroCom II biogenic OC)",
         f"/ham/{HAM_BIOGENIC_OC_RELPATH}"),)


def test_stage_biogenic_copies_the_site_tree_or_fetches_the_mirror(monkeypatch):
    """The HAM file verbatim from the site's tree; without one, the published copy."""
    from jcm.data.mirror.emissions import (HAM_BIOGENIC_OC_PRODUCT,
                                           HAM_BIOGENIC_OC_RELPATH)
    from jcm.data.mirror.sites import current

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        monkeypatch.setattr(build_mirror, "BUILD", build_dir)
        tree = tmp_path / "ham"
        src = tree / HAM_BIOGENIC_OC_RELPATH
        src.parent.mkdir(parents=True)
        src.write_bytes(b"site copy")
        monkeypatch.setattr(build_mirror, "SITE",
                            dataclasses.replace(current(), ham_emissions=str(tree)))
        build_mirror.stage_biogenic()
        assert (build_dir / HAM_BIOGENIC_OC_PRODUCT).read_bytes() == b"site copy"

        published = tmp_path / "published.nc"
        published.write_bytes(b"mirror copy")
        fetched = []
        monkeypatch.setattr("jcm.data.remote.fetch",
                            lambda path: fetched.append(path) or str(published))
        monkeypatch.setattr(build_mirror, "SITE",
                            dataclasses.replace(current(), ham_emissions=None))
        build_mirror.stage_biogenic()
        assert fetched == [f"products/{HAM_BIOGENIC_OC_PRODUCT}"]
        assert (build_dir / HAM_BIOGENIC_OC_PRODUCT).read_bytes() == b"mirror copy"


def test_stage_bundles_writes_emis_biogenic_oc(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_dir = tmp_path / "build"
        upload_dir = tmp_path / "upload"
        build_dir.mkdir()
        _synthetic_ceds_bb(build_dir)

        monkeypatch.setattr(build_mirror, "BUILD", build_dir)
        monkeypatch.setattr(build_mirror, "UPLOAD", upload_dir)
        monkeypatch.setattr(build_mirror, "_grids", lambda transient=False: {_TEST_GRID: 8})
        monkeypatch.setattr(build_mirror, "_want", lambda product: product == "emissions")
        monkeypatch.setattr(
            "jcm.data.mirror.emissions.load_biogenic_oc",
            _tier_a_biogenic(build_dir))

        build_mirror.stage_bundles()

        for era in ("pd", "pi"):
            out = xr.open_dataset(
                upload_dir / "bundles" / _TEST_GRID / f"emissions_{era}.nc")
            assert "emis_biogenic_oc" in out.data_vars, (
                f"emis_biogenic_oc missing from the {era} bundle -- "
                "stage_bundles did not forward load_biogenic_oc() into "
                "build_emissions_nc")
            assert np.all(np.isfinite(out["emis_biogenic_oc"].values))


def test_stage_bundles_omits_biogenic_oc_when_not_wanted(monkeypatch):
    """``_want("emissions")=False`` must not even load the biogenic source."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_dir = tmp_path / "build"
        upload_dir = tmp_path / "upload"
        build_dir.mkdir()

        monkeypatch.setattr(build_mirror, "BUILD", build_dir)
        monkeypatch.setattr(build_mirror, "UPLOAD", upload_dir)
        monkeypatch.setattr(build_mirror, "_grids", lambda transient=False: {_TEST_GRID: 8})
        monkeypatch.setattr(build_mirror, "_want", lambda product: False)

        def _boom(path):
            raise AssertionError("load_biogenic_oc must not be called")

        monkeypatch.setattr("jcm.data.mirror.emissions.load_biogenic_oc", _boom)

        build_mirror.stage_bundles()


def test_stage_amip_writes_emis_biogenic_oc(monkeypatch):
    """The yearly transient twin of the climatology test above: ``stage_amip``

    must also forward ``load_biogenic_oc()`` into ``build_emissions_year``.
    """
    year = 2000
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_dir = tmp_path / "build"
        upload_dir = tmp_path / "upload"
        build_dir.mkdir()
        _synthetic_transient_ceds_bb(build_dir, year)

        monkeypatch.setattr(build_mirror, "BUILD", build_dir)
        monkeypatch.setattr(build_mirror, "UPLOAD", upload_dir)
        monkeypatch.setattr(build_mirror, "_grids",
                            lambda transient=False: {_TEST_GRID: 8})
        monkeypatch.setattr(build_mirror, "_want", lambda product: product == "emissions")
        monkeypatch.setattr(build_mirror, "_AMIP_YEARS", (year, year))
        # The staged-coverage sidecar path is a module-level constant
        # derived from BUILD at import time, not re-derived from the
        # monkeypatched BUILD above -- redirect it too so
        # _record_staged_coverage (called unconditionally at the end of
        # stage_amip) does not try to create /glade on this machine.
        monkeypatch.setattr(build_mirror, "_STAGED_COVERAGE_PATH",
                            build_dir / "staged_coverage.json")
        monkeypatch.setattr(
            "jcm.data.mirror.emissions.load_biogenic_oc",
            _tier_a_biogenic(build_dir))

        build_mirror.stage_amip()

        out = xr.open_dataset(
            upload_dir / "bundles" / _TEST_GRID / "emissions_amip" / f"{year}.nc")
        assert "emis_biogenic_oc" in out.data_vars
        assert np.all(np.isfinite(out["emis_biogenic_oc"].values))
