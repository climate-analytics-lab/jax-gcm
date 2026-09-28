"""Unit tests for the mirror builders' pure-math pieces.

The builders themselves read multi-GB Glade sources; these tests cover
the statistics/registry machinery on synthetic inputs. The shared
regridding helpers are tested in ``jcm/data/regridding_test.py``, the
downloader in ``jcm/data/remote_test.py``.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from jcm.data.mirror.sso import finalize
from jcm.data.mirror.registry import build_registry, write_registry


class SsoFinalizeTest(unittest.TestCase):
    def _acc(self, **over):
        n = np.full(4, 10.0)
        acc = {"n": n, "land": n.copy(), "sh": np.zeros(4),
               "sh2": np.zeros(4), "shx2": np.zeros(4),
               "shy2": np.zeros(4), "shxy": np.zeros(4),
               "pic": np.zeros(4), "val": np.zeros(4)}
        acc.update(over)
        return acc

    def test_isotropic_slope_has_zero_anisotropy(self):
        # equal x/y gradient variance, no correlation -> gamma = 1 (round)
        acc = self._acc(shx2=np.full(4, 10.0), shy2=np.full(4, 10.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 1.0)
        np.testing.assert_allclose(out["orosig"], 1.0)  # sqrt(K+L=1)

    def test_pure_xslope_is_fully_anisotropic(self):
        # gradient variance only in x -> gamma = 0, theta = 0 deg
        acc = self._acc(shx2=np.full(4, 20.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 0.0)
        np.testing.assert_allclose(out["orothe"], 0.0)
        np.testing.assert_allclose(out["orosig"], np.sqrt(2.0))

    def test_ocean_cells_are_zeroed(self):
        acc = self._acc(land=np.array([10.0, 0.0, 10.0, 0.0]),
                        sh=np.full(4, 50.0), shx2=np.full(4, 4.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], [5.0, 0.0, 5.0, 0.0])
        np.testing.assert_allclose(out["lsm"], [1.0, 0.0, 1.0, 0.0])

    def test_variance_from_sufficient_statistics(self):
        h = np.array([1.0, 3.0, 5.0, 7.0])
        acc = self._acc(n=np.full(4, 2.0), land=np.full(4, 2.0),
                        sh=2 * h, sh2=2 * h ** 2 + 2.0)
        # per-cell: two samples h +/- 1 -> mean h, std 1
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], h)
        np.testing.assert_allclose(out["orostd"], 1.0)


class RegistryTest(unittest.TestCase):
    def test_registry_hashes_files(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sub").mkdir()
            (Path(d) / "sub" / "a.nc").write_bytes(b"hello")
            path = write_registry(d)
            reg = json.loads(Path(path).read_text())
            self.assertIn("sub/a.nc", reg["files"])
            self.assertEqual(reg["files"]["sub/a.nc"]["size"], 5)
            # registry.json itself is excluded
            reg2 = build_registry(d)
            self.assertNotIn("registry.json", reg2["files"])

    def test_paths_keep_the_base_entry_of_every_other_file(self):
        # A stale copy the build did not write must not override the
        # published entry.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            for name in ("forcing_pd.nc", "emissions_pd.nc"):
                (Path(d) / name).write_bytes(b"new")
            base = {"files": {"emissions_pd.nc": {"sha256": "published",
                                                  "size": 9}}}
            reg = build_registry(d, base=base, paths={"forcing_pd.nc"})
            self.assertEqual(reg["files"]["forcing_pd.nc"]["size"], 3)
            self.assertEqual(reg["files"]["emissions_pd.nc"]["sha256"],
                             "published")


class LedgerTest(unittest.TestCase):
    """Registry and upload act only on what this site's builds wrote."""

    F = "bundles/t63/forcing_pd.nc"

    def setUp(self):
        from pathlib import Path
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.t63 = self.root / "upload" / "bundles" / "t63"
        self.t63.mkdir(parents=True)
        (self.root / "build").mkdir()

    def _patched(self, **extra):
        from contextlib import ExitStack

        from jcm.data.mirror import build_mirror as bm
        stack = ExitStack()
        for name, value in {"BUILD": self.root / "build",
                            "UPLOAD": self.root / "upload", **extra}.items():
            stack.enter_context(patch.object(bm, name, value))
        return stack

    def _write(self, rel, data, ledger=True):
        from jcm.data.mirror import build_mirror as bm
        path = self.root / "upload" / rel
        path.write_bytes(data)
        if ledger:
            with self._patched():
                bm._write_ledger(bm._ledger() | {rel})
        return path

    def _registry(self, tip_files, dates=None, **extra):
        """Run stage_registry against a mocked mirror tip."""
        import json

        from jcm.data.mirror import build_mirror as bm
        dates = dates or {}
        with self._patched(_PRODUCTS=frozenset({"forcing"}), **extra), \
                patch.object(bm, "_mirror_tip", lambda: "tip"), \
                patch.object(bm, "_registry_at", lambda rev: json.loads(
                    json.dumps({"files": tip_files}))), \
                patch.object(bm, "_published_dates", lambda paths, rev: {
                    p: dates[p] for p in paths if p in dates}):
            bm.stage_registry()

    def _sha(self, data):
        import hashlib
        return hashlib.sha256(data).hexdigest()

    def test_a_stage_records_only_the_files_it_writes(self):
        from jcm.data.mirror import build_mirror as bm
        stale = self._write("bundles/t63/emissions_pd.nc", b"old", ledger=False)
        with self._patched():
            before = bm._upload_snapshot()
            (self.t63 / "forcing_pd.nc").write_bytes(b"new")
            after = bm._record_writes(before)
            self.assertEqual(bm._ledger(), {self.F})
            # A later stage adds to it; rewriting a file in place counts.
            stale.write_bytes(b"rebuilt")
            bm._record_writes(after)
            self.assertEqual(bm._ledger(),
                             {self.F, "bundles/t63/emissions_pd.nc"})

    def test_registry_merges_the_ledger_onto_the_tip_and_retires(self):
        import json

        from jcm.data.mirror import build_mirror as bm
        self._write(self.F, b"new")
        self._write("bundles/t63/emissions_pd.nc", b"stale", ledger=False)
        with self._patched():
            bm._write_ledger(bm._ledger() | {"bundles/t63/vanished.nc"})
        self._registry({"bundles/t63/emissions_pd.nc": {"sha256": "pub"},
                        self.F: {"sha256": "old"},
                        "bundles/t63/dust.nc": {"sha256": "gone"}},
                       _RETIRE=("bundles/*/dust.nc",))
        reg = json.loads((self.root / "upload" / "registry.json"
                          ).read_text())["files"]
        side = json.loads((self.root / "build" / "registry_base.json"
                           ).read_text())
        self.assertEqual(reg[self.F]["sha256"], self._sha(b"new"))
        self.assertEqual(reg["bundles/t63/emissions_pd.nc"]["sha256"], "pub")
        self.assertNotIn("bundles/t63/dust.nc", reg)
        self.assertEqual(side["parent_commit"], "tip")
        self.assertEqual(side["retired"], ["bundles/t63/dust.nc"])
        # A written file that has since gone is dropped, not published.
        self.assertEqual(list(side["written"]), [self.F])
        with self._patched():
            self.assertEqual(bm._ledger(), {self.F})

    def test_registry_refuses_only_a_newer_different_published_copy(self):
        import datetime
        path = self._write(self.F, b"mine")
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime,
                                                datetime.timezone.utc)
        later = {self.F: mtime + datetime.timedelta(hours=1)}
        earlier = {self.F: mtime - datetime.timedelta(hours=1)}
        # Another site published after our content was made: refused.
        with self.assertRaises(SystemExit) as ctx:
            self._registry({self.F: {"sha256": "theirs"}}, later)
        self.assertIn("older than a different published copy",
                      str(ctx.exception))
        # Ours is newer (rebuilt after their publish): allowed.
        self._registry({self.F: {"sha256": "theirs"}}, earlier)
        # Our own earlier publish matches our hash: allowed.
        self._registry({self.F: {"sha256": self._sha(b"mine")}}, later)

    def test_registry_refuses_retiring_a_file_it_wrote(self):
        self._write(self.F, b"new")
        with self.assertRaises(SystemExit) as ctx:
            self._registry({}, _RETIRE=("bundles/t63/forcing_*",))
        self.assertIn("narrow the globs", str(ctx.exception))

    def test_an_interrupted_registry_run_leaves_nothing_to_upload(self):
        from jcm.data.mirror import build_mirror as bm
        self._write(self.F, b"new")
        self._registry({})
        registry = (self.root / "upload" / "registry.json").read_text()
        with self._patched(), \
                patch.object(bm, "_mirror_tip", lambda: "tip"), \
                patch.object(bm, "_registry_at",
                             lambda rev: (_ for _ in ()).throw(OSError)), \
                self.assertRaises(OSError):
            bm.stage_registry()
        self.assertFalse((self.root / "build" / "registry_base.json").exists())
        self.assertEqual((self.root / "upload" / "registry.json").read_text(),
                         registry)
        with self.assertRaises(SystemExit) as ctx:
            self._upload(lambda **kw: None)
        self.assertIn("run --stage registry first", str(ctx.exception))

    def _upload(self, create_commit, tip=lambda: "tip", **extra):
        import contextlib
        import io
        from types import SimpleNamespace

        import huggingface_hub

        from jcm.data.mirror import build_mirror as bm
        api = SimpleNamespace(create_commit=create_commit)
        out = io.StringIO()
        with self._patched(**extra), \
                patch.object(huggingface_hub, "HfApi", lambda: api), \
                patch.object(bm, "_mirror_tip", tip), \
                patch("time.sleep", lambda s: None), \
                contextlib.redirect_stdout(out):
            bm.stage_upload()
        return out.getvalue()

    def test_upload_commits_exactly_what_the_registry_hashed(self):
        from types import SimpleNamespace

        import huggingface_hub

        from jcm.data.mirror import build_mirror as bm
        self._write("bundles/t63/emissions_pd.nc", b"stale", ledger=False)
        path = self._write(self.F, b"new")
        self._registry({"bundles/t63/dust.nc": {"sha256": "x"}},
                       _RETIRE=("bundles/*/dust.nc",))
        commits = []

        def create_commit(**kw):
            commits.append(kw)
            return SimpleNamespace(oid="f" * 40)

        # --retire belongs to the registry run; a different one is refused.
        with self.assertRaises(SystemExit) as ctx:
            self._upload(create_commit, _RETIRE=("other",))
        self.assertIn("applied by --stage registry", str(ctx.exception))
        # A file touched after the registry hashed it is refused.
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
        with self.assertRaises(SystemExit) as ctx:
            self._upload(create_commit)
        self.assertIn("changed since --stage registry", str(ctx.exception))
        self.assertEqual(commits, [])
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        out = self._upload(create_commit)
        (kw,) = commits
        self.assertEqual(kw["parent_commit"], "tip")
        added = sorted(op.path_in_repo for op in kw["operations"]
                       if isinstance(op, huggingface_hub.CommitOperationAdd))
        deleted = [op.path_in_repo for op in kw["operations"]
                   if isinstance(op, huggingface_hub.CommitOperationDelete)]
        self.assertEqual(added, [self.F, "registry.json"])
        self.assertEqual(deleted, ["bundles/t63/dust.nc"])
        self.assertIn('MIRROR_REVISION = "' + "f" * 40 + '"', out)
        with self._patched():
            self.assertEqual(bm._ledger(), set())
        self.assertTrue((self.root / "build" / f"upload_ledger.{'f' * 12}.json"
                         ).exists())

    def test_upload_stops_once_the_tip_has_moved_and_retries_otherwise(self):
        from types import SimpleNamespace

        from jcm.data.mirror import build_mirror as bm
        self._write(self.F, b"new")
        self._registry({})
        attempts = []

        def failing(**kw):
            attempts.append(kw)
            raise RuntimeError("412 parent commit mismatch")

        with self.assertRaises(SystemExit) as ctx:
            self._upload(failing, tip=lambda: "moved")
        self.assertIn("mirror tip is now moved", str(ctx.exception))
        self.assertEqual(len(attempts), 1)
        with self._patched():
            self.assertEqual(bm._ledger(), {self.F})

        # A failure that also breaks the tip query is retried, not fatal.
        def flaky(**kw):
            attempts.append(kw)
            if len(attempts) < 3:
                raise TimeoutError("xet")
            return SimpleNamespace(oid="e" * 40)

        def no_network():
            raise ConnectionError("down")

        self._upload(flaky, tip=no_network)
        self.assertEqual(len(attempts), 3)

    def test_main_records_each_stage_and_nothing_from_a_failed_one(self):
        import contextlib
        import io

        from jcm.data.mirror import build_mirror as bm

        def good():
            (self.t63 / "forcing_pd.nc").write_bytes(b"new")

        def bad():
            (self.t63 / "forcing_amip.nc").write_bytes(b"trunc")
            raise RuntimeError("walltime")

        stages = {**bm.STAGES, "bundles": good, "amip": bad}
        out = io.StringIO()
        with self._patched(STAGES=stages), \
                patch.object(bm, "check_sources", lambda *a, **k: None), \
                patch("sys.argv", ["build_mirror", "--stage", "bundles,amip"]), \
                contextlib.redirect_stdout(out), \
                self.assertRaises(RuntimeError):
            bm.main()
        with self._patched():
            self.assertEqual(bm._ledger(), {self.F})
        self.assertIn("stage amip failed: nothing it wrote is recorded",
                      out.getvalue())


class MirrorRevisionStagesTest(unittest.TestCase):
    def test_pull_is_pinned(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest import mock

        import huggingface_hub

        from jcm.data import remote
        from jcm.data.mirror import build_mirror as bm

        pulled = []

        def snapshot(**kw):
            pulled.append(kw["revision"])
            root = Path(kw["local_dir"])
            for name in ("era5_land_climo_2005-2014_0p25.nc",
                         "ceds_anthro.zarr", "bb4cmip7.zarr"):
                (root / "products" / name).mkdir(parents=True)

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(bm, "BUILD", Path(d)), \
                mock.patch.object(huggingface_hub, "snapshot_download",
                                  snapshot), \
                contextlib.redirect_stdout(io.StringIO()):
            bm.stage_pull()
        self.assertEqual(pulled, [remote.mirror_revision()])


class DustProductTest(unittest.TestCase):
    """The dust bundle builder on a synthetic stand-in for the HAMMOZ pool.

    The real inputs are in the ECHAM-HAMMOZ pool (``HammozPoolTest`` checks
    them where it is mounted); what needs covering everywhere is the pool
    layout lookup, the orientation flip, the conservative remap for a grid
    HAMMOZ does not ship, the regenerated region mask and the build-time guards.
    """

    NLAT, NLON = 96, 192
    T63 = "v0007/hammoz/T63"

    def _sources(self, tmp, soils=None, regions=None):
        """Write a fake T63-only pool under ``tmp``; return the model lat/lon."""
        import xarray as xr

        from jcm.data.mirror.dust import region_mask
        from jcm.data.regridding import gaussian_latlon
        lats, lons = gaussian_latlon(self.NLAT)
        # The HAMMOZ files store latitude DESCENDING; the builder flips it.
        desc = lats[::-1]
        shape = (1, self.NLAT, self.NLON)
        month = (12, self.NLAT, self.NLON)
        # pot_source varies with latitude so the flip is observable.
        pot = np.broadcast_to(np.abs(desc)[None, :, None], month).copy()
        if regions is None:
            regions = region_mask(desc, lons)[None]
        if soils is None:
            soils = {f"type{i}": np.full(shape, 0.2 if i in (2, 3, 4, 6) else 0.0)
                     for i in (2, 3, 4, 6, 13, 14, 15, 16, 17)}
        # Roughness is NaN away from dust sources, as in the HAMMOZ map.
        rough = np.full(month, np.nan)
        rough[:, 30:60, 20:80] = 0.02
        months = np.array([np.datetime64(f"2000-{m:02d}-01") for m in range(1, 13)])
        coords = {"lat": desc, "lon": lons}
        files = {
            "dust_potential_sources_T63.nc": xr.Dataset(
                {"pot_source": (("time", "lat", "lon"), pot)},
                coords={**coords, "time": months}),
            "dust_preferential_sources_T63.nc": xr.Dataset(
                {"source": (("time", "lat", "lon"), np.full(shape, 0.3))},
                coords={**coords, "time": months[:1]}),
            "soil_type_all_T63.nc": xr.Dataset(
                {k: (("time", "lat", "lon"), v) for k, v in soils.items()},
                coords={**coords, "time": months[:1]}),
            "dust_regions_T63.nc": xr.Dataset(
                {"regions": (("time", "lat", "lon"), regions.astype(float))},
                coords={**coords, "time": months[:1]}),
            "surface_rough_12m_T63.nc": xr.Dataset(
                {"surfrough": (("time", "lat", "lon"), rough)},
                coords={**coords, "time": months}),
        }
        os.makedirs(os.path.join(tmp, self.T63))
        for name, ds in files.items():
            ds.to_netcdf(os.path.join(tmp, self.T63, name))
        return lats, lons

    def _build(self, tmp, name, nlat):
        from jcm.data.mirror.dust import build_dust_product
        out = os.path.join(tmp, f"{name}_{nlat}.nc")
        with patch.dict("jcm.data.mirror.dust.NATIVE_SOURCES",
                        {63: _t63_only()}, clear=True):
            build_dust_product(name, nlat, out, source_dir=tmp)
        return out

    def test_native_grid_is_copied_with_ascending_latitude(self):
        import xarray as xr
        with tempfile.TemporaryDirectory() as tmp:
            lats, _ = self._sources(tmp)
            out = self._build(tmp, "dust_potential_sources", self.NLAT)
            with xr.open_dataset(out, decode_times=False) as ds:
                np.testing.assert_allclose(ds.lat.values, lats)
                np.testing.assert_allclose(
                    ds.pot_source.values[0, :, 0], np.abs(lats), atol=1e-9)
                self.assertNotIn("regrid_approximation", ds.attrs)
                self.assertIn(self.T63, ds.attrs["source"])

    def test_a_coarser_grid_is_remapped_conservatively(self):
        # A grid HAMMOZ does not ship is coarsened from the finest native file
        # with the exact-overlap remap: the area mean is conserved and nothing
        # is flagged as an approximation.
        import xarray as xr
        with tempfile.TemporaryDirectory() as tmp:
            lats, _ = self._sources(tmp)
            src = self._build(tmp, "dust_potential_sources", self.NLAT)
            out = self._build(tmp, "dust_potential_sources", 48)
            weights = lambda n: np.polynomial.legendre.leggauss(n)[1]  # noqa: E731
            with xr.open_dataset(src) as a, xr.open_dataset(out) as b:
                self.assertEqual(b.pot_source.shape, (12, 48, 96))
                mean_a = (a.pot_source.values * weights(96)[:, None]).sum() / 192
                mean_b = (b.pot_source.values * weights(48)[:, None]).sum() / 96
                self.assertAlmostEqual(mean_a, mean_b, places=10)
                self.assertIn("conservative", b.attrs["history"])
                self.assertNotIn("regrid_approximation", b.attrs)

    def test_a_finer_grid_is_flagged_as_an_approximation(self):
        import xarray as xr
        with tempfile.TemporaryDirectory() as tmp:
            self._sources(tmp)
            out = self._build(tmp, "dust_surface_roughness", 160)
            with xr.open_dataset(out) as ds:
                self.assertIn("regrid_approximation", ds.attrs)
                values = ds.surfrough.values
                # NaN stays "no source", and the valid patch keeps its value
                # rather than being diluted toward zero at its edges.
                self.assertTrue(np.isnan(values).any())
                np.testing.assert_allclose(values[np.isfinite(values)], 0.02)

    def test_the_region_mask_is_regenerated_and_categorical(self):
        import xarray as xr
        with tempfile.TemporaryDirectory() as tmp:
            self._sources(tmp)
            out = self._build(tmp, "dust_regions", 32)
            with xr.open_dataset(out) as ds:
                values = ds.regions.values
                self.assertEqual(values.shape, (32, 64))
                self.assertEqual(set(np.unique(values)), set(range(1, 9)))
                self.assertIn(self.T63, ds.attrs["history"])

    def test_a_native_mask_the_recipe_does_not_reproduce_is_refused(self):
        from jcm.data.mirror.dust import region_mask
        from jcm.data.regridding import gaussian_latlon
        lats, lons = gaussian_latlon(self.NLAT)
        regions = region_mask(lats[::-1], lons)
        regions[10, 10] = 5.0 if regions[10, 10] != 5.0 else 6.0
        with tempfile.TemporaryDirectory() as tmp:
            self._sources(tmp, regions=regions[None])
            with self.assertRaisesRegex(ValueError, "no longer reproduces"):
                self._build(tmp, "dust_regions", self.NLAT)

    def test_a_built_product_round_trips_through_its_reader(self):
        # The build and the read are the two halves of one contract; nothing
        # else checks that a published file is loadable, decodable and on the
        # model grid.
        import jax.numpy as jnp
        import xarray as xr

        from jcm.forcing import (WRAP_YEAR, read_dust_preferential,
                                 read_dust_regions, read_dust_roughness,
                                 read_dust_soil_types, read_dust_source)
        readers = {"dust_potential_sources": read_dust_source,
                   "dust_preferential_sources": read_dust_preferential,
                   "dust_soil_types": read_dust_soil_types,
                   "dust_regions": read_dust_regions,
                   "dust_surface_roughness": read_dust_roughness}
        with tempfile.TemporaryDirectory() as tmp:
            lats, lons = self._sources(tmp)
            for name, reader in readers.items():
                out = self._build(tmp, name, self.NLAT)
                # Decoded (not decode_times=False): a monthly product must carry
                # real CF time units, not bare numbers read as nanoseconds.
                with xr.open_dataset(out) as ds:
                    if "time" in ds.dims and ds.sizes["time"] > 1:
                        self.assertEqual(
                            str(ds.time.values[1])[:10], "2000-02-01",
                            f"{name}: time axis lost its units")
                    leaf = reader(ds, lat_deg=lats, lon_deg=lons)
                values = (next(iter(leaf.values())) if isinstance(leaf, dict)
                          else getattr(leaf, "values", leaf))
                self.assertEqual(jnp.asarray(values).shape[-2:],
                                 (self.NLON, self.NLAT), name)
                if name in ("dust_potential_sources", "dust_surface_roughness"):
                    self.assertEqual(int(leaf.align_mode), WRAP_YEAR, name)

    def test_a_broken_global_partition_is_refused(self):
        soils = {f"type{i}": np.full((1, self.NLAT, self.NLON), 0.3)
                 for i in (2, 3, 4, 6, 13, 14, 15, 16, 17)}
        with tempfile.TemporaryDirectory() as tmp:
            self._sources(tmp, soils=soils)
            with self.assertRaisesRegex(ValueError, "would go negative"):
                self._build(tmp, "dust_soil_types", self.NLAT)


def _t63_only():
    from jcm.data.mirror.dust import NATIVE_SOURCES
    return NATIVE_SOURCES[63]


@unittest.skipUnless(os.path.isdir("/pool/data/ECHAM6-HAMMOZ"),
                     "needs the ECHAM-HAMMOZ input pool (DKRZ Levante)")
class HammozPoolTest(unittest.TestCase):
    """The provenance claims in ``jcm.data.mirror.dust``, checked on the pool."""

    ROOT = "/pool/data/ECHAM6-HAMMOZ"

    def _field(self, rel, var):
        import xarray as xr
        with xr.open_dataset(os.path.join(self.ROOT, rel),
                             decode_times=False) as ds:
            return np.squeeze(ds[var].values), ds.lat.values, ds.lon.values

    def test_the_region_recipe_reproduces_every_native_mask(self):
        from jcm.data.mirror.dust import NATIVE_SOURCES, region_mask
        for trunc, table in NATIVE_SOURCES.items():
            if "dust_regions" not in table:
                continue
            ref, lats, lons = self._field(*table["dust_regions"]["regions"])
            np.testing.assert_array_equal(region_mask(lats, lons), ref,
                                          err_msg=f"T{trunc}")

    def test_the_old_lineage_names_the_same_products(self):
        # Where both lineages exist (T63, T127) the v01_001 files ARE the
        # current products — the basis for taking T255 from v01_001.
        from jcm.data.mirror.dust import NATIVE_SOURCES
        for trunc in (63, 127):
            t = f"T{trunc}"
            new = NATIVE_SOURCES[trunc]
            old = {"pot_source": (f"v01_001/hammoz/{t}/ndvi_lai_eff.12m.{t}.nc",
                                  "laieff"),
                   "source": (f"v01_001/hammoz/{t}/pot_sources.{t}.nc",
                              "source"),
                   "type2": (f"v01_001/hammoz/{t}/soil_type2.{t}.nc", "type")}
            pairs = {"pot_source": new["dust_potential_sources"]["pot_source"],
                     "source": new["dust_preferential_sources"]["source"],
                     "type2": new["dust_soil_types"]["type2"]}
            for key, cur in pairs.items():
                a = self._field(*cur)[0]
                b = self._field(*old[key])[0]
                np.testing.assert_allclose(a, b, atol=1e-6,
                                           err_msg=f"{t} {key}")


class LandChannelCoverageTest(unittest.TestCase):
    """Every surface-forcing writer must carry each ``translate_land`` channel.

    ``translate_land`` is the single land translation, but three builders
    serialize its output independently — the climatological bundles, the
    transient ERA5 years and the transient AMIP years — and a builder that
    silently omits a channel produces files that load with that channel
    ``None``. For ``soilw_rel`` that means the dust saturation cut-off runs
    inert with no other symptom (#787), which is exactly the failure this
    guards: the builders read multi-GB Glade sources, so no unit test runs
    them end to end.
    """

    _WRITERS = ("bundles", "era5_yearly", "amip_yearly")

    def test_no_writer_drops_a_channel(self):
        import ast
        import inspect
        import importlib

        translated = set(_translate_land_keys())
        self.assertIn("soilw_rel", translated)
        for name in self._WRITERS:
            module = importlib.import_module(f"jcm.data.mirror.{name}")
            # Dict KEYS only: every one of these builders assembles the
            # variables it writes as a dict literal, and a channel merely
            # mentioned in a comment or read back out of ``land[...]`` is
            # exactly the case that passes review and writes nothing.
            written = {key.value for node in
                       ast.walk(ast.parse(inspect.getsource(module)))
                       if isinstance(node, ast.Dict)
                       for key in node.keys
                       if isinstance(key, ast.Constant)
                       and isinstance(key.value, str)}
            missing = sorted(translated - written)
            self.assertEqual(missing, [], f"{name}.py never names {missing}")

    def test_every_writer_regrids_land_through_the_convention_helper(self):
        """Every writer splices :func:`land_surface_fields` into its dict.

        It is the one place the land-conditional channels (and ``glac``,
        ``forest``, ``lsm``) are regridded with their masks (#672); a writer
        that interpolated them itself would dilute them at coasts again.
        """
        import ast
        import inspect
        import importlib

        for name in self._WRITERS:
            module = importlib.import_module(f"jcm.data.mirror.{name}")
            spliced = any(
                key is None and isinstance(value, ast.Call)
                and getattr(value.func, "id", None) == "land_surface_fields"
                for node in ast.walk(ast.parse(inspect.getsource(module)))
                if isinstance(node, ast.Dict)
                for key, value in zip(node.keys, node.values))
            self.assertTrue(spliced, f"{name}.py bypasses land_surface_fields")


def _translate_land_keys():
    """Return the channel names ``translate_land`` produces on a minimal input."""
    import xarray as xr

    from jcm.data.mirror.bundles import translate_land

    base = xr.DataArray(np.ones((1, 2)), dims=("time", "cell"))
    cell = lambda v: xr.DataArray(np.full(2, v), dims=("cell",))  # noqa: E731
    era5 = xr.Dataset({"sd": base * 0.0, "stl1": base * 280.0,
                       "swvl1": base * 0.2, "swvl2": base * 0.2,
                       "cvh": cell(0.0), "cvl": cell(0.0), "slt": cell(2.0)})
    return translate_land(
        era5, permanent_snow=xr.DataArray(np.zeros(2, dtype=bool),
                                          dims=("cell",))).keys()


class RegistryMergeTest(unittest.TestCase):
    """A ``--grids`` build must not drop the rest of the mirror's registry."""

    def test_partial_tree_merges_onto_the_published_registry(self):
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            Path(d, "bundles/t127").mkdir(parents=True)
            Path(d, "bundles/t127/terrain.nc").write_bytes(b"new")
            base = {"files": {"bundles/t63/terrain.nc": {"sha256": "x", "size": 1},
                              "bundles/t127/terrain.nc": {"sha256": "old",
                                                          "size": 9}}}
            reg = json.loads(Path(write_registry(d, base=base)).read_text())
            self.assertIn("bundles/t63/terrain.nc", reg["files"])
            self.assertEqual(reg["files"]["bundles/t127/terrain.nc"]["size"], 3)
            self.assertNotEqual(
                reg["files"]["bundles/t127/terrain.nc"]["sha256"], "old")


class SitesTest(unittest.TestCase):
    def test_explicit_site_and_overrides(self):
        from jcm.data.mirror import sites
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "levante",
                                     "JCM_HAMMOZ_DIR": "/copy/of/pool"}):
            site = sites.current()
            self.assertEqual(site.name, "levante")
            self.assertIsNone(site.rda)
            self.assertEqual(site.hammoz, "/copy/of/pool")
            self.assertTrue(sites.input4mips("CMIP7/x").startswith(
                "/pool/data/INPUT4MIP/data/input4MIPs/"))
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "glade"}):
            self.assertIsNotNone(sites.current().rda)
        # An inputdata override carries the WACCM oxidant root with it; the
        # dedicated oxidant override wins over both.
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "levante",
                                     "JCM_CESM_INPUTDATA": "/my/inputdata"}):
            site = sites.current()
            self.assertEqual(site.cesm_inputdata, "/my/inputdata")
            self.assertEqual(site.waccm_oxidants, "/my/inputdata/atm/cam/ozone")
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "glade",
                                     "JCM_CESM_INPUTDATA": "/my/inputdata",
                                     "JCM_WACCM_OXIDANTS_DIR": "/cseg/ozone"}):
            self.assertEqual(sites.current().waccm_oxidants, "/cseg/ozone")
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "nowhere"}):
            with self.assertRaisesRegex(ValueError, "known sites"):
                sites.current()

    def test_a_source_the_site_lacks_is_refused_up_front(self):
        from jcm.data.mirror import build_mirror, sites
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "levante"}):
            levante = sites.current()
        with patch.object(build_mirror, "SITE", levante):
            self.assertEqual(build_mirror._unavailable(["era5", "ozone"]),
                             {"era5": ["RDA ERA5 monthly means"]})
            with self.assertRaises(SystemExit) as ctx:
                build_mirror.check_sources(["era5"])
            self.assertIn("--stage pull", str(ctx.exception))
            build_mirror.check_sources(["manifest", "pull"])   # source-free
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "glade"}):
            glade = sites.current()
        with patch.object(build_mirror, "SITE", glade):
            # Glade lacks the HAMMOZ pool: the dust stage names it and says
            # where it can be built.
            with self.assertRaises(SystemExit) as ctx:
                build_mirror.check_sources(["dust"])
            self.assertIn("ECHAM-HAMMOZ pool", str(ctx.exception))
            self.assertIn("JCM_HAMMOZ_DIR", str(ctx.exception))

    def test_build_tree_inputs_are_checked_only_before_their_stage(self):
        # A one-shot pull,...,bundles on a fresh root must not be refused up
        # front for build/ outputs an earlier stage in the same run produces.
        from jcm.data.mirror import build_mirror
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            root = Path(d)
            i4m = root / "i4m"
            (i4m / "CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10").mkdir(parents=True)
            site = build_mirror.SITE.__class__(**{
                **build_mirror.SITE.__dict__, "input4mips": str(i4m)})
            with patch.object(build_mirror, "SITE", site), \
                    patch.object(build_mirror, "ROOT", root), \
                    patch.object(build_mirror, "BUILD", root / "build"), \
                    patch.object(build_mirror, "UPLOAD", root / "upload"):
                build_mirror.check_sources(["bundles"])
                with self.assertRaises(SystemExit) as ctx:
                    build_mirror.check_sources(["bundles"], include_build=True)
                self.assertIn("Tier A ERA5 land climatology", str(ctx.exception))

    def test_a_partial_transient_build_is_refused(self):
        from jcm.data.mirror import build_mirror as bm
        argv = ["build_mirror", "--grids", "t63", "--stage", "amip"]
        with patch("sys.argv", argv), self.assertRaises(SystemExit) as ctx:
            bm.main()
        self.assertIn("every transient grid", str(ctx.exception))

class GridSelectionTest(unittest.TestCase):
    def test_grids_filter_and_transient_scope(self):
        from jcm.data.mirror import build_mirror as bm
        with patch.object(bm, "_SELECTED", None):
            self.assertEqual(set(bm._grids()), {"t63", "t106", "t127", "t255"})
            self.assertEqual(set(bm._grids(transient=True)), {"t63", "t106"})
            self.assertTrue(bm._column_selected())
        with patch.object(bm, "_SELECTED", frozenset({"t127", "t255"})):
            self.assertEqual(bm._grids(), {"t127": 192, "t255": 384})
            self.assertEqual(bm._grids(transient=True), {})
            self.assertFalse(bm._column_selected())
        self.assertEqual(bm._truncation("t255"), 255)

    def test_pulled_tier_a_marks_the_build_partial(self):
        from jcm.data.mirror import build_mirror as bm
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            build = Path(d) / "build"
            (build / "pulled" / "products" / "ceds_anthro.zarr").mkdir(
                parents=True)
            (build / "ceds_anthro.zarr").symlink_to(
                build / "pulled" / "products" / "ceds_anthro.zarr")
            with patch.object(bm, "BUILD", build), \
                    patch.object(bm, "_SELECTED", None), \
                    patch.object(bm, "_PRODUCTS", None):
                self.assertTrue(bm._pulled_emissions())
                self.assertTrue(bm._partial_build())

    def test_a_missing_ne30_topography_skips_only_ne30(self):
        from jcm.data.mirror import build_mirror as bm, sites
        with patch.dict(os.environ, {"JCM_MIRROR_SITE": "levante"}):
            levante = sites.current()
        with patch.object(bm, "SITE", levante), \
                patch.object(bm, "NE30_TOPO", None):
            with patch.object(bm, "_SELECTED", None):
                self.assertNotIn("sso", bm._unavailable(["sso"]))
                self.assertFalse(bm._column_buildable())
            with patch.object(bm, "_SELECTED", frozenset({"ne30pg3"})):
                self.assertEqual(bm._unavailable(["sso"]),
                                 {"sso": ["CESM ne30 topography"]})

    def test_products_filter_narrows_what_bundles_reads(self):
        from jcm.data.mirror import build_mirror as bm
        with patch.object(bm, "_PRODUCTS", frozenset({"emissions"})):
            self.assertTrue(bm._want("emissions"))
            self.assertFalse(bm._want("terrain"))
            labels = {label for label, _ in bm._stage_sources()["bundles"]}
            self.assertEqual(labels, {"Tier A CEDS store",
                                      "Tier A BB4CMIP7 store"})
        with patch.object(bm, "_PRODUCTS", None):
            self.assertTrue(all(bm._want(p) for p in bm.BUNDLE_PRODUCTS))


if __name__ == "__main__":
    unittest.main()
