# CAM6 SOAG offline test fixture

`anthro.nc`, `biogenic.nc` and `bb.nc` are **synthetic** stand-ins for the
three real NCAR CMIP6 CAM6 SOAG inventory files that
`jcm.data.emissions.cam6_soa.prepare_cam6_soa` normally downloads from
`svn-ccsm-inputdata.cgd.ucar.edu` (see that module's `HISTORICAL_SOURCES`).
They exist so the `jcm/data/mirror/` unit tests that exercise
`build_emissions_nc` -> `add_cam6_soa` -> `prepare_cam6_soa` can run offline,
without depending on that external host at test time (#1152 — PR #1150's
`fast-tests` job failed six of these tests the same day NCAR's SVN server
had a TLS outage).

**Not a subset of the real data.** Nothing here was downloaded, cropped or
derived from NCAR's files — it is generated from scratch by
`generate_fixture.py` in this directory. What is preserved is the
**contract** `prepare_cam6_soa` requires: variable names (`emiss_anthro`,
`emiss_biogenic`, `emiss_bb`), the `units: molecules/cm2/s` attribute, the
`(time, lat, lon)` dimension order, and a time axis whose calendar decodes
into the two decades `jcm.data.mirror.bundles.add_cam6_soa` ever requests —
1850–1859 ("pi") and 2005–2014 ("pd") — at twelve distinct calendar months
each. That is enough for `fetch()` (a local-path passthrough, see
`jcm/data/emissions/downloader.py`), the unit/sign validation, and the
conservative regrid to run through the exact same code the real files
exercise.

Grid: a regular 4×8 lat/lon grid (`generate_fixture.py` has the exact
values) — enough points for the conservative regridder to do real area
weighting, deliberately far coarser than NCAR's 0.9°×1.25° file.

**Size**: ~70 KB per file, ~210 KB total (well under the 1 MB the issue
asked for), dominated by netCDF/HDF5 header overhead on the 240-timestep
axis rather than the payload itself.

Pass these three paths as `prepare_cam6_soa`'s (or `add_cam6_soa`'s)
`sources=` dict, or `build_emissions_nc`'s `cam6_soa_sources=` argument, to
use them in a test — see `jcm/data/mirror/bundles_test.py` and
`jcm/data/mirror/emissions_test.py` for the pattern.

One test, `build_mirror_test.py::test_stage_bundles_writes_emis_biogenic_oc`,
deliberately keeps hitting the real download (`@pytest.mark.requires_network`)
so the extras/network CI job still verifies that live path end to end.

## Regenerate

Only needed if the fixture's grid or time coverage must change:

```bash
python jcm/data/test/cam6_soag_fixture/generate_fixture.py
```

See `provenance.json` for the generation parameters (seed, grid, years).
