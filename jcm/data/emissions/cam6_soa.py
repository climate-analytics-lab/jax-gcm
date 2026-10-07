"""Prepare the published CAM6 one-bin SOA precursor inventory.

CAM6 prescribes SOAG from fixed VOC yields, partitions it reversibly with
SOA, and removes the aerosol through wet and dry deposition. Its SOAG has
no gas deposition or photolysis. This is the CAM6 formulation (Liu et al.
2012; Jo et al. 2023 section 2.2), not the distinct CAM6.3 SOAE scheme.

The official inventories already contain the VOC yields and CAM's 1.5
factor. Applying either again would inflate the source. Their fluxes are
carbon-equivalent molecules, converted with CAM's tracer MW of 12.011,
not the physical molecular weight used for gas–aerosol exchange.

Usage::

    python -m jcm.data.emissions.cam6_soa --truncation 63 --output soag.nc

Add the result to ``forcing.emissions_file=[hf://bundles/t63/emissions_pd.nc,/absolute/path/soag.nc]``
with ``forcing.emissions_align=[auto,wrap_year]``. With ``--year``, select
that year from the official 1750–2015 historical inventories. Without it,
use the 1995–2005 climatology. Do not substitute either for a different era.
"""

from __future__ import annotations

from jcm.data.emissions.prepare import (
    SpeciatedChannel, prepare_speciated_emissions,
)

INPUTDATA = (
    "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/cam/chem/emis/"
    "CMIP6_emissions_2000climo/"
)
SOURCES = {
    "anthro": "emissions-cmip6_SOAGx1.5_anthro_surface_2000climo_0.9x1.25_c20170608.nc",
    "biogenic": "emissions-cmip6_SOAGx1.5_biogenic_surface_2000climo_0.9x1.25_c20170322.nc",
    "bb": "emissions-cmip6_SOAGx1.5_bb_surface_2000climo_0.9x1.25_c20170322.nc",
}
SOURCE_SHA256 = {
    "anthro": "d2836e80b7f079c6891d140c29cebd7e7924cc8f4d0a0c7835f21673b90e3968",
    "biogenic": "ae60c6b42172492c0199c2a86813ecbdd19dab4b798126f5218aff0e2b2b14c0",
    "bb": "541f4df48cac18b7fd39075b07e5650ba18b7bb7493a61d4b4c17741fa54e008",
}


HISTORICAL_INPUTDATA = INPUTDATA.replace("CMIP6_emissions_2000climo/", "CMIP6_emissions_1750_2015/")
HISTORICAL_SOURCES = {k: v.replace("2000climo", "1750-2015") for k, v in SOURCES.items()}
HISTORICAL_SHA256 = {
    "anthro": "ce458da7050202ce804edc1102156391c832e4db35faccd47bbdf09b93d3b51e",
    "biogenic": "fdb012bc697b3aa64f934ec87755771213a7e5e726d577c9a406deea330c470c",
    "bb": "832b71f62675ebe2cb25fd7bd5925252977f4c8042fd98f43ca86f22bbe3d242",
}


def prepare_cam6_soa(coords, sources=None, *, year=None, year_range=None):
    """Conservatively remap and sum the three CAM6 surface SOAG sources.

    ``sources`` optionally maps anthro/biogenic/bb to local paths or URLs.
    All three categories are required so an incomplete inventory fails.
    ``year`` selects one complete monthly year; ``year_range`` averages matching
    months across its inclusive range. Both require twelve distinct months in
    every source year, with source hashes
    recorded for the original historical files, not an untraceable subset.
    """
    import hashlib
    from pathlib import Path
    import numpy as np
    import xarray as xr
    from jcm.data.emissions.downloader import fetch

    if year is not None and year_range is not None:
        raise ValueError("Specify year or year_range, not both")
    if year_range is not None and year_range[0] > year_range[1]:
        raise ValueError("SOAG year_range must be increasing")
    historical = year is not None or year_range is not None
    official = sources is None
    if official:
        base, catalog, known = ((INPUTDATA, SOURCES, SOURCE_SHA256) if not historical
                                else (HISTORICAL_INPUTDATA, HISTORICAL_SOURCES, HISTORICAL_SHA256))
        sources = {k: base + v for k, v in catalog.items()}
    if set(sources) != set(SOURCES):
        raise ValueError("CAM6 SOAG requires anthro, biogenic and bb inventories")
    local = {}
    hashes = {}
    time = None
    time_index = None
    for key in SOURCES:
        local[key] = fetch(sources[key], known_hash=(
            "sha256:" + known[key] if official else None))
        hashes[key] = hashlib.sha256(Path(local[key]).read_bytes()).hexdigest()
        with xr.open_dataset(local[key], decode_times=False) as source:
            field = source["emiss_" + key]
            if field.attrs.get("units") != "molecules/cm2/s":
                raise ValueError("CAM6 SOAG inventories require molecules/cm2/s")
            if time is not None and not time.identical(source.time):
                raise ValueError("CAM6 SOAG source calendars/time axes must agree")
            time = source.time.copy(deep=True)
            if historical:
                decoded = xr.decode_cf(source[["time"]]).time
                first, last = year_range if year_range is not None else (year, year)
                source_years = decoded.dt.year.values
                time_index = np.flatnonzero((source_years >= first) & (source_years <= last))
                for selected_year in range(first, last + 1):
                    months = decoded.dt.month.values[source_years == selected_year]
                    if len(months) != 12 or set(months) != set(range(1, 13)):
                        raise ValueError(
                            f"CAM6 SOAG requires twelve distinct monthly fields for {selected_year}")
                field = field.isel(time=time_index)
            values = field.values
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("CAM6 SOAG fluxes must be finite and nonnegative")
    channels = tuple(
        SpeciatedChannel("g_soag", local[k], var="emiss_" + k,
                         molar_mass=12.011)
        for k in SOURCES
    )
    ds = prepare_speciated_emissions(channels, coords, time_index=time_index)
    if year_range is not None:
        # Average matching months, never all 120 fields into one annual rate.
        ds = xr.decode_cf(ds).groupby("time.month").mean("time", keep_attrs=True)
        ds = ds.rename(month="time").assign_coords(
            time=np.array([np.datetime64(f"{year_range[0]}-{m:02d}-01")
                           for m in range(1, 13)]))
    period = (f"{year_range[0]}–{year_range[1]} climatology" if year_range is not None
              else "1995–2005 climatology" if year is None else str(year))
    ds.attrs.update(
        title=f"CAM6 prescribed SOAG: {period}",
        inventory_year=period,
        source="; ".join(str(sources[k]) for k in SOURCES),
        source_sha256="; ".join(f"{k}:{hashes[k]}" for k in SOURCES),
        reference="https://doi.org/10.5194/gmd-16-3893-2023 (section 2.2)",
        scheme="CAM6 one-bin SOA; published yields and 1.5 factor already applied",
    )
    return ds


def main():
    """Write a model-grid inventory through the shared preparation pipeline."""
    import argparse
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truncation", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--year", type=int, help="Monthly year from CAM6 historical source")
    args = parser.parse_args()
    coords = get_speedy_coords(layers=8, spectral_truncation=args.truncation)
    ds = prepare_cam6_soa(coords, year=args.year)
    ds.to_netcdf(args.output)


if __name__ == "__main__":
    main()
