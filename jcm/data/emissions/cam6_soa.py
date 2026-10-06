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
with ``forcing.emissions_align=[auto,wrap_year]``. This inventory averages
1995–2005 and must not be silently substituted for PI/transient emissions.
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


def prepare_cam6_soa(coords, sources=None):
    """Conservatively remap and sum the three CAM6 surface SOAG sources.

    ``sources`` optionally maps anthro/biogenic/bb to local paths or URLs.
    All three categories are required so an incomplete inventory fails.
    """
    import hashlib
    from pathlib import Path
    import numpy as np
    import xarray as xr
    from jcm.data.emissions.downloader import fetch

    official = sources is None
    if official:
        sources = {k: INPUTDATA + v for k, v in SOURCES.items()}
    if set(sources) != set(SOURCES):
        raise ValueError("CAM6 SOAG requires anthro, biogenic and bb inventories")
    local = {}
    hashes = {}
    time = None
    for key in SOURCES:
        local[key] = fetch(sources[key], known_hash=(
            "sha256:" + SOURCE_SHA256[key] if official else None))
        hashes[key] = hashlib.sha256(Path(local[key]).read_bytes()).hexdigest()
        with xr.open_dataset(local[key], decode_times=False) as source:
            field = source["emiss_" + key]
            if field.attrs.get("units") != "molecules/cm2/s":
                raise ValueError("CAM6 SOAG inventories require molecules/cm2/s")
            if not np.isfinite(field.values).all() or (field.values < 0).any():
                raise ValueError("CAM6 SOAG fluxes must be finite and nonnegative")
            if time is not None and not time.identical(source.time):
                raise ValueError("CAM6 SOAG source calendars/time axes must agree")
            time = source.time.copy(deep=True)
    channels = tuple(
        SpeciatedChannel("g_soag", local[k], var="emiss_" + k,
                         molar_mass=12.011)
        for k in SOURCES
    )
    ds = prepare_speciated_emissions(channels, coords)
    ds.attrs.update(
        title="CAM6 prescribed SOAG: 1995–2005 climatology",
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
    args = parser.parse_args()
    coords = get_speedy_coords(layers=8, spectral_truncation=args.truncation)
    ds = prepare_cam6_soa(coords)
    ds.to_netcdf(args.output)


if __name__ == "__main__":
    main()
