# ARM Data Product, Download, And Access Guide

## Purpose

This guide is a practical entry point for researchers who want to use
Atmospheric Radiation Measurement (ARM) observations, including aerosol, cloud,
radiation, atmospheric-state, surface, and precipitation products. It explains:

- which ARM product families to consider;
- how to discover exact datastream names and coverage;
- how to download files through ARM Live Data;
- how to inspect NetCDF files and apply QC safely; and
- what provenance and citation information to retain.

ARM Data Discovery is the authoritative source for current availability,
documentation, and product-level citation:
<https://adc.arm.gov/discovery/>.

## Product Selection Principles

An ARM datastream name encodes site, instrument/product, facility, and data
level. For example:

```text
sgpaoscpcfE13.b1
```

means an SGP AOS condensation-particle-counter stream at facility E13, data
level b1. Exact facilities and levels change with instrument deployment and
date. Do not construct names from memory; query the catalog.

Data levels such as `a1`, `b1`, `b2`, and `c1` identify processing stages, but
they are not a universal ranking where one level is always scientifically
better. Instrument streams and value-added products answer different questions.

Preserve site and facility. At SGP, aerosol instruments are often at E13 while
ARMBE products are at C1. They are not the same point, and collocation distance
can matter.

## Recommended Product Families

| Scientific need | ARM products/codes | Typical role |
| --- | --- | --- |
| Atmospheric profiles | `armbeatm`, `armbeatmhires`, `sonde`, `mergesonde1mace` | Temperature, humidity, pressure, winds |
| Cloud fraction and radiation | `armbecldrad`, `armbecldradhires` | Total/layer cloud, LWP, PWV, radiation |
| Cloud condensate | `microbase`, `microbasekaplus` | Retrieved liquid/ice concentration and effective radius |
| Cloud boundaries | `arscl1cloth`, `arsclkazr1kollias`, MPL masks, ceilometer | Cloud occurrence, base, and top |
| Surface radiation | `qcrad1long`, `radflux1long`, `sirs` | QC broadband surface fluxes |
| Surface state | `met`, `aosmet`, `towermet` | Temperature, RH, pressure, wind, precipitation |
| Surface fluxes | `30ecor`, `30qcecor`, `30ebbr` | Sensible/latent heat and turbulence |
| Precipitation | `rain`, `precipmet`, `vdis`, `disdrometer` | Rate, accumulation, and drop-size context |
| Boundary layer | `pblhtrl1zhang`, `pblhtsonde1mcfarl`, `ceilpblht` | PBL-height diagnostics |
| Aerosol | AOS families listed below | Number, size, chemistry, CCN, optics, hygroscopicity |

The broader process map is in `ARM_JCM_OBSERVATION_MAP.md`.

## Aerosol Product Guide

### Recommended SGP Starting Set

The following are confirmed catalog examples for SGP around the modern AOS
period. Always query the requested dates before downloading.

| Quantity | Example datastream | Important contents |
| --- | --- | --- |
| Harmonized in-situ aerosol | `sgpmergedaerosolE13.c1` | Merged aerosol variables, source/QC metadata |
| Total particle number | `sgpaoscpcfE13.b1` | CPC number concentration, flow/inlet status, QC |
| Fine size distribution | `sgpaossmpsE13.b1` | Mobility-diameter bins and `dN/dlogDp`-type distribution |
| Accumulation size | `sgpaosuhsasE13.b1` | Optical-size number distribution and integrated number |
| Coarse size | `sgpaosapsE13.b1` | Aerodynamic-size distribution and integrated number |
| Merged size spectrum | `sgpmergedsmpsapsE13.c1` | Common merged SMPS/APS size grid and source flags |
| Non-refractory chemistry | `sgpaosacsmE13.b2` | Organics, sulfate, nitrate, ammonium, chloride |
| CCN | `sgpaosccn2colaE13.b1` | CCN concentration and measured/set supersaturation |
| Derived activation | `sgpaosccnsmpskappaE13.c1` | Kappa, activation diameter, CCN/CN ratio |
| Direct hygroscopic growth | `sgpaoshtdmaE13.b1` | Growth-factor distribution, RH, dry diameter |
| Dry scattering | `sgpaosnephdryE13.b1` | Total/backscatter by wavelength and inlet size cut |
| Humidity response | `sgpaosnephwetE13.b1` | Wet/dry scattering and `f(RH)` context |
| Absorption | `sgpaospsap3wE13.b1` | Multi-wavelength absorption, flow, transmittance, QC |
| Corrected absorption | `sgpaoppsap1flynn1mE13.c1` | Corrected PSAP absorption product |
| Spectral AOD | `sgpmfrsraod1michC1.c1` | Cloud-screened spectral aerosol optical depth |
| Best-estimate column/profile optics | `sgpaerosolbe1turnC1.c1` | AOD and estimated extinction/SSA/asymmetry profiles |

The exact NetCDF variable names are intentionally not prescribed here. ARM
variable names, dimensions, corrections, and QC fields vary by datastream
generation. Inspect a representative file before defining a schema.

### What The Measurements Mean

| Instrument/product | Main interpretation caveat |
| --- | --- |
| CPC | Lower size cutoff and inlet losses determine what “total number” includes |
| SMPS | Electrical mobility diameter, usually dried aerosol |
| UHSAS | Optical diameter depends on refractive-index assumptions |
| APS | Aerodynamic diameter is not directly interchangeable with mobility or optical diameter |
| ACSM | Non-refractory submicron composition; collection efficiency and calibration matter |
| CCN | Concentration is meaningful only with supersaturation and instrument status |
| Nephelometer | Scattering depends on wavelength, RH, truncation correction, and size cut |
| PSAP/CLAP | Filter loading and scattering corrections materially affect absorption |
| MFRSR AOD | Column quantity requiring cloud screening; not a surface concentration |
| AEROSOLBE | Retrieval/VAP combining observations and assumptions, not a single direct instrument |

Never merge SMPS, UHSAS, and APS bins as if their diameters were identical.
Retain native diameter type, bin edges, corrections, RH, flow, and inlet-cut
metadata.

### Suggested Bundles

**Aerosol-cloud activation:** CPC, SMPS or merged size distribution, CCN,
ACSM, local AOS meteorology, ARMBE cloud/state, and MICROBASE where condensate is
needed.

**Aerosol-radiation:** dry/wet nephelometer, corrected absorption, MFRSR AOD,
AEROSOLBE, humidity profiles, QCRAD, and cloud screening.

**Minimal exploratory download:** one month of CPC, SMPS, CCN, ACSM, dry/wet
nephelometer, corrected PSAP, AOS meteorology, and matching ARMBEATM/CLDRAD.

## Discovering Datastreams

### Browser

Use ARM Data Discovery:
<https://adc.arm.gov/discovery/>.

Filter by site, date, measurement, instrument, or value-added product. Use the
“Cite Data” function for product-level DOI/citation information.

### Public Catalog Script

The metadata inventory does not require an ARM token:

```bash
python experiments/armbe_sgp/inventory_arm_datastreams.py \
  --site sgp \
  --available-only \
  --visible-only \
  --output experiments/armbe_sgp/outputs/arm_catalog_sgp.json \
  --class-summary-output experiments/armbe_sgp/outputs/arm_catalog_sgp_classes.csv
```

The output records exact datastream names, facility, level, start/end coverage,
retirement status, visibility, and site provenance. It does not provide the
NetCDF variable schema.

The repository also contains a full catalog snapshot:

```text
outputs/arm_catalog_all.json
outputs/arm_catalog_classes.csv
```

## ARM Account And Credentials

ARM data are free, but an ARM user account is required for downloads:

- Registration: <https://adc.arm.gov/armuserreg/#/new>
- Login/token: <https://adc.arm.gov/armlive/home>

Store credentials in environment variables:

```bash
export ARM_USERID="your_arm_user_id"
export ARM_TOKEN="your_access_token"
```

Do not place tokens in notebooks, Git, shared shell scripts, command-line flags,
URLs saved in browser history, or copied logs.

## Querying And Downloading

### Validate Coverage Without Downloading

```bash
python experiments/armbe_sgp/download.py \
  --datastreams sgpaoscpcfE13.b1 sgpaossmpsE13.b1 \
  --start 2018-06-01 \
  --end 2018-07-01 \
  --output /path/to/arm-data \
  --list-only
```

Inspect the returned filenames and date coverage before removing `--list-only`.
An empty result usually indicates the wrong facility, data level, product
generation, or deployment dates.

### Download

```bash
python experiments/armbe_sgp/download.py \
  --datastreams sgpaoscpcfE13.b1 sgpaossmpsE13.b1 \
  --start 2018-06-01 \
  --end 2018-07-01 \
  --output /path/to/arm-data
```

Files are stored in one directory per datastream. Downloads stream through a
`.part` file and are renamed only when complete. Existing nonempty files are
skipped, so the command is resumable.

ARM Live Data uses:

```text
GET https://adc.arm.gov/armlive/query
GET https://adc.arm.gov/armlive/saveData
```

The repository client passes credentials through `requests` query parameters.
Do not print or archive prepared request URLs because they contain the token.

For large products, start with days or one month. Size distributions,
high-frequency radar, MICROBASE, and scanning products can become very large.

## Inspecting Native Files

ARM `.cdf` and `.nc` files are NetCDF and can be opened with xarray.

### Inspect One File First

```python
from pathlib import Path
import xarray as xr

path = next(Path("/path/to/arm-data/sgpaoscpcfE13.b1").glob("*.cdf"))

with xr.open_dataset(path, decode_times=True) as ds:
    print(ds)
    print(ds.attrs)
    for name, variable in ds.data_vars.items():
        print(name, variable.dims, variable.attrs.get("units"),
              variable.attrs.get("long_name"))
```

Record variable names, units, dimensions, fill values, valid ranges, QC fields,
and global processing/version attributes before combining files.

### Open A Consistent Time Series

```python
from pathlib import Path
import xarray as xr

files = sorted(Path("/path/to/arm-data/sgpaoscpcfE13.b1").glob("*.cdf"))
ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 10_000},
    data_vars="minimal",
    coords="minimal",
)
```

Do not combine different datastream generations or facilities until their
variables, units, and processing changes have been audited.

## Quality Control

ARM commonly provides `qc_<variable>` fields and flag metadata. QC can be
bit-packed; inspect attributes such as `flag_masks`, `flag_values`, and
`flag_meanings` rather than assuming every nonzero value has the same meaning.

A conservative first mask is often:

```python
value = ds["variable_name"]
qc = ds["qc_variable_name"]
conservative_good = value.where(qc == 0)
```

This may discard usable “indeterminate” or advisory samples. A final scientific
policy should decode the documented bits and state which conditions are
accepted.

General rules:

- Do not interpret missing or retrieval-failure values as zero.
- Apply QC before temporal averaging.
- Do not average QC integers themselves.
- Preserve valid-sample counts for every aggregate.
- Preserve retrieval uncertainty and source flags.
- Check for instrument transitions and calibration changes.
- Keep native units in the observation layer and document every conversion.

For aerosol data, also preserve:

- dry versus ambient convention;
- STP versus ambient concentration convention;
- inlet size cut and transmission;
- wavelength;
- CCN supersaturation;
- diameter type and bin edges;
- correction algorithm/version; and
- facility/inlet location.

## Collocation Guidance

Join products only after selecting explicit time windows and tolerances. Different
ARM products can have second, minute, hourly, or sounding cadence.

Recommended workflow:

1. Retain a native observation layer with original time, QC, and instrument ID.
2. Define a harmonized layer with canonical units and explicit averaging windows.
3. Build task-specific examples from the harmonized layer.
4. Preserve pointers from every derived sample to source datastreams/files.
5. Split train/validation/test by site and time blocks, never random dense rows.

Do not assume measurements at SGP C1 and E13 are collocated. Preserve facility
coordinates and quantify separation when combining aerosol and cloud products.

## Provenance, Sharing, And Citation

ARM encourages data access and sharing, but collaborators should be able to
trace processed values to the authoritative archive.

For every downloaded module, save:

- ARM datastream names and facilities;
- query start/end dates and access date;
- returned filename list;
- checksums;
- global NetCDF attributes and processing versions;
- retained variables and units;
- QC interpretation and exclusions;
- temporal/vertical aggregation recipe;
- code revision and environment; and
- ARM product DOI/citation from Data Discovery.

ARM states that the ARM User Facility should be acknowledged as the source of
data and provides product-level DOIs. Consult:

- Data overview: <https://www.arm.gov/data>
- Data use guidance: <https://www.arm.gov/guidance/datause>
- Acknowledgment/DOI guidance:
  <https://www.arm.gov/working-with-arm/acknowledging-arm>

For collaboration, prefer sharing compact processed artifacts plus manifests and
source pointers. Let recipients retrieve raw files from ARM when feasible so
they receive authoritative versions, documentation, and DOI information.

## Repository References

**NOTE:** Referenced files are available on branch `single-column-exp-armbe` of jax-gcm repo.

- `download.py`: authenticated, resumable ARM Live Data client
- `inventory_arm_datastreams.py`: public metadata-only catalog generator
- `ARM_JCM_OBSERVATION_MAP.md`: scientific product-family map
- `LOCAL_DATA_AUDIT.md`: local raw/processed inventory and release policy
- `DATA_PRODUCTS.md`: ARMBE variable interpretation
- `ARMBE_STANDARD_VARIABLE_CADENCE.md`: standard-product cadence audit
- `outputs/arm_catalog_all.json`: full catalog snapshot
- `outputs/arm_catalog_classes.csv`: instrument-class summary
