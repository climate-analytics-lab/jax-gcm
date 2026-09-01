# Mac-Local MICROBASE Reduction Plan

## Purpose

This document defines the acquisition and reduction workflow for the ARM
MICROBASE archive in order `268516`. Raw files are downloaded and processed on
the Mac, while only verified reduced products are transferred to MOSAIC shared
storage. This replaces the earlier raw-data transfer through `roselab1`, whose
measured throughput would make a 6.7 TB campaign take approximately six to
eight weeks.

The immediate scope is SGP data with matching ARMBEATM and ARMBECLDRAD products.
Expansion to other ARM sites is conditional on a schema and companion-product
audit. The existence of a MICROBASE file alone does not make that day usable by
the current observational operator.

## Decision

Use this data path:

```text
ARM THREDDS
    -> Mac raw staging
    -> Mac reduction and verification
    -> monthly reduced archive and checksum
    -> roselab1
    -> shared /data/MOSAIC storage and verification
    -> raw cleanup on the Mac
```

Do not transfer raw MICROBASE files from the Mac to `roselab1` during routine
production. Server-side raw processing remains a recovery option for files that
are already on MOSAIC storage.

## Implemented Components

The SGP-first implementation is now concrete:

| File | Role |
| --- | --- |
| `microbase_mac_pipeline.py` | Restartable CLI, SQLite ledger, download, atomic reduction, packaging, upload, remote publication, receipt verification, and cleanup |
| `microbase_physics.py` | Frozen schema-4 physical constants and vapor-pressure calculation without a JAX dependency |
| `collocate_microbase_pilot.py` | Scientific hourly aggregation and atmospheric collocation kernel |
| `process_microbase_month.py` | Existing Linux batch processor and day-output verifier |
| `microbase_mac_pipeline_test.py` | State, archive, publication, receipt, cleanup, and equivalence safety tests |
| `requirements-microbase-mac.txt` | Pinned Python 3.12 runtime and test dependencies |
| `microbase_campaign.example.toml` | Production paths, THREDDS endpoint, limits, companions, and remote settings |
| `microbase_benchmark.toml` | Isolated seven-day benchmark workspace and ledger |
| `setup_microbase_mac.sh` | Apple Silicon environment creation and `doctor` check |
| `run_microbase_mac_benchmark.sh` | June 24-30 processing and Linux-reference comparison |
| `build_mac_microbase_bundle.py` | Credential-free bundle builder with companions and references |
| `MAC_MICROBASE_README_FIRST.md` | Operator instructions |
| `MAC_MICROBASE_CHATGPT_HANDOFF.md` | Execution and recovery context for another session |

Normal production uses:

```bash
.venv-microbase/bin/python microbase_mac_pipeline.py run \
  --config microbase_campaign.toml \
  --month YYYY-MM
```

Durable stopping points are `downloaded`, `processed_verified`, `packaged`,
`server_verified`, and `raw_deleted`. A partial date-range benchmark is marked
noncanonical and cannot be packaged or published as a complete calendar month.

## Basis For The Decision

Measured rates and output sizes are:

| Quantity | Measurement |
| --- | ---: |
| ARM THREDDS to Mac | 11-12.4 MB/s |
| Mac to `roselab1` using one SCP stream | approximately 1.8 MB/s |
| Typical raw daily file | 670,485,008 bytes |
| Seven raw June 2011 days | approximately 4.69 GB |
| Seven reduced June 2011 days | approximately 13 MB on disk |
| Approximate reduction ratio | 0.28% of raw size |

For 6.7 TB, these measurements imply:

| Stage | Expected duration |
| --- | ---: |
| ARM download | 6.3-7.1 continuous days |
| Mac processing | expected to fit behind download; benchmark required |
| Upload of approximately 19-20 GB reduced output | 3-5 hours |
| Practical campaign | 7-10 days continuous, or 10-14 days with interruptions |

All estimates use decimal network units and should be treated as planning
ranges. The first Mac benchmark is a required gate before committing to the
full campaign.

## Scientific Scope Gate

The current processor has been validated for SGP 2011 and 2018. It currently
assumes:

- an SGP MICROBASE filename and facility;
- a 596-level MICROBASE/ARMBECLDRAD height grid;
- 24 hourly ARMBEATM and ARMBECLDRAD samples per day;
- matching annual ARMBEATM and ARMBECLDRAD files;
- known legacy and modern SGP ARMBEATM schemas; and
- the audited SGP MICROBASE units and retrieval/QC fields.

Order `268516` contains MICROBASE data from eight sites, but companion-product
availability and schema compatibility have not been established for all of
them. Data from another site must not enter production merely because its
filename appears in the order.

Before enabling a new site/year, record:

- MICROBASE datastream, facility, date range, grid, units, and schema;
- matching ARMBEATM and ARMBECLDRAD datastreams and coverage;
- exact time and height alignment behavior;
- processing success on representative clear, liquid, ice, and mixed-phase
  days; and
- any site-specific interpretation or exclusion rules.

## Streaming Lifecycle

### 1. Inventory

Build a machine-readable inventory from the ARM order before production. One
record represents one MICROBASE source file and contains at least:

```text
site
facility
day
source URL
expected filename
expected size, when available
companion atmosphere file
companion cloud-radiation file
state
attempt count
last error
raw SHA-256
reduced archive SHA-256
```

Allowed lifecycle states are:

```text
discovered
companion_blocked
queued
downloading
downloaded
processing
processed_verified
packaged
uploading
server_verified
raw_deleted
failed_retryable
failed_manual_review
```

State transitions must be atomic. Restarting the coordinator must resume from
the ledger rather than infer state from terminal output.

### 2. Download

Download to a bounded Mac staging directory grouped by site, year, and month.
Use `.part` files and rename only after the expected minimum size and NetCDF
open checks pass. Authentication uses the temporary browser OAuth cookie; the
cookie must be supplied through the environment or a protected local file and
must never be written into the ledger, logs, manifests, or repository.

The Mac must remain awake and connected during acquisition. Use the operating
system's supported mechanism to prevent sleep for the duration of an active
batch.

### 3. Process

Run the existing daily reduction logic locally against the raw MICROBASE file
and its annual ARMBEATM and ARMBECLDRAD companions. The processor produces:

```text
DAY/microbase_hourly.nc
DAY/observed_atmosphere_paired.nc
DAY/manifest.json
```

The processor must remain scientifically identical across Mac and Linux. Mac
portability work may change packaging, imports, command-line orchestration, and
filesystem paths, but must not silently alter constants, QC rules, averaging,
hydrostatic reconstruction, condensate conversion, or validity masks.

### 4. Verify

The existing day-level verification remains mandatory. It checks dimensions,
timestamps, required fields, atmospheric support, finite valid samples, and
monotonically decreasing pressure. Every manifest records source and output
sizes and SHA-256 checksums.

Mac and Linux processing of the same representative day must produce equivalent
scientific arrays. Exact NetCDF file bytes are not required to match because
backend metadata and encoding can differ, but coordinates, masks, values, and
manifest statistics must match within explicit numerical tolerances.

The June 24-30 Apple Silicon benchmark established `rtol=1e-6` and
`atol=1e-12` as the strictest tested cross-platform threshold that passes all
seven days. The difference comes from propagation of platform math-library
rounding through hydrostatic reconstruction and density-derived fields. Integer
counts, boolean masks, timestamps, dimensions, and discrete coordinates still
require exact equality.

### 5. Package

Package verified output by site and calendar month. A monthly package avoids
opening a new SSH connection for each of tens of thousands of small files.

Recommended package layout:

```text
SITE/YYYY-MM/
    YYYY-MM-DD/
        microbase_hourly.nc
        observed_atmosphere_paired.nc
        manifest.json
    batch_manifest.json
```

Create an uncompressed tar archive unless a benchmark demonstrates a material
benefit from compression. NetCDF data may already be compressed, so expensive
recompression is not assumed to help. Write the archive to a temporary name,
compute SHA-256, then rename it atomically.

### 6. Upload And Server Verification

Upload the monthly archive to `roselab1` using a `.part` suffix. Compare local
and remote byte sizes, compute the remote SHA-256, and rename only after the
checksums match. Extract into a temporary shared-storage directory, validate all
day manifests and outputs, then atomically publish the month under
`/data/MOSAIC/jax-gcm/experiments/armbe_sgp/outputs/`.

The monthly archive may be deleted from `roselab1` local storage after the
shared extracted month passes verification.

### 7. Cleanup

Delete a Mac raw file only when its reduced outputs have passed local validation
and the corresponding monthly package has reached `server_verified`. This is
more conservative than deleting raw data immediately after local processing and
ensures that at least one recoverable copy exists until MOSAIC has accepted the
result.

Retaining reduced Mac output after server verification is optional. The ledger,
source provenance, day manifests, batch manifest, and shared reduced outputs are
permanent campaign artifacts.

## Storage Budget

The workflow uses a bounded queue rather than storing the complete raw archive.
A typical 30-day month is approximately 20 GB raw. Budget:

| Item | Working allowance |
| --- | ---: |
| Raw monthly staging | 20-25 GB |
| Temporary and completed reduced products | 1-2 GB |
| Package construction and retry headroom | 5-10 GB |
| Recommended free working space | 30-50 GB |

If actual files or a site's monthly coverage exceed this allowance, the
coordinator must reduce the batch size rather than allow the disk to fill.

## Mac Environment

The production environment should be reproducible and isolated from the Mac's
base Conda environment. Use a pinned Python version and lock the minimal runtime
dependencies. The reduction path needs NumPy, Xarray, a NetCDF backend, and the
small set of experiment modules used by the processor; it does not need PySR,
model training, GPU JAX, or the complete GCM runtime.

The collocation module uses `microbase_physics.py`, a frozen snapshot of the
schema-4 JCM constants. Its fingerprint is recorded in each new manifest and
repository tests check the values against the scientific contract. This avoids
loading JAX, Dinosaur, or the eager `jcm` package on the Mac while preventing a
silent constants change from altering an existing data version.

Required portability checks are:

- Apple Silicon-compatible Python and NetCDF packages install cleanly;
- one source file can be opened and reduced without the full JAX-GCM runtime;
- output verification passes locally;
- Mac output agrees scientifically with the verified Linux output; and
- paths contain no assumptions about `/data/MOSAIC` or `/home/ubuntu`.

## Companion Data

Annual ARMBEATM and ARMBECLDRAD files are much smaller than daily MICROBASE
files and should be cached locally by site/year. A source file must not be
downloaded into the production queue until both companions are present and
validated, unless the ledger marks it `companion_blocked` rather than queued.

Companion files are retained across monthly batches. Their path, size, and
SHA-256 are included in every batch manifest so a future user can reconstruct
the exact collocation inputs.

## Failure And Recovery Rules

| Failure | Required behavior |
| --- | --- |
| OAuth redirect or expired cookie | stop download, preserve `.part`, refresh authentication |
| Interrupted download | resume or restart only the `.part` file |
| Invalid or undersized NetCDF | quarantine source and mark manual review |
| Missing companion data | mark `companion_blocked`; do not process |
| Processing exception | preserve raw file and logs; increment attempt count |
| Output verification failure | preserve raw file; remove or quarantine incomplete output |
| Interrupted upload | retain local archive and remote `.part`; resume safely |
| Remote checksum mismatch | do not publish or delete local data |
| Shared output verification failure | do not mark server verified or delete Mac raw data |

No cleanup step may run on a broad glob without consulting verified ledger
records. Raw deletion must always be scoped to explicitly verified files.

## Implementation Phases

### Phase 1: Seven-Day Mac Benchmark

- Build the minimal Mac environment.
- Transfer or download the matching 2011 ARMBEATM and ARMBECLDRAD files.
- Process June 24-30 from the existing Mac raw files.
- Compare all seven days with the verified Linux products.
- Measure processing wall time, peak disk use, and reduced size.

Exit criterion: all scientific arrays and validity masks agree, all manifests
pass, and local processing is faster than the 11 MB/s acquisition stream on
average.

### Phase 2: One-Month End-To-End Pilot

- Implement the persistent ledger and bounded queue.
- Download, process, package, upload, and verify one complete SGP month.
- Interrupt and resume one download and one upload deliberately.
- Confirm raw cleanup occurs only after server verification.

Exit criterion: a restarted pipeline completes the month without duplicate
processing, lost files, or manual state reconstruction.

### Phase 3: SGP Production

- Inventory all SGP years with compatible companions.
- Process in chronological monthly batches.
- Monitor download rate, processing lag, free disk space, failures, and verified
  output counts.
- Stop automatically if the processing queue or disk budget exceeds configured
  thresholds.

Exit criterion: every eligible SGP source is either `server_verified` or has an
explicit exclusion/failure record.

### Phase 4: Additional Sites

- Audit one representative year per site.
- Generalize filename discovery and remove SGP-only assumptions explicitly.
- Add schema fixtures and tests before enabling production.
- Re-estimate scientific utility and cost for each site.

Exit criterion: site-specific tests, companion coverage, and scientific review
are complete. Full eight-site processing is not an automatic consequence of SGP
success.

## Monitoring And Completion Criteria

The coordinator should report at least:

```text
files and bytes discovered
files and bytes downloaded
download MB/s and estimated completion
files processed and verified
processing seconds per day
queued raw bytes and free disk bytes
packages uploaded and server verified
retry and manual-review counts
```

The campaign is complete when:

- every eligible inventory record is `server_verified` or has a documented
  disposition;
- every shared day passes the current schema verifier;
- batch manifests reconcile source, output, and ledger counts;
- no `.part` or unpublished temporary directories remain;
- raw cleanup is recorded rather than assumed; and
- the reduced cohort can be opened and summarized independently on MOSAIC.

## Immediate Next Actions

1. Download and extract the generated Apple Silicon bundle.
2. Run `setup_microbase_mac.sh` and inspect the `doctor` report.
3. Run `run_microbase_mac_benchmark.sh` against the seven raw June files already
   on the Mac.
4. Record benchmark runtime, storage, and all seven comparison results.
5. Run one complete SGP month as an interruption/recovery pilot.
6. Add an exported ARM order inventory before unattended multi-year production;
   the current SGP CLI constructs calendar-day filenames and deliberately stops
   on a missing source rather than silently omitting it.
7. Audit companion-product coverage before scheduling non-SGP data.
