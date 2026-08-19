# Local ARMBE Data Audit

This audit separates completeness of individual ARM archive orders from
completeness of the current public ARMBE catalog. The snapshot was generated on
2026-08-13 with:

```text
python experiments/armbe_sgp/inventory_arm_datastreams.py \
  --output experiments/armbe_sgp/outputs/arm_catalog_all.json \
  --class-summary-output experiments/armbe_sgp/outputs/arm_catalog_classes.csv
python experiments/armbe_sgp/audit_local_armbe.py \
  --catalog experiments/armbe_sgp/outputs/arm_catalog_all.json \
  --output experiments/armbe_sgp/outputs/local_armbe_audit.json
```

The local audit reads filesystem metadata and archive `file_list.txt` files. It
does not inspect NetCDF contents. Add `--hash-duplicates` to prove whether
same-name, same-size duplicate candidates are byte-identical; the 2026-08-13
snapshot did not hash the large payloads.

## Result

The local raw tree contains 420 payload files totaling 43,042,792,274 bytes
(about 41 GiB). Both downloaded archive orders are internally complete:

| Collection | Payloads | Bytes | Datastreams | Interpretation |
|---|---:|---:|---:|---|
| `order-267892` | 298/298 | 31,423,065,436 | 38 | Complete relative to that order's manifest; broad ARMBE collection. |
| `order-267737` | 59/59 | 5,394,731,112 | 2 | Complete SGP standard ATM/CLDRAD order. |
| `full_range_armlive` | 59 | 5,394,731,112 | 2 | Same names and sizes as order `267737`; duplicate candidate, not independently hashed. |
| `microbase_probe` | 2 | 830,024,008 | 2 | One MICROBASE and one MICROBASEPI2 probe; not ARMBE. |
| `synthetic` | 2 | 240,606 | 0 parsed | Test fixtures, not observations. |

The public catalog snapshot contains 40 ARMBE datastreams across seven product
families. Local observations contain 38 of those streams across five families.
The missing streams and families are:

- `sgparmbe2dgridX1.c1` (`armbe2dgrid`)
- `sgparmbestnsX1.c1` (`armbestns`)

Therefore, the workspace contains every file requested in orders `267892` and
`267737`, but it does **not** contain every currently cataloged ARMBE product.
Catalog presence also does not establish public variable metadata or scientific
fitness; those remain unresolved for `armbe2dgrid` and `armbestns`.

## Raw Collection Policy

Use `order-267892` as the provenance-bearing raw source for the pooled standard
and high-resolution experiments. It includes standard `armbeatm` and
`armbecldrad`, SGP `armbeland`, and 2023-2024 high-resolution ATM/CLDRAD at ENA,
NSA, and SGP.

Do not combine these collections as independent observations:

- `order-267737` and `full_range_armlive` repeat the SGP standard streams already
  represented in `order-267892`; keep them only as acquisition records until
  hashes and retention policy justify deletion.
- `microbase_probe` is exploratory input for condensate-variable assessment.
- `synthetic` is test-only and must never enter scientific samples.

No files were deleted by this audit.

## Processed Artifacts

The processed tree at audit time contained 117 files totaling 231,743,254 bytes
(about 221 MiB). It mixes reusable data products with experiment results:

| Class | Top-level artifacts | Files | Bytes |
|---|---:|---:|---:|
| Observational/model caches | 4 | 15 | 92,859,514 |
| Feature exports and searches | 4 | 47 | 117,240,475 |
| Calibration runs | 3 | 21 | 1,319,596 |
| Diagnostic runs | 3 | 15 | 703,318 |
| Evaluation runs | 1 | 5 | 241,039 |
| Hindcast/SCM intermediates | 11 | 11 | 1,836,053 |
| Standalone plots | 1 | 1 | 506,418 |
| Catalog metadata | 2 | 2 | 17,036,841 |

The two approximately 39 MB unified caches are alternative split products, not
two independent observational datasets. The randomized month-block cache is the
source of the selected pooled symbolic-regression experiment.

## Canonical Shareable Release

The minimal canonical release should contain only the selected data products
and enough metadata to reproduce and interpret them:

```text
armbe-cloud-closure-<version>/
  README.md
  provenance.json
  data_dictionary.json
  qc_and_exclusions.json
  coverage.csv
  standard_t30/
    samples.nc
    split_manifest.json
    site_terrain.json
    features.nc
    train.csv
    validation.csv
    test.csv
  layerwise_hires/
    features.npz
    train.csv
    validation.csv
    test.csv
    manifest.json
```

The release should use
`cache_armbe_unified_paired_standard_random_month_blocks` and
`symbolic_features_unified_t30` for the standard T30 product. The layerwise
high-resolution export may be included, but its manifest must retain the absent
`qc`/`qi` warning and identify MICROBASE integration as unresolved.

Exclude the alternative unified cache, smoke products, SGP-only predecessor
exports, calibration/search directories, predictions, plots, hindcasts, SCM
intermediates, raw ARM payloads, MICROBASE probes, and synthetic fixtures. Raw
ARM data should remain referenced by archive order, datastream, filename, and
source path rather than silently repackaged into the processed release.
