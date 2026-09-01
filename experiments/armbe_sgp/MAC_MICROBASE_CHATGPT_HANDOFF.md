# ChatGPT Handoff: SGP MICROBASE Mac Pipeline

## Mission

Run and supervise the Apple Silicon SGP MICROBASE pipeline. First prove that
the seven June 2011 raw files on the Mac reproduce the included verified Linux
outputs. Do not begin bulk production until that benchmark passes.

## Start Here

Read, in order:

1. `MAC_MICROBASE_README_FIRST.md`
2. `MAC_LOCAL_MICROBASE_PIPELINE_PLAN.md`
3. `microbase_campaign.toml`

Then run:

```bash
./setup_microbase_mac.sh
./run_microbase_mac_benchmark.sh ~/Downloads/arm-microbase
```

## Scientific Invariants

- Processing schema is version 4.
- Windows are 60 minutes centered on ARMBE hourly timestamps and half-open.
- Missing retrievals are never converted to clear sky.
- Clear retrieval cells contribute zero condensate.
- Liquid and ice QC must both pass for cloudy condensate pairs.
- Atmospheric profiles are selected without time interpolation.
- Pressure is reconstructed hydrostatically from observed surface pressure.
- Concentration is converted from `g m-3` to `kg kg-1` with paired density.
- A day with no model-valid samples may still be a valid reduced day.
- Cross-platform acceptance compares scientific arrays, not NetCDF bytes.
- Cross-platform floating fields use `rtol=1e-6, atol=1e-12`, established by
  the seven-day Apple Silicon benchmark; discrete fields remain exact.

Do not change these rules while debugging orchestration. A scientific change
requires a new processing schema and an explicit review.

## Safety Invariants

- Never request or store the OAuth cookie in chat, config, logs, or the ledger.
- Never package SSH keys.
- Never delete raw data before a matching server acceptance receipt exists.
- Cleanup uses exact ledger paths, never a broad glob.
- Do not publish over an existing month with a different archive hash.
- Keep the SQLite ledger on local APFS storage.

## Recovery

Run `status` first. The normal `run` command is restartable. A failed download
keeps its `.part`; a failed reduction removes only its temporary output; a
failed upload retains the local archive. If a failure is not clearly retryable,
stop and inspect the ledger and filesystem without deleting either.

## Expected Benchmark Inputs

Raw files:

```text
sgpmicrobaseC1.c1.20110624.000000.nc
...
sgpmicrobaseC1.c1.20110630.000000.nc
```

Included companions and references are under `benchmark/`.

## After Benchmark

Report processing wall time, peak working storage, all comparison results, and
the final status JSON. If successful, refresh the THREDDS login, export the
cookie privately in Terminal, and process one complete month as the recovery
pilot before scheduling continuous production.
