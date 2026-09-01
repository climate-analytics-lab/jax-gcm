# Local OpenCode Prompt: Fresh SGP MICROBASE Setup

Paste the text below into a local OpenCode session, or ask it to read this file
and follow it exactly.

---

We are setting up the SGP MICROBASE Mac-local processing pipeline from scratch.

## Safety Rules

- Do not use or modify any previous extracted bundle.
- Work in a new directory under `~/Desktop/microbase-fresh`.
- Do not delete any raw MICROBASE files.
- Do not upload or publish anything yet.
- Do not request, print, store, or paste an ARM OAuth cookie.
- Stop after the seven-day benchmark and report the result.
- Do not loosen the bundled comparison tolerance beyond `rtol=1e-6`.
- The seven raw benchmark files should already be in
  `~/Downloads/arm-microbase`.

## Procedure

### 1. Create A Fresh Directory

Create:

```text
~/Desktop/microbase-fresh
```

### 2. Download The Bundle

Download these two files from `roselab1`:

```bash
scp -P 12000 -i ~/.ssh/mfisher \
  ubuntu@roselab1.ucsd.edu:/data/MOSAIC/jax-gcm/experiments/armbe_sgp/dist/microbase-sgp-mac-apple-silicon-20260825.tar.gz \
  ~/Desktop/microbase-fresh/
```

```bash
scp -P 12000 -i ~/.ssh/mfisher \
  ubuntu@roselab1.ucsd.edu:/data/MOSAIC/jax-gcm/experiments/armbe_sgp/dist/microbase-sgp-mac-apple-silicon-20260825.tar.gz.sha256.json \
  ~/Desktop/microbase-fresh/
```

### 3. Verify The Archive

The archive SHA-256 must be exactly:

```text
8aae02e52a6133f0de21892a27f380aad12ce4a681fee03922e6a1eb3fafb1da
```

Run:

```bash
cd ~/Desktop/microbase-fresh
shasum -a 256 microbase-sgp-mac-apple-silicon-20260825.tar.gz
```

Stop if it does not match.

### 4. Extract The Archive

```bash
tar -xzf microbase-sgp-mac-apple-silicon-20260825.tar.gz
cd microbase-sgp-mac-apple-silicon-20260825
```

### 5. Read The Documentation

Read these files before proceeding:

- `MAC_MICROBASE_README_FIRST.md`
- `MAC_MICROBASE_CHATGPT_HANDOFF.md`
- `MAC_LOCAL_MICROBASE_PIPELINE_PLAN.md`

### 6. Verify Bundle Members

```bash
python3 verify_microbase_bundle.py
```

Expected result:

```text
verified 41 bundle files
```

Stop if bundle verification fails.

### 7. Check The Requirements

Confirm `requirements-microbase-mac.txt` includes:

- Pandas
- Packaging
- cftime
- Certifi
- python-dateutil
- Six
- NumPy
- Xarray
- NetCDF4
- Pytest

### 8. Create And Diagnose The Environment

```bash
chmod +x setup_microbase_mac.sh run_microbase_mac_benchmark.sh
./setup_microbase_mac.sh
```

The doctor check must confirm:

- Python 3.12 is active.
- The machine is Apple Silicon `arm64`.
- NumPy, Pandas, Xarray, and NetCDF4 import successfully.
- A NetCDF write/read round trip succeeds.
- Both 2011 companion files exist.
- `~/.ssh/mfisher` exists.

Diagnose and fix environment problems if needed, but do not alter the
scientific processing code.

### 9. Check The Seven Raw Inputs

Confirm these files exist under `~/Downloads/arm-microbase`:

```text
sgpmicrobaseC1.c1.20110624.000000.nc
sgpmicrobaseC1.c1.20110625.000000.nc
sgpmicrobaseC1.c1.20110626.000000.nc
sgpmicrobaseC1.c1.20110627.000000.nc
sgpmicrobaseC1.c1.20110628.000000.nc
sgpmicrobaseC1.c1.20110629.000000.nc
sgpmicrobaseC1.c1.20110630.000000.nc
```

### 10. Run The Benchmark

```bash
./run_microbase_mac_benchmark.sh ~/Downloads/arm-microbase
```

The bundled comparison threshold is `rtol=1e-6` and `atol=1e-12`. Discrete
coordinates, masks, timestamps, dimensions, and counts must still match
exactly.

Expected final output includes:

```text
scientific arrays match
```

printed seven times, followed by a status containing:

```text
batch_month: 2011-06
count: 7
state: processed_verified
bytes: 4693395056
```

and finally:

```text
seven-day Apple Silicon benchmark passed
```

### 11. Stop After The Benchmark

Do not run a production month, package, upload, publish, or clean up raw files.

Report:

- Archive checksum result.
- Bundle verification result.
- Doctor output summary.
- Number of scientific comparisons passed.
- Final ledger status.
- Any warnings or failures.
