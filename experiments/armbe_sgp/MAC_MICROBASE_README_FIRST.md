# SGP MICROBASE Mac Pipeline

This bundle downloads and reduces SGP MICROBASE data locally on an Apple
Silicon Mac, then transfers only verified monthly reduced archives to MOSAIC.
It does not require JAX or the JCM model environment.

## Security

The bundle contains no OAuth cookie, ARM token, SSH private key, or raw
MICROBASE file. Never paste credentials into configuration, logs, the SQLite
ledger, or a ChatGPT conversation. Export the temporary THREDDS cookie only in
the terminal environment used for a production run.

## Included Benchmark Data

- 2011 annual SGP ARMBEATM companion file.
- 2011 annual SGP ARMBECLDRAD companion file.
- Verified Linux reduced outputs for June 24-30, 2011.
- Source and output checksums.

The seven approximately 670 MB raw MICROBASE files are not included because
they are already on the Mac.

## First Run

1. Extract the bundle.
2. Open Terminal in the extracted directory.
3. Verify every bundled file before creating local state:

```bash
python3 verify_microbase_bundle.py
```

4. Copy the example configuration:

```bash
cp microbase_campaign.example.toml microbase_campaign.toml
```

5. Inspect `microbase_campaign.toml`, especially the SSH identity path.
6. Create the environment and run diagnostics:

```bash
chmod +x setup_microbase_mac.sh run_microbase_mac_benchmark.sh
./setup_microbase_mac.sh
```

7. Run the equivalence benchmark, replacing the directory if needed:

```bash
./run_microbase_mac_benchmark.sh ~/Downloads/arm-microbase
```

The benchmark must end with:

```text
seven-day Apple Silicon benchmark passed
```

## Production Run

Refresh the ARM browser login and export the `_oauth2_proxy` cookie without
putting it in a file:

```bash
read -s THREDDS_COOKIE
export THREDDS_COOKIE
```

Run one month:

```bash
.venv-microbase/bin/python microbase_mac_pipeline.py run \
  --config microbase_campaign.toml \
  --month 2011-07
```

The default run downloads, reduces, packages, uploads, verifies, publishes, and
then deletes only the exact raw files covered by the server acceptance receipt.

Useful stopping points are:

```bash
--stop-after downloaded
--stop-after processed_verified
--stop-after packaged
--stop-after server_verified
```

Inspect state at any time:

```bash
.venv-microbase/bin/python microbase_mac_pipeline.py status \
  --config microbase_campaign.toml
```

Rerunning the same command resumes from `workspace/pipeline.sqlite3`.

## Limits

This package intentionally accepts only SGP C1 production MICROBASE data with
the audited 596-height schema. Other sites, facilities, and PI2 files require a
separate scientific and companion-product audit.
