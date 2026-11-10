# RTX performance experiments with unchanged numerical output

This experiment tests execution settings on one dedicated RTX PRO 6000
Blackwell Server Edition. It does not yet establish a speedup. The source
baseline is `dev` commit `5138e4c2f6f6f412e887f1dbc8f3342d7e50b95c`;
`main` was still at the older benchmark revision when the checkout was made.
The model source and physics remain unchanged on the experiment branch.

## Reproduce the experiment

`tools/rtx_spot/plan.json` records the model preset, variants and execution
order. `tools/rtx_spot/constraints.txt` pins the previously working Python
3.12 cloud environment, including JAX/jaxlib 0.10.2, dinosaur 1.5.1 and
jax-rrtmgp 0.7.0. It deliberately avoids confounding code changes with a
JAX upgrade. The VM uses Ubuntu 24.04 and the CUDA 12.9 deep learning image.

Export the existing Terraform `spot_job_config` output to a JSON file, then
run from the repository root:

```bash
python tools/rtx_spot/launch.py --config /path/to/spot-config.json \
  --output /path/to/results --dry-run
python tools/rtx_spot/launch.py --config /path/to/spot-config.json \
  --output /path/to/results
```

The launcher requires committed source, archives only tracked files and
records its revision, archive hash, exact VM command and result URI. Push
the branch before allocating a VM so temporary configurations are available
to collaborators. `--zone` can select another allowed central-US or Oregon
Spot pool. The VM is a Spot-only `g4-standard-48`, with a dedicated full GPU,
an auto-delete boot disk and a server-enforced two-hour deletion cap. It uses
the existing restricted runtime service account and private result bucket;
it neither changes IAM nor puts temporary VMs in shared Terraform state.
The launcher deletes the VM after collecting results. A preempted attempt
is a failed attempt, never a resumed timing sample.

`--on-demand-35m` is an explicit alternative for use only when an on-demand
run has been authorized. It changes provisioning to STANDARD, imposes a
35-minute VM deletion limit and a 25-minute experiment timeout. It never
follows a Spot failure automatically. At the 2026-11-10 Oregon catalog rate
of $4.49993/hour, the compute cap is about $2.63; setup, profiling and
compilation consume part of this shorter window, so the entire eight-run
sweep may not finish. Partial summaries and logs upload under unique names
while running; completion publishes the full result bundle before its marker.

On a dedicated local RTX with the pinned environment, set `RESULT_DIR` to a
writable absolute directory and run `tools/rtx_spot/sweep.py`. It orchestrates
the existing tools rather than implementing a model loop.

## What is held fixed

All runs use `+configuration=t63-echam-rrtmgp`: T63L47, ECHAM 1M physics,
two-stream RRTMGP, dry JW initialization, native terrain, prescribed
present-day SST/sea ice, packaged/mirrored ozone, the production sponge and
the preset's 12-minute timestep. The data mirror is pinned by the source;
the production benchmark tool records the resolved mirror commit and
prefetches its required inputs. No precision, physics, timestep or radiation
cadence change is allowed.

This is a different workload from the historical flat-ocean
`echam_speed_test.py` runs at the older `main` revision. Compare variants
against this experiment's baseline, not against the historical L4/RTX
absolute times.

## Candidates and measurement

The baseline already uses all-g-point gas optics, 64-g-point batches and
scanned vertical recurrences, settings tuned on an A100. The candidates are:

* `graphs`: lower the CUDA graph capture threshold to one operation.
* `chunk128`: increase the radiation spectral batch from 64 to 128. This
  uses more scratch memory and changes the spectral reduction grouping;
  mathematically equivalent physics alone does not qualify it as unchanged
  numerical output.
* `latency`: enable the latency-hiding scheduler with a 95% memory slop
  factor. This explores a memory-for-scheduling tradeoff without requesting
  different arithmetic or physics.

Run baseline, graphs, chunk128, latency, latency, chunk128, graphs, baseline
sequentially on the same GPU. Each is 30 simulated days in six 5-day chunks.
`tools/benchmark.py` discards compilation chunk 1, uses per-chunk wall times
and requires the final two chunks to agree within 3%. Its GPU telemetry
records peak resident memory. The separate baseline `profile_terms.py`
trace disables CUDA graphs for attribution; its wall time is not a
throughput measurement. Read its mixed-kernel fraction before interpreting
the per-term split.

After every successful run, compare the original bytes, shapes and dtypes
of all checkpoint dynamics, physics and clock arrays, and every output
netCDF variable and coordinate, including signed zeros. Record hashes and
finiteness in `<label>-arrays.json`; mismatches appear in `summary.json`.
No missing output is accepted. The repeated baseline must also reproduce
itself before claiming that a candidate preserves numerical results.
Raw netCDF/checkpoint files are deleted after fingerprinting to bound disk
use. This proves equality of the recorded state and samples, not every
intermediate operation or an arbitrary longer trajectory. Changes with
different hashes are unaccepted exploratory results, even if physically
plausible. The comparison has CPU unit tests covering small changes,
signed zeros, dtype changes, missing state and NaNs.

## Related CPU shard-size follow-up

The older C4D tests did not vary a device mesh. A single logical JAX CPU
device does not imply a single CPU thread: thread-level execution and
explicit device sharding are separate controls. To test the latter, set
both flags before Python starts and vary `N`:

```bash
JAX_PLATFORMS=cpu \
XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_cpu_enable_concurrency_optimized_scheduler=false" \
python -m jcm.main +configuration=t63-echam-rrtmgp \
  grid.spmd_mesh='[4,1,1]'
```

Use the same VM and thread/affinity settings while varying N. Start with
1, 2, 4 and 8 partitions; measure cache/communication tradeoffs rather than
assuming one partition per vCPU is optimal. Preserve whole vertical
columns. Verify that `coords.horizontal.nodal_shape` is still `(192, 96)`
for every setting, because the fast transform may pad an incompatible mesh.

A non-null mesh also selects `FastSphericalHarmonics`; the previous
unsharded path uses `RealSphericalHarmonics`. Include both the original
unsharded baseline and a one-device `[1,1,1]` fast-transform control, then
compare multi-device fast transforms with that control. Otherwise a
transform algorithm change gets misattributed to shard size. Check outputs
and record any rounding differences. See {doc}`parallelization` for the
implementation and existing regression coverage. No CPU speedup from this
follow-up has been measured yet.
