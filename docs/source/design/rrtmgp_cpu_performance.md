# RRTMGP radiation on a CPU: cache-bound, solved in column blocks

On a CPU the RRTMGP call of an ECHAM model (`t63-echam-2m`, T63L47, 18,432
columns) is bound by cache, not by arithmetic. This page records what
`RRTMGPRadiation` does about it, why, and the measurements behind it. The
accelerator side of the same call is in {doc}`rrtmgp_gpu_performance`; the
two differ in the one place that matters here, the column loop.

All timings are float32, `compute_cre=True`, `gas_optics_mode="all_gpts"`,
`gpt_chunk=64` with `use_scan` (the shipped preset) unless stated, one radiation
call on a saved day-3 `t63-echam-2m` state, on a shared 128-core host with
the benchmark pinned to 32 cores (or one); earlier runs of the single vmap on the same host took 7.9-8.5 s,
so differences of that size are host load. Times are the minimum and median of 3 timed
calls after the compiling call; "core-ms / column" is the minimum times the
cores, over the columns.

## Why the single vmap is slow on a CPU

A solve vmapped over `ncols` columns materialises arrays of
`n_gpt x ncols x nlev` between its stages (gas optics, cloud and aerosol
combination, two-stream coefficients, the level recurrences). Their size, not the
flop count, sets the speed on a CPU: the per-column cost on one core is 2.7 ms
when at most 8 columns are solved per call, 5.2 ms at 384 columns or more, and 13.7 core-ms
when 24 CPU devices that share a socket's cache each solve their slice of the
model's column axis. On a GPU the same program is launch- and gather-bound, so
the whole batch in one kernel is the right shape there and the CPU conclusion
does not transfer.

## What the code is

`RRTMGPRadiation(column_block=...)` selects the column loop (static, no effect
on the physics; Hydra `physics.terms.rrtmgp_radiation.column_block`, or
`physics.radiation_column_block` on the factory-built presets):

- an integer B: `lax.map` over scan steps that each solve B columns per device;
- `None`: one `jax.vmap` over all columns;
- `"auto"` (default): B = 8 when the columns are on CPU devices, `None` otherwise.
  "On CPU" is the platform of the model's SPMD mesh devices
  (`flattened_column_shard_count`, read in `cache_coords`), and
  `jax.default_backend()` when there is no mesh. The GPU program is therefore the
  single vmap it has always been.

**Strided layout.** Physics flattens the grid to one column axis that the model
shards contiguously over the mesh devices. Mapping `lax.map` over contiguous
blocks of that axis puts each block on one device and the scan runs the devices
one after another: the same call took 65 s against 8 s for the single vmap. The
blocks are therefore laid out `(ncols) -> (n_shards, steps, B) -> (steps,
n_shards, B)` (`_split_columns`): the `n_shards` axis stays the sharded one, and
every scan step takes B columns from every device. The body is a doubly vmapped
solve, with no cross-device traffic: the blocks add no collective to the
program (on a longitude x latitude mesh the flatten of the two sharded axes
still reshards once, as it does for the single vmap).
Without a mesh, `n_shards = 1` and the blocks are the contiguous ones.

**Block resolution** (`resolve_column_block`). The per-device count
(`ncols // n_shards`) must be tiled exactly, so B is lowered to its largest
divisor that is <= B; when that is the whole per-device count (a single column,
a tiny batch, a test) there is nothing to scan and the program is the single
vmap. A prime per-device count above B gives B = 1.

**Memory path.** `JCM_RRTMGP_COL_CHUNKS=n` (the reverse pass's rematerialised
blocks, {doc}`rrtmgp_gpu_performance`) uses the same layout helper with `n` steps and a
`jax.checkpoint` around the body, so under a mesh its blocks are strided too, and `ncols`
must be divisible by `n_shards * n`. When both are set the environment variable
wins: it is the explicit memory request, and its guarantee (peak falls like `1/n`) belongs to
the step count it names, which a per-device block size does not fix.

## Measurements

Through the real helper (`rrtmgp._maybe_chunked_vmap` with `n_shards` and
`column_block`), 18,432 columns over 24 CPU devices, 32 cores, column axis
sharded contiguously, one RRTMGP call:

| column loop | min s / call | median s / call | core-ms / column | max RSS | max flux difference vs single vmap |
|---|---|---|---|---|---|
| single vmap (`None`) | 9.12 | 9.14 | 15.8 | 18.6 GB | reference |
| B = 4 | 2.70 | 2.94 | 4.68 | 3.4 GB | 3.1e-3 W/m2 |
| **B = 8** | **2.78** | **2.80** | 4.82 | 3.5 GB | 3.1e-3 W/m2 |
| B = 16 | 3.57 | 3.74 | 6.20 | 3.7 GB | 3.1e-3 W/m2 |
| B = 32 | 4.88 | 5.05 | 8.46 | 3.7 GB | 3.1e-3 W/m2 |
| B = 8, `gpt_chunk=16` | 2.13 | 2.26 | 3.69 | 3.4 GB | 3.1e-3 W/m2 (heating 1.8e-6 K/s) |

One device, one core, 768 columns (what one device of the 24-device layout holds):

| column loop | min s / call | median s / call | core-ms / column | max flux difference vs single vmap |
|---|---|---|---|---|
| single vmap (`None`) | 3.95 | 3.96 | 5.14 | reference |
| B = 4 | 1.85 | 1.86 | 2.41 | 6.1e-5 W/m2 |
| B = 8 | 1.97 | 1.97 | 2.56 | 6.1e-5 W/m2 |
| B = 16 | 2.48 | 2.49 | 3.23 | 6.1e-5 W/m2 |
| B = 32 | 2.55 | 2.55 | 3.32 | 6.1e-5 W/m2 |

The sharded fluxes agree with the single vmap to <= 3.1e-3 W/m2 (on fluxes of
order 1e3 W/m2, a few float32 ulp from the different fusion), and the heating rates to
1.2e-6 K/s. The optimum is flat between B = 4 and 8 and rises beyond 8 because the working
set leaves cache: B = 8 is the default. The measurement came from a standalone
harness, with the model's input columns, that calls the same helper; the
whole-model effect is in the next section.

## In the whole model

`t63-echam-2m` on the level mesh `[1, 1, 24]` ({doc}`parallelization`), 32 cores
of one socket, seconds of wall time for the third simulated day of a run (the
first two carry compilation), one job per socket on a shared host:

| column loop | socket 0 | socket 1 | max RSS |
|---|---|---|---|
| single vmap (`column_block=null`) | 190 | 218 | 21.9 GB |
| `auto` (B = 8) | 146 | 158 | 11.6 GB |
| `auto`, `gpt_chunk=16` | 146 | 157 | 11.7 GB |

The blocks take 23-28 % off the model's wall time, about what twelve radiation
calls per simulated day at the per-call saving above predict. The single vmap on
socket 0 was run from the tree without this option, on socket 1 with
`column_block=null`.

## `gpt_chunk` on a CPU

`gpt_chunk=16` (g-points solved per spectral-loop iteration) is faster in the
standalone harness: 2.78 -> 2.13 s per call at 24 devices with B = 8, with the
same flux agreement against the single vmap at `gpt_chunk=64` as above. In the
whole model the difference is not measurable (table above), so the shipped
presets keep `gpt_chunk=64`, the A100 optimum ({doc}`rrtmgp_gpu_performance`),
on every backend. The override is `physics.terms.rrtmgp_radiation.gpt_chunk`
(`physics.radiation_gpt_chunk` on the factory-built presets).

## What was not measured

Block sizes other than the
`auto` one were measured only at the resolution above (T63L47, 47 levels); the
working-set argument (`n_gpt x B x nlev`) says the optimum moves with `nlev`, and
the option is exposed for that reason.
