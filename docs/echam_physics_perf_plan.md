# ECHAM physics performance plan

This is a staged plan to close the ~10x performance gap between ECHAM physics
and SPEEDY physics for the same grid. Profiling on T21 × 10 levels (2048
columns), CPU:

```
SPEEDY full compute_tendencies: ~1.6 ms / step
ICON   full compute_tendencies: ~23.7 ms / step
```

The sub-stepping work on `feature/radiation-substepping` handles the cost of
RRTMGP (which is called every N steps instead of every step). This plan
tackles the *remaining* ~10× gap that shows up independently of the radiation
scheme, i.e. the cost we still pay on non-radiation steps.

The three steps are independent and additive:

| Step | Target                                                         | Risk   |
| ---- | -------------------------------------------------------------- | ------ |
| 1    | Cheap local fixes: redundant work, per-term inefficiencies     | Low    |
| 2    | Move vmap-over-columns from inside each term to the outer loop | Medium |
| 3    | Fortran-style NPROMA column batching (`lax.map(vmap(fn))`)     | Medium |

Step 1 is purely mechanical and should land first. Steps 2 and 3 are more
invasive and must be benchmarked against step 1 before committing.

---

## Step 1 — Cheap local fixes (this branch)

These are all local perf wins that don't change the physics-dynamics
interface. Each sub-item is independent and can land on its own.

### 1.1 Remove vmap-of-1 wrapping in `apply_vertical_diffusion`
**File:** `jcm/physics/echam/echam_physics.py`

The old implementation wrapped each column in a helper `apply_vdiff_to_column`
that ran through `jax.vmap`, even though `vertical_diffusion_column` (and all
of `turbulence_coefficients.py`) already supports batched `(ncol, nlev)`
arrays — the word "column" in the name refers to *column format*, not to
*one column at a time*.

Fix: call `prepare_vertical_diffusion_state` and `vertical_diffusion_column`
directly on transposed `(ncols, nlev)` arrays. Drop the wrapper vmap and the
`thv_variance` `hasattr` check (`thv_variance` isn't stored on
`VerticalDiffusionData`, so it was always zeros anyway).

### 1.2 Drop dead chemistry init from `_prepare_common_physics_state`
**File:** `jcm/physics/echam/echam_physics.py`

`_prepare_common_physics_state` called
`initialize_chemistry_tracers(...)` every step, which ran Python-level loops
to build realistic ppbv profiles for ozone, CH₄ and CO₂. The very next term in
the sequence, `apply_forcing_data`, *unconditionally* overwrites
`physics_data.chemistry` with uniform constants:

```python
chemistry_data = physics_data.chemistry.copy(
    co2_vmr=jnp.ones_like(...) * 420.0,
    methane_vmr=jnp.ones_like(...) * 1900.0 * 1e-3,
    ozone_vmr=jnp.ones_like(...) * 300.0 * 1e-3,
)
```

So the realistic init was immediately discarded. Delete the init block in
`_prepare_common_physics_state` and remove the now-unused
`initialize_chemistry_tracers` import. Zero behavior change, pure win.

### 1.3 Dedupe aerosol `spatial_dist` computation
**File:** `jcm/physics/echam/aerosol/simple_aerosol.py`

The MACv2-SP plume Gaussian distribution was recomputed three times per step:
once inside `get_anthropogenic_aod`, once inside `get_background_aod`, and
once at the top of `get_simple_aerosol`. Compute it once and pass it through.

Signature changes (internal, tests updated):
```python
get_anthropogenic_aod(parameters, year_weight, ann_cycle, spatial_dist)
get_background_aod(parameters, ann_cycle, spatial_dist, constant_background=0.02)
```

### 1.4 Drop `max_bands = 10` padding in the grey radiation scheme
**Files:** `jcm/physics/echam/radiation/two_stream.py`,
`jcm/physics/echam/radiation/radiation_scheme.py`,
`jcm/physics/echam/radiation/gas_optics.py`

`longwave_fluxes` and `shortwave_fluxes` were vmapped over a fixed
`max_bands = 10` buffer with a mask that zeroed out bands ≥ `n_bands` (= 3 for
LW, 2 for SW in the grey scheme). That means ~70% of the two-stream work —
layer reflectance/transmittance, the LW downward/upward flux scans, and the
SW direct-beam + diffuse scans — was computed and thrown away.

Fix: make `n_bands` a static arg via
`functools.partial(jax.jit, static_argnames=('n_bands',))`, vmap over exactly
`n_bands` and drop the mask. The flux shape becomes `(nlev + 1, n_bands)`
instead of `(nlev + 1, 10)`; tests that hardcoded `10` are updated.

Also:
- `radiation_scheme.py` had a matching `max_sw_bands = 10` TOA-flux buffer
  that can be replaced with a direct allocation of shape `(default_n_sw_bands,)`.
- `gas_optics.create_gas_optics` and its `test_gas_optics` were dead code that
  built `(nlev, 10)` SSA/asymmetry arrays mismatched with the `(nlev, 8)` tau
  arrays. Delete entirely.

### 1.5 Vmap the band loop in `gas_optical_depth_lw` / `gas_optical_depth_sw`
**File:** `jcm/physics/echam/radiation/gas_optics.py`

Both functions had a Python `for band in range(N_BANDS)` loop that staged
`N_BANDS` separate `.at[:, band].set(...)` updates into XLA, producing an
unrolled dependency chain. Replace with a single `jax.vmap` over
`jnp.arange(N_BANDS)` and a final transpose. Same numerics, shorter HLO,
better fusion.

### 1.6 Re-run profiler
**File:** `/tmp/profile_icon.py`

Re-run the per-term profiler and the full-step profiler and compare against
the SPEEDY baseline. Note the new per-step number in the PR description and
call out which terms moved most.

---

## Step 2 — Move vmap-over-columns to the outer compute_tendencies loop

**Files:** `jcm/physics/echam/echam_physics.py`, all `apply_*` terms

Currently each term (`apply_convection`, `apply_clouds_and_microphysics`,
`apply_radiation`, etc.) vmaps over columns internally. Each per-term vmap is
a separate jit boundary that XLA cannot fuse across — so we pay the cost of
realizing intermediate `(nlev, ncols)` buffers at each term boundary and lose
cross-term fusion opportunities (e.g. convection → microphysics, which share
temperature/humidity profiles).

### Plan

1. Write each `apply_*` term as a single-column function that takes
   `PhysicsState[nlev]` and `PhysicsData[nlev]` and returns per-column
   tendencies/data. Most terms already *do* single-column work inside their
   own vmap — this is mostly unwrapping the vmap.
2. In `compute_tendencies`, call all of the terms inside one
   `jax.vmap(column_compute_tendencies, in_axes=(1, ..., 1), out_axes=1)` that
   threads state → tendencies → data through the whole sequence at the column
   level.
3. Surface terms and gravity-wave drag may need special handling if they
   currently rely on cross-column state (they don't appear to, but double
   check).

### Why this might help

- One fused `vmap` gives XLA a single giant kernel to optimize instead of
  ten kernels with forced materializations between them.
- Per-term `@jit` boundaries become dead weight; drop them.
- Intermediate `PhysicsData` field writes fuse into column-local registers
  instead of round-tripping through `(nlev, ncols)` global memory.

### Risks / caveats

- Sub-stepping already relies on physics data being carried across steps
  (radiation tendency cache in `apply_radiation`). That works at the outer
  level so should be unaffected, but the outer-vmap shape has to be compatible
  with the NNX `DiagnosticsCollector.physics_data_cache` mechanism.
- If XLA decides the fused kernel is too register-heavy it can spill and
  *hurt* perf. We should benchmark this against step 1 before committing, not
  just trust the theory.
- `apply_surface` creates zero tendencies at all levels except the lowest —
  its cross-term dependency on the ambient profile is limited to the bottom
  cell, which should survive the outer-vmap rewrite but watch for broadcast
  bugs.

### Verification

- Before/after per-step timings on the T21 × 10 lvl profiler.
- Numerical parity against step 1 to machine precision (or tight `atol` /
  `rtol`).
- Full `jcm/physics/echam/` test suite, `-m "not slow"` and slow.

---

## Step 3 — NPROMA-style column batching with `lax.map(vmap(fn))`

**Files:** `jcm/physics/echam/echam_physics.py` (thin wrapper over the step 2
outer vmap)

The Fortran version of ICON batches columns in *blocks* of size `NPROMA` (~32
to 128) to fit a block in cache. The JAX equivalent is
`jax.lax.map(jax.vmap(column_fn), columns)` — `vmap` vectorizes within a
block, `lax.map` iterates over blocks sequentially without unrolling into a
giant HLO trace.

### Plan

1. Take the outer vmap from step 2 and wrap it in a chunked driver:

   ```python
   def chunked_compute(state, ...):
       # Split columns into NPROMA-sized chunks, pad the last.
       chunks = _split_into_chunks(state, nproma)
       per_chunk = jax.vmap(column_compute_tendencies, in_axes=(1, ...), out_axes=1)
       return jax.lax.map(per_chunk, chunks)
   ```

2. Make `nproma` a `static_argnum` knob on `EchamPhysics.__init__` (defaulting
   to "no chunking" i.e. one big block, same as step 2).
3. Tune `nproma` empirically: start at 32, sweep [16, 32, 64, 128, 256] on
   both CPU and one device type representative of the target GPU/TPU and pick
   the sweet spot.

### Why this might help

- Better L1/L2 cache locality — a block of 32 columns fits in cache; 2048
  columns don't.
- Smaller HLO program size → faster compile, cheaper autodiff.
- On multi-device setups, the block dimension can become the sharded axis
  more naturally than the full-column axis.

### Risks / caveats

- `lax.map` introduces a scan-like control flow that XLA may not fuse as
  aggressively as a straight `vmap`. On some workloads this is a loss.
- Padding the last chunk wastes up to `nproma - 1` columns' worth of work.
  With `ncols = 2048`, `nproma = 32` gives a clean divide; `nproma = 48`
  would need padding.
- Gradient pass through `lax.map` has more overhead than through `vmap`.
  If training the model end-to-end is a goal, benchmark the backward pass too.
- Autoparallelism / sharding interactions are subtle. If we already shard
  columns over devices via `shard_map`, adding `lax.map` inside that is
  fine but worth verifying on a 2-device run.

### Verification

- Sweep `nproma` on CPU (benchmark above) and on the target device if
  available.
- Compare forward perf and backward (gradient) perf independently.
- Full ECHAM physics test suite.

---

## Open questions that might invalidate parts of this plan

- **Is the RRTMGP vmap the same story as the two-stream vmap?** RRTMGP has
  its own per-column implementation that's already vmapped inside
  `_apply_radiation_rrtmgp_inner`. Step 2's outer-vmap rewrite would include
  it — worth checking that the RRTMGP column function is actually safe to
  vmap at the outer level (it is, but let's verify the g-point allocations
  don't balloon).
- **Does XLA actually fail to fuse across `@jit` boundaries here?** Testing
  assumption is based on HLO inspection of `_apply_radiation_inner` vs
  `_apply_clouds_and_microphysics_inner`, each of which generates a separate
  compiled kernel. If the cost is actually dominated by *one* term (e.g.
  gas_optics), step 2 is a weaker win than expected.
- **Is step 3 net positive on GPUs?** `lax.map` on GPU historically produces
  inner loops that don't get the same fusion treatment as vmap. Get a
  benchmark on the actual target device before committing to chunking.
