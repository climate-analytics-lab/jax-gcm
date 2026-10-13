# Parallelization & sharding

JAX-GCM runs single-device by default. To split a simulation across
several devices — multiple GPUs, or (the main motivation for this page)
multiple CPU cores — you configure an SPMD **device mesh** once, on the
coordinate system, and the model shards both the dynamical core and the
physics across it.

## TL;DR

```python
from jcm.utils import get_coords
from jcm.physics.echam.echam_levels import get_echam_levels

# Split longitude across 4 devices; keep latitude and level whole.
coords = get_coords(get_echam_levels(47), spectral_truncation=63,
                    spmd_mesh=(4, 1, 1))
```

* The mesh is a `(x, y, z) = (longitude, latitude, level)` tuple; its
  product must equal `len(jax.devices())`.
* **Use a longitude-only mesh `(N, 1, 1)`.** See *Why longitude-only*
  below — it is the layout that costs no extra communication at the
  dynamics↔physics boundary.
* On CPU, expose the cores to JAX *before it initialises* (see
  *Multiple CPU devices*).

## The two layouts

The dynamical core and the physics want data laid out differently, and
this is the crux of how sharding has to work.

* **Dynamical core** — spectral transforms (FFT in longitude, Legendre
  in latitude) couple the whole horizontal plane, so the transform needs
  to communicate across whatever horizontal axis is split. Dinosaur's
  `dycore_partition_spec` is `P('z', 'x', 'y')`: it can use all three
  mesh axes.
* **Physics** — every column scheme (convection, radiation, vertical
  diffusion, …) operates on a *whole vertical column at once* and each
  column is independent of its neighbours. So physics must **never**
  shard the level axis (a split column would force communication inside
  every term) and is happiest sharding the horizontal. Dinosaur's
  `physics_partition_spec` is `P(None, ('x', 'z'), 'y')`: the level axis
  is replicated, longitude carries the `x` (and, if present, `z`) split,
  latitude carries `y`.

These two specs are *different*, so in the general case the model
reshards the gridpoint state at the boundary between dynamics and
physics every step. dinosaur's `with_dycore_sharding` /
`with_physics_sharding` helpers express exactly this.

## Why longitude-only `(N, 1, 1)`

Under a longitude-only mesh the two specs **collapse to the same thing**
— longitude split by `N`, latitude and level replicated — so the
per-step reshard becomes a no-op. Concretely:

* `dycore_partition_spec = P('z', 'x', 'y')` → `P(1, N, 1)` → longitude
  split.
* `physics_partition_spec = P(None, ('x', 'z'), 'y')` → `P(None, N, 1)`
  → longitude split, level replicated.

There is a second reason. Physics flattens the gridpoint state from
`(nlev, nlon, nlat)` to `(nlev, ncols)` with **longitude as the major
axis**. Flattening a sharded major axis (longitude) against a
*replicated* minor axis (latitude) is free — each device keeps a
contiguous block of columns. But flattening *two* sharded axes
(longitude **and** latitude, i.e. `y > 1`) into one axis is not
representable as a contiguous sharding, so XLA inserts a reshard. Keeping
`y = 1` avoids that.

So: split longitude, leave latitude and level alone, unless you have
*measured* a reason to do otherwise. `nlon` is always ≥ 64 (T21), so a
longitude-only mesh scales comfortably to the core counts that matter on
a CPU host.

> **Trade-off not taken (option C).** One could instead keep the dycore
> on `P('z', 'x', 'y')` (level-split transforms, which are
> communication-free) and reshard to physics layout each step — paying
> one boundary transpose to buy a cheaper transform. We deliberately do
> *not* do this by default: it only helps with a 2-D/3-D mesh, needs the
> dycore to support level-batch sharding, and the boundary reshard plus
> the column-flatten reshard can cost more than it saves. The wiring
> below leaves that door open (it is just a different `spmd_mesh`), but
> the supported, tested path is the longitude-only mesh.

## How it is wired

There is no bespoke sharding code in the physics packages — the boundary
is pinned with `with_sharding_constraint` at three points, all no-ops
without an `spmd_mesh`:

1. **`DinosaurDycore.to_physics_state`** pins the gridpoint state to the
   physics sharding as it leaves the dycore.
2. **`ComposablePhysics`** (the `vectorize_columns` path) pins the
   flattened `(nlev, ncols)` column state — and the accumulated
   tendencies before the un-flatten — to the merged column sharding
   (`_flattened_column_sharding` merges the `physics_partition_spec`'s
   longitude and latitude mesh axes, longitude-major). This is what stops
   XLA from silently gathering the columns at the flatten reshape.
3. **`DinosaurDycore.step`** pins the returned tendency to the physics
   sharding before the gridpoint→modal transform reshards it back to the
   dycore layout.

The mesh is built with **`AxisType.Auto`** axes (`jcm.utils.get_coords`),
because `with_sharding_constraint` requires Auto axes; JAX's `make_mesh`
otherwise defaults to Explicit axes and the constraints raise.

## Multiple CPU devices

JAX exposes a **single** CPU device by default no matter how many cores
the host has. The device count must be raised *before the CPU backend
initialises*, so the most reliable way is an environment variable set
before the process starts. Set **two** flags:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=8 \
                  --xla_cpu_enable_concurrency_optimized_scheduler=false"
python -m jcm.main grid.spmd_mesh='[8,1,1]'
```

The second flag serialises CPU collectives. Without it, complex graphs
(notably ECHAM physics) crash at **>= 8 CPU devices** with an XLA-CPU
`collective permute` rendezvous failure (`Check failed: id < num_threads`)
— the spectral transform's concurrent all-to-all ops over-subscribe the
CPU thread rendezvous. It is an XLA-CPU backend limitation, not a
jax-gcm sharding bug (it does not occur on GPU, which uses NCCL
collectives, and simpler graphs such as SPEEDY are unaffected even at 40
CPU devices). The flag is harmless on GPU and for simple models, so the
recipe above always sets it.

The Hydra CLI also accepts `host_device_count`, but importing the model
stack already initialises the JAX backend, so under the CLI it can no
longer *raise* the count — it only **validates** that the running device
count matches and warns (pointing back at `XLA_FLAGS`) if not. So set the
env var, and use `host_device_count` as a guard against forgetting it:

```bash
export XLA_FLAGS=--xla_force_host_platform_device_count=8
python -m jcm.main host_device_count=8 grid.spmd_mesh='[8,1,1]'
```

In a script or notebook, `jcm.runners.configure_host_device_count(8)`
*does* set the count when called as the very first thing after
`import jax` (before importing `jcm`). The `spmd_mesh` product must equal
the device count. On GPU,
`host_device_count` is irrelevant — set `CUDA_VISIBLE_DEVICES` and a mesh
whose product matches the number of visible GPUs.

## A subtlety: the diffusion filter and modal-axis padding

`FastSphericalHarmonics` pads the modal (total-wavenumber) axis with
zeros so it divides evenly across devices. The hyperdiffusion filters
normalise by the largest-magnitude Laplacian eigenvalue, and the old code
read that as `eigenvalues[-1]` — which the padding turns into **0**, giving
`scale = dt / 0 = inf` and NaN-ing every diffused field on the first step
of any multi-device run. The fix (`jcm.diffusion`,
`DinosaurDycore._make_diffusion_fn`) normalises by `max(|eigenvalues|)`
instead, which is the true largest eigenvalue with or without padding.
This is the one substantive change needed to make the *dynamical core*
run multi-device; the spectral transforms themselves shard correctly as
shipped.

## Grid divisibility

The mesh size must divide the grid, or `FastSphericalHarmonics` pads the
nodal grid up to an FFT-friendly multiple — silently changing the
resolution. For a longitude-only mesh `(N, 1, 1)` this means the nodal
longitude count must be `N × (FFT-friendly integer)`. Examples: T31 (96
lon) is fine at N ∈ {2, 4} but pads at N = 8; T106 (320 lon) and T213
(640 lon) are clean at N = 40. If `coords.horizontal.nodal_shape` changes
when you add the mesh, pick a different N (or resolution).

## Validation

Verified that a sharded run reproduces the single-device result of the
*same* (`FastSphericalHarmonics`) algorithm — the apples-to-apples test
that isolates sharding from the `RealSphericalHarmonics` ↔
`FastSphericalHarmonics` algorithm difference:

- **2× A100 GPU**, SPEEDY and ECHAM T63: a single sharded step is
  **bit-identical** to the single-GPU step (`rel = 0`). Over a multi-step
  run the two diverge at ~1e-3, which is chaotic amplification of GPU
  non-deterministic reductions, not a sharding error.
- **40× CPU**, SPEEDY T106: matches single-device to `rel ≈ 9e-6`.
- **40× CPU**, ECHAM T106: matches once the serialized-collective flag is
  set (see *Multiple CPU devices*).

`jcm/dycore/dinosaur/sharding_test.py` pins the full-model case (a 2-CPU
SPEEDY integration must stay finite and match the single-device run);
`jcm/physics/composable_physics_sharding_test.py` pins the physics path;
`jcm/diffusion_test.py` pins the modal-padding eigenvalue fix.

## Throughput scaling in practice

Reproducing the single-device answer (above) is correctness; it says nothing
about *speed*. This section records measured throughput so you can decide
when multi-device is worth it. The numbers below are from an **8× A100 80 GB
PCIe** host (**no NVLink** — inter-GPU collectives go over PCIe), full ECHAM +
RRTMGP physics, longitude-only mesh `(N, 1, 1)`. Treat them as
order-of-magnitude guidance for *this* class of interconnect, not portable
constants.

**The headline: the multi-GPU benefit grows with horizontal resolution, and
there is a crossover.** Every spherical-harmonic transform needs a global
all-to-all (longitude is split, so the FFT must communicate). At small grids
that fixed communication dominates the modest compute per step, so adding
devices *loses*. As the grid grows, compute per step rises faster than the
communication, and 2-way sharding crosses into a real win:

| grid (lon×lat)     | 2-GPU speedup | 4-GPU speedup |
| ------------------ | ------------- | ------------- |
| T63   (192×96)     | 0.95×         | 0.94× (8-GPU 0.53×) |
| TL127 (256×128)    | 0.92×         | 0.81×         |
| TL179 (360×180)    | 1.08×         | 0.88×         |
| TL255 (512×256)    | **1.79×** (89 % eff) | 1.46×  |

Speedups are per-grid versus that grid's single-GPU run. By **TL255** two
GPUs deliver a **near-linear 1.79×**; four stays positive (1.46×) but below
two — past two devices the PCIe all-to-all reasserts itself. So on this box:

* **Low resolution (≤ TL127): run single-GPU.** Multi-GPU only buys *capacity*
  (a model too big for one card), not speed.
* **High resolution (TL255+): two GPUs are worth it.** Four or more would need
  a faster interconnect (NVLink / NVSwitch, i.e. A100 SXM or H100) to pay off.

### Why not split the level axis instead? (option C)

A natural idea is to shard the *dynamics* by level (`(1, 1, N)`, the
`dycore_partition_spec = P('z', 'x', 'y')` layout): the spherical-harmonic
transform is independent per level, so a level split makes it
communication-free. It runs — the semi-implicit solve and the physics
boundary tolerate it — but it does **not** beat the longitude split. Measured
on SPEEDY T106 L8 (same dycore, cheap physics, so it isolates the dynamics):
2-GPU `B = 0.96×` vs `C = 0.89×`; 4-GPU `B = 0.71×` vs `C = 0.72×`. The freed
transform cost simply reappears elsewhere: (a) the physics↔dynamics boundary
becomes an all-to-all *transpose* (level-split dynamics is the transpose of
column-local physics), and (b) the dynamics couples levels (the geopotential
integral and semi-implicit solve), so splitting `z` adds communication inside
the dynamics too. It also can't shard a prime level count — `(1, 1, 2)` on the
ECHAM L47 grid fails outright (`47` is not divisible). The wiring leaves the
door open (it is just a different `spmd_mesh`), but option B is the supported,
faster path.

### Caveats

* **Interconnect-specific.** All of the above is dominated by PCIe collective
  cost. NVLink/NVSwitch would shift every crossover point and likely make
  4-GPU and level-sharding viable.
* **Cross-grid absolute throughput is not apples-to-apples** if the RRTMGP
  implementation changed between runs (e.g. the single-vmap refactor roughly
  doubled single-GPU TL255 throughput). The *per-grid* speedup ratios are
  valid regardless — numerator and denominator use the same code — but do not
  compare raw sim-days/wall-day across a code change.
* Benchmark harness: `run_logs/benchmark_rrtmgp_scaling.py` (`--grid TL127`
  etc. for the linear grids); the dynamics-layout A/B test is
  `run_logs/benchmark_speedy_scaling.py`.
