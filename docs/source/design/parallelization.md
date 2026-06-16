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
before the process starts:

```bash
export XLA_FLAGS=--xla_force_host_platform_device_count=8
python -m jcm.main grid.spmd_mesh='[8,1,1]'
```

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

## Known limitation: dycore transform on >1 device

The physics sharding described here is independent of, and tested
separately from, the dynamical core's spectral-transform sharding. The
dinosaur spectral transform currently trips over a `shard_map` strict
spec check on its (replicated, numpy-constant) basis matrices, which a
multi-device run must work around by pre-sharding those constants (see
the documented monkey-patch in `run_logs/benchmark_rrtmgp_scaling.py`).
Folding that fix into jax-gcm proper is tracked separately; until then,
full end-to-end multi-device *model* runs need that workaround even
though the physics path itself shards cleanly.
