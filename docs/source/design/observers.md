# Virtual observation operators (`jcm.observers`)

Comparing the model against in-situ measurements (station networks, ship and
aircraft campaigns) or satellite curtains needs model fields **at the
observation's location and time** — not at the nearest `save_interval` frame
of gridded output. `jcm.observers` provides sampling operators that run at
**every model timestep** inside the integration scan and emit their samples on
a dedicated per-`dt` output channel.

Because the whole path is JAX-differentiable, an `Observer` is also an
observation operator H(x): gradients of `‖H(x) − y_obs‖` flow back into
physics parameters, so the same machinery serves data assimilation and
parameter calibration against point observations.

## User API

```python
from jcm.observers import TrackObserver, LocalSolarTimeObserver

flight = TrackObserver.from_dataframe(
    df,                                   # time / latitude / longitude / altitude
    variables=("temperature", "so4", "cloud_fraction"),
    name="atom_f05",
)
stations = TrackObserver.stations(
    latitudes=[47.55, -15.94], longitudes=[7.98, 345.6],
    altitudes=[3580.0, 2160.0],           # Jungfraujoch, Cape Verde-ish
    variables=("so4", "temperature"), name="gaw",
)
swath = LocalSolarTimeObserver(
    variables=("cloud_fraction",), local_solar_hour=13.5, name="a_train",
)

model = Model(coords=..., physics=..., observers=[flight, stations, swath])
preds = model.run(...)
datasets = preds.observation_datasets()   # {name: xr.Dataset (time[, level], point)}
```

Variables resolve against the physics diagnostics dict: top-level diagnostic
keys (`"cloud_fraction"`), dotted sub-struct fields (`"radiation.tsr"`), and —
through the `StateSampler` term the `Model` appends automatically — the state
fields `temperature`, `u_wind`, `v_wind`, `specific_humidity`,
`surface_pressure`, `z_full`, `p_full`, and every `state.tracers` entry by
name (so JAM aerosol tracers sample directly).

Vertical modes: `"altitude"` (linear in height against the sampled `z_full`
profile; heights above the geoid), `"pressure"` (linear in log-p), `"surface"`
(2-D fields only), `"profile"` (whole columns — e.g. for satellite curtains).

In `"profile"` mode the sampler returns columns in the top-first physics frame
(`preds.observations` is raw), and `to_dataset` reverses them so the emitted
`level` axis is **surface-first**, carrying the same sigma coordinate and CF
attributes as the trajectory file. A curtain and the gridded output can
therefore be compared level-for-level without an orientation check — see
[output_vertical_conventions](output_vertical_conventions.md).

## Design decisions

**Time is snapped to the model timestep in the scan; exact obs times are a
post-processing interpolation.** Platform positions are resampled onto the
`dt` grid offline (numpy) at `prepare()` time, so every scan step samples a
statically-shaped set of points. The alternative — exact obs times inside the
scan — implies ragged per-step point counts, which JAX cannot trace
efficiently. Interpolating the dt-resolution output series onto exact
observation times afterwards is cheap xarray work and completes the exact
(lat, lon, alt, t) sampling.

**Horizontal weights are precomputed and cached; only the vertical
interpolation is state-dependent.** On separable grids (dinosaur:
Gaussian latitudes × uniform longitudes) weights are true bilinear with
longitude wraparound and pole clamping; on unstructured column grids (pySES
PG2) they are k-nearest inverse-great-circle-distance (SE basis evaluation is
a possible upgrade). Fixed stations resolve their geometry once; moving
tracks/swaths get per-step tables. The tables enter the scan as `xs`
(`(n_steps, npts, k)`), so nothing horizontal is recomputed in traced code.

**Sampling lives in the inner scan, not in a PhysicsTerm.** The op-split
trajectory decimates everything to `save_interval` (snapshot) or averages it
(averaged mode); a diagnostics-dict entry cannot survive at `dt` resolution.
The observers therefore hook `_op_split_trajectory`'s inner scan and emit
their samples as scan `ys` — the only per-`dt` channel — while regular output
is untouched. Samples are taken from the post-step diagnostics dict, i.e.
from the state the physics *saw* at the start of each `dt`.

**State fields travel via `_sampler_state`.** The `StateSampler` term
(zero tendencies, broadcasting-native) publishes state fields and the
vertical coordinates into the diagnostics dict under `"_sampler_state"`.
The key rides the scan carry (structure stability) but is stripped from
saved trajectory frames and from `to_xarray()`, so gridded output does not
duplicate the dynamics fields.

**Masking is NaN-out, weights stay finite.** Timesteps outside a track's
time window have their weights zeroed (finite gather) and the sampled value
replaced with NaN through `jnp.where`, keeping gradients clean.

## Chunked runs

`prepare()` is called per `run`/`resume` window with the window's absolute
start time (days since 1970 — the same axis `ModelPredictions.to_xarray`
uses), so a track spanning several chunks is sliced consistently and
`run(N) + resume(M)` reproduces a single `run(N+M)` bitwise (covered by
tests). Note the per-step tables scale as `n_steps × npts × k`; for very long
single windows with many points, prefer the chunked driver pattern anyway.

## Caveats / follow-ups

- Times are interpreted on the model's output time axis; with the
  `"365_day"` calendar a real (Gregorian) campaign date drifts across long
  runs — use `calendar="gregorian"` for real-campaign comparisons.
- Observers are fixed at `Model` construction (the jitted runner treats the
  Model as static; mutating `model.observers` later won't retrace).
- Fields with extra trailing axes (per-band optics) are not sampleable.
- Under SPMD sharding the neighbour gather is a cross-device gather; cheap
  at realistic point counts but not free.
- Possible extensions: SE basis-function weights on pySES grids, along-track
  averaging kernels (satellite footprints), and a Hydra config hook.
