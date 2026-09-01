# Vertical conventions and CF metadata in JCM output

## One direction per file

A JCM output netCDF has two vertical dimensions:

| dimension | length | holds |
|---|---|---|
| `level`   | `nlev`   | layer mid-level fields — temperature, tracers, `pressure_full`, heating rates |
| `level_i` | `nlev+1` | layer interface fields — `pressure_half`, `height_half`, radiative fluxes |

**Both run surface-first.** Index 0 of either axis is the level nearest the
ground; the sigma coordinate decreases with index towards the model top. The
two axes interleave in the obvious way: `level[k]` lies between `level_i[k]`
(below it) and `level_i[k+1]` (above it), so

```python
dp = -np.diff(ds.pressure_half, axis=iface_axis)   # > 0, aligned with `level`
burden = (tracer * dp / g).sum("level")            # correct as written
```

needs no orientation guard. This is the property that makes the file usable:
pairing an interface quantity with a mid-level one is the natural thing to
write, so it has to be the correct thing to write.

Better still, the ECHAM physics stacks emit the layer thickness directly on
the `level` axis as `pressure_thickness` [Pa] (positive), so the mass-weight
needs no `diff` and no interface axis at all:

```python
burden = (tracer * ds.pressure_thickness / g).sum("level")   # aligned by construction
```

`pressure_thickness` is the model's own `diff(pressure_half)` (computed in
`MoistAirColumnState`), so it lands on `level` and cannot be mis-paired with
the interface axis — the exact trap the `diff` form invites. It is present
only where the moist-air prepare term runs (the ECHAM stacks, `rce.py`), not
in SPEEDY output; for a file without it, fall back to the `diff` form above.

Two other frames exist and are *not* this one, which is why the conversion is
centralised rather than done per-variable:

- the **physics-internal** frame is TOA-first (`a_boundaries[0] = 0` gives
  `p = 0` at index 0), and so is every intermediate array inside a scheme;
- the **ECHAM/HAMMOZ input files** are TOA-first as well (`hybm[0] = 0`).

`jcm/cf_metadata.py` is the single place where the internal frame becomes the
file frame. Every dycore backend's `to_xarray` ends in
`cf_metadata.finalize_output(...)`; no other code flips a vertical axis on the
way out.

### Why centralised

The two axes used to be flipped independently, which meant they could — and
did — disagree. `ModelPredictions._trajectory_dataset` reversed `level` inline
and left `level_i` in the TOA-first frame it came out of the physics in, so
`pressure_full` and `pressure_half` in the *same file* ran in opposite
directions in every column ([#710]). Nothing raised: a heating rate computed
from the saved fluxes and compared against the saved temperature, or a
mass-weighted tracer burden, simply came out vertically reversed with a
plausible magnitude.

## Self-describing metadata

Orientation used to be discoverable only by reading pressure values off each
end of each axis and comparing. The file now states it:

- **`level` / `level_i` are real coordinate variables**, carrying nominal sigma
  `a/p0 + b` (mid-levels and interfaces respectively, from the same hybrid
  table), plus `standard_name = atmosphere_hybrid_sigma_pressure_coordinate`,
  `units = "1"`, `axis = "Z"` and `positive = "down"`. Pure-sigma grids are the
  `a = 0` case of the same formula, so one standard name covers both coordinate
  families.
- **The hybrid tables travel with the file** as the coordinate variables
  `hybrid_a_full` / `hybrid_b_full` (on `level`) and `hybrid_a_half` /
  `hybrid_b_half` (on `level_i`), written in the file's own surface-first
  order. `p = a + b·p_s` is therefore reproducible from the file alone, and CF
  `formula_terms` names them — emitted only when `surface_pressure` is
  actually present, so a dynamics-only file has no dangling reference.
- **`pressure_*` / `height_*` / the core prognostics** carry `standard_name`
  and `units`.

`positive` describes the direction in which the coordinate *values* increase
(CF-1.11 §4.3), not the storage order — a sigma coordinate is `positive =
"down"` whichever way it is stored. The storage order is legible from the
coordinate values themselves (they descend from ~1 towards 0) and is stated in
`long_name` for anyone skimming `ncdump -h`.

`positive` is deliberately **not** set on `pressure_full` / `pressure_half` /
`height_*`: CF defines it for vertical coordinate variables, and those are data
variables. Their orientation is that of the axis they sit on, and both axes now
agree.

## Reading states back

`jcm.utils.load_states_from_xarray` is the reader-side counterpart: it detects
a surface-first file from the `level` coordinate *values* (descending sigma)
and always returns a `PhysicsState` in the top-first physics frame ([#741]).
`PrescribedStateModel` output follows the file convention like every other
product ([#739]), so the pair round-trips. Physics diagnostics beyond the
vertical-coordinate neighbourhood carry their own units and standard names via
`PhysicsTerm.output_attrs`, declared next to the code that computes them
([#740]).

## Reading pre-#710 files

Output written before this change has `level_i` as a bare integer index with
no attributes, and interface variables stored TOA-first. The presence of
`level_i.attrs["positive"]` distinguishes the two; a pre-#710 file needs its
interface variables reversed before being paired with anything on `level`.

[#710]: https://github.com/climate-analytics-lab/jax-gcm/issues/710
[#739]: https://github.com/climate-analytics-lab/jax-gcm/issues/739
[#740]: https://github.com/climate-analytics-lab/jax-gcm/issues/740
[#741]: https://github.com/climate-analytics-lab/jax-gcm/issues/741
