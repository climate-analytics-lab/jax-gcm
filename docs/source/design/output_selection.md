# Choosing which variables a run writes

A run's trajectory output is every prognostic plus every diagnostic the
composed physics publishes: about 130 for the ECHAM package, and one 3-D field
is roughly 10 GB per simulated year at T63L95 with 6-hourly output. A caller
that wants a handful would otherwise pay to accumulate, stack, transfer and
serialize all of them. `run.output_keep` and `run.output_drop` (and the
`output_keep=` / `output_drop=` arguments of `Model.run`, `Model.resume`,
`Model.run_from_state` and `Model.run_from_state_with_carry`, which the
runner forwards to) choose what is written.

## The selector

* `output_keep`: `None`/`null` (default) writes everything; otherwise only the
  listed names. An empty list is refused (use `null`).
* `output_drop`: names never written, applied after `output_keep`, so a name
  in both is dropped.
* Names are the data-variable names of the output Dataset
  (`temperature`, a tracer's name, `radiation.toa_sw_up`,
  `surface_exchange.tile_fraction.0`). `Model.output_variable_names()` lists
  them for a model. An integer or boolean diagnostic is named but, as for any
  interval-mean output, omitted from averaged files.
* Both unset is the unchanged output: the same compiled program and the same
  Dataset. A selection removes outputs from the program XLA compiles, which
  changes its fusion, so a selected run agrees with the unselected one to
  round-off (last-ulp differences in the saved fields, growing as any
  perturbation does in a chaotic integration) and not bit for bit. It is
  bit-identical to itself across checkpoint and resume.
* The coordinates (`lon`, `lat`, `time`, `level`, the hybrid coefficients) and
  `time_bounds` are always written. A keep-list does not otherwise retain
  `pressure_full` / `pressure_half`: list them if the file should carry them.
  `run.pressure_levels` interpolates with `pressure_full`, so with it set a
  keep-list retains `pressure_full` implicitly and a drop-list naming it is
  refused.
* An unknown name raises `ValueError` listing the closest matches and every
  available name. The names are known as soon as the initial state and physics
  carry exist (the dynamics' from the state, the diagnostics' from the carry,
  which has the structure of the diagnostics the scan returns), so the check
  runs before the integration is compiled; the chunked runner repeats it before
  its first chunk and before any checkpoint is read.

## Where it acts

The selection is applied to the *saved copy*, before the output accumulation,
in `jcm.output_selection`:

| output | pruned |
| --- | --- |
| physics diagnostics | `make_physics_pruner`, applied where `Model` strips the never-written carry slots, to the template that seeds the running mean and to every step's diagnostics. A dict entry no selected name comes from is removed; a typed sub-struct (`radiation`, `surface_exchange`, ...) loses its unselected fields (they become `None`, which the flattening already omits); an entry that flattens to no output name (carry plumbing such as `_prev_step`) is removed. The plan is read from the physics' own `data_struct_to_dict` under `jax.eval_shape`, so renames, withheld fields and channel expansion agree with the Dataset by construction. |
| dynamics tracers | removed from the gridpoint state before it is accumulated and stacked. |
| dynamics prognostics (`u_wind` ... `normalized_surface_pressure`) | replaced by `None` leaves, which carry no arrays through the scan. The one exception: one 3-D prognostic (`temperature` when all would go) always stays in the saved state, because every consumer of a trajectory reads its frame count and grid shape off it; it is removed from the Dataset. |
| anything left | `ModelPredictions.to_xarray` removes whatever the Dataset still holds that is unselected (that reference field, the sibling channel of a partly selected multi-channel array, a backend's extras). |

The carry, the snapshot stream and the observers read the *unpruned*
per-step diagnostics, so selecting output cannot change what the simulation
sees. A snapshot variable that is also in `output_drop` is still written to
the snapshot file: `snapshot_variables` is an independent selector.

Checkpoints hold the dynamical state and the physics carry, never outputs, so
what a run writes has no bearing on a restart. A selection does change the
compiled program (dead outputs disappear and XLA fuses differently), so a
selected run matches a full-output run to rounding, not bit for bit; a
selected run reproduces itself bit for bit across a checkpoint and resume.

## What the chunked runner reads back

`run_chunked` builds each chunk's Dataset once and hands it to several
consumers. Under a selection each is decided on its own:

| consumer | reads | under a selection |
| --- | --- | --- |
| `check_health` temperature, wind and humidity checks | the chunk-boundary state passed as `boundary_state` | unaffected: the state does not come from the output |
| `check_health` emission-wind check (fatal when persistent) | `wind_10m_model_level` | computed for the gate, left out of the file unless selected |
| `check_health` report lines | `radiation.toa_lw_up`, `surface_lw_down`, `toa_sw_down`, `toa_sw_up`, `convection.precip_conv`, the surface temperature | computed for the gate, left out of the file unless selected |
| `check_health` NaN scan | every Dataset variable | scans the variables computed (selected plus the above); a NaN confined to an unwritten diagnostic is not seen, and one in the state is, through the temperature check |
| `aerosol_budget_report` | the `budget_*`, `emi_*`, `wet_*`, `dry_*` gauges | computed for the report, left out of the file unless selected |
| monthly accumulator, chunk files, pressure-level interpolation | the written Dataset | the selection, plus `pressure_full` when `run.pressure_levels` is set |
| checkpoint and restart logic | model state and carry | unaffected |

`jcm.diagnostics.runner_read_fields` is the list of names in the second to
fifth rows, kept beside the functions that read them and checked against the
source of `check_health` by a test, so a new check cannot read a variable the
runner does not provide.
