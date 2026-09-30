# Resolution-dependent parameter defaults

ECHAM6.3 sets a number of tunable parameters per model grid: the cloud cover's
critical relative humidities, the convective precipitation conversion, the
cloud-optics inhomogeneity factors and others take different values at T63 and
at T127. They are tunables, not constants of the formulation, and calibration
has to be able to move them. This page records how jcm chooses them for a grid
and how a scheme adopts the mechanism.

## The mechanism

`jcm/physics/resolution_defaults.py` holds one function for every scheme:

* A scheme keeps a table of its reference values per spectral truncation,
  `{truncation: {field: value}}`, next to its `Parameters` class
  (`jcm/physics/clouds/echam_cloud_defaults.py` for the cloud schemes).
* `resolution_defaults(table, truncation, nearest=..., fallback=..., ...)`
  returns one `{field: value}` mapping. At a tabulated truncation it returns
  the table's values exactly. Between two tabulated truncations it
  interpolates linearly in the truncation number; fields that are integers or
  switches in the reference (`nearest`) take the nearer truncation's value, the
  finer one on a tie. Outside the tabulated range it holds the end value and
  warns once per grid. A grid without a spectral truncation (the pySES cubed
  sphere) takes the `fallback` row, with a warning.
* The scheme's `Parameters.default(truncation=...)` builds its defaults from
  that mapping; the numeric fields stay ordinary differentiable pytree leaves.

Interpolation continues the resolution trend the reference encodes instead of
switching between its rows. The interpolated values are jcm's choice and are
untuned; the reference itself has no configuration between its truncations
(ECHAM6.3 stops with "Truncation not supported").

## Chosen at construction

The defaults are fixed when the physics is built, so that the parameter pytree
a user holds afterwards is final: calibration code builds its optimiser state
from it, and nothing may change it behind the optimiser's back.

* `echam_physics(coords=...)` builds the defaults of every scheme that has
  them for the grid's truncation; without `coords` it builds the T63 defaults.
* The Hydra runner builds the coordinates before the physics and passes them
  to both configuration doors: the factory presets (`physics.builder`) get
  `coords`, and the term-list presets build each term's `Parameters` base with
  `default_parameters(ParamsCls, truncation)`. With the pySES dycore the
  coordinates are the dycore's, which exists only after the physics has named
  its tracers: the runner reads the tracer declarations from a first build and
  builds the physics the model runs with `dycore.coords`. That grid has no
  spectral truncation, so its defaults are the T63 row, with the warning that
  says so.
* `default_parameters` calls `default(truncation=...)` on a class that accepts
  it and plain `default()` otherwise, so a scheme gains resolution defaults by
  adding the keyword and nothing else.

## Precedence

Highest first:

1. An explicit `Parameters` object passed by the caller is used as given.
2. A field override (a Hydra `physics.clouds.crs=...`, a factory mapping, a
   term-list `params:` block) replaces that one field; the base it is applied
   to is the grid's defaults, so the other fields keep them.
3. The grid's resolution default.

## The grid check

`cache_coords` does not re-resolve anything. A term that holds
resolution-dependent parameters records, as static metadata, the truncation its
defaults were built for (`defaults_truncation` on the `Parameters`) and whether
they were built by the factory or runner (`params_are_defaults=True`) or
supplied by the user. For factory-built defaults it calls
`check_defaults_grid`, which warns once, naming both grids, when they differ; a
user-supplied object is never checked, and a single-column grid, which has no
horizontal resolution, is never checked.

Grid geometry that the reference derives from the vertical grid, such as
ECHAM's inversion-search levels `jbmin`/`jbmax`, is not a tunable. It is
computed from the model's own levels in `cache_coords`.

## Adopting it in another scheme

1. Put the reference table next to the scheme's `Parameters`.
2. Give `default` a `truncation: int | None = 63` keyword that fills the
   table-driven fields (explicit keyword values winning) and records
   `defaults_truncation` as a `pytree_node=False` field.
3. Give the term a `params_are_defaults: bool = False` keyword and call
   `check_defaults_grid` from its `cache_coords` when it is set.

The factory and the runner then need no change. Today the cloud cover
(`CloudParameters`) and the 1M cloud scheme (`MicrophysicsParameters`: `cvtfall`,
`csecfrl`, `clwprat`) use it. ECHAM6.3 also sets these by truncation, which jcm
holds fixed at present: convection (`cmfctop`, `cprcon`,
`mo_echam_conv_constants.f90` l.121-137, and `cmftau = min(3 h, 7200 s·63/nn)`),
the ice cloud-optics inhomogeneity `zinhomi` and the deep-convective liquid
`zinhoml3` (`mo_cloud_optics.f90` l.115-134), and the subgrid-orography wake
coefficient `gkwake` (`mo_ssodrag.f90` l.93-122). Horizontal diffusion already
follows ECHAM's truncation table through its own interpolation in
`jcm/diffusion.py`.
