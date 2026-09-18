# Checkpoint compatibility and migration

A checkpoint written by `jcm.checkpoint.save_checkpoint` holds exactly the
pair a run needs to continue — the backend-native dycore state and the
cross-step physics carry — plus the elapsed sim-day count. It is restored
against a *destination* model that has already been bootstrapped, so the
model, not the file, supplies the pytree structure the arrays are poured
back into.

That makes a checkpoint a serialization of a live `Model`'s state, not a
self-describing archive format. Two consequences shape everything below:
a checkpoint is only portable between models that agree on the grid, the
levels and the physics composition; and the file needs enough metadata of
its own to say *which version's* conventions its numbers follow.

## What is stamped

Every checkpoint records:

| Key | Meaning |
| --- | --- |
| `schema_version` | integer on-disk layout version (`jcm.checkpoint.SCHEMA_VERSION`) |
| `jcm_version` | the writing installation's version, for provenance |
| `elapsed_days` | sim-days the donor run had completed |
| `dycore`, `physics` | the state arrays, each keyed by its **pytree name** |
| `physics_fields` | the ordered child names of each physics-carry struct/group |
| `dycore_tracers` | the saved tracer names, each with its `nondimensionalize` flag |
| `prognostic_carry_slots` | carry keys the writing composition called prognostic |

The array keys are what make migration possible. `tree_math.struct`
registers its structs without key paths, so JAX labels their children by
position; `jcm.checkpoint` recovers the field *name* from the dataclass
field order (child *i* is field *i*) and joins the path into
`tracers.qc`, `radiation.lw_flux_up`, `_prev_step.specific_humidity`.
Matching on those names instead of on a flat leaf index is what decouples
a checkpoint from the exact field set of the jcm that wrote it.

`physics_fields` and `dycore_tracers` are not needed to match arrays —
the names already do that. They record what the *writing* model declared,
which is the only thing a future unit migration can key on (see below)
and the first thing to read when a restore is refused.

## What migrates automatically

**A changed physics-carry field set.** A field the destination model
carries but the file does not have takes the freshly bootstrapped model's
value for it (`PhysicsTerm.initial_carry_state`, i.e. `.zeros()` for most
slots, or the term's documented seed such as TTE-TKE's turbulence floor).
A field the file has but this model no longer carries is dropped. Both are
logged at INFO on the `jcm.checkpoint` logger, naming the fields, so a
resume is auditable.

This is safe for the class of change that keeps occurring: carry fields
are diagnostics that terms rewrite within a step or two of the restart.
The one bounded exception is a radiation sub-cycle cache seeded this way,
which starts one radiation interval (default 2 h) stale — the same
staleness `init=from_state` already accepts, and negligible against
losing the restart.

**Except where a slot is prognostic.** A term whose carry slot holds the
only copy of a physical quantity declares it in
`PhysicsTerm.prognostic_carry_slots`; JAM's cloud-borne aerosol phase
(`_jam_cloud_borne`) is the case, stored in the carry and in no dycore
tracer (#602). Seeding such a slot would invent mass and dropping it
would destroy mass, neither recoverable, so a restore that would have to
do either is refused. Both directions are covered: the destination
model's declaration protects a slot the file predates, and the file's own
recorded declaration protects one the reading composition no longer
carries. Every other slot in the tree today is a diagnostic — the
boundary-condition, cloud, aerosol-optics and `_prev_step` groups are
overwritten at the top of each step, TTE-TKE's prior-step TKE reseeds at
the ECHAM floor a cold start would use anyway, and the aerosol budget
gauge's lagged expectation is zero on a first step by construction.

The dycore state gets the same name matching but no filling or dropping:
its leaves are the prognostic state, which is never invented. A name
difference there means a different composition or backend and is refused.

## What is refused

* **A different grid or level count** — a shared name whose array shape
  differs. The error names the file and the leaf.
* **A different precision** — a shared name whose dtype differs (a float64
  pySES state is not a float32 one). Scalars are exempt: a bootstrapped
  template's `sim_time` is a Python float where a run leaves a 0-d float32
  array, and that pairing is exact either way.
* **A different physics composition or dycore backend** — a dycore-state
  name in one and not the other (a tracer added or removed).
* **A changed field set under a prognostic carry slot** — see the
  exception under "What migrates automatically".
* **A newer `schema_version`** than this build reads: a later schema may
  store values this version would silently misread.
* **An unstamped file** — anything written before this policy. See the
  next section for why, and for the explicit escape hatch.

Refusal is always a `ValueError` naming the file and, where there is one,
the offending leaf.
The alternative — restoring a state that deserializes cleanly but means
something else — surfaces as a NaN days later, far from the cause.

## Why an unstamped file is refused

PR #824 gave every mass mixing ratio one contract at the Dinosaur
boundary: the dimensionless kg/kg value, stored unscaled, because the
hybrid dynamics reads condensate directly in its virtual-temperature
loading term. Before that the bridge nondimensionalised the gridpoint
number as g/kg, i.e. scaled it by 1e-3 on the way in and by 1e3 on the
way out.

Feeding the same gridpoint state (`q = 8.0`, `qc = 1e-5`) through
`DinosaurDycore.initial_state` on either side of that change shows what
the *store* became:

| | stored `specific_humidity` | stored `qc` |
| --- | --- | --- |
| before #824 | 0.0284 | 1e-8 |
| since #824 | 28.36 | 1e-5 |

(Modal coefficients for the humidity column, nodal values for the
semi-Lagrangian `qc`; both rescale by exactly 1000.)

For a mass mixing-ratio tracer that settles it: the gridpoint value was
kg/kg on both sides, so a pre-#824 file holds a thousandth of the current
convention and needs ×1000. For `specific_humidity` it does not. #666
changed what the gridpoint number *means* in the same release: it was
g/kg in SPEEDY's self-consistent internals (so that package's store was
already the physical kg/kg value and must not be rescaled) and kg/kg in
ECHAM's (so that package's store was physical/1000 and must be). The
physics carry inherits the same split through
`_prev_step.specific_humidity`, which holds the gridpoint value.

The correct factor per leaf therefore depends on which package and which
generation wrote the file — and an unstamped file records neither. Nor can
it be dated: a file written between #824 and this policy is already in the
current convention and carries no stamp either. jcm does not guess between
those cases, so **pre-3.0 checkpoints are not resumable** and
`load_checkpoint` says so.

`load_checkpoint(..., unstamped_scale=...)` is the way through when the
caller *does* know how a file was written. It takes a `{leaf name: factor}`
mapping, applies it to exactly those arrays, logs each factor at INFO, and
rejects a name that is not a float leaf of this model rather than ignoring
it. `{}` is the meaningful assertion "this file needs no rescale", and the
refusal message lists this model's mass mixing-ratio leaves as the
candidates to decide about.

For a pre-#824 **ECHAM-family** donor that means the dycore's humidity
*and* every `nondimensionalize=True` tracer — condensate, and aerosol and
gas mass for a JAM composition — all by 1000, with the physics carry left
alone, because ECHAM's gridpoint values were already kg/kg:

```python
load_checkpoint(model, path, unstamped_scale={
    "tracers.specific_humidity": 1000.0,
    "tracers.qc": 1000.0,
    "tracers.qi": 1000.0,
    # ... every other nondimensionalize=True tracer the composition carries
})
```

A SPEEDY-family donor is the mirror image and is **not** written out here
as a recipe: by the table above its dycore humidity store was already the
physical value and must not be touched, while the gridpoint values its
carry holds moved from g/kg to kg/kg. Derive it from the table for the
composition in hand rather than copying the ECHAM one.

The same assertion reaches the `init=from_state` warm start as
`init.unstamped_scale` (entries spelled
`"tracers.specific_humidity=1000"`, because the leaf names contain dots
that a Hydra override cannot put in a dictionary key) — a
warm-start donor spun up with an earlier jcm is exactly the case. It is
deliberately *not* wired into `run.checkpoint_path`: resuming a campaign
whose own physics has changed underneath it is not something to make a
one-flag operation, and migrating the file once is explicit about what was
assumed.

## Bumping the schema

Bump `SCHEMA_VERSION` when a change alters what a stored value *means* —
units, sign, frame, layout — rather than which fields exist. Adding or
removing a physics-carry field needs no bump; name matching already
handles it.

For a bump:

1. Add the new version, with a one-line description, to the
   `SCHEMA_VERSION` docstring in `jcm/checkpoint.py`.
2. Write the migration as a step from the old version to the new one,
   keyed on the metadata the old file *does* carry (`dycore_tracers`'
   flags, `physics_fields`) — never on the destination model's identity,
   which describes the reader rather than the file.
3. If the change cannot be inverted from the file's own metadata, refuse
   that schema with an error that names what the reader would have to
   assume, and record the reasoning here.
4. Cover the migration in `jcm/checkpoint_test.py` by building a payload
   at the old schema and asserting both the migrated values and the INFO
   log.

## Related

* `jcm/checkpoint.py` — the implementation and the per-argument contract.
* `jcm.initial_states.checkpoint_state` — the same file read as a warm
  start with the clock reset, rather than as a resume.
* `docs/source/running_at_scale.rst` — the chunked/preemptible run loop
  that writes these files, including the `.prev` rotation.
* `docs/source/science/dynamical_core.md` — the tracer contract at the
  Dinosaur boundary that the unit discussion above rests on.
