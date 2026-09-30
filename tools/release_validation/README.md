# Release validation

Pre-release validation of the supported configuration matrix
(issue #638): a full-output year per member on one A100, climatological
health gates, and an SCM aerosol-pathway check. Run before tagging a
release or merging `dev` → `main`.

| member | physics | grid | pairing policy |
|---|---|---|---|
| speedy-t31 | `speedy` | T31 L8 | grey radiation |
| echam-1m-{t63,t106} | `echam` | L47 | RRTMGP + MACv2-SP |
| echam-2m-{t63,t106} | `echam-rrtmgp-2m` | L47 | RRTMGP + MACv2-SP |
| echam-jam-t63-{l47,l95} | `echam-jam` | T63 | RRTMGP + JAM |
| scm | full ECHAM+JAM physics | 1 column L47 | `scm_check.py` |

## The fast regression fixtures

The year-long runs above are the release gate; they are far too slow to catch
an accidental change during development. The same matrix therefore also backs
a **fast** regression — a few minutes per member — in
`jcm/model_test.py::test_release_matrix_default_statistics`, gated behind
`JCM_RUN_GPU_INTEGRATION_TESTS=1`.

Each member's fixture is a pair: the **bands** (`<member>_statistics.nc`,
committed under `jcm/data/test/release_matrix/`, tens to a few hundred KB so a
change shows up as a reviewable diff) and the **init state** it resumes from
(hosted on the data mirror under `bundles/<grid>_<levels>/init_states/`, since
it runs from a few MB to several GB). The bands describe the window that follows that exact state, so the two
are only meaningful together and are regenerated together — one command per
member, on a GPU:

```bash
CUDA_VISIBLE_DEVICES=<idx> python -c "import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'; from jcm.data.test.release_matrix.generate_stats import generate; generate('echam-1m-t63', out_dir='/scr/$USER/fixtures')"
```

Set the variable first, as above. Importing jcm does not initialise a JAX
backend, and `generate` does no device work itself, but XLA reads the setting
only when a backend first comes up: a process that has already touched the
device (a notebook that built a model first) holds 75% of the card under the
default, starving the workers that actually integrate the model. `generate`
refuses to run without it.

**Generate in a CI-parity environment** — a fresh venv with
`pip install -e ".[mam4]"` plus a CUDA jax build of the pinned version
(`pip install "jax[cuda12]==<the pinned jax>"`), so every dependency resolves
to the repo's pins — **never in a shared or long-lived environment**. The bands
are only valid under the dependencies the test later runs with: a jax-rrtmgp
release a pin has moved past shifts the radiation of the whole column, so bands
drawn under it fail a correct model across the whole column, indistinguishably
from a physics regression. Each band file records the versions it was drawn under
(`bands_environment`) and the ones its init state was spun up under
(`init_state_environment`); check them first when a whole member fails
together. To re-derive bands on an already-published state, pass
`write_state=False` with the state in `out_dir`, plus
`state_environment=<its init_state_environment>` so that record is kept.

then upload the file it wrote — `<member>_fixture_<digest>.msgpack`, whose
name carries a digest of its own contents — additively under that member's
`init_states/` prefix (see `docs/source/design/data_mirror.md`). Upload it
under exactly the name `generate` produced: the band file records that path.
Mirror reads resolve at the pinned commit (`MIRROR_REVISION` in
`jcm/data/remote.py`), so bump the pin to the upload's commit in the same PR as
the bands, or no run can read the new state.

Before uploading, validate the new pair locally: point
`JCM_FIXTURE_STATE_DIR` at the directory holding the generated state(s) and
run the test. With it set, each member reads its state from there (digest
checked) and never from the mirror; a member whose state is absent is skipped,
named. A band file marked `hosted_state="pending"` (a state deliberately not
published yet) is skipped for the same reason when its state is not local, but
is validated like any other member when `JCM_FIXTURE_STATE_DIR` holds it.

**Bands are tied to a data-mirror commit.** `generate` records the commit its
inputs were read at as the band file's `data_mirror_revision`. The regression
fails a member whose band file names a different commit than the run's, with
"fixture generated at <A>, run uses <B>: regenerate the bands", instead of
reporting an input change as a physics regression. A band file without the
attribute predates it and is compared with a warning.

Every band is an area-weighted global mean — the grid's Gauss-Legendre
quadrature weights, via `jcm.analysis.global_mean`, never an equal-weight mean
over latitude rings — computed by the one reduction that both `generate` and
the test use. Band number concentrations are column burdens weighted by the layer air mass
`dp/g` (the `pressure_thickness` diagnostic): `air_density * layer_thickness`
is not a mass weight, because `layer_thickness` is floored at 10 m. Every
reduction propagates NaN, so a partially non-finite run fails its member
rather than averaging the finite cells into a band-sized mean.

Both are built through the member's **validated preset**, the same recipe this
directory's `matrix.yaml` names, so the regression covers what the project
claims to support rather than a composition invented for the test.

Two things to know before reading a failure:

- These are **regression** bands, not a climatology. They come from a short
  window after a short spin-up from the preset's own init, because the
  equilibrated states on the mirror are unreadable by current jcm (#762). A
  failure means "something changed", not "the physics is wrong".
- The **JAM members' bands describe the post-dust-retune aerosol climate**
  (#787/#808/#840): the relative-soil-wetness saltation gate and the
  `nduscale_reg` recalibration for jcm's winds. They were regenerated against
  that code and the rebuilt forcing bundle, so a failure is a regression, not
  the known-provisional state the pre-#840 bands were.

## Workflow

On Derecho (PBS); the same members as Kubernetes Jobs are under
[Workflow on Nautilus](#workflow-on-nautilus-kubernetes).

```bash
# 1. Generate + submit the year runs (Derecho; JAM aux inputs are the
#    present-day climatology mirror bundles, fetched at generation time)
python tools/release_validation/launch.py --repo . --submit

# 2. SCM member (CPU, ~15 min)
python tools/release_validation/scm_check.py 10

# 3. Health-check each finished run (exit 0 = all gates pass)
python tools/release_validation/health.py $SCRATCH/jam_runs/mx_<member>_<tag> \
    --last-n 40 --log runs/mx_<member>_<tag>.log
```

**The data-mirror commit is part of a validation run's provenance.**
`launch.py` records it in `<rundir>/mirror_revision.json` at first launch and
exports it into every job (a PBS job does not inherit the submitting shell).
`--resume` reuses the recorded commit; an explicit, different
`JCM_MIRROR_REVISION` is refused unless `--force-mirror-revision`, which
records the new commit and opts the job in to resuming across the switch.
If a forced launch dies before its first checkpoint, re-issue the same
`--resume --force-mirror-revision` command, which regenerates the opt-in; the
record stores only requested/source/commit/written.
Compare two validation runs only at the same commit.

Every artefact of a launch — rundir, PBS job name, outputs, log — is
namespaced by a run tag, which defaults to the launched repo's HEAD short
SHA (`--tag` overrides it; outside a git checkout it falls back to the UTC
date). A member is a *fresh* year, so `launch.py` refuses to write a job
whose rundir already holds a `checkpoint.msgpack`: continue that
integration with `--resume`, or launch under a new `--tag`. The tag is what
keeps two branches' validation runs of the same member apart: sharing a
rundir lets `run_chunked` silently resume the other branch's checkpoint
(#701).

A FAIL is a recorded verdict, not necessarily a blocker: members with
known characteristics (the 1m bright-cloud TOA, SPEEDY's wet bias, the
JAM soa/ss calibration items tracked in JEM-Cal#4) fail their gates by
design until fixed or the matrix declares them expected. Post the table
as-is.

Gates: NaN scan on every saved variable; TOA net |≤10| W/m²; precip
2–4 mm/day; cloud cover 0.5–0.9 (SPEEDY 0.4–0.8, see below);
near-surface T 278–295 K; AOD₅₅₀
0.02–0.35; JAM per-species burdens vs loose AeroCom ranges. `--last-n 40`
scores the settled ~200 days of a from-zero spin-up year (full spin-up is
~9 months — see #638). The checker speaks both the ECHAM and SPEEDY field
dialects. Post the table to the release issue; compare settled sim-days/hr
against the baselines in #638 (>15% drop = runtime regression).

**Cloud cover** is ECHAM's own total cover `aclcov` — maximum-random
overlap of `clouds.cloud_fraction`, `mo_cloud.f90` §10.2, via
`jcm.analysis.total_cloud_cover` — because that is the construction the
reference model uses and a total cover is the basis the satellite
climatologies are quoted on, and because it is computable from any saved
output.

The ECHAM band is **0.5–0.9**, calibrated on this definition: max-random
reads +0.11 to +0.15 above the column max the gate used to score, so a band
carried over from column-max experience would fail correct members on the
ceiling for a purely definitional reason.

SPEEDY scores its own `shortwave_rad.cloudc` — an RH-based column cover with
no profile to overlap, and untouched by this work — so it gates on its own
**0.4–0.8**. Shifting it with the ECHAM band would tighten the floor of the
member that sits closest to it (recorded 0.57 and 0.58) for a reason that
does not apply to it.

Two more covers are **printed and not gated**: `cloud_cover_colmax`, the
column maximum the gate used to score (a lower bound, kept so the #638/#782
tables stay readable), and `cloud_cover_radiation`, the McICA sub-column
cover the RRTMGP flux solve integrates (dropped when the run saved none, or
an all-zero field under grey radiation; the NOTE says which). The McICA
cover is a **different measurement, not a cross-check** — a time mean of an
instantaneous cover from a differently-preprocessed field, against an
overlap of the output-averaged fraction — and the two differ by ~0.25 on a
measured arm, which is expected.

**Cover numbers from before #707 are not comparable with these** — that PR
gave the 1M scheme ECHAM's `ccwmin` cover write-back, which redefined what
`clouds.cloud_fraction` counts. Its measured size (−0.066 of low cloud for
+0.15 W/m²: bookkeeping, not cloud) is a **column-max** figure and does not
carry over to the other two definitions, where it is unmeasured. Rationale,
the measured table and the #782 decomposition:
`docs/source/design/cloud_cover_gate.md`.

### The JAM aerosol block

On a run whose saved variables show JAM is composed (modal mass tracers
plus a `jam_*` diagnostic namespace), `health.py` adds the statistics from
`aerosol_stats.py` — also runnable on its own:

```bash
python tools/release_validation/aerosol_stats.py $SCRATCH/jam_runs/mx_<member>_<tag>
```

It reduces each chunk file separately (a JAM year is ~60 GB and must never
be opened as one array, so budget ~40 min for a full T63 L47 year;
`--series-out` saves the reduction and `--series-in` re-scores it without
re-reading the run) and reports per-species burdens including the
cloud-borne phase, their logarithmic drift over the final six months,
lifetimes, the mass-budget residual, sulfate's upper-level and hemispheric
distribution, AOD/Ångström, near-surface CDNC and N100, and the modal dry
radii. Three of those are **absolute gates**, not climatological ranges:
`|d ln B/dt| < 0.002 /day`, `|budget residual| < 5 %`, and the per-step
dynamics residual `budget_dyn/mass < 0.1 %/step` from the #713 in-step gauge.
They exist
because an aerosol runaway (#658) stays inside a ×3-slack range gate until
its final fortnight — the drift statistic is what sees it coming, and the
residual says whether the cause is a source or a sink.

The residual's **sign** matters. Negative (more deposited than entered) is
mass creation and cannot be explained away. Positive (emitted mass
unaccounted for) is what a missing sink *diagnostic* also looks like, and on
output written before the #722 removal-ledger fix `dry_*` omits Slinn dry
deposition entirely — so the gate names that caveat instead of calling it a
leak. The drift gate reads burdens only and is unaffected either way.

The dynamics gate answers a different question from both: whether the
*transport* conserved mass. The runaway that motivated these gates was
semi-Lagrangian non-conservation, not aerosol physics — see the design doc.

**Nothing passes by absence.** The drift and closure statistics need a window
of at least 90 days (below that a fitted slope is its own noise), the dynamics
gate needs, per species, the gauge, its mass denominator and a timestep, a
lifetime needs both deposition ledgers, and a species the run does not carry
has no burden to score. Every one of those is printed as `UNSCORED` with its
reason and counted in the summary line, because a missing row would otherwise
be indistinguishable from one that passed. Use `--last-n` to pick the settled
months, not to shrink the window below the floor.

**An UNSCORED gate does not fail the exit code, by design.** The commonest
causes are the user's own `--last-n` and a species the configuration does not
carry, and failing those would make the tool unusable for the windows it is
documented to support — so the report names them and the exit status ignores
them. The one exception: scoring *nothing at all* exits non-zero, because a
report that measured nothing has not passed anything and a harness reading only
the return code would otherwise see success. A release gate that wants a
stricter rule should read the `UNSCORED` lines, which is what they are for.

Non-JAM members skip the block. Rationale and the regression tolerance tiers:
`docs/source/design/jam_regression.md`.

Lean by construction: `matrix.yaml` members reference the validated
preset table in `tools/benchmark.py` (`PRESETS` — the single home of
known-good override sets), `health.py`'s burden gates derive from
`tools/jam_burden_report.py`'s shared species/anchor table (anchor
range × slack 3) applied to `aerosol_stats.py`'s annual means, and
per-grid inputs resolve automatically (`terrain=auto`,
`forcing.ozone_file=auto`), prefetched at submit time by `launch.py`.
SPEEDY profile facts (default run group + init; the longrun sponge
spans the whole L8 atmosphere) live with its preset in benchmark.py.

## Workflow on Nautilus (Kubernetes)

The same members as Kubernetes Jobs on the NRP Nautilus cluster (skill
`kubernetes-jcm-runs`): a pod has its A100-80GB to itself, and output lands
on the `jcm-runs` volume under `/runs/mx_<member>_<tag>/`. The Job is the one
`.claude/skills/kubernetes-jcm-runs/scripts/mkrun.py` builds for every
production run (`job_manifest`: pinned clone, GPU check, resume from the
checkpoint on every retry, health and day-count gates); `launch.py` supplies
the member's overrides — the list the PBS job runs, with the rundir on the
volume.

```bash
export SCRATCH=/scr/$USER    # launch records and fetched runs ($SCRATCH/jam_runs)
export PATH=/data/dwatsonparris/micromamba/bin:$PATH             # kubectl

# 1. Inspect, validate against the API server, submit. The commit must be on
#    GitHub: the pod clones it from there.
python tools/release_validation/launch.py --site nautilus > /tmp/matrix.json
kubectl apply --dry-run=server -f /tmp/matrix.json
python tools/release_validation/launch.py --site nautilus --submit

# 2. Watch (the COMPLETIONS column; each Job tees its log to run.log too)
kubectl get jobs -l jcm-run
kubectl logs -f job/jcm-run-mx-<member>-<tag>

# 3. A Job that spent its retries (or was deleted): continue its run
python tools/release_validation/launch.py --site nautilus --members <member> \
    --tag <tag> --resume --submit

# 4. Copy a finished run off the volume, then score it as on Derecho
python tools/release_validation/launch.py --site nautilus --fetch \
    --members <member> --tag <tag>
R=$SCRATCH/jam_runs/mx_<member>_<tag>
python tools/release_validation/health.py $R --last-n 40 --log $R/run.log \
    --json $R/health.json
```

What the Kubernetes door adds, and why:

- **Pinned code.** `--pin jcm=REF` (default: the launched checkout's HEAD) is
  resolved to a full SHA when the Job is generated, and refused unless a
  branch or tag on GitHub contains it — an unpushed commit would otherwise
  fail only once a node was found and the image pulled. A branch or tag name
  means the one on GitHub. The default tag is the pinned commit's short SHA,
  so the run's name matches the `jcm=<sha>` its outputs record; the member
  definitions (`matrix.yaml`, `PRESETS`) are read from the launching
  checkout, which is warned about when it is not the pin.
- **The pinned commit's own environment.** The image carries an older jcm
  release, so the pod installs the pinned commit with its own requirements and
  the `mam4` extra (`pip install -e '/work/jcm[mam4]'`, what CI installs),
  holding the image's CUDA jax fixed by a constraint; the GPU check follows.
- **Inputs resolve in the pod,** which has network: the JAM aux inputs stay
  `hf://` URLs read at the mirror commit the Job exports
  (`JCM_MIRROR_REVISION`, recorded in `mirror_revision.json` exactly as on
  the PBS path). They are still prefetched here first, so a missing input
  refuses before a GPU is claimed.
- **One run directory, one launch.** Each launch is recorded as
  `$SCRATCH/jam_runs/<run>/launch.json` — the code pin, image, override list,
  length and Job name, with a digest — beside the manifest (`job.json`) and
  the mirror record. The pod writes the same record into the rundir on its
  first attempt and refuses any Job whose definition differs, or a checkpoint
  no record claims: the generating node cannot see the volume, and without
  this a reused tag (another machine, a deleted Job) would resume someone
  else's integration and report it as its own (#701). Regenerating the same
  launch is a no-op, so inspect-then-`--submit` works; a different definition
  under a recorded tag is refused.
- **Resuming.** Evictions need nothing: every retry (`--retries`, default 20,
  is the eviction budget) resumes from the checkpoint. `--resume` re-emits the
  recorded launch — whatever HEAD is now — and refuses `--pin`, `--days`,
  `--init` and `--extra`, since a resumed run must continue as it began. With
  `--submit` it replaces the finished Job of that name (Jobs are immutable;
  the output stays on the volume and its log in `run.log`) and never touches
  a running one. `--force-mirror-revision` works as on the PBS path.
- A Job that fails deterministically (a refused rundir, a bad override)
  restarts until its retries are spent, holding its GPU in back-off:
  `kubectl delete job <name>` once the log shows why.
- `--job-prefix` (default `jcm-run`, mkrun.py's) gives a campaign sharing the
  namespace its own Job names. A Job name longer than 63 characters is
  refused rather than truncated, because truncation could fold two arms onto
  one Job.

### Retune arms (#682)

An arm is one member with a warm start, a length and extra Hydra overrides,
under a suffix that goes into its run name — its own rundir, Job and label.
A sweep is one invocation per arm, or a small loop; the control is the same
command without `--extra`:

```bash
TAG=$(git rev-parse --short HEAD)
STATE=/runs/mx_echam_jam_t63_l47_<spin-tag>/mx_echam_jam_t63_l47_<spin-tag>_day180.ckpt
L="python tools/release_validation/launch.py --site nautilus \
   --members echam-jam-t63-l47 --tag $TAG --days 60 --init $STATE --submit"
$L --suffix control
for cape in 50 150 200; do
  $L --suffix cape$cape --extra +physics.convection.trigger_cape=$cape.0
done
```

`--init` takes an `hf://bundles/<grid>_<levels>/init_states/...` state or a
path on the volume; `--extra` overrides land last, so they win. Warm-start
from something that does not move: a permanent archive
(`<prefix>_day<N>.ckpt`, written every 30 days by the JAM members) or the
final checkpoint of a finished run — not the rotating `checkpoint.msgpack` of
one still running. The factory-built JAM presets take per-scheme fields as
`+physics.convection.<field>=` (#935); the term-list presets (echam-1m/2m)
as `+physics.terms.tiedtke_convection.params.<field>=`. On the PBS path the
same flags build the same arm, but `--resume` there regenerates from the
command line, so repeat them.

### Fetching, scoring and ingesting

`--fetch` copies `/runs/<run>/` into `$SCRATCH/jam_runs/<run>/` through a
throwaway CPU pod that mounts the volume read-only
(`kubernetes-jcm-runs/scripts/fetch_run.py`), skipping checkpoints unless
`--with-checkpoints` (the JAM archives run to ~1 GB each). The copy is
incremental and size-checked: re-running it after an interrupted stream copies
only what is missing or short, and it exits non-zero while anything is.

Copying was chosen over scoring inside a pod because it needs less new code:
one small copy helper, after which `health.py`, `aerosol_stats.py` and
jcm-monitor's ingest run unchanged on a local directory, exactly as on a
Derecho run. Scoring in a pod would mean rebuilding the run's checkout and
environment there, and the monitor re-scores a raw run directory with the
run's own `health.py` anyway. The cost is moving the chunk files (tens of GB
for a JAM year) through `kubectl exec`.

A fetched run directory is what jcm-monitor's `ingest` takes: it reads the
run's `.hydra/` (the Job sets `hydra.run.dir` to the rundir), the chunk files'
`*.provenance.json` sidecars and `run.log`, and scores the run with the
`health.py` of the commit the run records. From the monitor's checkout
(`~/jcm-monitor`, see its README):

```bash
<monitor root>/venv/bin/python -m monitor.run ingest \
    $SCRATCH/jam_runs/mx_<member>_<tag> \
    --experiment rc-3.0.0 --arm <member> --member <member> --last-n 40
# a retune arm: the monitor keys it by the arm's model-defining overrides
<monitor root>/venv/bin/python -m monitor.run ingest \
    $SCRATCH/jam_runs/mx_echam_jam_t63_l47_<tag>_cape150 \
    --experiment tiedtke-retune --arm cape150 --member echam-jam-t63-l47
```
