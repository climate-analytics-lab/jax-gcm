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

## Workflow

```bash
# 1. Generate + submit the year runs (Derecho; JAM aux inputs staged per
#    jcm/data/mirror/SOURCES.md, pointed at by JAM_INPUTS/JCM_EMISSIONS)
python tools/release_validation/launch.py --repo . --submit

# 2. SCM member (CPU, ~15 min)
python tools/release_validation/scm_check.py 10

# 3. Health-check each finished run (exit 0 = all gates pass)
python tools/release_validation/health.py $SCRATCH/jam_runs/mx_<member>_<tag> \
    --last-n 40 --log runs/mx_<member>_<tag>.log
```

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
2–4 mm/day; cloud cover 0.4–0.8; near-surface T 278–295 K; AOD₅₅₀
0.02–0.35; JAM per-species burdens vs loose AeroCom ranges. `--last-n 40`
scores the settled ~200 days of a from-zero spin-up year (full spin-up is
~9 months — see #638). The checker speaks both the ECHAM and SPEEDY field
dialects. Post the table to the release issue; compare settled sim-days/hr
against the baselines in #638 (>15% drop = runtime regression).

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
dynamics residual `budget_dyn/mass < 0.1 %/step` from the #713 in-step gauge
(unscored on output that predates it, or in a run directory with no saved
Hydra config to read the timestep from). They exist
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
*transport* conserved mass. The August-2026 runaway was semi-Lagrangian
non-conservation, not aerosol physics — see the design doc.

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
