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
python tools/release_validation/health.py $SCRATCH/jam_runs/mx_<member> \
    --last-n 40 --log runs/mx_<member>.log
```

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

Lean by construction: `matrix.yaml` members reference the validated
preset table in `tools/benchmark.py` (`PRESETS` — the single home of
known-good override sets), `health.py`'s burden gates derive from
`tools/jam_burden_report.py`'s shared species/anchor table (anchor
range × slack 3), and per-grid inputs resolve automatically
(`terrain=auto`, `forcing.ozone_file=auto` fall back to the mirror).
SPEEDY profile facts (default run group + init; the longrun sponge
spans the whole L8 atmosphere) live with its preset in benchmark.py.
