# Next Steps: ARMBE Symbolic Cloud Cover

These tasks extend the current SGP-only, offline SPEEDY diagnostic and
symbolic-regression experiment. The existing fixed train, validation, and test
splits must remain intact while each new experiment records its own data recipe,
search configuration, selected equation, and held-out evaluation.

## 1. Expand Across ARMBE Sites

Generalize the data cache and feature export from the Southern Great Plains
(SGP) site to every ARMBE site with the variables and quality-control fields
required by the independent SPEEDY diagnostic.

Completion criteria:

- Audit available years, variables, units, vertical coordinates, and QC fields
  for each site.
- Build site-aware caches and record samples excluded for missing or invalid
  inputs.
- Preserve whole-year train, validation, and test separation, preferably with
  site-stratified splits.
- Report within-site and cross-site performance, including transfer to sites not
  used in the SR search where feasible.

## 2. Enrich the SR Search Space

Expand beyond the current four SPEEDY scalar predictors and limited operator
set, while retaining interpretable expressions and Pareto-frontier selection on
validation data.

Candidate directions:

- Add physically meaningful profile summaries, vertical humidity-gradient
  measures, temperature or stability diagnostics, and precipitation-regime
  indicators when consistently available across sites.
- Test richer but controlled operator families, including smooth threshold and
  saturation behavior.
- Keep explicit operator-complexity costs, nesting restrictions, deterministic
  search seeds, and full Pareto-frontier export.
- Compare richer searches against the current selected equation, not only
  against default SPEEDY.

## 3. Formulate Physical Constraints

Develop constraints before treating an SR equation as a candidate cloud scheme.
Constraints should be encoded during search where possible and screened after
search otherwise.

Initial constraint targets:

- Cloud fraction must remain in `[0, 1]` over the relevant feature domain.
- Predictions must be finite and avoid singular denominators or unstable
  extrapolation.
- Cloud cover should have a physically justified moisture response, especially
  with respect to cloud-relevant and lowest-level relative humidity.
- Stability effects should be checked across moist and dry boundary-layer
  regimes rather than assumed globally monotonic.
- Any constraint must be evaluated on training, validation, test, and
  cross-site feature ranges without using test performance to select equations.

## 4. Use AgentSR / KeplerAgent

Use the local AgentSR (KeplerAgent) workflow rather than only its standalone
PySR wrapper for future equation-discovery experiments.

Relevant local code:

```text
/data/MOSAIC/sr-agent/agentsr/
```

Completion criteria:

- Configure the workflow with the site-aware feature tables, scientific search
  instructions, allowed operators, complexity costs, and physical constraints.
- Preserve tool calls, resolved configurations, Pareto frontiers, and validation
  selections as experiment artifacts.
- Use the agent workflow to propose and document candidate equations, while
  retaining deterministic numerical evaluation and explicit held-out testing.

## 5. Inventory ARM Products Before Ordering

Start with `ARM_JCM_OBSERVATION_MAP.md`. It organizes candidate ARM families by
JCM process, observational role, caveats, and backbone versus specialized use.
The generated JSON and CSV below are technical references for coverage checks,
not the recommended human entry point.

`inventory_arm_datastreams.py` creates a local metadata inventory from the
public Data Discovery datastream feed. It does not query or download data files,
so no ARM token is needed. The full inventory is written with:

```text
python experiments/armbe_sgp/inventory_arm_datastreams.py \
  --output outputs/arm_catalog_all.json \
  --class-summary-output outputs/arm_catalog_classes.csv
```

The output retains individual datastream names, sites, facilities, data levels,
availability, retirement state, and temporal coverage, and groups variants by
ARM product/instrument code. The optional class CSV has one row for each of
ARM's roughly 400 instrument classes and is the recommended first-pass view.
Each record also preserves ARM's site description and a conservative deployment
category: `mobile_facility`, `off_site_campaign`, or `fixed_or_other`. The last
category does not assert permanent-observatory status and must be reviewed for a
given experiment. By default, product families are also labelled
`anchor_supported`, `other_fixed_support`, or `mobile_or_campaign_only` using
SGP, NSA, and ENA as durable reference sites. Override the reference set with
repeatable `--anchor-site` arguments when an experiment has a different domain.
Use `--site`, `--facility`, `--available-only`, and `--visible-only` to create a
smaller candidate inventory. Inspect schemas with the ARM metadata API before
requesting files through the authenticated ARM Live Data API.

`audit_local_armbe.py` compares that catalog snapshot with local raw collections,
checks archive orders against their supplied manifests, identifies duplicate
candidates, and classifies processed artifacts:

```text
python experiments/armbe_sgp/audit_local_armbe.py \
  --catalog experiments/armbe_sgp/outputs/arm_catalog_all.json \
  --output experiments/armbe_sgp/outputs/local_armbe_audit.json
```

The interpreted 2026-08-13 result and canonical shareable-release policy are in
`LOCAL_DATA_AUDIT.md`.
