# Agentic Equation Discovery Roadmap

## Purpose

This is the near-term prototype of the MOSAIC discovery loop, scoped to one
replaceable SPEEDY cloud-cover equation. Its purpose is to give a future agent
a reproducible way to propose, test, calibrate, and reject physical equations.
It is not yet the full MOSAIC system, global JEM integration, or observational
targeting workflow.

The first target is the SPEEDY cloud-cover diagnostic, `cloudc`. The first
observational validation is the existing ARMBE single-column forecast setup at
SGP. ERA5 provides the next validation level, initially for globally resolved
state/forcing and nudged or short-window experiments rather than as a
replacement for the site-level cloud target.

```text
ARMBE, ERA5 observations
        |
        v
candidate equation discovery
(KeplerAgent, ZEPHYRUS, or symbolic regression)
        |
        v
differentiable JCM PhysicsTerm
        |
        v
short-window forecast and calibration
(ARMBE SCM, then ERA5/global cases)
        |
        v
validated metrics, gradients, and failure records
        |
        +--------------------> discovery agent
```

## What Already Exists

The current ARMBE/SPEEDY work has most of the first evaluation layer:

- ARMBE atmospheric profiles initialize independent SPEEDY single-column
  forecasts.
- The model can run free physics forecasts with a configurable timestep,
  horizon, stride, and observation cadence.
- A cache stores initial states, prescribed surface-temperature forcing,
  QC-masked cloud targets, and a versioned comparison recipe.
- The evaluator produces per-lead cloud-fraction RMSE, predictions, and
  provenance manifests.
- The September 2018 experiment runs 24-hour forecasts every six hours and
  compares SPEEDY `cloudc` to QC-passed ARMBE `tot_cld`.

This is an evaluation scaffold, not yet a discovery system. In particular, it
does not yet accept a candidate cloud equation, optimize its coefficients, or
select among candidates.

## Candidate Contract

Every proposed equation must be a structured artifact, not only generated
source code. The artifact should record:

- Symbolic expression, named inputs, output units, and the SPEEDY diagnostic or
  tendency it replaces.
- Coefficients with initial values, scientifically plausible bounds, and units.
- Claimed physical properties, such as cloud fraction in `[0, 1]`, finite
  behavior at dry and saturated limits, and monotonicity assumptions.
- Required JCM state/diagnostic inputs, so a candidate cannot use unavailable
  information.
- Data used to propose it, its source revision, and the discovery method.

An implementer renders this artifact as a pure, differentiable JCM
`PhysicsTerm`. The module must preserve JAX differentiation, use smooth bounded
operators instead of hard thresholds, and expose only continuous coefficients
to JEM-Cal. The equation's symbolic structure is discrete and is selected by
the discovery system, not differentiated through.

## Validation Ladder

1. **Mechanical gate.** Check the rendered module for required inputs, output
   range, finite forward values, smooth behavior near thresholds, and
   automatic-differentiation versus finite-difference agreement on a small
   column battery.
2. **ARMBE SCM gate.** Run independent short forecasts from observed SGP
   profiles. Compare the cloud trajectory to QC-passed `tot_cld`; retain the
   per-lead RMSE and the prediction-target pairs. This is the inexpensive
   inner-loop evidence for most candidates.
3. **ERA5 gate.** Test surviving candidates in short globally resolved or
   nudged JCM windows. This asks whether an equation that helps one observed
   column has a useful effect in a broader atmospheric state distribution.
4. **Promotion gate.** Only stable, calibrated candidates proceed to longer
   online integrations and later satellite or ClimateBench evaluation.

Each result, including a rejected candidate, must write a versioned artifact:
candidate definition, rendered module revision, input/cache recipe, metrics,
runtime, gradient checks, and failure reason. This artifact store is the
agent's memory and makes its decisions reproducible without an LLM.

## Structure And Coefficient Learning

Symbolic regression and calibration answer different questions:

- **Symbolic regression** searches the discrete equation structure and normally
  fits provisional coefficients against its offline training data.
- **JEM-Cal calibration** holds a chosen structure fixed and tunes its
  continuous coefficients through the forecast loss, using JCM gradients.

We should not assume that one replaces the other. The first research comparison
is:

| Arm | Structure | Coefficients | Question |
| --- | --- | --- | --- |
| Offline symbolic regression | Discovered | Fit offline, then fixed | How far does offline equation fitting transfer into JCM? |
| Separate online calibration | Same discovered form | Initialized from offline fit, then tuned by JEM-Cal | Does forecast-aware gradient calibration improve held-out forecast skill? |
| Joint symbolic search | Discovered | Fitted within the symbolic-regression objective | Does coupling coefficient fitting to structure search help enough to justify its cost? |

These arms need the same candidate forms, observation split, coefficient bounds,
and compute accounting. The primary result is held-out forecast skill, with
parameter count, runtime, and physical-gate failures reported alongside it.

## Effective Use Of JCM Gradients

Gradients are useful after a candidate has a fixed, differentiable structure.
For an SCM window, JEM-Cal can differentiate the masked cloud-trajectory loss
with respect to the candidate coefficients and update them through smooth,
bounded parameter transforms. The practical rules are:

- Use short independent forecast windows and accumulate information over many
  windows. Do not treat a long free rollout as one reliable gradient sample.
- Verify each new scheme with finite-difference versus automatic-differentiation
  tests before using its gradients for calibration.
- Reject or repair a scheme with non-finite gradients, even if its forward
  forecast is finite.
- Use gradients for coefficient fitting and sensitivity diagnosis, not to choose
  among discrete equation strings.
- Keep the loss, QC mask, data split, and parameter bounds in the candidate
  artifact so gradient results remain scientifically comparable.

The immediate calibration baseline is the current default SPEEDY cloud equation.
An identical-twin recovery experiment should precede real-observation
calibration: synthesize targets from known coefficients, recover them with the
same forecast/evaluation path, then move to ARMBE observations.

## Near-Term Milestones

1. **Generic candidate runner:** replace the default `cloudc` diagnostic with a
   supplied `PhysicsTerm`, then run the existing ARMBE cache and evaluator
   unchanged.
2. **Candidate gates and artifacts:** implement the mechanical checks and a
   durable candidate/result record before generating many equations.
3. **Coefficient calibration:** connect a fixed candidate's bounded coefficient
   pytree to JEM-Cal; demonstrate identical-twin recovery and one ARMBE SCM
   calibration.
4. **Discovery baseline:** run symbolic regression on a declared cloud-cover
   feature set, render several candidate equations, and compare the three
   coefficient-learning arms above.
5. **ERA5 promotion case:** take the best SCM survivors into a short global or
   nudged JCM experiment before claiming generality.

## Decisions Still Needed

- Which state and diagnostic variables are permitted as cloud-cover inputs?
- What physical constraints are mandatory for the first candidate grammar?
- Which ARMBE periods/sites form training, validation, and held-out test blocks?
- What is the first ERA5 target: nudged tendency diagnostics, cloud/radiation
  observables, or a short free forecast metric?
- How should multi-lead cloud errors be weighted into a calibration objective?

The present ARMBE target remains an imperfect observation operator:
SPEEDY `cloudc` is compared with narrow-field-of-view ARMBE `tot_cld`. A future
candidate must not be declared better solely from this one metric without the
documented caveat and broader validation.
