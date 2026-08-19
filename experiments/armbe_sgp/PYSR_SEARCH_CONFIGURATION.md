# PySR Search Configuration

This records the effective configuration of the nested atmospheric-state
searches documented in `UNIFIED_EXPERIMENT.md`: the original six 2,000-row
searches and their six 5,000-row repeats. Values were checked against the saved
`result.json` files, AgentSR's PySR wrapper, and the installed PySR 1.5.10
defaults. "PySR default" means the value was not selected for this experiment
and may change if PySR is upgraded.

## Experiment-Selected Settings

| Setting | Value | Role |
| --- | --- | --- |
| Training sample | 2,000 or 5,000 unique rows, without replacement | One fixed sample from the 54,229 training rows |
| Sample and search seed | `20260731` | Seeds NumPy sampling and PySR's `random_state` |
| Iterations | 80 | Search iterations per feature group |
| Populations | 12 | Independent evolving populations |
| Population size | 33 | Expressions per population |
| Cycles per iteration | 200 | Evolutionary work between iteration boundaries |
| Binary operators | `+`, `-`, `*`, `/` | Search primitives |
| Unary operators | `square`, `cube`, `sqrt_abs`, `log_abs`, `exp`, `tanh`, `relu` | Search primitives |
| Operator costs | `/=3`, `square=3`, `cube=3`, `sqrt_abs=3`, `log_abs=3`, `exp=5`, `tanh=3`, `relu=3` | All unlisted operators cost 1 |
| Nested constraints | no `exp` in `exp`; no `log_abs` in `log_abs`; no `sqrt_abs` in `sqrt_abs` | Prevents self-nesting of these operators |
| Original limits | `maxsize=20`, `maxdepth=5` | First three searches |
| Expanded limits | `maxsize=50`, `maxdepth=8` | Only changes in the three expanded searches |

Constants and variables each have the inherited complexity cost 1. There was no
template expression, feature preselection, dimensional constraint, general
operator-argument constraint, size warmup, timeout, evaluation cap, or early-stop
condition.

The wrapper creates the sample once before `model.fit`; every population,
mutation, crossover, loss evaluation, and BFGS fit in that run therefore sees the
same sampled rows. The same seed and identically ordered training table make each
sample-size draw identical across feature groups and both size/depth settings.
With the installed NumPy implementation, the 5,000-row draw contains all 2,000
original rows plus 3,000 additional rows. Sampling without replacement avoids
duplicate weighting and uses the full unique-row budget. All other settings were
unchanged in the 5,000-row repeats.

## AgentSR-Pinned Settings

These are wrapper defaults rather than choices made separately for the nested
experiment. They were nevertheless passed explicitly to PySR.

| Setting | Effective value |
| --- | --- |
| Constant fitting | enabled |
| Optimizer | BFGS with backtracking line search |
| Optimization probability | 0.14 per population member at each iteration boundary |
| Optimizer restarts | 2 |
| Optimizer iterations | 8 |
| Optimizer function-call limit | inherited backend default, 10,000 |
| Warm start | false |
| Denoising | false |
| Parallel execution | multithreading |

BFGS optimization is distinct from the `optimize` tree-mutation weight below,
which is zero. Gradients use the inherited finite-difference behavior because no
automatic-differentiation backend was selected.

## Inherited PySR 1.5.10 Settings

The following search-relevant values came from PySR rather than the experiment or
AgentSR configuration.

| Area | Effective value |
| --- | --- |
| Objective | unweighted mean squared error (`L2DistLoss`), logarithmic loss scaling |
| Explicit parsimony | 0.0 |
| Adaptive complexity frequency | enabled in population costs and tournaments; scaling 1040.0 |
| Tournament | 15 candidates; rank-selection probability 0.982 |
| Population survival | age-regularized evolution: accepted offspring replace the oldest member, not necessarily the least fit |
| Crossover | probability 0.0259; otherwise a one-parent mutation is attempted |
| Simulated annealing | disabled (`alpha=3.17` is therefore inactive) |
| Migration | population migration and hall-of-fame migration enabled |
| Migration fractions | 0.00036 population replacement; 0.0614 hall-of-fame replacement |
| Migration candidates | top 12 |
| Simplification | enabled; failed mutations are skipped |
| Constant mutation | perturbation factor 0.129; negation probability 0.00743 |
| Batching | disabled, so search loss uses all 2,000 or 5,000 sampled rows |
| Numeric precision | 32 bit |
| Fast/turbo/bumper modes | disabled |
| Search determinism | disabled |
| PySR model selection | `best`, though the study does not use it for final selection |

PySR's mutation weights are relative weights normalized at runtime and may be
conditioned when a mutation is invalid for the current tree:

| Mutation | Weight | Unconditioned share |
| --- | ---: | ---: |
| Rotate tree | 4.26 | 50.640% |
| Add node | 2.47 | 29.361% |
| Delete node | 0.870 | 10.342% |
| Mutate operator | 0.293 | 3.483% |
| Do nothing | 0.273 | 3.245% |
| Swap operands | 0.198 | 2.354% |
| Mutate constant | 0.0346 | 0.411% |
| Insert node | 0.0112 | 0.133% |
| Simplify | 0.00209 | 0.025% |
| Randomize | 0.000502 | 0.006% |
| Optimize constants as a mutation | 0.0 | 0.000% |

These weights were not inferred from ARMBE. The installed backend describes its
current defaults as empirically tuned from benchmark discussion values, and its
v1.0 changelog says revised hyperparameter defaults were selected using Pareto
front volume rather than only the accuracy of the single best expression. The
installed source does not provide a separate analytical derivation for each
weight. They should therefore be treated as changeable library heuristics and
pinned explicitly in any future strict-reproduction run.

## Selection and Reproducibility

The `best_equation` field in AgentSR's `result.json` is the search wrapper's
maximum-score choice and is not the reported scientific result. The study
evaluates every saved Pareto-front equation on all validation rows, selects the
minimum validation RMSE, and evaluates only that equation on test. PySR's
training score, adaptive complexity penalty, and `model_selection="best"` do not
replace this external procedure.

The row draw is exactly reproducible. The evolutionary trajectory is not
guaranteed bitwise reproducible despite `random_state=20260731`, because these
runs use multithreading with PySR's `deterministic=false`. Exact reruns should
also pin PySR/SymbolicRegression.jl versions and use serial execution with
deterministic mode enabled.
