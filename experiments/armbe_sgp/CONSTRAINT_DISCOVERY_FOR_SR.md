# Discovering Physical Constraints Before Symbolic Regression

## Research Question

Predictive accuracy does not establish that a symbolic expression is physically
valid. The upstream research question is:

> Which low-complexity equalities, inequalities, monotonicities, thresholds,
> invariances, and regime dependencies are reproducible across environments and
> scales, and with what uncertainty should they restrict symbolic regression?

This differs from conventional physics-informed ML. Known physics remains one
source of constraints, but candidate constraints inferred from data are treated
as empirical objects that can be supported, qualified, or falsified.

## Known Versus Discoverable Constraints

Known hard constraints generally should not be rediscovered from finite data:

```text
dimensional consistency
mass and energy conservation
nonnegative mass and condensate
0 <= cloud fraction <= 1
positive diffusivity
finite outputs over the physical domain
```

Potentially data-discoverable behavior includes:

```text
monotonic response
saturation
activation thresholds
convexity or concavity
important and negligible interactions
regime boundaries
approximate invariance across environments
dependence on spatial or temporal averaging scale
```

These empirical behaviors should not be called physical laws without stronger
causal or experimental evidence.

## Data-Cube Perturbations

Let the atmospheric state and target be:

```text
x = [temperature, humidity, pressure, condensate, stability, wind, ...]
y = cloud fraction, precipitation, radiative flux, tendency, ...
```

First fit one or more flexible conditional response models:

```text
y_hat = f(x)
```

Around each observed state, construct small perturbations of selected inputs and
estimate:

```text
local slope:       dy/dx_j
curvature:         d2y/dx_j2
interaction:       d2y/(dx_j dx_k)
finite response:   f(x + delta) - f(x)
```

Aggregating these quantities over the dataset gives derivative-sign,
saturation, threshold, and interaction maps. The analysis can ask where a
response has one sign, where it changes sign, and whether those patterns remain
stable across sites, seasons, climates, and resolutions.

The phrase `data-cube perturbation` is useful descriptively but is not a standard
method name. Related established methods include response-surface methodology,
individual conditional expectation, accumulated local effects, derivative-based
sensitivity analysis, and conditional or manifold-respecting perturbations.

## Stay On The Physical Data Manifold

Atmospheric predictors are dependent. Independently perturbing temperature,
humidity, pressure, and condensate can generate combinations that do not occur
physically. A model response at such a point characterizes extrapolation, not an
observationally supported relationship.

Prefer:

```text
accumulated local effects within populated bins
conditional perturbations sampled from p(x_j | x_-j)
regional or subgroup effect analysis
learned-manifold perturbations
perturbations in physically meaningful coordinates
```

Useful physical coordinates may include relative humidity, potential
temperature, saturation deficit, moist static energy, dimensionless stability
measures, and conserved moist variables. Basic thermodynamic validity checks
should reject inadmissible perturbed states.

## Discovery Workflow

### 1. Fit Multiple Response Models

Use several flexible model classes, such as boosted trees, neural networks,
Gaussian processes, random forests, and local regression. A candidate constraint
is less credible if it appears only under one fitted model class.

### 2. Estimate Conditional Effects

Use accumulated local effects, individual conditional expectation, signed local
derivatives, finite differences, and interaction surfaces. Retain distributions
of local effects instead of only their global average, which can hide
sign-reversing regimes.

### 3. Condition On Regimes

Evaluate candidate behavior separately across:

```text
sites and deployments
seasons and years
land and ocean
day and night
stable and unstable boundary layers
cloud and precipitation regimes
present and perturbed climates
model resolutions and averaging scales
```

For example, test the sign of cloud response to stability conditional on
humidity and boundary-layer regime rather than asserting one global monotonic
relationship.

### 4. Quantify Uncertainty

Use block bootstraps over time, site, year, or deployment instead of treating
dense rows as independent. Candidate evidence can be summarized as:

```text
P(derivative >= 0 | regime)
fraction of supported state space satisfying the constraint
bootstrap interval for a threshold or saturation point
```

Multiple derivative cells and candidate constraints require multiplicity-aware
inference or independent confirmation.

### 5. Test Invariance Across Environments

Evidence is stronger when a relation survives SGP, NSA, ENA, mobile campaigns,
seasons, LES datasets, observations, and altered climates. Classify constraints
as global, regime-specific, or dataset-specific instead of forcing one equation
to represent every environment.

### 6. Pass Supported Structure Into SR

Hard, well-established constraints can restrict the representation or grammar.
Uncertain data-supported constraints should be soft penalties, probabilistic
requirements, or post-search selection criteria.

### 7. Validate Symbolic Equations

Repeat the perturbation analysis on each symbolic candidate. Test physical
bounds, finite extrapolation, derivative behavior, held-out environments,
resolution transfer, conservation, and stability when coupled into JCM.

## Enforcing Constraints In Symbolic Regression

Constraint enforcement can occur at several stages.

### Representation By Construction

Choose an expression form that cannot violate the constraint:

```text
bounded fraction:     sigmoid(g(x)) or 0.5 * (tanh(g(x)) + 1)
nonnegative output:   softplus(g(x)), exp(g(x)), or g(x)^2
positive diffusivity: exp(g(x))
normalized partition: softmax(g_1(x), ..., g_n(x))
```

This is the strongest approach when the constraint is exact. The transform can
affect interpretability and saturation, so it remains a scientific design
choice.

### Restricted Operators And Grammar

Limit the search language to dimensionally and numerically valid expressions:

```text
allow addition only between like units
require dimensionless arguments to exp, log, and trigonometric functions
use safe division only where the denominator has a physical lower bound
forbid singular operator nesting
construct expressions from dimensionless groups
```

Grammar restrictions prevent invalid candidates from consuming search effort.

### Constrained Coefficients

Parameterize coefficients to enforce signs or ranges:

```text
a = softplus(alpha)      ensures a >= 0
a = a_min + (a_max - a_min) * sigmoid(alpha)
```

This works when monotonicity follows directly from coefficient signs. In more
complex expressions, positive coefficients alone may not guarantee a global
derivative sign.

### Penalized Fitness

Evaluate a candidate on a physically supported constraint grid and augment its
prediction loss:

```text
fitness = data_loss
        + lambda_bound * mean(bound_violation^2)
        + lambda_mono  * mean(max(0, -dy_hat/dx_j)^2)
        + lambda_cons  * mean(conservation_residual^2)
```

This supports approximate or uncertain constraints. Penalty scales must be
reported, and finite sampled checks do not prove global satisfaction.

### Feasibility Filtering

Reject candidates before or after coefficient fitting when they violate:

```text
units
finite-domain requirements
hard bounds
known conservation identities
monotonicity over an accepted physical domain
```

For low-dimensional symbolic expressions, interval arithmetic or satisfiability
tools can sometimes verify a property over a continuous domain. Grid tests alone
can miss narrow violations between sampled points.

### Multi-Objective Selection

Treat prediction error, symbolic complexity, and physical violation as separate
Pareto objectives instead of hiding them in one weighted score:

```text
minimize [validation error, complexity, constraint violation]
```

This exposes the tradeoff and avoids pretending that one arbitrary penalty
weight defines the scientifically best equation.

### Regime-Specific Structure

If a constraint is supported only in named regimes, fit a smooth mixture or
piecewise equation with a physically interpretable gate. Do not impose a local
monotonicity result globally.

## LES And Other Evidence Sources

LES stands for **Large-Eddy Simulation**. It explicitly resolves the larger
turbulent eddies while parameterizing smaller subgrid turbulence. Atmospheric
LES provides high-frequency three-dimensional fields and process tendencies
that routine observations often cannot measure directly.

LES is useful for controlled perturbations, process budgets, and constraint
discovery, but it is simulated evidence rather than independent truth. A robust
constraint should ideally be discovered or screened in LES and then tested
against ARM, satellite, or other observational environments.

## Selected Literature

| Topic | Reference | Contribution |
| --- | --- | --- |
| Response surfaces | Box and Wilson (1951), https://doi.org/10.1111/j.2517-6161.1951.tb00067.x | Local factorial perturbations, gradients, curvature, and interactions |
| Partial dependence | Friedman (2001), https://doi.org/10.1214/aos/1013203451 | Feature-grid response surfaces, with off-manifold risks |
| Individual conditional expectation | Goldstein et al. (2015), https://doi.org/10.1080/10618600.2014.907095 | Heterogeneous local response curves and derivative behavior |
| Accumulated local effects | Apley and Zhu (2020), https://doi.org/10.1111/rssb.12377 | Conditional effects under correlated predictors |
| Dependent-feature decomposition | Hooker (2007), https://doi.org/10.1198/106186007X237892 | Functional decomposition for dependent inputs |
| Regional effects | Herbinger et al. (2022), https://proceedings.mlr.press/v151/herbinger22a.html | Data-driven regions with distinct response behavior |
| Monotonicity tests | Hall and Heckman (2000), https://doi.org/10.1214/aos/1016120363 | Inferential support for monotonic regression relationships |
| Derivative sensitivity | Sobol and Kucherenko (2009), https://doi.org/10.1016/j.matcom.2009.01.023 | Global sensitivity based on partial derivatives |
| Invalid perturbations | Hooker and Mentch (2019), https://arxiv.org/abs/1905.03151 | Why independent permutations create unsupported states |
| Invariant prediction | Peters, Buhlmann, and Meinshausen (2016), https://doi.org/10.1111/rssb.12167 | Stable conditional relations across environments |
| Stable kinetic structures | Pfister, Bauer, and Peters (2019), https://doi.org/10.1073/pnas.1815006116 | Differential structures ranked by cross-environment stability |
| Conservation discovery | Liu and Tegmark (2021), https://doi.org/10.1103/PhysRevLett.126.180604 | Conserved quantities learned from trajectories |
| Dimensionless groups | Bakarji et al. (2022), https://doi.org/10.1038/s43588-022-00355-5 | Learning useful Buckingham-Pi groups |
| Symbolic laws | Schmidt and Lipson (2009), https://doi.org/10.1126/science.1165893 | Foundational free-form law discovery |
| Physics-guided SR | Udrescu and Tegmark (2020), https://doi.org/10.1126/sciadv.aay2631 | Symmetry, separability, and compositional tests |
| Climate constraints | Beucler et al. (2021), https://doi.org/10.1103/PhysRevLett.126.098302 | Enforcing analytic constraints in climate ML |
| Cloud equation discovery | Grundner et al. (2024), https://doi.org/10.1029/2023MS003763 | Symbolic cloud-cover parameterization |

## Proposed Research Pipeline

```text
observational and LES datasets
    -> manifold-respecting perturbation cubes
    -> signed derivative and interaction maps
    -> block-bootstrap uncertainty
    -> invariance tests across environments and scales
    -> probabilistic candidate constraints
    -> constrained symbolic regression
    -> held-out environment and online JCM validation
```

The central opportunity is to make constraint discovery a formal,
uncertainty-aware stage before equation discovery instead of choosing constraints
only from intuition or inspecting physical behavior after fitting.
