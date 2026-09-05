.. _bias_correction:

Neural Bias Correction
======================

Overview
--------

Intermediate-complexity models such as SPEEDY buy their speed by simplifying
the physics, and the price is a systematic error that repeats from year to
year. This module learns that error and removes it from inside the running
model, rather than post-processing the output.

The correction is a small multilayer perceptron applied independently to every
atmospheric column, with weights shared across the whole grid. It is trained
**through** the model: rollouts are differentiated end to end, so the loss sees
the downstream consequences of a correction rather than only its instantaneous
value. Once trained, the network reads the model's own state and needs no
reanalysis at run time.

.. note::

   This distinguishes the approach from nudging. Nudging requires the target
   field at every step and therefore only works over periods that have already
   been observed. The correction here uses ERA5 as a training signal only.

Architecture
------------

Each column supplies ``4 * nlev`` values, the profiles of temperature, specific
humidity and both wind components, in the order given by
:data:`~jcm.physics.bias_correction.nn_bias_correction.FIELD_ORDER`. At the
default T31L8 configuration that is 32 inputs per column and 4,608 columns.

Optional per-column *context features* are appended to the standardised
profiles. These carry information a single column cannot infer from its own
state, such as whether it sits over land or how much sunlight it is receiving.
The shipped term uses one, ``insol``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Component
     - Description
   * - Input
     - ``4 * nlev`` standardised profile values plus ``n_context`` features
   * - Hidden layers
     - Three layers of 256 units, ``tanh`` activation
   * - Output
     - ``4 * nlev`` values, scaled by ``out_scale`` into per-second tendencies
   * - Applied to
     - Temperature and specific humidity; wind outputs are computed and discarded
   * - Parameters
     - 148,512 for the shipped term, shared by every column

Inputs are standardised per level and per field against climatological
statistics, because temperature and specific humidity differ by several orders
of magnitude and an unstandardised network cannot learn from both. Context
features bypass this step, since
:func:`~jcm.physics.bias_correction.nn_bias_correction.build_context`
normalises them to order one as it builds them.

Training
--------

Four stages, each addressing a specific failure of the one before it.

**Stage 1, offline.** Run the model nudged toward ERA5 and record the nudging
tendency, which estimates the model's instantaneous error. Fit the network to
predict it. This converges quickly but optimises a proxy: a network that scores
well here can still degrade the free-running climate, because offline its
mistakes have no downstream consequences.

**Stage 2, online rollout.** Start from a balanced near-ERA5 state, integrate
forward with the correction active, and compare the final state against ERA5 at
that lead time. The error is backpropagated through every timestep into the
weights. Rollout length grows across a curriculum of 12, 24, 48 and 96 steps,
which at a 30-minute timestep is 6 h, 12 h, 24 h and 48 h. The curriculum stops
there because gradients through a chaotic system grow without bound as the
window lengthens.

**Stage 3, climatology.** Short rollouts reward short-horizon accuracy, which
is not the same objective as a faithful long-run climate. This stage fine-tunes
against long-period means.

**Stage 4, surface taper.** Applied after training rather than learned. The
raw correction improves the mid-troposphere while degrading the near-surface
temperature, so it is faded smoothly to zero in the lowest model levels by
:func:`~jcm.physics.bias_correction.nn_bias_correction.surface_taper_factor`.

Evaluation protocol
-------------------

Results are area-weighted RMS errors of the time-mean bias against ERA5, taken
from free-running integrations with no nudging and no reanalysis input. The
first 90 days of every run are discarded as spin-up.
:mod:`~jcm.physics.bias_correction.eval_protocol` pins the canonical
configuration so published numbers remain reproducible.

The shipped term was trained on 1995-2015 and is evaluated here on 2016-2022,
seven years absent from training. Each configuration is compared against plain
SPEEDY **over the same period**: the model's prescribed ocean is a fixed
climatology, so its own error is not constant across eras, and a single fixed
baseline would flatter or penalise a term according to which years it was
tested on.

Every score on this page was produced on the SPEEDY physics as of commit
748413a (July 2026), the base this work was developed on. The surface flux and
vertical diffusion schemes changed on ``dev`` after that. The term loads,
composes and runs on the current code, but the free-run scores have not been
recomputed against the current plain SPEEDY; rerunning stage 5 of the
reproduction chain for ``plain`` and for the shipped term is the first thing
to do before quoting these numbers for the current physics.

Sampling noise and window length
--------------------------------

Every metric here is read off a finite mean of a free-running integration, so
it carries the model's own internal variability. Seven years does not average
that out. Two things follow, and both are measurable rather than matters of
opinion.

A run can be scored in consecutive blocks. Set ``M4_BLOCK_YEARS`` and give
``tools/bias_correction/evaluate_term.py`` a run several blocks long: it writes an extra ``_blkN`` file per
block, each holding the same trained network and a different stretch of the
same trajectory. The spread of a metric across those blocks is sampling noise
with the network held fixed, which is the smallest difference between two
terms that can mean anything at that window length. Blocks are anchored at the
spin-up drop and are a whole number of years long, so the first block of a long
run covers exactly the days the 2645-day holdout scored.

A run can also simply be longer. The free run is forced by a climatology, so
its calendar year selects which ERA5 years the reference averages and nothing
else; lengthening the run buys down the model side of the sampling noise and
leaves the observed side untouched, and the observed side is common to every
configuration so it cannot change a ranking. ``M4_CHUNK_DAYS`` integrates such
a run in segments, reducing each to per-season sums before the next, so a
21-year run does not have to be held in memory at once. Segments are threaded
through ``Model.resume``, which carries the cross-step physics state, so they
form one trajectory rather than a sequence of restarts.
``tools/bias_correction/run_longeval.sh`` drives the whole thing.

Comparisons against plain SPEEDY are unaffected by any of this: those gaps are
several times larger than the spread. What the spread governs is comparisons
between corrected terms, which are much closer together.

Results
-------

Held-out evaluation, 2016-2022. Lower is better.

+----------------------------+----------------+------------------+------------+
| Metric                     | plain SPEEDY   | with correction  | reduction  |
+============================+================+==================+============+
| Near-surface T, annual     | 3.64 K         | 3.29 K           | 10 %       |
+----------------------------+----------------+------------------+------------+
| Near-surface T, DJF        | 3.60 K         | 3.32 K           | 8 %        |
+----------------------------+----------------+------------------+------------+
| Near-surface T, JJA        | 4.17 K         | 3.87 K           | 7 %        |
+----------------------------+----------------+------------------+------------+
| T at 500 hPa, annual       | 4.73 K         | 2.53 K           | 46 %       |
+----------------------------+----------------+------------------+------------+
| Specific humidity, annual  | 1.46 g/kg      | 0.79 g/kg        | 46 %       |
+----------------------------+----------------+------------------+------------+

The correction improves every metric. Gains are largest in the free
troposphere and in moisture, and smallest near the surface, where a single
column carries little information about the underlying surface type.

These are the numbers for one training run. The shipped recipe was trained
twice, from seeds 0 and 7, and beat plain SPEEDY on all five metrics both
times (the seed table below). The same architecture without the insolation
input was trained three times, and there the spread is much larger: up to
0.89 K on annual near-surface temperature and 2.79 K at 500 hPa, with one of
the three seeds improving three metrics rather than five. The improvement over
plain SPEEDY is far larger than either spread and is not in question. A
ranking of one corrected term against another on this window is not
supported by the evaluation: ``tools/bias_correction/holdout_table.py``
reports the measured floors and a paired within-seed comparison, and neither
establishes an ordering among the 148k terms.

Network capacity
^^^^^^^^^^^^^^^^

December-February near-surface temperature was not improved by any
configuration until the network was enlarged. The following comparison
isolates the cause.

+------------------------+-----------+-----------------+-----------+
| Configuration          | Weights   | Training years  | DJF RMS   |
+========================+===========+=================+===========+
| plain SPEEDY           | n/a       | n/a             | 3.60 K    |
+------------------------+-----------+-----------------+-----------+
| ``clim_vt``            | 8,352     | 2001            | 3.69 K    |
+------------------------+-----------+-----------------+-----------+
| ``20yr_clim_vt``       | 8,352     | 1995-2015       | 3.89 K    |
+------------------------+-----------+-----------------+-----------+
| ``big_clim_vt``        | 148,256   | 1995-2015       | 3.55 K    |
+------------------------+-----------+-----------------+-----------+
| ``big_fmask_vt``       | 148,512   | 1995-2015       | 3.39 K    |
+------------------------+-----------+-----------------+-----------+
| ``big_insol_vt``       | 148,512   | 1995-2015       | 3.32 K    |
+------------------------+-----------+-----------------+-----------+

The 8,352-parameter network fails on this metric whether it is trained on one
year or twenty one, and additional data makes it worse, which is characteristic
of underfitting rather than of missing information. At 148k parameters the
metric improves with a seasonal input, with a static land-sea mask, or with no
context feature at all. The context features account for 256 weights, so
capacity rather than feature choice is the operative difference.

Seed sensitivity
^^^^^^^^^^^^^^^^

Two independent trainings differing only in weight initialisation.

+----------------------------+--------------+--------------+
| Metric                     | seed 0       | seed 7       |
+============================+==============+==============+
| Near-surface T, annual     | 3.29 K       | 3.23 K       |
+----------------------------+--------------+--------------+
| Near-surface T, DJF        | 3.32 K       | 3.38 K       |
+----------------------------+--------------+--------------+
| Near-surface T, JJA        | 3.87 K       | 3.85 K       |
+----------------------------+--------------+--------------+
| T at 500 hPa, annual       | 2.53 K       | 3.70 K       |
+----------------------------+--------------+--------------+
| Specific humidity, annual  | 0.79 g/kg    | 1.00 g/kg    |
+----------------------------+--------------+--------------+

For this recipe the count of improved metrics reproduced; the magnitude at
500 hPa did not, moving by 1.2 K between the two seeds. The no-context recipe
did not reproduce its count at all (three seeds: five, three and five metrics
improved). Quote the 5 of 5 as a two-seed result and the 500 hPa reduction as
a range, 2.5 to 3.7 K against plain's 4.73 K.

What did not work
-----------------

Every configuration below went through the same chain and was scored the same
way. They are listed because the shipped recipe is what is left after these
results, and because several of them overturned a conclusion that had already
been written down.

.. list-table::
   :header-rows: 1
   :widths: 26 52 22

   * - Tried
     - What happened
     - Verdict
   * - Offline term run on its own
     - Non-finite or wildly biased within a 450-day free run. The rollout stage
       is what makes a term stable.
     - Refuted; stage 2 is mandatory
   * - Latitude (polar) taper
     - Removed the polar overshoot and most of the 500 hPa gain with it (2.6 to
       3.2 K), because plain SPEEDY's mid-tropospheric error is itself polar.
     - Refuted; height is the right axis
   * - Training with the surface taper active
     - The network re-routed the polar warming through the untapered levels
       (polar mean bias back to +2.3 K). Masking after training beats
       retraining.
     - Refuted
   * - Land-sea mask input, 8k parameters
     - Best surface score of the small networks, 500 hPa worse than plain,
       winter unchanged.
     - Refuted at 8k; works at 148k
   * - Insolation input, 8k parameters
     - Same pattern: surface improved, aloft lost, winter unchanged.
     - Refuted at 8k; works at 148k
   * - Loss weighted toward 500 hPa
     - Made 500 hPa worse (3.3 to 4.9 K); the free-run response overshoots.
     - Refuted
   * - 21 training years, 8k parameters
     - Winter worse (3.69 to 3.89 K) and 500 hPa past plain. More data without
       more capacity backfires.
     - Refuted at 8k
   * - Retraining at T63
     - Worse than running the T31 term at T63 with no retraining, which still
       beat plain on 4 of 5.
     - Refuted
   * - Sea-ice fraction input, 148k
     - 5 of 5 against plain, but DJF 3.44 K against the insolation term's
       3.32 K.
     - Not better
   * - Ensemble mean of trained terms
     - Lands near the member average, not the best member: the members are
       different biased corrections, not independent noise.
     - Refuted
   * - Selecting the network on offline error
     - Offline error fell 28% across a width and activation sweep while the
       climate score went from 5 of 5 to 3 of 5. No correlation.
     - Refuted; select on climate metrics
   * - Seasonal climatology windows
     - A first run improved DJF by 0.17 K; the paired three-seed repeat flipped
       sign (+0.40, +0.15 K). The improvement was the seed.
     - Not established

Not yet tested
^^^^^^^^^^^^^^

Context features retrained with a raised learning rate on the new input row
(the shipped context rows trained so little that they are numerically inert,
so no context feature except sea ice has had a fair test); activation and
width selected on the climate score rather than on offline error; a held-out
region rather than held-out years, which would settle whether the network
memorises a per-location correction; a neighbouring-column stencil; transfer
to a different physics package.

Shipped terms
-------------

One trained term ships: ``jcm/data/bias_correction/online_term_t31_big_insol_vt.npz``,
the reference term. The file carries its own normalisation, output scale,
context feature and taper, so nothing else is needed to run it. The other
terms named above back the comparisons on this page and are kept in the
author's research repository rather than here, since nothing in this
repository loads them; ``TERMS_README.md`` lists them with their scores, and
the chain below regenerates any of them.

.. code-block:: python

   from jcm.model import Model
   from jcm.physics.bias_correction import load_bias_correction
   from jcm.physics.speedy.speedy_coords import get_speedy_coords
   from jcm.physics.speedy.speedy_terms import speedy_physics

   coords = get_speedy_coords(layers=8, spectral_truncation=31)
   term = load_bias_correction(
       "jcm/data/bias_correction/online_term_t31_big_insol_vt.npz")
   model = Model(coords=coords, terrain=terrain,
                 physics=speedy_physics() + term)

The term is an ordinary :class:`~jcm.physics.physics_term.PhysicsTerm` with
empty ``requires`` and ``provides``, so it composes anywhere in the stack.
Append it last.

Reproducing these results
-------------------------

The full chain, from an empty cache to a scored term, with every argument.
Every stage needs a GPU except the taper. The drivers live in
``tools/bias_correction/``; its README documents the environment knobs.

.. code-block:: bash

   # 0. per-year ERA5 and nudged-state cache. Lives outside the repo,
   #    roughly 1.1 GB per year, and is resumable.
   python tools/bias_correction/build_era5_cache.py --years 1995-2015
   python tools/bias_correction/build_era5_cache.py --years 2016-2022

   # 1. offline warm start
   python tools/bias_correction/train_offline_multiyear.py --years 1995-2015 --val-years 2016-2022 \
       --hidden 256,256,256 \
       --out jcm/data/bias_correction/offline_term_t31_big.npz

   # 2. rollout curriculum
   python tools/bias_correction/train_online_multiyear.py \
       jcm/data/bias_correction/offline_term_t31_big.npz \
       jcm/data/bias_correction/online_term_t31_big.npz \
       --years 1995-2015 --start-pool 21 --updates 500,500,500,250

   # 3. climatology fine-tune, adding the insolation input
   python tools/bias_correction/finetune_climatology.py \
       jcm/data/bias_correction/online_term_t31_big.npz \
       jcm/data/bias_correction/online_term_t31_big_insol.npz \
       --years 1995-2015 --start-pool 21 --n-steps 1440 --rollout-steps 48 \
       --lam 1.0 --updates 210 --lr 2.5e-5 --pole-floor 0.3 --context insol

   # 4. surface taper (no GPU)
   CUDA_VISIBLE_DEVICES="" python tools/bias_correction/add_surface_taper.py \
       jcm/data/bias_correction/online_term_t31_big_insol.npz \
       jcm/data/bias_correction/online_term_t31_big_insol_vt.npz 0.7 1.0

   # 5. score on the held-out years
   M4_YEAR=2016 M4_DAYS=2645 M4_OUT_DIR=$PWD/eval_out_holdout \
       M4_TERM=jcm/data/bias_correction/online_term_t31_big_insol_vt.npz \
       M4_TAG=big_insol_vt python tools/bias_correction/evaluate_term.py

Drop ``--context insol`` in step 3 to reproduce ``big_clim_vt``, or use
``--context fmask`` for ``big_fmask_vt``. Drop ``--hidden 256,256,256`` in
step 1 to reproduce the 8,352-parameter network.

Three settings in this recipe were found by the term going non-finite in
evaluation rather than by reasoning ahead: the rollout stage needs roughly 1750
updates, the climatology window must be 30 days rather than 90, and rollout
starts must be spread across the seasonal cycle as well as across years.
``tools/bias_correction/README.md`` records what each of those failures
looked like.

Limitations
-----------

- **Climate, not weather.** Every quantity reported is a multi-year mean. The
  correction is not evaluated at synoptic timescales.
- **Configuration-specific.** The shipped term is trained for SPEEDY at T31L8.
  An earlier 8k term run at T63 without retraining still beat plain on 4 of 5
  metrics, and retraining it at T63 scored worse than that transfer. The
  shipped term has not been tried at T63, and transfer to a different physics
  package is untested.
- **Column-local.** Each column is corrected without reference to its
  neighbours, so error driven by horizontal structure is out of reach by
  construction. This is a plausible contributor to the modest near-surface
  gains.
- **Not attributable.** The correction reduces the aggregate error but does
  not identify which parameterisation is deficient.

Module reference
----------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Module
     - Purpose
   * - :mod:`~jcm.physics.bias_correction.nn_bias_correction`
     - The ``PhysicsTerm``, the network, and context features
   * - :mod:`~jcm.physics.bias_correction.offline_training`
     - Stage 1: supervised fit to recorded nudging tendencies
   * - :mod:`~jcm.physics.bias_correction.online_training`
     - Stage 2: rollout loss and curriculum driver
   * - :mod:`~jcm.physics.bias_correction.era5_data`
     - ERA5 on the model grid, leap-safe across multi-year spans
   * - :mod:`~jcm.physics.bias_correction.era5_cache`
     - Per-year on-disk cache for multi-year training
   * - :mod:`~jcm.physics.bias_correction.multiyear`
     - Sampling start states across seasons and years
   * - :mod:`~jcm.physics.bias_correction.eval_protocol`
     - The pinned evaluation configuration
   * - :mod:`~jcm.physics.bias_correction.optim`
     - Shared Adam with gradient clipping

Repository map
--------------

Everything importable lives in ``jcm/physics/bias_correction/`` and is listed
above. The drivers live in ``tools/bias_correction/``: they chdir to the repo
root, assume a GPU for everything except the taper and the tables, and are run
once per experiment rather than imported.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Script
     - Role
   * - ``build_era5_cache.py``
     - Stage 0. Builds the per-year ERA5 and nudged-state cache.
   * - ``train_offline_multiyear.py``
     - Stage 1. Supervised warm start across the cached years.
   * - ``train_online_multiyear.py``
     - Stage 2. Rollout curriculum across seasons and years.
   * - ``finetune_climatology.py``
     - Stage 3. Long-horizon climatology or combined fine-tune. Also where a
       context feature is added and the first layer widened.
   * - ``add_surface_taper.py``
     - Stage 4. Applies the post-hoc near-surface taper that makes a term
       ``_vt``.
   * - ``evaluate_term.py``
     - Runs a free simulation and writes the mean fields used for scoring.
   * - ``holdout_table.py``
     - Scores every term in an evaluation directory against the measured
       floors and runs the paired within-seed comparison. Needs no GPU.
   * - ``perturb_term.py``
     - Weight-perturbed copies of a term, for the perturbation floor.
   * - ``run_longeval.sh``
     - A 21-year free run cut into 7-year blocks, for the sampling floor.

Which artifact backs which claim: the held-out scorecard comes from
``online_term_t31_big_insol_vt.npz``; the capacity argument compares it against
``online_term_t31_clim_vt.npz`` and ``online_term_t31_20yr_clim_vt.npz`` at
8,352 parameters; the claim that the context feature is not doing the work
compares it against ``online_term_t31_big_clim_vt.npz`` and
``online_term_t31_big_fmask_vt.npz`` at the same width. Only the first of these
ships. The others are regenerated by the chain above or shared on request, and
their scores are recorded in ``TERMS_README.md``.

References
----------

The nudge-then-learn-the-nudge construction of stages 1 and 2 follows
Watt-Meyer et al. (2021); training through the rollout follows Rasp (2020);
the dynamical core is the one described in Kochkov et al. (2024), and the
physics is SPEEDY (Molteni, 2003). None of the method is new; the contribution
here is the differentiable online training inside this code base and the
record of what did and did not work.

- Kochkov, D., et al. (2024). Neural general circulation models for weather
  and climate. *Nature*, 632, 1060-1067.
- Molteni, F. (2003). Atmospheric simulations using a GCM with simplified
  physical parametrizations. I: Model climatology and variability in
  multi-decadal experiments. *Climate Dynamics*, 20, 175-191.
- Rasp, S. (2020). Coupled online learning as a way to tackle instabilities and
  biases in neural network parameterizations: general algorithms and Lorenz 96
  case study. *Geoscientific Model Development*, 13, 2185-2196.
- Watt-Meyer, O., Brenowitz, N. D., Clark, S. K., Henn, B., Kwa, A., McGibbon,
  J., Perkins, W. A., and Bretherton, C. S. (2021). Correcting weather and
  climate models by machine learning nudged historical simulations.
  *Geophysical Research Letters*, 48, e2021GL092555.
