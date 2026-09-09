Running at scale — CLI, Hydra and batch jobs
============================================

The :doc:`getting_started` guide drives the model from Python. This page is the
one place in the documentation that speaks **Hydra**: the ``python -m jcm.main``
command-line interface, the config-group tree it composes, the ``+``/``++``
override rules that trip people up, and the chunked/resumable, containerised and
batch-queue patterns that production runs use. Everything a run needs — physics,
grid, forcing, checkpointing — is expressible as ``python -m jcm.main`` with
Hydra groups and overrides; there are no bespoke driver scripts.

Command-line interface
----------------------

Most simulations can be launched without writing any Python. ``jcm/main.py`` is
executable, so it can be invoked either as a module or directly::

   ./jcm/main.py                                               # direct invocation
   python -m jcm.main                                          # equivalent module form

   # Default 10-day SPEEDY aquaplanet
   python -m jcm.main

   # ECHAM T63L47 with the production RRTMGP radiation (physics=echam IS the
   # RRTMGP one-moment package; there is no separate ``echam-rrtmgp`` group)
   python -m jcm.main physics=echam grid=echam_t63_l47_hybrid

   # Held-Suarez dynamical-core test
   python -m jcm.main physics=held_suarez grid=held_suarez_t31_l8 \
       run.total_time=30 run.save_interval=1

   # Chunked, resumable long run
   python -m jcm.main physics=echam grid=echam_t63_l47_hybrid run=longrun

The state-file modes (``run.mode=scm`` and ``run.mode=prescribed``) read a
netCDF written by an earlier run::

   python -m jcm.main run.mode=scm run.state_file=path/to/state.nc \
       run.column.lat_deg=0 run.column.lon_deg=180

Both the vertical orientation and the tracer list are handled for you: output
files are surface-first and are flipped into the top-first physics frame on
load, and with ``run.tracer_vars`` unset (the default) every tracer the
configured physics declares — ``qc``/``qi`` for the one-moment cloud scheme,
plus ``qnc``/``qni`` for the two-moment one — is loaded from the file when it
carries it. Pass an explicit mapping to rename variables, or
``run.tracer_vars={}`` to load none.

Config groups
-------------

Config groups live under ``jcm/config/``. Selecting a different option for a
group is a bare ``group=option`` override:

============= =================================================================
Group         What it selects
============= =================================================================
``physics``   ``speedy``, ``held_suarez``, ``echam`` (RRTMGP 1-moment),
              ``echam-rrtmgp-2m``, ``echam-emulated-2m``, the ``echam-jam*``
              prognostic-aerosol packages, …
``grid``      ``speedy_t31_l8``, ``held_suarez_t31_l8``,
              ``echam_t{63,106,119}_l{47,95}_hybrid``,
              ``echam_t85_l47_hybrid``, …
``run``       ``default``, ``longrun``, ``smoke``, ``pyses_year``
``init``      ``isothermal``, ``balanced_isothermal``, ``jw``, ``era5``,
              ``from_state``
``terrain``   ``aquaplanet``, ``auto`` (native-grid orography), ``from_file``
``forcing``   ``default``, ``amip``, ``era5``, ``from_file``, ``macv2_sp``
``nudging``   ``none``, ``era5``
``diffusion`` ``default``, ``strong``
``dycore``    ``dinosaur`` (default), the ``pyses_*`` spectral-element
              (CAM-SE) backends
============= =================================================================

Inspect the available choices and the fully-composed config::

   python -m jcm.main --help                                   # config-group choices
   python -m jcm.main --cfg job                                # composed config
   python -m jcm.main --cfg job grid=echam_t63_l47_hybrid      # with overrides

Validated configurations — the ``configuration`` group
-------------------------------------------------------

Composing a run by hand (``physics=… grid=… init=… run=… terrain=… forcing=…``)
is powerful but easy to get subtly wrong — an isothermal cold start with no
sponge, for instance, goes NaN within days at L47. The ``configuration`` group
promotes each *known-good* combination to a single named composition, so one
command is one validated configuration::

   python -m jcm.main +configuration=t63-echam-jam     # T63L47 ECHAM + JAM aerosol
   python -m jcm.main +configuration=speedy-t31        # SPEEDY T31L8 reference
   python -m jcm.main +configuration=ma-t63-l95        # middle-atmosphere JAM sweep

Note the leading ``+``: a configuration is *added* to the default composition
and then overrides the physics/grid/init/run/terrain/forcing groups it selects.
Each ``jcm/config/configuration/*.yaml`` carries comments explaining WHY every
setting is what it is (the dry-JW init, the production sponge, the
semi-Lagrangian off-centering, the level-matched ozone, …), and is the single
source of truth for that configuration — ``tools/benchmark.py`` and the
release-validation matrix compose the very same yaml rather than a private
override list. Override individual keys on top as usual, e.g.
``python -m jcm.main +configuration=t63-echam-jam run.total_time=30``.

The very same recipes are loadable from Python without touching Hydra via
:func:`jcm.configurations.load` — see :ref:`configurations-from-python` in the
getting-started guide.

Override semantics: ``key=`` vs ``+key=`` vs ``++key=``
-------------------------------------------------------

Hydra distinguishes three override prefixes, and picking the wrong one is the
most common CLI trap:

============ ================================================================
Prefix       Meaning
============ ================================================================
``key=v``    **Set an existing key.** Errors if the key is not already in the
             composed config.
``+key=v``   **Append a new key** that the config does not yet have. Errors if
             it *does* already exist.
``++key=v``  **Add-or-override** — set the key whether or not it exists.
============ ================================================================

Two consequences worth committing to memory:

* **Every ``run`` group exposes the same complete key schema.**
  ``run/default.yaml`` is the base schema and the others
  (``longrun``, ``smoke``, ``pyses_year``) inherit it via
  ``defaults: [default, _self_]``, overriding only what they change. So any run
  key sets with a plain override on any group —
  ``run=longrun run.checkpoint_path=/scratch/x.ckpt`` composes even though the
  ``longrun`` yaml never mentions ``checkpoint_path``. The rule is simply:
  **``run.<key>=<value>`` always works.**

* **Reserve ``+``/``++`` for keys outside a group's schema.** A per-scheme
  physics parameter block, for instance, is intentionally absent from
  ``physics/echam.yaml`` (each field falls back to the scheme's
  ``Parameters.default()``), so overriding one field needs the append prefix::

     python -m jcm.main physics=echam \
         +physics.terms.tiedtke_convection.params.entrpen=4e-4

  Physical-constant overrides are the same story — ``constants`` starts as an
  empty mapping, so each base field is *added*::

     python -m jcm.main +constants.grav=9.80665 +constants.rearth=6.4e6

  Constants are applied process-globally before the model is built; see
  :ref:`overriding-constants` in the getting-started guide for the semantics and
  caveats.

Online-aerosol (JAM) inputs default to ``auto``
------------------------------------------------

When a prognostic-aerosol package is active (``physics=echam-jam`` and its
AeroCom variants), the prescribed-emission inputs — ``forcing.emissions_file``,
``forcing.dms_file``, ``forcing.dust_file`` and ``forcing.oxidants_file`` —
default to ``auto``. ``auto`` resolves the per-grid present-day bundle from the
project data mirror for the composed grid (e.g. ``bundles/t63/emissions_pd.nc``)
at build time, so ``python -m jcm.main +configuration=t63-echam-jam`` composes a
fully-specified online-aerosol run with no hand-managed emission paths. For any
non-JAM package these keys resolve to nothing; set an explicit path or ``hf://``
bundle to override one, or ``null`` to opt out (the runner then warns the run is
emission-free). ``auto`` always resolves the *present-day* ``*_pd`` bundle, so a
transient by-date run (``forcing=amip``/``era5``) left on ``auto`` breathes
present-day aerosol emissions over a historical circulation — the runner warns
and names the keys; override ``forcing.emissions_file``/``forcing.oxidants_file``
with year-matched products for a consistent transient run.

The data mirror — ``hf://`` paths
----------------------------------

Boundary-condition and emissions files can be pulled straight from the project
data mirror on Hugging Face by prefixing any file path with ``hf://`` (fetch
once on a node with internet — afterwards the local cache serves compute nodes
offline)::

   python -m jcm.main physics=echam-jam grid=echam_t63_l47_hybrid \
       terrain=from_file terrain.file=hf://bundles/t63/terrain.nc \
       forcing=from_file forcing.file=hf://bundles/t63/forcing_pd.nc

See :doc:`design/data_mirror` for the full bundle catalogue. The Python door
onto the same bundles is :meth:`jcm.forcing.ForcingData.from_bundles` (in the
getting-started guide).

Emulated radiation
------------------

``physics=echam-emulated-2m`` swaps RRTMGP for a GRU emulator trained to
reproduce it — a settled 4.4x end-to-end at T63L47 (22.7 → 5.1 s per sim day).
Trained weights ship with the package, so it runs out of the box::

   python -m jcm.main physics=echam-emulated-2m grid=echam_t63_l47_hybrid

Point ``physics.terms.nn_emulator_radiation.weights_file`` at another checkpoint
to swap networks. The emulator sees ozone and CO2 but **not CH4 or N2O**, so
runs varying those gases are refused with a pointer to ``physics=echam-rrtmgp-2m``
(jax-gcm#738). See :doc:`design/radiation_nn_emulator` for the training workflow
and the full accuracy/cost analysis.

Chunked, resumable runs and checkpoints
----------------------------------------

Long integrations run in **chunks** with a health gate between them. Setting
``run.chunk_days`` (``run=longrun`` uses 30) breaks the integration into pieces,
writes each to ``{run.output_prefix}_day{N}.nc``, and runs
:func:`jcm.diagnostics.check_health` after each one. With
``run.bail_on_unhealthy`` (the runner default) the run aborts on the first
unhealthy chunk instead of integrating a doomed state for hours; set it
``false`` to log and keep going.

Multi-day integrations on preemptible compute (spot instances, Slurm
``--requeue`` queues, NRP Nautilus) can be killed at short notice. Set
``run.checkpoint_path`` to make a chunked run resumable: after each chunk the
runner persists the modal + physics state and the elapsed sim-day count to that
file (atomic write via tmpfile + rename, so a kill mid-write leaves the previous
checkpoint intact). When the same command is launched again with the file
already in place, the run restores from the checkpoint and only steps the
remaining chunks::

   python -m jcm.main physics=echam grid=echam_t63_l47_hybrid \
       run=longrun run.checkpoint_path=/scratch/$JOB_ID.ckpt

Set ``run.archive_ckpt_every`` (sim-days; 0 = off) to also copy the rotating
checkpoint to a dated, never-overwritten archive, so a later experiment can
restart from before a slowly-developing failure.

The same primitives are available directly to bring-your-own-driver workflows
via :py:mod:`jcm.checkpoint`:

.. code-block:: python

   from jcm.checkpoint import save_checkpoint, load_checkpoint

   model.run(forcing=forcing, total_time=10)
   save_checkpoint(model, '/scratch/run.ckpt', elapsed_days=10.0)

   # ... later, in a fresh process ...
   model = build_model(cfg)            # same coords + physics
   model.bootstrap_state()             # populate template pytrees
   elapsed = load_checkpoint(model, '/scratch/run.ckpt')
   model.resume(forcing=forcing, total_time=20 - elapsed)

The on-disk format is flax's msgpack codec applied to flattened lists of arrays
— small (state pytrees are a few MB even at T63L47) and portable across hosts as
long as the destination ``Model`` was built with the same coords and physics
term composition. (Warm-starting from an *equilibrated* state while resetting the
clock is the separate :func:`jcm.initial_states.checkpoint_state` path in the
getting-started guide.)

Nudging from config
-------------------

Relaxing the model toward an external reference state ("nudging") is one flag on
the CLI: ``nudging=era5`` pulls the run window from WeatherBench2's public cloud
ERA5 (regridded to the model grid and cached locally by :mod:`jcm.data.era5`),
and ``init=era5`` starts the run from the ERA5 state at the same date::

   python -m jcm.main physics=echam grid=echam_t63_l47_hybrid \
       init=era5 nudging=era5 run.start_date=2010-01-01 run.total_time=30

Prefetch on a login node first when compute nodes lack internet
(``python -m jcm.data.era5 --grid echam_t63_l47_hybrid --start 2010-01-01 --end
2010-01-31 --init``). The WB2 stores carry 13 pressure levels up to 50 hPa, so
nudging is automatically masked off above ``nudging.min_pressure_hpa`` (default
60); see ``jcm/config/nudging/era5.yaml`` for the knobs (``tau_hours``,
``pbl_levels``, ``nudge_temperature``, ``freq``). Wiring nudging in code against
an arbitrary reference dataset is covered in the getting-started guide.

Multi-device parallelization
----------------------------

JCM supports multi-device execution through JAX's SPMD (Single Program Multiple
Data) sharding, splitting the computation across several GPUs or TPUs. The mesh
is passed to the *coords* helper (so it is a Python-side choice even for CLI
runs, which build coords from the ``grid`` group); if you never set
``spmd_mesh`` JCM runs on a single device, which is the right default for
smaller resolutions (T31, T42) or a single accelerator.

The mesh has three dimensions, ``(x, y, z)`` = ``(longitude, latitude,
vertical)``, and their product must equal the device count:

.. code-block:: python

   import jax
   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   print(f"Available devices: {jax.devices()}")

   # Split longitude across 4 devices: (4, 1, 1). Other layouts: (2, 2, 1)
   # splits longitude and latitude 2x2; (8, 1, 1) uses 8 devices.
   coords = get_speedy_coords(spmd_mesh=(4, 1, 1))
   model = Model(coords=coords)
   predictions = model.run(save_interval=5.0, total_time=30.0)

Rules of thumb: longitude (x) usually has the most grid points, so split it
first; physics with many layers (32/64) can benefit from vertical sharding of
the dycore instead; and higher resolutions (T85+) gain the most from sharding.
See :doc:`design/parallelization` for the sharding design and the strong-scaling
results.

Docker
------

Build the CUDA-enabled image locally::

   docker build -t jcm .
   docker run --rm --gpus all jcm physics=echam grid=echam_t63_l47_hybrid

Arguments after the image name are passed straight to ``python -m jcm.main``.
Mount ``/app/outputs`` to persist Hydra output directories::

   docker run --rm --gpus all -v "$(pwd)/outputs:/app/outputs" jcm \
       physics=echam grid=echam_t63_l47_hybrid run.total_time=30

Kubernetes examples for the NRP Nautilus cluster are in ``deploy/k8s/``.

GPU and batch-queue patterns
----------------------------

On a shared multi-GPU host, pin a run to one device with
``CUDA_VISIBLE_DEVICES`` and keep it detached so it survives a disconnected
shell, writing the state and a resumable checkpoint to fast scratch::

   CUDA_VISIBLE_DEVICES=0 nohup python -m jcm.main \
       +configuration=t63-echam-jam \
       run.output_prefix=/scratch/$USER/run \
       run.checkpoint_path=/scratch/$USER/run.ckpt \
       run.total_time=365 > /scratch/$USER/run.log 2>&1 &

On a PBS/Torque cluster the same command drops into a job script. The
scheduler does not extend a job past its walltime, so a long run is a *chain*
of jobs sharing one checkpoint: each job resumes from
``run.checkpoint_path``, integrates until the walltime kills it, and the next
job in the chain picks up at the last completed chunk. Once the run reaches
``run.total_time`` a resubmitted job restores the checkpoint and exits
immediately, so over-provisioning the chain is harmless:

.. code-block:: bash

   #!/bin/bash
   #PBS -l select=1:ngpus=1
   #PBS -l walltime=12:00:00
   cd "$PBS_O_WORKDIR"
   python -m jcm.main +configuration=t63-echam-jam run=longrun \
       run.total_time=365 \
       run.checkpoint_path="$SCRATCH/t63-echam-jam.ckpt"

Submit the chain with ``afterany`` dependencies (each link starts only when
the previous one ends, however it ended)::

   jid=$(qsub run.pbs)
   for i in $(seq 5); do jid=$(qsub -W depend=afterany:$jid run.pbs); done

The checkpoint path must be **stable across jobs** — deriving it from
``$PBS_JOBID`` would give every link a fresh checkpoint and restart the run
from day zero. Size ``run.chunk_days`` so at least one chunk completes
comfortably inside the walltime; progress only persists at chunk boundaries.

.. _packaged-config-tree:

Downstream apps: the packaged config tree
-----------------------------------------

The whole ``jcm/config`` tree is a **public, packaged** Hydra config tree. A
downstream Hydra app (a coupled Earth-system CLI, say) reaches every jcm group
through ``hydra.searchpath: [pkg://jcm.config]`` and can re-root a whole
validated configuration under one of its own nodes with
``+configuration@<node>=<name>``. That contract — the public group names, the
load-bearing ``# @package _global_`` header plus absolute-override recipe style,
and the group-rename policy (this release renamed the ``experiment`` group to
``configuration``, so a searchpath user must change ``+experiment@<node>=`` to
``+configuration@<node>=``) — is documented in :doc:`design/packaged_config_tree`.
