# The packaged config tree as a public API

`jcm/config` is not only the source of `python -m jcm.main`'s own Hydra
composition — it is a **public, packaged config tree** that a downstream Hydra
application (a coupled Earth-system CLI, for example) composes into its own
configuration without vendoring copies. This page states that contract, because
the properties it depends on look like incidental style until a refactor
silently breaks a downstream release (issue #757).

## The tree is packaged and importable

The whole tree ships in the wheel: `pyproject.toml`'s
`[tool.setuptools.package-data]` lists `config/**/*.yaml`, and
`jcm/config/__init__.py` makes `jcm.config` a real (not namespace) package.
Together these are a **guarantee**, not an implementation detail — a downstream
app reaches every jcm group by adding one line to its own config:

```yaml
hydra:
  searchpath:
    - pkg://jcm.config
```

Hydra's `pkg://` provider only reports a directory as *available* when it is a
regular package (its check looks for `__init__.py`); the `__init__.py` is
therefore load-bearing. Without it `jcm.config` is a namespace package that
Hydra reads but flags `provider=hydra.searchpath ... is not available`, and a
stricter Hydra release could drop it outright.

With the search path in place, the app selects jcm groups by name exactly as
`jcm.main` does — `physics`, `grid`, `dycore`, `run`, `init`, `terrain`,
`forcing`, `nudging`, `diffusion`, and the validated `configuration` group.
These names are the **public API**.

## Re-rooting a whole configuration under one node

A coupled app runs jcm as one component among several, so it needs a whole
validated jcm configuration nested under a single node (say `atmosphere`), not
spread across the global config. That works because of two intentional,
**load-bearing** properties of every `jcm/config/configuration/*.yaml`:

1. the file opens with `# @package _global_`, so its package is declared *in the
   file* rather than implied by its directory; and
2. it refers to groups by **absolute** path — `- override /physics: echam`,
   `- override /grid: echam_t63_l47_hybrid`, ….

Because the package is declared in the file, Hydra's package-override syntax can
re-root the entire recipe under an arbitrary node. The coupler mounts jcm's base
groups under that node and then re-roots a recipe onto them:

```yaml
# coupler config.yaml
defaults:
  - _self_
  - physics@atmosphere.physics: speedy
  - grid@atmosphere.grid: speedy_t31_l8
  - dycore@atmosphere.dycore: dinosaur
  - run@atmosphere.run: default
  - init@atmosphere.init: isothermal
  - terrain@atmosphere.terrain: aquaplanet
  - forcing@atmosphere.forcing: default
  - nudging@atmosphere.nudging: none
  - diffusion@atmosphere.diffusion: default

hydra:
  searchpath:
    - pkg://jcm.config
```

```bash
python -m coupler.main +configuration@atmosphere=speedy-t31
```

lands the complete, validated `speedy-t31` configuration under `cfg.atmosphere`
— its grid, its 30-minute step, its terrain/forcing file overrides — while the
coupler's own top-level keys are untouched. A maintainer refactoring toward
directory-implied packages or relative overrides would break this without any
in-tree test noticing, so it is pinned by
`TestPackagedConfigSearchpath` in `jcm/configurations_test.py`, which composes a
foreign app exactly as above and asserts the re-rooted values.

## Group names are public; renames are coordinated, not aliased

Because a downstream searchpath user selects jcm groups **by name**, a group
name is a public symbol: renaming one is a breaking change for every coupled
CLI, and — unlike a Python rename — nothing in jcm's own tests or CI sees the
downstream `+<group>@<node>=` override that goes stale. The policy is therefore:

- a group rename gets a **release-notes entry** naming the old and new group and
  the exact override a searchpath user must change, and
- it is **coordinated in lockstep** with the coupled repositories (both jcm and
  JAX-ESM are in-house and pre-1.0), which update their configs in the same
  release cycle.

**We deliberately keep no back-compatibility alias directory** (e.g. a shim
`experiment/` group forwarding to `configuration/`). Both repos are in-house and
pre-1.0, so a permanent alias would be dead weight; the contract, the
release note, and the lockstep update are the whole policy. This release is the
first application of it: the `experiment` group was renamed to `configuration`
(the word "experiment" already means a *realized simulation* elsewhere in the
project), so a downstream searchpath user must change
`+experiment@<node>=<name>` to `+configuration@<node>=<name>`.

## Driving jcm from a host process

A co-simulation host that owns the time loop (it advances jcm a segment at a
time between exchanges with other components) uses jcm as a library through
three public pieces, none of them underscore-prefixed.

**Build from a composed config.** `jcm.configurations.from_config(cfg)` takes a
composed config (an OmegaConf `DictConfig` or a plain nested dict) and returns
the `LoadedConfiguration` that `jcm.configurations.load` returns; `load(name)`
is "compose, then `from_config`", so the two cannot drift. A host that keeps its
own config directory composes it with jcm's either through
`load(name, extra_config_dir=...)` (the directory joins `hydra.searchpath` behind
jcm's own, so its new groups and recipes compose with jcm's) or by composing
against `jcm.configurations.CONFIG_DIR` itself and passing the result in.
`from_config` rejects a config that lacks `run`, `grid`, `physics`, `terrain`,
`forcing` or `run.save_interval`, or has neither `run.total_time` nor
`run.end_time`, and carries every other
section through untouched, so an overlay's extra groups are not an error. The
build is the sequence the CLI's full mode runs: `constants.*` overrides,
`build_model`, `runners.build_run_forcing` (boundary forcing, the ERA5 nudging
target, the forcing guards and config-trap warnings) and
`runners.initial_state_kwargs` (`init.kind`); `runners._run_full` calls the same
two helpers, so the doors share one implementation. `model.run(**exp.run_kwargs)`
is the CLI's single integration.

**Process-global steps are explicit.** The logger level and JAX's compilation
cache are performance/verbosity settings of the process, not of the model, so
the door does not apply them unasked. A host calls `runners.apply_log_level(cfg)`
and `runners.maybe_enable_compilation_cache()` itself, or passes
`from_config(..., apply_log_level=True, enable_compilation_cache=True)`.
`configure_host_device_count`, `provenance.start_run` and the `from_state`
path pre-flight belong to `runners.run` alone.

**Step with `run`/`resume`, window the target.** The nudging target the door
returns is host-resident (numpy leaves; a year of 6-hourly T63L47 ERA5 is about
15 GB, which the compiled step must not carry on the device). `Model.run` takes
it whole. A host that calls `Model.run` once and `Model.resume` repeatedly
passes each call `jcm.nudging.window_forcing(forcing, start, span_seconds)`:
the forcing with the target cut to the `samples_per_span(target, span_seconds)`
samples starting at the last sample at or before `start` (the sample before the
segment and one after, for an interpolating leaf), moved to the device. The
window length depends only on the span and the target's spacing, so equal
segments share one compiled program; a window past the end of the series is
clamped to end there, and a `persist="hold"` leaf holds its last sample. A
forcing with no target, or with only a static or climatological one, is returned
unchanged. `runners.run_chunked` windows each chunk through this same function,
so a segmented run is bit-identical to the same chunks of a chunked run and to
the same segments with the whole target attached.
