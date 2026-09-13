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
— its grid, its 15-minute step, its terrain/forcing file overrides — while the
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
