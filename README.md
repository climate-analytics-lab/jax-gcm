# JAX-GCM (JCM)

[![Docs](https://readthedocs.org/projects/jax-gcm/badge/?version=latest)](https://jax-gcm.readthedocs.io/en/latest/)
[![Tests](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_test.yaml/badge.svg?branch=dev)](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_test.yaml)
[![Lint](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_linter.yaml/badge.svg?branch=dev)](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_linter.yaml)
[![PyPI](https://img.shields.io/pypi/v/jcm.svg)](https://pypi.org/project/jcm/)
[![Python](https://img.shields.io/pypi/pyversions/jcm.svg)](https://pypi.org/project/jcm/)
[![License](https://img.shields.io/github/license/climate-analytics-lab/jax-gcm.svg)](LICENSE)

<img src="logo.png" alt="JAX-GCM logo" width="180">

JAX-GCM is a differentiable atmospheric general circulation model written
entirely in JAX. A pluggable dynamical-core interface couples the
[Dinosaur](https://github.com/neuralgcm/dinosaur) spectral backend to modular
SPEEDY, Held-Suarez, and ECHAM-style physics packages.

## Why JCM

- **Flexible.** Switch from lightweight SPEEDY physics to full-fat ECHAM with
  online aerosol by changing one flag — and compose anything in between, term
  by term, through one composable-physics API.
- **Differentiable.** Any output is differentiable with respect to the physics
  parameters, the initial conditions, and the boundary conditions, so gradient
  calibration, data assimilation, and hybrid physics-ML experiments come for
  free from JAX's `jit`, `grad`, and `vmap`.
- **Fast, on whatever you have.** The same configuration runs on CPU, GPU, or
  TPU. ECHAM T63L47 climate with RRTMGP radiation costs roughly 22.7 s per
  simulated day on a single GPU — about 4x faster again with the
  neural-emulated radiation backend — and SPEEDY is cheaper still.

## Installation

```bash
pip install jcm
```

JCM requires Python 3.11 or newer. See
[the getting-started guide](https://jax-gcm.readthedocs.io/en/latest/getting_started.html)
for the development install and the full dependency set.

## Quick start

Run a short SPEEDY aquaplanet integration from Python:

```python
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# T31 spectral resolution, 8 vertical levels. Omitting time_step lets the
# Model resolve a numerically stable default from the physics and grid
# (SPEEDY picks 30 min here; the no-limit default for ECHAM/Held-Suarez is
# 12 min).
coords = get_speedy_coords(layers=8, spectral_truncation=31)
model = Model(coords=coords)

predictions = model.run(save_interval=10.0, total_time=120.0)  # days
ds = predictions.to_xarray()
print(ds)
```

## Documentation

- [Getting started](https://jax-gcm.readthedocs.io/en/latest/getting_started.html) — the Python quick start: coords, terrain, physics, running, and analysing output.
- [Running at scale](https://jax-gcm.readthedocs.io/en/latest/running_at_scale.html) — the `python -m jcm.main` Hydra CLI, validated configurations, chunked/resumable runs, Docker, and GPU/batch patterns.
- [ECHAM physics](https://jax-gcm.readthedocs.io/en/latest/echam_physics.html) and [SPEEDY physics](https://jax-gcm.readthedocs.io/en/latest/speedy_physics.html) — scheme notes and references.
- Example notebooks live in [`notebooks/`](notebooks/).

Full documentation is hosted at
[jax-gcm.readthedocs.io](https://jax-gcm.readthedocs.io/en/latest/).

## Citation

If you use JAX-GCM in your research, please cite:

```bibtex
@article{jcm_gmd_2026,
  title   = {{JCM} v1.1: a differentiable, intermediate-complexity atmospheric model},
  author  = {Davenport, Ellen H. and Madan, J. Varan and Gjini, Rebecca and
             Brzenski, Jared and Ho, Nick and Hsu, Tien-Yiao and Liang, Yueshan and
             Liu, Zhixing and Manivannan, Veeramakali and Pham, Eric and
             Vutukuru, Rohith and Williams, Andrew I. L. and Yang, Zhiqi and
             Yu, Rose and Lutsko, Nicholas J. and Hoyer, Stephan and
             Watson-Parris, Duncan},
  journal = {Geoscientific Model Development},
  volume  = {19},
  pages   = {6451--6466},
  year    = {2026},
  doi     = {10.5194/gmd-19-6451-2026},
  url     = {https://gmd.copernicus.org/articles/19/6451/2026/}
}
```

## License

JAX-GCM is licensed under Apache 2.0. See [`LICENSE`](LICENSE).
