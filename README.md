# JAX-GCM (JCM)

[![Docs](https://readthedocs.org/projects/jax-gcm/badge/?version=latest)](https://jax-gcm.readthedocs.io/en/latest/)
[![Tests](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_test.yaml/badge.svg?branch=dev)](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_test.yaml)
[![Lint](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_linter.yaml/badge.svg?branch=dev)](https://github.com/climate-analytics-lab/jax-gcm/actions/workflows/run_linter.yaml)
[![PyPI](https://img.shields.io/pypi/v/jcm.svg)](https://pypi.org/project/jcm/)
[![Python](https://img.shields.io/pypi/pyversions/jcm.svg)](https://pypi.org/project/jcm/)
[![License](https://img.shields.io/github/license/climate-analytics-lab/jax-gcm.svg)](LICENSE)

<img src="logo.png" alt="JAX-GCM logo" width="180">

JAX-GCM is a differentiable atmospheric general circulation model written
in JAX. Its pluggable dynamical-core interface currently ships with the
[Dinosaur](https://github.com/neuralgcm/dinosaur) spectral backend and
couples it to modular SPEEDY, Held-Suarez, and ECHAM-style physics packages,
with support for gradient-based calibration, ML-physics experiments, and
accelerated CPU/GPU/TPU runs.

The v2.0 release focus is the ECHAM T63L47 hybrid-coordinate stack with
RRTMGP radiation:

```bash
python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid run=longrun
```

## Highlights

- Fully differentiable JAX implementation compatible with `jit`, `grad`, and `vmap`
- Pluggable dynamical-core protocol with a shipped Dinosaur spectral backend
- Dycore-agnostic, operator-split gridpoint physics coupling
- SPEEDY, Held-Suarez, and composable ECHAM physics configurations
- ECHAM T63L47 hybrid-coordinate target setup with grey, RRTMGP, or neural-emulated radiation
- xarray/netCDF output, chunked long-run health checks, and resumable checkpoints
- Docker and Kubernetes deployment examples for GPU batch runs

## Installation

Install the latest release:

```bash
pip install jcm
```

For the development branch:

```bash
git clone https://github.com/climate-analytics-lab/jax-gcm.git
cd jax-gcm
git switch dev
pip install -e .
```

JCM requires Python 3.11 or newer. The full dependency set is listed in
[`requirements.txt`](requirements.txt), including JAX, Flax, Dinosaur,
Hydra, xarray, and optional RRTMGP support.

## Quick Start

Run a short SPEEDY aquaplanet integration from Python:

```python
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# Build coords (pass spmd_mesh=(x, y, z) here to enable multi-device sharding)
coords = get_speedy_coords(layers=8, spectral_truncation=31)

# Create a model with default configuration
model = Model(
    coords=coords,
    time_step=30.0,  # minutes
)

predictions = model.run(save_interval=10.0, total_time=120.0)
ds = predictions.to_xarray()
print(ds)
```

Most production runs use the Hydra CLI:

```bash
# Default 10-day SPEEDY aquaplanet
python -m jcm.main

# ECHAM T63L47 with production RRTMGP radiation
python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid

# Cheaper ECHAM development run with grey two-stream radiation
python -m jcm.main physics=echam grid=echam_t63_l47_hybrid

# Chunked, resumable long run
python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid run=longrun \
    run.checkpoint_path=/scratch/$JOB_ID.ckpt

# Inspect available config groups and the composed config
python -m jcm.main --help
python -m jcm.main --cfg job grid=echam_t63_l47_hybrid
```

Config groups live under [`jcm/config/`](jcm/config/) (`physics`, `grid`,
`run`, `init`, `terrain`, `forcing`, and `diffusion`).

## Physics Packages

**SPEEDY** provides a compact climate-physics package for development,
testing, and optimization examples: simplified convection, large-scale
condensation, radiation, surface fluxes, vertical diffusion, and
orographic drag.

**ECHAM** is the v2.0 release target for climate-quality integrations. It
includes Tiedtke-Nordeng convection, Sundqvist cloud cover, 1M/2M cloud
microphysics, TTE-TKE vertical diffusion, multi-tile surface physics,
gravity-wave drag, MACv2-SP aerosols, simple chemistry, and selectable
radiation backends (`grey`, `rrtmgp`, `emulated`). See
[`docs/source/echam_physics.rst`](docs/source/echam_physics.rst) for
scheme notes, references, and current performance guidance.

Individual parameterizations are `PhysicsTerm` modules and can be
combined with the composable physics API:

```python
from jcm.physics.echam.echam_terms import echam_physics

physics = echam_physics(radiation_scheme="rrtmgp")
physics = echam_physics().remove("hines")
```

## Notebooks

Examples live in [`notebooks/`](notebooks/):

- `01_jcm_demo.ipynb`: SPEEDY aquaplanet and basic xarray analysis
- `02_optimization_example.ipynb`: differentiable parameter optimization
- `03_generate_speedy_default_stats.ipynb`: bundled SPEEDY reference statistics
- `04_jcm_era5_example.ipynb`: ERA5-style initial state workflow
- `04_jcm_slides.ipynb`: presentation overview
- `05_jcm_echam_demo.ipynb`: ECHAM and composable physics demo
- `06_macv2_aerosols.py`: MACv2-SP aerosol parameter workflow

## Docker And Deployment

Build the CUDA-enabled image locally:

```bash
docker build -t jcm .
docker run --rm --gpus all jcm physics=echam-rrtmgp grid=echam_t63_l47_hybrid
```

Mount `/app/outputs` to persist Hydra output directories:

```bash
docker run --rm --gpus all -v "$(pwd)/outputs:/app/outputs" jcm \
    physics=echam-rrtmgp grid=echam_t63_l47_hybrid run.total_time=30
```

Kubernetes examples for the NRP Nautilus cluster are in
[`deploy/k8s/`](deploy/k8s/).

## Documentation

Read the hosted documentation at
[jax-gcm.readthedocs.io](https://jax-gcm.readthedocs.io/en/latest/) or
build it locally:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html`.

## Testing And Development

Run the fast suite locally:

```bash
pytest -m "not slow"
ruff check .
```

The GitHub test workflow enforces coverage on push and pull requests. New
work should target the `dev` branch; clean release points are merged to
`main` and tagged.

## Citation

If you use JAX-GCM in your research, please cite:

```bibtex
@software{jax_gcm,
  title = {JAX-GCM: A Differentiable General Circulation Model},
  author = {J. Madan, E. Davenport, et al.},
  year = {2025},
  url = {https://github.com/climate-analytics-lab/jax-gcm}
}
```

## License

JAX-GCM is licensed under Apache 2.0. See [`LICENSE`](LICENSE).
