# JAX-GCM (JCM)

<img src="logo.png" alt="Logo" width="200">

A fully differentiable General Circulation Model (GCM) for climate science and machine learning applications, written entirely in JAX.

## Overview

JCM is a physical climate model that combines the [Dinosaur](https://github.com/google-research/dinosaur) dynamical core with JAX implementations of atmospheric physics parameterizations. The entire model is differentiable, enabling gradient-based optimization, data assimilation, and ML-enhanced climate modeling.

### Key Features

- **Fully Differentiable**: Automatic differentiation through the entire model using JAX
- **GPU/TPU Accelerated**: JIT compilation and hardware acceleration via JAX
- **Modular Physics**: SPEEDY and ECHAM physics packages with radiation, convection, clouds, and surface processes
- **Composable**: Mix and match parameterizations across physics packages (e.g., SPEEDY convection + ECHAM radiation)
- **Flexible Grids**: Spectral dynamical core supporting multiple resolutions (T21 to T425), including hybrid (a + b·P_s) vertical coordinates
- **Climate-Quality Target Config**: T63L47 hybrid grid with ECHAM physics and RRTMGP radiation (`physics=echam-rrtmgp grid=echam_t63_l47_hybrid`) for production-grade runs
- **ML-Ready**: Designed for hybrid physics-ML workflows and parameter optimization

## Installation

Easily install using pip with all the associated requirements:

```bash
pip install jcm
```

Or, from conda forge:
```bash
conda install -c conda-forge jcm
```

Or, clone the repository and install in development mode:

```bash
git clone https://github.com/climate-analytics-lab/jax-gcm.git
cd jax-gcm
pip install -e .
```

### Requirements

- Python ≥ 3.11
- JAX
- [Dinosaur](https://github.com/google-research/dinosaur) (dynamical core)
- XArray (for I/O and data handling)

See `requirements.txt` for the complete list of dependencies.

## Quick Start

Run a simple aquaplanet simulation:

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

# Run a 120-day simulation
predictions = model.run(
    save_interval=10.0,  # save every 10 days
    total_time=120.0     # total simulation time in days
)

# Convert output to xarray Dataset for analysis
ds = predictions.to_xarray()
print(ds)
```

## Command-line interface

Most simulations can be launched with the bundled Hydra CLI without writing
any Python:

```bash
# Default 10-day SPEEDY aquaplanet
python -m jcm.main

# Targeted production setup: ECHAM physics on T63L47 hybrid coords with RRTMGP radiation
python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid

# ECHAM physics with the default grey two-stream radiation (cheaper, good for tuning)
python -m jcm.main physics=echam grid=echam_t63_l47_hybrid

# Held-Suarez 30-day integration
python -m jcm.main physics=held_suarez grid=held_suarez_t31_l8 \
    run.total_time=30 run.save_interval=1

# Override individual physics parameters (the leading `+` is required because
# `params` defaults to the scheme's `.default()` and is not in the yaml).
python -m jcm.main physics=echam +physics.terms.tiedtke_convection.params.entrpen=4e-4

# Long ECHAM + RRTMGP run with chunked health checks
python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid run=longrun

# Same, but resumable across preemptions: model state is saved to
# ``run.checkpoint_path`` after each chunk; if the file already exists at
# launch, the run picks up at the recorded sim-day.
python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid run=longrun \
    run.checkpoint_path=/scratch/$JOB_ID.ckpt

# Single-column physics evolution from a saved JCM run
python -m jcm.main run.mode=scm run.state_file=path/to/state.nc \
    run.column.lat_deg=0 run.column.lon_deg=180
```

Inspect the available config groups and the fully-composed config:

```bash
python -m jcm.main --help                  # config-group choices + Hydra usage
python -m jcm.main --cfg job               # print the composed config
python -m jcm.main --cfg job grid=echam_t63_l47_hybrid   # with overrides
```

Config groups live under `jcm/config/` (`physics`, `grid`, `run`, `init`,
`terrain`, `forcing`, `diffusion`).

## Running in Docker

The bundled `Dockerfile` is built on `nvidia/cuda` and ships `jax[cuda12]`,
so the same image runs accelerated on GPU hosts and falls back to CPU
elsewhere (no TPU support). Its entrypoint is the JCM Hydra CLI, which
makes it straightforward to launch non-interactive simulations on
Kubernetes, GCP, NRP or any other batch/queued compute. Build once:

```bash
docker build -t jcm .
```

### Default run

```bash
# On a GPU host
docker run --rm --gpus all jcm

# Or without GPUs (CPU fallback)
docker run --rm jcm
```

This runs the default 10-day SPEEDY aquaplanet configuration.

### Hydra overrides

Anything after the image name is forwarded to `python -m jcm.main`, so the
full Hydra CLI (config groups, dotted overrides, multirun) is available:

```bash
# Switch physics package and grid (recommended ECHAM + RRTMGP setup)
docker run --rm --gpus all jcm physics=echam-rrtmgp grid=echam_t63_l47_hybrid

# Override individual run options
docker run --rm --gpus all jcm run.time_step=20 run.total_time=30 run.save_interval=1

# Parameter sweep (multirun)
docker run --rm --gpus all jcm -m run.time_step=10,20,30
```

### Persisting outputs

By default outputs land in `outputs/YYYY-MM-DD/HH-MM-SS/` *inside the
container* and are lost when it exits. Mount a host directory at
`/app/outputs` to keep them:

```bash
docker run --rm --gpus all -v "$(pwd)/outputs:/app/outputs" jcm \
    physics=echam-rrtmgp grid=echam_t63_l47_hybrid run.total_time=30
```

After the run finishes the netCDF state file is available on the host
under `./outputs/`.

### Interactive shell

Override the entrypoint to drop into a shell for debugging or exploration:

```bash
docker run --rm -it --entrypoint bash jcm
```

### Kubernetes example

Ready-to-use manifests for the NRP Nautilus cluster (Job, interactive
pod, PVC) live under [`deploy/k8s/`](deploy/k8s/). For other clusters,
pass Hydra overrides via the container `args` field, request a GPU,
and mount a persistent volume for outputs:

```yaml
spec:
  containers:
    - name: jcm
      image: your-registry/jcm:latest
      args: ["physics=echam-rrtmgp", "grid=echam_t63_l47_hybrid", "run.total_time=30"]
      resources:
        limits:
          nvidia.com/gpu: 1
      volumeMounts:
        - name: outputs
          mountPath: /app/outputs
```

## Example notebooks

Example notebooks are available in the `notebooks/` directory:

- **`01_jcm_demo.ipynb`**: Basic model simulation with SPEEDY physics
- **`02_optimization_example.ipynb`**: Parameter optimization examples
- **`03_generate_speedy_default_stats.ipynb`**: Regenerate the bundled SPEEDY reference statistics
- **`04_jcm_era5_example.ipynb`**: Initialising and verifying runs against ERA5
- **`04_jcm_slides.ipynb`**: Presentation-ready overview
- **`05_jcm_echam_demo.ipynb`**: Running with ECHAM physics and composable physics
- **`06_macv2_aerosols.py`**: Driving ECHAM with MACv2-SP aerosol parameters

## Physics Packages

JCM provides two physics packages and a composable framework for mixing parameterizations across them.

### SPEEDY Physics

The SPEEDY (Simplified Parameterizations, primitivE-Equation DYnamics) physics package includes:

- Convection (simplified mass-flux scheme)
- Large-scale condensation
- Shortwave and longwave radiation
- Surface fluxes (land, ocean, sea ice)
- Vertical diffusion
- Orographic drag

### ECHAM Physics

The ECHAM physics package is a JAX port of the MPI-M ECHAM6 / ICON-A
atmospheric physics. It is the recommended package for climate-quality
runs and is the target for the `T63L47` hybrid-coordinate setup
(`grid=echam_t63_l47_hybrid`):

- Tiedtke-Nordeng mass-flux convection
- Sundqvist diagnostic cloud fraction
- 1-moment (default) or 2-moment cloud microphysics (Lohmann & Roeckner)
- Radiation, selectable per run via `radiation_scheme`:
  - Grey two-stream (default, fast)
  - RRTMGP correlated-k (production accuracy, `physics=echam-rrtmgp`)
  - Neural-network RRTMGP emulator (RRTMGP accuracy at grey-like cost)
- TTE-TKE vertical diffusion (Pithan & Brinkop)
- Multi-tile surface scheme (ocean, sea ice, land)
- Hines (1997) non-orographic + Lott-Miller (1997) sub-grid orographic gravity-wave drag
- MACv2-SP aerosol scheme (Stevens et al., 2017) with aerosol-cloud coupling
- Simple chemistry (ozone, CO2, CH4)

See [`docs/source/echam_physics.rst`](docs/source/echam_physics.rst) for
the per-scheme references and performance numbers on T63L47.

> **Note:** ECHAM physics was previously called "ICON" in JCM. The
> `jcm.physics.icon` namespace and `physics=icon` config group have been
> renamed to `jcm.physics.echam` and `physics=echam` (PR #457).

### Composable Physics

Individual parameterizations are wired together via the composable physics API. Both `speedy_physics()` and `echam_physics()` return a `ComposablePhysics` whose terms can be swapped (`replace`), removed (`remove`) or extended (`+`):

```python
from jcm.physics.echam.echam_terms import echam_physics

# Recommended T63L47 production setup — full RRTMGP correlated-k radiation
physics = echam_physics(radiation_scheme="rrtmgp")

# Cheaper alternative: NN radiation emulator at RRTMGP-like accuracy
physics = echam_physics(radiation_scheme="emulated")

# Drop a term by category, e.g. for sensitivity studies
physics = echam_physics().remove("hines")

# Or swap an individual term in by category
from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
physics = echam_physics().replace("radiation", RRTMGPRadiation())
```

Each `PhysicsTerm` stores its own tunable parameters as `flax.nnx.Param`, enabling per-scheme gradient-based optimization.

## Documentation

For more details, see the [documentation](https://jax-gcm.readthedocs.io) (or build locally):

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in your browser.

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest jcm/model_test.py

# Run with verbose output
pytest -v
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. Note, the latest development work should target the `dev` branch. Clean, working releases are periodically merged into `main` and tagged. 

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dinosaur**: JAX-GCM builds on the [Dinosaur](https://github.com/google-research/dinosaur) dynamical core developed by Google Research
- **SPEEDY**: Physics parameterizations adapted from the [SPEEDY](https://users.ictp.it/~kucharsk/speedy-net.html) model by F. Molteni
- **SPEEDY.f90**: We referenced the [Fortran 90 version](https://github.com/samhatfield/speedy.f90) of SPEEDY by Sam Hatfield and Leo Saffin for our specific implementation.

## Contact

For questions or collaboration inquiries, please open an issue or contact dwatsonparris@ucsd.edu.
