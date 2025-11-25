# JAX-GCM (JCM)

![Logo](logo.png)

A fully differentiable General Circulation Model (GCM) for climate science and machine learning applications, written entirely in JAX.

## Overview

JCM is a physical climate model that combines the [Dinosaur](https://github.com/google-research/dinosaur) dynamical core with JAX implementations of atmospheric physics parameterizations. The entire model is differentiable, enabling gradient-based optimization, data assimilation, and ML-enhanced climate modeling.

### Key Features

- **Fully Differentiable**: Automatic differentiation through the entire model using JAX
- **GPU/TPU Accelerated**: JIT compilation and hardware acceleration via JAX
- **Modular Physics**: SPEEDY physics package with radiation, convection, clouds, and surface processes
- **Flexible Grids**: Spectral dynamical core supporting multiple resolutions (T21 to T425)
- **ML-Ready**: Designed for hybrid physics-ML workflows and parameter optimization

## Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/jax-gcm.git
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

# Create a model with default configuration
model = Model(
    time_step=30.0,          # minutes
    layers=8,                 # vertical levels
    horizontal_resolution=31  # T31 spectral grid
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

## Examples

Example notebooks are available in the `notebooks/` directory:

- **`run-speedy.ipynb`**: Basic model simulation with SPEEDY physics
- **`run-speedy-gradients.ipynb`**: Computing gradients through the model
- **`optimization_example.ipynb`**: Parameter optimization examples
- **`autodiff_userguide.ipynb`**: Guide to automatic differentiation features

## Physics Packages

### SPEEDY Physics

The SPEEDY (Simplified Parameterizations, primitivE-Equation DYnamics) physics package includes:

- Convection (simplified mass-flux scheme)
- Large-scale condensation
- Shortwave and longwave radiation
- Surface fluxes (land, ocean, sea ice)
- Vertical diffusion
- Orographic drag

> **Note**: ICON atmospheric physics is currently under development but not yet available in released versions.

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

Contributions are welcome! Please feel free to submit issues or pull requests.

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
