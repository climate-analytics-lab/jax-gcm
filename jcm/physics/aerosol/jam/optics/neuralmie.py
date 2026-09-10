"""NeuralMie aerosol bulk-optics emulator in JAX.

A JAX port of the NeuralMie inference path: two small pretrained MLPs that
predict the *mode-integrated* optical properties of a lognormal population of
homogeneous or coated spheres in a single forward pass.

Derived from pnnl/NEURALMIE @ 0b5b1d8 (BSD-2-Clause,
Copyright 2024 Battelle Memorial Institute). Reference:
Geiss, A. and Ma, P.-L., *NeuralMie (v1.0): An Aerosol Optics Emulator*,
Geosci. Model Dev., doi:10.5194/gmd-2024-30.

Vendored from ``reflective-org/neuralmie-jax`` (BSD-2-Clause), which holds the
NumPy Mie ground truth and the validation suite behind the accuracy figures
quoted below. The module is self-contained -- it imports only the standard
library, ``numpy`` and ``jax`` -- so keeping it in sync with upstream is a file
copy plus the weights path in :func:`default_weights`.

Accuracy against a NumPy Bohren-Huffman kernel (an independent implementation
from the TAMie the networks were trained on) wrapped in the reference
1024-point quadrature, over the range modal schemes occupy: extinction 0.023%
median / 0.18% max, ``ssa`` within 9.2e-4, ``g`` within 1.4e-3. All 12
published reference cases reproduce to <=1e-6.

Units and conventions, all of which are easy to get wrong:

* ``wavelength`` is in metres.
* ``r_g`` is the number-median **radius** of the lognormal, in metres --
  *not* a diameter. This is the single most common coupling error against
  MAM/E3SM, whose ``dgnum`` is a diameter.
* ``m_i`` is the **positive** imaginary refractive index.
* ``core_fraction`` is a **radius** ratio ``r_core/r_total``; the corresponding
  volume fraction is its cube.
* The primary output ``ke_rho`` is the extinction cross-section per unit
  *particle volume*, in 1/m. Divide by the particle mass density to get the
  mass extinction coefficient in m^2/kg -- see :meth:`BulkOptics.per_mass`.
* ``g`` is the scattering-weighted asymmetry parameter.
"""

from __future__ import annotations

import functools
import importlib.resources
import math
from collections.abc import Callable
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

# --------------------------------------------------------------------------
# Constants transcribed from the reference implementation.
#
# These are not tunables and must not be "cleaned up" -- they define the
# checkpoint. Each cites its upstream source.
# --------------------------------------------------------------------------

#: sqrt(2)*erfinv(0.999). Appears in the Rayleigh switch and the integration
#: bounds (``sphere_inference_demo.py:11-21``). erfinv's argument is 0.999,
#: i.e. the t=0.9995 quantile, matching the [0.0005, 0.9995] bounds upstream
#: integrates between.
SQRT2_ERFINV_999: Final[float] = 3.2905267314918945

#: The emulator is bypassed for an analytic Rayleigh limit when 99.9% of
#: particles have a size parameter below this (``sphere_inference_demo.py:47``).
RAYLEIGH_SIZE_PARAMETER: Final[float] = 0.1

# Training domain (``utils.py:9-14``). Inputs are clipped to this box; outside
# it the emulator extrapolates and carries no validity claim.
MU_RANGE: Final[tuple[float, float]] = (5.0e-9, 5.0e-5)
SIGMA_G_RANGE: Final[tuple[float, float]] = (1.2, 2.8)
M_R_RANGE: Final[tuple[float, float]] = (1.1, 3.0)
M_I_RANGE: Final[tuple[float, float]] = (1.0e-8, 1.0)
CORE_FRACTION_RANGE: Final[tuple[float, float]] = (0.0, 0.98)
WAVELENGTH_RANGE: Final[tuple[float, float]] = (2.0e-7, 1.0e-3)

#: Size-parameter range implied by the corners of MU_RANGE and WAVELENGTH_RANGE.
MU_X_RANGE: Final[tuple[float, float]] = (
    2.0 * math.pi * MU_RANGE[0] / WAVELENGTH_RANGE[1],
    2.0 * math.pi * MU_RANGE[1] / WAVELENGTH_RANGE[0],
)

#: Inert guard on ln(ke_rho*lambda) before exp(). The observed range over the
#: whole training box is [-11.8, 3.0], so this never binds; it exists so a
#: future weight swap cannot produce inf.
LN_KE_LAMBDA_CLIP: Final[tuple[float, float]] = (-40.0, 40.0)

_SPHERE_LAYERS: Final[tuple[int, ...]] = (4, 69, 69, 69, 69, 3)
_CORESHELL_LAYERS: Final[tuple[int, ...]] = (7, 112, 112, 112, 112, 3)


def _swish(x):
    """Keras ``swish`` / SiLU, ``x*sigmoid(x)`` with beta=1.

    The upstream README warns that running these weights through an
    activation *without* swish produces no error but silently wrong output.
    """
    return x * jax.nn.sigmoid(x)


ACTIVATIONS: Final[dict[str, Callable]] = {
    "swish": _swish,
    "linear": lambda x: x,
}


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


class DenseWeights(NamedTuple):
    """One dense layer, oriented so that ``y = x @ kernel + bias``.

    Contains arrays only -- no activation names -- so the whole structure
    remains a valid pytree that can cross a ``jit`` boundary.
    """

    kernel: jnp.ndarray  # (n_in, n_out)
    bias: jnp.ndarray  # (n_out,)


class MLPWeights(NamedTuple):
    """A serial dense stack with swish hidden layers and a linear head."""

    layers: tuple[DenseWeights, ...]


class NeuralMieWeights(NamedTuple):
    """Both pretrained networks."""

    sphere: MLPWeights
    coreshell: MLPWeights


def load_mlp_weights(path, *, expect_layers=None, dtype=jnp.float32) -> MLPWeights:
    """Load one network from an ``.npz`` written by ``tools/convert_weights.py``.

    Args:
        path: path to the ``.npz`` file.
        expect_layers: if given, the layer sizes the file must declare.
        dtype: dtype to cast parameters to.

    Returns:
    -------
        The loaded :class:`MLPWeights`.

    Raises:
    ------
        ValueError: if the declared architecture, per-layer shapes or
            activation names are not what this module can evaluate.

    """
    with np.load(path, allow_pickle=False) as npz:
        layer_sizes = tuple(int(v) for v in npz["layer_sizes"])
        activations = [str(a) for a in npz["activations"]]
        if expect_layers is not None and layer_sizes != tuple(expect_layers):
            raise ValueError(f"{path}: layer sizes {layer_sizes} != {tuple(expect_layers)}")
        unknown = set(activations) - set(ACTIVATIONS)
        if unknown:
            raise ValueError(f"{path}: unsupported activations {sorted(unknown)}")
        if activations[-1] != "linear" or set(activations[:-1]) != {"swish"}:
            raise ValueError(f"{path}: expected swish hidden layers and a linear head, got {activations}")

        layers = []
        for i in range(len(layer_sizes) - 1):
            kernel = jnp.asarray(npz[f"kernel_{i}"], dtype)
            bias = jnp.asarray(npz[f"bias_{i}"], dtype)
            want = (layer_sizes[i], layer_sizes[i + 1])
            if kernel.shape != want:
                raise ValueError(f"{path}: kernel_{i} shape {kernel.shape} != {want}")
            if bias.shape != (want[1],):
                raise ValueError(f"{path}: bias_{i} shape {bias.shape} != {(want[1],)}")
            layers.append(DenseWeights(kernel, bias))
    return MLPWeights(tuple(layers))


@functools.cache
def default_weights(dtype=jnp.float32) -> NeuralMieWeights:
    """Return the packaged pretrained weights, memoised per dtype.

    Resolved with ``importlib.resources`` relative to this package, so the
    lookup keeps working when this module is vendored under another path.
    """
    data = importlib.resources.files("jcm") / "data" / "neuralmie"
    with importlib.resources.as_file(data) as d:
        return NeuralMieWeights(
            sphere=load_mlp_weights(d / "sphere.npz", expect_layers=_SPHERE_LAYERS, dtype=dtype),
            coreshell=load_mlp_weights(d / "coreshell.npz", expect_layers=_CORESHELL_LAYERS, dtype=dtype),
        )


def apply_mlp(weights: MLPWeights, x):
    """Evaluate the MLP on ``x`` of shape ``(..., n_in)``, returning ``(..., n_out)``.

    Leading axes broadcast through ``jnp.matmul``, so scalars, columns and
    whole grids all work without reshaping. Safe under jit/grad/vmap.
    """
    *hidden, head = weights.layers
    for layer in hidden:
        x = _swish(x @ layer.kernel + layer.bias)
    return x @ head.kernel + head.bias


# --------------------------------------------------------------------------
# Domain handling
# --------------------------------------------------------------------------


def _clip_ste(x, lo, hi):
    """Clip ``x`` to ``[lo, hi]`` with a straight-through gradient.

    The forward value is clipped, but the derivative is the identity, so an
    out-of-domain input still reports the emulator's sensitivity *at the
    boundary* rather than an exactly-zero gradient. This is the pattern jcm
    issue #664 ("hard clips and floors make physics tunables unlearnable")
    names as the fix, and mirrors ``physics_interface.py``'s
    ``_cap_negative_tend``. A softplus smooth-clamp is deliberately *not*
    used: it perturbs values inside the box too, degrading the emulator's
    validated accuracy where it is valid.
    """
    clipped = jnp.clip(x, lo, hi)
    return jax.lax.stop_gradient(clipped - x) + x


def size_parameter(wavelength, radius):
    """Size parameter ``2*pi*r/lambda`` (dimensionless)."""
    return 2.0 * jnp.pi * radius / wavelength


def _rayleigh_boundary_mu_x(sigma_g):
    """``mu_x`` at which the 99.9th-percentile size parameter hits the switch."""
    return RAYLEIGH_SIZE_PARAMETER / jnp.exp(SQRT2_ERFINV_999 * jnp.log(sigma_g))


def is_rayleigh(mu_x, sigma_g):
    """Return True where 99.9% of particles have size parameter <= 0.1."""
    return mu_x <= _rayleigh_boundary_mu_x(sigma_g)


# --------------------------------------------------------------------------
# Analytic Rayleigh limit
# --------------------------------------------------------------------------


def _clausius_mossotti(m_r, m_i):
    """Return ``(|(m^2-1)/(m^2+2)|^2, Im((m^2-1)/(m^2+2)))`` in real arithmetic.

    Algebraically exact but avoids complex dtypes, which keeps this path
    float32-friendly and free of ``abs(complex)``'s non-differentiable point.
    With ``A = n^2-k^2-1``, ``C = n^2-k^2+2``, ``B = 2nk``, ``D = C^2+B^2``:
    the squared modulus is ``(A^2+B^2)/D`` and the imaginary part ``6nk/D``.
    """
    nn, kk = m_r * m_r, m_i * m_i
    a = nn - kk - 1.0
    c = nn - kk + 2.0
    b = 2.0 * m_r * m_i
    d = c * c + b * b
    return (a * a + b * b) / d, 6.0 * m_r * m_i / d


def _lognormal_moment_ratio(sigma_g):
    """Return the truncated 6th- to 3rd-moment integral ratio of the lognormal.

    A function of ``sigma_g`` alone: upstream forms this from ``erf``
    differences at ``ln(x2/mu_x) = +/- sqrt(2) erfinv(0.999) ln(sigma)``, in
    which the ``mu`` and ``lambda`` dependence cancels. Evaluated with
    ``erfc`` differences rather than ``erf`` differences because the arguments
    saturate -- at sigma_g = 2.8 the ``erf`` form loses ~7e-6 relative
    accuracy in float32 against ~1e-6 for ``erfc``.
    """
    lns = jnp.log(sigma_g)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    top = erfc((6.0 * lns - SQRT2_ERFINV_999) * inv_sqrt2) - erfc(
        (6.0 * lns + SQRT2_ERFINV_999) * inv_sqrt2
    )
    bot = erfc((3.0 * lns - SQRT2_ERFINV_999) * inv_sqrt2) - erfc(
        (3.0 * lns + SQRT2_ERFINV_999) * inv_sqrt2
    )
    return top / bot


def rayleigh_bulk_optics(wavelength, r_g, sigma_g, m_r, m_i):
    """Analytic small-particle limit, returning ``(ke_rho, ssa, g)``.

    ``ke_rho`` is in 1/m and ``g`` is exactly zero, matching upstream
    ``rayleigh_approx`` (``sphere_inference_demo.py:8-33``) with the density
    divided out.
    """
    mu_x = size_parameter(wavelength, r_g)
    lns = jnp.log(sigma_g)
    cm_abs2, cm_imag = _clausius_mossotti(m_r, m_i)
    coef = 4.0 * jnp.pi * mu_x**3 * jnp.exp(13.5 * lns * lns) * cm_abs2
    ks_rho = coef * _lognormal_moment_ratio(sigma_g) / wavelength
    ka_rho = 6.0 * jnp.pi * cm_imag / wavelength
    ke_rho = ks_rho + ka_rho
    return ke_rho, ks_rho / ke_rho, jnp.zeros_like(ke_rho)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


class MassOptics(NamedTuple):
    """Bulk optics expressed per unit particle mass."""

    ke: jnp.ndarray  # mass extinction coefficient, m^2/kg
    ssa: jnp.ndarray
    g: jnp.ndarray


class BulkOptics(NamedTuple):
    """Bulk optics of a lognormal aerosol population."""

    ke_rho: jnp.ndarray  # extinction cross-section per unit particle volume, 1/m
    ssa: jnp.ndarray  # single-scattering albedo
    g: jnp.ndarray  # scattering-weighted asymmetry parameter
    rayleigh: jnp.ndarray  # True where the analytic Rayleigh limit was used

    def per_mass(self, rho) -> MassOptics:
        """Convert to a mass extinction coefficient given a density in kg/m^3."""
        return MassOptics(self.ke_rho / rho, self.ssa, self.g)


def _bulk_optics(weights, features, wavelength, r_g, sigma_g, m_r, m_i, ray_args, clip):
    """Shared branch logic for both networks.

    The evaluation order is the whole safety argument -- see
    :func:`sphere_bulk_optics` for why the ``mu_x`` clamp is mandatory.
    """
    if clip:
        r_g = _clip_ste(r_g, *MU_RANGE)
        sigma_g = _clip_ste(sigma_g, *SIGMA_G_RANGE)
        m_r = _clip_ste(m_r, *M_R_RANGE)
        m_i = _clip_ste(m_i, *M_I_RANGE)

    mu_x = size_parameter(wavelength, r_g)
    mask = is_rayleigh(mu_x, sigma_g)

    # Clamp mu_x UP to the switch boundary before it reaches the network. The
    # nets were never trained below it (upstream drops those rows via
    # `keep = upper_bounds > 0.1`), and the coreshell net's raw output there
    # reaches +2142, so exp() overflows to inf across ~1.4% of the training
    # box and poisons `where` gradients. The clamp is active only where the
    # Rayleigh arm is selected anyway, so it is bitwise identity on every
    # value this function actually returns.
    mu_x_nn = jnp.maximum(mu_x, _rayleigh_boundary_mu_x(sigma_g))

    raw = apply_mlp(weights, features(mu_x_nn, sigma_g, m_r, m_i))
    ln_ke_lambda = jnp.clip(raw[..., 0], *LN_KE_LAMBDA_CLIP)
    ke_nn = jnp.exp(ln_ke_lambda) / wavelength

    ke_r, ssa_r, g_r = rayleigh_bulk_optics(wavelength, r_g, sigma_g, *ray_args)
    return BulkOptics(
        ke_rho=jnp.where(mask, ke_r, ke_nn),
        ssa=jnp.where(mask, ssa_r, jax.nn.sigmoid(raw[..., 1])),
        g=jnp.where(mask, g_r, jax.nn.sigmoid(raw[..., 2])),
        rayleigh=mask,
    )


def sphere_bulk_optics(wavelength, r_g, sigma_g, m_r, m_i, *, weights=None, clip=True) -> BulkOptics:
    """Bulk optics of a lognormal population of homogeneous spheres.

    Args:
        wavelength: wavelength in metres.
        r_g: number-median radius in metres (a radius, not a diameter).
        sigma_g: geometric standard deviation, dimensionless.
        m_r: real refractive index.
        m_i: imaginary refractive index, positive convention.
        weights: sphere-network weights; defaults to the packaged ones.
        clip: clip inputs to the training domain (default True).

    Returns:
    -------
        A :class:`BulkOptics` whose fields have the broadcast shape of the
        inputs.

    Notes:
    -----
        All inputs broadcast against one another. Both the analytic Rayleigh
        limit and the network are evaluated unconditionally and selected with
        ``jnp.where``, so the function is safe under ``jit``, ``vmap`` and
        ``grad``. ``wavelength`` is not clipped: the network sees it only
        through the size parameter, and it re-enters through the ``1/lambda``
        output factor.

    """
    w = weights if weights is not None else default_weights().sphere

    def features(mu_x, sigma_g, m_r, m_i):
        return jnp.stack(
            jnp.broadcast_arrays(
                (jnp.log(mu_x) + 1.5) / 3.6,
                2.0 * sigma_g - 4.0,
                2.0 * m_r - 4.0,
                (jnp.log(m_i) + 9.0) / 5.0,
            ),
            axis=-1,
        )

    return _bulk_optics(w, features, wavelength, r_g, sigma_g, m_r, m_i, (m_r, m_i), clip)


def coreshell_bulk_optics(
    wavelength, r_g, sigma_g, m_r, m_i, m_r_core, m_i_core, core_fraction,
    *, weights=None, clip=True,
) -> BulkOptics:
    """Bulk optics of a lognormal population of concentric core-shell spheres.

    Args:
        wavelength: wavelength in metres.
        r_g: number-median radius in metres.
        sigma_g: geometric standard deviation.
        m_r: shell real refractive index.
        m_i: shell imaginary refractive index, positive.
        m_r_core: core real refractive index.
        m_i_core: core imaginary refractive index, positive.
        core_fraction: ``r_core/r_total``; the volume fraction is its cube.
        weights: core-shell network weights; defaults to the packaged ones.
        clip: clip inputs to the training domain (default True).

    Returns:
    -------
        A :class:`BulkOptics` with the broadcast shape of the inputs.

    Notes:
    -----
        The Rayleigh arm volume-mixes the two indices by ``core_fraction**3``,
        which is also why this function reduces smoothly to the homogeneous
        case as the core vanishes. Upstream documents that the core-shell
        Rayleigh limit is markedly less accurate than the emulator itself.

    """
    w = weights if weights is not None else default_weights().coreshell
    if clip:
        m_r_core = _clip_ste(m_r_core, *M_R_RANGE)
        m_i_core = _clip_ste(m_i_core, *M_I_RANGE)
        core_fraction = _clip_ste(core_fraction, *CORE_FRACTION_RANGE)

    def features(mu_x, sigma_g, m_r, m_i):
        return jnp.stack(
            jnp.broadcast_arrays(
                (jnp.log(mu_x) + 1.5) / 3.6,
                2.0 * sigma_g - 4.0,
                2.0 * m_r - 4.0,
                (jnp.log(m_i) + 9.0) / 5.0,
                2.0 * m_r_core - 4.0,
                (jnp.log(m_i_core) + 9.0) / 5.0,
                4.0 * core_fraction - 2.0,
            ),
            axis=-1,
        )

    # Volume mixing for the Rayleigh arm (`coreshell_inference_demo.py`).
    f3 = core_fraction**3
    ray_args = (m_r_core * f3 + m_r * (1.0 - f3), m_i_core * f3 + m_i * (1.0 - f3))
    return _bulk_optics(w, features, wavelength, r_g, sigma_g, m_r, m_i, ray_args, clip)
