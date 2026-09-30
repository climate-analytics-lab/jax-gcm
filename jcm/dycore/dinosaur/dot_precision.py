"""Keep dinosaur's float32 GPU matmuls at ``Precision.HIGHEST``.

Since 1.5.0, dinosaur resolves float32 ``einsum``/``dot`` calls to bfloat16
emulation on GPU, with two separate defaults. jcm overrides both, for
different reasons.

``jax_numpy_utils.FLOAT32_DOT_ALGORITHM`` (``BF16_BF16_F32_X6``) covers the
vertical operators and the ``RealSphericalHarmonics`` transforms of
single-device runs. The XLA bundled with jaxlib < 0.11.2 miscompiles one
shape of it: a contraction whose operand is a compile-time constant (the numpy
Legendre basis) while the other operand has a size-1 free dimension. Constant
folding permutes the constant's bf16 split, leaving ~2e-5 relative error
instead of ~1e-7. That shape is exactly the inverse transform of a
single-level field, so ``log_surface_pressure`` picks up the same error
pattern every step. In the hybrid semi-Lagrangian core this integrates into a
hemispheric mass drift: a T63L47 Held-Suarez aquaplanet reaches a -116 hPa
NH-SH surface-pressure asymmetry in 30 days, against +0.03 hPa with
``HIGHEST``. The override is gated on the jaxlib version and lifts on its own
once jcm's jax pin reaches a fixed release.

``spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM`` (``BF16_BF16_F32_X3``)
covers the ``FastSphericalHarmonics`` transforms used when ``spmd_mesh`` is
set. The constant-folding bug does not cause this one: disabling that XLA pass
leaves it unchanged. On GPU, 3-pass bfloat16 transforms measurably alter the
solution. A T63L47 Held-Suarez aquaplanet loses 0.1 hPa of global-mean
surface pressure on the first day and spins up about 4 hPa too weakly at
40-60N over 10 days, and ``HIGHEST`` restores the single-device reference. On
CPU, X3 already gives results identical to ``HIGHEST``. This override is
therefore not version-gated.

Measured cost for the dycore-only Held-Suarez case on one A100 (the worst
case, as physics dilutes it in full runs): +2 % per simulated day
single-device, +8 % with SPMD. Upstream report: neuralgcm/dinosaur#147.

``dot_precision_test.py`` has a GPU-only regression test of the triggering
shape, which fails if the override is removed on an affected jaxlib. CI runs
on CPU and skips it, so run it on a GPU host when changing the jax pin.
"""

from __future__ import annotations

import logging

import jax
import jaxlib
from dinosaur import jax_numpy_utils, spherical_harmonic
from packaging.version import Version

logger = logging.getLogger(__name__)

# First jaxlib whose XLA computes the constant-operand, size-1 bf16 contraction
# correctly (verified on A100: jaxlib 0.10.2, 0.11.0 and 0.11.1 affected).
FIXED_JAXLIB_VERSION = Version("0.11.2")

# dinosaur's own defaults. Only these values are replaced, so a caller who
# chose an algorithm before importing jcm keeps their choice.
_DINOSAUR_FLOAT32_DEFAULT = jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6
_DINOSAUR_TRANSFORM_DEFAULT = jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3


def jaxlib_is_affected(version: str | None = None) -> bool:
    """Whether ``version`` (default: the running jaxlib) has the XLA bf16 miscompile."""
    return Version(version or jaxlib.__version__) < FIXED_JAXLIB_VERSION


def apply_dot_precision_workaround(version: str | None = None) -> dict[str, bool]:
    """Resolve dinosaur's float32 GPU dots to ``Precision.HIGHEST``.

    Must run before any dinosaur computation is traced, because dinosaur reads
    both defaults at trace time. It is called on import of
    :mod:`jcm.dycore.dinosaur`, which ``import jcm`` performs.

    ``FLOAT32_DOT_ALGORITHM`` is only consulted on GPU, so that override does
    not change CPU or TPU. ``FAST_TRANSFORM_DOT_ALGORITHM`` applies on every
    platform; on CPU, X3 already matches ``HIGHEST``, and on TPU the override
    trades speed for the same accuracy as the other backends.

    Returns which of the two defaults were changed.
    """
    changed = {"float32": False, "fast_transform": False}
    if (jaxlib_is_affected(version)
            and jax_numpy_utils.FLOAT32_DOT_ALGORITHM == _DINOSAUR_FLOAT32_DEFAULT):
        jax_numpy_utils.FLOAT32_DOT_ALGORITHM = jax.lax.Precision.HIGHEST
        changed["float32"] = True
    if spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM == _DINOSAUR_TRANSFORM_DEFAULT:
        spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM = jax.lax.Precision.HIGHEST
        changed["fast_transform"] = True
    if any(changed.values()):
        logger.debug("dinosaur float32 dots -> Precision.HIGHEST: %s (jaxlib %s)",
                     changed, version or jaxlib.__version__)
    return changed
