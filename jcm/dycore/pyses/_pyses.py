"""Lazy loader for the optional ``pyses`` dependency.

The pySES CAM-SE backend is an optional extra (``pip install jcm[pyses]``).
Nothing under :mod:`jcm.dycore.pyses` imports ``pyses`` at module import time —
everything goes through :func:`require_pyses` so that

* ``import jcm.dycore.pyses`` (e.g. via the dycore registry) succeeds even
  when pyses is not installed, failing only at construction time with a
  clear, actionable message; and
* the pyses backend selection happens *before* the first ``pyses`` import.
  pyses freezes its array backend (numpy / jax / torch) from the
  ``PYSES_BACKEND`` environment variable at import time; jax-gcm requires the
  jax backend, so we default the variable here. A user who has already
  imported pyses with a non-jax backend gets an explicit error rather than
  silent numpy arrays leaking into the jitted model step.

Side effect to be aware of: importing pyses with its default
``PYSES_USE_DOUBLE=1`` calls ``jax.config.update("jax_enable_x64", True)``
process-wide. The CAM-SE dynamics *needs* float64 (the explicit RK core is
noise-sensitive in the thin upper layers), so this backend deliberately keeps
that default and instead casts the physics-facing state down to float32 in
:meth:`PysesCamSEDycore.to_physics_state` (see ``dycore.py`` for the full
float64-dynamics / float32-physics precision contract).
"""

from __future__ import annotations

import os


_INSTALL_MSG = (
    "The pySES CAM-SE dynamical core requires the optional 'pyses' package "
    "(>= 0.1.3a2). Install it with `pip install jcm[pyses]` or "
    "`pip install pyses`."
)


def require_pyses():
    """Import pyses on the jax backend and return its backend object.

    Returns the pyses backend struct (``pyses._config.Backend``), whose
    ``.np`` attribute is the backend array namespace (``jax.numpy`` for the
    jax backend). Raises :class:`ImportError` with an actionable message when
    pyses is missing or was already imported on a non-jax backend.
    """
    # Must be set before the first `import pyses` anywhere in the process.
    os.environ.setdefault("PYSES_BACKEND", "jax")
    try:
        from pyses._config import get_backend
    except ImportError as e:  # pragma: no cover - exercised only without pyses
        raise ImportError(_INSTALL_MSG) from e

    backend = get_backend()
    if backend.wrapper_type != "jax":
        raise ImportError(
            "jax-gcm's pyses backend requires PYSES_BACKEND=jax, but pyses "
            f"was initialised with the {backend.wrapper_type!r} backend. "
            "Set PYSES_BACKEND=jax before importing pyses."
        )
    return backend
