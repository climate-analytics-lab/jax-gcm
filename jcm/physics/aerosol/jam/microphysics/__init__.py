"""Interchangeable aerosol microphysics cores for the JAM harness.

Cores whose adapter module imports an **optional/heavy dependency at import
time** (e.g. ``Mam4JaxMicrophysics``, which pulls the GPL-3.0 ``mam4-jax``) are
resolved **lazily** via module ``__getattr__``, so importing *this* package — or
the ``jam`` package above it — never triggers that import. Plain jcm imports
therefore stay Apache-only, and CI without the ``jcm[mam4]`` extra can still
collect every non-mam4 JAM test. The dependency is pulled in only when the core
is actually accessed (``from ...microphysics import Mam4JaxMicrophysics``,
construction, or an explicit import of its adapter module). When that dependency
is not installed, the access raises a ``ModuleNotFoundError`` that names the
install line, rather than a bare ``No module named ...``: the default JAM
configuration runs M7, whose m7-jax is not on PyPI, so the bare error would
leave a PyPI user with no way forward.

To add a microphysics core:

* **pure jcm** (no optional dep): import it eagerly below, like
  :class:`PlaceholderMicrophysics`.
* **wraps an optional dependency**: add one entry to ``_LAZY_CORES`` (the
  module that defines it, the optional package, and how to install that
  package). The ``__getattr__`` mechanism below never needs to change.
"""

import importlib

from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.mam4_data import (
    MAM4_SPEC,
)
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
# After mam4_data/placeholder: m7_data.py pulls in emissions.dust and
# ice_nucleation.ham_freezing, neither of which import this package, but
# mam4_data must already be fully loaded first (dust.py reads MAM4_SPEC).
from jcm.physics.aerosol.jam.microphysics.m7_data import (
    M7_SPEC,
)

#: The m7-jax commit the ``jcm[m7]`` extra is validated against. m7-jax is not
#: on PyPI yet, so the extra names it (``m7-jax==0.2.0.dev0``, every build of
#: this commit's branch) and it installs from GitHub first. This is the one
#: copy a user reads, in the error a missing core raises; a test holds it equal
#: to ``tools/ci/extras_preinstall.txt``'s line and to the install docs.
M7_JAX_REQUIREMENT = (
    "m7-jax @ git+https://github.com/reflective-org/M7-JAX.git"
    "@8fe0fa7496b86026523984d6c573c1555c86c90e")

#: Cores deferred to first access: attribute name -> (module that defines it,
#: the optional package that module imports, how to install that package).
#: Listed here precisely because importing their module pulls an optional
#: dependency we don't want on the plain-import path.
_LAZY_CORES = {
    "Mam4JaxMicrophysics": (
        "jcm.physics.aerosol.jam.microphysics.mam4_jax", "mam4_jax",
        "The MAM4 JAM core (jam_microphysics: mam4_jax or mam4_jax_astem; "
        "physics=echam-jam, which t63-echam-jam-mam4, the ma-* sweep and the "
        "AeroCom configurations compose) needs the jcm[mam4] extra: "
        "pip install 'jcm[mam4]'"),
    "M7JaxMicrophysics": (
        "jcm.physics.aerosol.jam.microphysics.m7_jax", "m7_jax",
        "The M7 JAM core (jam_microphysics: m7_jax; the default JAM "
        "configuration t63-echam-jam) needs the jcm[m7] extra. m7-jax is "
        "not on PyPI yet, so install its pinned commit from GitHub first:\n"
        f"    pip install \"{M7_JAX_REQUIREMENT}\"\n"
        "    pip install 'jcm[m7]'\n"
        "or run the MAM4 configuration, t63-echam-jam-mam4, whose extra "
        "(jcm[mam4]) installs from PyPI."),
}

__all__ = [
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "MAM4_SPEC",
    "M7_SPEC",
    *_LAZY_CORES,
]


def _missing_core(exc: ModuleNotFoundError, package: str) -> bool:
    """Whether ``exc`` is the optional core ``package`` itself being absent.

    Not a missing module the core's own code imports, which is a broken
    install and keeps its own message.
    """
    return (exc.name or "").split(".")[0] == package


def __getattr__(name):
    """Resolve a deferred core (PEP 562) on first access, keeping its import lazy.

    Raises:
        ModuleNotFoundError: The core's optional package is not installed; the
            message names its install line (``_LAZY_CORES``).

    """
    entry = _LAZY_CORES.get(name)
    if entry is not None:
        module, package, install = entry
        try:
            return getattr(importlib.import_module(module), name)
        except ModuleNotFoundError as exc:
            if not _missing_core(exc, package):
                raise
            raise ModuleNotFoundError(install, name=exc.name) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
