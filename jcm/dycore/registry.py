"""Name → factory registry for dynamical-core backends.

Backends register themselves at import time via the :func:`register_dycore`
decorator. :func:`build_dycore` looks up a registered name and invokes the
factory with the supplied kwargs. The v2.0 Hydra runner still constructs the
shipped Dinosaur backend explicitly; the registry is currently used by
Python-API integrations and tests.

The registry is deliberately minimal: it carries names and factories and nothing
else. Anything that needs to vary per-backend (config schema, default kwargs,
required imports) is the factory's concern.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from jcm.dycore.base import DynamicalCore


_REGISTRY: Dict[str, Callable[..., DynamicalCore]] = {}


def register_dycore(name: str) -> Callable[[Callable[..., DynamicalCore]], Callable[..., DynamicalCore]]:
    """Register a dycore factory under ``name``.

    Usage::

        @register_dycore("dinosaur")
        def _build_dinosaur(**kwargs) -> DinosaurDycore:
            return DinosaurDycore(**kwargs)

    Re-registering the same name overwrites the previous entry (useful for
    tests that swap in fakes). Raises if ``name`` is empty.
    """
    if not name:
        raise ValueError("dycore name must be a non-empty string")

    def _wrap(factory: Callable[..., DynamicalCore]) -> Callable[..., DynamicalCore]:
        _REGISTRY[name] = factory
        return factory

    return _wrap


def build_dycore(name: str, /, **kwargs: Any) -> DynamicalCore:
    """Construct the dycore registered under ``name`` with ``kwargs``.

    Raises :class:`KeyError` with a descriptive message listing the
    currently-registered names if ``name`` is unknown — much friendlier than
    the bare ``KeyError`` a dict access would produce.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown dycore {name!r}. Registered: {available}. "
            "Did you forget to import the backend module so its "
            "@register_dycore call runs?"
        )
    return _REGISTRY[name](**kwargs)


def list_dycores() -> tuple[str, ...]:
    """Return the names of all currently-registered dycore backends."""
    return tuple(sorted(_REGISTRY))
