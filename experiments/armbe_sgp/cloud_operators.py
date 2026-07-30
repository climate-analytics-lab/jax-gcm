"""Reviewed cloud observation operators for ARMBE diagnostic experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax


def cloudc(diagnostics: Mapping[str, jax.Array]) -> jax.Array:
    """Return SPEEDY's primary cloud-fraction diagnostic."""
    return diagnostics["cloudc"]


def cloudc_plus_cloudstr_raw(diagnostics: Mapping[str, jax.Array]) -> jax.Array:
    """Return the literal cloudc plus cloudstr sum without an overlap assumption."""
    return diagnostics["cloudc"] + diagnostics["cloudstr"]


OPERATORS: Mapping[str, Callable[[Mapping[str, jax.Array]], jax.Array]] = {
    "cloudc": cloudc,
    "cloudc_plus_cloudstr_raw": cloudc_plus_cloudstr_raw,
}


def get_operator(name: str) -> Callable[[Mapping[str, jax.Array]], jax.Array]:
    """Return a reviewed cloud operator selected by configuration name."""
    try:
        return OPERATORS[name]
    except KeyError as error:
        raise ValueError(f"unknown cloud operator {name!r}; choose from {sorted(OPERATORS)}") from error
