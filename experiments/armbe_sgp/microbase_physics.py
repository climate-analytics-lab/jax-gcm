"""Lightweight thermodynamic constants for MICROBASE reduction.

These values are a frozen snapshot of ``jcm.constants`` for processing schema 4.
Keeping the snapshot explicit makes the observational operator reproducible on
machines that do not have the full JCM/JAX runtime installed. The repository
test suite checks that the snapshot still agrees with the canonical constants.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


GRAV = 9.81
CPD = 1004.64
AKAP = 2.0 / 7.0
RD = AKAP * CPD
RV = 461.0
EPS = 0.622
TMELT = 273.15


def constants_record() -> dict[str, float | int | str]:
    """Return the frozen constants and their stable fingerprint."""
    values: dict[str, float | int | str] = {
        "schema_version": 1,
        "grav": GRAV,
        "cpd": CPD,
        "akap": AKAP,
        "rd": RD,
        "rv": RV,
        "eps": EPS,
        "tmelt": TMELT,
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    values["sha256"] = hashlib.sha256(encoded).hexdigest()
    return values


def saturation_vapor_pressure_hpa(t_kelvin: np.ndarray) -> np.ndarray:
    """Return Magnus/Tetens saturation vapor pressure over water in hPa."""
    t_c = np.asarray(t_kelvin) - TMELT
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
