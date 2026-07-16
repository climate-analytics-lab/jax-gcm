"""Smoke test: does SPEEDY physics run inside the shipped SingleColumnModel?

SPEEDY is not (yet) exercised by ``jcm/single_column_model_test.py`` (only
Held-Suarez and ECHAM are), so before building the ARMBE data adapter we
confirm ``speedy_physics()`` drops into ``SingleColumnModel`` and produces
finite tendencies plus its precipitation diagnostics (``precls``/``precnv``).

Mirrors ``jcm/single_column_model_test.py::_make_column_state`` for the
synthetic column. Run with::

    JAX_PLATFORMS=cpu /data/MOSAIC/.venv/bin/python smoke_test.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.constants import grav
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics_interface import PhysicsState
from jcm.single_column_model import SingleColumnModel


def make_column_state(nlev: int) -> PhysicsState:
    """A vertically stratified 1-D column (troposphere-like profile)."""
    z = jnp.linspace(0, 30000, nlev)[::-1]
    t_profile = jnp.maximum(288.0 - 6.5e-3 * z, 200.0)
    q_profile = 0.012 * jnp.exp(-z / 3000.0)
    return PhysicsState(
        u_wind=jnp.full(nlev, 5.0),
        v_wind=jnp.zeros(nlev),
        temperature=t_profile,
        specific_humidity=q_profile,
        geopotential=grav * z,
        normalized_surface_pressure=jnp.asarray(1.0),
        tracers={"qc": jnp.zeros(nlev), "qi": jnp.zeros(nlev)},
    )


def _unwrap(o):
    """SPEEDY stores each term's diagnostics as a 0-d object array wrapping a
    dataclass (CondensationData, SurfaceFluxData, ...). Pull the object out."""
    if getattr(o, "dtype", None) == object and getattr(o, "shape", None) == ():
        return o.item()
    return o


def _term_field(physics_data, term, field):
    """Read ``physics_data[term].field`` through the object-array wrapper."""
    obj = _unwrap(physics_data[term])
    d = vars(obj) if hasattr(obj, "__dict__") else obj
    return np.asarray(d[field])


def main() -> int:
    nlev = 8
    scm = SingleColumnModel(
        physics=speedy_physics(),
        vertical=SigmaCoordinates.equidistant(nlev),
        lat_deg=36.6,
        lon_deg=-97.5,
        dt_seconds=3600.0,
    )
    state = make_column_state(nlev)
    n_steps = 4
    pred = scm.run([state] * n_steps)

    # 1. Tendencies are finite and correctly shaped (n_steps, nlev).
    temp_tend = np.asarray(pred.tendencies.temperature)
    assert temp_tend.shape == (n_steps, nlev), temp_tend.shape
    assert np.all(np.isfinite(temp_tend)), "non-finite temperature tendency"
    print(f"temperature tendency shape={temp_tend.shape} finite=OK "
          f"range=[{temp_tend.min():.2e}, {temp_tend.max():.2e}] K/s")

    # 2. The diagnostics we will compare against ARMBE are present and finite.
    #    (Field names confirmed by inspecting the SPEEDY term dataclasses.)
    targets = {
        "precls (large-scale precip)": ("_condensation", "precls"),
        "precnv (convective precip)": ("_convection", "precnv"),
        "rsds (sfc SW down)": ("_shortwave_rad", "rsds"),
        "rlds (sfc LW down)": ("_surface_flux", "rlds"),
        "shf (sensible heat)": ("_surface_flux", "shf"),
        "evap (evaporation)": ("_surface_flux", "evap"),
    }
    print("\nARMBE comparison diagnostics:")
    for label, (term, field) in targets.items():
        arr = _term_field(pred.physics_data, term, field)
        finite = bool(np.all(np.isfinite(arr)))
        assert finite, f"non-finite {term}.{field}"
        print(f"  {label:32s} {term}.{field:8s} shape={str(arr.shape):14s} "
              f"finite=OK range=[{arr.min():.3e}, {arr.max():.3e}]")

    total_precip = (_term_field(pred.physics_data, "_condensation", "precls")
                    + _term_field(pred.physics_data, "_convection", "precnv"))
    print(f"\ntotal precip (precls+precnv) range: "
          f"[{total_precip.min():.3e}, {total_precip.max():.3e}] (SPEEDY units)")

    print(f"\nbackend: {jax.default_backend()}  devices: {len(jax.devices())}")
    print("SMOKE TEST PASSED: SPEEDY runs in SingleColumnModel with the "
          "ARMBE-comparison diagnostics available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
