"""Shared data structures and timestep constants for the Lohmann 2M scheme.

Split out of the monolithic ``lohmann_2m.py`` module (pure move, no
numerical change) so the process submodules can share the tendency
container and the ECHAM timestep-derived constants without cycles.
This module must not import any other module from this package.
"""

import jax.numpy as jnp
from typing import NamedTuple
from math import pi

import jcm.constants as c

from ..lohmann_2m_params import CloudParams2M


class MicrophysicsTendencies_2M(NamedTuple):
    """Tendencies from microphysics processes"""
    
    dtedt: jnp.ndarray          # Temperature tendency (K/s)
    dqdt: jnp.ndarray           # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray          # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray          # Cloud ice tendency (kg/kg/s)
    dqncdt: jnp.ndarray         # Cloud droplet number tendency (1/kg/s)
    dqnidt: jnp.ndarray         # Cloud ice crystal number tendency (1/kg/s)
    dqrdt: jnp.ndarray          # Rain water tendency (kg/kg/s)
    dqsdt: jnp.ndarray          # Snow tendency (kg/kg/s)

class ScavengingLedger(NamedTuple):
    """The ECHAM-HAM wet-scavenging interface (``cloud_subm_2``, #708).

    ECHAM-HAM does not reconstruct aerosol scavenging from cover x
    grid-mean state: ``cloud_subm_2`` receives the process-time ledger the
    microphysics itself integrated. These are its per-level equivalents,
    per column (each field is ``(nlev,)`` inside the scheme, stacked to
    ``(nlev, ncols)`` by the term's vmap). Published on ``CloudData`` for
    the JAM wet-deposition and cloud-borne exchange terms.

    ``incloud_liquid`` / ``incloud_ice`` are ECHAM's ``zmlwc``/``zmiwc``:
    the IN-CLOUD condensate captured just before precipitation formation
    (section 7), after the assembly's faithful zeroing (cells whose
    post-write-back cover fell below ``clc_min``, or whose value is below
    1e-20, carry zero — mo_cloud_micro_2m.f90:3660-3665). A zeroed pool
    with a positive formation rate therefore marks a cell the step fully
    converted to precipitation; consumers must treat that as
    scavenged-fraction 1, not 0 (the one documented deviation from
    HAMMOZ's ``prep_wetdep_hydro``, which maps it to 0 and misses the
    removal — the #708 dead zone).

    The formation rates are IN-CLOUD [kg/kg/s]: ``rain_formation`` is
    ``zmratepr`` (KK2000 autoconversion + both accretion pathways),
    ``snow_formation`` is ``zmrateps`` (ice-sedimentation seed per ECHAM
    1243, overwritten by aggregation + ice accretion where the cold chain
    runs, per the Fortran MERGE at 3310), and ``liquid_riming`` is
    ``zmsnowacl`` (cloud droplets collected by falling snow — a LIQUID
    sink into the frozen precip).

    ``process_cloud_fraction`` is the cover the processes actually ran
    under (pre-write-back ``paclc``): the in-droplet share of interstitial
    aerosol during the step, where the post-microphysics
    ``CloudData.cloud_fraction`` is already 0 in emptied cells.

    ``condensate_evaporation`` is the grid-mean cloud-condensate
    evaporation ledger [kg/kg/s] (``zxlevap + zxievap``, #707): condensate
    in cloud-free cells plus the clear-sky share of positive increments
    returned to vapour. This is the resuspension key — a droplet that
    evaporates releases its aerosol; one that rains out does not.
    """

    incloud_liquid: jnp.ndarray            # zmlwc [kg/kg, in-cloud]
    incloud_ice: jnp.ndarray               # zmiwc [kg/kg, in-cloud]
    rain_formation: jnp.ndarray            # zmratepr/dt [kg/kg/s, in-cloud]
    snow_formation: jnp.ndarray            # zmrateps/dt [kg/kg/s, in-cloud]
    liquid_riming: jnp.ndarray             # zmsnowacl/dt [kg/kg/s, in-cloud]
    process_cloud_fraction: jnp.ndarray    # paclc at process time [1]
    condensate_evaporation: jnp.ndarray    # (zxlevap+zxievap)/dt [kg/kg/s]


def microphysics_dt_constants(dt: jnp.ndarray, params: CloudParams2M) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constants that depend on the microphysics timestep. Here for consistency with ECHAM6,
    where they cannot be parameters.
    Constants are defined locally in each subroutine where needed.
    """
    ztmst = dt
    ztmst_rcp = 1.0 / jnp.maximum(ztmst, params.eps)
    zcons1 = c.cpd*c.vtmpc2
    # Match the ECHAM Fortran (mo_cloud_micro_2m.f90 line 535):
    # ``zcons2 = ztmst_rcp * rgrav = 1 / (dt * g)``. The earlier port had
    # ``ztmst * rgrav`` which was dt^2 too large in every site that uses
    # zcons2 to convert ``pdp * mmr`` into a flux (kg/m^2/s) — so the
    # large-scale surface precip diagnostic came out ~dt^2 (~5x10^5 at
    # dt=12 min) too large, and the latent heat in melt/sub paths was
    # similarly mis-scaled.
    zcons2 = ztmst_rcp * c.rgrav
    zcons3 = 1.0 / ( pi*params.crhosno*params.cn0s*params.cvtfall**(1.0/1.16) )**0.25
    
    return ztmst, ztmst_rcp, zcons1, zcons2, zcons3
