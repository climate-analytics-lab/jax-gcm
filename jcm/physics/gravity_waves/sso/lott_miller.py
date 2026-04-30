"""Lott & Miller (1997) + Lott (1999) sub-grid orographic GW drag.

Port target: ECHAM ``mo_ssodrag.f90`` (atm_phy_echam version, 1564 lines).
References: Lott & Miller (1997), QJRMS 123:101-127; Lott (1999),
MWR 127:788-801.

Phase 0 status: scaffold only. ``sso_drag`` returns zero tendencies. Real
algorithm (subroutines ``orodrag``, ``orosetup``, ``gwstress``,
``gwprofil``, ``orolift``) lands in Phase 2.

Default values below come from ``mo_ssodrag.f90`` (echam6 inline namelist
lines 41-113) and ``mo_echam_sso_config``.
"""
from typing import NamedTuple, Tuple

import jax.numpy as jnp
import tree_math


@tree_math.struct
class SSOParameters:
    """Parameters for the Lott & Miller (1997) SSO drag scheme.

    Splits cleanly into:

    - **Activation thresholds** (``gpicmea``, ``gstd``): minimum subgrid
      orography stats below which the scheme is skipped at a column.
    - **Drag coefficients** (``gkdrag``, ``gkwake``, ``gklift``): tuning
      knobs for the wave-drag, blocked-flow-wake, and mountain-lift
      branches respectively. ``gklift = 0`` disables the lift branch.
    - **Critical Froude / Richardson numbers** (``gfrcrit``, ``grcrit``):
      controlling blocking and wave saturation.
    - **Security mins** (``gssec``, ``gtsec``, ``gvsec``, ``gsigcr``):
      protect divisions from going singular.
    - **Vertical structure** (``nktopg``, ``ntop``, ``grahilo``).
    """

    # --- Activation thresholds (per-domain config) ---
    gpicmea: jnp.ndarray        # min peak-mean elevation diff (m), default 1.0
    gstd: jnp.ndarray            # min std-dev of orography (m), default 1.0

    # --- Drag-branch coefficients (per-domain config) ---
    gkdrag: jnp.ndarray          # GW drag coefficient, default 0.2
    gkwake: jnp.ndarray          # blocked-flow wake coefficient, default 1.0
    gklift: jnp.ndarray          # mountain-lift coefficient, default 0.0 (off)

    # --- Critical numbers (PARAMETER constants in echam6 source) ---
    gfrcrit: jnp.ndarray         # critical non-dim mountain height, default 0.5
    grcrit: jnp.ndarray          # critical Richardson number, default 0.25
    grahilo: jnp.ndarray         # trapped-wave fraction, default 1.0

    # --- Security / minimum values ---
    gsigcr: jnp.ndarray          # min blocked-flow depth fraction, default 0.80
    gssec: jnp.ndarray           # min low-level B-V freq, default 1e-4
    gtsec: jnp.ndarray           # min anisotropy / GW stress, default 1e-5
    gvsec: jnp.ndarray           # min ulow, default 0.10

    # --- Vertical-structure indices ---
    nktopg: jnp.ndarray          # top level for orography effects (1-indexed)
    ntop: jnp.ndarray            # top level for stress profile (1-indexed)

    @classmethod
    def default(
        cls,
        gpicmea: float = 1.0,
        gstd: float = 1.0,
        gkdrag: float = 0.2,
        gkwake: float = 1.0,
        gklift: float = 0.0,
        gfrcrit: float = 0.5,
        grcrit: float = 0.25,
        grahilo: float = 1.0,
        gsigcr: float = 0.80,
        gssec: float = 1e-4,
        gtsec: float = 1e-5,
        gvsec: float = 0.10,
        nktopg: int = 1,
        ntop: int = 1,
    ) -> "SSOParameters":
        return cls(
            gpicmea=jnp.array(gpicmea),
            gstd=jnp.array(gstd),
            gkdrag=jnp.array(gkdrag),
            gkwake=jnp.array(gkwake),
            gklift=jnp.array(gklift),
            gfrcrit=jnp.array(gfrcrit),
            grcrit=jnp.array(grcrit),
            grahilo=jnp.array(grahilo),
            gsigcr=jnp.array(gsigcr),
            gssec=jnp.array(gssec),
            gtsec=jnp.array(gtsec),
            gvsec=jnp.array(gvsec),
            nktopg=jnp.array(nktopg),
            ntop=jnp.array(ntop),
        )


class SSOState(NamedTuple):
    """Diagnostic outputs from the SSO scheme."""

    u_stress: jnp.ndarray        # column-integrated u-stress (Pa)
    v_stress: jnp.ndarray        # column-integrated v-stress (Pa)
    dissip_total: jnp.ndarray    # column-integrated energy dissipation (W/m^2)


class SSOTendencies(NamedTuple):
    """Tendencies from the SSO scheme."""

    dudt: jnp.ndarray            # m/s^2
    dvdt: jnp.ndarray            # m/s^2
    dissip: jnp.ndarray          # energy dissipation rate per mass (W/kg)


def sso_drag(
    pdtime: jnp.ndarray,
    pcoriol: jnp.ndarray,
    pzf: jnp.ndarray,
    pzs: jnp.ndarray,
    paphm1: jnp.ndarray,
    papm1: jnp.ndarray,
    pmair: jnp.ndarray,
    ptm1: jnp.ndarray,
    pum1: jnp.ndarray,
    pvm1: jnp.ndarray,
    pmea: jnp.ndarray,
    pstd: jnp.ndarray,
    psig: jnp.ndarray,
    pgam: jnp.ndarray,
    pthe: jnp.ndarray,
    ppic: jnp.ndarray,
    pval: jnp.ndarray,
    psftlf: jnp.ndarray,
    config: SSOParameters,
) -> Tuple[SSOTendencies, SSOState]:
    """Compute SSO drag tendencies for a single column.

    Mirrors the entry point ``ssodrag`` in ``mo_ssodrag.f90`` line 26. Input
    arrays use the ECHAM convention: index 0 = top of atmosphere.
    Half-level arrays (``paphm1``) have length nlev+1; full-level arrays
    have length nlev. The seven SSO descriptors (``pmea``, ``pstd``,
    ``psig``, ``pgam``, ``pthe``, ``ppic``, ``pval``) are scalars per
    column, sourced from boundary data.

    Phase 0: returns zeros. Phase 2 will replace the body with ``orodrag``
    (wave drag + blocked-flow wake) and optionally ``orolift`` (mountain
    lift, default off).
    """
    nlev = pum1.shape[0]
    zeros_full = jnp.zeros(nlev)
    zero_scalar = jnp.array(0.0)
    return (
        SSOTendencies(dudt=zeros_full, dvdt=zeros_full, dissip=zeros_full),
        SSOState(u_stress=zero_scalar, v_stress=zero_scalar, dissip_total=zero_scalar),
    )
