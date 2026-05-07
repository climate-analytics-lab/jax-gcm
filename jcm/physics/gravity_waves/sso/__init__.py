"""Sub-grid orographic gravity-wave drag (Lott & Miller 1997, Lott 1999).

Port of ECHAM ``mo_ssodrag.f90``. The real implementation lives in
:mod:`.lott_miller`. This package init re-exports the public symbols.
"""

from .lott_miller import (
    LottMillerSso,
    SSOParameters,
    SSOState,
    SSOTendencies,
    sso_drag,
)

__all__ = [
    "LottMillerSso",
    "SSOParameters",
    "SSOState",
    "SSOTendencies",
    "sso_drag",
]
