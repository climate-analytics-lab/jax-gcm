"""ECHAM's ``lccnclim`` climatological mixed-phase freezing inputs (#1149).

Port of the mixed-phase block of ``mo_ccnclim.f90::ccnclim_IN_setup``
(ECHAM6.3-HAM2.3 r7492, F 524-583): the inputs ``lccnclim`` feeds
:func:`jcm.physics.clouds.lohmann_2m.deposition_freezing.het_mxphase_freezing`
in place of a prognostic aerosol's own
:func:`jcm.physics.aerosol.jam.ice_nucleation.ham_freezing.ham_freezing_aerosol`.

What the Fortran actually depends on
-------------------------------------
Despite the module's name ("CCN climatology"), the mixed-phase block
(F 547-558) has NO spatial or temporal dependence at all. Every one of the
eight fields ``het_mxphase_freezing`` needs is assigned straight from a
module-level ``CCNCLIMCTL`` namelist default (F 71-79), broadcast uniformly
over every column and level::

    prwetki(1:kproma,:)      = rwetki        ! F 549
    prwetai(1:kproma,:)      = rwetai        ! F 550
    prwetci(1:kproma,:)      = rwetci        ! F 551
    pfracdusol(1:kproma,:)   = fracdusol     ! F 554
    pfracduai(1:kproma,:)    = fracduai      ! F 555
    pfracduci(1:kproma,:)    = fracduci      ! F 556
    pfracbcsol(1:kproma,:)   = fracbcsol     ! F 557
    pfracbcinsol(1:kproma,:) = fracbcinsol   ! F 558

There is no ``READ`` of the three-dimensional CCN climatology anywhere in
this block. The actual ``ccnYYYY.nc`` climatology (``read_ccnclim``,
F 282-346, via the module's own ``ccn_tot_strat``/``ccn_tot_conv`` fields)
feeds only ``ccnclim_IN_setup``'s CIRRUS-freezing outputs (``pascs``/
``papnx``/``paprx``/``papsigx``, F 561-581) -- which ``het_mxphase_freezing``
does not take as an argument at all (compare its signature) and are out of
scope for #1149 (the cirrus scheme's own soluble-aerosol-number cap,
ECHAM's ``zascs``, is a separate, still-open gap tracked under #1159 --
see ``docs/source/design/lohmann_2m_column_processes.md``'s "Deliberate
omissions" section -- not this one). So issue #1149's own framing -- "port
``ccnclim_IN_setup`` as a pure function of location/season" -- turns out,
on reading F 524-583, not to describe the part ``het_mxphase_freezing``
needs: these eight values are the SAME constant at every location, season
and level, by construction of the compiled routine, not a climatology
lookup. The subroutine's own remark bears this out ("Here, some quantities
are surrogates for their m7 modes counterparts ... There is no dependency
to HAM and m7. These quantities were diagnosed in a preliminary full HAM
run", F 517-521): a single global snapshot, hard-coded as the namelist's
default, not a live field.

Because every output is a plain scalar, :func:`ccnclim_freezing_aerosol` is
trivially "broadcasting-native" (CLAUDE.md's column-physics convention):
whatever shape the caller's ``reference`` array has -- a single ``(kx,)``
column, a vectorized ``(kx, ncols)`` block or a whole ``(kx, ix, il)`` grid
-- the same constants broadcast onto it unchanged.

The eight constants themselves live on :class:`~jcm.physics.clouds.
lohmann_2m_params.CloudParams2M` (``ccnclim_fracdusol`` etc.) as
differentiable leaves, like the scheme's other physical tunables
(``immersion_coefficient_dust``/``_bc``) -- ECHAM hard-codes them, jcm need
not.
"""

import jax.numpy as jnp

from ..lohmann_2m_params import CloudParams2M
from .types import HeterogeneousFreezingAerosol


def ccnclim_freezing_aerosol(
    reference: jnp.ndarray,
    params: CloudParams2M,
) -> HeterogeneousFreezingAerosol:
    """ECHAM's ``lccnclim`` mixed-phase freezing inputs (``ccnclim_IN_setup``, F 524-583).

    Every field is ``params.ccnclim_*`` broadcast to ``reference``'s shape
    and dtype -- see the module docstring for why a broadcast, rather than
    any per-level or per-column computation, is the faithful port here: the
    compiled Fortran assigns the same scalar everywhere.

    Args:
        reference: any array shaped like the level-resolved fields
            ``het_mxphase_freezing`` is about to receive (e.g. the scheme's
            ``air_density``, ``(kx, *horiz)``) -- used only for its shape
            and dtype.
        params: the scheme's tunable parameters, holding the eight
            ``lccnclim`` constants as differentiable leaves.

    Returns:
        :class:`HeterogeneousFreezingAerosol` shaped and dtyped like
        ``reference``, matching :func:`jcm.physics.aerosol.jam.
        ice_nucleation.ham_freezing.ham_freezing_aerosol`'s own "shaped
        like ..." contract, so section 6.2 of ``cloud_microphysics_2m`` can
        feed either's output to ``het_mxphase_freezing`` without caring
        which one ran.

    """
    def broadcast(value):
        return jnp.broadcast_to(value, reference.shape).astype(reference.dtype)

    return HeterogeneousFreezingAerosol(
        dust_soluble=broadcast(params.ccnclim_fracdusol),
        dust_insoluble_accumulation=broadcast(params.ccnclim_fracduai),
        dust_insoluble_coarse=broadcast(params.ccnclim_fracduci),
        bc_soluble=broadcast(params.ccnclim_fracbcsol),
        bc_insoluble=broadcast(params.ccnclim_fracbcinsol),
        wet_radius_insoluble_aitken=broadcast(params.ccnclim_rwetki),
        wet_radius_insoluble_accumulation=broadcast(params.ccnclim_rwetai),
        wet_radius_insoluble_coarse=broadcast(params.ccnclim_rwetci),
    )
