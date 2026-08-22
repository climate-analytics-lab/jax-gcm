"""Tunable parameters for the 2-moment cloud microphysics.

Based on mo_echam_cloud_params + mo_cloud_utils from ECHAM6/ICON.

``CloudParams2M`` is a ``flax.struct.dataclass`` so every numeric tunable
is a pytree leaf visible to ``jax.grad`` when the struct is threaded
through the scheme (the term holds it in an ``nnx.Param``). The two
genuinely *static* switches — ``nic_cirrus`` (cirrus-scheme selector) and
``ldyn_cdnc_min`` (dynamic-vs-fixed CDNC floor) — select code paths with
ordinary Python ``if`` at trace time, so they are ``pytree_node=False``
aux data and must stay plain ``int``/``bool``.

There is deliberately no module-level default instance: parameters reach
the scheme only through the threaded struct, so overrides (and gradients
with respect to any leaf) actually take effect. The previous import-time
``cloud_params = CloudParams2M.default()`` bake plus per-name module
exports silently severed both.
"""

from math import pi

import jax.numpy as jnp
import numpy as np
from flax import struct

from jcm import constants as c


@struct.dataclass
class CloudParams2M:
    """Cloud parameters for the ECHAM6/ICON 2-moment microphysical scheme."""

    # Physical constants snapshot (resolved from the live jcm.constants
    # singleton when default() is called, honouring set_constants).
    tmelt: jnp.ndarray
    grav: jnp.ndarray
    cthomi: jnp.ndarray      # homogeneous freezing threshold [K]

    # Snow / ice microphysics
    cn0s: jnp.ndarray        # snow size-distribution intercept [1/m^4]
    crhoi: jnp.ndarray       # bulk ice density [kg/m^3]
    crhosno: jnp.ndarray     # bulk snow density [kg/m^3]
    ccsaut: jnp.ndarray      # Levkov ice autoconversion timescale factor
    ccraut: jnp.ndarray      # warm autoconversion tuning
    ceffmax: jnp.ndarray     # max ice effective radius [um]
    ceffmin: jnp.ndarray     # min ice effective radius [um]
    ccwmin: jnp.ndarray      # cloud water limit for cover > 0 [kg/kg]
    cqtmin: jnp.ndarray      # total water minimum for cloud presence
    cvtfall: jnp.ndarray     # snow fall-speed factor

    # Utility guards. ``epsec``/``eps`` are PHYSICAL/numerical floors (~1e-12,
    # ~1e-7) that bound denominators/quantities away from zero and *do* change
    # the forward there. ``d_epsilon`` (~1e-30) is a DIFFERENTIABILITY floor:
    # it keeps the masked/dead branch of a ``where`` strictly positive so a
    # ``sqrt``/fractional-power there has a finite derivative (issue #558), and
    # must be far below any real value so it never perturbs the physics. Do NOT
    # substitute ``eps`` for it — a 1e-7 floor on e.g. the ice-radius base
    # inflates the effective radius of small ice.
    epsec: jnp.ndarray       # small number to avoid division by zero
    xsec: jnp.ndarray        # 1 - epsec
    eps: jnp.ndarray         # float32 machine epsilon
    d_epsilon: jnp.ndarray   # absolute floor for differentiability guards only

    # Ice-crystal mass / fall-speed relation
    mi: jnp.ndarray          # ice crystal mass at volume-mean radius cri [kg]
    ri_vol_mean_1: jnp.ndarray  # vol-mean ice crystal radius, range border 1 [m]
    ri_vol_mean_2: jnp.ndarray  # vol-mean ice crystal radius, range border 2 [m]
    alfased_1: jnp.ndarray   # ice crystal fall velocity coefficients ...
    alfased_2: jnp.ndarray
    alfased_3: jnp.ndarray
    betased_1: jnp.ndarray   # ... and exponents, per size range
    betased_2: jnp.ndarray
    betased_3: jnp.ndarray

    # Minimum-CDNC bounds (SF #475: implied by max droplet size)
    cdnc_min_upper: jnp.ndarray  # [1/m^3]
    cdnc_min_lower: jnp.ndarray  # [1/m^3]
    rcd_vol_max: jnp.ndarray     # [m] max mean-volume droplet radius
    cdnc_min_fixed: jnp.ndarray  # [cm^-3] fixed floor when not dynamic
    # Half-width [1/m^3] of the smooth replacement for the hard
    # ``max(activated_cdnc - cdnc, 0)`` nucleation increment in
    # ``update_in_cloud_water``. The hard max makes the loss piecewise in the
    # aerosol activation parameters (SPA prefactor/exponent): a cell whose
    # carried CDNC crosses the activation floor flips branches and the exact
    # local gradient stops tracking the large-scale Twomey response. The
    # smooth form overshoots by at most half this width at the corner.
    # Default 1e6 m^-3 (1 cm^-3), a few percent of typical activated CDNC;
    # 0.0 recovers the hard max exactly.
    activation_smoothing: jnp.ndarray  # [1/m^3]

    # Ice crystal number concentration bounds
    icemin: jnp.ndarray      # [1/m^3]
    icemax: jnp.ndarray      # [1/m^3]

    # Reference droplet/crystal mass parameters
    mi0_rcp: jnp.ndarray     # [1/kg] reciprocal reference crystal mass

    # Fall-speed / density conversions
    fall: jnp.ndarray        # [-] fall-speed tuning constant
    rhoice: jnp.ndarray      # [kg/m^3] solid ice density
    conv_effr2mvr: jnp.ndarray  # effective radius -> mean volume radius
    clc_min: jnp.ndarray     # lower cloud-fraction limit in conversions

    # KK2000-style integrated sink exponents (autoconversion)
    exm1_1: jnp.ndarray
    exp_1: jnp.ndarray

    # Snow collection / accretion tuning
    fact_coll_eff: jnp.ndarray  # temp-dependent collection efficiency factor
    fact_tke: jnp.ndarray       # turbulence enhancement factor

    # Local accretion geometry (ECHAM zauloc = cauloc·dz/5000, clipped to
    # [clmin, clmax]): the fraction of the box in which NEWLY formed
    # rain/snow participates in accretion, a resolution-(layer-depth-)
    # dependent ramp (mo_cloud_micro_2m.f90:1618-1619). ECHAM zeroes it
    # additionally at the two levels around a diagnosed low-level
    # inversion when the large-scale vertical velocity is downward
    # (knvb/lonacc gate) — that needs pvervel, which is not plumbed to
    # this scheme yet (#705).
    cauloc: jnp.ndarray      # [-] zauloc ramp factor (ECHAM 2M hardwires 3)
    clmin: jnp.ndarray       # [-] lower zauloc clip (ECHAM 0.0)
    clmax: jnp.ndarray       # [-] upper zauloc clip (ECHAM 0.5)

    # Pruppacher & Klett (1997) ice mass-size relation
    fact_PK: jnp.ndarray     # [-] (g, cm) parameter; see cloud_utils notes
    pow_PK: jnp.ndarray      # [-]

    # Derived density shortcuts
    pirho_rcp: jnp.ndarray   # 1 / (pi * rhoh2o)
    cap: jnp.ndarray         # 2 / pi (capacitance factor, plate-like ice)
    cons4: jnp.ndarray       # 1 / (pi*crhosno*cn0s)^0.8125

    # Prescribed coarse-mode aerosol number (d > 0.5 um) for the DeMott
    # (2010) INP heterogeneous nucleation parameterization [cm^-3 at STP].
    n_aer_coarse: jnp.ndarray

    # Static code-path selectors (trace-time Python branches; NOT leaves).
    # nic_cirrus=1: diagnostic ICNC from ice mass and mean crystal radius
    # (ECHAM licnc-diagnostic configuration) — the working default.
    # nic_cirrus=2 expects an EXTERNAL cirrus-nucleation source (ECHAM-HAM
    # Kaercher-Lohmann pnicex) that jcm does not compute yet (issue #552):
    # with it, sub-cthomi cells never nucleate ICNC, depositional growth
    # stalls, and supersaturation grows unboundedly (RH 5-10 over the
    # Antarctic winter surface killed the first coupled year runs).
    nic_cirrus: int = struct.field(pytree_node=False, default=1)
    ldyn_cdnc_min: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def default(
        cls,
        tmelt: float | None = None,
        grav: float | None = None,
        cthomi: float | None = None,
        cn0s: float = 3e6,
        crhoi: float = 500.0,
        crhosno: float = 100.0,
        ccsaut: float = 95.0,
        ccraut: float = 15.0,
        ceffmax: float = 150.0,
        ceffmin: float = 10.0,
        ccwmin: float = 1e-7,
        cqtmin: float = 1e-12,
        cvtfall: float = 2.5,
        epsec: float = 1e-12,
        # cri: assumed volume-mean radius of ice crystals produced when
        # melting; only enters through the derived crystal mass ``mi``.
        cri: float = 10e-6,
        ri_vol_mean_1: float = 2.166e-9,
        ri_vol_mean_2: float = 4.264e-8,
        alfased_1: float = 63292.4,
        alfased_2: float = 8.78,
        alfased_3: float = 329.75,
        betased_1: float = 0.5727,
        betased_2: float = 0.0954,
        betased_3: float = 0.3091,
        cdnc_min_upper: float = 40.0e6,
        cdnc_min_lower: float = 1.0e6,
        rcd_vol_max: float = 19.0e-6,
        icemin: float = 10.0,
        icemax: float = 1.0e7,
        mi0: float = 1.0e-12,
        fall: float = 3.0,
        rhoice: float = 925.0,
        conv_effr2mvr: float = 0.9,
        clc_min: float = 0.01,
        exm1_1: float = 2.47 - 1.0,
        exp_1: float = -1.0 / (2.47 - 1.0),
        fact_coll_eff: float = 0.09,
        fact_tke: float = 0.7,
        cauloc: float = 3.0,
        clmin: float = 0.0,
        clmax: float = 0.5,
        fact_PK: float = 8.253e-3,
        pow_PK: float = 2.475,
        cdnc_min_fixed: float = 40.0,  # [cm^-3] ECHAM warm-microphysics floor; KK2000 autoconv (rate ∝ Nc^-1.79) runs away below this in clean air
        activation_smoothing: float = 1.0e6,  # [1/m^3] smooth-max half-width for the nucleation increment
        n_aer_coarse: float = 0.5,
        nic_cirrus: int = 1,
        ldyn_cdnc_min: bool = False,
    ) -> 'CloudParams2M':
        """Return default cloud parameters for the 2-moment scheme."""
        # Derived helpers — Python/numpy math so default() stays safe to
        # call under any tracing context.
        xsec = 1.0 - epsec
        eps_val = float(np.finfo(np.float32).eps)
        mi_val = 4.0 / 3.0 * cri ** 3 * pi * crhoi
        pirho_rcp_val = 1.0 / (pi * 1000.0)
        cap_val = 2.0 / pi
        cons4_val = 1.0 / (pi * crhosno * cn0s) ** 0.8125

        # Resolve constant-derived defaults here (not in the signature) so
        # they read the live jcm.constants singleton when default() is
        # *called*, honouring a set_constants override applied before model
        # construction.
        if tmelt is None:
            tmelt = c.tmelt
        if grav is None:
            grav = c.grav
        if cthomi is None:
            cthomi = c.tmelt - 35.0

        return cls(
            tmelt=jnp.array(tmelt),
            grav=jnp.array(grav),
            cthomi=jnp.array(cthomi),
            cn0s=jnp.array(cn0s),
            crhoi=jnp.array(crhoi),
            crhosno=jnp.array(crhosno),
            ccsaut=jnp.array(ccsaut),
            ccraut=jnp.array(ccraut),
            ceffmax=jnp.array(ceffmax),
            ceffmin=jnp.array(ceffmin),
            ccwmin=jnp.array(ccwmin),
            cqtmin=jnp.array(cqtmin),
            cvtfall=jnp.array(cvtfall),
            epsec=jnp.array(epsec),
            xsec=jnp.array(xsec),
            eps=jnp.array(eps_val),
            d_epsilon=jnp.array(1.0e-30),
            mi=jnp.array(mi_val),
            ri_vol_mean_1=jnp.array(ri_vol_mean_1),
            ri_vol_mean_2=jnp.array(ri_vol_mean_2),
            alfased_1=jnp.array(alfased_1),
            alfased_2=jnp.array(alfased_2),
            alfased_3=jnp.array(alfased_3),
            betased_1=jnp.array(betased_1),
            betased_2=jnp.array(betased_2),
            betased_3=jnp.array(betased_3),
            cdnc_min_upper=jnp.array(cdnc_min_upper),
            cdnc_min_lower=jnp.array(cdnc_min_lower),
            rcd_vol_max=jnp.array(rcd_vol_max),
            cdnc_min_fixed=jnp.array(cdnc_min_fixed),
            activation_smoothing=jnp.array(activation_smoothing),
            icemin=jnp.array(icemin),
            icemax=jnp.array(icemax),
            mi0_rcp=jnp.array(1.0 / mi0),
            fall=jnp.array(fall),
            rhoice=jnp.array(rhoice),
            conv_effr2mvr=jnp.array(conv_effr2mvr),
            clc_min=jnp.array(clc_min),
            exm1_1=jnp.array(exm1_1),
            exp_1=jnp.array(exp_1),
            fact_coll_eff=jnp.array(fact_coll_eff),
            fact_tke=jnp.array(fact_tke),
            cauloc=jnp.array(cauloc),
            clmin=jnp.array(clmin),
            clmax=jnp.array(clmax),
            fact_PK=jnp.array(fact_PK),
            pow_PK=jnp.array(pow_PK),
            pirho_rcp=jnp.array(pirho_rcp_val),
            cap=jnp.array(cap_val),
            cons4=jnp.array(cons4_val),
            n_aer_coarse=jnp.array(n_aer_coarse),
            nic_cirrus=int(nic_cirrus),
            ldyn_cdnc_min=bool(ldyn_cdnc_min),
        )
