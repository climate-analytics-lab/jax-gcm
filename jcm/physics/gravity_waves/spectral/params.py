r"""Configuration for the frontal spectral gravity-wave drag term.

:class:`FrontalGWParameters` is a JAX pytree (``flax.struct.dataclass``):
the numeric tunables are differentiable leaves — gradients can be taken
with respect to them, like every other physics parameter container in
this repo — while the fields that fix array sizes or select code paths at
trace time (``ngwv``, spectrum shape, the level-selection pressures, the
static flags) are ``pytree_node=False`` aux data.

Numeric defaults are the CESM2.2 CAM6 **ne30** configuration of
``use_gw_front`` (ESCOMP/CAM ref ``cam_cesm2_2_rel``):

===================  ==========================  =============================
parameter            value                       CAM source
===================  ==========================  =============================
``taubgnd``          1.25e-3 Pa                  namelist_defaults ``ne30np4``
``frontgfc``         3.0e-15 K²/m²/s             namelist_defaults ``ne30np4``
``effgw``            1.0                         namelist_defaults ``effgw_cm``
``ngwv``             32                          namelist_defaults ``pgwv``
``dc``               2.5 m/s                     build-namelist ``gw_dc``
``fcrit2``           1.0                         gw_drag.F90 ``band_mid``
``wavelength``       1e5 m                       gw_drag.F90 ``wavelength_mid``
``gaussian_width``   30 m/s                      gw_drag.F90
                                                 ``front_gaussian_width``
``tndmax``           400/86400 m/s²              gw_common.F90 ``tndmax``
``umcfac``           0.5                         gw_common.F90 ``umcfac``
``satfac``           2.0                         gw_drag_prof default
``source_pressure``  50000 Pa                    gw_drag.F90 ``kbot_front``
``front_pressure``   60000 Pa                    gw_drag.F90 ``kfront``
===================  ==========================  =============================
"""

from __future__ import annotations

import enum

import jax.numpy as jnp
from flax import struct


class SpectrumShape(enum.Enum):
    """Shape of the launched stress spectrum over phase speed."""

    #: Bin-averaged Gaussian in phase speed (CAM's production choice —
    #: ``gaussian_cm_desc`` wired in ``gw_drag.F90::gw_init``).
    GAUSSIAN = "gaussian"
    #: Flat spectrum (``flat_cm_desc``; kept for completeness/testing).
    FLAT = "flat"


@struct.dataclass
class FrontalGWParameters:
    """Configuration for :class:`~...term.FrontalGravityWaveDrag`.

    Attributes:
        taubgnd: Background stress source strength [Pa]. For the Gaussian
            spectrum this is the *peak* of the Gaussian (CAM passes
            ``taubgnd`` as ``gaussian_cm_desc``'s ``height``); for the
            flat spectrum it is the per-wavenumber stress.
        frontgfc: Frontogenesis-function critical threshold [K^2/m^2/s].
        effgw: Tendency efficiency (CAM ``effgw_cm``).
        gaussian_width: Width of the Gaussian source spectrum [m/s].
        gaussian_center: Center of the Gaussian source spectrum [m/s].
            CAM has no such knob (its Gaussian is centered on c = 0); the
            default 0.0 reproduces CAM.
        dc: Phase-speed bin width [m/s].
        fcrit2: Critical Froude number squared for the band.
        wavelength: Horizontal wavelength of the band [m].
        tndmax: Maximum wind tendency [m/s^2] (400 m/s/day).
        umcfac: Maximum allowed fractional change in ``u - c`` per step.
        satfac: Saturation factor in the Lindzen stress (CAM default 2).
        ngwv: Spectrum half-width (static; wavenumbers ``-ngwv..ngwv``).
        spectrum: :class:`SpectrumShape` (static; selects the builder).
        source_pressure: Waves launch from the deepest interface with
            reference pressure below this [Pa] (static — it selects a
            level *index* at coordinate-cache time; CAM: 500 hPa).
        front_pressure: The frontogenesis trigger is evaluated at the
            deepest midpoint whose upper interface is above this [Pa]
            (static; CAM: 600 hPa).
        tau_0_ubc: Enforce tau = 0 at the model-top interface (static;
            CAM6 non-WACCM default False).
        apply_fixers: Apply the momentum and energy fixers below the
            source level (static; CAM's ``use_gw_front`` driver always
            does — disable only for solver-level debugging).
        fallback_frontogenesis: Value [K^2/m^2/s] used for the trigger
            field when no ``"frontogenesis"`` diagnostic is provided
            (static config, not a physical tunable). The default 0.0 is
            below any positive ``frontgfc``, so **without a frontogenesis
            provider the term is inert** (no waves launch, tendencies are
            exactly zero). Setting it above ``frontgfc`` forces a uniform
            launch — a testing/spin-up aid with no CAM counterpart.

    """

    # --- Differentiable tunables (pytree leaves) ----------------------------
    taubgnd: jnp.ndarray = 1.25e-3
    frontgfc: jnp.ndarray = 3.0e-15
    effgw: jnp.ndarray = 1.0
    gaussian_width: jnp.ndarray = 30.0
    gaussian_center: jnp.ndarray = 0.0
    dc: jnp.ndarray = 2.5
    fcrit2: jnp.ndarray = 1.0
    wavelength: jnp.ndarray = 1.0e5
    tndmax: jnp.ndarray = 400.0 / 86400.0
    umcfac: jnp.ndarray = 0.5
    satfac: jnp.ndarray = 2.0

    # --- Static configuration (sizes/code paths; not differentiated) --------
    ngwv: int = struct.field(pytree_node=False, default=32)
    spectrum: SpectrumShape = struct.field(
        pytree_node=False, default=SpectrumShape.GAUSSIAN)
    source_pressure: float = struct.field(pytree_node=False, default=50000.0)
    front_pressure: float = struct.field(pytree_node=False, default=60000.0)
    tau_0_ubc: bool = struct.field(pytree_node=False, default=False)
    apply_fixers: bool = struct.field(pytree_node=False, default=True)
    fallback_frontogenesis: float = struct.field(
        pytree_node=False, default=0.0)

    @classmethod
    def default(cls) -> "FrontalGWParameters":
        """Return the CESM2.2 CAM6 ne30 defaults."""
        return cls()
