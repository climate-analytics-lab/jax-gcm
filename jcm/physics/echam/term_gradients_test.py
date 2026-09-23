"""Every ECHAM term's derivative, term by term, on one column.

``gradient_finiteness_test`` asks whether a *whole rollout* differentiates; this
module asks the same question of each ``PhysicsTerm`` on its own, so a term that
loses its gradient is named rather than merely making the stack NaN. Two things
are checked per (term, operating point):

* **finiteness** — one jvp and one vjp through the term, with a per-leaf
  direction from :mod:`jcm.testing`, must be finite. This is the #558 /
  masked-``inf`` class and needs no tolerance and no assumption about which
  outputs are live. It is asserted directly rather than through
  ``check_gradients(reference="adjoint")`` because that reference additionally
  requires *every* differentiable output leaf to be non-zero, which no term
  satisfies across a whole diagnostics struct at a single operating point — a
  warm column has no snow flux, a cloud-free layer no effective radius.
* **a reference** — ``check_gradients`` against a central difference where one
  exists, and against the adjoint identity plus named live inputs where the
  term's own activation boundary means no two-sided derivative exists (see
  ``_CHECKS``).

Three choices shape what these checks mean.

**One column, not a grid.** The perturbation is a fraction of each leaf's RMS,
so on a 2048-column grid *some* column crosses *some* ``where`` threshold at
every rung of the ladder and no step is ever usable — the failure would be a
property of the grid, not of the term. One column at a chosen sounding makes
every branch the term takes one this file picked.

**Two operating points.** ``stable`` is a subtropical column with no convection
and no orography; ``convecting`` is a moist tropical column over a hill, where
convection, precipitation and the orographic drag are all active. Several terms
are differentiable at one and sit on an activation boundary at the other, and
that difference is itself the result.

**The column's geometry is held fixed.** ``MoistAirColumnState`` publishes
pressure, density, layer thickness and heights as deterministic functions of
the state; the model recomputes them every step. Perturbing them *independently
of* the state they come from is not a direction the model can move in, and with
a step relative to each leaf's RMS it is actively destructive: ``pressure_half``
spans 1 Pa to 1e5 Pa, so a 0.05% displacement of its RMS is 25 Pa — many times
the whole pressure of the top levels, which it drives negative. Those eight
fields (:data:`_FIXED_GEOMETRY`) are therefore closed over, not passed as
arguments. The rest of what that term publishes is *not*: ``thermo_run`` is the
environment temperature and humidity most schemes actually read — Tiedtke's
``temperature_env`` comes from there, not from ``state.temperature`` — so
freezing it would sever the state's own path into half the package and report
``state/temperature`` as a dead input. Everything else the term reads — the
prognostic state, the upstream diagnostics, forcing and terrain — is perturbed.

The outputs checked are the tendency ledger plus the diagnostics the term
declares in ``provides``, i.e. its own output contract. Diagnostics that merely
pass through would otherwise contribute an identity derivative to the
projection and dilute what the term actually computed. A cell may narrow that
further (``_Check.outputs`` / ``_Check.skip_outputs``) where the term returns a
field as a structural zero — a microphysics scheme's momentum tendency, a
gravity-wave scheme's moisture tendency — because the ``"adjoint"`` reference
requires every output it is given to be live.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import jcm.constants as c
from jcm.column_coordinates import ColumnCoordinates
from jcm.forcing import ForcingData, SolarGeometry
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    saturation_mixing_ratio,
)
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData
from jcm.testing import (_cotangent, _leaf_names, _tangent,
                         check_gradients)

# Replaying twelve terms and walking a step ladder for each takes minutes in
# total, which is a slow-suite cost rather than a per-push one. The individual
# cases are seconds each once the module-scoped replay is cached.
pytestmark = pytest.mark.slow

_NLEV = 47


@dataclasses.dataclass(frozen=True)
class _OperatingPoint:
    """A single column and the surface it sits over."""

    sst: float
    #: Relative humidity in the moist lower troposphere and well above it; the
    #: profile decays from one to the other over ``moist_depth_m``. A single
    #: constant will not do: Sundqvist diagnoses no cloud at all below its
    #: critical RH, and a term downstream of a cloud-free column is inert, so
    #: the check would be measuring nothing.
    rh_boundary_layer: float
    rh_free_troposphere: float
    moist_depth_m: float
    #: Strength [K] of a temperature inversion at the mixed-layer top, which is
    #: what ECHAM's stratocumulus enhancement looks for.
    inversion_k: float
    wind: float
    mixed_layer_top_m: float
    orography_m: float
    latitude_deg: float
    #: Peak cloud liquid / ice mixing ratio [kg/kg] of the seeded deck.
    cloud_liquid: float
    cloud_ice: float


_POINTS = {
    # Subtropical marine stratocumulus over a flat ocean: a moist, capped
    # boundary layer under dry free troposphere, statically stable, no
    # convection, and no sub-grid orography — so the Lott-Miller gate never
    # opens and every SSO denominator is zero, which is the aquaplanet
    # degeneracy issue #558 exists for.
    "stable": _OperatingPoint(
        sst=292.0, rh_boundary_layer=0.97, rh_free_troposphere=0.3,
        moist_depth_m=1500.0, inversion_k=5.0, wind=6.0,
        mixed_layer_top_m=1000.0, orography_m=0.0, latitude_deg=25.0,
        cloud_liquid=2.0e-4, cloud_ice=0.0),
    # Moist tropical over a 400 m hill: conditionally unstable through a deep
    # moist layer, with a well-mixed sub-cloud layer (ECHAM's ``cubase`` lifts
    # a dry parcel from the lowest level, so an unmixed sounding cannot trigger
    # Tiedtke at any physical ``zlift``), a deep condensate load, and enough
    # orography to activate Lott-Miller.
    "convecting": _OperatingPoint(
        sst=302.0, rh_boundary_layer=0.95, rh_free_troposphere=0.7,
        moist_depth_m=9000.0, inversion_k=0.0, wind=8.0,
        mixed_layer_top_m=900.0, orography_m=400.0, latitude_deg=5.0,
        cloud_liquid=4.0e-4, cloud_ice=4.0e-5),
}

# Inputs held fixed, because they are absent at both operating points rather
# than small there, and every one of them is a non-negative fraction: a land
# fraction and a sea-ice fraction on an open tropical ocean, and the snow cover
# that goes with them. A leaf pinned at the boundary of its own domain has no
# two-sided derivative — displacing it downwards asks the surface scheme for
# its response to a negative tile fraction — so the whole direction along it is
# wasted, and with the tile selectors it crosses it reports a jump rather than
# a gradient. This is about the domain, never about convergence: no leaf is
# here because the check reads better without it.
_FIXED_INPUTS = ("fmask", "sice_am", "snowc_am")

# The purely geometric part of what ``MoistAirColumnState`` publishes: closed
# over rather than perturbed (see the module docstring). Spelled out rather
# than taken as ``MOIST_AIR_FIELDS`` so that adding a field to that tuple
# cannot silently remove it from the checked direction.
_FIXED_GEOMETRY = (
    "pressure_full", "pressure_half", "height_full", "height_half",
    "air_density", "pressure_thickness", "layer_thickness", "surface_pressure",
)

# The composition this module sweeps, in order. Asserted against the factory in
# ``test_every_term_is_covered`` so a term added to ``echam_physics`` cannot
# slip through unchecked.
_TERM_NAMES = (
    "moist_air_column_state",
    "echam_boundary_conditions",
    "macv2_sp_aerosol",
    "simple_chemistry",
    "sundqvist_cloud_fraction",
    "grey_two_stream_radiation",
    "tte_tke_vertical_diffusion",
    "echam_surface",
    "tiedtke_convection",
    "echam_1m_microphysics",
    "echam_surface_exchange",
    "hines_gwd",
    "lott_miller_sso",
)

_CONDENSATE_CLIP_KINK = (
    "jcm/physics/radiation/grey_two_stream/radiation_scheme.py:266-267 — "
    "cloud_water = jnp.maximum(cloud_water, 0.0), and the same for cloud_ice, "
    "applied to the condensate this scheme reads straight from the state "
    "tracers (cloud_data.py:311, radiation_cloud_fields, ECHAM's cover-then-"
    "radiation order). The seeded deck leaves that tracer at exactly 0 in most "
    "layers, so the clip sits ON the operating point: the plus and minus "
    "displacements are clipped in disjoint sets of layers, the two one-sided "
    "secants therefore measure different physics and stay O(1) apart at every "
    "rung, and the cloud optics' fractional dependence on the condensate path "
    "makes the central secant grow rather than converge as the step shrinks. "
    "Bisecting the direction one input leaf at a time puts the whole gap on "
    "one leaf: at the stable point state/tracers/qi alone gives D-=-1549, "
    "D+=0 against the full direction's -1606/-101, and at the convecting one "
    "state/tracers/qc alone gives 2070/-9.1 against 1243/-584. Freezing both "
    "drops the remaining gap an order of magnitude without restoring a "
    "reference — a one_sided of 0.3 to 0.5 survives on the cloud-fraction and "
    "aerosol leaves, and below eps ~ 1e-5 the projection is float32 noise. "
    "Not a lost gradient: every derivative here is finite in both modes, and "
    "the clip is the physics (negative condensate must not radiate). (#843)"
)

# The environment a column scheme actually reads. ECHAM hands ``cucall`` and
# ``cloud`` the provisional ``ztp1 = ptm1 + ptte*dt``, and the port follows: the
# schemes take T and q from the running ``thermo_run`` view, and touch
# ``state.temperature`` only for its shape. Naming ``state/temperature`` live
# would therefore assert something false about the port rather than about the
# scheme.
_ENVIRONMENT = ("['thermo_run']/['temperature']",
                "['thermo_run']/['specific_humidity']")

_PBL_HEIGHT_DEFECT = (
    "jcm/physics/vertical_diffusion/tte_tke/turbulence_coefficients.py:392 — "
    "compute_boundary_layer_height picks the PBL top with jnp.argmax over the "
    "exchange-coefficient threshold and indexes height_full with it, so "
    "vertical_diffusion/pbl_height is a staircase whose derivative is "
    "identically zero. It is not only a diagnostic: it sets the mixing length "
    "at turbulence_coefficients.py:108-111, so the diffusion's dependence on "
    "the PBL depth is invisible to a gradient. (#843)"
)

_ONE_MOMENT_SATURATION_CANCELLATION = (
    "no float32 adjoint identity holds on the convecting column. The "
    "saturation adjustment runs at 95 % relative humidity, where `q - qs` and "
    "the pass-2 `q_p1 - qs_p1 - 0.01*qs_p1` are each a difference of two "
    "nearly equal numbers, and the Rotstayn rain evaporation below the deck "
    "compounds it: the per-level derivatives in the lowest ten layers land "
    "50 % from their float64 values and the projection's summands reach 2.9e3 "
    "against a total of 5.2e2. jvp and vjp are that same double sum "
    "contracted in opposite orders, so they split — 4.5 % in the checked "
    "direction and 0.19 % to 37 % over seeds 0-5, against 1.3e-6 with the "
    "inputs promoted to float64. It is reduction order and not an asymmetry: "
    "there is no `custom_jvp`, `custom_vjp` or `stop_gradient` in the scheme "
    "for one to come from, and the gap does not move when the phase-partition "
    "floor this column's condensate tail sits on is swept from 1e-18 to 1e-9. "
    "Recorded rather than absorbed into `adjoint_rtol`: with no difference "
    "reference on this cell the adjoint identity is the only quantitative "
    "check left, and a tolerance loose enough to pass it — the 1.2e-1 this "
    "cell used to carry — could not detect a 10 % gradient error, nor did it "
    "hold at seeds 4 and 5. The stable column of the same term sits at 3.6e-4 "
    "and keeps a real tolerance. (#843)"
)

_CHEMISTRY_RELAXATION_KINK = (
    "jcm/physics/chemistry/simple_chemistry.py:313-314 — the term splits its net "
    "ozone relaxation rate into ozone_production = jnp.maximum(ozone_tendency, "
    "0.0) and ozone_loss = jnp.maximum(-ozone_tendency, 0.0), a corner at "
    "ozone_tendency == 0. On the replay's first step the term relaxes toward "
    "target = fixed_ozone_distribution(state) while current_ozone is the "
    "climatology EchamBoundaryConditions seeded from that SAME distribution, so "
    "ozone_tendency = (target - current)/tau is bit-identically 0 at every level "
    "and the operating point sits exactly on that corner (ozone_production == 0, "
    "ozone_loss rms 2.5e-13). Perturbing temperature (which moves target through "
    "the height profile) or current_ozone lifts the tendency off zero into "
    "production on one side and loss on the other, so the two one-sided secants "
    "measure the two arms of the split — D- ~ +5e7, D+ ~ -4e7 — and stay O(1) "
    "apart at every rung while the central secant sits stably at their mean; no "
    "central difference exists. Both operating points sit on it for the same "
    "reason. Not a lost gradient: both AD modes are finite (the finiteness case "
    "passes) and the production/loss split is the physics. (#843)"
)


@dataclasses.dataclass(frozen=True)
class _Check:
    """How one (term, operating point) cell is checked.

    ``reference="adjoint"`` is for a cell whose central difference reports the
    term's activation boundary rather than its derivative — the projection
    scales as ``jump/eps`` or the two one-sided secants stay O(1) apart however
    small the step. What remains there is the adjoint identity and a per-leaf
    assertion that the named inputs are live, which is what ``live_inputs`` is
    for.
    """

    reference: str = "difference"
    rtol: float = 2.0e-2
    adjoint_rtol: float = 1.0e-4
    live_inputs: tuple[str, ...] = ()
    #: ``"all"`` checks the tendency ledger and the ``provides`` diagnostics;
    #: ``"tendency"`` checks only the ledger. The narrow form is for a cell
    #: under the ``"adjoint"`` reference, which demands every checked output be
    #: live: a diagnostics struct always has fields that are legitimately zero
    #: at one operating point (no CAPE diagnostic is filled, no ice detrains
    #: from a warm plume), and listing them one by one would be a running
    #: inventory of the struct rather than a check.
    outputs: str = "all"
    #: Output keys dropped even from the ledger, each because the term returns
    #: that field as a structural zero rather than computing it.
    skip_outputs: tuple[str, ...] = ()
    xfail_reference: str | None = None
    xfail_finiteness: str | None = None


# Cells that are not the default. Keyed by (term, operating point); a term name
# alone applies to both points.
_CHECKS: dict = {
    # Finiteness holds at both points; only the two-sided reference is missing,
    # and for a reason that is the operating point rather than the scheme.
    "grey_two_stream_radiation": _Check(xfail_reference=_CONDENSATE_CLIP_KINK),

    # The chemistry relaxes ozone toward a target it also seeds the current
    # field from, so on the replay's first step the net rate is zero and the
    # production/loss split sits exactly on its corner at both points; finiteness
    # holds, the two-sided reference does not. See ``_CHEMISTRY_RELAXATION_KINK``.
    "simple_chemistry": _Check(xfail_reference=_CHEMISTRY_RELAXATION_KINK),

    # ``sundqvist_cloud_fraction`` at both points takes the default difference
    # reference. The stable column carried a strict xfail until #677: _qs_cover's
    # ice-memory phase switch (es_ice vs es_water at t_ice = 238.15 K, a
    # deliberate hard discontinuity — kept, see ``sundqvist.py`` _qs_cover) puts
    # a jump in cloud_fraction where a level crosses that boundary, and along
    # this test's fixed direction (seed 0) it defeated the central difference:
    # the secant grew as jump/eps until the step no longer crossed t_ice, by
    # which point it was float32 cancellation noise, so no rung was both past the
    # jump and above the noise. #677 did not touch the switch, but its
    # single-level tie-break and surface-interface height changed the
    # stable-column cloud_fraction field enough that the seed-0 projection now
    # resolves a converged reference in that gap (below the crossing step, above
    # the noise) which AD matches. The reference is genuine but delicate: the
    # switch is unchanged, so along other directions the jump can still defeat a
    # difference (verified: the cell passes at some seeds and not others on both
    # dev and this branch). If a future change re-crosses the boundary at seed 0
    # this fails loudly rather than silently — the honest signal for a delicate
    # reference — at which point it earns back an xfail naming the switch.

    # TTE-TKE, the 1M microphysics and Hines each cross an internal activation
    # boundary under this direction, and none of them has a central difference
    # because of it: the secant doubles as the step halves — ``jump/eps``, a
    # jump of order one RMS-unit inside the step — and the two one-sided
    # secants stay O(1) apart down to the last rung. Repeating it in float64
    # leaves the jump exactly where it was, so it is the scheme's own switch
    # and not float32 noise. The adjoint identity and per-leaf input liveness
    # are what remain meaningful, and both hold.
    #
    # ``adjoint_rtol`` is relaxed because each projection is a long float32
    # reduction with heavy cancellation: TTE-TKE's tridiagonal ``lax.scan``,
    # the 1M scheme's top-to-bottom flux ``lax.scan``, and the Hines
    # ``lax.scan`` over the wave spectrum. Each relaxation is sized to the
    # spread actually measured over seeds 0-5 rather than to what passes —
    # at most 3.0e-5 for TTE-TKE, 1.2e-5 for Hines and 3.6e-4 for the 1M
    # scheme's stable column — so the tolerances below keep 30x to 80x of
    # headroom and still detect a tenth-of-a-percent asymmetry. In float64
    # the same jvp and vjp projections agree to 1.6e-13 (TTE-TKE) and 1.9e-15
    # (Hines), so the float32 spread is reduction order, not a jvp/vjp
    # asymmetry — and there is no ``custom_jvp``, ``custom_vjp`` or
    # ``stop_gradient`` anywhere in the three schemes for one to come from.
    # The 1M scheme's two operating points part company here: the convecting
    # one reaches 37 % and is recorded as a defect rather than absorbed.
    "tte_tke_vertical_diffusion": _Check(
        reference="adjoint", adjoint_rtol=1.0e-3,
        live_inputs=("[0]/u_wind",), xfail_reference=_PBL_HEIGHT_DEFECT),
    # The microphysics moves water and the latent heat that goes with it; it
    # returns the momentum tendencies as structural zeros, and publishes
    # ``wbf`` as a zero on purpose (the 1M scheme has no explicit
    # Wegener-Bergeron-Findeisen transfer, but the key stays so the AeroCom
    # diagnostic set is scheme-independent — echam_1m.py:1418). There is
    # nothing for the liveness guard to find in any of the three.
    "echam_1m_microphysics": _Check(
        reference="adjoint", adjoint_rtol=2.0e-3, live_inputs=_ENVIRONMENT,
        skip_outputs=("u_wind", "v_wind", "wbf")),
    # The convecting column keeps the default adjoint tolerance and fails it;
    # see ``_ONE_MOMENT_SATURATION_CANCELLATION`` for why that is recorded
    # rather than absorbed. Everything else about the cell is the term entry
    # above.
    ("echam_1m_microphysics", "convecting"): _Check(
        reference="adjoint", live_inputs=_ENVIRONMENT,
        skip_outputs=("u_wind", "v_wind", "wbf"),
        xfail_reference=_ONE_MOMENT_SATURATION_CANCELLATION),
    # Hines returns a structurally zero moisture tendency — it moves momentum
    # and returns the dissipated energy as heat, nothing else — so there is
    # nothing for the liveness guard to find in that field.
    "hines_gwd": _Check(
        reference="adjoint", adjoint_rtol=1.0e-3, outputs="tendency",
        skip_outputs=("specific_humidity",),
        live_inputs=("[0]/u_wind", "[0]/temperature")),

    # On the stable column Tiedtke runs a one-layer shallow plume at the top
    # of the capped boundary layer (7e-9 kg/m²/s of precipitation) that sits
    # on its own existence boundary: along the seed-0 direction the plume
    # dies between +1e-3 and +1e-2 of the tangent and the one-sided secants
    # disagree between −1e-3 and 0 (a kink — the half-level environment's
    # dry-static-energy envelope is a max over the near-equal energies of the
    # well-mixed layer), so float32 finds no usable difference rung. In
    # float64 the central difference is usable and agrees with AD, and jvp
    # and vjp agree to 3.9e-13; float32 leaves them 4.4e-4 apart on this
    # small projection (|value| ≈ 89), the same reduction-order gap as the
    # convecting cell's. So the adjoint is the reference here too, at 1e-3.
    ("tiedtke_convection", "stable"): _Check(
        reference="adjoint", adjoint_rtol=1.0e-3, outputs="tendency",
        skip_outputs=("tracers/qi",), live_inputs=_ENVIRONMENT),

    # On the convecting column the column sits on the trigger: the minus secant is ~4e5 and the plus
    # secant ~0 at every rung, i.e. the perturbation switches the plume off.
    # That boundary is the documented discrete part of the scheme — ECHAM's
    # ``cubasmc`` mid-level conditions ARE the activation, and the deep/shallow
    # split is ECHAM's ``zdqcv`` switch (docs/source/science/convection.md) —
    # so no central difference exists there and the adjoint reference is what
    # the check can honestly assert. The plume on this column stays warm, so
    # it detrains no ice and the qi tendency is a structural zero rather than
    # a lost gradient.
    #
    # ``adjoint_rtol`` is relaxed to 3.0e-3 because the faithful #676/#669
    # reformulation lengthened the convecting-plume float32 reduction: the
    # cududv momentum tendency now sums SEPARATE updraft and downdraft
    # deviation-flux divergences, each built from a prognostic plume wind
    # mixed through the ascent/descent scans, plus the sub-cloud taper and the
    # surface-layer closure; and the organized entrainment/detrainment add the
    # ``zdrodz`` log-density term and the metre-based ``tan`` profile with
    # their ``centrmax`` clips. In float64 the seed-0 jvp and vjp agree to
    # 8.1e-13 (≤4e-14 over seeds 1-2), so the float32 gap is reduction order,
    # not a jvp/vjp asymmetry — and there is no ``custom_jvp``, ``custom_vjp``
    # or ``stop_gradient`` in the scheme for one to come from; the two float32
    # modes straddle the float64 truth (-2527.3: jvp -2530.1, vjp -2528.9).
    # The worst float32 spread over seeds 0-5 is 4.85e-4, all of it on seed 0
    # (the smallest-magnitude projection, |Δ|≈1.2 on a value of 2530; the
    # other five seeds are ≤2.6e-5). 3.0e-3 keeps ~6x headroom and still
    # detects a 0.3 % asymmetry — far tighter than the 1.2e-1 the 1M
    # convecting cell had to reject as unusable.
    ("tiedtke_convection", "convecting"): _Check(
        reference="adjoint", adjoint_rtol=3.0e-3, outputs="tendency",
        skip_outputs=("tracers/qi",), live_inputs=_ENVIRONMENT),
}


def _check_for(term_name: str, point: str) -> _Check:
    """Return the cell's configuration, most specific key first."""
    return _CHECKS.get((term_name, point), _CHECKS.get(term_name, _Check()))


def _steady_sun(day_fraction=0.22, time_of_day=0.5) -> SolarGeometry:
    """Return a fixed sun, so the column has a well-defined zenith angle."""
    two_pi = 2.0 * np.pi
    return SolarGeometry(
        tyear=jnp.float32(day_fraction),
        orbital_phase=jnp.float32(two_pi * day_fraction),
        synodic_phase=jnp.float32(two_pi * time_of_day),
    )


def _column_state(vertical, tracer_names, point: _OperatingPoint):
    """Build the ``(nlev, 1)`` sounding of ``point``.

    Dry-adiabatic below ``mixed_layer_top_m`` and at 6.5 K/km above it, capped
    at a 200 K stratosphere. The mixed layer is physics, not cosmetics: a
    column that runs at the free-tropospheric lapse rate all the way to the
    surface loses ~3.3 K/km of parcel buoyancy and cannot trigger Tiedtke at
    any physical ``zlift``, which is why the RCE helpers build the same shape.

    The vertical axis is whatever ``vertical`` gives — ECHAM's hybrid
    coefficients are top-first — and nothing here assumes which end is which:
    every profile is written as a function of ``height``.
    """
    ps = c.p0
    a_half = jnp.asarray(vertical.a_boundaries)
    b_half = jnp.asarray(vertical.b_boundaries)
    p_half = a_half + b_half * ps
    p_full = 0.5 * (p_half[:-1] + p_half[1:])

    height = -7.6e3 * jnp.log(p_full / ps)
    dry_lapse = c.grav / c.cpd
    temperature = jnp.maximum(
        jnp.where(
            height <= point.mixed_layer_top_m,
            point.sst - dry_lapse * height,
            point.sst - dry_lapse * point.mixed_layer_top_m
            - 6.5e-3 * (height - point.mixed_layer_top_m)),
        200.0)
    # A capping inversion at the mixed-layer top, which is the structure
    # ECHAM's stratocumulus enhancement searches for (a level warmer than the
    # one below it inside its inversion-height window).
    temperature = temperature + point.inversion_k * jnp.exp(
        -((height - point.mixed_layer_top_m) / 200.0) ** 2)
    qsat = jax.vmap(saturation_mixing_ratio)(p_full, temperature)
    relative_humidity = (
        point.rh_free_troposphere
        + (point.rh_boundary_layer - point.rh_free_troposphere)
        * jnp.exp(-(height / point.moist_depth_m) ** 2))
    humidity = relative_humidity * qsat

    # A condensate deck, so ``qc``/``qi`` are inputs with a magnitude rather
    # than leaves pinned at zero. A zero leaf has no RMS for the step to be
    # relative to, so ``check_gradients`` falls back to an absolute step — and
    # 1e-3 kg/kg of cloud water is ten times any real value, which drives the
    # microphysics and the cloud optics somewhere the model never goes.
    # Liquid fills the warm part of the column and ice the cold part, split at
    # the freezing level, which is also what makes both phases' pathways live.
    warm = temperature > c.tmelt
    shape = jnp.exp(-((height - 3.0e3) / 4.0e3) ** 2)
    liquid = jnp.where(warm, point.cloud_liquid * shape, 0.0)
    ice = jnp.where(warm, 0.0, point.cloud_ice * shape)

    # A sheared jet rather than a uniform wind. Hines launches its spectrum at
    # the bottom and deposits momentum where the Doppler shift breaks it, so a
    # column with no shear never breaks a wave and returns a drag that does not
    # depend on the wind at all — which reads as a lost gradient and is only
    # an inactive scheme. This profile peaks near 11 km, the height an
    # upper-tropospheric jet does.
    jet = 0.3 + 3.0 * jnp.exp(-((height - 11.0e3) / 7.0e3) ** 2)

    column = lambda x: jnp.asarray(x).reshape(_NLEV, 1)
    seeded = {"qc": column(liquid), "qi": column(ice)}
    return PhysicsState(
        u_wind=column(point.wind * jet),
        v_wind=column(0.5 * point.wind * jet),
        temperature=column(temperature),
        specific_humidity=column(humidity),
        geopotential=column(c.grav * height),
        normalized_surface_pressure=jnp.full((1,), ps / c.p0),
        tracers={name: seeded.get(name, jnp.zeros((_NLEV, 1)))
                 for name in tracer_names},
    )


@dataclasses.dataclass(frozen=True)
class _Replay:
    """One column, plus each term's input diagnostics at its turn to run."""

    physics: object
    state: PhysicsState
    forcing: ForcingData
    terrain: TerrainData
    snapshots: dict


_REPLAY_CACHE: dict[str, _Replay] = {}


def _replay(point_name: str) -> _Replay:
    """Run the package once, snapshotting the diagnostics each term is given.

    Reproduces ``ComposablePhysics._compute_tendencies_columns`` on a one-column
    grid — the same ``_dt_seconds`` / ``_band_config`` injection and the same
    running ``_tendency_run`` accumulator — so each term sees the diagnostics it
    would see on the model's first step. There is no dycore, so
    ``_dycore_fields`` is absent and Tiedtke's ``cubasmc`` mid-level trigger
    reads its documented zero-omega fallback (no resolved ascent, trigger
    dormant).
    """
    if point_name in _REPLAY_CACHE:
        return _REPLAY_CACHE[point_name]

    point = _POINTS[point_name]
    vertical = get_echam_levels(_NLEV)
    coords = ColumnCoordinates.at_location(vertical, point.latitude_deg, 0.0)
    physics = echam_physics(radiation_scheme="grey", checkpoint_terms=False)
    physics.cache_coords(coords)

    state = _column_state(
        vertical, [spec.name for spec in physics.required_tracers()], point)
    terrain = TerrainData.single_column(orog=point.orography_m)
    # Ocean-like surface boundary values rather than the zeros the factory
    # defaults to: a zero albedo or a zero soil wetness is a leaf pinned at the
    # edge of its range, which would have to be frozen like the tile fractions
    # above instead of carrying a real derivative.
    forcing = ForcingData.zeros((1, 1)).copy(
        sea_surface_temperature=jnp.full((1, 1), point.sst + 1.0),
        stl_am=jnp.full((1, 1), point.sst),
        alb0=jnp.full((1, 1), 0.07),
        soilw_am=jnp.full((1, 1), 0.3),
        solar=_steady_sun(),
    )

    diagnostics = dict(physics.initial_carry_state(coords))
    diagnostics["_dt_seconds"] = physics.dt_seconds
    diagnostics["_band_config"] = physics.band_config
    running = {
        "u_wind": jnp.zeros((_NLEV, 1)), "v_wind": jnp.zeros((_NLEV, 1)),
        "temperature": jnp.zeros((_NLEV, 1)),
        "specific_humidity": jnp.zeros((_NLEV, 1)),
        "tracers": {name: jnp.zeros((_NLEV, 1)) for name in state.tracers},
    }
    snapshots = {}
    for term in physics.terms:
        diagnostics["_tendency_run"] = running
        snapshots[term.name] = dict(diagnostics)
        tendency, diagnostics = term(state, diagnostics, forcing, terrain)
        running = {
            "u_wind": running["u_wind"] + tendency.u_wind,
            "v_wind": running["v_wind"] + tendency.v_wind,
            "temperature": running["temperature"] + tendency.temperature,
            "specific_humidity": (running["specific_humidity"]
                                  + tendency.specific_humidity),
            "tracers": {
                name: running["tracers"][name] + tendency.tracers.get(name, 0.0)
                for name in running["tracers"]},
        }

    _REPLAY_CACHE[point_name] = _Replay(
        physics=physics, state=state, forcing=forcing, terrain=terrain,
        snapshots=snapshots)
    return _REPLAY_CACHE[point_name]


def _term_function(replay: _Replay, term_name: str,
                   outputs: str = "all", skip_outputs: tuple[str, ...] = ()):
    """``(state, diagnostics, forcing, terrain) -> (tendency, provided)``.

    ``diagnostics`` here is only the part that is an independent input: the
    plumbing keys and the state-derived column geometry are closed over (see
    the module docstring).
    """
    term = next(t for t in replay.physics.terms if t.name == term_name)
    snapshot = replay.snapshots[term_name]
    fixed = {k: v for k, v in snapshot.items()
             if k.startswith("_") or k in _FIXED_GEOMETRY}
    free = {k: v for k, v in snapshot.items()
            if not k.startswith("_") and k not in _FIXED_GEOMETRY}
    provides = tuple(term.provides)

    def call(state, free_diagnostics, forcing, terrain):
        tendency, updated = term(
            state, {**fixed, **free_diagnostics}, forcing, terrain)
        out = {
            "u_wind": tendency.u_wind,
            "v_wind": tendency.v_wind,
            "temperature": tendency.temperature,
            "specific_humidity": tendency.specific_humidity,
            **{f"tracers/{name}": value
               for name, value in tendency.tracers.items()},
        }
        if outputs == "all":
            out.update({k: updated[k] for k in provides if k in updated})
        return {k: v for k, v in out.items() if k not in skip_outputs}

    return call, (replay.state, free, replay.forcing, replay.terrain)


def _assert_derivatives_are_finite(f, args, label):
    """One jvp and one vjp through ``f``, both finite everywhere.

    The direction and the cotangent are the same per-leaf-scaled ones
    ``check_gradients`` uses, so a leaf whose magnitude would otherwise leave
    it unperturbed still moves.
    """
    primal, vjp_fun = jax.vjp(f, *args)
    _, forward = jax.jvp(f, args, _tangent(args, 0))
    reverse = vjp_fun(_cotangent(primal, 1))
    # An integer leaf carries JAX's empty ``float0`` derivative in both modes,
    # which ``np.isfinite`` cannot even be asked about; there is nothing to
    # check on one.
    is_float = lambda x: jnp.issubdtype(jnp.result_type(x), jnp.floating)
    for mode, tree in (("forward", forward), ("reverse", reverse)):
        for leaf in jax.tree.leaves(tree):
            if is_float(leaf):
                assert np.all(np.isfinite(np.asarray(leaf))), (
                    f"{label}: a {mode}-mode derivative is not finite")


def _cases(attribute):
    """One case per (term, operating point), xfailed where ``attribute`` says.

    A cell that names a defect gets a STRICT xfail, so the day the scheme it
    names is fixed the case turns from xfail into an unexpected pass and says
    so, instead of quietly staying disabled. The two tests take their xfails
    from different fields, because a term can lose its reference while its
    derivatives stay perfectly finite.

    ``raises=AssertionError`` narrows it to the failure the reason describes.
    ``check_gradients`` raises ``ValueError`` for a ``live_inputs`` or
    ``fixed_inputs`` name that matches no leaf, and several of these cells
    carry both a name list and an xfail; without the constraint, renaming a
    state field would convert that loud error into a green xfail and the
    strictness that is meant to fire on a fix never would.
    """
    cases = []
    for term_name in _TERM_NAMES:
        for point_name in sorted(_POINTS):
            reason = getattr(_check_for(term_name, point_name), attribute)
            cases.append(pytest.param(
                term_name, point_name, id=f"{term_name}-{point_name}",
                marks=([pytest.mark.xfail(strict=True, reason=reason,
                                          raises=AssertionError)]
                       if reason else [])))
    return cases


def test_every_term_is_covered():
    """``_TERM_NAMES`` is the composition, so a new term cannot slip through."""
    physics = echam_physics(radiation_scheme="grey", checkpoint_terms=False)
    assert tuple(term.name for term in physics.terms) == _TERM_NAMES


@pytest.mark.parametrize("term_name,point_name", _cases("xfail_finiteness"))
def test_term_derivatives_are_finite(term_name, point_name):
    """No term turns a finite forward pass into a non-finite derivative."""
    f, args = _term_function(_replay(point_name), term_name)
    _assert_derivatives_are_finite(f, args, f"{term_name}/{point_name}")


@pytest.mark.parametrize("term_name,point_name", _cases("xfail_reference"))
def test_term_gradients_against_a_reference(term_name, point_name):
    """Each term's AD agrees with a central difference, or with its adjoint.

    Which of the two, and with what tolerance, is ``_CHECKS`` — every entry
    there carries the evidence for choosing it.
    """
    check = _check_for(term_name, point_name)
    f, args = _term_function(
        _replay(point_name), term_name,
        outputs=check.outputs, skip_outputs=check.skip_outputs)
    check_gradients(
        f, args,
        rtol=check.rtol if check.reference == "difference" else None,
        # The projection is dimensionless — each output leaf's cotangent is
        # divided by its RMS and each input leaf's tangent multiplied by its
        # own — so this absolute floor means the same thing for every term:
        # a response below a hundred-millionth of a fractional displacement is
        # no response at all.
        atol=1e-8,
        reference=check.reference,
        adjoint_rtol=check.adjoint_rtol,
        live_inputs=check.live_inputs,
        fixed_inputs=_FIXED_INPUTS,
    )


@pytest.mark.parametrize("point_name", sorted(_POINTS))
def test_package_tendency_is_finite_per_state_field(point_name):
    """The composed package's tendency, differentiated one state field at a time.

    The per-term checks above cannot see a poison that only exists once terms
    feed each other, and a projection over the whole state cannot say *which*
    field carries one. This runs ``compute_tendencies`` on the same single
    column and takes one jvp per ``PhysicsState`` leaf, so a non-finite
    derivative is reported against the field that produced it.

    It is the cheap counterpart of ``gradient_finiteness_test``'s two-step
    rollout, and a strictly wider direction than it: that test differentiates
    only ``solar_constant``, a single scalar whose path through the package on
    the first step touches almost none of it, while this one walks every
    prognostic field. Both the defects that used to surface here — the Planck
    denominator's float32 overflow and the 1M phase partition's ``0 * inf`` —
    were found through this direction and are fixed at their source.
    """
    replay = _replay(point_name)
    grid = lambda x: x.reshape(x.shape[0], 1, 1) if x.ndim == 2 else x.reshape(1, 1)
    state = PhysicsState(
        u_wind=grid(replay.state.u_wind),
        v_wind=grid(replay.state.v_wind),
        temperature=grid(replay.state.temperature),
        specific_humidity=grid(replay.state.specific_humidity),
        geopotential=grid(replay.state.geopotential),
        normalized_surface_pressure=grid(
            replay.state.normalized_surface_pressure),
        tracers={name: grid(value)
                 for name, value in replay.state.tracers.items()},
    )
    carry = replay.physics.initial_carry_state(
        ColumnCoordinates.at_location(
            get_echam_levels(_NLEV), _POINTS[point_name].latitude_deg, 0.0))

    def tendencies(state_):
        tendency, _ = replay.physics.compute_tendencies(
            state_, replay.forcing, replay.terrain, carry)
        return tendency

    # One compiled jvp, reused for every direction (all have the state's
    # shapes). Run eagerly, each direction dispatched the package op by op,
    # and the per-primitive executables alone took an xdist worker past the
    # kernel's memory-map limit (65,499 of vm.max_map_count's 65,530), where
    # the CPU JIT fails with "Failed to materialize symbols".
    jvp_tendencies = jax.jit(
        lambda state_, direction_: jax.jvp(
            tendencies, (state_,), (direction_,))[1])

    leaves, treedef = jax.tree_util.tree_flatten(state)
    names = _leaf_names(state)
    for index, name in enumerate(names):
        direction = jax.tree_util.tree_unflatten(treedef, [
            jnp.ones_like(leaf) if position == index else jnp.zeros_like(leaf)
            for position, leaf in enumerate(leaves)])
        forward = jvp_tendencies(state, direction)
        for leaf in jax.tree.leaves(forward):
            assert np.all(np.isfinite(np.asarray(leaf))), (
                f"{point_name}: the package tendency has a non-finite "
                f"derivative with respect to {name}")


@pytest.mark.parametrize("point_name", sorted(_POINTS))
def test_convection_parameter_gradients(point_name):
    """The convection tunables are differentiable, not closed-over constants.

    ``_term_function`` differentiates with respect to the term's *arguments*,
    which leaves the ``nnx.Param`` holding ``ConvectionParameters`` as a
    captured constant — and those are exactly the leaves a calibration run
    optimises. Splitting the params out and passing them in puts them back in
    the checked direction.
    """
    replay = _replay(point_name)
    term = next(t for t in replay.physics.terms
                if t.name == "tiedtke_convection")
    graphdef, params, rest = nnx.split(term, nnx.Param, ...)
    snapshot = replay.snapshots["tiedtke_convection"]
    provides = tuple(term.provides)

    def call(params_):
        rebuilt = nnx.merge(graphdef, params_, rest)
        tendency, updated = rebuilt(
            replay.state, dict(snapshot), replay.forcing, replay.terrain)
        return tendency, {k: updated[k] for k in provides if k in updated}

    _assert_derivatives_are_finite(
        call, (params,), f"tiedtke_convection params/{point_name}")
    # The scheme's smooth-trigger machinery is what makes these learnable at
    # all; the sigmoids saturate far from a threshold, so liveness is asserted
    # only on the column that is actually convecting.
    if point_name == "convecting":
        primal, vjp_fun = jax.vjp(call, params)
        gradients = vjp_fun(_cotangent(primal, 1))[0]
        assert any(
            np.any(np.asarray(leaf) != 0.0)
            for leaf in jax.tree.leaves(gradients)
            if jnp.issubdtype(jnp.result_type(leaf), jnp.floating)), (
            "no convection parameter carries a gradient on a convecting "
            "column — a trigger has been re-hardened")
