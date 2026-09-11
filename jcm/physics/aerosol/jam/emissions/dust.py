"""Tegen/HAMMOZ wind-erosion dust emission.

Port of the MPI-BGC dust scheme ``mo_ham_dust.f90`` (``bgc_dust_initialize``,
``bgc_dust_calc_emis``) as configured by HAM2 (``ndust = 4``): a size-resolved
Marticorena-Bergametti (1995) saltation flux over a 191-class soil size grid
and a per-cell mixture of prescribed soil textures, sandblasted into an emitted
spectrum and integrated onto MAM4's accumulation and coarse emission windows.

The chain, in CGS as the Fortran is (cm, cm/s, g, g cm⁻² s⁻¹):

* ``u* = vk·U10/ln(ZZ/z0)``                              MB95 (15)
* ``u*t(D)`` from the friction Reynolds number            MB95 (5)-(7)
* ``srel(D)``, ``srelV(D)`` from four lognormal soil populations per texture
                                                          MB95 (29)-(32)
* ``F(j,D) = srel·(1+R)²·(1−R)·cd·u*³·α_j``, ``R = u*t·s/u*``
                                                          MB95 (28)/(33), Tegen (2002) (3)
* sandblasting: each class ``k > 1`` redistributes over classes ``1…k`` in
  proportion to ``srelV``
* ``flux = Σ_bins · (1 − snow_cover) · pot_source``, zeroed where the soil is
  saturated.

Five prescribed fields drive it, all on the model grid (see
:mod:`jcm.forcing`): ``dust_source`` (monthly effective-LAI erodible fraction,
the gate *and* a linear factor), ``dust_preferential`` (paleolake area fraction,
a texture swap), ``dust_soil_types`` (nine texture area fractions),
``dust_regions`` (the 1-8 tuning index) and ``dust_roughness`` (read but
overwritten by the constant ``ndurough``, exactly as the Fortran does).

Documented in ``docs/source/science/aerosol.md``; the ``U10 = 10 m/s`` texture
switch is a hard step with zero gradient (#664).
"""

from __future__ import annotations

import math
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from flax import nnx, struct

from jcm.physics.aerosol.jam.emissions.distributors import distribute_surface_flux
from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
    accumulate_emission_fluxes, emission_flux_keys)
from jcm.physics.aerosol.jam.emissions.surface_wind import (
    MODEL_LEVEL_WIND_KEY, wind_10m)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

# --- structural constants (mo_ham_dust.f90 declaration block), CGS ----------
#: Soil size grid: 50 classes per decade from 0.2 µm; the Fortran's
#: ``DO WHILE (D <= Dmax + 1e-5)`` fills 191 of the 192 allocated classes.
DMIN = 2.0e-5          # cm
DMAX = 0.130           # cm
DSTEP = math.log(10.0) / 50.0
NCLASS = 191
ROA = 0.001227         # air density [g/cm³]
ROP = 2.65             # soil particle density [g/cm³]
VK = 0.4               # von Karman
ZZ = 1000.0            # reference height for the u* log law [cm]
GRAV_CGS = 9.80665 * 100.0
#: Flux dimensioning parameter ``cd = roa/(g·100)``, MB95 (33).
CD = ROA / GRAV_CGS
#: Above this 10 m wind the preferential source emits the clay (type 11)
#: spectrum instead of the silt (type 10) one — a hard step, see module docstring.
HIGH_WIND_MS = 10.0

#: MAM4 emission windows, dry diameter [µm]. HAM stops at an 8-bin M7 product;
#: these edges are MAM4's own convention and mass above ``SUPERCOARSE_UM`` is
#: discarded ("neglect the super-coarse mode", Stier et al. 2005 §2.3.4).
ACCUM_UM = (0.1, 1.0)
COARSE_UM = (1.0, 10.0)
SUPERCOARSE_UM = 10.0

#: Per-column diagnostic: the emitted mass flux discarded above
#: ``SUPERCOARSE_UM`` [kg/m²/s]. Not an ``emi_``/``emis_`` name — those prefixes
#: select emission fluxes in the forcing reader and the burden report, and this
#: mass deliberately never enters a tracer.
DUST_SUPERCOARSE_KEY = "dust_supercoarse_flux"

#: The 17×14 soil table of ``mo_ham_dust.f90::set_dust_data``: four
#: ``(D_med [cm], σ_g, mass fraction)`` populations, then α [cm⁻¹] (the
#: vertical:horizontal flux ratio) and the residual soil moisture. Rows 5, 7, 8,
#: 9 and 12 are never referenced by the flux.
SOIL_TABLE = np.array([
    [0.0707, 2.0, 0.43, 0.0158, 2.0, 0.40, 0.0015, 2.0, 0.17, 0.0002, 2.0, 0.00, 2.1e-06, 0.20],
    [0.0707, 2.0, 0.00, 0.0158, 2.0, 0.37, 0.0015, 2.0, 0.33, 0.0002, 2.0, 0.30, 4.0e-06, 0.25],
    [0.0707, 2.0, 0.00, 0.0158, 2.0, 0.00, 0.0015, 2.0, 0.33, 0.0002, 2.0, 0.67, 1.0e-07, 0.50],
    [0.0707, 2.0, 0.10, 0.0158, 2.0, 0.50, 0.0015, 2.0, 0.20, 0.0002, 2.0, 0.20, 2.7e-06, 0.23],
    [0.0707, 2.0, 0.00, 0.0158, 2.0, 0.50, 0.0015, 2.0, 0.12, 0.0002, 2.0, 0.38, 2.8e-06, 0.25],
    [0.0707, 2.0, 0.00, 0.0158, 2.0, 0.27, 0.0015, 2.0, 0.25, 0.0002, 2.0, 0.48, 1.0e-07, 0.36],
    [0.0707, 2.0, 0.23, 0.0158, 2.0, 0.23, 0.0015, 2.0, 0.19, 0.0002, 2.0, 0.35, 2.5e-06, 0.25],
    [0.0707, 2.0, 0.25, 0.0158, 2.0, 0.25, 0.0015, 2.0, 0.25, 0.0002, 2.0, 0.25, 0.0, 0.50],
    [0.0707, 2.0, 0.25, 0.0158, 2.0, 0.25, 0.0015, 2.0, 0.25, 0.0002, 2.0, 0.25, 0.0, 0.50],
    [0.0707, 2.0, 0.00, 0.0158, 2.0, 0.00, 0.0015, 2.0, 1.00, 0.0002, 2.0, 0.00, 1.0e-05, 0.25],
    [0.0707, 2.0, 0.00, 0.0158, 2.0, 0.00, 0.0015, 2.0, 0.00, 0.0002, 2.0, 1.00, 1.0e-05, 0.25],
    [0.0707, 2.0, 0.00, 0.0158, 2.0, 0.00, 0.0027, 2.0, 1.00, 0.0002, 2.0, 0.00, 1.0e-05, 0.25],
    [0.0442, 1.5, 0.03, 0.0084, 1.5, 0.85, 0.0015, 2.0, 0.11, 0.0002, 2.0, 0.02, 1.9e-06, 0.12],
    [0.0450, 1.5, 0.00, 0.0070, 1.5, 0.33, 0.0015, 2.0, 0.50, 0.0002, 2.0, 0.17, 1.9e-04, 0.15],
    [0.0457, 1.8, 0.31, 0.0086, 1.5, 0.22, 0.0015, 2.0, 0.34, 0.0002, 2.0, 0.12, 3.9e-05, 0.13],
    [0.0293, 1.8, 0.39, 0.0090, 1.5, 0.16, 0.0015, 2.0, 0.35, 0.0002, 2.0, 0.10, 3.1e-05, 0.13],
    [0.0305, 1.5, 0.46, 0.0101, 1.5, 0.41, 0.0015, 2.0, 0.10, 0.0002, 2.0, 0.03, 2.8e-06, 0.12],
])
ALPHA_COL = 12
WRES_COL = 13

#: Soil-type file variables, in the order :class:`DustEmissions` consumes them.
#: Type 1 (coarse) is NOT in the file — it is the residual ``1 − Σ`` (and, under
#: ``k_dust_easo = 2``, the East-Asian total).
SOIL_TYPE_VARS = ("type2", "type3", "type4", "type6",
                  "type13", "type14", "type15", "type16", "type17")
#: The nine file fields as 1-based soil-table rows.
SOIL_TYPE_INDEX = (2, 3, 4, 6, 13, 14, 15, 16, 17)
#: The East-Asian subset (Laurent et al. 2006 / Cheng et al. 2008 textures).
EAST_ASIA_INDEX = (13, 14, 15, 16, 17)

#: ``(flux texture, spectrum texture)`` rows of the per-cell mixture, 1-based.
#: The last two are the preferential source, which takes its flux magnitude from
#: type 10 (silt) always and its emitted spectrum from type 10 below
#: ``HIGH_WIND_MS`` and type 11 (clay) above it.
MIXTURE_ROWS = ((1, 1), (2, 2), (3, 3), (4, 4), (6, 6),
                (13, 13), (14, 14), (15, 15), (16, 16), (17, 17),
                (10, 10), (10, 11))
_N_EAST_ASIA_ROWS = len(EAST_ASIA_INDEX)

#: Number of regions in ``dust_regions.nc`` (1 = everywhere else, 2 = N America,
#: 3 = S America, 4 = N Africa, 5 = S Africa, 6 = Middle East, 7 = Asia,
#: 8 = Australia; Huneeus et al. 2011).
N_REGIONS = 8


def soil_diameters() -> np.ndarray:
    """Return the 191 soil-class diameters [cm], ``D_k = Dmin·exp((k−1)·Dstep)``."""
    return DMIN * np.exp(np.arange(NCLASS) * DSTEP)


def threshold_friction_velocity(diameters, a_rnolds, b_rnolds, x_rnolds,
                                d_thrsld, coeff):
    """MB95 (5)-(7) threshold friction velocity [cm/s] per soil class.

    A pure function of diameter — no soil type, no moisture, no cell dependence.
    The Reynolds branch switches at ``Re = 10``; the 0.129 prefactor is used in
    both branches.
    """
    re = a_rnolds * diameters ** x_rnolds + b_rnolds
    a = jnp.sqrt(ROP * GRAV_CGS * diameters / ROA)
    c = jnp.sqrt(1.0 + d_thrsld / diameters ** 2.5)
    low = 1.928 * re ** 0.092 - 1.0
    # Both branches are evaluated under `where`, so the low-Re sqrt must stay
    # finite (and its gradient defined) at large Re where the argument is >1 anyway.
    uth_low = coeff * a * c / jnp.sqrt(jnp.maximum(low, 1.0e-12))
    uth_high = coeff * a * c * (1.0 - 0.0858 * jnp.exp(-0.0617 * (re - 10.0)))
    return jnp.where(re < 10.0, uth_low, uth_high)


def soil_size_distributions(soil_table, diameters):
    """Relative surface ``srel`` and relative mass ``srelV`` per soil type.

    MB95 (29)-(32): each texture is a superposition of four lognormal
    populations; ``srel`` (basal surface, the flux weight) divides the mass
    density by ``xn = ρ_p·(2/3)·(D/2)``, ``srelV`` (mass, the sandblasting
    weight) does not. Returns ``(srel, srelV, su_srelV)``, each ``(17, 191)``
    with ``su_srelV`` the running cumulative ``srelV``.
    """
    d_med = soil_table[:, 0:12:3][:, :, None]      # (17, 4, 1)
    sigma = soil_table[:, 1:12:3][:, :, None]
    frac = soil_table[:, 2:12:3][:, :, None]
    ln_sig = jnp.log(sigma)
    xk = frac / (jnp.sqrt(2.0 * jnp.pi) * ln_sig)
    xl = (jnp.log(diameters)[None, None, :] - jnp.log(d_med)) ** 2 / (2.0 * ln_sig ** 2)
    xm = jnp.sum(xk * jnp.exp(-xl), axis=1)        # (17, 191) mass density
    xn = ROP * (2.0 / 3.0) * (diameters / 2.0)
    su = xm * DSTEP / xn
    su_v = xm * DSTEP
    srel = su / jnp.sum(su, axis=1, keepdims=True)
    srel_v = su_v / jnp.sum(su_v, axis=1, keepdims=True)
    return srel, srel_v, jnp.cumsum(srel_v, axis=1)


#: The five ``(window, moment)`` aggregates the emitted spectrum is reduced to:
#: mass and ``Σ F/D³`` (which fixes the number) for each MAM4 window, plus the
#: total mass so the discarded super-coarse remainder can be reported.
_AGGREGATES = (("acc", 0), ("acc", 1), ("cor", 0), ("cor", 1), ("all", 0))


def emission_weight_matrix(srel, srel_v, su_srel_v, diameters):
    """Reduce the sandblasting redistribution to one matrix per mixture row.

    The Fortran redistributes each saltating class ``k ≥ 2`` over classes
    ``1…k`` with weights ``srelV(k')/(su_srelV(k) − srelV(1))`` (the numerator
    includes ``k' = 1`` while the denominator excludes it, so the weights sum to
    slightly more than 1 — a faithful ~3e-5 non-conservation, not a typo). Since
    every aggregate is a fixed linear functional of the emitted spectrum, the
    double loop collapses exactly to

        ``Σ_{k'∈window} F(k')·D_{k'}^{-3q} = α · Σ_k W[c, k]·base(k)``

    with ``base(k) = (1+R)²(1−R)·cd·u*³`` the only column-dependent factor.
    That keeps the per-column working set at ``(191, ncols)`` instead of
    ``(ntype, 191, ncols)``; ``dust_test`` checks it against a direct transcription
    of the Fortran loops.

    Returns ``(nrow, 5, 191)`` for :data:`MIXTURE_ROWS` and :data:`_AGGREGATES`.
    """
    d_um = diameters * 1.0e4
    windows = {"acc": (d_um >= ACCUM_UM[0]) & (d_um < ACCUM_UM[1]),
               "cor": (d_um >= COARSE_UM[0]) & (d_um < COARSE_UM[1]),
               "all": jnp.ones_like(d_um, dtype=bool)}
    rows = []
    for flux_type, size_type in MIXTURE_ROWS:
        jf, js = flux_type - 1, size_type - 1
        # Weights of the classes k >= 2 redistributing into the window.
        denom = su_srel_v[js, 1:] - srel_v[js, 0]
        combos = []
        for name, q in _AGGREGATES:
            moment = jnp.where(windows[name], diameters ** (-3 * q), 0.0)
            cumulative = jnp.cumsum(srel_v[js] * moment)
            head = srel[jf, 0] * moment[0]                     # class 1 emits at its own size
            tail = srel[jf, 1:] / denom * cumulative[1:]
            combos.append(jnp.concatenate([head[None], tail]))
        rows.append(jnp.stack(combos))
    return jnp.stack(rows)


@struct.dataclass
class DustParameters:
    """Tegen/HAMMOZ dust knobs — numeric leaves, code-path switches static.

    Defaults are the ``ndust = 4`` preset (Stier et al. 2005 + Cheng's
    East-Asian soils = the HAM2 configuration of Zhang et al. 2012 §4.1.5) at
    T63 free-running. Build another preset with :meth:`preset`.
    """

    soil_table: jnp.ndarray        # (17, 14) populations, α [cm⁻¹], residual moisture
    nduscale_reg: jnp.ndarray      # (8,) per-region multiplier on the THRESHOLD
    threshold_scale: jnp.ndarray   # (5,) utsc for soil types 13-17
    r_dust_umin: jnp.ndarray       # u* pre-gate [cm/s]
    r_dust_lai: jnp.ndarray        # pot_source gate
    r_dust_z0s: jnp.ndarray        # smooth-roughness reference [cm]
    ndurough: jnp.ndarray          # constant roughness [cm] when the map is off
    r_dust_scz0: jnp.ndarray       # roughness scale factor
    r_dust_z0min: jnp.ndarray      # roughness floor [cm]
    w0: jnp.ndarray                # relative soil wetness above which flux is zero
    aeff: jnp.ndarray              # MB95 (17) drag-partition constants
    xeff: jnp.ndarray
    a_rnolds: jnp.ndarray          # MB95 (5) friction-Reynolds fit
    b_rnolds: jnp.ndarray
    x_rnolds: jnp.ndarray
    d_thrsld: jnp.ndarray          # MB95 cohesion term
    uth_coeff: jnp.ndarray         # MB95 (6)/(7) prefactor

    #: ``k_dust_smst = 0``: add the Fécan et al. (1999) soil-moisture correction
    #: to the threshold. Off in every preset but ``ndust = 2``.
    fecan_moisture: bool = struct.field(pytree_node=False, default=False)
    #: ``k_dust_easo``: 1 zeroes the East-Asian textures, 2 replaces the cell
    #: with them, 0 is the Cheng implementation whose residual can go negative.
    east_asia: int = struct.field(pytree_node=False, default=2)
    #: ``ndurough = 0``: use the monthly satellite roughness map instead of the
    #: constant. The constant itself (``ndurough``) stays a differentiable leaf.
    use_roughness_map: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def preset(cls, ndust: int = 4, truncation: int = 63, nudged: bool = False
               ) -> "DustParameters":
        """``mo_ham_dust.f90::get_dust_namelist_defaults`` for ``ndust`` 2-4."""
        table = np.array(SOIL_TABLE, dtype=float)
        scale = np.ones(N_REGIONS)
        thresh = np.ones(_N_EAST_ASIA_ROWS)
        if ndust == 2:                       # Cheng (2008)
            rough, lai, smst, easo = 0.0, 1.0e-10, True, 0
            scale[:] = 0.68
        elif ndust == 3:                     # Stier et al. (2005)
            rough, lai, smst, easo = 0.001, 1.0e-10, False, 1
            # Resolution fit tuned at T21/T42 to reproduce the T63 total; the
            # source warns it must be re-tuned above T63.
            poly = (-7.9365e-5 * truncation ** 2 + 0.0095238 * truncation + 0.575)
            scale[:] = 0.86 if truncation > 63 else poly
        elif ndust == 4:                     # Stier (2005) + East-Asia soils (HAM2)
            rough, lai, smst, easo = 0.001, 0.1, False, 2
            table[13, ALPHA_COL] = 1.0e-6    # r_dust_af14: loess
            thresh[0] = 0.6                  # r_dust_sf13: Taklimakan
            if truncation == 63:
                high = 1.25 if nudged else 1.45
                low = 0.95 if nudged else 1.05
                scale[:] = [low, high, high, low, low, low, high, low]
            else:
                scale[:] = 0.86
        else:
            raise ValueError(
                f"ndust={ndust}: only the 2 (Cheng), 3 (Stier 2005) and 4 "
                "(Stier + East-Asia soils, HAM2) presets are ported; ndust=5 "
                "needs the MSG-SEVIRI activation map, which is not available.")
        return cls(
            soil_table=jnp.asarray(table),
            nduscale_reg=jnp.asarray(scale),
            threshold_scale=jnp.asarray(thresh),
            r_dust_umin=jnp.asarray(21.0),
            r_dust_lai=jnp.asarray(lai),
            r_dust_z0s=jnp.asarray(0.001),
            ndurough=jnp.asarray(rough if rough > 0.0 else 0.001),
            r_dust_scz0=jnp.asarray(1.0),
            r_dust_z0min=jnp.asarray(1.0e-5),
            w0=jnp.asarray(0.99),
            aeff=jnp.asarray(0.35),
            xeff=jnp.asarray(10.0),
            a_rnolds=jnp.asarray(1331.647),
            b_rnolds=jnp.asarray(0.38194),
            x_rnolds=jnp.asarray(1.561228),
            d_thrsld=jnp.asarray(2.31e-6),
            uth_coeff=jnp.asarray(0.129),
            fecan_moisture=smst,
            east_asia=easo,
            use_roughness_map=(rough == 0.0),
        )

    @classmethod
    def default(cls) -> "DustParameters":
        return cls.preset()


def _column_field(forcing, name, ncols, default=0.0):
    """Return a ``(ncols,)`` forcing field, or ``default`` when absent/mis-shaped."""
    value = getattr(forcing, name, None) if forcing is not None else None
    if value is None or jnp.size(value) != ncols:
        return jnp.full((ncols,), default)
    return jnp.ravel(value)


def _soil_fractions(forcing, ncols):
    """Return the nine prescribed texture area fractions, zero where unsupplied."""
    types = getattr(forcing, "dust_soil_types", None) if forcing is not None else None
    if types is None:
        return {name: jnp.zeros((ncols,)) for name in SOIL_TYPE_VARS}
    return {name: (jnp.ravel(types[name])
                   if name in types and jnp.size(types[name]) == ncols
                   else jnp.zeros((ncols,)))
            for name in SOIL_TYPE_VARS}


class DustEmissions(PhysicsTerm):
    """Tegen/HAMMOZ dust emission into MAM4's accumulation and coarse modes."""

    name: ClassVar[str] = "jam_dust_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = (
        emission_flux_keys() + (MODEL_LEVEL_WIND_KEY, DUST_SUPERCOARSE_KEY))

    def __init__(
        self,
        params: DustParameters | None = None,
        *,
        ndust: int = 4,
        nudged: bool = False,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold the parameters, the population and the static soil size grid.

        ``ndust``/``nudged`` select the preset :meth:`cache_coords` rebuilds at
        the model's own truncation; an explicit ``params`` overrides both and is
        never rebuilt.
        """
        self._preset = None if params is not None else (ndust, nudged)
        self.params = nnx.Param(
            params if params is not None
            else DustParameters.preset(ndust, nudged=nudged))
        self._spec = spec or MAM4_SPEC
        self._diameters = jnp.asarray(soil_diameters())
        (self._accum, _), (self._coarse, _) = self._spec.primary_split("du")

    def cache_coords(self, coords) -> None:
        """Rebuild the preset at the model's truncation.

        ``nduscale_reg`` is resolution-dependent — the ``ndust = 4`` regional
        vector is defined only at T63 and the ``ndust = 3`` polynomial only up
        to it — so the preset cannot be fixed at construction, where the grid is
        not yet known. An explicitly supplied ``DustParameters`` is left alone.
        """
        if self._preset is None:
            return
        ndust, nudged = self._preset
        # truncation = total_wavenumbers - 2, the relation utils.get_coords uses.
        truncation = int(coords.horizontal.total_wavenumbers) - 2
        self.params = nnx.Param(
            DustParameters.preset(ndust, truncation=truncation, nudged=nudged))

    def _soil_weights(self, forcing, ncols, params):
        """Per-cell area weights of :data:`MIXTURE_ROWS`, before the wind switch.

        Reproduces the two guarded East-Asia branches of the Fortran. The nine
        file fields are two *overlapping* partitions (global Zobler textures and
        Cheng's Chinese textures, summing to 2.18 at Gobi), so a naive nine-way
        sum drives the type-1 residual negative.
        """
        frac = _soil_fractions(forcing, ncols)
        psrc = jnp.clip(_column_field(forcing, "dust_preferential", ncols), 0.0, 1.0)
        east = [frac[f"type{i}"] for i in EAST_ASIA_INDEX]
        global_types = [frac[f"type{i}"] for i in (2, 3, 4, 6)]
        east_total = sum(east)
        base = jnp.ones((ncols,))
        if params.east_asia == 1:
            east = [jnp.zeros((ncols,)) for _ in east]
        elif params.east_asia == 2:
            replaced = east_total > 0.0
            base = jnp.where(replaced, east_total, base)
            global_types = [jnp.where(replaced, 0.0, f) for f in global_types]
            psrc = jnp.where(replaced, 0.0, psrc)
        residual = base - sum(global_types) - sum(east)
        land = 1.0 - psrc
        return [land * residual] + [land * f for f in global_types] \
            + [land * f for f in east], psrc

    def _threshold_scale(self, forcing, ncols, params):
        """``utsc``: the East-Asian threshold multipliers, last match winning."""
        frac = _soil_fractions(forcing, ncols)
        utsc = jnp.ones((ncols,))
        for slot, index in enumerate(EAST_ASIA_INDEX):
            utsc = jnp.where(frac[f"type{index}"] > 0.0,
                             params.threshold_scale[slot], utsc)
        return utsc

    def _region_scale(self, forcing, ncols, params):
        """``nduscale_2d``: the per-region threshold multiplier from ``regions``."""
        regions = getattr(forcing, "dust_regions", None) if forcing is not None else None
        if regions is None or jnp.size(regions) != ncols:
            return jnp.full((ncols,), params.nduscale_reg[0])
        regions = jnp.round(jnp.ravel(regions))
        scale = jnp.zeros((ncols,))
        for r in range(N_REGIONS):
            scale = scale + params.nduscale_reg[r] * (regions == r + 1)
        # Region 1 is "everywhere else", so an unlabelled cell takes its value.
        return jnp.where(scale > 0.0, scale, params.nduscale_reg[0])

    def _roughness(self, forcing, ncols, params):
        """``Z01``/``Z02`` [cm], floored and scaled as ``bgc_read_fpar_field`` does."""
        if params.use_roughness_map:
            # ndust=2 is the only preset with a live roughness map, and the
            # Fortran aborts without the file; silently flooring z0 to z0min
            # would give feff = 1 and quietly emit the wrong flux.
            supplied = (getattr(forcing, "dust_roughness", None)
                        if forcing is not None else None)
            if supplied is None or jnp.size(supplied) != ncols:
                raise ValueError(
                    "DustParameters.use_roughness_map is set (ndust=2) but "
                    "forcing.dust_roughness is missing or the wrong shape. Set "
                    "forcing.dust_roughness_file (or 'auto'), or use the "
                    "default constant roughness.")
            z0 = jnp.ravel(supplied)
        else:
            z0 = jnp.full((ncols,), 1.0) * params.ndurough
        return jnp.maximum(z0, params.r_dust_z0min) * params.r_dust_scz0

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape
        d = self._diameters

        # The saltation threshold and the log law are both calibrated to a 10 m
        # wind, so a column falling back to the lowest model level (~33 m at
        # L47) emits nothing rather than over-driving u*³ with a sheared wind.
        u10, from_model_level = wind_10m(state, diagnostics)
        u10 = jnp.where(from_model_level > 0.0, 0.0,
                        jnp.maximum(jnp.ravel(u10), 0.0))
        pot = jnp.clip(_column_field(forcing, "dust_source", ncols), 0.0, 1.0)
        snow = jnp.clip(_column_field(forcing, "snowc_am", ncols), 0.0, 1.0)
        wetness = jnp.clip(_column_field(forcing, "soilw_am", ncols), 0.0, 1.0)

        z0 = self._roughness(forcing, ncols, p)
        # MB95 (17). With the default ndurough = z0s the log is zero and feff ≡ 1;
        # the obstacle-spacing factor is 1 because the Fortran never sets d1.
        feff = jnp.clip(
            1.0 - jnp.log(z0 / p.r_dust_z0s)
            / jnp.log(p.aeff * (p.xeff / p.r_dust_z0s) ** 0.8), 0.0, 1.0)
        u_star = VK * (u10 * 100.0) / jnp.log(ZZ / z0)     # MB95 (15), cm/s

        nduscale = self._region_scale(forcing, ncols, p)
        utsc = self._threshold_scale(forcing, ncols, p)
        safe_feff = jnp.where(feff > 0.0, feff, 1.0)
        mask = ((feff > 0.0)
                & (u_star > 0.0)
                & (u_star >= p.r_dust_umin * nduscale / safe_feff)
                & (pot > p.r_dust_lai))

        srel, srel_v, su_srel_v = soil_size_distributions(p.soil_table, d)
        weight_matrix = emission_weight_matrix(srel, srel_v, su_srel_v, d)
        alpha = jnp.stack([p.soil_table[jf - 1, ALPHA_COL]
                           for jf, _ in MIXTURE_ROWS])                   # (nrow,)

        uth = threshold_friction_velocity(
            d, p.a_rnolds, p.b_rnolds, p.x_rnolds, p.d_thrsld, p.uth_coeff)
        safe_u = jnp.where(mask, u_star, 1.0)
        if p.fecan_moisture:
            # Fécan et al. (1999): the threshold rises once the soil water
            # exceeds the texture's residual moisture. jcm has no ECHAM ws/wsmx,
            # so the relative wetness stands in for the Fortran's min(ws/ρ_p,1)
            # — a declared approximation on an off-by-default path (#787).
            w_res = p.soil_table[:, WRES_COL] * 100.0                    # (17,)
            w = wetness * 100.0
            excess = jnp.maximum(w[None, :] - w_res[:, None], 0.0)       # (17, ncols)
            wet = jnp.sqrt(1.0 + 1.21 * excess ** 0.68)
            wet = jnp.stack([wet[js - 1] for _, js in MIXTURE_ROWS])     # (nrow, ncols)
            ratio = (uth[None, :, None] * wet[:, None, :]
                     * (nduscale * utsc / safe_feff / safe_u)[None, None, :])
            base = _saltation(ratio, safe_u)
            aggregates = jnp.einsum("rck,rki->rci", weight_matrix, base)
        else:
            ratio = uth[:, None] * (nduscale * utsc / safe_feff / safe_u)[None, :]
            base = _saltation(ratio, safe_u)                             # (191, ncols)
            aggregates = jnp.einsum("rck,ki->rci", weight_matrix, base)

        row_weights, psrc = self._soil_weights(forcing, ncols, p)
        # The preferential source keeps type 10's magnitude but switches its
        # emitted spectrum to type 11 (clay) above HIGH_WIND_MS — a hard step
        # with zero gradient (#664), ported as the step it is.
        high = u10 > HIGH_WIND_MS
        row_weights = row_weights + [psrc * (1.0 - high), psrc * high]
        weights = jnp.stack(row_weights) * alpha[:, None]                # (nrow, ncols)
        moments = jnp.einsum("rci,ri->ci", aggregates, weights)          # (5, ncols)

        # g cm⁻² s⁻¹ -> kg m⁻² s⁻¹ is ×1e4 (area) ×1e-3 (mass); the snow factor
        # and the potential-source multiplier are both unconditional, and a
        # saturated soil emits nothing in every preset.
        scale = jnp.where(mask & (wetness <= p.w0), 10.0 * (1.0 - snow) * pot, 0.0)
        mass_acc = moments[0] * scale
        num_acc = moments[1] * scale
        mass_cor = moments[2] * scale
        num_cor = moments[3] * scale
        mass_all = moments[4] * scale

        # Number-conserving emitted diameter within each window: the diameter at
        # which one particle's mass equals mass/number, D = (ΣF / Σ(F/D³))^(1/3).
        d_acc = _effective_diameter(mass_acc, num_acc, ACCUM_UM)
        d_cor = _effective_diameter(mass_cor, num_cor, COARSE_UM)

        fluxes = [("du", self._accum.short, mass_acc, d_acc),
                  ("du", self._coarse.short, mass_cor, d_cor)]
        tracer_tends = distribute_surface_flux(self._spec, fluxes, air_density, dz)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        diagnostics = accumulate_emission_fluxes(
            diagnostics, tracer_tends, air_density, dz)
        diagnostics = {**diagnostics,
                       MODEL_LEVEL_WIND_KEY: jnp.maximum(
                           diagnostics.get(MODEL_LEVEL_WIND_KEY, 0.0),
                           from_model_level),
                       DUST_SUPERCOARSE_KEY: mass_all - mass_acc - mass_cor}
        return tendency, diagnostics


def _saltation(ratio, u_star):
    """MB95 (28)/(33) saltation factor ``(1+R)²(1−R)·cd·u*³``, zero below threshold."""
    factor = (1.0 + ratio) ** 2 * (1.0 - ratio)
    return jnp.where(ratio < 1.0, factor, 0.0) * CD * u_star ** 3


def _effective_diameter(mass, number_integral, window_um):
    """``(Σ F / Σ F/D³)^(1/3)`` [m], falling back to the window's centre when idle."""
    fallback = math.sqrt(window_um[0] * window_um[1]) * 1.0e-6
    ok = number_integral > 0.0
    ratio = jnp.where(ok, mass, 1.0) / jnp.where(ok, number_integral, 1.0)
    return jnp.where(ok, ratio ** (1.0 / 3.0) * 1.0e-2, fallback)
