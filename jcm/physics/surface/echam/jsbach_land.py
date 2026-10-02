"""JSBACH's land-tile evaporation form and the land skin energy balance (#979).

The ECHAM hosts' land tile is a *prescribed-moisture* land: soil moisture,
snow cover and the deep soil temperature are prescribed climatologies, while
the skin temperature is prognostic and closes an energy balance coupled
implicitly to the lowest model level. This module holds the column
physics of that tile, ported from ECHAM6.3 / JSBACH (r7492) and pinned against
the compiled Fortran (``jcm/data/test/echam_land_reference``):

* :func:`humidity_factors` — JSBACH's ``cair``/``csat`` (``mo_soil.f90::
  update_soil``): bare soil evaporates only while ``h·q_s > q_a``, the canopy
  transpires through a conductance limited by the water-stress factor, snow and
  glacier at the potential rate.
* :func:`unstressed_canopy_conductance` — ECHAM3's canopy conductance
  (``mo_canopy.f90::unstressed_canopy_cond_par``).
* :func:`top_layer_thermal_properties` — the capacity and conductance of
  JSBACH's top soil layer (``update_soiltemp.f90``) over a reservoir at the
  prescribed soil temperature.
* :func:`update_surfacetemp` — the implicit surface energy balance
  (``update_surfacetemp.f90``), verbatim.
* :func:`richtmyer_morton` — the per-tile Richtmyer–Morton coefficients
  (``mo_surface_land.f90::richtmyer_land``; ``_ocean``/``_ice`` are the same
  with ``cair = csat = 1``).

Every function is broadcasting-native: arguments are per column ``(*horiz)``
and broadcast numpy-style. The design, the stand-ins and their derivations
are in ``docs/source/design/land_skin_energy_balance.md``; what the land tile
is, scientifically, in ``docs/source/science/surface.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

import jcm.constants as c
from jcm.physics.surrogate_gradient import with_surrogate_gradient

__all__ = [
    "JsbachLandParameters",
    "bare_soil_relative_humidity",
    "water_stress_factor",
    "unstressed_canopy_conductance",
    "humidity_factors",
    "top_layer_thermal_properties",
    "update_surfacetemp",
    "richtmyer_morton",
    "melt_cap",
]

#: JSBACH's "no stomatal resistance limit" (``mo_soil.f90:1745-1757``).
_CANOPY_RESISTANCE_CLOSED = 1.0e20
#: ``EPSILON(1._dp)`` in the canopy gate (``mo_soil.f90:1752``).
_EPS_DP = 2.220446049250313e-16
#: ``relative_humidity > 1.e-10`` in the bare-soil switch (``mo_soil.f90:2531``).
_H_MIN = 1.0e-10
#: Floor of the exchange velocity in the canopy factor [m/s]: the port's own
#: floor on every surface exchange coefficient (``surface_layer.py``). ECHAM's
#: ``zchl·max(1, |U|)`` is never 0; a cold-start carry is, and the canopy
#: factor's slope ``g/(g + C_h|U|)²`` at ``g ≈ 1e-20`` and ``C_h|U| = 0`` is
#: infinite in float32, which poisons a reverse pass through a zero land
#: fraction.
_EXCHANGE_FLOOR = 1.0e-6


@struct.dataclass
class JsbachLandParameters:
    """Constants of the prescribed-moisture JSBACH land tile.

    Every numeric value is a differentiable pytree leaf; the surrogate widths
    configure only derivatives and are static (``docs/source/design/
    surrogate_gradients.md``). Provenance of each value is ECHAM6.3 / JSBACH
    r7492 unless marked as a jcm stand-in.
    """

    # --- soil moisture thresholds (mo_soil.f90::config_soil, 326-327) -------
    #: ``moist_wilt_fract``: transpiration stops below this root-zone fill.
    moisture_wilting_fraction: jnp.ndarray = 0.35
    #: ``moist_crit_fract``: unstressed transpiration above this fill.
    moisture_critical_fraction: jnp.ndarray = 0.75

    # --- canopy (mo_canopy.f90, ECHAM3 manual eq. 3.3.2.12) -----------------
    #: Leaf area index of the vegetated fraction. jcm stand-in: the forcing
    #: bundle carries no LAI; 4 is a closed canopy, consistent with the
    #: vegetated fraction being the forest fraction.
    leaf_area_index: jnp.ndarray = 4.0
    #: ECHAM3's canopy extinction ``k``, ``a`` [J/m3], ``b`` [W/m2], ``c`` [s/m].
    canopy_extinction: jnp.ndarray = 0.9
    canopy_a: jnp.ndarray = 5000.0
    canopy_b: jnp.ndarray = 10.0
    canopy_c: jnp.ndarray = 100.0
    #: Share of the net shortwave that is photosynthetically active. jcm
    #: stand-in for ECHAM's net visible band (``sw_vis_net``): the radiation
    #: here publishes broadband surface fluxes.
    par_fraction: jnp.ndarray = 0.5

    # --- soil heat (mo_soil.f90:485-497 grid, 996-997 FAO row 0) -----------
    #: ``VolHeatCap(0)`` [J m-3 K-1]: volumetric heat capacity of the soil.
    soil_heat_capacity: jnp.ndarray = 2.25e6
    #: ``ThermalDiff(0)`` [m2/s]: thermal diffusivity of the soil.
    soil_thermal_diffusivity: jnp.ndarray = 7.4e-7
    #: ``cdel(1)``, ``cdel(2)`` [m]: thickness of JSBACH's top two soil layers.
    top_layer_thickness: jnp.ndarray = 0.065
    second_layer_thickness: jnp.ndarray = 0.254

    # --- snow and ice (update_soiltemp.f90:82-86) ---------------------------
    #: ``zsn_capa`` [J m-3 K-1] and ``zsn_cond`` [W m-1 K-1] of snow.
    snow_heat_capacity: jnp.ndarray = 634500.0
    snow_conductivity: jnp.ndarray = 0.31
    #: ``snow_density`` [kg/m3] (mo_jsbach_constants.f90:75).
    snow_density: jnp.ndarray = 330.0
    #: ``zrici`` [J m-3 K-1] and ``zdifiz`` [m2/s] of glacier ice.
    ice_heat_capacity: jnp.ndarray = 2.09e6
    ice_thermal_diffusivity: jnp.ndarray = 12.0e-7
    #: Snow water equivalent [m] at full prescribed cover: the forcing
    #: bundle's own ``snowc = min(1, SWE / 60 mm)`` (jcm.data.mirror.bundles),
    #: inverted to give the snow depth the top-layer grading needs.
    full_cover_snow_water_equivalent: jnp.ndarray = 0.06
    #: ``crit_snow_depth`` [m water] (mo_soil.f90:328): above it the skin
    #: cannot warm past the melting point.
    critical_snow_depth: jnp.ndarray = 5.85036e-3

    # --- surrogate-gradient widths (static) --------------------------------
    #: Width of the bare-soil and dew switches, in relative-humidity units.
    hinge_width: float = struct.field(pytree_node=False, default=0.02)
    #: Width of the water-stress clip.
    stress_width: float = struct.field(pytree_node=False, default=0.02)
    #: Width of the melt cap [K].
    melt_width: float = struct.field(pytree_node=False, default=0.5)

    # --- configuration (static) ------------------------------------------
    #: How the land skin temperature is set. ``"prognostic"`` (default) solves
    #: the surface energy balance. ``"prescribed"`` holds the skin at the
    #: forcing's land temperature every step
    #: (:func:`~jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion.forcing_land_temperature`)
    #: with the same evaporation form, humidity factors and per-tile coupling;
    #: the surface energy budget is then open and its residual is published as
    #: ``surface.land_energy_residual``. That is the fixed-SST and fixed-land-
    #: temperature configuration of Andrews et al. (2021) for the effective
    #: radiative forcing. From the command line::
    #:
    #:     python -m jcm.main +configuration=t63-echam-1m \
    #:         +physics.terms.tte_tke_vertical_diffusion.land_params.land_temperature=prescribed
    #:
    #: (``+physics.land_surface.land_temperature=prescribed`` for the presets
    #: built by ``echam_physics``).
    land_temperature: str = struct.field(pytree_node=False, default="prognostic")

    @classmethod
    def default(cls) -> "JsbachLandParameters":
        """ECHAM6.3 / JSBACH values with the documented jcm stand-ins."""
        return cls()


#: The values of :attr:`JsbachLandParameters.land_temperature`.
LAND_TEMPERATURE_MODES = ("prognostic", "prescribed")


def check_land_temperature_mode(params: JsbachLandParameters) -> str:
    """Return ``params.land_temperature``, refusing a value that is not a mode."""
    mode = params.land_temperature
    if mode not in LAND_TEMPERATURE_MODES:
        raise ValueError(f"JsbachLandParameters.land_temperature must be one of "
                         f"{LAND_TEMPERATURE_MODES}, got {mode!r}")
    return mode


# ---------------------------------------------------------------------------
# Smooth stand-ins used only for derivatives
# ---------------------------------------------------------------------------

def _sigmoid(x, width):
    return jax.nn.sigmoid(x / width)


def _soft_clip01(x, width):
    """Softplus clip to ``[0, 1]``: ``w·sp(x/w) − w·sp((x − 1)/w)``."""
    return width * (jax.nn.softplus(x / width) - jax.nn.softplus((x - 1.0) / width))


# ---------------------------------------------------------------------------
# Moisture
# ---------------------------------------------------------------------------

def bare_soil_relative_humidity(upper_layer_fill: jnp.ndarray) -> jnp.ndarray:
    """Relative humidity of the bare soil, ``calc_relative_humidity_upper``.

    ``h = ½(1 − cos(π·min(w, 1)))`` for ``w > 0`` and 0 otherwise
    (``mo_soil.f90:2754-2777``), ``w`` the upper layer's water content as a
    fraction of its field capacity. This is the ``nsoil > 1`` branch of
    ``update_soil`` (2492-2499), the 5-layer configuration ECHAM6.3 runs. ``h``
    is C¹ — flat at both ends — so it needs no surrogate.
    """
    w = jnp.minimum(upper_layer_fill, 1.0)
    return jnp.where(upper_layer_fill > 0.0, 0.5 * (1.0 - jnp.cos(jnp.pi * w)), 0.0)


def _stress_exact(w, wilt, crit):
    return jnp.clip((w - wilt) / (crit - wilt), 0.0, 1.0)


def water_stress_factor(root_zone_fill, wilt, crit, width: float = 0.0):
    """JSBACH's water-stress factor, ``calc_water_stress_factor``.

    ``clip((w − w_wilt)/(w_crit − w_wilt), 0, 1)`` (``mo_soil.f90:2689-2703``),
    ``w`` the root-zone fill. Exact value; with ``width > 0`` the derivative is
    that of a softplus clip, so the plateaux below wilting and above the
    critical fill still see the thresholds.
    """
    if width == 0.0:
        return _stress_exact(root_zone_fill, wilt, crit)

    def surrogate(w, a, b):
        return _soft_clip01((w - a) / (b - a), width)

    return with_surrogate_gradient(_stress_exact, surrogate)(root_zone_fill, wilt, crit)


def unstressed_canopy_conductance(leaf_area_index, absorbed_par, params: JsbachLandParameters):
    """Unstressed canopy conductance [m/s], ``unstressed_canopy_cond_par``.

    ECHAM3's integral of the stomatal conductance over the canopy
    (``mo_canopy.f90:23-47``, ECHAM3 manual eq. 3.3.2.12)::

        d = (a + b·c)/(c·PAR),
        g = [ ln((d·e^{kL} + 1)/(d + 1))·b/(d·PAR) − ln((d + e^{−kL})/(d + 1)) ] / (k·c)

    with ``PAR`` floored at 1e-10 W/m2 and ``g = 1e-20`` without leaves. At
    night it tends to ``L·b/(k·(a + b·c))·k = L/600`` m/s for the default
    constants, so stomata narrow but do not close.
    """
    p = params
    par = jnp.maximum(1.0e-10, absorbed_par)
    d = (p.canopy_a + p.canopy_b * p.canopy_c) / (p.canopy_c * par)
    k_lai = p.canopy_extinction * leaf_area_index
    g = ((jnp.log((d * jnp.exp(k_lai) + 1.0) / (d + 1.0)) * p.canopy_b / (d * par)
          - jnp.log((d + jnp.exp(-k_lai)) / (d + 1.0)))
         / (p.canopy_extinction * p.canopy_c))
    return jnp.where(leaf_area_index > _EPS_DP, g, 1.0e-20)


def _factors(h, w_root, snow, glacier, veg, gc, chu, qa, qs, wilt, crit, *, bare, dew, stress,
             wilting_gate=True):
    """``update_soil``'s humidity factors given the three switch values.

    ``bare``/``dew`` are the bare-soil and dew indicators (exact: 0/1), and
    ``stress`` the water-stress factor; the caller decides whether they are
    the reference or the smooth surrogate. One non-glacier tile (cover
    ``1 − g``, snow ``snow``, vegetation ratio ``veg``, wet-skin fraction 0)
    and one glacier tile (cover ``g``, snow 1), tile-averaged as
    ``average_tiles`` does (``mo_soil.f90:2561-2566``).

    ``wilting_gate`` is JSBACH's separate ``moisture > wilt`` test on the
    canopy (2511). Below wilting the stress factor is already 0, so the gate
    changes the value by at most ``1e-20/C_h|U|``; the surrogate leaves it out,
    because a hard gate would hide the wilting point from the derivative its
    smooth stress clip provides.
    """
    # Canopy resistance (mo_soil.f90:1741-1761): open while stressed > eps,
    # conductance > eps and the air below saturation; the factor
    # 1/(1 + C_h|U|·r_c) written with the conductance so it stays finite.
    conductance = gc * stress + 1.0e-20
    open_canopy = (stress > _EPS_DP) & (gc > _EPS_DP)
    canopy_open = conductance / (conductance + chu)
    canopy_shut = 1.0 / (1.0 + chu * _CANOPY_RESISTANCE_CLOSED)
    canopy = jnp.where(open_canopy, (1.0 - dew) * canopy_open + dew * canopy_shut, canopy_shut)
    # qsat_veg / qair_veg (2509-2528): transpiration only above wilting.
    if wilting_gate:
        qsat_veg = jnp.where(w_root > wilt, snow + (1.0 - snow) * canopy, snow)
    else:
        qsat_veg = snow + (1.0 - snow) * canopy
    # Bare soil (2530-2552): RH form while h > q_a/q_s, nothing otherwise,
    # the potential rate when the air is supersaturated over the surface.
    qsat_fact = snow + (1.0 - snow) * h * bare
    qair_fact = snow + (1.0 - snow) * bare
    qsat_fact = dew + (1.0 - dew) * qsat_fact
    qair_fact = dew + (1.0 - dew) * qair_fact
    csat = veg * qsat_veg + (1.0 - veg) * qsat_fact
    cair = veg * qsat_veg + (1.0 - veg) * qair_fact
    # Glacier tile: snow_fract = 1 makes both factors 1.
    return (1.0 - glacier) * cair + glacier, (1.0 - glacier) * csat + glacier


def humidity_factors(bare_soil_humidity, root_zone_fill, snow_cover, glacier_fraction,
                     vegetated_fraction, canopy_conductance, exchange_velocity,
                     air_humidity, surface_saturation, params: JsbachLandParameters):
    """JSBACH's land humidity factors ``(cair, csat, stress)``.

    The land moisture flux is ``E = ρ·C_h|U|·(csat·q_s − cair·q_a)``
    (``mo_soil.f90:1900-1902``). The factors are those of ``update_soil``'s
    canopy-resistance and humidity-factor blocks (1741-1761, 2503-2574),
    collapsed to one non-glacier and one glacier tile:

    * bare soil, ``h > q_a/q_s``: ``qsat_fact = s + (1−s)·h``, ``qair_fact = 1``
      (the relative-humidity form); otherwise both ``s`` (no bare-soil flux);
      both 1 when ``q_a > q_s`` (deposition at the potential rate);
    * vegetation above wilting: ``s + (1−s)/(1 + C_h|U|·r_c)``,
      ``r_c = 1/(g_c·β)`` (1e20 when ``β`` or ``g_c`` vanish or ``q_a > q_s``);
    * ``csat = v·qsat_veg + (1−v)·qsat_fact``, likewise ``cair``; the glacier
      share evaporates at the potential rate.

    The values are the reference's exactly. The derivatives are those of the
    same expressions with sigmoid switches and a softplus stress clip of the
    widths in ``params`` (one surrogate for the pair, applied where they are
    formed, so every flux that reads them sees one derivative).

    Args:
        bare_soil_humidity: ``h`` (:func:`bare_soil_relative_humidity`).
        root_zone_fill: root-zone water as a fraction of capacity, for ``β``.
        snow_cover: snow-covered share of the non-glacier land.
        glacier_fraction: glacier share of the land.
        vegetated_fraction: vegetation ratio of the non-glacier land.
        canopy_conductance: unstressed canopy conductance [m/s].
        exchange_velocity: the land tile's ``C_h·max(|U|, 1)`` [m/s] (ECHAM
            ``zchl·max(1, |U|)``), floored at the port's 1e-6 m/s.
        air_humidity: lowest-level specific humidity [kg/kg].
        surface_saturation: saturation specific humidity at the skin [kg/kg].
        params: the land constants.

    Returns:
        ``(cair, csat, stress)``, each ``(*horiz)``.

    """
    wilt = params.moisture_wilting_fraction
    crit = params.moisture_critical_fraction

    def exact(h, w, s, g, v, gc, chu, qa, qs, a, b):
        rh = qa / qs
        bare = ((h > rh) & (h > _H_MIN)).astype(h.dtype)
        dew = (qa > qs).astype(h.dtype)
        stress = _stress_exact(w, a, b)
        cair, csat = _factors(h, w, s, g, v, gc, chu, qa, qs, a, b,
                              bare=bare, dew=dew, stress=stress)
        return cair, csat, stress

    def surrogate(h, w, s, g, v, gc, chu, qa, qs, a, b):
        rh = qa / qs
        bare = _sigmoid(h - rh, params.hinge_width) * (h > _H_MIN)
        dew = _sigmoid(rh - 1.0, params.hinge_width)
        stress = _soft_clip01((w - a) / (b - a), params.stress_width)
        cair, csat = _factors(h, w, s, g, v, gc, chu, qa, qs, a, b,
                              bare=bare, dew=dew, stress=stress, wilting_gate=False)
        return cair, csat, stress

    args = jnp.broadcast_arrays(
        jnp.asarray(bare_soil_humidity), jnp.asarray(root_zone_fill), jnp.asarray(snow_cover),
        jnp.asarray(glacier_fraction), jnp.asarray(vegetated_fraction),
        jnp.asarray(canopy_conductance),
        jnp.maximum(jnp.asarray(exchange_velocity), _EXCHANGE_FLOOR),
        jnp.asarray(air_humidity), jnp.asarray(surface_saturation))
    args = tuple(args) + (wilt, crit)
    if params.hinge_width == 0.0 and params.stress_width == 0.0:
        return exact(*args)
    return with_surrogate_gradient(exact, surrogate)(*args)


# ---------------------------------------------------------------------------
# Heat
# ---------------------------------------------------------------------------

def top_layer_thermal_properties(snow_cover, glacier_fraction, params: JsbachLandParameters):
    """Heat capacity [J m-2 K-1] and conductance [W m-2 K-1] of the skin layer.

    JSBACH's surface temperature is its top soil layer's (``update_soiltemp.f90:
    142``). With the layer below held at the prescribed soil temperature, the
    surface sees ``C_s = ρc₁·cdel(1)`` and ``Λ = κ₁/(cmid(2) − cmid(1))``
    (``zdz2(1)·Δt`` and ``zdz1(1)``, lines 194-200). The top layer is graded
    between soil and snow by snow depth, in series (lines 157-170):
    ``x = min(h_sn/cmid(2), 1)``, ``ρc = x·ρc_sn + (1−x)·ρc``,
    ``κ = 1/(x/κ_sn + (1−x)/κ)``, with the depth from the prescribed cover,
    ``h_sn = snowc·SWE_full·ρ_w/ρ_sn``. The glacier share takes ice
    (lines 105-110), and the two tiles average by cover (``update_soil``
    1834-1840).
    """
    p = params
    cdel1, cdel2 = p.top_layer_thickness, p.second_layer_thickness
    cmid1 = 0.5 * cdel1
    cmid2 = cdel1 + 0.5 * cdel2
    k_soil = p.soil_heat_capacity * p.soil_thermal_diffusivity
    k_ice = p.ice_heat_capacity * p.ice_thermal_diffusivity
    depth = snow_cover * p.full_cover_snow_water_equivalent * c.rhow / p.snow_density
    x = jnp.minimum(depth / cmid2, 1.0)
    rhoc_land = x * p.snow_heat_capacity + (1.0 - x) * p.soil_heat_capacity
    k_land = 1.0 / (x / p.snow_conductivity + (1.0 - x) / k_soil)
    heat_capacity = cdel1 * ((1.0 - glacier_fraction) * rhoc_land
                             + glacier_fraction * p.ice_heat_capacity)
    conductance = ((1.0 - glacier_fraction) * k_land + glacier_fraction * k_ice) / (cmid2 - cmid1)
    return heat_capacity, conductance


def update_surfacetemp(pcp, pescoe, pfscoe, peqcoe, pfqcoe, psold, pqsold, pdqsold,
                       pnetrad, pgrdfl, pcfh, pcair, pcsat, pfracsu, pgrdcap,
                       dt, tpfac1, emissivity, *, latent_heat_vaporization=None,
                       latent_heat_sublimation=None, stefan_boltzmann=None):
    """Solve the implicit surface energy balance, ``update_surfacetemp.f90``.

    Verbatim: with ``α = tpfac1`` and the linearised emission and saturation,
    the surface dry static energy ``ŝ = α·s_new + (1−α)·s_old`` solves
    ``pgrdcap·(ŝ − s_old)/c_p = α·Δt·[Rn + H + LE + G]`` (fluxes positive into
    the surface) with the lowest level eliminated through the
    Richtmyer–Morton relations ``s_K = pescoe·ŝ + pfscoe``,
    ``q_K = peqcoe·q̂_s + pfqcoe``. Arguments carry ECHAM's names and units
    (``pcp`` J kg-1 K-1, ``ps*`` J/kg, ``pcfh = ρ·C_h|U|`` kg m-2 s-1,
    ``pnetrad``/``pgrdfl`` W/m2, ``pgrdcap`` J m-2 K-1). The latent heats and
    the Stefan–Boltzmann constant default to ``jcm.constants``; the reference
    test passes ECHAM's.
    """
    alv = c.alhc if latent_heat_vaporization is None else latent_heat_vaporization
    als = c.alhs if latent_heat_sublimation is None else latent_heat_sublimation
    stbo = c.sbc if stefan_boltzmann is None else stefan_boltzmann
    pdt = tpfac1 * dt
    zicp = 1.0 / pcp
    zca = als * pfracsu + alv * (pcair - pfracsu)
    zcs = als * pfracsu + alv * (pcsat - pfracsu)
    zcolin = (pgrdcap * zicp
              + pdt * (zicp * 4.0 * emissivity * stbo * ((zicp * psold) ** 3)
                       - pcfh * (zca * peqcoe - zcs) * zicp * pdqsold))
    zcohfl = -pdt * pcfh * (pescoe - 1.0)
    zcoind = pdt * (pnetrad + pcfh * pfscoe
                    + pcfh * ((zca * peqcoe - zcs) * pqsold + zca * pfqcoe) + pgrdfl)
    return (zcolin * psold + zcoind) / (zcolin + zcohfl)


def richtmyer_morton(den_heat, den_moisture, rhs_heat, rhs_moisture, k_heat, k_moisture,
                     cair, csat, tpfac1):
    """Per-tile Richtmyer–Morton coefficients, ``richtmyer_land``.

    After the top-down elimination the lowest level obeys
    ``X̂_K = E·X̂_s + F`` (``mo_surface_land.f90:337-392``)::

        E_s = k/(D + k),               F_s = α·R_s/(D + k)
        E_q = csat·k/(D + cair·k),     F_q = α·R_q/(D + cair·k)

    with ``D = 1 + zfac·(1 − zebsh_{K−1})`` the eliminated bottom diagonal,
    ``R = ztdif_K + zfac·ztdif_{K−1}`` the eliminated bottom right-hand side
    (both in the solver's ``X/α`` units), ``k = zcfh_sfc·zqdp`` the tile's
    dimensionless exchange and ``α = tpfac1``. ``richtmyer_ocean``/``_ice`` are
    this with ``cair = csat = 1``. ECHAM eliminates heat and moisture with one
    coefficient; the port's two matrices carry their own (``den_heat``,
    ``den_moisture``), identical when the interior coefficients are.

    Returns:
        ``(E_s, F_s, E_q, F_q)``.

    """
    disc_s = den_heat + k_heat
    disc_q = den_moisture + cair * k_moisture
    return (k_heat / disc_s, tpfac1 * rhs_heat / disc_s,
            csat * k_moisture / disc_q, tpfac1 * rhs_moisture / disc_q)


def _melt_exact(t, capped, tmelt):
    return jnp.where(capped, jnp.minimum(t, tmelt), t)


def melt_cap(temperature, capped, params: JsbachLandParameters):
    """``min(T, tmelt)`` where the land holds snow or glacier (``mo_soil.f90:1859-1863``).

    ECHAM keeps a snow- or glacier-covered surface at the melting point and
    spends the excess on melt (``update_surf_down.f90:246-263``). Exact value;
    with ``params.melt_width > 0`` the derivative is that of a softplus
    minimum, so a melting surface still sees the energy it is given.
    """
    tmelt = jnp.asarray(c.tmelt, jnp.asarray(temperature).dtype)
    if params.melt_width == 0.0:
        return _melt_exact(temperature, capped, tmelt)
    w = params.melt_width

    def smooth(t, cap, tm):
        # t - w·softplus((t - tmelt)/w): t well below the melting point,
        # tmelt well above it, slope sigmoid((tmelt - t)/w).
        return jnp.where(cap, t - w * jax.nn.softplus((t - tm) / w), t)

    return with_surrogate_gradient(_melt_exact, smooth)(temperature, capped, tmelt)
