"""Column physics of SLIM, the Simple Land Interface Model (Laguë et al. 2019).

SLIM (ESCOMP/SimpleLand, ``src/biogeophys/mml_main.F90``) is a land made of a
soil-water bucket, a snow bucket and a ten-layer soil temperature column, with
its surface properties (albedos, bucket capacity, evaporative resistance,
vegetation height, soil thermal properties, glacier mask) prescribed as monthly
maps. This module holds the column physics jcm runs for it on the ECHAM hosts'
land tile; the coupling to the atmosphere (the implicit surface energy balance)
is the tile's own Richtmyer–Morton solve
(:func:`jcm.physics.vertical_diffusion.tte_tke.matrix_solver.couple_surface_tiles`).

* :func:`soil_layer_geometry` — SLIM's layer depths, thicknesses and interface
  depths (``atm2lndType.F90::InitCold``) and its freezable water per layer.
* :func:`soil_thermal_properties` — ``mml_main.F90::soil_thermal_properties``.
* :func:`eliminate_soil_column` / :func:`complete_soil_column` — the implicit
  tridiagonal soil solve of ``mml_main`` (upward sweep, surface solve,
  downward sweep).
* :func:`phase_change` — ``mml_main.F90::phase_change``.
* :func:`bucket_beta`, :func:`snow_indicator`, :func:`evaporation_factor` — the
  evaporation form ``E = ρ(q_s − q_a)·β/(r_evap + r_a)`` written as the
  humidity factor of the tile's implicit coupling.
* :func:`melt_energy` — ``mml_main``'s snow melt (``ptl_snow_melt``, the melt
  cap at the freezing point).
* :func:`update_buckets` — ``mml_main``'s bucket hydrology.
* :func:`land_albedos` — the snow-masked blend of ground and snow albedos.

Every function is broadcasting-native: the soil layer is axis 0 of the column
arrays, every other axis is horizontal. Design, operator split and
provenance: ``docs/source/design/land_skin_energy_balance.md`` (section
"The SLIM land"); the science: ``docs/source/science/surface.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

import jcm.constants as c
from jcm.physics.surrogate_gradient import with_surrogate_gradient

__all__ = [
    "NSOIL",
    "SlimParameters",
    "soil_layer_geometry",
    "soil_thermal_properties",
    "eliminate_soil_column",
    "complete_soil_column",
    "phase_change",
    "seed_soil_ice",
    "bucket_beta",
    "snow_indicator",
    "evaporation_factor",
    "cap_evaporation_factor",
    "melt_energy",
    "update_buckets",
    "snow_masking_factor",
    "land_albedos",
]

#: Number of soil layers (``mml_nsoi = 10`` in ``mml_main.F90``).
NSOIL = 10


@struct.dataclass
class SlimParameters:
    """Constants of SLIM (``mml_main.F90``, ESCOMP/SimpleLand).

    Every numeric value is a differentiable pytree leaf; the surrogate widths
    configure only derivatives and are static
    (``docs/source/design/surrogate_gradients.md``). The latent heats and the
    freezing point are jcm's (:mod:`jcm.constants`), the values the atmosphere
    exchanges energy with, so the land and the atmosphere close one budget.
    """

    #: The ``0.75`` in ``beta = min(water/(0.75·bucket_cap), 1)``.
    bucket_beta_fraction: jnp.ndarray = 0.75
    #: ``snowcap`` [kg/m2]: snow above it runs off.
    snow_cap: jnp.ndarray = 5000.0
    #: Upper bound of the total resistance ``r_evap + r_a`` [s/m] (``res`` cap).
    resistance_cap: jnp.ndarray = 1.0e5
    #: ``z0m = momentum_roughness_fraction·roughness`` (vegetation height).
    momentum_roughness_fraction: jnp.ndarray = 0.1
    #: ``z0h = heat_roughness_fraction·z0m``.
    heat_roughness_fraction: jnp.ndarray = 0.1
    #: Layer depths ``z_i = −scale·(exp(growth·(i − 0.5)) − 1)`` [m], i = 1..10
    #: (``atm2lndType.F90::InitCold``; SLIM does not read ``soilz`` from file).
    soil_depth_scale: jnp.ndarray = 0.025
    soil_depth_growth: jnp.ndarray = 0.5
    #: Freezable water of every layer but the top one [kg/m3]
    #: (``max_freezable``); the top layer holds none.
    max_freezable_water: jnp.ndarray = 300.0

    # --- surrogate-gradient widths (static) --------------------------------
    #: Width of the bucket-β clip, in β units.
    beta_width: float = struct.field(pytree_node=False, default=0.02)
    #: Width of the snow switch ``S > 0`` [kg/m2].
    snow_width: float = struct.field(pytree_node=False, default=1.0)
    #: Width of the melt hinge at the freezing point [K].
    melt_width: float = struct.field(pytree_node=False, default=0.5)

    # --- configuration (static) ------------------------------------------
    #: Where the land albedo comes from: ``"slim"`` (default) the snow-masked
    #: blend of SLIM's ground and snow albedo maps; ``"jsbach"`` jcm's
    #: JSBACH form (:func:`jcm.physics.surface.echam.albedo.land_albedo`) fed
    #: by the snow bucket's masking factor as its snow cover.
    albedo: str = struct.field(pytree_node=False, default="slim")

    @classmethod
    def default(cls) -> "SlimParameters":
        """SLIM's values."""
        return cls()


#: The values of :attr:`SlimParameters.albedo`.
SLIM_ALBEDO_MODES = ("slim", "jsbach")


def check_albedo_mode(params: SlimParameters) -> str:
    """Return ``params.albedo``, refusing a value that is not a mode."""
    if params.albedo not in SLIM_ALBEDO_MODES:
        raise ValueError(f"SlimParameters.albedo must be one of {SLIM_ALBEDO_MODES}, "
                         f"got {params.albedo!r}")
    return params.albedo


# ---------------------------------------------------------------------------
# Soil column
# ---------------------------------------------------------------------------

def soil_layer_geometry(params: SlimParameters):
    """SLIM's soil grid, ``atm2lndType.F90::InitCold``.

    Returns ``(z, dz, zh, max_water)``, each ``(NSOIL,)``: layer depths
    ``z_i = −a·(exp(b·(i − ½)) − 1)`` (negative down, m), thicknesses
    ``dz_1 = ½|z_1 + z_2|``, ``dz_i = ½(z_{i−1} − z_{i+1})``,
    ``dz_N = z_{N−1} − z_N``, bottom-interface depths ``zh_i = zh_{i−1} − dz_i``
    (``zh_1 = −dz_1``) and the freezable water ``300·dz_i`` kg/m2 below the top
    layer (0 in it). The deepest node is at 2.87 m, the column ends at
    3.43 m and the top layer is 1.75 cm thick.
    """
    i = jnp.arange(1, NSOIL + 1, dtype=jnp.result_type(float))
    z = -params.soil_depth_scale * (jnp.exp(params.soil_depth_growth * (i - 0.5)) - 1.0)
    dz_top = 0.5 * jnp.abs(z[0] + z[1])
    dz_mid = 0.5 * (z[:-2] - z[2:])
    dz_bot = z[-2] - z[-1]
    dz = jnp.concatenate([dz_top[None], dz_mid, dz_bot[None]])
    zh = -jnp.cumsum(dz)
    max_water = jnp.concatenate([jnp.zeros(1, dz.dtype),
                                 params.max_freezable_water * dz[1:]])
    return z, dz, zh, max_water


def _col(x, ndim):
    """Reshape a ``(NSOIL,)`` profile to broadcast against ``(NSOIL, *horiz)``."""
    return jnp.reshape(x, (-1,) + (1,) * (ndim - 1))


def soil_thermal_properties(soil_tk, soil_cv, glacier_tk, glacier_cv, glacier, z, zh):
    """``mml_main.F90::soil_thermal_properties``.

    Every layer takes the column's conductivity and heat capacity (the
    glacier's on glacier cells), and the interface conductance below layer
    ``i`` is the harmonic combination over the two half layers::

        tkh_i = tk_i·tk_{i+1}·(z_i − z_{i+1}) /
                (tk_i·(zh_i − z_{i+1}) + tk_{i+1}·(z_i − zh_i))

    Args:
        soil_tk: soil conductivity [W m-1 K-1], per column ``(*horiz)``.
        soil_cv: soil volumetric heat capacity [J m-3 K-1], per column.
        glacier_tk: glacier conductivity, per column.
        glacier_cv: glacier volumetric heat capacity, per column.
        glacier: glacier indicator ``(*horiz)`` (0/1 or bool).
        z: :func:`soil_layer_geometry`'s layer depths ``(NSOIL,)``.
        zh: its interface depths ``(NSOIL,)``.

    Returns:
        ``(tk, cv, tkh)``: ``(NSOIL, *horiz)``, ``(NSOIL, *horiz)`` and the
        interface conductivities ``(NSOIL − 1, *horiz)`` [W m-1 K-1].

    """
    g = jnp.asarray(glacier) > 0.5
    tk1 = jnp.where(g, glacier_tk, soil_tk)
    cv1 = jnp.where(g, glacier_cv, soil_cv)
    ndim = jnp.ndim(tk1) + 1
    ones = jnp.ones((NSOIL,) + jnp.shape(tk1), jnp.result_type(tk1, z))
    tk = ones * tk1
    cv = ones * cv1
    zc, zhc = _col(z, ndim), _col(zh, ndim)
    tkh = (tk[:-1] * tk[1:] * (zc[:-1] - zc[1:])
           / (tk[:-1] * (zhc[:-1] - zc[1:]) + tk[1:] * (zc[:-1] - zhc[:-1])))
    return tk, cv, tkh


def eliminate_soil_column(t_soil, cv, tkh, z, dz, dt):
    """Upward sweep of SLIM's soil tridiagonal (``mml_main.F90`` ~1100-1145).

    With the surface flux ``F`` (net energy into the top layer, positive down)
    left out, the column's implicit conduction (backward Euler in the layer
    increments ``ΔT_i``) is::

        a_i ΔT_{i−1} + b_i ΔT_i + c_i ΔT_{i+1} = d_i       (+ F on row 1)

    ``c_i = −tkh_i/(z_i − z_{i+1})``, ``a_i = c_{i−1}``, ``b_i = cv_i·dz_i/Δt −
    a_i − c_i``, ``d_i`` the explicit conduction divergence. Eliminating from
    the bottom gives ``ΔT_i = f_i − e_i·ΔT_{i−1}`` for ``i ≥ 2`` and leaves the
    top row as ``den·ΔT_1 = num + F``: the top layer sees the rest of the
    column as a capacity ``den`` [W m-2 K-1] and a flux ``num`` [W m-2]. SLIM
    then adds the linearised surface fluxes to that row (its ``−df0`` and
    ``f0``); jcm hands ``den·Δt`` and ``num`` to the implicit surface balance
    as ECHAM hands JSBACH's ``pgrndcapc``/``pgrndhflx`` to it.

    Returns:
        ``(den, num, e, f)``: ``den``/``num`` ``(*horiz)`` and the sweep
        coefficients ``e``/``f`` ``(NSOIL, *horiz)`` (row 0 unused).

    """
    ndim = jnp.ndim(t_soil)
    zc, dzc = _col(z, ndim), _col(dz, ndim)
    dzz = zc[:-1] - zc[1:]                                   # z_i − z_{i+1} > 0
    cc = -tkh / dzz                                          # c_i, i = 1..N−1
    flux = tkh * (t_soil[:-1] - t_soil[1:]) / dzz            # conduction i → i+1
    cap = cv * dzc / dt
    # Bottom layer: no flux through its base.
    a_n = cc[-1]
    e_next = a_n / (cap[-1] - a_n)
    f_next = flux[-1] / (cap[-1] - a_n)
    e = [e_next]
    f = [f_next]
    for i in range(NSOIL - 2, 0, -1):                        # layers N−1 .. 2
        a_i, c_i = cc[i - 1], cc[i]
        b_i = cap[i] - a_i - c_i
        d_i = flux[i - 1] - flux[i]
        den_i = b_i - c_i * e_next
        e_next = a_i / den_i
        f_next = (d_i - c_i * f_next) / den_i
        e.append(e_next)
        f.append(f_next)
    c_1 = cc[0]
    den = cap[0] - c_1 - c_1 * e_next
    num = -flux[0] - c_1 * f_next
    zero = jnp.zeros_like(den)
    e = jnp.stack([zero] + e[::-1])
    f = jnp.stack([zero] + f[::-1])
    return den, num, e, f


def complete_soil_column(t_soil, top_increment, e, f):
    """Downward sweep: ``ΔT_i = f_i − e_i·ΔT_{i−1}`` from the top layer's ``ΔT_1``."""
    increments = [top_increment]
    for i in range(1, NSOIL):
        increments.append(f[i] - e[i] * increments[-1])
    return t_soil + jnp.stack(increments)


def phase_change(t_soil, soil_ice, cv, dz, max_water, dt):
    """``mml_main.F90::phase_change``: freeze or thaw each layer's water.

    Each layer holds a fixed mass of thermodynamic water ``max_water`` split
    into ice and liquid. A layer with ice above the freezing point (melt) or
    liquid below it (freeze) is set to the freezing point; the energy that
    takes, ``hfm = (T_f − T)·cv·dz/Δt``, changes the ice by ``hfm/L_f`` within
    what is there, and the part the water cannot absorb returns to the layer's
    temperature. Energy is conserved: ``Σ cv·dz·ΔT/Δt = L_f·Σ Δice/Δt``.

    Returns:
        ``(t_soil, soil_ice, phase_heat)``: the new temperatures and ice
        ``(NSOIL, *horiz)``, and the column's latent heat release
        ``L_f·Σ Δice/Δt`` [W/m2] (positive when freezing).

    """
    ndim = jnp.ndim(t_soil)
    dzc = _col(dz, ndim)
    wmass = _col(max_water, ndim) * jnp.ones_like(t_soil)
    tf = c.tmelt
    lf = c.alhf
    ice0 = soil_ice
    liq0 = jnp.maximum(wmass - ice0, 0.0)
    melting = (ice0 > 0.0) & (t_soil > tf)
    freezing = (~melting) & (liq0 > 0.0) & (t_soil < tf)
    active = melting | freezing
    hfm = (tf - t_soil) * cv * dzc / dt
    ice = jnp.clip(ice0 + hfm / lf * dt, 0.0, wmass)
    ice = jnp.where(active, ice, ice0)
    alhf = lf * (ice - ice0) / dt
    t_new = jnp.where(active, tf - (hfm - alhf) * dt / (cv * dzc), t_soil)
    return t_new, ice, jnp.sum(alhf, axis=0)


def seed_soil_ice(t_soil, max_water):
    """Seed a soil column's ice at ``t_soil``: frozen below the freezing point."""
    wmass = _col(max_water, jnp.ndim(t_soil)) * jnp.ones_like(t_soil)
    return jnp.where(t_soil < c.tmelt, wmass, 0.0)


# ---------------------------------------------------------------------------
# Evaporation
# ---------------------------------------------------------------------------

def _soft_clip01(x, width):
    """Softplus clip to ``[0, 1]``: ``w·sp(x/w) − w·sp((x − 1)/w)``."""
    return width * (jax.nn.softplus(x / width) - jax.nn.softplus((x - 1.0) / width))


def bucket_beta(water, capacity, params: SlimParameters):
    """SLIM's bucket factor ``β = clip(W/(0.75·cap), 0, 1)`` (``mml_main`` ~1050).

    SLIM takes ``min(·, 1)`` and floors a negative ``β`` at 0. Exact value;
    the derivative is that of a softplus clip of width ``params.beta_width``,
    so a full or empty bucket still sees its water.
    """
    def exact(w, cap, frac):
        return jnp.clip(w / (frac * cap), 0.0, 1.0)

    def surrogate(w, cap, frac):
        return _soft_clip01(w / (frac * cap), params.beta_width)

    return with_surrogate_gradient(exact, surrogate)(
        water, capacity, jnp.asarray(params.bucket_beta_fraction, jnp.result_type(water)))


def snow_indicator(snow, glacier, params: SlimParameters):
    """1 where the land holds snow (``S > 0``) or is glacier, else 0.

    SLIM evaporates a snow-covered surface at ``β = 1``, and a glacier never
    runs out of snow. Exact value; the derivative with respect to ``S`` is that
    of ``σ(S/w)`` with ``w = params.snow_width``.
    """
    g = jnp.asarray(glacier) > 0.5

    def exact(s, g):
        return jnp.where(g | (s > 0.0), 1.0, 0.0).astype(jnp.result_type(s))

    def surrogate(s, g):
        return jnp.where(g, 1.0, jax.nn.sigmoid(s / params.snow_width)).astype(
            jnp.result_type(s))

    return with_surrogate_gradient(exact, surrogate)(snow, g)


def evaporation_factor(beta, snow_on, exchange_velocity, evaporative_resistance,
                       params: SlimParameters):
    """Return the land humidity factor of SLIM's evaporation, ``cair = csat``.

    SLIM's latent heat flux (``mml_main.F90`` ~1050-1070) is
    ``LE = ρ·λ·(q_s − q_a)·β/(r_evap + r_a)``, with ``β = 1`` over snow and the
    total resistance capped at ``resistance_cap``. With ``r_a = 1/(C_h|U|)``
    that is ECHAM's ``E = ρ·C_h|U|·(csat·q_s − cair·q_a)`` with::

        cair = csat = β_eff·r_a/(r_a + r_evap) = β_eff/(1 + C_h|U|·r_evap)

    ``β_eff = s + (1 − s)·β`` (``s`` the :func:`snow_indicator`). The exchange
    velocity is floored at 1e-6 m/s, the port's floor on every surface
    exchange coefficient.
    """
    beta_eff = snow_on + (1.0 - snow_on) * beta
    chu = jnp.maximum(exchange_velocity, 1.0e-6)
    resistance = jnp.minimum(1.0 / chu + evaporative_resistance, params.resistance_cap)
    return beta_eff / (chu * resistance)


def cap_evaporation_factor(factor, potential_rate, available_water, dt):
    """Scale the factor so one step cannot evaporate more than the land holds.

    SLIM limits the latent heat flux to ``λ·(W + S)/Δt`` before its surface
    solve (``mml_main.F90`` ~1072-1078). Here ``potential_rate`` is the
    step-start ``ρ·C_h|U|·(q_s − q_a)`` [kg m-2 s-1]: where ``factor·rate·Δt``
    exceeds ``available_water`` the factor is scaled down to match. Dew
    (``rate < 0``) is never limited. The scale is a ratio of smooth functions
    selected by a comparison, whose one-sided derivative is the true one.
    """
    demand = factor * potential_rate * dt
    avail = jnp.maximum(available_water, 0.0)
    over = demand > avail
    scale = jnp.where(over, avail / jnp.where(over, demand, 1.0), 1.0)
    return factor * scale


# ---------------------------------------------------------------------------
# Snow melt
# ---------------------------------------------------------------------------

def melt_energy(t_new, capacity, snow, glacier, dt, params: SlimParameters):
    """SLIM's snow melt energy [W/m2] (``mml_main.F90`` ~1150-1175).

    ``ptl_snow_melt = max(0, (T − T_f)·den/L_f)`` is the melt rate that would
    bring the provisional top-layer temperature ``T`` back to the freezing
    point, ``den`` the top layer's implicit capacity [W m-2 K-1] (SLIM's
    ``den``; here that of the full implicit system, atmosphere included), and
    the realised rate is ``min(S/Δt, ptl)``. A glacier never runs out of snow,
    so there the melt is the potential one. Returns ``L_f·melt``.

    Exact value; the derivatives are those of the same expression with the
    hinge ``max(0, ΔT)`` a softplus of width ``params.melt_width`` and the
    ``min`` a smooth minimum of the same width in energy units.
    """
    tf = c.tmelt
    lf = c.alhf

    def exact(t, den, s, g, dt):
        potential = jnp.maximum(t - tf, 0.0) * den
        available = jnp.maximum(s, 0.0) * lf / dt
        return jnp.where(g, potential, jnp.minimum(available, potential))

    def surrogate(t, den, s, g, dt):
        w = params.melt_width
        potential = w * jax.nn.softplus((t - tf) / w) * den
        available = jnp.maximum(s, 0.0) * lf / dt
        wd = w * den
        smooth_min = available - wd * jax.nn.softplus((available - potential) / wd)
        return jnp.where(g, potential, smooth_min)

    # The step length is an argument, not a closure: it is traced inside the
    # vertical diffusion's jit.
    return with_surrogate_gradient(exact, surrogate)(
        t_new, capacity, snow, jnp.asarray(glacier) > 0.5, dt)


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------

def update_buckets(water, snow, rain, snowfall, melt, evaporation, capacity, glacier,
                   dt, params: SlimParameters):
    """SLIM's bucket hydrology for one step (``mml_main.F90`` ~1455-1560).

    In SLIM's order: the melt moves from the snow to the water bucket; rain and
    snowfall enter their buckets; evaporation (positive up, dew negative) is
    taken from the snow while there is snow (``wat2snow`` takes the rest from
    the water) and from the water otherwise; water above the bucket capacity
    and snow above ``snow_cap`` run off. A glacier never runs out of snow: the
    melt and sublimation its snow cannot supply come from the ice
    (``glacier_supply``), and its water bucket does not evaporate. SLIM then
    clips a negative bucket to zero; jcm does not, so the water budget closes
    exactly (the evaporation cap keeps any undershoot to the difference
    between the step-start and the implicit flux, and ``β`` floors at 0, which
    stops further evaporation).

    All arguments ``(*horiz)``; the fluxes in kg m-2 s-1 (``melt`` the melt
    rate). Returns a dict with ``water``, ``snow``, ``runoff``,
    ``snow_evaporation``, ``water_evaporation`` and ``glacier_supply`` (rates in
    kg m-2 s-1), satisfying ``ΔW + ΔS = (rain + snowfall − E − runoff +
    glacier_supply)·Δt`` exactly.
    """
    g = jnp.asarray(glacier) > 0.5
    # Melt: from the snow (a glacier supplies what the snow lacks).
    snow1 = snow - melt * dt
    supply_melt = jnp.where(g, jnp.maximum(-snow1, 0.0), 0.0)
    snow1 = snow1 + supply_melt
    water1 = water + melt * dt + rain * dt
    snow1 = snow1 + snowfall * dt
    # Evaporation: from the snow first, the remainder from the water; a
    # glacier sublimates from its snow and then its ice, never the bucket.
    e_dt = evaporation * dt
    from_snow = jnp.where(snow1 > 0.0, jnp.minimum(e_dt, snow1), 0.0)
    from_snow = jnp.where(g, e_dt, from_snow)
    from_water = e_dt - from_snow
    snow2 = snow1 - from_snow
    supply_evap = jnp.where(g, jnp.maximum(-snow2, 0.0), 0.0)
    snow2 = snow2 + supply_evap
    water2 = water1 - from_water
    # Runoff above the capacities.
    runoff_w = jnp.maximum(water2 - capacity, 0.0)
    runoff_s = jnp.maximum(snow2 - params.snow_cap, 0.0)
    return dict(
        water=water2 - runoff_w,
        snow=snow2 - runoff_s,
        runoff=(runoff_w + runoff_s) / dt,
        snow_evaporation=from_snow / dt,
        water_evaporation=from_water / dt,
        glacier_supply=(supply_melt + supply_evap) / dt,
    )


# ---------------------------------------------------------------------------
# Albedo
# ---------------------------------------------------------------------------

def snow_masking_factor(snow, snow_mask_depth):
    """``S/(S + snowmask)`` for ``S > 0``, else 0 (``mml_main.F90`` ~495-517)."""
    s = jnp.maximum(snow, 0.0)
    denom = s + snow_mask_depth
    ok = (s > 0.0) & (denom > 0.0)
    return jnp.where(ok, s / jnp.where(ok, denom, 1.0), 0.0)


def land_albedos(props, snow):
    """SLIM's land albedos, the snow-masked blend of ground and snow.

    ``α_b = (1 − m)·α_ground,b + m·α_snow,b`` for the four streams (visible
    and near-infrared, direct and diffuse), ``m`` the
    :func:`snow_masking_factor` (``mml_main.F90`` ~537-544).

    Args:
        props: mapping of the SLIM property fields for this step
            (``alb_{g,s}{v,n}{d,f}``, ``snowmask``), each ``(*horiz)``.
        snow: the snow bucket [kg/m2].

    Returns:
        ``(vis_dir, nir_dir, vis_dif, nir_dif)``.

    """
    m = snow_masking_factor(snow, props["snowmask"])

    def blend(ground, snowy):
        return (1.0 - m) * props[ground] + m * props[snowy]

    return (blend("alb_gvd", "alb_svd"), blend("alb_gnd", "alb_snd"),
            blend("alb_gvf", "alb_svf"), blend("alb_gnf", "alb_snf"))
