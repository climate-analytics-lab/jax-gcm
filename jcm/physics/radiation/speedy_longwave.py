import jax
from jax import jit
import jax.numpy as jnp
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
import jcm.constants as c
# sbc (Stefan-Boltzmann) is a shared constant, read as a module attribute
# from jcm.constants.
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.speedy_coords import stratosphere_mask

nband = 4

@jit
def get_downward_longwave_rad_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Calculate the downward longwave radiation fluxes
    
    Args:
        ta: Absolute temperature - state.temperature
        fband: Energy fraction emitted in each LW band = f(T) - modradcon.fband
        st4a: Blackbody emission from full and half atmospheric levels - modradcon.st4a
        flux: Radiative flux in different spectral bands - modradcon.flux

    Returns:
        rlds: Downward flux of long-wave radiation at the surface
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer

    """
    kx, ix, il = state.temperature.shape
    ta = state.temperature
    st4a = physics_data.mod_radcon.st4a
    flux = physics_data.mod_radcon.flux
    tau2 = physics_data.mod_radcon.tau2

    nl1 = kx - 1

    # 1. Blackbody emission from atmospheric levels.
    # The linearized gradient of the blakbody emission is computed
    # from temperatures at layer forcing, which are interpolated
    # assuming a linear dependence of T on log_sigma.
    # Above the first (top) level, the atmosphere is assumed isothermal.
    
    # Identify the stratosphere by sigma threshold (sigma<0.2) instead of by the
    # original hardcoded top-two-levels. fsg is static so strat_mask is a
    # compile-time-constant boolean array -> fully differentiable, no jnp.where
    # on traced state needed.
    #
    # We use a STATIC sigma threshold (not SpeedyWeather's UniformCooling
    # T<207.5 K state-dependent threshold) because SPEEDY's stratosphere
    # treatment here is structural — it always treats the top of the column as
    # isothermal/blackbody by grid position, independent of the actual
    # temperature. The sigma mask is the faithful nlev-scaling of that intent and
    # keeps the blackbody construction differentiable. (See speedy_coords for the
    # shared threshold.) This deviates from SpeedyWeather, which replaces this
    # whole scheme with a T-relaxation; the deviation is intentional.
    strat_mask = stratosphere_mask(physics_data.speedy_coords.fsg)  # (kx,)
    strat3 = strat_mask[:, jnp.newaxis, jnp.newaxis]
    # The topmost stratospheric layer (k=0) is treated isothermal above TOA;
    # all other stratospheric layers are interior (boundaries above and below).
    is_top = jnp.zeros((kx,), dtype=bool).at[0].set(True)

    # Temperature at level boundaries
    st4a = st4a.at[:nl1,:,:,0].set(ta[:nl1]+physics_data.speedy_coords.wvi[:nl1,1,jnp.newaxis,jnp.newaxis]*(ta[1:nl1+1]-ta[:nl1]))

    # Mean temperature in stratospheric layers.
    # Topmost layer: isothermal above, mean T = 0.75*T + 0.25*T_boundary_below.
    # Interior stratospheric layer: mean T = 0.50*T + 0.25*(T_bound_above +
    # T_bound_below). T_bound_above is the boundary temperature of the layer
    # above (st4a[k-1,...,0]); for k=0 there is none. These reduce exactly to
    # SPEEDY's original k=0 and k=1 formulas at nlev=8.
    tbound_above = jnp.concatenate([st4a[:1, :, :, 0], st4a[:-1, :, :, 0]], axis=0)
    tmean_top = 0.75 * ta + 0.25 * st4a[:, :, :, 0]
    tmean_interior = 0.50 * ta + 0.25 * (tbound_above + st4a[:, :, :, 0])
    strat_tmean = jnp.where(is_top[:, jnp.newaxis, jnp.newaxis], tmean_top, tmean_interior)
    st4a = st4a.at[:, :, :, 1].set(jnp.where(strat3, strat_tmean, st4a[:, :, :, 1]))

    # Temperature gradient in tropospheric layers (sigma>=0.2). The lowest layer
    # (PBL) uses the full-level temperature against the boundary above it; other
    # tropospheric layers use the boundary difference. SPEEDY hardcoded these as
    # k=2..kx-2 and k=kx-1; we select them with ~strat_mask and the PBL index.
    anis = 1
    trop_grad = 0.5 * anis * jnp.maximum(st4a[:, :, :, 0] - tbound_above, 0.0)
    pbl_grad = anis * jnp.maximum(ta - tbound_above, 0.0)
    is_pbl = jnp.zeros((kx,), dtype=bool).at[kx - 1].set(True)
    trop_mask = (~strat_mask) & (~is_pbl)
    st4a = st4a.at[:, :, :, 1].set(jnp.where(trop_mask[:, jnp.newaxis, jnp.newaxis], trop_grad, st4a[:, :, :, 1]))
    st4a = st4a.at[:, :, :, 1].set(jnp.where(is_pbl[:, jnp.newaxis, jnp.newaxis], pbl_grad, st4a[:, :, :, 1]))

    # Blackbody emission in the stratosphere: emit at the mean temperature, and
    # zero the gradient-correction term there.
    strat_bb = c.sbc * st4a[:, :, :, 1] ** 4.0
    st4a = st4a.at[:, :, :, 0].set(jnp.where(strat3, strat_bb, st4a[:, :, :, 0]))
    st4a = st4a.at[:, :, :, 1].set(jnp.where(strat3, 0.0, st4a[:, :, :, 1]))

    # Blackbody emission in the troposphere (sigma>=0.2).
    trop3 = (~strat_mask)[:, jnp.newaxis, jnp.newaxis]
    st3a = c.sbc * ta ** 3.0
    st4a = st4a.at[:, :, :, 0].set(jnp.where(trop3, st3a * ta, st4a[:, :, :, 0]))
    st4a = st4a.at[:, :, :, 1].set(jnp.where(trop3, 4.0 * st3a * st4a[:, :, :, 1], st4a[:, :, :, 1]))

    # 2. Initialization of fluxes
    rlds = jnp.zeros((ix, il))
    dfabs = jnp.zeros((kx, ix, il))

    # 3. Emission and absorption of longwave downward flux.
    #    For downward emission, a correction term depending on the
    #    local temperature gradient and on the layer transmissivity is
    #    added to the average (full-level) emission of each layer.
    
    # 3.1 Stratosphere
    k = 0
    emis = 1 - tau2
    brad = radset(ta, parameters.mod_radcon.epslw) * (st4a[:,:,:,0,jnp.newaxis] + emis*st4a[:,:,:,1,jnp.newaxis])
    emis_brad = emis * brad
    flux = emis_brad[k].at[:,:,2:nband].set(0.0)
    dfabs = dfabs.at[k].add(-jnp.sum(flux,axis=-1))

    # 3.2 Troposphere
    _flux_3d = jnp.zeros((kx, ix, il, nband)).at[0].set(flux)
    _flux_3d = _flux_3d.at[1:].set(jax.lax.scan(
        jax.checkpoint(lambda carry, k: 2*(tau2[k] * carry + emis_brad[k],)),
        flux,
        jnp.arange(1, kx) # scan from TOA to surface
    )[1])
    flux = _flux_3d[-1]
    
    dfabs = dfabs.at[1:].add(-jnp.diff(jnp.sum(_flux_3d, axis=-1), axis=0))

    rlds = jnp.sum(parameters.mod_radcon.emisfc * flux, axis=-1)

    corlw = parameters.mod_radcon.epslw * parameters.mod_radcon.emisfc * st4a[kx-1,:,:,0]
    dfabs = dfabs.at[-1].add(-corlw)
    rlds += corlw

    surface_flux_out = physics_data.surface_flux.copy(rlds=rlds)
    longwave_out = physics_data.longwave_rad.copy(dfabs=dfabs)
    mod_radcon_out = physics_data.mod_radcon.copy(st4a=st4a)
    physics_data = physics_data.copy(
        surface_flux=surface_flux_out, longwave_rad=longwave_out, mod_radcon=mod_radcon_out
    )
    physics_tendencies = PhysicsTendency(
        jnp.zeros_like(state.u_wind),
        jnp.zeros_like(state.v_wind),
        jnp.zeros_like(state.temperature),
        jnp.zeros_like(state.temperature)
    )

    return physics_tendencies, physics_data

@jit
def get_upward_longwave_rad_fluxes(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """Calculate the upward longwave radiation fluxes
    
    Args:
        ta: Absolute temperature
        ts: Surface temperature - surface_fluxes.tsfc
        rlds: Downward flux of long-wave radiation at the surface
        rlus: Surface blackbody emission - taken from rlus from surface fluxes
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
        st4a: Blackbody emission from full and half atmospheric levels - mod_radcon.st4a
    
    Returns:
        fsfc: Net upward flux of long-wave radiation at the surface
        ftop: Outgoing flux of long-wave radiation at the top of the atmosphere
        dfabs: Flux of long-wave radiation absorbed in each atmospheric layer
        st4a: Blackbody emission from full and half atmospheric levels - mod_radcon.st4a

    """
    kx, ix, il = state.temperature.shape
    ta = state.temperature
    dfabs = physics_data.longwave_rad.dfabs
    rlds = physics_data.surface_flux.rlds

    st4a = physics_data.mod_radcon.st4a
    flux = physics_data.mod_radcon.flux
    tau2 = physics_data.mod_radcon.tau2
    stratc = physics_data.mod_radcon.stratc

    rlus = physics_data.surface_flux.rlus
    ts = physics_data.surface_flux.tsfc # called tsfc in surface_fluxes.f90
    refsfc = 1.0 - parameters.mod_radcon.emisfc
    epslw = parameters.mod_radcon.epslw
    fsfc = rlus - rlds
    
    flux = radset(ts, epslw) * rlus[:,:,jnp.newaxis] + refsfc * flux

    # Troposphere
    # correction for 'black' band
    dfabs = dfabs.at[-1].add(parameters.mod_radcon.epslw * rlus)

    emis = 1. - tau2
    brad = radset(ta, epslw) * (st4a[:,:,:,0,jnp.newaxis] - emis*st4a[:,:,:,1,jnp.newaxis])
    emis_brad = emis * brad

    _flux_3d = jnp.zeros((kx, ix, il, nband)).at[-1].set(flux)
    _flux_3d = _flux_3d.at[:-1].set(jax.lax.scan(
        jax.checkpoint(lambda carry, k: 2*(tau2[k] * carry + emis_brad[k],)),
        flux,
        jnp.arange(kx-1, 0, -1) # scan from surface to TOA
    )[1][::-1])
    flux = _flux_3d[0]

    dfabs = dfabs.at[1:].add(jnp.diff(jnp.sum(_flux_3d, axis=-1), axis=0))

    flux = flux.at[:,:,:2].set((tau2[0] * flux + emis_brad[0])[:,:,:2])
    dfabs = dfabs.at[0].add(jnp.sum((_flux_3d[0] - flux)[:,:,:2], axis=-1))

    # Stratospheric cooling correction. SPEEDY applied this to exactly the top
    # two layers (corlw1 at k=0, corlw2 at k=1), with the polar-night term
    # stratc[:,:,0] only at k=0. We apply the per-layer eps1-based cooling
    # dhs[k]*stratc[:,:,1]*st4a[k,0] to every stratospheric layer (sigma<0.2)
    # and keep the polar-night term on the single topmost layer. stratc[:,:,1]
    # already carries eps1 = epslw/sum(dhs over strat) (set in the shortwave
    # routine), so the column-integrated cooling is preserved. strat_mask/dhs
    # are static; ftop is the column sum of the corrections (TOA outgoing).
    strat_mask = stratosphere_mask(physics_data.speedy_coords.fsg)
    dhs = physics_data.speedy_coords.dhs
    corlw = dhs[:, jnp.newaxis, jnp.newaxis] * stratc[:, :, 1][jnp.newaxis] * st4a[:, :, :, 0]
    corlw = jnp.where(strat_mask[:, jnp.newaxis, jnp.newaxis], corlw, 0.0)
    corlw = corlw.at[0].add(stratc[:, :, 0])  # polar-night cooling at the top layer
    dfabs = dfabs - corlw
    ftop = jnp.sum(corlw, axis=0)

    ftop += jnp.sum(flux, axis = -1)

    surface_flux_out = physics_data.surface_flux.copy(rlns=fsfc)
    longwave_out = physics_data.longwave_rad.copy(ftop=ftop, dfabs=dfabs)
    mod_radcon_out = physics_data.mod_radcon.copy(st4a=st4a, flux=flux)
    physics_data = physics_data.copy(
        surface_flux=surface_flux_out, longwave_rad=longwave_out, mod_radcon=mod_radcon_out
    )
    
    # Compute temperature tendency due to absorbed lw flux: logic from physics.f90:182-184
    ttend_lwr = dfabs * physics_data.speedy_coords.grdscp[:, jnp.newaxis, jnp.newaxis] / state.normalized_surface_pressure
    physics_tendencies = PhysicsTendency.zeros(shape=state.temperature.shape,temperature=ttend_lwr)
    
    return physics_tendencies, physics_data

@jit
def radset(temp, epslw):
    """Compute energy fractions in longwave bands as a function of temperature

    Args:
        temp: Absolute temperature
        epslw: Longwave emissivity in PBL

    Returns:
        fband: Energy fraction emitted in each LW band

    """
    jtemp = jnp.clip(temp, 200, 320) # To retain backwards compatibility with F90 code

    fband_123 = jnp.stack((
        0.148 - 3.0e-6 * (jtemp - 247) ** 2,
        0.356 - 5.2e-6 * (jtemp - 282) ** 2,
        0.314 + 1.0e-5 * (jtemp - 315) ** 2,
    ), axis=-1)
    fband_0 = 1. - fband_123.sum(axis=-1)

    fband = jnp.concatenate([
        fband_0[..., jnp.newaxis],
        fband_123
    ], axis=-1)
    
    return (1. - epslw) * fband