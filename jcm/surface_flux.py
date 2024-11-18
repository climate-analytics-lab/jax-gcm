import jax.numpy as jnp
from jax import jit

# importing custom functions from library
from jcm.physics import PhysicsTendency, PhysicsState
from jcm.physics_data import PhysicsData
from jcm.physical_constants import p0, rgas, cp, alhc, sbc, sigl, wvi, grav, grdsig, grdscp
from jcm.geometry import coa
from jcm.mod_radcon import emisfc
from jcm.humidity import get_qsat, rel_hum_to_spec_hum

# constants for surface fluxes
fwind0 = 0.95 # Ratio of near-sfc wind to lowest-level wind

# Weight for near-sfc temperature extrapolation (0-1) :
# 1 : linear extrapolation from two lowest levels
# 0 : constant potential temperature ( = lowest level)
ftemp0 = 1.0

# Weight for near-sfc specific humidity extrapolation (0-1) :
# 1 : extrap. with constant relative hum. ( = lowest level)
# 0 : constant specific hum. ( = lowest level)
fhum0 = 0.0

cdl = 2.4e-3   # Drag coefficient for momentum over land
cds = 1.0e-3   # Drag coefficient for momentum over sea
chl = 1.2e-3   # Heat exchange coefficient over land
chs = 0.9e-3   # Heat exchange coefficient over sea
vgust = 5.0    # Wind speed for sub-grid-scale gusts
ctday = 1.0e-2 # Daily-cycle correction (dTskin/dSSRad)
dtheta = 3.0   # Potential temp. gradient for stability correction
fstab = 0.67   # Amplitude of stability correction (fraction)
clambda = 7.0  # Heat conductivity in skin-to-root soil layer
clambsn = 7.0  # Heat conductivity in soil for snow cover = 1

lscasym = True   # true : use an asymmetric stability coefficient
lskineb = True   # true : redefine skin temp. from energy balance

hdrag = 2000.0 # Height scale for orographic correction

@jit
def get_surface_fluxes(state: PhysicsState, physics_data: PhysicsData):
    '''

    Parameters
    ----------
    psa : 2D array
        - Normalised surface pressure, physics_data.convection.psa
    ua : 3D array
        - u-wind, state.u_wind
    va : 3D array
        - v-wind, state.v_wind
    ta :  3D array
        - Temperature, state.temperature
    qa : 3D array
        - Specific humidity [g/kg], state.specific_humidity
    rh : 3D array
        - Relative humidity, physics_data.humidity.rh
    phi : 3D array
        - Geopotential, state.geopotential
    phi0 : 2D array
        - Surface geopotential, physics_data.surface_flux.phi0 
    fmask : 2D array
        - Fractional land-sea mask, physics_data.surface_flux.fmask
    tsea : 2D array
        - Sea-surface temperature, physics_data.sea_model.tsea
    rsds : 2D array 
        - Downward flux of short-wave radiation at the surface, physics_data.shortwave_rad.rsds
    rlds : 2D array 
        - Downward flux of long-wave radiation at the surface, physics_data.longwave_rad.rlds
    lfluxland : boolean, physics_data.surface_flux.lfluxland

    '''
    stl_am = physics_data.surface_flux.stl_am
    soilw_am = physics_data.surface_flux.soilw_am
    ix, il, kx = state.temperature.shape

    psa = physics_data.convection.psa
    ua = state.u_wind
    va = state.v_wind
    ta = state.temperature
    qa = state.specific_humidity
    phi = state.geopotential
    fmask = physics_data.surface_flux.fmask

    lfluxland = physics_data.surface_flux.lfluxland
    rsds = physics_data.shortwave_rad.rsds
    rlds = physics_data.longwave_rad.rlds

    rh = physics_data.humidity.rh
    phi0 = physics_data.surface_flux.phi0 # surface geopotentail
    tsea = physics_data.sea_model.tsea

    snowc = physics_data.mod_radcon.snowc
    alb_l = physics_data.mod_radcon.alb_l
    alb_s = physics_data.mod_radcon.alb_s

    forog = set_orog_land_sfc_drag(phi0)

    # Initialize variables
    esbc  = emisfc*sbc
    ghum0 = 1.0 - fhum0

    ustr = jnp.zeros((ix, il, 3))
    vstr = jnp.zeros((ix, il, 3))
    shf = jnp.zeros((ix, il, 3))
    evap = jnp.zeros((ix, il, 3))
    slru = jnp.zeros((ix, il, 3))
    hfluxn = jnp.zeros((ix, il, 2))
    t1 = jnp.zeros((ix, il, 2))
    q1 = jnp.zeros((ix, il, 2))
    t2 = jnp.zeros((ix, il, 2))
    qsat0 = jnp.zeros((ix, il, 2))
    denvvs = jnp.zeros((ix, il, 3))

    ##########################################################
    # Land surface
    ##########################################################

    lfluxland = True # so that jax can trace the function
    if lfluxland: # leaving this here in case we want to implement a better workaround later

        # 1. Extrapolation of wind, temp, hum. and density to the surface

        # 1.1 Wind components
        u0 = fwind0*ua[:, :, kx-1]
        v0 = fwind0*va[:, :, kx-1]

        gtemp0 = 1.0 - ftemp0
        rcp = 1.0/cp 
        nl1 = kx-1

        # substituting the for loop at line 109
        # Temperature difference between lowest level and sfc
        # line 112
        dt1 = wvi[kx-1, 1]*(ta[:, :, kx-1] - ta[:, :, nl1-1])
        
        # Extrapolated temperature using actual lapse rate (0:land, 1:sea)
        # line 115 - 116
        t1 = t1.at[:, :, 0].add(ta[:, :, kx-1] + dt1)
        t1 = t1.at[:, :, 1].set(t1[:, :, 0] - phi0*dt1/(rgas*288.0*sigl[kx-1]))

        # Extrapolated temperature using dry-adiab. lapse rate (0:land, 1:sea)
        # line 119 - 120
        t2 = t2.at[:, :, 1].set(ta[:, :, kx-1] + rcp*phi[:, :, kx-1])
        t2 = t2.at[:, :, 0].set(t2[:, :, 1] - rcp*phi0)

        # lines 124 - 137
        t1 = jnp.where((ta[:, :, kx-1] > ta[:, :, nl1-1])[:, :, jnp.newaxis],
                    ftemp0*t1 + gtemp0*t2,
                    ta[:, :, kx-1][:, :, jnp.newaxis])
        
        t0 = t1[:, :, 1] + fmask * (t1[:, :, 0] - t1[:, :, 1])

        # 1.3 Density * wind speed (including gustiness factor)
        denvvs = denvvs.at[:, :, 0].set((p0*psa/(rgas*t0))*jnp.sqrt(u0**2 + v0**2 + vgust**2))

        # 2. Using Presribed Skin Temperature to Compute Land Surface Fluxes 
        # 2.1 Compensating for non-linearity of Heat/Moisture Fluxes by definig effective skin temperature

        # Vectorized computation using JAX arrays
        tskin = stl_am + ctday * jnp.sqrt(coa) * rsds * (1.0 - alb_l) * psa

        # 2.2 Stability Correlation
        rdth  = fstab / dtheta
        if lscasym: astab = 0.5
        else: astab = 1.0

        dthl = jnp.where(
            tskin > t2[:, :, 0], 
            jnp.minimum(dtheta, tskin - t2[:, :, 0]), 
            jnp.maximum(-dtheta, astab * (tskin - t2[:, :, 0]))
        )

        denvvs = denvvs.at[:, :, 1].set(denvvs[:, :, 0] * (1.0 + dthl * rdth))

        # 2.3 Computing Wind Stress
        cdldv = cdl * denvvs[:, :, 0] * forog
        ustr = ustr.at[:, :, 0].set(-cdldv * ua[:, :, kx-1])
        vstr = vstr.at[:, :, 0].set(-cdldv * va[:, :, kx-1])

        # 2.4 Computing Sensible Heat Flux
        chlcp = chl * cp
        shf = shf.at[:, :, 0].set(chlcp * denvvs[:, :, 1] * (tskin - t1[:, :, 0]))

        # 2.5 Computing Evaporation
        if fhum0 > 0.0:
            q1_val, qsat0_val = rel_hum_to_spec_hum(t1[:, :, 0], psa, 1.0, rh[:, :, kx-1])
            q1 = q1.at[:, :, 0].set(fhum0*q1_val + ghum0*qa[:, :, kx-1])
            qsat0 = qsat0.at[:, :, 0].set(qsat0_val)
        else:
            q1 = q1.at[:, :, 0].set(qa[:, :, kx-1])

        qsat0 = qsat0.at[:, :, 0].set(get_qsat(tskin, psa, 1.0))

        evap = evap.at[:, :, 0].set(chl * denvvs[:, :, 1] *\
                    jnp.maximum(0.0, soilw_am * qsat0[:, :, 0] - q1[:, :, 0]))

        # 3. Computing land-surface energy balance; Adjust skin temperature and heat fluxes
        # 3.1 Emission of lw radiation from the surface and net heat fluxes into land surface
        tsk3 = tskin ** 3.0
        dslr = 4.0 * esbc * tsk3
        slru = slru.at[:, :, 0].set(esbc * tsk3 * tskin)

        hfluxn = hfluxn.at[:, :, 0].set(
                        rsds * (1.0 - alb_l) + rlds -\
                            (slru[:, :, 0] + shf[:, :, 0] + (alhc * evap[:, :, 0]))
                    )

        # 3.2 Re-definition of skin temperature from energy balance
        if lskineb:
            # Compute net heat flux including flux into ground
            clamb = clambda + (snowc * (clambsn - clambda))
            hfluxn = hfluxn.at[:, :, 0].set(hfluxn[:, :, 0] - (clamb * (tskin - stl_am)))
            dtskin = tskin + 1.0

            # Compute d(Evap) for a 1-degree increment of Tskin
            qsat0 = qsat0.at[:, :, 1].set(get_qsat(dtskin, psa, 1.0))
            qsat0 = qsat0.at[:, :, 1].set(
                    jnp.where(
                        evap[:, :, 0] > 0.0,
                        soilw_am * (qsat0[:, :, 1] - qsat0[:, :, 0]),
                        0.0
                    )
                )

            # Redefine skin temperature to balance the heat budget
            dtskin = hfluxn[:, :, 0] / (clamb + dslr + (chl * denvvs[:, :, 1] * (cp + (alhc * qsat0[:, :, 1]))))
            tskin = tskin + dtskin

            # Add linear corrections to heat fluxes
            shf = shf.at[:, :, 0].set(shf[:, :, 0] + chlcp*denvvs[:, :, 1]*dtskin)
            evap = evap.at[:, :, 0].set(evap[:, :, 0] + chl*denvvs[:, :, 1]*qsat0[:, :, 1]*dtskin)
            slru = slru.at[:, :, 0].set(slru[:, :, 0] + dslr*dtskin)
            hfluxn = hfluxn.at[:, :, 0].set(clamb*(tskin - stl_am))

        dths = jnp.where(
            tsea > t2[:, :, 1],
            jnp.minimum(dtheta, tsea - t2[:, :, 1]),
            jnp.maximum(-dtheta, astab * (tsea - t2[:, :, 1]))
        )
        
        denvvs = denvvs.at[:, :, 2].set(denvvs[:, :, 0] * (1.0 + dths * rdth))

        if fhum0 > 0.0:
            q1_val, qsat0_val = rel_hum_to_spec_hum(t1[:, :, 1], psa, 1.0, rh[:, :, kx-1])
            q1 = q1.at[:, :, 1].set(fhum0*q1_val + ghum0*qa[:, :, kx-1])
            qsat0 = qsat0.at[:, :, 1].set(qsat0_val)
        else:
            q1 = q1.at[:, :, 1].set(qa[:, :, kx-1])

        # 4.2 Wind Stress
        ks = 2

        cdsdv = cds * denvvs[:, :, ks]
        ustr = ustr.at[:, :, 1].set(-cdsdv * ua[:, :, kx-1])
        vstr = vstr.at[:, :, 1].set(-cdsdv * va[:, :, kx-1])

    ##########################################################
    # Sea Surface
    ##########################################################

    ks = 2

    # 4.3 Sensible heat flux
    shf = shf.at[:, :, 1].set(chs * cp * denvvs[:, :, ks] * (tsea - t1[:, :, 1]))

    # 4.4 Evaporation
    qsat0 = qsat0.at[:, :, 1].set(get_qsat(tsea, psa, 1.0))
    evap = evap.at[:, :, 1].set(chs * denvvs[:, :, ks] * (qsat0[:, :, 1] - q1[:, :, 1]))
    
    # 4.5 Lw emission and net heat fluxes
    slru = slru.at[:, :, 1].set(esbc * (tsea ** 4.0))
    hfluxn = hfluxn.at[:, :, 1].set(rsds * (1.0 - alb_s) + rlds - slru[:, :, 1] + shf[:, :, 1] + alhc * evap[:, :, 1])

    # Weighted average of surface fluxes and temperatures according to land-sea mask
    if lfluxland:
        weighted_average = lambda var: var[:, :, 1] + fmask * (var[:, :, 0] - var[:, :, 1])
        ustr = ustr.at[:, :, 2].set(weighted_average(ustr))
        vstr = vstr.at[:, :, 2].set(weighted_average(vstr))
        shf = shf.at[:, :, 2].set(weighted_average(shf))
        evap = evap.at[:, :, 2].set(weighted_average(evap))
        slru = slru.at[:, :, 2].set(weighted_average(slru))

        t0 = weighted_average(t1)

        tsfc  = tsea + fmask * (stl_am - tsea)
        tskin = tsea + fmask * (tskin  - tsea)
    
    surface_flux_out = physics_data.surface_flux.copy(ustr=ustr, vstr=vstr, shf=shf, evap=evap, slru=slru, hfluxn=hfluxn, tsfc=tsfc, tskin=tskin, u0=u0, v0=v0, t0=t0)
    physics_data = physics_data.copy(surface_flux=surface_flux_out)

    # Compute tendencies due to surface fluxes (physics.f90:197-205)
    rps = 1.0 / physics_data.convection.psa
    utend = jnp.zeros_like(state.u_wind).at[:,:,-1].add(ustr[:,:,2]*rps*grdsig[-1])
    vtend = jnp.zeros_like(state.v_wind).at[:,:,-1].add(vstr[:,:,2]*rps*grdsig[-1])
    ttend = jnp.zeros_like(state.temperature).at[:,:,-1].add(shf[:,:,2]*rps*grdscp[-1])
    qtend = jnp.zeros_like(state.specific_humidity).at[:,:,-1].add(evap[:,:,2]*rps*grdsig[-1])
    physics_tendencies = PhysicsTendency(utend, vtend, ttend, qtend)

    return physics_tendencies, physics_data

@jit
def set_orog_land_sfc_drag(phi0):
    '''
    Parameters
    ----------
    phi0 : Array
        - Array used for calculating the forog
    '''

    rhdrag = 1/(grav*hdrag)

    forog = 1.0 + rhdrag*(1.0 - jnp.exp(-jnp.maximum(phi0, 0.0)*rhdrag))

    return forog

    

    

