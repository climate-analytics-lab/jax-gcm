import jax.numpy as jnp
from jax import jit
from jax import vmap
from jcm.physical_constants import solc,epssw
from jcm.params import il, ix
# from jcm.geometry import sia,coa


@jit
def get_zonal_average_fields(tyear, sia, coa):
    """
    Calculate zonal average fields including solar radiation, ozone depth, 
    and polar night cooling in the stratosphere using JAX.
    
    Parameters:
    tyear : float
        Time as fraction of year (0-1, 0 = 1 Jan)
    sia : jnp.ndarray
        Sine of latitude array
    coa : jnp.ndarray
        Cosine of latitude array
    solc : float
        Solar constant
    il : int
        Number of latitude zones
    epssw : float
        Ozone absorption constant

    Returns:
    fsol : jnp.ndarray
        Solar radiation at the top
    ozupp : jnp.ndarray
        Ozone depth in upper stratosphere
    ozone : jnp.ndarray
        Ozone concentration in lower stratosphere
    stratz : jnp.ndarray
        Polar night cooling in the stratosphere
    """

    # Alpha = year phase (0 - 2pi, 0 = winter solstice = 22 Dec)
    alpha = 4.0 * jnp.arcsin(1.0) * (tyear + 10.0 / 365.0)
    dalpha = 0.0

    coz1 = jnp.maximum(0.0, jnp.cos(alpha - dalpha))
    coz2 = 1.8

    azen = 1.0
    nzen = 2

    rzen = -jnp.cos(alpha) * 23.45 * jnp.arcsin(1.0) / 90.0

    fs0 = 6.0

    # Solar radiation at the top
    topsr = jnp.zeros(il)
    topsr = solar(tyear)
    
    def compute_fields(sia_j, coa_j, topsr_j):
        flat2 = 1.5 * sia_j ** 2 - 0.5

        # Solar radiation at the top
        fsol_i_j = topsr_j

        # Ozone depth in upper stratosphere
        ozupp_i_j = 0.5 * epssw
        ozone_i_j = 0.4 * epssw * (1.0 + coz1 * sia_j + coz2 * flat2)

        # Zenith angle correction to (downward) absorptivity
        zenit_i_j = 1.0 + azen * (1.0 - (coa_j * jnp.cos(rzen) + sia_j * jnp.sin(rzen))) ** nzen

        # Ozone absorption in upper and lower stratosphere
        ozupp_i_j = fsol_i_j * ozupp_i_j * zenit_i_j
        ozone_i_j = fsol_i_j * ozone_i_j * zenit_i_j

        # Polar night cooling in the stratosphere
        stratz_i_j = jnp.maximum(fs0 - fsol_i_j, 0.0)

        return jnp.full(ix, fsol_i_j), jnp.full(ix, ozupp_i_j), jnp.full(ix, ozone_i_j), jnp.full(ix, zenit_i_j), jnp.full(ix, stratz_i_j)

    vectorized_compute_fields = vmap(compute_fields, in_axes=0, out_axes=1)

    fsol, ozupp, ozone, zenit, stratz = vectorized_compute_fields(sia, coa, topsr)

    return fsol, ozupp, ozone, zenit, stratz

@jit
def solar(tyear):
    """
    Example implementation of the solar subroutine using JAX.

    Parameters:
    tyear : float
        Time as fraction of year (0-1, 0 = 1 Jan)
    solc : float
        Solar constant
    topsr : jnp.ndarray
        Array to hold solar radiation values

    Returns:
    topsr : jnp.ndarray
        Updated array with solar radiation values
    """
    # Example calculation (you may replace this with the actual implementation)
    return jnp.array([  0.        ,  11.7592291 ,  36.17967221,  62.37982749,
        89.00025789, 115.57292167, 141.84417253, 167.63513079,
       192.80057852, 217.21274849, 240.75546083, 263.32066537,
       284.8073584 , 305.12085103, 324.17260453, 341.88024969,
       358.16757826, 372.96491594, 386.20901108, 397.84346693,
       407.81888344, 416.09288126, 422.63043185, 427.40389234,
       430.39312664, 431.58561647, 430.97645518, 428.56841898,
       424.37194347, 418.40514039, 410.69366489, 401.27070803,
       390.17695405, 377.4603214 , 363.17606856, 347.38657514,
       330.16140263, 311.57736317, 291.71876667, 270.67820204,
       248.55772488, 225.47230974, 201.55584544, 176.97731864,
       151.98202326, 127.03117273, 103.47897701,  92.20329543])