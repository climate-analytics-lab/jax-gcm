import jax.numpy as jnp
from jax import jit
from jcm.physical_constants import solc,epssw
from jcm.params import il
from jcm.geometry import sia,coa


@jit
def get_zonal_average_fields(tyear):
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

    # Initialize arrays
    fsol = jnp.zeros((il,))
    ozupp = jnp.zeros((il,))
    ozone = jnp.zeros((il,))
    zenit = jnp.zeros((il,))
    stratz = jnp.zeros((il,))

    def compute_fields(j, fsol, ozupp, ozone, zenit, stratz):
        flat2 = 1.5 * sia[j] ** 2 - 0.5

        # Solar radiation at the top
        fsol = fsol.at[j].set(topsr[j])

        # Ozone depth in upper stratosphere
        ozupp = ozupp.at[j].set(0.5 * epssw)
        ozone = ozone.at[j].set(0.4 * epssw * (1.0 + coz1 * sia[j] + coz2 * flat2))

        # Zenith angle correction to (downward) absorptivity
        zenit = zenit.at[j].set(1.0 + azen * (1.0 - (coa[j] * jnp.cos(rzen) + sia[j] * jnp.sin(rzen))) ** nzen)

        # Ozone absorption in upper and lower stratosphere
        ozupp = ozupp.at[j].set(fsol[j] * ozupp[j] * zenit[j])
        ozone = ozone.at[j].set(fsol[j] * ozone[j] * zenit[j])

        # Polar night cooling in the stratosphere
        stratz = stratz.at[j].set(jnp.maximum(fs0 - fsol[j], 0.0))

        return fsol, ozupp, ozone, zenit, stratz

    for j in range(il):
        fsol, ozupp, ozone, zenit, stratz = compute_fields(j, fsol, ozupp, ozone, zenit, stratz)

    return fsol, ozupp, ozone, stratz

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