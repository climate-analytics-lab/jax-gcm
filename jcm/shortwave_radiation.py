import jax.numpy as jnp
from jax import jit
from jax import vmap
from jcm.physical_constants import solc,epssw
from jcm.params import il, ix
# from jcm.geometry import sia,coa


sia = jnp.array([-0.99882019, -0.99358201, -0.98417646, -0.97064298, -0.95303822,
       -0.93143612, -0.90592718, -0.87661856, -0.843633  , -0.80710906,
       -0.76719975, -0.72407258, -0.67790836, -0.62890077, -0.57725537,
       -0.52318871, -0.46692774, -0.40870813, -0.34877434, -0.28737777,
       -0.22477564, -0.16123083, -0.09700976, -0.03238179,  0.03238179,
        0.09700976,  0.16123083,  0.22477564,  0.28737777,  0.34877434,
        0.40870813,  0.46692774,  0.52318871,  0.57725537,  0.62890077,
        0.67790836,  0.72407258,  0.76719975,  0.80710906,  0.843633  ,
        0.87661856,  0.90592718,  0.93143612,  0.95303822,  0.97064298,
        0.98417646,  0.99358201,  0.99882019])

coa = jnp.array([0.04856168, 0.11311405, 0.17719114, 0.24052484, 0.30285006,
       0.36390487, 0.42343352, 0.48118592, 0.53692026, 0.59040238,
       0.64140824, 0.68972379, 0.73514642, 0.77748558, 0.81656368,
       0.85221686, 0.88429548, 0.91266515, 0.93720673, 0.95781732,
       0.97441055, 0.98691672, 0.99528343, 0.99947557, 0.99947557,
       0.99528343, 0.98691672, 0.97441055, 0.95781732, 0.93720673,
       0.91266515, 0.88429548, 0.85221686, 0.81656368, 0.77748558,
       0.73514642, 0.68972379, 0.64140824, 0.59040238, 0.53692026,
       0.48118592, 0.42343352, 0.36390487, 0.30285006, 0.24052484,
       0.17719114, 0.11311405, 0.04856168])

@jit
def get_zonal_average_fields(tyear, sia, coa, solc, il, ix, epssw):
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
    ix : int
        Number of vertical layers
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
    topsr = solar(tyear)  # Assuming solar is another JAX function you defined

    # Initialize arrays
    fsol = jnp.zeros((ix, il,))
    ozupp = jnp.zeros((ix, il,))
    ozone = jnp.zeros((ix, il,))
    zenit = jnp.zeros((ix, il,))
    stratz = jnp.zeros((ix, il,))

    def compute_fields(j, fsol, ozupp, ozone, zenit, stratz):
        flat2 = 1.5 * sia[j] ** 2 - 0.5

        # Solar radiation at the top
        fsol = fsol.at[:, j].set(topsr[j])

        # Ozone depth in upper stratosphere
        ozupp = ozupp.at[:, j].set(0.5 * epssw)
        ozone = ozone.at[:, j].set(0.4 * epssw * (1.0 + coz1 * sia[j] + coz2 * flat2))

        # Zenith angle correction to (downward) absorptivity
        zenit = zenit.at[:, j].set(1.0 + azen * (1.0 - (coa[j] * jnp.cos(rzen) + sia[j] * jnp.sin(rzen))) ** nzen)

        # Ozone absorption in upper and lower stratosphere
        ozupp = ozupp.at[:, j].set(fsol[:, j] * ozupp[:, j] * zenit[:, j])
        ozone = ozone.at[:, j].set(fsol[:, j] * ozone[:, j] * zenit[:, j])

        # Polar night cooling in the stratosphere
        stratz = stratz.at[:, j].set(jnp.maximum(fs0 - fsol[:, j], 0.0))

        return fsol, ozupp, ozone, zenit, stratz

    # Use vmap to vectorize the compute_fields function over the latitude index j
    vmap_compute_fields = vmap(compute_fields, in_axes=(0, None, None, None, None, None), out_axes=(None, None, None, None, None))

    # Apply vmap across all latitudes
    fsol, ozupp, ozone, zenit, stratz = vmap_compute_fields(jnp.arange(il), fsol, ozupp, ozone, zenit, stratz)

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