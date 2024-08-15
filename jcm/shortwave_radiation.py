import jax.numpy as jnp
from jax import jit

@jit
def get_zonal_average_fields(tyear, sia, coa, solc, il, epssw):
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
    topsr = solar(tyear, 4.0 * solc, topsr)

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
def solar(tyear, solc, topsr):
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
    return topsr.at[:].set(solc * (1 + 0.034 * jnp.cos(2 * jnp.pi * tyear)))