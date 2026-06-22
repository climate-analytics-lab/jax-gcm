import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem
import jcm.constants as c
from jcm.physics.speedy.physical_constants import compute_sigma_boundaries
from jcm.utils import get_coords

def get_speedy_coords(layers=8, spectral_truncation=31, nodal_shape=None, spmd_mesh=None) -> CoordinateSystem:
    """Create a CoordinateSystem with SPEEDY's standard sigma layers.

    This is a convenience wrapper around jcm.utils.get_coords() that uses
    SPEEDY's sigma layer boundaries. The boundaries come from
    :func:`compute_sigma_boundaries`, which returns the hand-tuned SPEEDY tables
    for ``layers`` in {7, 8} and an analytic Frierson (2006) stretch for any
    other count, so an arbitrary number of vertical levels is supported.

    Args:
        layers: Number of vertical levels (any int >= 2; 7/8 use the exact
            SPEEDY tables, other counts use the Frierson (2006) stretch).
        spectral_truncation: Spectral truncation number (default 31)
        nodal_shape: Optional nodal shape (ix, il) to infer spectral_truncation
        spmd_mesh: Optional ``(x, y, z)`` tuple giving the SPMD device mesh over
            (longitude, latitude, vertical). The product must equal ``len(jax.devices())``.
            Pass this here (not to ``Model``) to enable multi-device sharding.

    Returns:
        CoordinateSystem object with SPEEDY sigma levels

    """
    return get_coords(
        vertical_coords=compute_sigma_boundaries(layers),
        spectral_truncation=spectral_truncation,
        nodal_shape=nodal_shape,
        spmd_mesh=spmd_mesh
    )

def compute_speedy_vertical_coords(kx: int):
        """Compute SPEEDY vertical coordinate transformations.

        The sigma half-level boundaries come from
        :func:`compute_sigma_boundaries`, so any ``kx >= 2`` is supported (7/8
        reproduce the original SPEEDY tables exactly). Everything below is
        derived generically from those boundaries.

        Args:
            kx: Number of vertical levels (>= 2)

        Returns:
            Tuple of (hsg, fsg, dhs, sigl, grdsig, grdscp, wvi)

        """
        # Layer boundaries and midpoints
        hsg = compute_sigma_boundaries(kx)
        fsg = (hsg[1:] + hsg[:-1]) / 2.
        dhs = jnp.diff(hsg)
        sigl = jnp.log(fsg)

        # Conversion factors for fluxes -> tendencies
        grdsig = c.grav / (dhs * c.p0)
        grdscp = grdsig / c.cpd

        # Weights for vertical interpolation at half-levels(1,kx) and surface
        # Note that for phys.par. half-lev(k) is between full-lev k and k+1
        # Fhalf(k) = Ffull(k) + WVI(K,2) * (Ffull(k+1) - Ffull(k))
        # Fsurf = Ffull(kx) + WVI(kx,2) * (Ffull(kx) - Ffull(kx-1))
        wvi = jnp.zeros((kx, 2))
        wvi = wvi.at[:-1, 0].set(1. / jnp.diff(sigl))
        wvi = wvi.at[:-1, 1].set((jnp.log(hsg[1:-1]) - sigl[:-1]) * wvi[:-1, 0])
        wvi = wvi.at[-1, 1].set((jnp.log(0.99) - sigl[-1]) * wvi[-2, 0])

        return hsg, fsg, dhs, sigl, grdsig, grdscp, wvi
@tree_math.struct
class SpeedyCoords:
    """SPEEDY-specific coordinate system data.

    This struct caches precomputed coordinate transformations needed by SPEEDY physics.
    All fields are constant during a simulation.

    Attributes:
        # Vertical coordinate data
        hsg: Sigma layer boundaries (kx+1,)
        fsg: Sigma layer midpoints (kx,)
        dhs: Sigma layer thicknesses (kx,)
        sigl: Log of sigma layer midpoints (kx,)
        grdsig: g/(d_sigma * p0) - converts fluxes to tendencies (kx,)
        grdscp: g/(d_sigma * p0 * c_p) - converts energy fluxes to temperature tendencies (kx,)
        wvi: Weights for vertical interpolation (kx, 2)

        # Horizontal coordinate data
        radang: Latitude in radians (il,)
        sia: Sin of latitude (il,)
        coa: Cos of latitude (il,)

    """

    # Vertical
    hsg: jnp.ndarray
    fsg: jnp.ndarray
    dhs: jnp.ndarray
    sigl: jnp.ndarray
    grdsig: jnp.ndarray
    grdscp: jnp.ndarray
    wvi: jnp.ndarray

    # Horizontal
    radang: jnp.ndarray
    sia: jnp.ndarray
    coa: jnp.ndarray

    def copy(self,
            hsg=None,
            fsg=None,
            dhs=None,
            sigl=None,
            grdsig=None,
            grdscp=None,
            wvi=None,
            radang=None,
            sia=None,
            coa=None):
        """Copy an instance of SpeedyCoords"""
        return SpeedyCoords(
            hsg=hsg if hsg is not None else self.hsg,
            fsg=fsg if fsg is not None else self.fsg,
            dhs=dhs if dhs is not None else self.dhs,
            sigl=sigl if sigl is not None else self.sigl,
            grdsig=grdsig if grdsig is not None else self.grdsig,
            grdscp=grdscp if grdscp is not None else self.grdscp,
            wvi=wvi if wvi is not None else self.wvi,
            radang=radang if radang is not None else self.radang,
            sia=sia if sia is not None else self.sia,
            coa=coa if coa is not None else self.coa
        )

    def xarray_additional_coords(self):
        """Return additional xarray coordinates for SPEEDY physics fields."""
        return {
            'wvi_id': jnp.array([1, 2]),
            'hsg_level': jnp.arange(len(self.hsg)),
        }

    @classmethod
    def single_column_coords(cls, radang=0., num_levels=8):
            """Initialize a speedy_coords instance for a single column model.

            Args:
                radang (optional): Latitude of the single column in radians (default 0).
                num_levels (optional): Number of vertical levels (default 8).

            Returns:
                Speedy coords object

            """
            sia, coa = jnp.sin(radang), jnp.cos(radang)

            # Vertical functions of sigma
            hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = compute_speedy_vertical_coords(num_levels)

            return cls(radang=jnp.array([[radang]]), sia=jnp.array([[sia]]), coa=jnp.array([[coa]]),
                    hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl,
                    grdsig=grdsig, grdscp=grdscp, wvi=wvi)

    @classmethod
    def from_coordinate_system(cls, coords: CoordinateSystem):
        """Create SpeedyCoords from a dinosaur CoordinateSystem.

        This function extracts and transforms coordinate data from a CoordinateSystem
        into the specific form needed by SPEEDY physics parameterizations.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object

        Returns:
            SpeedyCoords struct containing all coordinate transformations

        """
        # Compute vertical coordinates
        kx = coords.nodal_shape[0]
        hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = compute_speedy_vertical_coords(kx)

        # Compute horizontal coordinates
        radang = coords.horizontal.latitudes
        sia = jnp.sin(radang)
        coa = jnp.cos(radang)

        return cls(
            hsg=hsg,
            fsg=fsg,
            dhs=dhs,
            sigl=sigl,
            grdsig=grdsig,
            grdscp=grdscp,
            wvi=wvi,
            radang=radang,
            sia=sia,
            coa=coa
        )