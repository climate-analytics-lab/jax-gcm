"""The ECHAM stack advances the sea-ice surface temperature (``s_sicetemp``)."""
from datetime import datetime

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.date import DateData
from jcm.forcing import ForcingData
from jcm.physics.coords_util import column_lat_lon
from jcm.physics.echam.testing import idealized_echam_physics
from jcm.physics.surface.echam.albedo import CTFREEZ
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData
from jcm.utils import get_coords


_COORDS = {}


def ice_shape(nlat, nlon):
    return (nlat, nlon)


def physics_coords(shape):
    return _COORDS[shape].horizontal


def _setup(nlev=8, nlat=64, nlon=32):
    state = PhysicsState(
        u_wind=jnp.ones((nlev, nlat, nlon)) * 5.0,
        v_wind=jnp.zeros((nlev, nlat, nlon)),
        temperature=jnp.ones((nlev, nlat, nlon)) * 255.0,
        specific_humidity=jnp.ones((nlev, nlat, nlon)) * 5e-4,
        geopotential=jnp.linspace(1e5, 1e3, nlev)[:, None, None] * jnp.ones((nlev, nlat, nlon)),
        normalized_surface_pressure=jnp.ones((nlat, nlon)),
        tracers={"qc": jnp.zeros((nlev, nlat, nlon)), "qi": jnp.zeros((nlev, nlat, nlon))},
    )
    coords = get_coords(np.linspace(0, 1, nlev + 1), nodal_shape=(nlat, nlon))
    physics = idealized_echam_physics()
    physics.cache_coords(coords)
    _COORDS[ice_shape(nlat, nlon)] = coords
    terrain = TerrainData.aquaplanet(coords)
    ice = jnp.zeros((nlat, nlon)).at[:8, :].set(0.9).at[-8:, :].set(0.9)
    forcing = ForcingData.zeros(
        (nlat, nlon), sea_surface_temperature=jnp.full((nlat, nlon), 271.0), sice_am=ice,
    ).select(DateData.set_date(jdt.Datetime.from_pydatetime(datetime(2020, 1, 15))))
    return physics, state, forcing, terrain, ice


def test_cold_air_cools_the_ice_surface_below_the_freezing_seed():
    physics, state, forcing, terrain, ice = _setup()
    _, carry = physics.compute_tendencies(state, forcing=forcing, terrain=terrain)
    surf = carry["surface"]
    t_ice = np.asarray(surf.ice_surface_temperature).reshape(ice.shape)
    iced = np.asarray(ice) > 0
    assert np.all(np.isfinite(t_ice))
    # 255 K air over ice seeded at min(SST, ctfreez) = 271 K: the surface
    # loses energy, so it cools below the seed, by one step's worth only.
    assert np.all(t_ice[iced] < 271.0)
    assert np.all(t_ice[iced] > 250.0)
    # Without ice the field follows the SST seed and the balance is zero.
    np.testing.assert_allclose(t_ice[~iced], 271.0)
    cond = np.asarray(surf.ice_conductive_heat_flux).reshape(ice.shape)
    assert np.all(cond[~iced] == 0.0)
    # The conduction flux is alice*(T - ctfreez)/siced, with siced 2 m in the
    # northern and 1 m in the southern hemisphere.
    lat, _ = column_lat_lon(physics_coords(ice.shape))
    north = (np.asarray(lat) > 0).reshape(ice.shape)
    siced = np.where(north, 2.0, 1.0)
    want = 2.1656 * (t_ice - CTFREEZ) / siced
    np.testing.assert_allclose(cond[iced], want[iced], rtol=1e-4, atol=1e-5)


def test_the_carried_value_is_the_next_steps_start():
    physics, state, forcing, terrain, ice = _setup()
    _, carry1 = physics.compute_tendencies(state, forcing=forcing, terrain=terrain)
    _, carry2 = physics.compute_tendencies(
        state, forcing=forcing, terrain=terrain, prev_physics_data=carry1)
    iced = np.asarray(ice) > 0
    t1 = np.asarray(carry1["surface"].ice_surface_temperature).reshape(ice.shape)
    t2 = np.asarray(carry2["surface"].ice_surface_temperature).reshape(ice.shape)
    assert np.all(np.isfinite(t2))
    # Still cooling under the same cold air, continuing from where it was.
    assert np.all(t2[iced] < t1[iced])
