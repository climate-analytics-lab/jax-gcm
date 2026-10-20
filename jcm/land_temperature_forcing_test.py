"""Dated control-run land-temperature input, without external data downloads."""

from types import SimpleNamespace

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from omegaconf import OmegaConf
import pytest
import xarray as xr

from jcm.date import DateData
from jcm.forcing import (ForcingData, check_forcing_coverage,
                         read_land_skin_temperature)
from jcm.forcing_assembly import _attach_land_skin_temperature
from jcm.physics.surface.echam.jsbach_land import JsbachLandParameters
from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion import (
    TteTkeVerticalDiffusion, forcing_land_temperature,
)


def _archive():
    times = np.array(['2001-01-01T00', '2001-01-01T03', '2001-01-01T06'],
                     dtype='datetime64[h]')
    # Each column is distinct; reversing latitudes must preserve its identity.
    values = 280. + np.arange(18).reshape(3, 2, 3)
    return xr.Dataset({'land_skin_temperature': (
        ('time', 'lat', 'lon'), values, {'units': 'K'})},
        coords={'time': times, 'lat': [30., -30.], 'lon': [0., 120., 240.]})


def _date(text):
    return DateData.set_date(jdt.Datetime.from_isoformat(text))


def test_sort_orientation_and_subdaily_interpolation():
    ds = _archive().isel(time=[2, 0, 1])
    series = read_land_skin_temperature(ds, [-30., 30.], [0., 120., 240.])
    forcing = ForcingData.zeros((3, 2), land_skin_temperature=series)
    selected = forcing.select(_date('2001-01-01T01:30:00'))
    expected = _archive().land_skin_temperature.mean('time').values - 3.
    np.testing.assert_allclose(selected.land_skin_temperature, expected[::-1].T)
    np.testing.assert_allclose(forcing_land_temperature(selected), expected[::-1].T)


def test_copy_and_legacy_fallback():
    base = ForcingData.zeros((3, 2))
    assert base.land_skin_temperature is None
    np.testing.assert_array_equal(forcing_land_temperature(base), base.stl_am)
    custom = base.copy(land_skin_temperature=jnp.full((3, 2), 300.))
    np.testing.assert_array_equal(custom.copy().land_skin_temperature, 300.)
    assert custom.copy(land_skin_temperature=None).land_skin_temperature is None


@pytest.mark.parametrize('problem', ['units', 'duplicate', 'numeric_time', 'nan', 'grid'])
def test_invalid_archives_fail_before_physics(problem):
    ds = _archive()
    if problem == 'units':
        ds.land_skin_temperature.attrs['units'] = 'degC'
    elif problem == 'duplicate':
        ds = ds.isel(time=[0, 0, 2])
    elif problem == 'numeric_time':
        ds = ds.assign_coords(time=[0., 3., 6.])
    elif problem == 'nan':
        ds.land_skin_temperature.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        read_land_skin_temperature(ds, [-40., 40.] if problem == 'grid' else None)


def test_archive_does_not_replay_in_another_year():
    series = read_land_skin_temperature(_archive())
    forcing = ForcingData.zeros((3, 2), land_skin_temperature=series)
    seconds = float(np.datetime64('2002-01-01', 's').astype(np.int64))
    with pytest.raises(ValueError, match='land_skin_temperature'):
        check_forcing_coverage(forcing, (seconds, seconds + 3600.))


def test_config_attachment_uses_the_same_reader(tmp_path):
    path = tmp_path / 'skin.nc'
    _archive().to_netcdf(path)
    coords = SimpleNamespace(horizontal=SimpleNamespace(
        latitudes=np.deg2rad([-30., 30.]), longitudes=np.deg2rad([0., 120., 240.])))
    cfg = OmegaConf.create({'land_skin_temperature_file': str(path)})
    attached = _attach_land_skin_temperature(ForcingData.zeros((3, 2)), cfg, coords)
    selected = attached.select(_date('2001-01-01T03:00:00'))
    expected = _archive().land_skin_temperature.values[1, ::-1, :].T
    np.testing.assert_allclose(selected.land_skin_temperature, expected)


def test_archive_requires_prescribed_land_mode():
    forcing = ForcingData.zeros((3, 2), land_skin_temperature=jnp.full((3, 2), 290.))
    with pytest.raises(ValueError, match="land_temperature='prescribed'"):
        TteTkeVerticalDiffusion().validate_forcing(forcing)
    term = TteTkeVerticalDiffusion(land_params=JsbachLandParameters(land_temperature='prescribed'))
    term.validate_forcing(forcing)


def test_snapshot_conversion_round_trip(tmp_path):
    from tools.build_land_skin_temperature import build_archive

    source = tmp_path / 'snapshots.nc'
    output = tmp_path / 'land_skin.nc'
    ds = _archive().rename({'land_skin_temperature': 'surface.land_surface_temperature'})
    ds.attrs['jcm_prov_code'] = 'control provenance'
    ds.to_netcdf(source)
    build_archive([source], output, control_sha='abc123')
    with xr.open_dataset(output) as archive:
        assert archive.attrs['control_sha'] == 'abc123'
        assert archive.attrs['coverage_days'] == .25
        assert 'control provenance' in archive.attrs['source_snapshots']
        series = read_land_skin_temperature(archive)
    forcing = ForcingData.zeros((3, 2), land_skin_temperature=series)
    np.testing.assert_allclose(forcing.select(_date('2001-01-01T03:00:00')).land_skin_temperature,
                               _archive().land_skin_temperature.values[1, ::-1, :].T)
