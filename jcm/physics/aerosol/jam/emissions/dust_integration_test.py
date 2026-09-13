"""End-to-end integration of the Tegen dust chain (issue #802).

Drives the whole path the unit tests only touch in pieces: five netCDF fixtures
through the real readers and ``_attach_dust``, into a composed ``echam-jam``
model on a small grid, integrated for a few steps. The assertions are the ones
that only an end-to-end run can make — that a field written to disk reaches the
term on the right columns, that the gates select the cells the data says they
should, and that the MAM4 split the term computes survives into the tracers.

Marked slow: it builds a real ``Model`` and integrates it.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest
import xarray as xr

# A T21 Gaussian grid: 32 latitudes x 64 longitudes.
_NLAT, _NLON = 32, 64
_TRUNC, _NLEV = 21, 20
_MONTHS = np.array([np.datetime64(f"2005-{m:02d}-01") for m in range(1, 13)])


def _grid():
    from jcm.data.regridding import gaussian_latlon
    return gaussian_latlon(_NLAT)


def _write_dust_fixtures(tmp, source_mask, regions_value=4):
    """Write the five products on the model grid; emission only where masked.

    ``source_mask`` is a ``(nlat, nlon)`` boolean: those cells get
    ``pot_source = 1`` in every month and everything else exactly 0, so the
    potential-source gate has an unambiguous expected answer.
    """
    lats, lons = _grid()
    coords2 = {"lat": lats, "lon": lons}
    pot = np.where(source_mask, 1.0, 0.0)
    # A seasonal cycle: month 6 switches the whole field off, so a month
    # boundary is observable end to end.
    monthly = np.broadcast_to(pot, (12, _NLAT, _NLON)).copy()
    monthly[6] = 0.0
    paths = {}
    xr.Dataset({"pot_source": (("time", "lat", "lon"), monthly)},
               coords={**coords2, "time": _MONTHS}).to_netcdf(
        paths.setdefault("dust_file", os.path.join(tmp, "pot.nc")))
    xr.Dataset({"source": (("lat", "lon"), np.where(source_mask, 0.5, 0.0))},
               coords=coords2).to_netcdf(
        paths.setdefault("dust_preferential_file", os.path.join(tmp, "pref.nc")))
    soils = {f"type{i}": (("lat", "lon"),
                          np.full((_NLAT, _NLON), 0.25 if i == 2 else 0.0))
             for i in (2, 3, 4, 6, 13, 14, 15, 16, 17)}
    xr.Dataset(soils, coords=coords2).to_netcdf(
        paths.setdefault("dust_soil_types_file", os.path.join(tmp, "soil.nc")))
    xr.Dataset({"regions": (("lat", "lon"),
                            np.full((_NLAT, _NLON), float(regions_value)))},
               coords=coords2).to_netcdf(
        paths.setdefault("dust_regions_file", os.path.join(tmp, "reg.nc")))
    xr.Dataset({"surfrough": (("time", "lat", "lon"),
                              np.full((12, _NLAT, _NLON), 0.001),
                              {"units": "1."})},
               coords={**coords2, "time": _MONTHS}).to_netcdf(
        paths.setdefault("dust_roughness_file", os.path.join(tmp, "rough.nc")))
    return paths


def _forcing_with_dust(coords, paths):
    """Attach all five products through the real assembly door."""
    from omegaconf import OmegaConf

    from jcm import forcing_assembly as fa
    from jcm.forcing import default_forcing
    cfg = OmegaConf.create({k: str(v) for k, v in paths.items()})
    return fa._attach_dust(default_forcing(coords.horizontal), cfg, coords)


@pytest.mark.slow
class DustEndToEndTest(unittest.TestCase):
    """The five products -> readers -> attach -> composed model -> tracers."""

    def _model(self, physics=None):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, _NLEV + 1),
                            spectral_truncation=_TRUNC)
        terrain = TerrainData.aquaplanet(coords)
        return coords, Model(
            coords=coords, time_step=30, terrain=terrain,
            physics=physics or echam_physics(aerosol_module="jam",
                                             cloud_scheme="2m"))

    def test_emission_is_confined_to_the_gated_cells(self):
        """A field on disk reaches the term on the right columns, and only there.

        The strongest end-to-end statement available: the emission footprint
        must equal the potential-source footprint, because every other gate is
        satisfied uniformly here. It fails if the orientation flip, the column
        ravel or the gate is wrong in any way a per-column unit test cannot see.
        """
        from jcm.physics.aerosol.jam import mass_name

        coords, model = self._model()
        lats, lons = _grid()
        mask = np.zeros((_NLAT, _NLON), dtype=bool)
        mask[_NLAT // 2:_NLAT // 2 + 4, 10:20] = True     # one contiguous patch
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _forcing_with_dust(coords, _write_dust_fixtures(tmp, mask))

        # ``dust_source`` is a WRAP_YEAR TimeSeries; ``select`` collapses it.
        from jcm.date import DateData
        import jax_datetime as jdt
        date = DateData.set_date(jdt.Datetime.from_pydatetime(
            jdt.to_datetime("2005-01-15")))
        sliced = forcing.select(date)
        got = np.asarray(sliced.dust_source)              # (lon, lat)
        self.assertEqual(got.shape, (_NLON, _NLAT))
        np.testing.assert_array_equal(got > 0.0, mask.T)

        # And the term, driven with a wind well above threshold, emits on
        # exactly those columns.
        from jcm.physics.aerosol.jam.emissions.dust import DustEmissions
        import jax.numpy as jnp
        from jcm.physics_interface import PhysicsState
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types \
            import VerticalDiffusionData
        ncols = _NLON * _NLAT
        state = PhysicsState.zeros((_NLEV, ncols)).copy(
            temperature=jnp.full((_NLEV, ncols), 295.0))
        vd = VerticalDiffusionData.zeros((ncols,), _NLEV).copy(
            wind_10m=jnp.full((ncols,), 20.0))
        diagnostics = {"air_density": jnp.full((_NLEV, ncols), 1.2),
                       "layer_thickness": jnp.full((_NLEV, ncols), 100.0),
                       "vertical_diffusion": vd}
        term = DustEmissions()
        term.cache_coords(coords)

        class _F:
            pass
        f = _F()
        for name in ("dust_source", "dust_preferential", "dust_regions",
                     "dust_soil_types", "dust_roughness"):
            setattr(f, name, getattr(sliced, name))
        f.snowc_am = jnp.zeros((ncols,))
        f.soilw_am = jnp.zeros((ncols,))
        tend, _ = term(state, diagnostics, f, None)
        emitted = np.asarray(
            tend.tracers[mass_name("du", "cor")][-1]).reshape(_NLON, _NLAT) > 0.0
        np.testing.assert_array_equal(emitted, mask.T)
        del lats, lons

    def test_month_boundary_switches_the_source_off(self):
        """The WRAP_YEAR climatology steps by month, end to end.

        Month 7 of the fixture is all-zero, so selecting a mid-July date must
        produce no source at all while mid-January produces the patch.
        """
        import jax_datetime as jdt

        from jcm.date import DateData
        from jcm.utils import get_coords
        coords = get_coords(np.linspace(0, 1, _NLEV + 1),
                            spectral_truncation=_TRUNC)
        mask = np.zeros((_NLAT, _NLON), dtype=bool)
        mask[4:8, 4:8] = True
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _forcing_with_dust(coords, _write_dust_fixtures(tmp, mask))
        pick = lambda d: np.asarray(forcing.select(  # noqa: E731
            DateData.set_date(jdt.Datetime.from_pydatetime(
                jdt.to_datetime(d)))).dust_source)
        self.assertGreater(pick("2005-01-15").max(), 0.0)
        self.assertEqual(pick("2005-07-15").max(), 0.0)
        # Back on in August, so this is a month step and not a one-way latch.
        self.assertGreater(pick("2005-08-15").max(), 0.0)

    def test_composed_model_integrates_with_dust_forcing(self):
        """A real JAM model runs with the dust chain wired and stays finite.

        Exercises the term inside ``ComposablePhysics`` under the scan — the
        carry-state path, the diagnostics pytree and the tracer tendencies —
        which the per-call unit tests never reach.
        """
        import jax.numpy as jnp

        from jcm.physics.aerosol.jam import mass_name, number_name
        coords, model = self._model()
        mask = np.ones((_NLAT, _NLON), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _forcing_with_dust(coords, _write_dust_fixtures(tmp, mask))
        preds = model.run(forcing=forcing, save_interval=0.125, total_time=0.125)
        dyn = preds.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        for key in (mass_name("du", "acc"), mass_name("du", "cor"),
                    number_name("acc"), number_name("cor")):
            self.assertIn(key, dyn.tracers)
            self.assertFalse(bool(jnp.any(jnp.isnan(dyn.tracers[key]))))

    def test_mode_split_and_number_reach_the_tracers(self):
        """The online MAM4 split survives into the tendencies.

        Coarse mass exceeds accumulation mass, and each mode's number is
        consistent with its own emitted effective diameter rather than the
        mode's equilibrium geometry — the property the whole O5 mapping exists
        to provide.
        """
        import jax.numpy as jnp

        from jcm.physics.aerosol.jam import mass_name, number_name
        from jcm.physics.aerosol.jam.emissions.dust import DustEmissions
        from jcm.physics.aerosol.jam.emissions.distributors import (
            particle_mean_mass)
        from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
        from jcm.physics_interface import PhysicsState
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types \
            import VerticalDiffusionData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, _NLEV + 1),
                            spectral_truncation=_TRUNC)
        mask = np.ones((_NLAT, _NLON), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            forcing = _forcing_with_dust(coords, _write_dust_fixtures(tmp, mask))
        import jax_datetime as jdt

        from jcm.date import DateData
        sliced = forcing.select(DateData.set_date(
            jdt.Datetime.from_pydatetime(jdt.to_datetime("2005-01-15"))))
        ncols = _NLON * _NLAT
        state = PhysicsState.zeros((_NLEV, ncols)).copy(
            temperature=jnp.full((_NLEV, ncols), 295.0))
        vd = VerticalDiffusionData.zeros((ncols,), _NLEV).copy(
            wind_10m=jnp.full((ncols,), 18.0))
        diagnostics = {"air_density": jnp.full((_NLEV, ncols), 1.2),
                       "layer_thickness": jnp.full((_NLEV, ncols), 100.0),
                       "vertical_diffusion": vd}

        class _F:
            pass
        f = _F()
        for name in ("dust_source", "dust_preferential", "dust_regions",
                     "dust_soil_types", "dust_roughness"):
            setattr(f, name, getattr(sliced, name))
        f.snowc_am = jnp.zeros((ncols,))
        f.soilw_am = jnp.zeros((ncols,))
        term = DustEmissions()
        term.cache_coords(coords)
        tend, _ = term(state, diagnostics, f, None)
        rho_dz = 1.2 * 100.0
        density = MAM4_SPEC.species_props("du").density
        acc = float(np.asarray(tend.tracers[mass_name("du", "acc")][-1, 0]))
        cor = float(np.asarray(tend.tracers[mass_name("du", "cor")][-1, 0]))
        self.assertGreater(cor, acc)
        for short in ("acc", "cor"):
            mass = float(np.asarray(
                tend.tracers[mass_name("du", short)][-1, 0])) * rho_dz
            num = float(np.asarray(
                tend.tracers[number_name(short)][-1, 0])) * rho_dz
            geom = mass / particle_mean_mass(MAM4_SPEC.mode(short), density)
            self.assertGreater(max(num / geom, geom / num), 3.0)


if __name__ == "__main__":
    unittest.main()
