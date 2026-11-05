"""The SLIM land as composed: properties, carry, migration, wiring and a model step."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from jcm.forcing import default_forcing
from jcm.physics.surface.slim import SLIM_LAND_KEY, SlimLand, SlimLandData, SlimParameters
from jcm.physics.surface.slim.properties import MONTHLY_FIELDS, read_slim_properties

_VALUES = dict(alb_gvd=0.1, alb_svd=0.95, alb_gnd=0.25, alb_snd=0.65, alb_gvf=0.1,
               alb_svf=0.95, alb_gnf=0.25, alb_snf=0.65, bucketdepth=150.0, snowmask=50.0,
               evap_res=150.0, roughness=1.0, soil_tk_1d=1.5, soil_cv_1d=2.0e6,
               glc_tk_1d=2.29, glc_cv_1d=2.1e6)


def uniform_properties(shape, **overrides):
    """SLIM properties on a grid of ``shape``, uniform per field."""
    values = {**_VALUES, **overrides}
    props = {k: jnp.full(shape, float(values[k])) for k in MONTHLY_FIELDS}
    props["glc_mask"] = jnp.zeros(shape)
    return props


def _dataset(lat, lon):
    times = np.array([np.datetime64(f"2000-{m:02d}-01") for m in range(1, 13)])
    data = {k: (("time", "lat", "lon"),
                np.full((12, lat.size, lon.size), v, np.float32)
                + np.arange(12, dtype=np.float32)[:, None, None] * (k == "bucketdepth"))
            for k, v in _VALUES.items()}
    data["glc_mask"] = (("lat", "lon"), (lat[:, None] > 60).astype(np.float32)
                        * np.ones(lon.size))
    return xr.Dataset(data, coords={"time": times, "lat": lat, "lon": lon})


class TestProperties:

    def test_reader_steps_by_month_and_orients_to_the_model_grid(self):
        from jcm.date import DateData
        import jax_datetime as jdt
        from jcm.forcing import ForcingData
        lat, lon = np.array([-70.0, 0.0, 70.0]), np.array([0.0, 120.0, 240.0, 300.0])
        props = read_slim_properties(_dataset(lat, lon), lat, lon)
        forcing = ForcingData.zeros((4, 3)).copy(slim_properties=props)
        for date, month in (("2001-03-01T00:00", 2), ("2001-03-31T23:00", 2),
                            ("2001-04-01T00:00", 3)):
            sel = forcing.select(DateData.set_date(jdt.Datetime.from_isoformat(date)))
            np.testing.assert_allclose(sel.slim_properties["bucketdepth"], 150.0 + month)
        np.testing.assert_array_equal(np.asarray(props["glc_mask"])[:, 2], 1.0)
        np.testing.assert_array_equal(np.asarray(props["glc_mask"])[:, :2], 0.0)

    def test_reader_rejects_a_missing_field(self):
        lat, lon = np.array([-30.0, 30.0]), np.array([0.0, 180.0])
        with pytest.raises(ValueError, match="evap_res"):
            read_slim_properties(_dataset(lat, lon).drop_vars("evap_res"))

    def test_config_attachment(self, tmp_path):
        from omegaconf import OmegaConf
        from jcm.forcing_assembly import _attach_slim_properties, _model_latlon_deg
        from jcm.utils import get_coords
        coords = get_coords(np.linspace(0, 1, 9), spectral_truncation=21)
        lat, lon = _model_latlon_deg(coords)
        path = tmp_path / "slim.nc"
        _dataset(np.asarray(lat), np.asarray(lon)).to_netcdf(path)
        forcing = _attach_slim_properties(None, OmegaConf.create({"slim_file": str(path)}),
                                          coords)
        assert set(forcing.slim_properties) == set(MONTHLY_FIELDS) | {"glc_mask"}
        assert _attach_slim_properties(None, OmegaConf.create({"slim_file": None}),
                                       coords) is None


class TestCarry:

    def test_unset_state_is_seeded_from_the_forcing(self):
        from jcm.physics.surface.slim.slim_term import SNOW_COVER_SWE, resolve_state
        forcing = default_forcing.__wrapped__ if hasattr(default_forcing, "__wrapped__") else None
        from jcm.forcing import ForcingData
        n = 4
        forcing = ForcingData.zeros((n,), stl_am=jnp.asarray([260.0, 280.0, 300.0, 290.0]),
                                    soilw_am=jnp.asarray([0.5, 0.2, 2.0, 0.0]),
                                    snowc_am=jnp.asarray([1.0, 0.0, 0.0, 0.5]))
        props = {k: v for k, v in uniform_properties((n,)).items()}
        data = SlimLandData.zeros((n,))
        data = data.copy(soil_temperature=data.soil_temperature.at[:, 3].set(285.0),
                         water=data.water.at[3].set(42.0))
        out = resolve_state(data, forcing, props, n, SlimParameters())
        np.testing.assert_allclose(out.soil_temperature[:, :3], [[260.0, 280.0, 300.0]] * 10)
        np.testing.assert_allclose(out.water[:3], [75.0, 30.0, 150.0])
        np.testing.assert_allclose(out.snow[:3], [SNOW_COVER_SWE, 0.0, 0.0])
        assert float(out.soil_ice[0, 0]) == 0.0 and float(out.soil_ice[5, 0]) > 0.0
        assert float(out.soil_ice[5, 1]) == 0.0
        # A set column is left alone.
        np.testing.assert_allclose(out.soil_temperature[:, 3], 285.0)
        assert float(out.water[3]) == 42.0


class TestWiring:

    def test_echam_physics_composes_slim_after_the_microphysics(self):
        from jcm.physics.echam.echam_terms import echam_physics
        phys = echam_physics(land_model="slim", cloud_scheme="1m")
        names = [t.name for t in phys.terms]
        i = names.index("slim_land")
        assert names.index("tiedtke_convection") < i < names.index("echam_surface_exchange")
        vd = next(t for t in phys.terms if t.name == "tte_tke_vertical_diffusion")
        assert vd.land_model == "slim"
        with pytest.raises(ValueError, match="prescribed_surface_fluxes"):
            echam_physics(land_model="slim", prescribed_surface_fluxes=True)
        with pytest.raises(ValueError, match="slim_land"):
            echam_physics(slim_land={"snow_cap": 1.0})
        p = echam_physics(land_model="slim", slim_land={"albedo": "jsbach", "snow_cap": 100.0})
        slim = next(t for t in p.terms if t.name == "slim_land")
        assert float(slim.params.get_value().snow_cap) == 100.0
        bc = next(t for t in p.terms if t.name == "echam_boundary_conditions")
        assert bc._slim_albedo == "jsbach"

    def test_term_list_presets_take_the_switch(self):
        from jcm.runners import _apply_land_model
        terms = {name: {"_target_": "x"} for name in (
            "echam_boundary_conditions", "tte_tke_vertical_diffusion", "tiedtke_convection",
            "echam_1m_microphysics", "echam_surface_exchange", "hines_gwd")}
        assert _apply_land_model(terms, "prescribed", {}) is terms
        out = _apply_land_model(terms, "slim", {"albedo": "jsbach"})
        names = list(out)
        assert names.index("slim_land") == names.index("echam_surface_exchange") - 1
        assert out["tte_tke_vertical_diffusion"]["land_model"] == "slim"
        assert out["echam_boundary_conditions"]["slim_albedo"] == "jsbach"
        with pytest.raises(ValueError, match="needs the ECHAM terms"):
            _apply_land_model({"echam_boundary_conditions": {}}, "slim", {})
        with pytest.raises(ValueError, match="only"):
            _apply_land_model(terms, "prescribed", {"snow_cap": 1.0})

    def test_the_hydra_presets_build_with_slim(self):
        """physics.land_model=slim composes for the 1M, 2M and JAM-M7 presets."""
        from hydra import compose, initialize_config_module
        from jcm.runners import build_physics
        for config in ("t63-echam-1m", "t63-echam-2m", "t63-echam-jam-m7"):
            with initialize_config_module(config_module="jcm.config", version_base=None):
                cfg = compose(config_name="config", overrides=[
                    f"+configuration={config}", "physics.land_model=slim"])
            if config.endswith("m7"):
                # Composition only: the placeholder core needs no optional extra.
                cfg.physics.jam_microphysics = "m7_placeholder"
                cfg.physics.jam_microphysics_options = {}
                cfg.physics.jam_optics_backend = "jcm"
            phys = build_physics(cfg)
            assert "slim_land" in [t.name for t in phys.terms], config


class TestMigration:

    def test_a_state_without_slim_restores_and_seeds_from_the_forcing(self, caplog):
        from jcm.checkpoint import load_checkpoint, save_checkpoint
        from jcm.forcing import ForcingData
        from jcm.model import Model
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.diagnostics.moist_air_state import MoistAirColumnState
        from jcm.physics.forcing.echam_boundary_conditions import EchamBoundaryConditions
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        from jcm.terrain import TerrainData

        coords = get_held_suarez_coords(spectral_truncation=21)

        def model(slim):
            terms = [MoistAirColumnState(),
                     EchamBoundaryConditions(slim_albedo="slim" if slim else None)]
            if slim:
                terms.append(SlimLand())
            return Model(coords=coords, terrain=TerrainData.from_coords(coords), time_step=30,
                         physics=ComposablePhysics(terms=terms, vectorize_columns=True))

        donor = model(False)
        donor.bootstrap_state()
        target = model(True)
        target.bootstrap_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.msgpack"
            save_checkpoint(donor, path, elapsed_days=0.0)
            with caplog.at_level("INFO", logger="jcm.checkpoint"):
                load_checkpoint(target, path)
        assert any(f"{SLIM_LAND_KEY}.soil_temperature" in r.message and "seeded" in r.message
                   for r in caplog.records)
        np.testing.assert_array_equal(
            np.asarray(target.physics_carry[SLIM_LAND_KEY].soil_temperature), 0.0)
        shape = coords.horizontal.nodal_shape
        base = default_forcing(coords.horizontal)
        forcing = base.copy(slim_properties=uniform_properties(shape),
                            soilw_am=jnp.full(shape, 0.4))
        assert isinstance(forcing, ForcingData)
        target.resume(forcing=forcing, save_interval=1 / 48, total_time=1 / 48)
        carry = target.physics_carry[SLIM_LAND_KEY]
        stl = np.asarray(base.stl_am).ravel()
        np.testing.assert_allclose(np.asarray(carry.soil_temperature)[-1].ravel(), stl,
                                   rtol=1e-6)
        np.testing.assert_allclose(np.asarray(carry.water).ravel(), 0.4 * 150.0, rtol=1e-6)


@pytest.mark.slow
def test_a_t21_model_steps_with_the_slim_land():
    """A T21 ECHAM model runs with land_model=slim; land state stays finite and moves."""
    import jax
    from jcm.model import Model
    from jcm.physics.echam.echam_terms import echam_physics
    from jcm.terrain import TerrainData
    from jcm.utils import get_coords

    coords = get_coords(np.linspace(0, 1, 9), spectral_truncation=21)
    shape = coords.horizontal.nodal_shape
    fmask = np.zeros(shape)
    fmask[: shape[0] // 2] = 1.0                      # half the longitudes are land
    terrain = TerrainData.from_coords(coords, fmask=fmask)
    forcing = default_forcing(coords.horizontal).copy(
        slim_properties=uniform_properties(shape), soilw_am=jnp.full(shape, 0.5))
    model = Model(coords=coords, time_step=30, terrain=terrain,
                  physics=echam_physics(land_model="slim", cloud_scheme="1m"))
    pred = model.run(forcing=forcing, save_interval=1 / 24, total_time=1 / 12)
    assert bool(jnp.all(jnp.isfinite(pred.dynamics.temperature)))
    carry = model._final_physics_state
    for leaf in jax.tree.leaves(carry[SLIM_LAND_KEY]):
        assert bool(jnp.all(jnp.isfinite(leaf)))
    land = np.asarray(terrain.fmask).ravel() > 0.5
    t0 = np.asarray(default_forcing(coords.horizontal).stl_am).ravel()
    moved = np.abs(np.asarray(carry[SLIM_LAND_KEY].soil_temperature)[0].ravel() - t0)[land]
    assert moved.max() > 0.1
    # The soil column is saved on its own axis.
    ds = pred.to_xarray()
    assert "soil_layer" in ds["slim_land.soil_temperature"].dims
    assert ds.sizes["soil_layer"] == 10


def test_calibration_levers_scale_the_maps():
    from jcm.forcing import ForcingData
    from jcm.physics.surface.slim.slim_term import column_properties
    forcing = ForcingData.zeros((3,)).copy(slim_properties=uniform_properties((3,)))
    p = SlimParameters(evap_res_scale=2.0, bucket_depth_scale=0.5)
    props = column_properties(forcing, 3, p)
    np.testing.assert_allclose(props["evap_res"], 300.0)
    np.testing.assert_allclose(props["bucketdepth"], 75.0)
    np.testing.assert_allclose(column_properties(forcing, 3)["evap_res"], 150.0)
    from jcm.physics.echam.echam_terms import echam_physics
    phys = echam_physics(land_model="slim", slim_land={"evap_res_scale": 1.5})
    vd = next(t for t in phys.terms if t.name == "tte_tke_vertical_diffusion")
    assert float(vd.slim_params.get_value().evap_res_scale) == 1.5
