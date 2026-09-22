"""Tests for the package-independent surface-exchange coupling contract.

Covers (jax-gcm#754 / #301):

- the ``SurfaceExchange`` struct round-trip, zeros, and optional-field
  ``None`` semantics (absence omitted from output, never a fake zero);
- the ``ComposablePhysics`` publish/accessor seam and Held-Suarez opt-out;
- both publishers filling the contract EXACTLY from the scheme's own
  delivered internal fluxes (SPEEDY inline, ECHAM terminal term);
- forced mode (#301): fed a run's own published fluxes, the forced
  configuration reproduces the interactive one — SPEEDY bit-for-bit,
  ECHAM to solver round-off on the column-integrated budgets — plus the
  loud-failure guards when the prescribed forcing is absent.
"""

import jax
import jax.numpy as jnp
import pytest

import jcm.constants as c
from jcm.forcing import default_forcing
from jcm.physics.surface.surface_exchange import (
    SURFACE_EXCHANGE_KEY,
    SURFACE_EXCHANGE_OUTPUT_ATTRS,
    SurfaceExchange,
    surface_exchange_from,
)
from jcm.physics_interface import PhysicsState


# ---------------------------------------------------------------------------
# Struct: round-trip, zeros, optional-None semantics
# ---------------------------------------------------------------------------

class TestSurfaceExchangeStruct:
    """The typed contract itself."""

    def _full(self, n=4):
        base = jnp.arange(1.0, n + 1.0)
        return SurfaceExchange(
            net_heat_flux=base, sensible_heat_flux=base * 2,
            latent_heat_flux=base * 3, evaporation=base * 4,
            precipitation=base * 5, stress_u=base * 6, stress_v=base * 7,
            wind_speed=base * 8, air_density=base * 9,
            air_potential_temperature=base * 10,
        )

    def test_tree_round_trip(self):
        """A tree_math struct flattens and unflattens losslessly."""
        se = self._full()
        leaves, treedef = jax.tree_util.tree_flatten(se)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        for name in ("net_heat_flux", "evaporation", "stress_v",
                     "air_potential_temperature"):
            assert jnp.array_equal(getattr(se, name), getattr(rebuilt, name))

    def test_tree_map_preserves_none_optionals(self):
        """Optional tile fields stay out of the leaf set until filled."""
        se = self._full()
        assert se.tile_fraction is None
        assert se.precip_rain is None
        doubled = jax.tree_util.tree_map(lambda x: x * 2, se)
        assert doubled.tile_fraction is None
        assert jnp.array_equal(doubled.evaporation, se.evaporation * 2)

    def test_zeros(self):
        z = SurfaceExchange.zeros((3, 2))
        assert z.net_heat_flux.shape == (3, 2)
        assert float(jnp.sum(jnp.abs(z.sensible_heat_flux))) == 0.0
        assert z.stress_u_tile is None

    def test_output_attrs_cover_every_guaranteed_field(self):
        """Every guaranteed field has CF/units metadata; all state units."""
        guaranteed = [
            "net_heat_flux", "sensible_heat_flux", "latent_heat_flux",
            "evaporation", "precipitation", "stress_u", "stress_v",
            "wind_speed", "air_density", "air_potential_temperature",
        ]
        for field in guaranteed:
            key = f"surface_exchange.{field}"
            assert key in SURFACE_EXCHANGE_OUTPUT_ATTRS, key
            assert "units" in SURFACE_EXCHANGE_OUTPUT_ATTRS[key]

    def test_none_optionals_omitted_from_flattened_output(self):
        """A None optional field is dropped, not published as a zero."""
        from jcm.physics.composable_physics import ComposablePhysics
        cp = ComposablePhysics(terms=[])
        flat = cp.data_struct_to_dict(self._full(), nodal_shape=(4,))
        assert "evaporation" in flat
        assert "net_heat_flux" in flat
        # Optional tile / phase-split fields left None must not appear.
        assert "tile_fraction" not in flat
        assert "precip_rain" not in flat
        assert "net_heat_flux_tile" not in flat


# ---------------------------------------------------------------------------
# Accessor + composition-time seam
# ---------------------------------------------------------------------------

class TestPublishSeam:
    """publishes_surface_exchange / require_surface_exchange / accessor."""

    def test_speedy_publishes(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        phys = speedy_physics()
        assert phys.publishes_surface_exchange()
        phys.require_surface_exchange()  # no raise

    def test_held_suarez_opts_out(self):
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        phys = held_suarez_physics()
        assert not phys.publishes_surface_exchange()
        with pytest.raises(ValueError, match="surface_exchange"):
            phys.require_surface_exchange()

    def test_accessor_raises_pointed_error(self):
        with pytest.raises(KeyError, match="no 'surface_exchange'"):
            surface_exchange_from({})

    def test_accessor_returns_struct(self):
        se = SurfaceExchange.zeros((2,))
        assert surface_exchange_from({SURFACE_EXCHANGE_KEY: se}) is se


# ---------------------------------------------------------------------------
# SPEEDY publisher — exact agreement with the scheme's own internals
# ---------------------------------------------------------------------------

def _speedy_setup():
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics
    from jcm.terrain import TerrainData

    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    terrain = TerrainData.aquaplanet(coords)
    phys = speedy_physics()
    phys.cache_coords(coords)
    nodal = coords.horizontal.nodal_shape
    nlev = coords.nodal_shape[0]
    shape = (nlev,) + nodal
    state = PhysicsState.zeros(shape).copy(
        temperature=jnp.full(shape, 280.0)
        + jnp.linspace(-40, 10, nlev)[:, None, None],
        specific_humidity=jnp.full(shape, 5e-3),
        u_wind=jnp.full(shape, 5.0),
        normalized_surface_pressure=jnp.ones(nodal),
    )
    forcing = default_forcing(coords.horizontal)
    return coords, terrain, phys, state, forcing


class TestSpeedyPublisher:
    """SPEEDY fills the contract from ``_surface_flux`` / precip exactly."""

    def setup_method(self):
        (self.coords, self.terrain, self.phys,
         self.state, self.forcing) = _speedy_setup()
        _, self.diag = self.phys.compute_tendencies(
            self.state, self.forcing, self.terrain)
        self.se = self.diag[SURFACE_EXCHANGE_KEY]
        self.sf = self.diag["_surface_flux"]

    def test_net_heat_flux_is_hfluxn(self):
        assert jnp.allclose(self.se.net_heat_flux, self.sf.hfluxn)

    def test_sensible_heat_positive_up_matches_shf(self):
        assert jnp.allclose(self.se.sensible_heat_flux, self.sf.shf)

    def test_evaporation_g_to_kg(self):
        assert jnp.allclose(self.se.evaporation, self.sf.evap / 1000.0)

    def test_stress_flipped_into_surface(self):
        # SPEEDY ``ustr`` is stress ON the atmosphere; the contract wants
        # the flux INTO the surface (its negative).
        assert jnp.allclose(self.se.stress_u, -self.sf.ustr)
        assert jnp.allclose(self.se.stress_v, -self.sf.vstr)

    def test_precip_is_conv_plus_lsc_scaled(self):
        conv = self.diag["_convection"].precnv
        lsc = self.diag["_condensation"].precls
        assert jnp.allclose(self.se.precipitation, (conv + lsc) / 1000.0)
        assert float(self.se.precipitation.min()) >= 0.0

    def test_thermodynamics_physical(self):
        assert float(self.se.air_density.min()) > 0.5
        assert float(self.se.air_density.max()) < 2.0
        assert float(self.se.air_potential_temperature.min()) > 250.0

    def test_tiles_and_split_absent(self):
        assert self.se.tile_fraction is None
        assert self.se.precip_rain is None


# ---------------------------------------------------------------------------
# SPEEDY forced mode (#301) — self-consistency + guards
# ---------------------------------------------------------------------------

class TestSpeedyForcedMode:
    """Fed its own fluxes, forced SPEEDY reproduces the interactive step."""

    def setup_method(self):
        from jcm.physics.speedy.speedy_terms import (
            speedy_physics, SpeedySurfaceFlux,
        )
        (self.coords, self.terrain, self.phys,
         self.state, self.forcing) = _speedy_setup()
        self.tend_i, self.diag_i = self.phys.compute_tendencies(
            self.state, self.forcing, self.terrain)
        self.se = self.diag_i[SURFACE_EXCHANGE_KEY]
        self.forced = speedy_physics().replace(
            "surface", SpeedySurfaceFlux(prescribed_fluxes=True))
        self.forced.cache_coords(self.coords)
        self.forcing_p = self.forcing.copy(
            prescribed_sensible_heat_flux=self.se.sensible_heat_flux,
            prescribed_evaporation=self.se.evaporation,
            prescribed_stress_u=self.se.stress_u,
            prescribed_stress_v=self.se.stress_v,
        )

    def test_forced_reproduces_interactive_tendencies(self):
        tend_f, _ = self.forced.compute_tendencies(
            self.state, self.forcing_p, self.terrain)
        diffs = jax.tree_util.tree_map(
            lambda a, b: float(jnp.max(jnp.abs(a - b))), self.tend_i, tend_f)
        assert diffs.u_wind < 1e-10
        assert diffs.v_wind < 1e-10
        assert diffs.temperature < 1e-8
        assert diffs.specific_humidity < 1e-10

    def test_forced_echoes_prescribed_fluxes(self):
        _, diag_f = self.forced.compute_tendencies(
            self.state, self.forcing_p, self.terrain)
        sef = diag_f[SURFACE_EXCHANGE_KEY]
        for name in ("sensible_heat_flux", "evaporation",
                     "stress_u", "stress_v"):
            assert jnp.allclose(getattr(sef, name), getattr(self.se, name))

    def test_missing_forcing_raises(self):
        # The loud check lives in validate_forcing (called by Model on the
        # concrete run forcing); __call__ itself falls back to zeros so the
        # abstract shape probe stays well-defined.
        with pytest.raises(ValueError, match="prescribed"):
            self.forced.validate_forcing(self.forcing)
        self.forced.validate_forcing(self.forcing_p)  # complete forcing: no raise


# ---------------------------------------------------------------------------
# ECHAM publisher + forced mode
# ---------------------------------------------------------------------------

def _echam_setup(**echam_kwargs):
    from jcm.utils import get_coords
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.physics.echam.echam_terms import echam_physics
    from jcm.terrain import TerrainData

    levels = get_echam_levels(47)
    coords = get_coords(levels, spectral_truncation=21)
    terrain = TerrainData.aquaplanet(coords)
    phys = echam_physics(**echam_kwargs)
    phys.cache_coords(coords)
    nodal = coords.horizontal.nodal_shape
    nlev = coords.nodal_shape[0]
    shape = (nlev,) + nodal
    state = PhysicsState.zeros(shape).copy(
        temperature=jnp.linspace(210.0, 288.0, nlev)[:, None, None]
        * jnp.ones(shape),
        specific_humidity=jnp.full(shape, 3e-3),
        u_wind=jnp.full(shape, 5.0),
        normalized_surface_pressure=jnp.ones(nodal),
        tracers={s.name: jnp.zeros(shape) for s in phys.required_tracers()},
    )
    forcing = default_forcing(coords.horizontal)
    return coords, terrain, phys, state, forcing, nodal, nlev


class TestEchamPublisher:
    """ECHAM's terminal publisher fills the contract from delivered fluxes."""

    def setup_method(self):
        (self.coords, self.terrain, self.phys, self.state,
         self.forcing, self.nodal, self.nlev) = _echam_setup()
        _, self.diag = self.phys.compute_tendencies(
            self.state, self.forcing, self.terrain,
            self.phys.initial_carry_state(self.coords))
        self.se = self.diag[SURFACE_EXCHANGE_KEY]
        self.sfc = self.diag["surface"]

    def test_turbulent_fluxes_match_delivered(self):
        assert jnp.allclose(
            self.se.sensible_heat_flux, self.sfc.sensible_heat_flux)
        assert jnp.allclose(self.se.latent_heat_flux, self.sfc.latent_heat_flux)
        assert jnp.allclose(self.se.evaporation, self.sfc.evaporation)

    def test_stress_matches_delivered_sign(self):
        # ECHAM ``momentum_flux_u`` is already positive-down (into surface),
        # so the contract copies it unnegated.
        assert jnp.allclose(self.se.stress_u, self.sfc.momentum_flux_u)
        assert jnp.allclose(self.se.stress_v, self.sfc.momentum_flux_v)

    def test_net_heat_flux_energy_balance(self):
        rad = self.diag["radiation"]
        net = ((rad.surface_sw_down - rad.surface_sw_up)
               + (rad.surface_lw_down - rad.surface_lw_up)
               - self.sfc.sensible_heat_flux - self.sfc.latent_heat_flux)
        assert jnp.allclose(self.se.net_heat_flux, net)

    def test_precip_is_total(self):
        clouds = self.diag["clouds"]
        conv = self.diag["convection"]
        expected = (clouds.precip_rain + clouds.precip_snow
                    + conv.precip_conv)
        assert jnp.allclose(self.se.precipitation, expected)
        assert float(self.se.precipitation.min()) >= 0.0

    def test_tiles_and_split_absent(self):
        assert self.se.tile_fraction is None
        assert self.se.precip_rain is None


class TestEchamForcedMode:
    """ECHAM: couple_surface off + prescribed delivery reproduces budgets."""

    def _colint(self, field, ph, nlev):
        dp = ph[1:] - ph[:-1]
        return jnp.sum(field.reshape(nlev, -1) * dp, axis=0) / c.grav

    def setup_method(self):
        (self.coords, self.terrain, self.phys, self.state,
         self.forcing, self.nodal, self.nlev) = _echam_setup()
        self.tend_i, self.diag_i = self.phys.compute_tendencies(
            self.state, self.forcing, self.terrain,
            self.phys.initial_carry_state(self.coords))
        self.se = self.diag_i[SURFACE_EXCHANGE_KEY]

        from jcm.physics.echam.echam_terms import echam_physics
        self.forced = echam_physics(prescribed_surface_fluxes=True)
        self.forced.cache_coords(self.coords)
        self.forcing_p = self.forcing.copy(
            prescribed_sensible_heat_flux=self.se.sensible_heat_flux.reshape(
                self.nodal),
            prescribed_evaporation=self.se.evaporation.reshape(self.nodal),
            prescribed_stress_u=self.se.stress_u.reshape(self.nodal),
            prescribed_stress_v=self.se.stress_v.reshape(self.nodal),
        )
        self.tend_f, self.diag_f = self.forced.compute_tendencies(
            self.state, self.forcing_p, self.terrain,
            self.forced.initial_carry_state(self.coords))

    def test_couple_surface_disabled_in_forced_vdiff(self):
        vdiff_term = next(t for t in self.forced.terms
                          if t.name == "tte_tke_vertical_diffusion")
        assert vdiff_term.couple_surface is False
        prescribed = [t for t in self.forced.terms
                      if t.name == "prescribed_surface_flux"]
        assert len(prescribed) == 1

    def test_column_integrated_budgets_match(self):
        ph = self.diag_i["pressure_half"]
        water_i = self._colint(self.tend_i.specific_humidity, ph, self.nlev)
        water_f = self._colint(self.tend_f.specific_humidity, ph, self.nlev)
        mom_i = self._colint(self.tend_i.u_wind, ph, self.nlev)
        mom_f = self._colint(self.tend_f.u_wind, ph, self.nlev)
        heat_i = self._colint(self.tend_i.temperature, ph, self.nlev) * c.cpd
        heat_f = self._colint(self.tend_f.temperature, ph, self.nlev) * c.cpd
        assert float(jnp.max(jnp.abs(water_i - water_f))) < 1e-8
        assert float(jnp.max(jnp.abs(mom_i - mom_f))) < 1e-5
        assert float(jnp.max(jnp.abs(heat_i - heat_f))) < 1e-1

    def test_forced_water_delivery_equals_prescribed_evaporation(self):
        ph = self.diag_i["pressure_half"]
        water_f = self._colint(self.tend_f.specific_humidity, ph, self.nlev)
        assert float(jnp.max(jnp.abs(water_f - self.se.evaporation))) < 1e-4

    def test_forced_echoes_prescribed_fluxes(self):
        sef = self.diag_f[SURFACE_EXCHANGE_KEY]
        assert jnp.allclose(sef.sensible_heat_flux, self.se.sensible_heat_flux)
        assert jnp.allclose(sef.evaporation, self.se.evaporation)
        assert jnp.allclose(sef.stress_u, self.se.stress_u)

    def test_missing_forcing_raises(self):
        # The loud check lives in validate_forcing (called by Model on the
        # concrete run forcing); __call__ falls back to zeros for the probe.
        with pytest.raises(ValueError, match="prescribed"):
            self.forced.validate_forcing(self.forcing)
        self.forced.validate_forcing(self.forcing_p)  # complete forcing: no raise


# ---------------------------------------------------------------------------
# Forcing doors: forcing.prescribed_surface_flux (constants + file)
# ---------------------------------------------------------------------------

class TestPrescribedFluxForcingAttach:
    """``_attach_prescribed_surface_fluxes`` constants/file/error paths."""

    def _coords(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        return get_speedy_coords(layers=8, spectral_truncation=21)

    def _cfg(self, block):
        from omegaconf import OmegaConf
        return OmegaConf.create({"prescribed_surface_flux": block})

    def test_unset_is_noop(self):
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        assert _attach_prescribed_surface_fluxes(
            None, self._cfg(None), self._coords()) is None

    def test_constants(self):
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        f = _attach_prescribed_surface_fluxes(None, self._cfg({"constants": {
            "sensible_heat_flux": 12.0, "evaporation": 3e-5,
            "stress_u": 0.05, "stress_v": 0.0}}), coords)
        nodal = coords.horizontal.nodal_shape
        assert f.prescribed_sensible_heat_flux.shape == nodal
        # The constant round-trips through the physics float32 working dtype,
        # so compare at a realistic float32 relative tolerance (~1e-5), not
        # pytest.approx's f32-impossible 1e-6 default (#849/#850 convention).
        assert float(f.prescribed_sensible_heat_flux.mean()) == pytest.approx(
            12.0, rel=1e-5)
        assert float(f.prescribed_evaporation.mean()) == pytest.approx(
            3e-5, rel=1e-5)

    def test_constants_missing_field_raises(self):
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        with pytest.raises(ValueError, match="missing"):
            _attach_prescribed_surface_fluxes(
                None, self._cfg({"constants": {"sensible_heat_flux": 1.0}}),
                self._coords())

    def test_both_sources_raises(self):
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        with pytest.raises(ValueError, match="exactly one"):
            _attach_prescribed_surface_fluxes(
                None,
                self._cfg({"constants": {
                    "sensible_heat_flux": 1.0, "evaporation": 1.0,
                    "stress_u": 1.0, "stress_v": 1.0}, "file": "x.nc"}),
                self._coords())

    def _write_flux_nc(self, path, coords, with_time=False):
        import numpy as np
        import xarray as xr
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        varnames = ("sensible_heat_flux", "evaporation", "stress_u", "stress_v")
        data = {}
        if with_time:
            t = np.array([np.datetime64("2000-01-15"),
                          np.datetime64("2000-02-15")])
            for i, v in enumerate(varnames):
                data[v] = (("time", "lat", "lon"),
                           np.full((2, nlat, nlon), float(i + 1)))
            ds = xr.Dataset(data, coords={"time": t, "lat": lat, "lon": lon})
        else:
            for i, v in enumerate(varnames):
                data[v] = (("lat", "lon"), np.full((nlat, nlon), float(i + 1)))
            ds = xr.Dataset(data, coords={"lat": lat, "lon": lon})
        ds.to_netcdf(path)

    def test_file_static(self, tmp_path):
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        p = tmp_path / "flux.nc"
        self._write_flux_nc(p, coords, with_time=False)
        f = _attach_prescribed_surface_fluxes(
            None, self._cfg({"file": str(p)}), coords)
        assert f.prescribed_sensible_heat_flux.shape == \
            coords.horizontal.nodal_shape
        assert float(f.prescribed_sensible_heat_flux.mean()) == 1.0
        assert float(f.prescribed_stress_v.mean()) == 4.0

    def test_file_timeseries_non_monthly_aligns_by_date(self, tmp_path):
        from jcm.forcing import BY_DATE, TimeSeries
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        p = tmp_path / "flux_t.nc"
        self._write_flux_nc(p, coords, with_time=True)  # 2 timestamps
        f = _attach_prescribed_surface_fluxes(
            None, self._cfg({"file": str(p)}), coords)
        # A time axis becomes a TimeSeries leaf, sliced per step by select().
        ts = f.prescribed_sensible_heat_flux
        assert isinstance(ts, TimeSeries)
        # A non-12-step archive MUST align on its absolute timestamps, not be
        # smeared into year bins (Codex #877).
        assert int(ts.align_mode) == BY_DATE

    def _write_flux_nc_at(self, path, coords, times):
        """Write a 4-variable flux file with an explicit ``time`` axis."""
        import numpy as np
        import xarray as xr
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        varnames = ("sensible_heat_flux", "evaporation", "stress_u", "stress_v")
        nt = len(times)
        xr.Dataset(
            {v: (("time", "lat", "lon"), np.zeros((nt, nlat, nlon)))
             for v in varnames},
            coords={"time": np.asarray(times), "lat": lat, "lon": lon},
        ).to_netcdf(path)

    def test_file_monthly_climatology_wraps_year(self, tmp_path):
        import numpy as np
        from jcm.forcing import WRAP_YEAR, TimeSeries
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        # 12 mid-month timestamps → month-sized gaps → WRAP_YEAR.
        t = [np.datetime64("2000-01-15") + np.timedelta64(30 * i, "D")
             for i in range(12)]
        p = tmp_path / "flux_monthly.nc"
        self._write_flux_nc_at(p, coords, t)
        f = _attach_prescribed_surface_fluxes(
            None, self._cfg({"file": str(p)}), coords)
        ts = f.prescribed_sensible_heat_flux
        assert isinstance(ts, TimeSeries)
        assert int(ts.align_mode) == WRAP_YEAR

    def test_file_real_month_starts_wrap_year(self, tmp_path):
        """Actual calendar month-starts (28-31 day gaps) also count as monthly."""
        import numpy as np
        from jcm.forcing import WRAP_YEAR
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        # Jan..Dec 1st of a leap year (Feb->Mar gap = 29 d, the tightest real
        # monthly gap) — all within the month-sized window.
        t = (np.datetime64("2000-01", "M") + np.arange(12)).astype(
            "datetime64[ns]")
        p = tmp_path / "flux_month_starts.nc"
        self._write_flux_nc_at(p, coords, t)
        f = _attach_prescribed_surface_fluxes(
            None, self._cfg({"file": str(p)}), coords)
        assert int(f.prescribed_sensible_heat_flux.align_mode) == WRAP_YEAR

    @pytest.mark.parametrize("step_days,label", [
        (0.5, "12-hourly"),   # sub-daily
        (1.0, "12-daily"),    # daily
        (365.0, "12-yearly"),  # yearly
    ])
    def test_file_twelve_samples_non_monthly_cadence_by_date(
            self, tmp_path, step_days, label):
        """A 12-SAMPLE archive that is not monthly (sub-daily / daily / yearly)
        must align by absolute date, not be wrapped as a fake climatology
        (Codex #877: count is not cadence).
        """
        import numpy as np
        from jcm.forcing import BY_DATE
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        # Fixed-duration hour steps so the arithmetic stays on the ns clock
        # (calendar "M"/"Y" timedeltas can't be added to datetime64[ns]).
        base = np.datetime64("2000-01-01", "ns")
        t = np.array([base + np.timedelta64(int(step_days * 24 * i), "h")
                      for i in range(12)])
        p = tmp_path / f"flux_{label}.nc"
        self._write_flux_nc_at(p, coords, t)
        f = _attach_prescribed_surface_fluxes(
            None, self._cfg({"file": str(p)}), coords)
        assert int(f.prescribed_sensible_heat_flux.align_mode) == BY_DATE, (
            f"{label} cadence must align BY_DATE"
        )

    def test_file_missing_variable_raises(self, tmp_path):
        import numpy as np
        import xarray as xr
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        p = tmp_path / "partial.nc"
        xr.Dataset(
            {"sensible_heat_flux": (("lat", "lon"), np.zeros((nlat, nlon)))},
            coords={"lat": lat, "lon": lon},
        ).to_netcdf(p)
        with pytest.raises(ValueError, match="missing"):
            _attach_prescribed_surface_fluxes(
                None, self._cfg({"file": str(p)}), coords)

    def test_file_descending_latitude_is_flipped(self, tmp_path):
        """A north-to-south file is reoriented to the model grid, not consumed
        positionally — a latitudinally varying field must come back matching
        the ascending-lat reference.
        """
        import numpy as np
        import xarray as xr
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))  # ascending
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        varnames = ("sensible_heat_flux", "evaporation", "stress_u", "stress_v")
        # A field that varies with latitude (index) so a wrong orientation
        # would be detectable; same physical content written N->S.
        asc = np.tile(np.arange(nlat, dtype=float), (nlon, 1))  # (lon, lat)
        p = tmp_path / "desc.nc"
        xr.Dataset(
            {v: (("lat", "lon"), asc.T[::-1]) for v in varnames},
            coords={"lat": lat[::-1], "lon": lon},  # descending latitude
        ).to_netcdf(p)
        f = _attach_prescribed_surface_fluxes(
            None, self._cfg({"file": str(p)}), coords)
        # Reoriented back to the model's (lon, lat) ascending layout.
        np.testing.assert_allclose(
            np.asarray(f.prescribed_sensible_heat_flux), asc)

    def test_file_wrong_grid_raises(self, tmp_path):
        """A file whose latitudes match neither the model grid nor its flip is
        rejected, not silently regridded by index.
        """
        import numpy as np
        import xarray as xr
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        # Right N points, wrong values (uniform 0..nlat spacing, not Gaussian).
        bogus_lat = np.linspace(-89.0, 89.0, nlat)
        varnames = ("sensible_heat_flux", "evaporation", "stress_u", "stress_v")
        p = tmp_path / "wrong.nc"
        xr.Dataset(
            {v: (("lat", "lon"), np.zeros((nlat, nlon))) for v in varnames},
            coords={"lat": bogus_lat, "lon": lon},
        ).to_netcdf(p)
        with pytest.raises(ValueError, match="latitude"):
            _attach_prescribed_surface_fluxes(
                None, self._cfg({"file": str(p)}), coords)


# ---------------------------------------------------------------------------
# Forced-mode validation fires from every public run entry point
# ---------------------------------------------------------------------------

class TestForcedForcingValidation:
    """A forced physics package with missing prescribed fields fails loudly at
    run start, from EVERY public entry point — the validation lives in the
    shared ``run_from_state_with_carry`` choke point, not one door.
    """

    def _forced_model(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import (
            speedy_physics, SpeedySurfaceFlux,
        )
        from jcm.terrain import TerrainData
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        physics = speedy_physics().replace(
            "surface", SpeedySurfaceFlux(prescribed_fluxes=True))
        return Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                     physics=physics, time_step=20)

    def test_run_rejects_missing(self):
        model = self._forced_model()
        bare = default_forcing(model.coords.horizontal)  # no prescribed_* fields
        with pytest.raises(ValueError, match="prescribed"):
            model.run(forcing=bare, save_interval=(1 / 24.0),
                      total_time=(1 / 24.0))

    def test_run_from_state_rejects_missing(self):
        model = self._forced_model()
        state = model._prepare_initial_dycore_state()
        bare = default_forcing(model.coords.horizontal)
        with pytest.raises(ValueError, match="prescribed"):
            model.run_from_state(state, bare, save_interval=(1 / 24.0),
                                 total_time=(1 / 24.0))

    def test_run_from_state_with_carry_rejects_missing(self):
        model = self._forced_model()
        state = model._prepare_initial_dycore_state()
        bare = default_forcing(model.coords.horizontal)
        with pytest.raises(ValueError, match="prescribed"):
            model.run_from_state_with_carry(
                state, bare, save_interval=(1 / 24.0), total_time=(1 / 24.0))


# ---------------------------------------------------------------------------
# pySES backend rejects forced fluxes loudly (out of scope for v3.0)
# ---------------------------------------------------------------------------

def test_pyses_forcing_rejects_prescribed_flux():
    """The pySES column-forcing path has no prescribed-flux wiring; it refuses
    the block with a clear message rather than dropping it silently.
    """
    from omegaconf import OmegaConf
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.runners import _build_pyses_forcing
    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    cfg = OmegaConf.create({"prescribed_surface_flux": {"constants": {
        "sensible_heat_flux": 10.0, "evaporation": 3e-5,
        "stress_u": 0.05, "stress_v": 0.0}}})
    # The guard runs before the dycore is touched, so a placeholder is fine.
    with pytest.raises(ValueError, match="pySES"):
        _build_pyses_forcing(cfg, dycore=None, coords=coords)


# ---------------------------------------------------------------------------
# Constant-flux forced aquaplanet smoke run (NaN-free)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_speedy_forced_constant_flux_aquaplanet_smoke():
    """A short forced SPEEDY run with constant prescribed fluxes stays finite."""
    from jcm.model import Model
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics, SpeedySurfaceFlux
    from jcm.terrain import TerrainData

    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    terrain = TerrainData.aquaplanet(coords)
    physics = speedy_physics().replace(
        "surface", SpeedySurfaceFlux(prescribed_fluxes=True))
    # ``time_step`` is in MINUTES; 20 min is comfortably stable at T21 and
    # divides the 1-hour save interval into 3 inner steps.
    model = Model(coords=coords, terrain=terrain, physics=physics,
                  time_step=20)

    nodal = coords.horizontal.nodal_shape
    forcing = default_forcing(coords.horizontal).copy(
        prescribed_sensible_heat_flux=jnp.full(nodal, 10.0),
        prescribed_evaporation=jnp.full(nodal, 3.0e-5),
        prescribed_stress_u=jnp.full(nodal, 0.05),
        prescribed_stress_v=jnp.zeros(nodal),
    )

    preds = model.run(forcing=forcing, save_interval=(1 / 24.0),
                      total_time=(1 / 12.0))
    se = preds.physics[SURFACE_EXCHANGE_KEY]
    # Published struct echoes the constant prescribed values.
    assert jnp.allclose(se.sensible_heat_flux, 10.0)
    assert jnp.allclose(se.evaporation, 3.0e-5)
    # No NaNs anywhere in the published contract.
    for leaf in jax.tree_util.tree_leaves(se):
        assert not bool(jnp.any(jnp.isnan(leaf)))
