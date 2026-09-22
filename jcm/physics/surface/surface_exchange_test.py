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
        with pytest.raises(ValueError, match="prescribed"):
            self.forced.compute_tendencies(
                self.state, self.forcing, self.terrain)


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
        with pytest.raises(ValueError, match="prescribed"):
            self.forced.compute_tendencies(
                self.state, self.forcing, self.terrain,
                self.forced.initial_carry_state(self.coords))


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
    model = Model(coords=coords, terrain=terrain, physics=physics,
                  time_step=600)

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
