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
    WIND_REFERENCES,
    SurfaceExchange,
    surface_exchange_from,
    surface_exchange_output_attrs,
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
            wind_speed=base * 8, wind_u=base * 8 * 0.6,
            wind_v=-base * 8 * 0.8, air_density=base * 9,
            air_potential_temperature=base * 10, wind_reference="10m",
        )

    def test_tree_round_trip(self):
        """The struct flattens and unflattens losslessly, metadata included."""
        se = self._full()
        leaves, treedef = jax.tree_util.tree_flatten(se)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        for name in ("net_heat_flux", "evaporation", "stress_v", "wind_u",
                     "air_potential_temperature"):
            assert jnp.array_equal(getattr(se, name), getattr(rebuilt, name))
        assert rebuilt.wind_reference == "10m"

    def test_wind_reference_is_static_metadata(self):
        """Not an array leaf: the struct is a valid jit output and the
        reference is readable on the host.
        """
        se = self._full()
        assert "10m" not in jax.tree_util.tree_leaves(se)
        out = jax.jit(lambda x: x.replace(wind_u=x.wind_u * 1.0))(se)
        assert out.wind_reference == "10m"

    def test_wind_reference_is_required(self):
        with pytest.raises(TypeError, match="wind_reference"):
            SurfaceExchange(*([jnp.zeros(2)] * 12))

    def test_validate_accepts_consistent_struct(self):
        self._full().validate()

    def test_validate_rejects_wind_vector_speed_mismatch(self):
        se = self._full().replace(wind_v=self._full().wind_v * 0.5)
        with pytest.raises(ValueError, match="hypot"):
            se.validate()

    def test_validate_rejects_unknown_reference(self):
        with pytest.raises(ValueError, match="wind_reference"):
            self._full().replace(wind_reference="2m").validate()

    def test_validate_tile_invariants(self):
        se = self._full(n=2)
        frac = jnp.array([[0.5, 0.2, 0.3], [1.0, 0.0, 0.0]])
        red = jnp.array([[1.2, 0.8, 0.6], [1.0, 0.5, 0.5]])
        # Tiles whose fraction-weighted sum IS the grid mean, all parallel
        # to it (as ECHAM's per-tile reductions of one lowest-level wind).
        speed_t = red * (se.wind_speed / jnp.sum(frac * red, axis=-1))[:, None]
        tiled = se.replace(
            tile_fraction=frac, wind_speed_tile=speed_t,
            wind_u_tile=speed_t * (se.wind_u / se.wind_speed)[:, None],
            wind_v_tile=speed_t * (se.wind_v / se.wind_speed)[:, None])
        tiled.validate()
        with pytest.raises(ValueError, match="tile_fraction"):
            tiled.replace(tile_fraction=None).validate()
        with pytest.raises(ValueError, match="wind_u_tile"):
            tiled.replace(wind_u_tile=tiled.wind_u_tile * 1.1).validate()

    def test_tree_map_preserves_none_optionals(self):
        """Optional tile fields stay out of the leaf set until filled."""
        se = self._full()
        assert se.tile_fraction is None
        assert se.precip_rain is None
        doubled = jax.tree_util.tree_map(lambda x: x * 2, se)
        assert doubled.tile_fraction is None
        assert jnp.array_equal(doubled.evaporation, se.evaporation * 2)

    def test_zeros(self):
        z = SurfaceExchange.zeros((3, 2), wind_reference="lowest_level")
        assert z.net_heat_flux.shape == (3, 2)
        assert z.wind_u.shape == (3, 2)
        assert float(jnp.sum(jnp.abs(z.sensible_heat_flux))) == 0.0
        assert z.stress_u_tile is None
        assert z.wind_u_tile is None
        assert z.wind_reference == "lowest_level"

    def test_output_attrs_cover_every_guaranteed_field(self):
        """Every guaranteed field has CF/units metadata; all state units."""
        guaranteed = [
            "net_heat_flux", "sensible_heat_flux", "latent_heat_flux",
            "evaporation", "precipitation", "stress_u", "stress_v",
            "wind_speed", "wind_u", "wind_v", "air_density",
            "air_potential_temperature",
        ]
        for field in guaranteed:
            key = f"surface_exchange.{field}"
            assert key in SURFACE_EXCHANGE_OUTPUT_ATTRS, key
            assert "units" in SURFACE_EXCHANGE_OUTPUT_ATTRS[key]
        assert (SURFACE_EXCHANGE_OUTPUT_ATTRS["surface_exchange.wind_u"]
                ["standard_name"] == "eastward_wind")
        assert (SURFACE_EXCHANGE_OUTPUT_ATTRS["surface_exchange.wind_v"]
                ["standard_name"] == "northward_wind")

    @pytest.mark.parametrize("reference", sorted(WIND_REFERENCES))
    def test_output_attrs_carry_the_wind_reference(self, reference):
        attrs = surface_exchange_output_attrs(reference)
        for field in ("wind_speed", "wind_u", "wind_v"):
            entry = attrs[f"surface_exchange.{field}"]
            assert entry["wind_reference"] == reference
            assert (entry["wind_reference_description"]
                    == WIND_REFERENCES[reference])
            assert ("height" in entry) == (reference == "10m")
        # The shared table itself is not mutated.
        assert "wind_reference" not in (
            SURFACE_EXCHANGE_OUTPUT_ATTRS["surface_exchange.wind_u"])
        with pytest.raises(ValueError, match="wind_reference"):
            surface_exchange_output_attrs("2m")

    def test_publishers_declare_their_reference(self):
        from jcm.physics.speedy.speedy_terms import SpeedySurfaceFlux
        from jcm.physics.surface.echam.surface_exchange_publisher import (
            EchamSurfaceExchange,
        )
        key = "surface_exchange.wind_u"
        assert (SpeedySurfaceFlux.output_attrs[key]["wind_reference"]
                == "lowest_level")
        assert EchamSurfaceExchange.output_attrs[key]["wind_reference"] == "10m"

    def test_none_optionals_omitted_from_flattened_output(self):
        """A None optional field is dropped, not published as a zero."""
        from jcm.physics.composable_physics import ComposablePhysics
        cp = ComposablePhysics(terms=[])
        flat = cp.data_struct_to_dict(self._full(), nodal_shape=(4,))
        assert "evaporation" in flat
        assert "net_heat_flux" in flat
        assert "wind_u" in flat
        # Static string metadata is an attribute, not a variable.
        assert "wind_reference" not in flat
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
        se = SurfaceExchange.zeros((2,), wind_reference="10m")
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

    def test_wind_is_the_closure_wind(self):
        """``(wind_u, wind_v)`` is SPEEDY's own ``(u0, v0)`` = fwind0 x the
        lowest-level wind, fwind0 = 0.95 by default.
        """
        assert self.se.wind_reference == "lowest_level"
        assert jnp.array_equal(self.se.wind_u, self.sf.u0)
        assert jnp.array_equal(self.se.wind_v, self.sf.v0)
        assert jnp.allclose(self.se.wind_u, 0.95 * self.state.u_wind[-1])
        assert jnp.allclose(self.se.wind_v, 0.95 * self.state.v_wind[-1])
        self.se.validate()

    def test_westerly_gives_positive_wind_and_stress(self):
        # The setup wind is a uniform 5 m/s westerly.
        assert float(self.se.wind_u.min()) > 0.0
        assert float(self.se.stress_u.min()) > 0.0

    def test_wind_gradient_finite_at_rest(self):
        """SPEEDY's default state is at rest: the published wind (speed and
        components) must not poison reverse mode there.
        """
        state0 = self.state.copy(u_wind=jnp.zeros_like(self.state.u_wind))

        def total(scale):
            state = state0.copy(u_wind=state0.u_wind * scale,
                                v_wind=state0.v_wind * scale)
            _, diag = self.phys.compute_tendencies(
                state, self.forcing, self.terrain)
            se = diag[SURFACE_EXCHANGE_KEY]
            return jnp.sum(se.wind_speed + se.wind_u + se.wind_v)

        assert jnp.isfinite(jax.grad(total)(1.0))

    def test_tiles_and_split_absent(self):
        assert self.se.tile_fraction is None
        assert self.se.precip_rain is None
        assert self.se.wind_u_tile is None
        assert self.se.wind_speed_tile is None


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

    def test_forced_wind_is_the_atmospheres_own(self):
        """The coupler prescribes stress only; the published wind is
        unchanged by forcing the fluxes.
        """
        _, diag_f = self.forced.compute_tendencies(
            self.state, self.forcing_p, self.terrain)
        sef = diag_f[SURFACE_EXCHANGE_KEY]
        for name in ("wind_speed", "wind_u", "wind_v"):
            assert jnp.array_equal(getattr(sef, name), getattr(self.se, name))
        assert sef.wind_reference == self.se.wind_reference

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

def _echam_setup(v_wind=0.0, **echam_kwargs):
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
        v_wind=jnp.full(shape, v_wind),
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

    def test_flux_tiles_and_split_absent(self):
        assert self.se.sensible_heat_flux_tile is None
        assert self.se.stress_u_tile is None
        assert self.se.precip_rain is None


class TestEchamPublishedWind:
    """ECHAM publishes vdiff's 10 m wind: grid mean, tiles, one 10 m wind."""

    def setup_method(self):
        (self.coords, self.terrain, self.phys, self.state,
         self.forcing, self.nodal, self.nlev) = _echam_setup(
            v_wind=-3.0, enable_aerocom=True, aerocom_groups=("nearsurface",))
        # A hydrostatic (isothermal 260 K) geopotential, so the lowest level
        # sits ~30 m up and the 10 m reduction is active (a zero
        # geopotential puts it below 10 m, where the wind is unreduced).
        vertical = self.coords.vertical
        sigma = jnp.asarray(vertical.centers if hasattr(vertical, "centers")
                            else vertical.get_sigma_centers(101325.0))
        geopotential = -c.rd * 260.0 * jnp.log(sigma)
        self.state = self.state.copy(geopotential=geopotential[:, None, None]
                                     * jnp.ones(self.state.temperature.shape))
        _, self.diag = self.phys.compute_tendencies(
            self.state, self.forcing, self.terrain,
            self.phys.initial_carry_state(self.coords))
        self.se = self.diag[SURFACE_EXCHANGE_KEY]
        self.vdiff = self.diag["vertical_diffusion"]

    def test_wind_is_the_vdiff_10m_wind(self):
        ncols = self.se.wind_speed.shape[0]
        assert self.se.wind_reference == "10m"
        for name, vname in (("wind_speed", "wind_10m"),
                            ("wind_u", "wind_10m_u"),
                            ("wind_v", "wind_10m_v")):
            assert jnp.array_equal(
                getattr(self.se, name),
                getattr(self.vdiff, vname).reshape(ncols))
        self.se.validate()

    def test_wind_parallel_to_lowest_level_and_reduced(self):
        """ECHAM ``u10 = zred * pum1``: the 10 m vector keeps the
        lowest-level direction (westerly, southward) and is weaker.
        """
        u_low, v_low = 5.0, -3.0
        speed_low = float(jnp.hypot(u_low, v_low))
        assert jnp.allclose(self.se.wind_u * speed_low,
                            u_low * self.se.wind_speed, rtol=1e-5)
        assert jnp.allclose(self.se.wind_v * speed_low,
                            v_low * self.se.wind_speed, rtol=1e-5)
        assert float(self.se.wind_speed.max()) < speed_low
        assert float(self.se.wind_u.min()) > 0.0
        assert float(self.se.stress_u.min()) > 0.0

    def test_grid_mean_is_the_tile_weighted_sum(self):
        frac = self.se.tile_fraction
        assert frac.shape == self.se.wind_u_tile.shape
        assert jnp.allclose(jnp.sum(frac, axis=-1), 1.0)
        for name in ("wind_u", "wind_v", "wind_speed"):
            assert jnp.allclose(
                jnp.sum(frac * getattr(self.se, name + "_tile"), axis=-1),
                getattr(self.se, name), rtol=1e-5, atol=1e-6)

    def test_aerocom_uas_vas_are_the_published_wind(self):
        """One 10 m wind in the output: AeroCom uas/vas == wind_u/wind_v."""
        ncols = self.se.wind_u.shape[0]
        assert jnp.array_equal(
            self.diag["aerocom_uas"].reshape(ncols), self.se.wind_u)
        assert jnp.array_equal(
            self.diag["aerocom_vas"].reshape(ncols), self.se.wind_v)


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

    def test_forced_wind_is_the_atmospheres_own(self):
        """The vdiff term diagnoses the 10 m wind before its surface-coupling
        branch, so forcing the fluxes leaves the published wind unchanged.
        """
        sef = self.diag_f[SURFACE_EXCHANGE_KEY]
        for name in ("wind_speed", "wind_u", "wind_v", "wind_u_tile",
                     "wind_speed_tile", "tile_fraction"):
            assert jnp.array_equal(getattr(sef, name), getattr(self.se, name))

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
            None, self._cfg({"file": str(p), "align": "by_date"}), coords)
        # A time axis becomes a TimeSeries leaf, sliced per step by select().
        ts = f.prescribed_sensible_heat_flux
        assert isinstance(ts, TimeSeries)
        assert int(ts.align_mode) == BY_DATE

    def _write_flux_nc_at(self, path, coords, times, tag_per_time=None):
        """Write a 4-variable flux file with an explicit ``time`` axis.

        ``tag_per_time`` (optional, one scalar per timestamp) fills every grid
        cell of that timestep with the scalar, so a test can detect whether the
        loader reordered the samples correctly.
        """
        import numpy as np
        import xarray as xr
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        varnames = ("sensible_heat_flux", "evaporation", "stress_u", "stress_v")
        nt = len(times)
        if tag_per_time is None:
            block = np.zeros((nt, nlat, nlon))
        else:
            block = np.stack([np.full((nlat, nlon), float(tag))
                              for tag in tag_per_time])
        times = np.asarray(times)
        xr.Dataset(
            {v: (("time", "lat", "lon"), block.copy()) for v in varnames},
            coords={"time": times, "lat": lat, "lon": lon},
        ).to_netcdf(path)

    @staticmethod
    def _jan_to_dec(year=2000, day=15):
        import numpy as np
        return [np.datetime64(f"{year}-{m:02d}-{day:02d}", "ns")
                for m in range(1, 13)]

    def _load(self, tmp_path, times, name, align=None, **kw):
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        p = tmp_path / name
        self._write_flux_nc_at(p, coords, times, **kw)
        block = {"file": str(p)}
        if align is not None:
            block["align"] = align
        return _attach_prescribed_surface_fluxes(
            None, self._cfg(block), coords).prescribed_sensible_heat_flux

    # -- the declared alignment decides; timestamps never do (#884) ---------

    def test_auto_raises_for_a_time_resolved_file(self, tmp_path):
        """Codex #877's case: a one-year TRANSIENT monthly archive (one sample
        per month Jan..Dec of 2000, e.g. a coupler history file) has exactly a
        climatology's timestamps. No data-mirror product carries fluxes, so
        ``auto`` has no manifest kind to resolve from and must raise, naming
        the knob — never guess and replay 2000's fluxes every year.
        """
        for day in (1, 15):  # month-start and mid-month stamps
            with pytest.raises(ValueError,
                               match="prescribed_surface_flux.align"):
                self._load(tmp_path, self._jan_to_dec(day=day),
                           f"transient_{day}.nc")

    def test_explicit_by_date_on_a_jan_dec_archive(self, tmp_path):
        from jcm.forcing import BY_DATE, BY_DATE_INTERP
        ts = self._load(tmp_path, self._jan_to_dec(), "a.nc", align="by_date")
        assert int(ts.align_mode) == BY_DATE
        ts = self._load(tmp_path, self._jan_to_dec(), "b.nc",
                        align="by_date_interp")
        assert int(ts.align_mode) == BY_DATE_INTERP

    def test_explicit_wrap_year_wraps_jan_dec(self, tmp_path):
        from jcm.forcing import WRAP_YEAR
        for day in (1, 15):
            ts = self._load(tmp_path, self._jan_to_dec(day=day),
                            f"clim_{day}.nc", align="wrap_year")
            assert int(ts.align_mode) == WRAP_YEAR

    def test_unknown_align_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown align"):
            self._load(tmp_path, self._jan_to_dec(), "x.nc",
                       align="climatology")

    # -- a declared climatology is VALIDATED as Jan->Dec --------------------

    @staticmethod
    def _axis(label):
        import numpy as np
        base = np.datetime64("2000-01-01", "ns")
        return {
            "jul_to_jun": (np.datetime64("2000-07", "M")
                           + np.arange(12)).astype("datetime64[ns]"),
            "every_4_weeks": [base + np.timedelta64(28 * i, "D")
                              for i in range(12)],
            "bimonthly": (np.datetime64("2000-01", "M")
                          + 2 * np.arange(12)).astype("datetime64[ns]"),
            "12_hourly": [base + np.timedelta64(12 * i, "h") for i in range(12)],
            "12_daily": [base + np.timedelta64(i, "D") for i in range(12)],
            "12_yearly": [base + np.timedelta64(365 * i, "D")
                          for i in range(12)],
            "seasonal": [np.datetime64(f"2000-{m:02d}-15", "ns")
                         for m in (1, 4, 7, 10)],
        }[label]

    @pytest.mark.parametrize("label", [
        "jul_to_jun", "every_4_weeks", "bimonthly", "12_daily", "seasonal"])
    def test_declared_climatology_not_jan_dec_raises(self, tmp_path, label):
        """WRAP_YEAR indexes sample ``floor(tyear*12) % 12`` (0 == January),
        so a file declared ``wrap_year`` must be exactly Jan..Dec, or it would
        replay out of phase. Each of these raises instead: Jul→Jun (six months
        out of phase), every-4-weeks (two Januaries, no December), bi-monthly
        over two years, 12 daily samples, a 4-sample seasonal climatology.
        """
        with pytest.raises(ValueError, match="January..December"):
            self._load(tmp_path, self._axis(label), f"{label}.nc",
                       align="wrap_year")

    @pytest.mark.parametrize("label", [
        "jul_to_jun", "every_4_weeks", "bimonthly", "12_hourly", "12_daily",
        "12_yearly"])
    def test_by_date_accepts_any_ascending_axis(self, tmp_path, label):
        """Declared ``by_date``, every axis earlier review rounds worried about
        aligns on its absolute dates (no month-position constraint).
        """
        from jcm.forcing import BY_DATE
        ts = self._load(tmp_path, self._axis(label), f"{label}.nc",
                        align="by_date")
        assert int(ts.align_mode) == BY_DATE

    # -- ordering and axis hygiene ------------------------------------------

    def test_descending_climatology_sorted_and_wraps(self, tmp_path):
        """A Jan→Dec climatology stored DESCENDING (Dec first) is sorted to
        ascending time and its samples reordered to match, so WRAP_YEAR's
        January-anchored position index lands on real January data. Each month
        is tagged with its number so a mis-order would be caught.
        """
        import numpy as np
        from jcm.forcing import WRAP_YEAR
        months_desc = list(range(12, 0, -1))  # 12, 11, …, 1  (Dec first)
        times = [np.datetime64(f"2000-{m:02d}-15", "ns") for m in months_desc]
        ts = self._load(tmp_path, times, "desc_clim.nc",
                        tag_per_time=months_desc, align="wrap_year")
        assert int(ts.align_mode) == WRAP_YEAR
        assert bool(np.all(np.diff(_bound_seconds(ts.times)) > 0))
        vals = np.asarray(ts.values)  # (time, lon, lat), ascending time
        assert float(vals[0].mean()) == 1.0    # January
        assert float(vals[-1].mean()) == 12.0  # December

    def test_descending_transient_sorted_for_by_date(self, tmp_path):
        """A descending (latest-first) axis routed to BY_DATE is sorted
        ascending (``searchsorted`` requires it) with samples reordered.
        """
        import numpy as np
        from jcm.forcing import BY_DATE
        days_desc = list(range(11, -1, -1))  # 11, 10, …, 0
        times = [np.datetime64("2000-06-01", "ns") + np.timedelta64(d, "D")
                 for d in days_desc]
        ts = self._load(tmp_path, times, "desc_daily.nc",
                        tag_per_time=days_desc, align="by_date")
        assert int(ts.align_mode) == BY_DATE
        assert bool(np.all(np.diff(_bound_seconds(ts.times)) > 0))
        vals = np.asarray(ts.values)
        assert float(vals[0].mean()) == 0.0
        assert float(vals[-1].mean()) == 11.0

    def test_align_with_constants_raises(self):
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        with pytest.raises(ValueError, match="only to a 'file'"):
            _attach_prescribed_surface_fluxes(
                None, self._cfg({"align": "wrap_year", "constants": {
                    "sensible_heat_flux": 1.0, "evaporation": 1.0,
                    "stress_u": 1.0, "stress_v": 1.0}}), self._coords())

    def test_missing_timestamp_raises(self, tmp_path):
        import numpy as np
        t = [np.datetime64("2000-01-15", "ns"), np.datetime64("NaT", "ns"),
             np.datetime64("2000-03-15", "ns")]
        with pytest.raises(ValueError, match="NaT"):
            self._load(tmp_path, t, "nat.nc")

    @pytest.mark.parametrize("calendar", ["noleap", "360_day"])
    def test_idealised_year_zero_climatology_wraps(self, tmp_path, calendar):
        """Codex #877 (P2): a climatology stamped with idealised CF calendar
        dates in year 0 (``cftime`` noleap / 360_day, stored Dec-first here)
        loads as WRAP_YEAR. WRAP_YEAR never reads absolute times, so the axis
        is sorted and validated on its calendar fields and never converted to
        (non-existent) Gregorian epoch seconds.
        """
        import cftime
        import numpy as np
        from jcm.forcing import WRAP_YEAR
        cls = {"noleap": cftime.DatetimeNoLeap,
               "360_day": cftime.Datetime360Day}[calendar]
        months_desc = list(range(12, 0, -1))
        times = [cls(0, m, 15) for m in months_desc]
        ts = self._load(tmp_path, times, f"y0_{calendar}.nc",
                        tag_per_time=months_desc, align="wrap_year")
        assert int(ts.align_mode) == WRAP_YEAR
        assert bool(np.all(np.diff(_bound_seconds(ts.times)) > 0))
        vals = np.asarray(ts.values)
        assert float(vals[0].mean()) == 1.0 and float(vals[-1].mean()) == 12.0

    def test_idealised_year_zero_by_date_raises(self, tmp_path):
        """Date alignment needs the model's Gregorian clock; a year-0 axis
        has no place on it, so declaring it ``by_date`` fails loudly.
        """
        import cftime
        times = [cftime.DatetimeNoLeap(0, m, 15) for m in range(1, 13)]
        with pytest.raises(ValueError):
            self._load(tmp_path, times, "y0_by_date.nc", align="by_date")

    def test_duplicate_timestamps_raise(self, tmp_path):
        import numpy as np
        t = [np.datetime64("2000-01-15", "ns")] * 2 + [
            np.datetime64("2000-02-15", "ns")]
        with pytest.raises(ValueError, match="duplicate"):
            self._load(tmp_path, t, "dup.nc")

    def test_numeric_time_axis_raises(self, tmp_path):
        """A time axis that does not decode to dates cannot be aligned."""
        import numpy as np
        import xarray as xr
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        p = tmp_path / "numeric.nc"
        xr.Dataset(
            {v: (("time", "lat", "lon"), np.zeros((12, nlat, nlon)))
             for v in ("sensible_heat_flux", "evaporation", "stress_u",
                       "stress_v")},
            coords={"time": np.arange(1, 13), "lat": lat, "lon": lon},
        ).to_netcdf(p)
        with pytest.raises(ValueError, match="does not decode to dates"):
            _attach_prescribed_surface_fluxes(
                None, self._cfg({"file": str(p), "align": "wrap_year"}),
                coords)

    def test_single_sample_time_axis_is_static(self, tmp_path):
        import numpy as np
        f = self._load(tmp_path, [np.datetime64("2000-01-15", "ns")],
                       "one.nc", tag_per_time=[7.0])
        assert not hasattr(f, "align_mode")  # bare array, not a TimeSeries
        assert float(np.asarray(f).mean()) == 7.0

    def test_python_door_reader_matches_hydra_door(self, tmp_path):
        """``read_prescribed_surface_fluxes`` is the Python door; the Hydra
        door returns the same leaves for the same file and ``align``.
        """
        import numpy as np
        import xarray as xr
        from jcm.forcing import read_prescribed_surface_fluxes, WRAP_YEAR
        from jcm.forcing_assembly import _model_latlon_deg
        coords = self._coords()
        p = tmp_path / "door.nc"
        self._write_flux_nc_at(p, coords, self._jan_to_dec(),
                               tag_per_time=list(range(1, 13)))
        lat, lon = _model_latlon_deg(coords)
        with xr.open_dataset(p) as ds:
            fields = read_prescribed_surface_fluxes(
                ds, lat, lon, align_mode="wrap_year")
        assert set(fields) == {
            "prescribed_sensible_heat_flux", "prescribed_evaporation",
            "prescribed_stress_u", "prescribed_stress_v"}
        via_cfg = self._load(tmp_path, self._jan_to_dec(), "door2.nc",
                             tag_per_time=list(range(1, 13)),
                             align="wrap_year")
        py = fields["prescribed_sensible_heat_flux"]
        assert int(py.align_mode) == int(via_cfg.align_mode) == WRAP_YEAR
        np.testing.assert_array_equal(np.asarray(py.values),
                                      np.asarray(via_cfg.values))

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
                state, bare, save_interval=(1 / 24.0), total_time=(1 / 24.0),
                initial_time=model.start_time, initial_step=0)


# ---------------------------------------------------------------------------
# A date-aligned flux archive must cover the run window (no silent clamping)
# ---------------------------------------------------------------------------

def _monthly_seconds(year=2000, day=1):
    import numpy as np
    import pandas as pd
    return np.array([
        (pd.Timestamp(f"{year}-{m:02d}-{day:02d}")
         - pd.Timestamp("1970-01-01")).total_seconds() for m in range(1, 13)])


def _secs(date):
    import pandas as pd
    return (pd.Timestamp(date) - pd.Timestamp("1970-01-01")).total_seconds()


def _dates(seconds):
    """Convert seconds since 1970 to the exact ``datetime64[s]`` labels a
    ``TimeSeries`` axis takes.
    """
    import numpy as np
    return np.rint(np.asarray(seconds, dtype=float)).astype(
        "int64").astype("datetime64[s]")


def _bound_seconds(bounds):
    """Return exact ``Datetime`` bounds or times as seconds since 1970."""
    from jcm.forcing import _host_epoch_seconds
    return _host_epoch_seconds(bounds).astype(float)


class TestByDateCoverage:
    """``by_date_coverage_error`` and its run-start wiring."""

    def _ts(self, align, day=1, year=2000):
        from jcm.forcing import make_time_series
        t = _monthly_seconds(year, day)
        return make_time_series(jnp.zeros((12, 2, 2)), _dates(t), align)

    @pytest.mark.parametrize("day", [1, 15])
    def test_monthly_archive_covers_its_calendar_year(self, day):
        from jcm.forcing import BY_DATE, by_date_coverage_error
        ts = self._ts(BY_DATE, day=day)
        assert by_date_coverage_error(
            ts, _secs("2000-01-01"), _secs("2000-12-31")) is None
        # The 366-day leap year ending exactly at 2001-01-01 00:00.
        assert by_date_coverage_error(
            ts, _secs("2000-01-01"), _secs("2001-01-01")) is None

    def test_run_past_archive_end_is_reported(self):
        from jcm.forcing import BY_DATE, by_date_coverage_error
        err = by_date_coverage_error(
            self._ts(BY_DATE), _secs("2000-06-01"), _secs("2001-03-01"),
            name="forcing.x")
        assert err is not None and "forcing.x" in err
        assert "align=wrap_year" in err  # names the climatology remedy

    def test_run_before_archive_start_is_reported(self):
        from jcm.forcing import BY_DATE_INTERP, by_date_coverage_error
        assert by_date_coverage_error(
            self._ts(BY_DATE_INTERP), _secs("1999-10-01"),
            _secs("2000-03-01")) is not None

    def test_run_in_another_year_is_reported(self):
        """Codex #877's consequence: a one-year archive aligned BY_DATE, run in
        a different year, fails instead of holding its December forever.
        """
        from jcm.forcing import BY_DATE, by_date_coverage_error
        assert by_date_coverage_error(
            self._ts(BY_DATE), _secs("2001-01-01"),
            _secs("2001-02-01")) is not None

    def test_wrap_year_always_covers(self):
        from jcm.forcing import WRAP_YEAR, by_date_coverage_error
        assert by_date_coverage_error(
            self._ts(WRAP_YEAR), _secs("2050-01-01"),
            _secs("2060-01-01")) is None

    def _forcing_with(self, leaf):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        return default_forcing(coords.horizontal).copy(
            prescribed_sensible_heat_flux=leaf,
            prescribed_evaporation=leaf,
            prescribed_stress_u=leaf,
            prescribed_stress_v=leaf,
        )

    def test_both_forced_terms_check_coverage(self):
        from jcm.forcing import BY_DATE
        from jcm.physics.speedy.speedy_terms import SpeedySurfaceFlux
        from jcm.physics.surface.prescribed_flux import PrescribedSurfaceFlux
        forcing = self._forcing_with(self._ts(BY_DATE))
        inside = (_secs("2000-02-01"), _secs("2000-03-01"))
        outside = (_secs("2001-02-01"), _secs("2001-03-01"))
        for term in (PrescribedSurfaceFlux(),
                     SpeedySurfaceFlux(prescribed_fluxes=True)):
            term.validate_forcing(forcing, run_window=inside)
            # No concrete window (traced sim_time): nothing to check.
            term.validate_forcing(forcing, run_window=None)
            with pytest.raises(ValueError, match="BY_DATE"):
                term.validate_forcing(forcing, run_window=outside)
        # Interactive SPEEDY surface ignores the prescribed fields entirely.
        SpeedySurfaceFlux().validate_forcing(forcing, run_window=outside)

    def _forced_model(self, start):
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
                     physics=physics, time_step=20,
                     start_time=start)

    def test_model_run_window_is_absolute(self):
        from jcm.model import _run_window_seconds
        model = self._forced_model("2000-03-01")
        start, end = _run_window_seconds(model.start_time, 10 * 86400)
        assert start == pytest.approx(_secs("2000-03-01"))
        assert end == pytest.approx(_secs("2000-03-11"))

    def test_model_run_outside_archive_fails_at_start(self):
        """End to end: ``Model.run`` passes its window to validate_forcing, so
        a BY_DATE archive of 2000 fails a 2001 run before compiling.
        """
        from jcm.forcing import BY_DATE, make_time_series
        model = self._forced_model("2001-06-01")
        nodal = model.coords.horizontal.nodal_shape
        leaf = make_time_series(jnp.zeros((12, *nodal)),
                                _dates(_monthly_seconds(2000)), BY_DATE)
        forcing = default_forcing(model.coords.horizontal).copy(
            prescribed_sensible_heat_flux=leaf, prescribed_evaporation=leaf,
            prescribed_stress_u=leaf, prescribed_stress_v=leaf)
        with pytest.raises(ValueError, match="run covers 2001-06-01"):
            model.run(forcing=forcing, save_interval=(1 / 24.0),
                      total_time=(1 / 24.0))


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


# ---------------------------------------------------------------------------
# Prescribed fluxes need a forced-mode consumer (never silently ignored)
# ---------------------------------------------------------------------------

class TestPrescribedFluxNeedsConsumer:
    """Supplied ``prescribed_*`` fields with no term that reads them are
    rejected, judged by the declared ``consumed_forcing_fields`` capability.
    """

    def _forcing(self, with_fluxes=True):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        forcing = default_forcing(coords.horizontal)
        if not with_fluxes:
            return forcing
        nodal = coords.horizontal.nodal_shape
        return forcing.copy(
            prescribed_sensible_heat_flux=jnp.full(nodal, 10.0),
            prescribed_evaporation=jnp.full(nodal, 3e-5),
            prescribed_stress_u=jnp.full(nodal, 0.05),
            prescribed_stress_v=jnp.zeros(nodal))

    def test_speedy_interactive_preset_rejects_fluxes(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.physics.surface.prescribed_flux import (
            check_prescribed_flux_consumers,
        )
        with pytest.raises(ValueError, match="forcing.prescribed_surface_flux"
                           ".*speedy-forced-flux"):
            check_prescribed_flux_consumers(speedy_physics(), self._forcing())

    def test_echam_interactive_preset_rejects_fluxes(self):
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.surface.prescribed_flux import (
            check_prescribed_flux_consumers,
        )
        with pytest.raises(ValueError, match="echam-forced-flux"):
            check_prescribed_flux_consumers(echam_physics(), self._forcing())

    def test_forced_compositions_accept_fluxes(self):
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_terms import (
            SpeedySurfaceFlux, speedy_physics,
        )
        from jcm.physics.surface.prescribed_flux import (
            PRESCRIBED_FLUX_FORCING_FIELDS, check_prescribed_flux_consumers,
        )
        speedy_forced = speedy_physics().replace(
            "surface", SpeedySurfaceFlux(prescribed_fluxes=True))
        echam_forced = echam_physics(prescribed_surface_fluxes=True)
        for physics in (speedy_forced, echam_forced):
            assert set(PRESCRIBED_FLUX_FORCING_FIELDS) <= set(
                physics.consumed_forcing_fields())
            check_prescribed_flux_consumers(physics, self._forcing())

    def test_no_fluxes_is_unchanged(self):
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.physics.surface.prescribed_flux import (
            check_prescribed_flux_consumers,
        )
        bare = self._forcing(with_fluxes=False)
        for physics in (speedy_physics(), echam_physics(), object()):
            check_prescribed_flux_consumers(physics, bare)
        check_prescribed_flux_consumers(speedy_physics(), None)

    def test_capability_survives_composition_edits(self):
        """Removing the consumer re-arms the check; a user-defined term that
        DECLARES the fields is honoured without any class-name matching.
        """
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.physics_term import PhysicsTerm
        from jcm.physics.surface.prescribed_flux import (
            PRESCRIBED_FLUX_FORCING_FIELDS, check_prescribed_flux_consumers,
        )
        forced = echam_physics(prescribed_surface_fluxes=True)
        stripped = forced.remove("prescribed_surface_flux")
        with pytest.raises(ValueError, match="no term"):
            check_prescribed_flux_consumers(stripped, self._forcing())

        class MyCouplerFlux(PhysicsTerm):
            name = "my_coupler_flux"
            category = "my_coupler_flux"

            def consumed_forcing_fields(self):
                return PRESCRIBED_FLUX_FORCING_FIELDS

            def __call__(self, state, diagnostics, forcing, terrain):
                raise NotImplementedError

        check_prescribed_flux_consumers(stripped + MyCouplerFlux(),
                                        self._forcing())

    def test_physics_without_hook_rejects_fluxes(self):
        from jcm.physics.surface.prescribed_flux import (
            check_prescribed_flux_consumers,
        )
        with pytest.raises(ValueError, match="no term"):
            check_prescribed_flux_consumers(object(), self._forcing())

    def test_model_run_rejects_unconsumed_fluxes(self):
        """End to end through the Python door: ``Model.run`` refuses before
        compiling instead of running interactive SPEEDY on ignored fluxes.
        """
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.terrain import TerrainData
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        model = Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                      physics=speedy_physics(), time_step=20)
        with pytest.raises(ValueError, match="no term in the composed"):
            model.run(forcing=self._forcing(), save_interval=(1 / 24.0),
                      total_time=(1 / 24.0))

    def test_cli_assembly_rejects_block_with_interactive_preset(self):
        """The CLI door: the block assembled for an interactive preset is
        refused right after forcing assembly (``runners`` guard).
        """
        from omegaconf import OmegaConf

        from jcm import runners
        from jcm.forcing_assembly import _attach_prescribed_surface_fluxes
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        cfg = OmegaConf.create({"prescribed_surface_flux": {"constants": {
            "sensible_heat_flux": 10.0, "evaporation": 3e-5,
            "stress_u": 0.05, "stress_v": 0.0}}})
        forcing = _attach_prescribed_surface_fluxes(
            default_forcing(coords.horizontal), cfg, coords)
        with pytest.raises(ValueError, match="forcing.prescribed_surface_flux"):
            runners.validate_run_forcing(speedy_physics(), forcing)

    def test_scm_cli_refuses_forced_flux(self):
        from omegaconf import OmegaConf

        from jcm.physics.speedy.speedy_terms import (
            SpeedySurfaceFlux, speedy_physics,
        )
        from jcm.runners import _reject_forced_flux_in_scm
        block = OmegaConf.create({"forcing": {"prescribed_surface_flux": {
            "constants": {"sensible_heat_flux": 1.0}}}})
        with pytest.raises(ValueError, match="run.mode=scm"):
            _reject_forced_flux_in_scm(block, speedy_physics())
        forced = speedy_physics().replace(
            "surface", SpeedySurfaceFlux(prescribed_fluxes=True))
        none_block = OmegaConf.create(
            {"forcing": {"prescribed_surface_flux": None}})
        with pytest.raises(ValueError, match="forced mode"):
            _reject_forced_flux_in_scm(none_block, forced)
        _reject_forced_flux_in_scm(none_block, speedy_physics())
        _reject_forced_flux_in_scm(OmegaConf.create({}), speedy_physics())


# ---------------------------------------------------------------------------
# Coverage slack comes from the END intervals (or declared time_bnds)
# ---------------------------------------------------------------------------

class TestByDateCoverageEndIntervals:
    """An interior gap never widens the usable window (Codex #877)."""

    def _gappy_daily(self):
        """Daily Jan 1..10 2000, a ~year-long gap, then daily Dec 1..31."""
        import numpy as np

        from jcm.forcing import BY_DATE, make_time_series
        days = ([f"2000-01-{d:02d}" for d in range(1, 11)]
                + [f"2000-12-{d:02d}" for d in range(1, 32)])
        t = np.array([_secs(d) for d in days])
        return make_time_series(jnp.zeros((t.size, 2, 2)), _dates(t),
                                BY_DATE)

    def test_run_past_end_cadence_is_rejected(self):
        from jcm.forcing import by_date_coverage_error
        ts = self._gappy_daily()
        # Last sample Dec 31 00:00, daily cadence: usable to Jan 1 00:00.
        assert by_date_coverage_error(
            ts, _secs("2000-12-15"), _secs("2001-01-03")) is not None
        assert by_date_coverage_error(
            ts, _secs("2000-12-15"), _secs("2001-06-01")) is not None
        # Leading edge: one day before Jan 1, not the gap's ~325 days.
        assert by_date_coverage_error(
            ts, _secs("1999-12-25"), _secs("2000-01-05")) is not None

    def test_run_within_end_interval_is_accepted(self):
        from jcm.forcing import by_date_coverage_error
        ts = self._gappy_daily()
        assert by_date_coverage_error(
            ts, _secs("2000-12-15"), _secs("2001-01-01")) is None
        assert by_date_coverage_error(
            ts, _secs("1999-12-31"), _secs("2000-01-05")) is None

    def test_declared_bounds_win(self):
        from jcm.forcing import BY_DATE, by_date_coverage_error
        from jcm.forcing import make_time_series
        # Mid-month stamps: the cadence rule would allow to ~Jan 14 2001.
        ts = make_time_series(jnp.zeros((12, 2, 2)),
                              _dates(_monthly_seconds(2000, 15)), BY_DATE)
        late = (_secs("2000-12-20"), _secs("2001-01-10"))
        assert by_date_coverage_error(ts, *late) is None
        bounds = (_secs("2000-01-01"), _secs("2001-01-01"))
        err = by_date_coverage_error(ts, *late, bounds=bounds)
        assert err is not None and "time_bnds" in err
        assert by_date_coverage_error(
            ts, _secs("2000-01-01"), _secs("2001-01-01"),
            bounds=bounds) is None

    def _write_bounded(self, path, coords, bounds_ok=True):
        import numpy as np
        import pandas as pd
        import xarray as xr
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        starts = pd.date_range("2000-01-01", periods=12, freq="MS")
        ends = pd.date_range("2000-02-01", periods=12, freq="MS")
        mids = starts + (ends - starts) / 2
        if not bounds_ok:
            ends = starts  # zero-length intervals that miss their samples
        data = {v: (("time", "lat", "lon"), np.ones((12, nlat, nlon)))
                for v in ("sensible_heat_flux", "evaporation",
                          "stress_u", "stress_v")}
        data["time_bnds"] = (("time", "nv"),
                             np.stack([starts.values, ends.values], axis=1))
        ds = xr.Dataset(data, coords={"time": mids.values, "lat": lat,
                                      "lon": lon})
        ds["time"].attrs["bounds"] = "time_bnds"
        # One units encoding for time and its bounds, as CF requires.
        ds["time"].encoding["units"] = "hours since 2000-01-01"
        ds.to_netcdf(path)

    def test_reader_carries_time_bnds_to_the_run_check(self, tmp_path):
        import xarray as xr

        from jcm.forcing import read_prescribed_surface_fluxes
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import SpeedySurfaceFlux
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        p = tmp_path / "bounded.nc"
        self._write_bounded(p, coords)
        import numpy as np
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        with xr.open_dataset(p) as ds:
            fields = read_prescribed_surface_fluxes(
                ds, lat, lon, align_mode="by_date", source=str(p))
        b = _bound_seconds(fields["prescribed_flux_time_bounds"])
        assert b.shape == (12, 2)
        assert float(b.min()) == pytest.approx(_secs("2000-01-01"))
        assert float(b.max()) == pytest.approx(_secs("2001-01-01"))
        forcing = default_forcing(coords.horizontal).copy(**fields)
        term = SpeedySurfaceFlux(prescribed_fluxes=True)
        term.validate_forcing(
            forcing, run_window=(_secs("2000-01-01"), _secs("2001-01-01")))
        with pytest.raises(ValueError, match="time_bnds"):
            term.validate_forcing(
                forcing, run_window=(_secs("2000-12-20"), _secs("2001-01-10")))
        # A climatology has no absolute coverage, so no bounds are carried.
        with xr.open_dataset(p) as ds:
            clim = read_prescribed_surface_fluxes(
                ds, lat, lon, align_mode="wrap_year", source=str(p))
        assert "prescribed_flux_time_bounds" not in clim

    def test_reader_rejects_bounds_that_miss_their_samples(self, tmp_path):
        import numpy as np
        import xarray as xr

        from jcm.forcing import read_prescribed_surface_fluxes
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        p = tmp_path / "bad_bounds.nc"
        self._write_bounded(p, coords, bounds_ok=False)
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        with xr.open_dataset(p) as ds, pytest.raises(
                ValueError, match="do not bracket"):
            read_prescribed_surface_fluxes(
                ds, lat, lon, align_mode="by_date", source=str(p))


# ---------------------------------------------------------------------------
# Every run entry point enforces BOTH directions of the forced-mode contract
# ---------------------------------------------------------------------------

def _entry_setup(forced, fluxes):
    """(coords, physics, forcing) for one scenario of the entry-point matrix.

    ``fluxes``: ``None`` (no prescribed fields), ``"static"`` (uniform maps)
    or ``"archive2000"`` (a BY_DATE monthly archive of the year 2000).
    """
    from jcm.forcing import BY_DATE, make_time_series
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import (
        SpeedySurfaceFlux, speedy_physics,
    )
    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    physics = (speedy_physics().replace(
        "surface", SpeedySurfaceFlux(prescribed_fluxes=True))
        if forced else speedy_physics())
    forcing = default_forcing(coords.horizontal)
    nodal = coords.horizontal.nodal_shape
    if fluxes == "static":
        leaf = jnp.full(nodal, 1.0)
    elif fluxes == "archive2000":
        leaf = make_time_series(jnp.ones((12, *nodal)),
                                _dates(_monthly_seconds(2000)), BY_DATE)
    else:
        return coords, physics, forcing
    return coords, physics, forcing.copy(
        prescribed_sensible_heat_flux=leaf, prescribed_evaporation=leaf,
        prescribed_stress_u=leaf, prescribed_stress_v=leaf)


def _door_model(method):
    def call(coords, physics, forcing, start):
        from jcm.model import Model
        from jcm.terrain import TerrainData
        model = Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                      physics=physics, time_step=20,
                      start_time=start)
        kw = dict(save_interval=(1 / 24.0), total_time=(1 / 24.0))
        if method == "run":
            return model.run(forcing=forcing, **kw)
        state = model._prepare_initial_dycore_state()
        if method == "run_from_state_with_carry":
            # The low-level door takes the complete RunState clock.
            kw.update(initial_time=model.start_time, initial_step=0)
        return getattr(model, method)(state, forcing, **kw)
    return call


def _door_scm(coords, physics, forcing, start):
    from jcm.single_column_model import SingleColumnModel
    scm = SingleColumnModel(physics=physics, vertical=coords.vertical,
                            dt_seconds=1200.0)
    # The contract is checked before the states are touched.
    return scm.run(None, forcing=forcing)


def _door_prescribed(coords, physics, forcing, start):
    from jcm.prescribed_state_model import PrescribedStateModel
    from jcm.prescribed_state_model_test import _make_test_state
    model = PrescribedStateModel(physics=physics, coords=coords,
                                 start_time=start)
    return model.run([_make_test_state(coords)], forcing=forcing)


def _door_cli(coords, physics, forcing, start):
    # The CLI/recipe doors (_run_full, _run_prescribed, configurations) call
    # this re-export right after assembly, with the CONFIGURED window
    # (``runners.configured_run_window``: run.start_time + total_time, #900).
    from jcm import runners
    runners.validate_run_forcing(
        physics, forcing, run_window=(_secs(start), _secs(start) + 3600.0))
    raise AssertionError("reached: no contract violation raised")


#: (door, has a concrete run window)
_DOORS = {
    "Model.run": (_door_model("run"), True),
    "Model.run_from_state": (_door_model("run_from_state"), True),
    "Model.run_from_state_with_carry": (
        _door_model("run_from_state_with_carry"), True),
    "SingleColumnModel.run": (_door_scm, False),
    "PrescribedStateModel.run": (_door_prescribed, True),
    "runners/configurations (CLI)": (_door_cli, True),
}


@pytest.mark.parametrize("door", list(_DOORS))
def test_entry_point_rejects_unconsumed_fluxes(door):
    """Direction (a): fluxes supplied to an interactive composition."""
    coords, physics, forcing = _entry_setup(forced=False, fluxes="static")
    with pytest.raises(ValueError, match="no term in the composed physics"):
        _DOORS[door][0](coords, physics, forcing, "2000-03-01")


@pytest.mark.parametrize("door", list(_DOORS))
def test_entry_point_rejects_forced_physics_without_fluxes(door):
    """Direction (b): a forced consumer with no prescribed fields."""
    coords, physics, forcing = _entry_setup(forced=True, fluxes=None)
    with pytest.raises(ValueError, match="are None"):
        _DOORS[door][0](coords, physics, forcing, "2000-03-01")


@pytest.mark.parametrize(
    "door", [d for d, (_, windowed) in _DOORS.items() if windowed])
def test_entry_point_rejects_uncovered_archive(door):
    """Direction (b), coverage: a 2000 BY_DATE archive cannot drive 2001."""
    coords, physics, forcing = _entry_setup(forced=True, fluxes="archive2000")
    with pytest.raises(ValueError, match="BY_DATE"):
        _DOORS[door][0](coords, physics, forcing, "2001-06-01")


def _dated_2000_forcing(coords, which):
    """Aquaplanet forcing whose ``which`` input is a BY_DATE archive of 2000.

    Built directly as ``TimeSeries`` leaves (the readers are covered in
    ``jcm/forcing_test.py::TestDatedInputPersistence``); what is under test
    here is that every door judges every input family.
    """
    from jcm.forcing import BY_DATE, BY_DATE_INTERP, make_time_series
    from jcm.ozone_climatology import OzoneClimatology
    forcing = default_forcing(coords.horizontal)
    nodal = coords.horizontal.nodal_shape
    nlev = coords.nodal_shape[0]
    t = _dates(_monthly_seconds(2000, day=15))

    def ts(*shape, mode=BY_DATE):
        return make_time_series(jnp.ones((12, *shape)), t, mode)

    if which == "sst":
        return forcing.copy(
            sea_surface_temperature=ts(*nodal, mode=BY_DATE_INTERP))
    if which == "ozone":
        return forcing.copy(ozone_climatology=OzoneClimatology(
            o3_ppmv=ts(nlev, nodal[0] * nodal[1], mode=BY_DATE_INTERP)))
    if which == "emissions":
        return forcing.copy(anthropogenic_emissions={
            "emis_ene_so2": ts(*nodal)})
    if which == "oxidants":
        return forcing.copy(oxidant_vmr={
            k: ts(nlev, *nodal) for k in ("oh", "no3", "o3", "h2o2")})
    if which == "macv2":
        return forcing.copy(aerosol_year_weight=make_time_series(
            jnp.ones((1, 9)), _dates([_secs("2000-01-01")]), BY_DATE))
    raise AssertionError(which)


#: input -> the knob its coverage error names
_DATED_INPUT_KNOBS = {
    "sst": "forcing.persist",
    "ozone": "forcing.ozone_persist",
    "emissions": "forcing.emissions_persist",
    "oxidants": "forcing.oxidants_persist",
    "macv2": "forcing.macv2_persist",
}


@pytest.mark.parametrize("which", list(_DATED_INPUT_KNOBS))
@pytest.mark.parametrize(
    "door", [d for d, (_, windowed) in _DOORS.items() if windowed])
def test_entry_point_rejects_uncovered_dated_input(door, which):
    """#900: a 2000 archive of ANY dated input cannot drive a 2001 run, at
    any door, and the error names that input's persist knob — before the
    run is compiled.
    """
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics
    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    forcing = _dated_2000_forcing(coords, which)
    with pytest.raises(ValueError,
                       match=f"{_DATED_INPUT_KNOBS[which]}=hold"):
        _DOORS[door][0](coords, speedy_physics(), forcing, "2001-06-01")


def test_cli_door_holds_a_declared_input_with_a_warning():
    """The same 2001 run passes the CLI door once the input declares hold."""
    import warnings

    from jcm import runners
    from jcm.forcing import BY_DATE, _HOLD_WARNED, make_time_series
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics
    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    nodal = coords.horizontal.nodal_shape
    leaf = make_time_series(jnp.ones((12, *nodal)),
                            _dates(_monthly_seconds(2000, day=15)), BY_DATE,
                            persist="hold")
    forcing = default_forcing(coords.horizontal).copy(
        anthropogenic_emissions={"emis_ene_so2": leaf})
    _HOLD_WARNED.clear()
    window = (_secs("2001-06-01"), _secs("2001-06-02"))
    with pytest.warns(UserWarning, match="emissions_persist=hold"):
        runners.validate_run_forcing(speedy_physics(), forcing,
                                     run_window=window)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # warned once per held input
        runners.validate_run_forcing(speedy_physics(), forcing,
                                     run_window=window)


def test_cli_configured_window_spans_the_whole_run():
    """``configured_run_window``: run.start_time through total/end time."""
    import types

    from jcm.date import to_datetime
    from jcm.runners import configured_run_window
    from jcm.runners_test import _compose
    model = types.SimpleNamespace(start_time=to_datetime("2001-03-01"))
    cfg = _compose(["run.total_time=10"])
    assert configured_run_window(cfg, model) == pytest.approx(
        (_secs("2001-03-01"), _secs("2001-03-11")))
    cfg = _compose(["run.total_time=null", "run.end_time=2002-03-01"])
    assert configured_run_window(cfg, model) == pytest.approx(
        (_secs("2001-03-01"), _secs("2002-03-01")))


def test_every_entry_point_calls_the_shared_contract_helper():
    """Structural guard: each door that steps physics on a forcing calls
    ``validate_run_forcing``, and the ``Model`` doors funnel into the one
    that does, so a new door cannot silently skip the contract.
    """
    import inspect

    from jcm import configurations, runners
    from jcm.model import Model
    from jcm.prescribed_state_model import PrescribedStateModel
    from jcm.single_column_model import SingleColumnModel
    for fn in (Model.run_from_state_with_carry, SingleColumnModel.run,
               PrescribedStateModel.run, runners._run_full,
               runners._run_prescribed, configurations.load):
        assert "validate_run_forcing(" in inspect.getsource(fn), fn
    # The CLI doors that know the run window pass it (#900): the configured
    # one for a full run, the states' span for prescribed mode.
    for fn in (runners._run_full, configurations.load):
        assert "configured_run_window(" in inspect.getsource(fn), fn
    assert "_run_window_seconds(times)" in inspect.getsource(
        runners._run_prescribed)
    for fn in (Model.run_from_state, Model.resume):
        assert "run_from_state_with_carry(" in inspect.getsource(fn), fn
    assert "self.resume(" in inspect.getsource(Model.run)
    # The SCM CLI builds no ForcingData, so it refuses forced flux outright.
    assert "_reject_forced_flux_in_scm(" in inspect.getsource(runners._run_scm)
    # run_chunked steps through model.run (checked above).
    assert "model.run(" in inspect.getsource(runners.run_chunked)


# ---------------------------------------------------------------------------
# Declared time_bnds gaps (Codex #877): kept per interval, never collapsed
# ---------------------------------------------------------------------------

class TestDeclaredBoundsGaps:
    """Disjoint CF bounds are a declared gap the run window may not cross."""

    def _ts(self):
        from jcm.forcing import BY_DATE, make_time_series
        t = _dates([_secs("2000-01-15"), _secs("2000-02-15"),
                    _secs("2000-06-15"), _secs("2000-07-15")])
        return make_time_series(jnp.zeros((4, 2, 2)), t, BY_DATE)

    def _bounds(self, contiguous):
        """Four monthly intervals; the gappy set has no March-May coverage."""
        if contiguous:
            edges = [("2000-01-01", "2000-02-01"), ("2000-02-01", "2000-06-01"),
                     ("2000-06-01", "2000-07-01"), ("2000-07-01", "2000-08-01")]
        else:
            edges = [("2000-01-01", "2000-02-01"), ("2000-02-01", "2000-03-01"),
                     ("2000-06-01", "2000-07-01"), ("2000-07-01", "2000-08-01")]
        import numpy as np
        return np.asarray([[_secs(a), _secs(b)] for a, b in edges])

    def test_contiguous_bounds_accepted(self):
        from jcm.forcing import by_date_coverage_error
        assert by_date_coverage_error(
            self._ts(), _secs("2000-01-01"), _secs("2000-08-01"),
            bounds=self._bounds(True)) is None

    def test_gap_inside_window_rejected(self):
        from jcm.forcing import by_date_coverage_error
        err = by_date_coverage_error(
            self._ts(), _secs("2000-02-10"), _secs("2000-06-20"),
            bounds=self._bounds(False))
        assert err is not None and "declare a gap from 2000-03-01" in err
        # A run starting INSIDE the gap is rejected too.
        assert by_date_coverage_error(
            self._ts(), _secs("2000-04-01"), _secs("2000-06-20"),
            bounds=self._bounds(False)) is not None

    def test_gap_outside_window_accepted(self):
        from jcm.forcing import by_date_coverage_error
        gappy = self._bounds(False)
        assert by_date_coverage_error(
            self._ts(), _secs("2000-01-05"), _secs("2000-02-25"),
            bounds=gappy) is None
        assert by_date_coverage_error(
            self._ts(), _secs("2000-06-05"), _secs("2000-07-31"),
            bounds=gappy) is None

    def test_reader_keeps_the_intervals(self, tmp_path):
        import numpy as np
        import pandas as pd
        import xarray as xr

        from jcm.forcing import read_prescribed_surface_fluxes
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        starts = pd.to_datetime(["2000-01-01", "2000-06-01"])
        ends = pd.to_datetime(["2000-02-01", "2000-07-01"])
        mids = starts + (ends - starts) / 2
        data = {v: (("time", "lat", "lon"), np.ones((2, nlat, nlon)))
                for v in ("sensible_heat_flux", "evaporation",
                          "stress_u", "stress_v")}
        data["time_bnds"] = (("time", "nv"),
                             np.stack([starts.values, ends.values], axis=1))
        ds = xr.Dataset(data, coords={"time": mids.values, "lat": lat,
                                      "lon": lon})
        ds["time"].attrs["bounds"] = "time_bnds"
        ds["time"].encoding["units"] = "hours since 2000-01-01"
        p = tmp_path / "gappy.nc"
        ds.to_netcdf(p)
        with xr.open_dataset(p) as opened:
            fields = read_prescribed_surface_fluxes(
                opened, lat, lon, align_mode="by_date", source=str(p))
        b = _bound_seconds(fields["prescribed_flux_time_bounds"])
        assert b.shape == (2, 2)
        assert b[0, 1] == pytest.approx(_secs("2000-02-01"))
        assert b[1, 0] == pytest.approx(_secs("2000-06-01"))


# ---------------------------------------------------------------------------
# run.mode=prescribed end to end: the state file's own clock drives forcing
# ---------------------------------------------------------------------------

def _daily_state_ds(n, start="1970-01-01", times=None):
    """Return a state-file stand-in carrying only its ``time`` coordinate."""
    import pandas as pd
    import xarray as xr
    t = (pd.date_range(start, periods=n, freq="D").values
         if times is None else times)
    return xr.Dataset(coords={"time": t})


class TestPrescribedModeStateClock:
    """``_prescribed_state_times_days`` and the prescribed driver's clock."""

    def test_daily_snapshots_give_day_offsets(self):
        from jcm.runners import _prescribed_state_times_days
        days = _prescribed_state_times_days(_daily_state_ds(3), 3, "f")
        assert list(days) == [0.0, 1.0, 2.0]

    def test_irregular_cadence_is_honoured(self):
        import numpy as np
        from jcm.runners import _prescribed_state_times_days
        t = np.array(["1970-01-01", "1970-01-02", "1970-01-05"],
                     dtype="datetime64[ns]")
        days = _prescribed_state_times_days(_daily_state_ds(3, times=t), 3, "f")
        assert list(days) == [0.0, 1.0, 4.0]

    def test_cftime_axis_uses_nominal_dates(self):
        import cftime
        import numpy as np
        from jcm.runners import _prescribed_state_times_days
        t = np.array([cftime.DatetimeNoLeap(2001, 2, 28),
                      cftime.DatetimeNoLeap(2001, 3, 1)], dtype=object)
        days = _prescribed_state_times_days(_daily_state_ds(2, times=t), 2, "f")
        assert list(days) == [0.0, 1.0]

    def test_single_state_needs_no_time_axis(self):
        import xarray as xr
        from jcm.runners import _prescribed_state_times_days
        assert list(_prescribed_state_times_days(xr.Dataset(), 1, "f")) == [0.0]

    @pytest.mark.parametrize("case,match", [
        ("missing", "no 'time' coordinate"),
        ("length", "entries for 3 states"),
        ("numeric", "unit is unknown"),
        ("numeric_nan", "non-finite"),
        ("nat", "missing"),
        ("unordered", "not strictly increasing"),
        ("duplicate", "not strictly increasing"),
    ])
    def test_bad_time_axis_rejected(self, case, match):
        import numpy as np
        import xarray as xr
        from jcm.runners import _prescribed_state_times_days
        d = lambda *x: np.array(x, dtype="datetime64[ns]")  # noqa: E731
        ds = {
            "missing": xr.Dataset(),
            "length": _daily_state_ds(2),
            "numeric": xr.Dataset(coords={"time": [0.0, 1.0, 2.0]}),
            "numeric_nan": xr.Dataset(coords={"time": (
                "time", [0.0, np.nan, 2.0], {"units": "d"})}),
            "nat": xr.Dataset(coords={"time": d("1970-01-01", "NaT",
                                                "1970-01-03")}),
            "unordered": xr.Dataset(coords={"time": d(
                "1970-01-01", "1970-01-03", "1970-01-02")}),
            "duplicate": xr.Dataset(coords={"time": d(
                "1970-01-01", "1970-01-01", "1970-01-02")}),
        }[case]
        with pytest.raises(ValueError, match=match):
            _prescribed_state_times_days(ds, 3, "f")

    @pytest.mark.parametrize("units,scale", [
        ("d", 1.0), ("days", 1.0), ("s", 86400.0), ("seconds", 86400.0)])
    def test_numeric_elapsed_axis_matches_datetime_axis(self, units, scale):
        """A numeric elapsed-time axis gives the same offsets as the
        equivalent datetime axis (seconds are converted).
        """
        import numpy as np
        import xarray as xr
        from jcm.runners import _prescribed_state_times_days
        elapsed = np.array([0.0, 1.0, 4.0]) * scale
        numeric = xr.Dataset(coords={"time": ("time", elapsed,
                                              {"units": units})})
        t = np.array(["1970-01-01", "1970-01-02", "1970-01-05"],
                     dtype="datetime64[ns]")
        assert list(_prescribed_state_times_days(numeric, 3, "f")) == list(
            _prescribed_state_times_days(_daily_state_ds(3, times=t), 3, "f"))

    def test_timedelta_axis_accepted(self):
        import numpy as np
        import xarray as xr
        from jcm.runners import _prescribed_state_times_days
        td = np.array([0, 12, 36], dtype="timedelta64[h]").astype(
            "timedelta64[ns]")
        days = _prescribed_state_times_days(
            xr.Dataset(coords={"time": td}), 3, "f")
        assert list(days) == [0.0, 0.5, 1.5]

    @pytest.mark.parametrize("units", [None, "hours", "days since 2000-01-01",
                                       "months"])
    def test_numeric_axis_without_known_units_rejected(self, units):
        import xarray as xr
        from jcm.runners import _prescribed_state_times_days
        attrs = {} if units is None else {"units": units}
        ds = xr.Dataset(coords={"time": ("time", [0.0, 1.0], attrs)})
        with pytest.raises(ValueError, match="numeric 'time' coordinate"):
            _prescribed_state_times_days(ds, 2, "f")

    def test_numeric_axis_round_trips_the_writer_convention(self, tmp_path):
        """What ``cf_metadata`` writes for a numeric elapsed axis (``units:
        "d"``) survives netCDF and is read back as elapsed days.
        """
        import numpy as np
        import xarray as xr
        from jcm import cf_metadata
        from jcm.runners import _prescribed_state_times_days
        ds = cf_metadata.apply_cf_attributes(
            xr.Dataset(coords={"time": np.array([10.0, 11.0, 13.0])}))
        assert ds["time"].attrs["units"] == "d"
        p = tmp_path / "numeric_time.nc"
        ds.to_netcdf(p)
        with xr.open_dataset(p) as opened:
            days = _prescribed_state_times_days(opened, 3, str(p))
        assert list(days) == [0.0, 1.0, 3.0]

    def test_times_length_must_match_states(self):
        from jcm.prescribed_state_model import PrescribedStateModel
        from jcm.prescribed_state_model_test import _make_test_state
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        model = PrescribedStateModel(physics=speedy_physics(), coords=coords)
        with pytest.raises(ValueError, match="2 times for 1 states"):
            model.run([_make_test_state(coords)], times=jnp.asarray([0., 1.]))

    def test_traced_times_give_no_window(self):
        import jax
        from jcm.prescribed_state_model import PrescribedStateModel
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        model = PrescribedStateModel(physics=speedy_physics(), coords=coords)
        seen = []

        @jax.jit
        def f(t):
            seen.append(model._run_window_seconds(t))
            return t
        f(jnp.asarray([0.0, 1.0]))
        assert seen == [None]
        start, end = model._run_window_seconds(jnp.asarray([0.0, 2.0]))
        assert (start, end) == (pytest.approx(_secs("2000-01-01")),
                                pytest.approx(_secs("2000-01-03")))

    def test_full_mode_only_knobs_rejected(self):
        from omegaconf import OmegaConf
        from jcm.runners import _reject_full_mode_only_knobs
        for extra in ({"chunk_days": 5}, {"checkpoint_path": "x.ckpt"}):
            cfg = OmegaConf.create({"run": {"mode": "prescribed", **extra}})
            with pytest.raises(ValueError, match="apply only to run.mode=full"):
                _reject_full_mode_only_knobs(cfg)
        _reject_full_mode_only_knobs(OmegaConf.create(
            {"run": {"mode": "prescribed", "chunk_days": 0,
                     "checkpoint_path": None}}))

    @pytest.mark.parametrize("state_file", [["a.nc", "b.nc"], "s_{year}.nc"])
    def test_single_state_file_required(self, state_file):
        from omegaconf import OmegaConf
        from jcm.runners import _load_states_from_cfg
        cfg = OmegaConf.create({"run": {"mode": "prescribed",
                                        "state_file": state_file}})
        with pytest.raises(ValueError, match="ONE netCDF state file"):
            _load_states_from_cfg(cfg, None)


class TestPrescribedModeForcedFlux:
    """Forced fluxes in ``run.mode=prescribed`` follow the state times."""

    def _archive_forcing(self, coords, n_days=5):
        from jcm.forcing import BY_DATE, make_time_series
        nodal = coords.horizontal.nodal_shape
        t = _dates([_secs(f"2000-01-{d + 1:02d}") for d in range(n_days)])
        vals = jnp.stack([jnp.full(nodal, float(k + 1)) for k in range(n_days)])
        leaf = make_time_series(vals, t, BY_DATE)
        return default_forcing(coords.horizontal).copy(
            prescribed_sensible_heat_flux=leaf, prescribed_evaporation=leaf,
            prescribed_stress_u=leaf, prescribed_stress_v=leaf)

    def _model(self, coords):
        from jcm.physics.speedy.speedy_terms import (
            SpeedySurfaceFlux, speedy_physics,
        )
        from jcm.prescribed_state_model import PrescribedStateModel
        physics = speedy_physics().replace(
            "surface", SpeedySurfaceFlux(prescribed_fluxes=True))
        return PrescribedStateModel(
            physics=physics, coords=coords, dt_seconds=3 * 3600.0,
            start_time="2000-01-01")

    def test_daily_snapshots_select_daily_archive_samples(self):
        """Daily snapshots on a 3-hour step: snapshot k sees day-k fluxes."""
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.prescribed_state_model_test import _make_test_state
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        state = _make_test_state(coords)
        preds = self._model(coords).run(
            [state, state, state], forcing=self._archive_forcing(coords),
            times=jnp.asarray([0.0, 1.0, 2.0]))
        shf = preds.physics_data[SURFACE_EXCHANGE_KEY].sensible_heat_flux
        for k in range(3):
            assert float(jnp.mean(shf[k])) == pytest.approx(k + 1.0)

    def test_coverage_checked_against_the_state_span(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.prescribed_state_model_test import _make_test_state
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        state = _make_test_state(coords)
        # States span 30 days; the archive covers 5 (+1 cadence day).
        with pytest.raises(ValueError, match="BY_DATE"):
            self._model(coords).run(
                [state, state], forcing=self._archive_forcing(coords),
                times=jnp.asarray([0.0, 30.0]))

    def test_cli_driver_passes_file_times_and_start_time(self):
        """``_run_prescribed`` hands the file's own times (not arange*dt) and
        the configured start time to the model, and validates the forced
        contract over that window before physics.
        """
        from unittest import mock

        from hydra import compose, initialize_config_module

        from jcm import runners
        from jcm.prescribed_state_model_test import _make_test_state
        with initialize_config_module("jcm.config", version_base=None):
            cfg = compose("config", overrides=[
                "physics=speedy-forced-flux", "forcing.ozone_file=analytic",
                "run.mode=prescribed", "run.time_step=180",
                "run.start_time=2000-01-01", "run.state_file=unused.nc"])
        coords = runners.build_coords(cfg)
        state = _make_test_state(coords)
        from jax.tree_util import tree_map
        states = tree_map(lambda *a: jnp.stack(a), state, state, state)
        archive = self._archive_forcing(coords)
        seen = {}
        from jcm.prescribed_state_model import PrescribedStateModel
        real_run = PrescribedStateModel.run

        def spy_run(self, states, forcing=None, times=None):
            seen.update(times=times, start=self.start_time)
            return real_run(self, states, forcing=forcing, times=times)

        with mock.patch.object(runners, "build_forcing",
                               return_value=archive), \
                mock.patch.object(runners, "_load_states_from_cfg",
                                  return_value=(_daily_state_ds(3), states)), \
                mock.patch.object(PrescribedStateModel, "run", spy_run):
            preds = runners._run_prescribed(cfg)
        assert list(seen["times"]) == [0.0, 1.0, 2.0]
        assert str(seen["start"].to_datetime64()).startswith("2000-01-01")
        shf = preds.physics_data[SURFACE_EXCHANGE_KEY].sensible_heat_flux
        assert float(jnp.mean(shf[2])) == pytest.approx(3.0)
        # An archive that stops before the last snapshot fails BEFORE physics.
        with mock.patch.object(runners, "build_forcing",
                               return_value=self._archive_forcing(coords, 1)), \
                mock.patch.object(runners, "_load_states_from_cfg",
                                  return_value=(_daily_state_ds(3), states)), \
                mock.patch.object(PrescribedStateModel, "run") as never:
            with pytest.raises(ValueError, match="BY_DATE"):
                runners._run_prescribed(cfg)
            never.assert_not_called()

    def test_cli_forced_preset_with_nothing_attached_raises_value_error(self):
        """``kind: default`` + ``ozone_file: analytic`` assembles ``None``
        forcing; a forced preset then gets the actionable ValueError, not an
        AttributeError.
        """
        from hydra import compose, initialize_config_module

        from jcm import runners
        with initialize_config_module("jcm.config", version_base=None):
            cfg = compose("config", overrides=[
                "physics=speedy-forced-flux", "forcing.ozone_file=analytic"])
        coords = runners.build_coords(cfg)
        forcing = runners.build_forcing(cfg, coords)
        assert forcing is None
        with pytest.raises(ValueError, match="forcing.prescribed_surface_flux"):
            runners.validate_run_forcing(runners.build_physics(cfg), forcing)


# ---------------------------------------------------------------------------
# Edge-slack cadence forms (_repeat_cadence), incl. month-end stamps
# ---------------------------------------------------------------------------

class TestEdgeCadenceForms:
    """Each cadence form the edge slack recognises, and the documented
    elapsed-seconds fallback for the rest.
    """

    def _ts(self, dates):
        from jcm.forcing import BY_DATE, make_time_series
        t = _dates([_secs(d) for d in dates])
        return make_time_series(jnp.zeros((len(dates), 2, 2)), t, BY_DATE)

    @pytest.mark.parametrize("dates", [
        # Gregorian month ends through a leap February.
        ["2000-01-31", "2000-02-29"],
        # A noleap (365-day) archive's month ends: Feb 28 even in 2000.
        ["2000-01-31", "2000-02-28"],
    ])
    def test_month_end_monthly_archive(self, dates):
        from jcm.forcing import by_date_coverage_error
        ts = self._ts(dates)
        assert by_date_coverage_error(
            ts, _secs("1999-12-31"), _secs("2000-03-31")) is None
        assert by_date_coverage_error(
            ts, _secs("1999-12-30"), _secs("2000-03-01")) is not None
        assert by_date_coverage_error(
            ts, _secs("2000-01-15"), _secs("2000-04-01")) is not None

    def test_month_end_from_a_cftime_noleap_file(self, tmp_path):
        """A real cftime noleap axis (read through the flux reader) gets the
        month-end slack on the model clock.
        """
        import cftime
        import numpy as np
        import xarray as xr

        from jcm.forcing import (
            by_date_coverage_error, read_prescribed_surface_fluxes,
        )
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nlon, nlat = coords.horizontal.nodal_shape
        lat = np.degrees(np.asarray(coords.horizontal.latitudes))
        lon = np.degrees(np.asarray(coords.horizontal.longitudes))
        t = [cftime.DatetimeNoLeap(2001, 1, 31), cftime.DatetimeNoLeap(2001, 2, 28),
             cftime.DatetimeNoLeap(2001, 3, 31)]
        data = {v: (("time", "lat", "lon"), np.ones((3, nlat, nlon)))
                for v in ("sensible_heat_flux", "evaporation",
                          "stress_u", "stress_v")}
        ds = xr.Dataset(data, coords={"time": t, "lat": lat, "lon": lon})
        fields = read_prescribed_surface_fluxes(
            ds, lat, lon, align_mode="by_date", source="noleap")
        ts = fields["prescribed_sensible_heat_flux"]
        assert by_date_coverage_error(
            ts, _secs("2000-12-31"), _secs("2001-04-30")) is None
        assert by_date_coverage_error(
            ts, _secs("2000-12-30"), _secs("2001-04-30")) is not None

    def test_same_day_monthly_still_steps_calendar_months(self):
        from jcm.forcing import by_date_coverage_error
        ts = self._ts(["2000-11-01", "2000-12-01"])
        # One December (31 days) after Dec 1; one October before Nov 1.
        assert by_date_coverage_error(
            ts, _secs("2000-10-01"), _secs("2001-01-01")) is None
        assert by_date_coverage_error(
            ts, _secs("2000-09-30"), _secs("2000-12-15")) is not None

    def test_yearly_same_day_cadence(self):
        from jcm.forcing import by_date_coverage_error
        ts = self._ts(["2000-01-01", "2001-01-01"])
        assert by_date_coverage_error(
            ts, _secs("1999-01-01"), _secs("2002-01-01")) is None

    @pytest.mark.parametrize("a,b,origin,expected", [
        # Varying mid-month day (CMIP style): elapsed seconds, 29.5 days.
        ("2000-01-16 12:00", "2000-02-15 00:00", "2000-02-15 00:00",
         "2000-03-15 12:00"),
        # Month end next to a non-month-end: elapsed seconds.
        ("2000-01-30", "2000-01-31", "2000-01-31", "2000-02-01"),
        # Month ends at different times of day: elapsed seconds.
        ("2000-01-31 00:00", "2000-02-29 06:00", "2000-02-29 06:00",
         "2000-03-29 12:00"),
        # Daily: exact in seconds.
        ("2000-01-01", "2000-01-02", "2000-01-02", "2000-01-03"),
    ])
    def test_other_cadences_fall_back_to_elapsed_seconds(
            self, a, b, origin, expected):
        from jcm.forcing import _repeat_cadence
        assert _repeat_cadence(_secs(a), _secs(b), _secs(origin)) == \
            pytest.approx(_secs(expected))

    def test_month_end_steps_reanchor_to_month_end(self):
        from jcm.forcing import _repeat_cadence
        # Backwards from Jan 31 → Dec 31; forwards from Feb 29 → Mar 31.
        assert _repeat_cadence(_secs("2000-02-29"), _secs("2000-01-31"),
                               _secs("2000-01-31")) == pytest.approx(
            _secs("1999-12-31"))
        assert _repeat_cadence(_secs("2000-01-31"), _secs("2000-02-29"),
                               _secs("2000-02-29")) == pytest.approx(
            _secs("2000-03-31"))
