"""Emulated radiation scheme using bidirectional GRU neural networks.

Drop-in replacement for ``radiation_scheme_rrtmgp`` that uses trained neural
networks to predict shortwave and longwave fluxes. The NN weights are passed
as JAX arrays through the ``emulator_weights`` argument, making them
fully differentiable for gradient-based optimization.

Reference architecture: Ukkonen (2024), https://github.com/peterukk/rte-rrtmgp-nn

Date: 2026-04-11
"""

import dataclasses
from typing import Tuple, Optional

import jax.numpy as jnp

from jcm.physics.coords_util import column_lat_lon

from jcm.physics.radiation.radiation_types import (
    RadiationParameters,
    RadiationTendencies,
    RadiationData,
)
from jcm.physics.radiation.nn_emulator import (
    EmulatorWeights,
    InputScaling,
    init_emulator_weights,
    load_emulator_weights,
    preprocess_sw_inputs,
    preprocess_lw_inputs,
    lw_emulator_column,
    n_input_features,
    reconstruct_sw_interface_fluxes,
    reconstruct_lw_interface_fluxes,
    flux_to_heating_rate,
)
from jcm.physics.radiation.cloud_optics import resolve_effective_radii
from jcm.physics.radiation.mcica import in_cloud_path
from jcm.physics.radiation.rrtmgp import _MAX_IN_CLOUD_CONDENSATE
import jcm.constants as c


def _max_random_total_cover(cloud_fraction: jnp.ndarray) -> jnp.ndarray:
    """Analytic maximum-random total column cover from a cf profile.

    The standard ECHAM diagnostic::

        C = 1 - prod_k (1 - max(cf_k, cf_{k-1})) / (1 - cf_{k-1})

    Adjacency-only, so vertical orientation does not matter. This is the
    expectation of the McICA sub-column draw RRTMGP's ``total_cloud_cover``
    reports, without the sampling noise.
    """
    cf = jnp.clip(cloud_fraction, 0.0, 1.0)
    cf_prev = jnp.concatenate([cf[:1] * 0.0, cf[:-1]])
    # An overcast layer above drives the ratio to 0/eps -> product 0 ->
    # cover 1, which is the correct limit.
    ratio = (1.0 - jnp.maximum(cf, cf_prev)) / jnp.maximum(
        1.0 - cf_prev, 1e-6,
    )
    return 1.0 - jnp.prod(ratio, axis=0)


def radiation_scheme_emulated(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_albedo_vis: jnp.ndarray,
    surface_albedo_nir: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    solar,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6,
    emulator_weights: Optional[EmulatorWeights] = None,
    sw_scaling: Optional[InputScaling] = None,
    lw_scaling: Optional[InputScaling] = None,
    band_mode: str = "per_band",
    r_eff_liq_um: Optional[jnp.ndarray] = None,
    r_eff_ice_um: Optional[jnp.ndarray] = None,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Emulated radiation scheme — drop-in replacement for ``radiation_scheme_rrtmgp``.

    Runs a GRU network per column to predict all-sky and clear-sky flux
    profiles, then derives heating rates from flux divergence. The call
    signature matches the other radiation schemes so it can be used
    interchangeably.

    Additional Args:
        emulator_weights: Trained NN weights (``EmulatorWeights``). Must be
            provided; passed through the parameters mechanism in EchamPhysics.
        sw_scaling: Input normalization for SW network.
        lw_scaling: Input normalization for LW network.
        band_mode: How aerosol optics enter the input features; see
            ``nn_emulator._band_features``. Must match what the weights
            were trained with, since it fixes the input width.

    """
    from jax_solar import OrbitalTime, radiation_flux, get_solar_sin_altitude

    nlev = temperature.shape[0]

    # --- Solar geometry ---
    # `solar` is a `jcm.forcing.SolarGeometry` precomputed by the Model;
    # the radiation scheme stays date-free.
    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    toa_flux = radiation_flux(
        orbital_time, longitude, latitude, parameters.solar_constant
    )
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    # The ACTUAL cosine is what RadiationData must carry: the cached-step
    # rescale divides by it, so storing the clipped value would rescale
    # twilight columns by mu_now/min_cos_zenith instead of mu_now/mu_at_solve
    # (PR #730 review). The clip exists only to keep the network FEATURE in
    # its trained range.
    cos_zenith = sin_altitude
    mu0_feature = jnp.maximum(sin_altitude, parameters.min_cos_zenith)

    # --- Prepare inputs common to SW and LW ---
    # Water vapour mixing ratio
    eps = c.eps  # Mv/Md ≈ 0.622
    h2o_vmr = specific_humidity / (eps * (1.0 - specific_humidity) + specific_humidity)

    # Ozone
    if ozone_vmr is None:
        ozone_vmr = jnp.full(nlev, 5e-6)

    # GRID-MEAN cloud water/ice paths (kg/m^2). ``cloud_water``/``cloud_ice``
    # arrive as grid-mean mixing ratios (prognostic qc/qi), so rho*dz alone
    # gives the grid-mean path; a further cloud_fraction factor would make
    # the feature scale as cf^2 (cover is its own feature). The trainer
    # builds the identical quantity — the two must change together.
    cwp = cloud_water * air_density * layer_thickness
    cip = cloud_ice * air_density * layer_thickness

    # Effective radii, resolved against the same diagnostic fallbacks RRTMGP
    # uses so a feature and the label it was trained against describe the same
    # cloud. Zero (1M microphysics, cold start, or no caller-supplied value)
    # selects the fallback rather than meaning "zero-radius droplets".
    # The ice fallback is a power law in IN-CLOUD IWC, so it needs the
    # in-cloud path (grid mean DIVIDED by cloud fraction), not the grid-mean
    # `cip` above. Using `cip` here would feed an IWC too small by ~cf^2 and
    # reintroduce exactly the feature/label mismatch this input exists to
    # remove -- the label generator resolves from the in-cloud path.
    ice_in_cloud = jnp.minimum(
        in_cloud_path(cloud_ice, cloud_fraction), _MAX_IN_CLOUD_CONDENSATE,
    )
    zeros = jnp.zeros_like(temperature)
    r_eff_liq_um, r_eff_ice_um = resolve_effective_radii(
        zeros if r_eff_liq_um is None else r_eff_liq_um,
        zeros if r_eff_ice_um is None else r_eff_ice_um,
        aerosol_data.cdnc_factor,
        ice_in_cloud * air_density * layer_thickness, layer_thickness,
    )

    n_sw = n_input_features(band_mode, aerosol_data.aod_sw_per_band.shape[0])
    n_lw = n_input_features(band_mode, aerosol_data.aod_lw_per_band.shape[0])
    if sw_scaling is None:
        sw_scaling = InputScaling(x_max=jnp.ones(n_sw))
    if lw_scaling is None:
        lw_scaling = InputScaling(x_max=jnp.ones(n_lw))

    # --- Shortwave ---
    sw_input = preprocess_sw_inputs(
        temperature, pressure_levels, h2o_vmr, ozone_vmr,
        cwp, cip, cloud_fraction, mu0_feature, sw_scaling,
        r_eff_liq_um, r_eff_ice_um,
        aerosol_data.aod_sw_per_band, aerosol_data.ssa_sw_per_band,
        aerosol_data.asy_sw_per_band, band_mode,
    )
    # Same 0.46/0.54 vis/NIR weighting RRTMGP is driven with, so the
    # emulator sees the boundary condition its labels were made under.
    surface_albedo = 0.46 * surface_albedo_vis + 0.54 * surface_albedo_nir
    sw_nn_output = lw_emulator_column(
        sw_input, jnp.atleast_1d(surface_albedo), emulator_weights.sw,
    )
    toa_sw_down = jnp.maximum(toa_flux, 0.0)
    (sw_flux_down, sw_flux_up, sw_flux_down_clear,
     sw_flux_up_clear) = reconstruct_sw_interface_fluxes(
        sw_nn_output, toa_sw_down,
    )

    # --- Longwave ---
    lw_input = preprocess_lw_inputs(
        temperature, pressure_levels, h2o_vmr, ozone_vmr,
        cwp, cip, cloud_fraction, co2_vmr, lw_scaling,
        r_eff_liq_um, r_eff_ice_um,
        aerosol_data.aod_lw_per_band, aerosol_data.ssa_lw_per_band,
        aerosol_data.asy_lw_per_band, band_mode,
    )
    lw_nn_output = lw_emulator_column(
        lw_input, jnp.atleast_1d(surface_emissivity), emulator_weights.lw,
    )
    (lw_flux_down, lw_flux_up, lw_flux_down_clear,
     lw_flux_up_clear) = reconstruct_lw_interface_fluxes(
        lw_nn_output, surface_temperature, temperature,
    )

    # --- Heating rates ---
    sw_heating = flux_to_heating_rate(sw_flux_down, sw_flux_up, pressure_interfaces)
    lw_heating = flux_to_heating_rate(lw_flux_down, lw_flux_up, pressure_interfaces)
    total_heating = sw_heating + lw_heating

    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating,
        shortwave_heating=sw_heating,
    )

    diagnostics = RadiationData(
        cos_zenith=cos_zenith,
        surface_albedo_vis=jnp.atleast_1d(surface_albedo_vis),
        surface_albedo_nir=jnp.atleast_1d(surface_albedo_nir),
        surface_emissivity=jnp.atleast_1d(surface_emissivity),
        sw_flux_up=sw_flux_up,
        sw_flux_down=sw_flux_down,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up,
        lw_flux_down=lw_flux_down,
        lw_heating_rate=lw_heating,
        sw_flux_up_clear=sw_flux_up_clear,
        sw_flux_down_clear=sw_flux_down_clear,
        lw_flux_up_clear=lw_flux_up_clear,
        lw_flux_down_clear=lw_flux_down_clear,
        surface_sw_down=sw_flux_down[-1],
        surface_lw_down=lw_flux_down[-1],
        surface_sw_up=sw_flux_up[-1],
        surface_lw_up=lw_flux_up[-1],
        toa_sw_up=sw_flux_up[0],
        toa_lw_up=lw_flux_up[0],
        toa_sw_down=toa_sw_down,
        # Clear sky comes from dedicated output channels, so CRE is a
        # real difference here rather than a placeholder.
        toa_sw_up_clear=sw_flux_up_clear[0],
        toa_sw_up_noa=jnp.zeros_like(sw_flux_up[0]),
        toa_lw_up_noa=jnp.zeros_like(sw_flux_up[0]),
        toa_sw_up_clear_noa=jnp.zeros_like(sw_flux_up[0]),
        noa_frac_toa_sw_up=jnp.zeros_like(sw_flux_up[0]),
        noa_frac_toa_lw_up=jnp.zeros_like(sw_flux_up[0]),
        noa_frac_toa_sw_up_clear=jnp.zeros_like(sw_flux_up[0]),
        noa_frac_toa_lw_up_clear=jnp.zeros_like(sw_flux_up[0]),
        toa_lw_up_clear_noa=jnp.zeros_like(sw_flux_up[0]),
        toa_lw_up_clear=lw_flux_up_clear[0],
        # No sub-column machinery here, so column cover is the ANALYTIC
        # maximum-random total — the expectation of the McICA draw RRTMGP
        # reports and the overlap its labels assume. Publishing the old
        # placeholder 0 fed aerocom_cmor's clt a clear sky under full
        # cloud (PR #730 review).
        total_cloud_cover=jnp.broadcast_to(
            _max_random_total_cover(cloud_fraction),
            jnp.shape(sw_flux_up[0]),
        ),
        # ``step`` is owned by the enclosing ``NNEmulatorRadiation``
        # carry — the standalone scheme emits 0 and the term bumps it
        # after its compute-vs-cache cond.
        step=jnp.int32(0),
    )

    return tendencies, diagnostics


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.clouds.cloud_data import radiation_cloud_fields  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics.radiation import (  # noqa: E402
    cached_radiation_tendency,
    current_cos_zenith,
    radiation_should_compute,
    rescale_cached_radiation,
)
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


def _column_vector_emulated(value: jnp.ndarray, ncols: int) -> jnp.ndarray:
    """Return a vmapped scalar diagnostic as one value per column."""
    return jnp.reshape(value, (ncols,))


# Config key to point at when the checkpoint and the term disagree.
_TERM_CONFIG_HINT = (
    "the nn_emulator_radiation term in "
    "jcm/config/physics/echam-emulated-2m.yaml"
)


def _validate_weights_file(
    filepath: str,
    weights: EmulatorWeights,
    sw_scaling: InputScaling,
    lw_scaling: InputScaling,
    metadata: dict,
    band_mode: str,
    n_bnd_sw: int,
    n_bnd_lw: int,
) -> None:
    """Reject a checkpoint that was not trained for this configuration.

    ``band_mode`` and the band counts jointly fix the network's input
    width, and all three come from config rather than from the file. A
    mismatch either dies deep inside a GRU matmul with a bare
    ``dot_general`` shape error or — when the widths happen to agree —
    runs the network on features unlike anything it was trained on,
    which produces plausible-looking but wrong fluxes.

    Args:
        filepath: Checkpoint path, quoted in every message.
        weights / sw_scaling / lw_scaling: What was loaded from it.
        metadata: Global attributes from the file; must carry
            ``band_mode``.
        band_mode / n_bnd_sw / n_bnd_lw: The term's configuration.

    Raises:
        ValueError: Naming the offending values and the config key.

    """
    file_band_mode = metadata.get("band_mode")
    if file_band_mode is None:
        raise ValueError(
            f"Emulator weights file {filepath!r} carries no 'band_mode' "
            f"metadata, so it cannot be checked against band_mode="
            f"{band_mode!r}. Re-save it with save_emulator_weights(..., "
            "metadata={'band_mode': ...})."
        )
    file_band_mode = str(file_band_mode)
    if file_band_mode != band_mode:
        raise ValueError(
            f"Emulator weights file {filepath!r} was trained with "
            f"band_mode={file_band_mode!r} but the term is configured "
            f"with band_mode={band_mode!r}. Set band_mode on "
            f"{_TERM_CONFIG_HINT} to {file_band_mode!r}, or train "
            f"weights for {band_mode!r}."
        )

    for side, n_bnd, band_key, w, scaling in (
        ("SW", n_bnd_sw, "n_bnd_sw", weights.sw, sw_scaling),
        ("LW", n_bnd_lw, "n_bnd_lw", weights.lw, lw_scaling),
    ):
        # The input-width check below cannot see a band-count mismatch
        # under band_mode='broadband' (the width is band-count-independent,
        # but _band_features sums/AOD-weights the supplied bands, so the
        # partition still changes every feature value). The trainer stores
        # the counts in metadata; hold the file to them when present.
        file_n_bnd = metadata.get(band_key)
        if file_n_bnd is not None and int(file_n_bnd) != n_bnd:
            raise ValueError(
                f"Emulator weights file {filepath!r} was trained with "
                f"{band_key}={int(file_n_bnd)} but the term supplies "
                f"{n_bnd} {side} aerosol bands. Match the band structure "
                f"or train weights for it."
            )
        stored = int(w.gru_fwd.kernel.shape[0])
        expected = n_input_features(band_mode, n_bnd)
        if stored != expected:
            raise ValueError(
                f"{side} weights in {filepath!r} take {stored} input "
                f"features, but band_mode={band_mode!r} with {band_key}="
                f"{n_bnd} gives {expected}. Fix band_mode, {band_key} or "
                f"weights_file on {_TERM_CONFIG_HINT} so the three agree."
            )
        got = int(scaling.x_max.shape[-1])
        if got != stored:
            raise ValueError(
                f"{side} scaling in {filepath!r} has {got} entries but "
                f"the {side} weights take {stored} input features. The "
                "checkpoint is internally inconsistent; re-save it."
            )
        # Flux reconstruction reads channels 0-3 (all-sky + clear-sky,
        # down/up). JAX clamps an out-of-bounds gather, so a narrower
        # checkpoint would not fail — it would silently publish channel 1
        # again as "clear sky" and corrupt every CRE diagnostic.
        n_out = int(w.output_dense.kernel.shape[-1])
        if n_out != 4:
            raise ValueError(
                f"{side} weights in {filepath!r} emit {n_out} output "
                "channels, but flux reconstruction needs exactly 4 "
                "(down, up, down_clear, up_clear). The file predates the "
                "clear-sky output channels; retrain it."
            )


class NNEmulatorRadiation(PhysicsTerm):
    """Bidirectional-GRU neural network radiation emulator as a PhysicsTerm.

    Drop-in replacement for :class:`GreyTwoStreamRadiation` /
    :class:`RRTMGPRadiation` that uses a pre-trained NN to predict
    SW + LW fluxes per column, then derives heating rates from flux
    divergence. Cheap and differentiable. Reads the same diagnostics
    set as the other radiation terms; the emulator weights / scaling
    live on ``parameters.radiation``.
    """

    name: ClassVar[str] = "nn_emulator_radiation"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half", "layer_thickness",
        "air_density", "chemistry", "aerosol",
        "radiation", "surface", "clouds",
    )
    provides: ClassVar[tuple[str, ...]] = ("radiation", "clouds")

    def __init__(
        self,
        params: RadiationParameters | None = None,
        band_mode: str = "per_band",
        units: int = 16,
        init_seed: int = 42,
        n_bnd_sw: int = 14,
        n_bnd_lw: int = 16,
        zero_tendency: bool = False,
        weights_file: str | None = None,
    ):
        """Hold the scheme-native :class:`RadiationParameters` (with NN weights).

        Args:
            params: Scheme parameters. If it carries no
                ``emulator_weights`` and no ``weights_file`` is given,
                randomly initialised ones sized for ``band_mode`` are
                built here, which is what training from scratch starts
                from and what lets Hydra construct the term at all.
            band_mode: How aerosol optics enter the features; fixes the
                input width, so it must match what the weights were
                trained with. A plain attribute rather than a parameter
                leaf because it selects a code path at trace time.
            units: GRU hidden size.
            init_seed: PRNG seed for randomly initialised weights.
            n_bnd_sw / n_bnd_lw: Band counts used to size the input layer.
            zero_tendency: Run the network at full cost but return zero
                heating. Untrained weights drive the model to NaN within
                a step, and a blown-up run cannot be timed; this measures
                what the scheme costs on a trajectory that stays finite.
                It is a cost-measurement aid and never a valid simulation.
            weights_file: NetCDF checkpoint written by
                :func:`nn_emulator.save_emulator_weights`. Supplies the
                trained weights *and* both input scalings, and is
                validated against ``band_mode`` / the band counts before
                use. This is jax-gcm's own format, not the upstream
                rte-rrtmgp-nn one. ``"auto"`` resolves the packaged
                default (``jcm/data/emulator_weights_per_band_u64.nc``).

        Raises:
            ValueError: If ``weights_file`` disagrees with ``band_mode``
                or the band counts.

        """
        params = params or RadiationParameters.default()
        if weights_file == "auto":
            # The packaged default (same convention as ``ozone_file: auto``):
            # per_band 14/16-band, 64-unit weights trained on the v3 labels
            # (provenance in the file's global attributes). Makes
            # ``physics=echam-emulated-2m`` runnable out of the box.
            from importlib import resources
            weights_file = str(
                resources.files("jcm") / "data"
                / "emulator_weights_per_band_u64.nc"
            )
        if weights_file is not None:
            weights, sw_scaling, lw_scaling, metadata = load_emulator_weights(
                weights_file,
            )
            _validate_weights_file(
                weights_file, weights, sw_scaling, lw_scaling, metadata,
                band_mode, n_bnd_sw, n_bnd_lw,
            )
            params = dataclasses.replace(
                params, emulator_weights=weights,
                sw_scaling=sw_scaling, lw_scaling=lw_scaling,
            )
        elif params.emulator_weights is None:
            params = dataclasses.replace(
                params,
                emulator_weights=init_emulator_weights(
                    sw_features=n_input_features(band_mode, n_bnd_sw),
                    lw_features=n_input_features(band_mode, n_bnd_lw),
                    units=units, key=jax.random.key(init_seed),
                ),
            )
        # The network weights live in their OWN Param, separate from the rest
        # of RadiationParameters, because that struct carries integer leaves
        # (band counts, the cloud-overlap enum) and jax.grad rejects a pytree
        # containing them. Keeping the weights partitioned means an optimizer
        # can address exactly the differentiable subtree -- which is what an
        # online fine-tuning loop needs -- without filtering.
        self.weights = nnx.Param(params.emulator_weights)
        self.params = nnx.Param(
            dataclasses.replace(params, emulator_weights=None),
        )
        self._band_mode = band_mode
        self._zero_tendency = bool(zero_tendency)
        self._coords_cached = False

    def withheld_output_keys(self) -> tuple[str, ...]:
        """Hide the aerosol-free slots the emulator never fills.

        The network makes no aerosol-free prediction, so the ``*_noa``
        fluxes and ``noa_frac_*`` ratios sit at their zero defaults.
        Publishing them turns a downstream ERFari (``rsut - rsutnoa``)
        into the entire all-sky flux (jax-gcm#647; same contract as
        ``RRTMGPRadiation.withheld_output_keys``). The clear-sky fluxes
        are NOT withheld: they come from dedicated network output
        channels, so CRE is a real difference here.
        """
        from jcm.physics.radiation.aerosol_free import NOA_KEYS
        return tuple(f"radiation.{k}_noa" for k in NOA_KEYS) + tuple(
            f"radiation.noa_frac_{k}" for k in NOA_KEYS
        )

    def _zero_if_requested(self, tendency: PhysicsTendency) -> PhysicsTendency:
        """Drop the heating under ``zero_tendency``, leaving the cost paid.

        Applied downstream of the compute-vs-cache branch so it covers
        both. The cached branch rebuilds the tendency from the heating
        rates on the carry, and radiation sub-steps, so most calls take
        it — zeroing only the compute branch leaves untrained heating
        driving the model.
        """
        if not self._zero_tendency:
            return tendency
        return PhysicsTendency(
            u_wind=tendency.u_wind, v_wind=tendency.v_wind,
            temperature=jnp.zeros_like(tendency.temperature),
            specific_humidity=tendency.specific_humidity,
            tracers=tendency.tracers,
        )

    def _check_band_counts(self, n_bnd_sw: int, n_bnd_lw: int) -> None:
        """Fail clearly when the aerosol bands do not match the weights."""
        weights = self.weights.get_value()
        for name, n_bnd, w in (
            ("SW", n_bnd_sw, weights.sw), ("LW", n_bnd_lw, weights.lw),
        ):
            expected = w.gru_fwd.kernel.shape[0]
            got = n_input_features(self._band_mode, n_bnd)
            if got != expected:
                raise ValueError(
                    f"{name} emulator weights expect {expected} input "
                    f"features but the aerosol term supplies {n_bnd} "
                    f"{name} bands, giving {got} under band_mode="
                    f"{self._band_mode!r}. The band config follows the "
                    "active radiation backend; a radiation term absent "
                    "from jcm.runners._band_config_for_terms falls back "
                    "to a single 550 nm SW band and no LW bands."
                )

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (deg) for the radiation scheme."""
        lat, lon = column_lat_lon(coords.horizontal)
        self._lats = nnx.Variable(lat * 180.0 / jnp.pi)
        self._lons = nnx.Variable(lon * 180.0 / jnp.pi)
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute or reuse cached NN-emulated heating rates."""
        params = self.params.get_value()
        radiation = diagnostics["radiation"]
        # Solar geometry now. Needed on both branches: the compute branch
        # stamps it so a later cached step knows which sun the fluxes were
        # solved under, and the cached branch rescales the shortwave by the
        # ratio of the two (#671). Pure trig, so it is cheap every step.
        mu0_now = current_cos_zenith(
            forcing.solar, self._lons.get_value(), self._lats.get_value(),
        ).astype(radiation.cos_zenith.dtype)

        def _compute():
            tend, rad = self._compute_full(state, diagnostics, forcing, params)
            # Pin the compute branch to the carry's leaf dtypes (see the
            # identical guard in rrtmgp.py / the grey scheme): keeps the
            # two lax.cond branches type-identical for float32 states
            # under jax_enable_x64.
            rad = jax.tree.map(lambda n, o: n.astype(o.dtype), rad, radiation)
            tend = jax.tree.map(
                lambda t: t.astype(state.temperature.dtype), tend)
            return tend, rad

        def _use_cached():
            rad = rescale_cached_radiation(radiation, mu0_now)
            tend = cached_radiation_tendency(rad, state.temperature.shape)
            # Same dtype pin as _compute: under x64 the cached heating ->
            # tendency arithmetic can promote through float64 scalars.
            tend = jax.tree.map(
                lambda t: t.astype(state.temperature.dtype), tend)
            return tend, rad

        tendency, new_radiation = jax.lax.cond(
            radiation_should_compute(diagnostics, params),
            _compute, _use_cached,
        )
        # Zeroed downstream of the cond, so it covers the cached branch
        # too: that one rebuilds the tendency from the heating rates on
        # the carry, and with radiation sub-stepping most steps take it.
        tendency = self._zero_if_requested(tendency)
        # Advance the radiation-local step counter on every call (both
        # compute and cached paths). Mirrors the carry-side step bump
        # in the grey two-stream / RRTMGP radiation terms so the
        # sub-stepping gate sees the same cadence regardless of scheme.
        new_radiation = new_radiation.copy(step=radiation.step + 1)
        # Mirror TOA fluxes onto the clouds sub-struct for CRE
        # diagnostics. Clear sky comes from dedicated network output
        # channels, so no second solve is needed.
        clouds = diagnostics["clouds"].copy(
            toa_sw_up_all=new_radiation.toa_sw_up,
            toa_sw_up_clear=new_radiation.toa_sw_up_clear,
            toa_lw_up_all=new_radiation.toa_lw_up,
            toa_lw_up_clear=new_radiation.toa_lw_up_clear,
        )
        return tendency, {
            **diagnostics, "radiation": new_radiation, "clouds": clouds,
        }

    def _compute_full(
        self, state, diagnostics, forcing, params,
    ):
        """Run the full NN-emulator scheme, return (tendency, RadiationData)."""
        nlev, ncols = state.temperature.shape

        latitudes = self._lats.get_value()
        longitudes = self._lons.get_value()
        solar = forcing.solar

        cloud_water, cloud_ice, cloud_fraction = radiation_cloud_fields(
            state, diagnostics,
        )

        chemistry = diagnostics["chemistry"]
        ozone_vmr = chemistry.ozone_vmr * 1e-6
        # CO2 is a prescribed forcing read straight from ForcingData.
        co2_vmr = forcing.co2_vmr * 1e-6

        # Microphysical effective radii from the clouds carry, sourced exactly
        # as RRTMGP sources them so the emulator sees the cloud its labels
        # describe. Zero (1M, or a cold start) selects the diagnostic fallback
        # inside the scheme.
        clouds_in = diagnostics["clouds"]
        r_eff_liq_um = clouds_in.r_eff_liq.reshape(nlev, ncols)
        r_eff_ice_um = clouds_in.r_eff_ice.reshape(nlev, ncols)

        surface_temperature_col = (
            diagnostics["surface"].surface_temperature.reshape(ncols)
        )
        radiation_in = diagnostics["radiation"]
        surface_albedo_vis_col = radiation_in.surface_albedo_vis.reshape(ncols)
        surface_albedo_nir_col = radiation_in.surface_albedo_nir.reshape(ncols)
        surface_emissivity_col = radiation_in.surface_emissivity.reshape(ncols)

        aerosol_in = diagnostics["aerosol"]
        n_bnd_sw = aerosol_in.aod_sw_per_band.shape[0]
        n_bnd_lw = aerosol_in.aod_lw_per_band.shape[0]
        # The weights were sized at construction from assumed band counts,
        # but the aerosol term's are set by the active band config. A
        # mismatch otherwise surfaces deep in a GRU matmul as a bare
        # dot_general shape error.
        self._check_band_counts(n_bnd_sw, n_bnd_lw)

        def _per_band_to_col(arr, n_bnd):
            """(n_bnd, nlev, ncols) → (ncols, n_bnd, nlev) for the column vmap."""
            return arr.reshape(n_bnd, nlev, ncols).transpose(2, 0, 1)

        # Every leaf must carry the column axis first: ``in_axes=0`` maps
        # the whole pytree, so a field left with its band axis leading
        # makes vmap reject the trace.
        aerosol_for_vmap = aerosol_in.copy(
            aod_profile=aerosol_in.aod_profile.reshape(nlev, ncols).T,
            ssa_profile=aerosol_in.ssa_profile.reshape(nlev, ncols).T,
            asy_profile=aerosol_in.asy_profile.reshape(nlev, ncols).T,
            cdnc_factor=aerosol_in.cdnc_factor.reshape(ncols),
            aod_total=aerosol_in.aod_total.reshape(ncols),
            aod_anthropogenic=aerosol_in.aod_anthropogenic.reshape(ncols),
            aod_background=aerosol_in.aod_background.reshape(ncols),
            angstrom=aerosol_in.angstrom.reshape(ncols),
            aod_sw_per_band=_per_band_to_col(aerosol_in.aod_sw_per_band, n_bnd_sw),
            ssa_sw_per_band=_per_band_to_col(aerosol_in.ssa_sw_per_band, n_bnd_sw),
            asy_sw_per_band=_per_band_to_col(aerosol_in.asy_sw_per_band, n_bnd_sw),
            aod_lw_per_band=_per_band_to_col(aerosol_in.aod_lw_per_band, n_bnd_lw),
            ssa_lw_per_band=_per_band_to_col(aerosol_in.ssa_lw_per_band, n_bnd_lw),
            asy_lw_per_band=_per_band_to_col(aerosol_in.asy_lw_per_band, n_bnd_lw),
        )

        emulator_weights = self.weights.get_value()
        sw_scaling = params.sw_scaling
        lw_scaling = params.lw_scaling
        band_mode = self._band_mode

        tendencies_vmapped, diagnostics_vmapped = jax.vmap(
            radiation_scheme_emulated,
            in_axes=(
                1, 1, 1, 1, 1,
                1, 1, 1, 1,
                0, 0, 0, 0,
                None, 0, 0,
                None, 0, 1, None,
                None, None, None, None,
                1, 1,
            ),
            out_axes=(0, 0),
            axis_size=ncols,
        )(
            state.temperature, state.specific_humidity,
            diagnostics["pressure_full"], diagnostics["pressure_half"],
            diagnostics["layer_thickness"], diagnostics["air_density"],
            cloud_water, cloud_ice, cloud_fraction,
            surface_temperature_col, surface_albedo_vis_col,
            surface_albedo_nir_col, surface_emissivity_col,
            solar, latitudes, longitudes,
            params, aerosol_for_vmap, ozone_vmr, co2_vmr,
            emulator_weights, sw_scaling, lw_scaling, band_mode,
            r_eff_liq_um, r_eff_ice_um,
        )

        rad_out = RadiationData(
            cos_zenith=_column_vector_emulated(
                diagnostics_vmapped.cos_zenith, ncols,
            ),
            surface_albedo_vis=_column_vector_emulated(
                diagnostics_vmapped.surface_albedo_vis, ncols,
            ),
            surface_albedo_nir=_column_vector_emulated(
                diagnostics_vmapped.surface_albedo_nir, ncols,
            ),
            surface_emissivity=_column_vector_emulated(
                diagnostics_vmapped.surface_emissivity, ncols,
            ),
            sw_flux_up=diagnostics_vmapped.sw_flux_up.T,
            sw_flux_down=diagnostics_vmapped.sw_flux_down.T,
            sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
            lw_flux_up=diagnostics_vmapped.lw_flux_up.T,
            lw_flux_down=diagnostics_vmapped.lw_flux_down.T,
            lw_heating_rate=tendencies_vmapped.longwave_heating.T,
            sw_flux_up_clear=diagnostics_vmapped.sw_flux_up_clear.T,
            sw_flux_down_clear=diagnostics_vmapped.sw_flux_down_clear.T,
            lw_flux_up_clear=diagnostics_vmapped.lw_flux_up_clear.T,
            lw_flux_down_clear=diagnostics_vmapped.lw_flux_down_clear.T,
            surface_sw_down=_column_vector_emulated(
                diagnostics_vmapped.surface_sw_down, ncols,
            ),
            surface_lw_down=_column_vector_emulated(
                diagnostics_vmapped.surface_lw_down, ncols,
            ),
            surface_sw_up=_column_vector_emulated(
                diagnostics_vmapped.surface_sw_up, ncols,
            ),
            surface_lw_up=_column_vector_emulated(
                diagnostics_vmapped.surface_lw_up, ncols,
            ),
            toa_sw_up=_column_vector_emulated(
                diagnostics_vmapped.toa_sw_up, ncols,
            ),
            toa_lw_up=_column_vector_emulated(
                diagnostics_vmapped.toa_lw_up, ncols,
            ),
            toa_sw_down=_column_vector_emulated(
                diagnostics_vmapped.toa_sw_down, ncols,
            ),
            toa_sw_up_noa=jnp.zeros((ncols,)),
            toa_lw_up_noa=jnp.zeros((ncols,)),
            toa_sw_up_clear_noa=jnp.zeros((ncols,)),
            noa_frac_toa_sw_up=jnp.zeros((ncols,)),
            noa_frac_toa_lw_up=jnp.zeros((ncols,)),
            noa_frac_toa_sw_up_clear=jnp.zeros((ncols,)),
            noa_frac_toa_lw_up_clear=jnp.zeros((ncols,)),
            toa_lw_up_clear_noa=jnp.zeros((ncols,)),
            toa_sw_up_clear=_column_vector_emulated(
                diagnostics_vmapped.toa_sw_up_clear, ncols,
            ),
            toa_lw_up_clear=_column_vector_emulated(
                diagnostics_vmapped.toa_lw_up_clear, ncols,
            ),
            total_cloud_cover=_column_vector_emulated(
                diagnostics_vmapped.total_cloud_cover, ncols,
            ),
            # Placeholder — the enclosing ``__call__`` overwrites
            # ``step`` after the compute-vs-cache cond.
            step=jnp.int32(0),
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros((nlev, ncols)),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=tendencies_vmapped.temperature_tendency.T,
            specific_humidity=jnp.zeros((nlev, ncols)),
            tracers={},
        )
        return tendency, rad_out
