"""Neural network emulator for radiative transfer.

Implements the bidirectional RNN architecture from Ukkonen (2024) for emulating
the RTE+RRTMGP radiation scheme. Separate models handle shortwave (SW) and
longwave (LW) radiation, each predicting normalized flux profiles that are
converted to heating rates via flux divergence.

Reference: https://github.com/peterukk/rte-rrtmgp-nn (nn_dev branch)

Date: 2026-04-11
"""

from typing import Optional

import jax
import jax.numpy as jnp
import tree_math

import jcm.constants as c


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def softsign(x: jnp.ndarray) -> jnp.ndarray:
    """Softsign activation: x / (|x| + 1)."""
    return x / (jnp.abs(x) + 1.0)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation."""
    return jax.nn.sigmoid(x)


ACTIVATIONS = {
    "softsign": softsign,
    "sigmoid": sigmoid,
    "relu": jax.nn.relu,
    "linear": lambda x: x,
    "tanh": jnp.tanh,
}


# ---------------------------------------------------------------------------
# Weight data structures
# ---------------------------------------------------------------------------

@tree_math.struct
class DenseWeights:
    """Weights for a single Dense layer: y = activation(x @ kernel + bias)."""

    kernel: jnp.ndarray  # (in_features, out_features)
    bias: jnp.ndarray    # (out_features,)


@tree_math.struct
class GRUWeights:
    """Weights for a GRU cell.

    Gate layout follows Keras convention: [z (update), r (reset), h (candidate)].
    ``kernel`` multiplies the input, ``recurrent_kernel`` multiplies the hidden state.
    """

    kernel: jnp.ndarray           # (input_dim, 3 * units)
    recurrent_kernel: jnp.ndarray # (units, 3 * units)
    bias: jnp.ndarray             # (2, 3 * units) — input bias + recurrent bias


@tree_math.struct
class SWEmulatorWeights:
    """Weights for the shortwave bidirectional-GRU emulator.

    Architecture (brnn.py):
      aux_dense_fwd  : albedo → initial state for forward GRU
      aux_dense_bwd  : albedo → initial state for backward GRU
      gru_fwd        : forward GRU (Bidirectional wrapper forward)
      gru_bwd        : backward GRU (Bidirectional wrapper backward)
      gru2           : second GRU on concatenated hidden states
      output_dense   : TimeDistributed Dense → 2 outputs (rsd_norm, rsu_norm)
    """

    aux_dense_fwd: DenseWeights
    aux_dense_bwd: DenseWeights
    gru_fwd: GRUWeights
    gru_bwd: GRUWeights
    gru2: GRUWeights
    output_dense: DenseWeights


@tree_math.struct
class LWEmulatorWeights:
    """Weights for the longwave GRU emulator.

    Architecture (brnn2.py):
      gru_fwd       : forward GRU
      surface_dense : Dense on [last_state, emissivity]
      gru_bwd       : backward GRU (go_backwards=True)
      gru3          : third GRU on concatenated hidden states
      output_dense  : TimeDistributed Dense → 2 outputs (rld_norm, rlu_norm)
    """

    gru_fwd: GRUWeights
    surface_dense: DenseWeights
    gru_bwd: GRUWeights
    gru3: GRUWeights
    output_dense: DenseWeights


@tree_math.struct
class EmulatorWeights:
    """All weights for the emulated radiation scheme.

    Both slots hold :class:`LWEmulatorWeights`-shaped weights: upstream's
    shipped shortwave model (``bigru_gru_16``) uses this surface-aux
    architecture, not the bidirectional one, and it is the only variant
    verified against a real checkpoint. It also emits at ``nlev+1``
    interfaces, where the fluxes actually live. The aux input is surface
    albedo for SW and emissivity for LW.

    :class:`SWEmulatorWeights` and :func:`sw_emulator_column` remain as
    the bidirectional alternative for the architecture sweep.
    """

    sw: LWEmulatorWeights
    lw: LWEmulatorWeights


@tree_math.struct
class InputScaling:
    """Min-max scaling coefficients for NN inputs: x_scaled = x / x_max."""

    x_max: jnp.ndarray  # (n_features,)


# ---------------------------------------------------------------------------
# GRU cell
# ---------------------------------------------------------------------------

def gru_cell(
    x: jnp.ndarray,
    h: jnp.ndarray,
    weights: GRUWeights,
) -> jnp.ndarray:
    """Single GRU step.

    Args:
        x: Input vector (input_dim,).
        h: Previous hidden state (units,).
        weights: GRU weights.

    Returns:
        New hidden state (units,).

    """
    units = h.shape[-1]

    # Gate projections
    x_z = x @ weights.kernel[:, :units]
    x_r = x @ weights.kernel[:, units:2*units]
    x_h = x @ weights.kernel[:, 2*units:]

    h_z = h @ weights.recurrent_kernel[:, :units]
    h_r = h @ weights.recurrent_kernel[:, units:2*units]

    # Keras uses two bias rows: input_bias and recurrent_bias
    bx = weights.bias[0]
    bh = weights.bias[1]

    z = sigmoid(x_z + bx[:units] + h_z + bh[:units])
    r = sigmoid(x_r + bx[units:2*units] + h_r + bh[units:2*units])

    h_candidate = jnp.tanh(
        x_h + bx[2*units:] + r * (h @ weights.recurrent_kernel[:, 2*units:] + bh[2*units:])
    )

    h_new = z * h + (1.0 - z) * h_candidate
    return h_new


# ---------------------------------------------------------------------------
# GRU sequence processing
# ---------------------------------------------------------------------------

def gru_forward_sequence(
    x_seq: jnp.ndarray,
    h0: jnp.ndarray,
    weights: GRUWeights,
) -> jnp.ndarray:
    """Run a GRU forward over a sequence.

    Args:
        x_seq: Input sequence (seq_len, input_dim).
        h0: Initial hidden state (units,).
        weights: GRU weights.

    Returns:
        Hidden states at all time steps (seq_len, units).

    """
    def step(h, x):
        h_new = gru_cell(x, h, weights)
        return h_new, h_new

    _, hidden_seq = jax.lax.scan(step, h0, x_seq)
    return hidden_seq


def gru_backward_sequence(
    x_seq: jnp.ndarray,
    h0: jnp.ndarray,
    weights: GRUWeights,
    realign: bool = True,
) -> jnp.ndarray:
    """Run a GRU backward over a sequence (go_backwards=True).

    Args:
        x_seq: Input sequence (seq_len, input_dim).
        h0: Initial hidden state (units,).
        weights: GRU weights.
        realign: If True, reverse the output back so element ``i`` is the
            state *at* level ``i``, which is what a concat against a
            forward pass needs (Keras ``Bidirectional`` semantics). If
            False, leave it in computation order — a bare Keras
            ``GRU(go_backwards=True, return_sequences=True)`` does not
            reverse its output, so reproducing upstream ``rte-rrtmgp-nn``
            checkpoints requires this.

    Returns:
        Hidden states at all time steps (seq_len, units).

    """
    hidden_rev = gru_forward_sequence(x_seq[::-1], h0, weights)
    return hidden_rev[::-1] if realign else hidden_rev


# ---------------------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------------------

def dense(x: jnp.ndarray, weights: DenseWeights, activation=None) -> jnp.ndarray:
    """Apply a Dense layer: y = activation(x @ kernel + bias).

    Works for both single vectors and batched (seq_len, features) inputs
    (TimeDistributed pattern).
    """
    y = x @ weights.kernel + weights.bias
    if activation is not None:
        y = activation(y)
    return y


# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------

# Smallest mixing ratio fed to the quartic-root transform. A spectral
# dycore delivers small negative humidity from Gibbs ringing and
# (-1e-12) ** 0.25 is NaN, so the floor is required, not defensive.
# 1e-15 kg/kg is far below any radiatively active amount.
GAS_FLOOR = 1e-15


def _band_features(
    aod: jnp.ndarray,
    ssa: jnp.ndarray,
    asy: jnp.ndarray,
    band_mode: str,
) -> list[jnp.ndarray]:
    """Turn per-band aerosol optics into per-level input features.

    Args:
        aod / ssa / asy: Per-band optics, ``(n_bnd, nlev)``.
        band_mode: ``"per_band"`` keeps every band (highest fidelity,
            ``3 * n_bnd`` features); ``"broadband"`` collapses to three
            (AOD summed over bands, AOD-weighted SSA and asymmetry);
            ``"none"`` drops aerosol entirely, matching the upstream
            feature set.

    Band optics are strongly correlated across bands, so ``"broadband"``
    buys a 16x narrower input for a modest fidelity loss. Which is the
    better trade is an empirical question for the training sweep, hence
    the switch rather than a fixed layout.

    Returns:
        List of ``(nlev,)`` feature arrays.

    """
    if band_mode == "none":
        return []
    if band_mode == "per_band":
        return [*aod, *ssa, *asy]
    if band_mode == "broadband":
        # AOD-weighted means: SSA and asymmetry are intensive, so summing
        # them across bands would be meaningless.
        total = jnp.sum(aod, axis=0)
        weight = jnp.maximum(total, 1e-30)
        return [total,
                jnp.sum(ssa * aod, axis=0) / weight,
                jnp.sum(asy * aod, axis=0) / weight]
    raise ValueError(
        f"Unknown band_mode {band_mode!r}; expected 'per_band', "
        "'broadband' or 'none'."
    )


def n_input_features(band_mode: str, n_bnd: int, n_base: int = 7) -> int:
    """Return the number of input features a given band handling produces.

    Lets the weight initialisers and the training driver size the input
    layer without building a dummy column first.
    """
    extra = {"none": 0, "broadband": 3, "per_band": 3 * n_bnd}
    if band_mode not in extra:
        raise ValueError(
            f"Unknown band_mode {band_mode!r}; expected 'per_band', "
            "'broadband' or 'none'."
        )
    return n_base + extra[band_mode]


def preprocess_sw_inputs(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o: jnp.ndarray,
    o3: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cos_zenith: jnp.ndarray,
    scaling: InputScaling,
    aerosol_aod: Optional[jnp.ndarray] = None,
    aerosol_ssa: Optional[jnp.ndarray] = None,
    aerosol_asy: Optional[jnp.ndarray] = None,
    band_mode: str = "none",
) -> jnp.ndarray:
    """Prepare SW NN inputs from atmospheric profiles.

    Follows the reference preprocessing: log(pressure), power transforms
    for gases, divide-by-max scaling.

    Args:
        temperature: Temperature profile (nlev,) [K].
        pressure: Pressure at full levels (nlev,) [Pa].
        h2o: Water vapour mass mixing ratio (nlev,) [kg/kg].
        o3: Ozone mass mixing ratio (nlev,) [kg/kg].
        cloud_water: Cloud liquid water path (nlev,) [kg/m^2].
        cloud_ice: Cloud ice water path (nlev,) [kg/m^2].
        cos_zenith: Cosine of solar zenith angle (scalar).
        scaling: Input normalization coefficients.
        aerosol_aod / aerosol_ssa / aerosol_asy: Per-SW-band optics,
            ``(n_bnd_sw, nlev)``. Required unless ``band_mode="none"``.
        band_mode: See :func:`_band_features`.

    Returns:
        Scaled input array (nlev, n_features), features ordered
        ``[T, log(p), h2o^1/4, o3^1/4, lwp, iwp, mu0, *aerosol]``.

    """
    log_p = jnp.log(jnp.maximum(pressure, 1.0))
    h2o_t = jnp.maximum(h2o, GAS_FLOOR) ** 0.25
    o3_t = jnp.maximum(o3, GAS_FLOOR) ** 0.25

    mu0 = jnp.broadcast_to(cos_zenith, temperature.shape)

    features = [temperature, log_p, h2o_t, o3_t, cloud_water, cloud_ice, mu0]
    if band_mode != "none":
        features += _band_features(
            aerosol_aod, aerosol_ssa, aerosol_asy, band_mode,
        )
    x = jnp.stack(features, axis=-1)

    # Divide-by-max scaling
    return x / jnp.maximum(scaling.x_max, 1e-30)


def preprocess_lw_inputs(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o: jnp.ndarray,
    o3: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    co2_vmr: float,
    scaling: InputScaling,
    aerosol_aod: Optional[jnp.ndarray] = None,
    aerosol_ssa: Optional[jnp.ndarray] = None,
    aerosol_asy: Optional[jnp.ndarray] = None,
    band_mode: str = "none",
) -> jnp.ndarray:
    """Prepare LW NN inputs from atmospheric profiles.

    Args:
        temperature: Temperature profile (nlev,) [K].
        pressure: Pressure at full levels (nlev,) [Pa].
        h2o: Water vapour mass mixing ratio (nlev,) [kg/kg].
        o3: Ozone mass mixing ratio (nlev,) [kg/kg].
        cloud_water: Cloud liquid water path (nlev,) [kg/m^2].
        cloud_ice: Cloud ice water path (nlev,) [kg/m^2].
        co2_vmr: CO2 volume mixing ratio (scalar).
        scaling: Input normalization coefficients.
        aerosol_aod / aerosol_ssa / aerosol_asy: Per-LW-band optics,
            ``(n_bnd_lw, nlev)``. Required unless ``band_mode="none"``.
        band_mode: See :func:`_band_features`.

    Returns:
        Scaled input array (nlev, n_features), features ordered
        ``[T, log(p), h2o^1/4, o3^1/4, lwp, iwp, co2, *aerosol]``.

    """
    log_p = jnp.log(jnp.maximum(pressure, 1.0))
    h2o_t = jnp.maximum(h2o, GAS_FLOOR) ** 0.25
    o3_t = jnp.maximum(o3, GAS_FLOOR) ** 0.25
    co2 = jnp.broadcast_to(jnp.asarray(co2_vmr), temperature.shape)

    features = [temperature, log_p, h2o_t, o3_t, cloud_water, cloud_ice, co2]
    if band_mode != "none":
        features += _band_features(
            aerosol_aod, aerosol_ssa, aerosol_asy, band_mode,
        )
    x = jnp.stack(features, axis=-1)

    return x / jnp.maximum(scaling.x_max, 1e-30)


# ---------------------------------------------------------------------------
# SW emulator (bidirectional GRU — brnn.py architecture)
# ---------------------------------------------------------------------------

def sw_emulator_column(
    x_seq: jnp.ndarray,
    surface_albedo: jnp.ndarray,
    weights: SWEmulatorWeights,
) -> jnp.ndarray:
    """Run the SW bidirectional-GRU emulator for one column.

    Args:
        x_seq: Preprocessed input features (nlev, n_features).
        surface_albedo: Surface albedo (1,).
        weights: SW emulator weights.

    Returns:
        Normalized flux predictions (nlev, 2) — (rsd_norm, rsu_norm).
        These represent the fraction of TOA flux reaching each level (down)
        and reflected upward (up), before boundary-condition reconstruction.

    """
    nneur = weights.gru_fwd.recurrent_kernel.shape[0]

    # Auxiliary inputs: albedo → initial hidden states for bidirectional GRU
    h0_fwd = dense(surface_albedo, weights.aux_dense_fwd, activation=None)
    h0_bwd = dense(surface_albedo, weights.aux_dense_bwd, activation=None)

    # Bidirectional GRU (merge_mode='concat')
    hidden_fwd = gru_forward_sequence(x_seq, h0_fwd, weights.gru_fwd)
    hidden_bwd = gru_backward_sequence(x_seq, h0_bwd, weights.gru_bwd)
    hidden_bi = jnp.concatenate([hidden_fwd, hidden_bwd], axis=-1)

    # Second GRU
    h0_gru2 = jnp.zeros(nneur)
    hidden2 = gru_forward_sequence(hidden_bi, h0_gru2, weights.gru2)

    # Output dense (sigmoid activation)
    output = dense(hidden2, weights.output_dense, activation=sigmoid)
    return output


def reconstruct_sw_fluxes(
    nn_output: jnp.ndarray,
    toa_sw_down: jnp.ndarray,
    surface_albedo: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct physical SW fluxes from normalized NN output.

    Args:
        nn_output: NN predictions (nlev, 2) — normalized (down, up) per layer.
        toa_sw_down: Incoming SW flux at TOA (scalar, W/m^2).
        surface_albedo: Surface albedo (scalar).

    Returns:
        sw_flux_down: Downward SW flux at interfaces (nlev+1,) [W/m^2].
        sw_flux_up: Upward SW flux at interfaces (nlev+1,) [W/m^2].

    """
    rsd_norm = nn_output[:, 0]  # normalized downwelling per layer
    rsu_norm = nn_output[:, 1]  # normalized upwelling per layer

    # Downwelling: TOA boundary = toa_sw_down, then NN predictions
    rsd = jnp.concatenate([jnp.array([1.0]), rsd_norm]) * toa_sw_down

    # Upwelling: NN predictions, then surface reflection = albedo * rsd_surface
    rsu_surface = surface_albedo * rsd[-1]
    rsu = jnp.concatenate([rsu_norm * toa_sw_down, jnp.array([rsu_surface])])

    return rsd, rsu


def reconstruct_sw_interface_fluxes(
    nn_output: jnp.ndarray,
    toa_sw_down: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scale normalized all-sky + clear-sky SW output to W/m^2.

    Args:
        nn_output: NN predictions at interfaces (nlev+1, 4), ordered
            (down, up, down_clear, up_clear) as fractions of the
            incoming TOA flux.
        toa_sw_down: Incoming SW flux at TOA (scalar, W/m^2).

    Normalizing by the incoming flux makes the target scale-free, so the
    network sees the same distribution at every solar zenith angle and at
    night. The TOA downward boundary is then exact by construction rather
    than something the network has to learn.

    Returns:
        ``(down, up, down_clear, up_clear)``, each (nlev+1,) [W/m^2].

    """
    fluxes = nn_output * toa_sw_down
    down = fluxes[:, 0].at[0].set(toa_sw_down)
    down_clear = fluxes[:, 2].at[0].set(toa_sw_down)
    return down, fluxes[:, 1], down_clear, fluxes[:, 3]


def lw_flux_scale(
    surface_temperature: jnp.ndarray,
    temperature: jnp.ndarray,
) -> jnp.ndarray:
    """Return the longwave flux normalizing scale sigma*T_max^4 [W/m^2].

    ``T_max`` is the warmest temperature anywhere in the column, surface
    included. No longwave flux at any interface can exceed black-body
    emission at that temperature, so every normalized target lands in
    [0, 1] — which is exactly the range the network's sigmoid output can
    represent.

    Scaling by the *surface emission* eps*sigma*T_s^4 instead does not
    have that property: over a cold surface under a warmer atmosphere
    (polar night, Antarctica) the outgoing longwave exceeds the surface
    emission, and in RRTMGP-labelled T63L47 columns ~14% of upward
    longwave values land above 1 that way, reaching 1.28 — unreachable
    for a sigmoid, so the error could not be trained away.
    """
    t_max = jnp.maximum(surface_temperature, jnp.max(temperature))
    return c.sbc * t_max ** 4


def reconstruct_lw_interface_fluxes(
    nn_output: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    temperature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scale normalized all-sky + clear-sky LW output to W/m^2.

    Args:
        nn_output: NN predictions at interfaces (nlev+1, 4), ordered
            (down, up, down_clear, up_clear), normalized by
            :func:`lw_flux_scale`.
        surface_temperature: Surface temperature (scalar, K).
        temperature: Column temperature profile (nlev,) [K].

    Surface emissivity is not part of the scale; the network receives it
    as the auxiliary input to its surface dense layer instead, so the
    dependence is learned rather than imposed.

    Returns:
        ``(down, up, down_clear, up_clear)``, each (nlev+1,) [W/m^2].

    """
    fluxes = nn_output * lw_flux_scale(surface_temperature, temperature)
    return fluxes[:, 0], fluxes[:, 1], fluxes[:, 2], fluxes[:, 3]


# ---------------------------------------------------------------------------
# LW emulator (forward-backward GRU — brnn2.py architecture)
# ---------------------------------------------------------------------------

def lw_emulator_column(
    x_seq: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    weights: LWEmulatorWeights,
    realign_backward: bool = True,
) -> jnp.ndarray:
    """Run the LW forward-backward GRU emulator for one column.

    Architecture: Forward GRU → surface Dense (last state + emissivity) →
    append to sequence → backward GRU → optional 3rd GRU → Dense output.

    Args:
        x_seq: Preprocessed input features (nlev, n_features).
        surface_emissivity: Surface emissivity (1,).
        weights: LW emulator weights.
        realign_backward: Pair the backward state *at* each level with the
            forward state there. False reproduces upstream checkpoints,
            whose bare Keras ``go_backwards`` GRU leaves its output in
            computation order and so concatenates level ``i`` against
            level ``nlev - i``. See :func:`gru_backward_sequence`.

    Returns:
        Normalized flux predictions (nlev+1, 2) — (rld_norm, rlu_norm).

    """
    nneur = weights.gru_fwd.recurrent_kernel.shape[0]
    h0 = jnp.zeros(nneur)

    # Forward GRU
    hidden_fwd = gru_forward_sequence(x_seq, h0, weights.gru_fwd)
    last_state = hidden_fwd[-1]

    # Surface processing: Dense on [last_state, emissivity]
    surface_input = jnp.concatenate([last_state, surface_emissivity])
    surface_hidden = dense(surface_input, weights.surface_dense, activation=None)

    # Append surface hidden state to forward sequence
    hidden_fwd_extended = jnp.concatenate(
        [hidden_fwd, surface_hidden[jnp.newaxis, :]], axis=0
    )

    # Backward GRU on extended sequence
    h0_bwd = jnp.zeros(nneur)
    hidden_bwd = gru_backward_sequence(
        hidden_fwd_extended, h0_bwd, weights.gru_bwd, realign=realign_backward,
    )

    # Concatenate forward and backward
    hidden_concat = jnp.concatenate([hidden_fwd_extended, hidden_bwd], axis=-1)

    # Third GRU
    h0_gru3 = jnp.zeros(nneur)
    hidden3 = gru_forward_sequence(hidden_concat, h0_gru3, weights.gru3)

    # Output dense (sigmoid activation)
    output = dense(hidden3, weights.output_dense, activation=sigmoid)
    return output


def reconstruct_lw_fluxes(
    nn_output: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct physical LW fluxes from normalized NN output.

    Args:
        nn_output: NN predictions (nlev+1, 2) — normalized (down, up) at interfaces.
        surface_temperature: Surface temperature (scalar, K).
        surface_emissivity: Surface emissivity (scalar).

    Returns:
        lw_flux_down: Downward LW flux at interfaces (nlev+1,) [W/m^2].
        lw_flux_up: Upward LW flux at interfaces (nlev+1,) [W/m^2].

    """
    # Scale factor: surface blackbody emission
    surface_emission = surface_emissivity * c.sbc * surface_temperature ** 4

    rld = nn_output[:, 0] * surface_emission
    rlu = nn_output[:, 1] * surface_emission

    return rld, rlu


# ---------------------------------------------------------------------------
# Heating rates from fluxes
# ---------------------------------------------------------------------------

def flux_to_heating_rate(
    flux_down: jnp.ndarray,
    flux_up: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
) -> jnp.ndarray:
    """Compute heating rate from flux profiles via flux divergence.

    dT/dt = -(g / c_p) * dF_net / dp

    Args:
        flux_down: Downward flux at interfaces (nlev+1,) [W/m^2].
        flux_up: Upward flux at interfaces (nlev+1,) [W/m^2].
        pressure_interfaces: Pressure at interfaces (nlev+1,) [Pa].

    Returns:
        Heating rate at full levels (nlev,) [K/s].

    """
    net_flux = flux_down - flux_up  # positive downward
    d_net_flux = jnp.diff(net_flux)  # (nlev,)
    dp = jnp.diff(pressure_interfaces)  # (nlev,)

    # dT/dt = (g/cp) * dF_net/dp  (positive heating when net flux increases downward)
    return (c.grav / c.cpd) * d_net_flux / dp


# ---------------------------------------------------------------------------
# Weight loading from NetCDF
# ---------------------------------------------------------------------------

# Sub-weight layout of :class:`LWEmulatorWeights`, written out explicitly
# so the on-disk variable names are fixed by this table. Reflecting over
# the pytree instead would let a struct field rename silently change the
# file format and break every previously trained checkpoint.
_GRU_FIELDS = ("kernel", "recurrent_kernel", "bias")
_DENSE_FIELDS = ("kernel", "bias")
_EMULATOR_LAYERS = (
    ("gru_fwd", GRUWeights, _GRU_FIELDS),
    ("surface_dense", DenseWeights, _DENSE_FIELDS),
    ("gru_bwd", GRUWeights, _GRU_FIELDS),
    ("gru3", GRUWeights, _GRU_FIELDS),
    ("output_dense", DenseWeights, _DENSE_FIELDS),
)


def _weight_dims(name: str, array) -> tuple[str, ...]:
    """Per-variable dimension names, so no two variables can clash.

    Every array gets private dims rather than shared ones like ``units``:
    the layers have genuinely independent sizes and a shared name would
    make xarray reject a file the moment two of them differed.
    """
    stem = name.replace(".", "_")
    return tuple(f"{stem}_dim{i}" for i in range(array.ndim))


def save_emulator_weights(
    filepath: str,
    weights: EmulatorWeights,
    sw_scaling: InputScaling,
    lw_scaling: InputScaling,
    metadata: Optional[dict] = None,
) -> None:
    """Write trained emulator weights and input scalings to NetCDF.

    This is jax-gcm's *own* checkpoint format — flat, explicitly named
    variables such as ``sw.gru_fwd.kernel`` plus ``sw_x_max`` /
    ``lw_x_max``. It is unrelated to :func:`load_weights_from_netcdf`,
    which reads the upstream rte-rrtmgp-nn dense-layer format
    (``w1``/``b1``/...). Only files written here can be fed to
    :func:`load_emulator_weights` or to ``NNEmulatorRadiation``'s
    ``weights_file``.

    Args:
        filepath: Destination ``.nc`` path.
        weights: Trained weights. Both slots are
            :class:`LWEmulatorWeights` (see :class:`EmulatorWeights`).
        sw_scaling / lw_scaling: Input normalization for each network.
        metadata: Small scalars/strings stored as NetCDF global
            attributes and returned by :func:`load_emulator_weights`.
            ``band_mode`` belongs here — the physics term refuses to load
            a file without it, since the band handling fixes the input
            width and a mismatch is otherwise silent.

    Arrays are stored in their native dtype with no packing, so a
    float32 round trip is bit-identical.

    """
    import numpy as np
    import xarray as xr

    data_vars = {}

    def _put(name: str, array) -> None:
        values = np.asarray(array)
        data_vars[name] = (_weight_dims(name, values), values)

    for side, side_weights in (("sw", weights.sw), ("lw", weights.lw)):
        for layer, _, fields in _EMULATOR_LAYERS:
            layer_weights = getattr(side_weights, layer)
            for field in fields:
                _put(f"{side}.{layer}.{field}", getattr(layer_weights, field))
    _put("sw_x_max", sw_scaling.x_max)
    _put("lw_x_max", lw_scaling.x_max)

    ds = xr.Dataset(data_vars, attrs=dict(metadata or {}))
    ds.to_netcdf(filepath)
    ds.close()


def load_emulator_weights(
    filepath: str,
) -> tuple[EmulatorWeights, InputScaling, InputScaling, dict]:
    """Read back a checkpoint written by :func:`save_emulator_weights`.

    Reads jax-gcm's own emulator format, *not* the upstream
    rte-rrtmgp-nn one handled by :func:`load_weights_from_netcdf`.

    Args:
        filepath: Path to the ``.nc`` file.

    Returns:
        ``(weights, sw_scaling, lw_scaling, metadata)``. ``metadata`` is
        the file's global attributes, empty if it carries none.

    """
    import xarray as xr

    with xr.open_dataset(filepath) as ds:
        def _get(name: str):
            if name not in ds.variables:
                raise KeyError(
                    f"{filepath!r} has no variable {name!r}; it is not an "
                    "emulator checkpoint written by save_emulator_weights."
                )
            return jnp.asarray(ds[name].values)

        def _side(side: str) -> LWEmulatorWeights:
            return LWEmulatorWeights(**{
                layer: cls(**{f: _get(f"{side}.{layer}.{f}") for f in fields})
                for layer, cls, fields in _EMULATOR_LAYERS
            })

        weights = EmulatorWeights(sw=_side("sw"), lw=_side("lw"))
        sw_scaling = InputScaling(x_max=_get("sw_x_max"))
        lw_scaling = InputScaling(x_max=_get("lw_x_max"))
        metadata = dict(ds.attrs)

    return weights, sw_scaling, lw_scaling, metadata


def load_weights_from_netcdf(filepath: str) -> tuple:
    """Load NN weights from a NetCDF file in the rte-rrtmgp-nn format.

    User-facing utility (no internal callers): use it to load pretrained
    rte-rrtmgp-nn weights when configuring :class:`NNEmulatorRadiation`.

    This reads the *upstream* per-layer ``w1``/``b1``/... dense format
    from peterukk/rte-rrtmgp-nn. It is NOT the format jax-gcm's own
    trainer writes — for that use :func:`load_emulator_weights`.

    Args:
        filepath: Path to the .nc file.

    Returns:
        Tuple of (list of DenseWeights, InputScaling, activation_names).

    """
    import xarray as xr

    ds = xr.open_dataset(filepath)

    layer_weights = []
    i = 1
    while f"w{i}" in ds:
        kernel = jnp.array(ds[f"w{i}"].values)
        bias = jnp.array(ds[f"b{i}"].values)
        layer_weights.append(DenseWeights(kernel=kernel, bias=bias))
        i += 1

    x_max = jnp.array(ds["xmax"].values) if "xmax" in ds else jnp.ones(1)
    scaling = InputScaling(x_max=x_max)

    activation_names = []
    if "activations" in ds:
        activation_names = [
            str(a) for a in ds["activations"].values
        ]

    ds.close()
    return layer_weights, scaling, activation_names


# ---------------------------------------------------------------------------
# Random weight initialization (for testing / training from scratch)
# ---------------------------------------------------------------------------

def init_gru_weights(
    input_dim: int,
    units: int,
    key: jax.Array,
) -> GRUWeights:
    """Initialize GRU weights with Glorot uniform."""
    k1, k2, k3 = jax.random.split(key, 3)
    scale_k = jnp.sqrt(2.0 / (input_dim + units))
    scale_r = jnp.sqrt(2.0 / (units + units))
    return GRUWeights(
        kernel=jax.random.normal(k1, (input_dim, 3 * units)) * scale_k,
        recurrent_kernel=jax.random.normal(k2, (units, 3 * units)) * scale_r,
        bias=jnp.zeros((2, 3 * units)),
    )


def init_dense_weights(
    input_dim: int,
    output_dim: int,
    key: jax.Array,
) -> DenseWeights:
    """Initialize Dense weights with Glorot uniform."""
    scale = jnp.sqrt(2.0 / (input_dim + output_dim))
    return DenseWeights(
        kernel=jax.random.normal(key, (input_dim, output_dim)) * scale,
        bias=jnp.zeros(output_dim),
    )


def init_sw_emulator_weights(
    n_features: int = 7,
    units: int = 16,
    n_outputs: int = 2,
    key: Optional[jax.Array] = None,
) -> SWEmulatorWeights:
    """Initialize random SW emulator weights.

    Args:
        n_features: Number of input features per layer.
        units: GRU hidden size.
        n_outputs: Number of outputs (default 2: rsd_norm, rsu_norm).
        key: PRNG key (default: key(0)).

    """
    if key is None:
        key = jax.random.key(0)
    keys = jax.random.split(key, 6)
    return SWEmulatorWeights(
        aux_dense_fwd=init_dense_weights(1, units, keys[0]),
        aux_dense_bwd=init_dense_weights(1, units, keys[1]),
        gru_fwd=init_gru_weights(n_features, units, keys[2]),
        gru_bwd=init_gru_weights(n_features, units, keys[3]),
        gru2=init_gru_weights(2 * units, units, keys[4]),
        output_dense=init_dense_weights(units, n_outputs, keys[5]),
    )


def init_lw_emulator_weights(
    n_features: int = 7,
    units: int = 16,
    n_outputs: int = 2,
    key: Optional[jax.Array] = None,
) -> LWEmulatorWeights:
    """Initialize random LW emulator weights.

    Args:
        n_features: Number of input features per layer.
        units: GRU hidden size.
        n_outputs: Number of outputs (default 2: rld_norm, rlu_norm).
        key: PRNG key (default: key(1)).

    """
    if key is None:
        key = jax.random.key(1)
    keys = jax.random.split(key, 5)
    return LWEmulatorWeights(
        gru_fwd=init_gru_weights(n_features, units, keys[0]),
        surface_dense=init_dense_weights(units + 1, units, keys[1]),
        gru_bwd=init_gru_weights(units, units, keys[2]),
        gru3=init_gru_weights(2 * units, units, keys[3]),
        output_dense=init_dense_weights(units, n_outputs, keys[4]),
    )


def init_emulator_weights(
    sw_features: int = 7,
    lw_features: int = 7,
    units: int = 16,
    n_outputs: int = 4,
    key: Optional[jax.Array] = None,
) -> EmulatorWeights:
    """Initialize random weights for both SW and LW emulators.

    The starting point for training from scratch. Both slots get the
    surface-aux architecture (see :class:`EmulatorWeights`). Four outputs
    by default: all-sky and clear-sky, up and down.

    Args:
        sw_features / lw_features: Input feature counts, from
            :func:`n_input_features` for the chosen ``band_mode``.
        units: GRU hidden size.
        n_outputs: Channels per interface.
        key: PRNG key (default: key(42)).

    """
    if key is None:
        key = jax.random.key(42)
    k1, k2 = jax.random.split(key)
    return EmulatorWeights(
        sw=init_lw_emulator_weights(
            n_features=sw_features, units=units,
            n_outputs=n_outputs, key=k1,
        ),
        lw=init_lw_emulator_weights(
            n_features=lw_features, units=units,
            n_outputs=n_outputs, key=k2,
        ),
    )
