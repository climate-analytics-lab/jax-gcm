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
    """All weights for the emulated radiation scheme."""

    sw: SWEmulatorWeights
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
) -> jnp.ndarray:
    """Run a GRU backward over a sequence (go_backwards=True).

    Args:
        x_seq: Input sequence (seq_len, input_dim).
        h0: Initial hidden state (units,).
        weights: GRU weights.

    Returns:
        Hidden states at all time steps (seq_len, units), reversed back
        to original ordering.

    """
    hidden_rev = gru_forward_sequence(x_seq[::-1], h0, weights)
    return hidden_rev[::-1]


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

def preprocess_sw_inputs(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o: jnp.ndarray,
    o3: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cos_zenith: jnp.ndarray,
    scaling: InputScaling,
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

    Returns:
        Scaled input array (nlev, n_features).

    """
    log_p = jnp.log(jnp.maximum(pressure, 1.0))
    h2o_t = h2o ** 0.25
    o3_t = o3 ** 0.25

    mu0 = jnp.broadcast_to(cos_zenith, temperature.shape)

    # Stack features: [T, log(p), h2o^1/4, o3^1/4, lwp, iwp, mu0]
    x = jnp.stack([temperature, log_p, h2o_t, o3_t,
                    cloud_water, cloud_ice, mu0], axis=-1)

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

    Returns:
        Scaled input array (nlev, n_features).

    """
    log_p = jnp.log(jnp.maximum(pressure, 1.0))
    h2o_t = h2o ** 0.25
    o3_t = o3 ** 0.25
    co2 = jnp.broadcast_to(jnp.asarray(co2_vmr), temperature.shape)

    # Stack features: [T, log(p), h2o^1/4, o3^1/4, lwp, iwp, co2]
    x = jnp.stack([temperature, log_p, h2o_t, o3_t,
                    cloud_water, cloud_ice, co2], axis=-1)

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


# ---------------------------------------------------------------------------
# LW emulator (forward-backward GRU — brnn2.py architecture)
# ---------------------------------------------------------------------------

def lw_emulator_column(
    x_seq: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    weights: LWEmulatorWeights,
) -> jnp.ndarray:
    """Run the LW forward-backward GRU emulator for one column.

    Architecture: Forward GRU → surface Dense (last state + emissivity) →
    append to sequence → backward GRU → optional 3rd GRU → Dense output.

    Args:
        x_seq: Preprocessed input features (nlev, n_features).
        surface_emissivity: Surface emissivity (1,).
        weights: LW emulator weights.

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
    hidden_bwd = gru_backward_sequence(hidden_fwd_extended, h0_bwd, weights.gru_bwd)

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
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant

    # Scale factor: surface blackbody emission
    surface_emission = surface_emissivity * sigma * surface_temperature ** 4

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
    g = 9.81
    cp = 1004.0

    net_flux = flux_down - flux_up  # positive downward
    d_net_flux = jnp.diff(net_flux)  # (nlev,)
    dp = jnp.diff(pressure_interfaces)  # (nlev,)

    # dT/dt = (g/cp) * dF_net/dp  (positive heating when net flux increases downward)
    return (g / cp) * d_net_flux / dp


# ---------------------------------------------------------------------------
# Weight loading from NetCDF
# ---------------------------------------------------------------------------

def load_weights_from_netcdf(filepath: str) -> tuple:
    """Load NN weights from a NetCDF file in the rte-rrtmgp-nn format.

    The NetCDF file contains weight matrices and bias vectors for each layer,
    along with activation function names and scaling coefficients.

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
    key: Optional[jax.Array] = None,
) -> EmulatorWeights:
    """Initialize random weights for both SW and LW emulators."""
    if key is None:
        key = jax.random.key(42)
    k1, k2 = jax.random.split(key)
    return EmulatorWeights(
        sw=init_sw_emulator_weights(n_features=sw_features, units=units, key=k1),
        lw=init_lw_emulator_weights(n_features=lw_features, units=units, key=k2),
    )
