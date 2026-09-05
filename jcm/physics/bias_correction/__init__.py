"""State-dependent NN bias correction for SPEEDY physics (issue #356).

A learned replacement for hand-written nudging: a small neural network reads
the model's own column state and outputs a correction tendency that is added
on top of the SPEEDY tendencies. ERA5 is needed only to train the network;
at run time the correction depends on the model state alone.
"""

from jcm.physics.bias_correction.nn_bias_correction import (
    NNBiasCorrection,
    bias_correction_tendency,
    build_context,
    make_bias_correction,
    load_bias_correction,
    polar_taper_factor,
    surface_taper_factor,
    state_to_features,
    widen_first_layer,
    remap_first_layer,
    init_mlp,
    mlp,
    dense,
    DenseWeights,
    FIELD_ORDER,
    CONTEXT_FEATURES,
)
from jcm.physics.bias_correction.online_training import (
    CurriculumStage,
    default_curriculum,
    lat_weights,
    level_weights,
    make_rollout_loss,
    make_train_step,
    state_error,
    stds_from_stats,
    train_online,
)

__all__ = [
    "NNBiasCorrection",
    "bias_correction_tendency",
    "build_context",
    "make_bias_correction",
    "load_bias_correction",
    "polar_taper_factor",
    "surface_taper_factor",
    "state_to_features",
    "widen_first_layer",
    "remap_first_layer",
    "CONTEXT_FEATURES",
    "init_mlp",
    "mlp",
    "dense",
    "DenseWeights",
    "FIELD_ORDER",
    "CurriculumStage",
    "default_curriculum",
    "lat_weights",
    "level_weights",
    "make_rollout_loss",
    "make_train_step",
    "state_error",
    "stds_from_stats",
    "train_online",
]
