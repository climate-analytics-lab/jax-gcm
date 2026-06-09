import logging

from jcm.model import Model, ModelPredictions
from jcm.prescribed_state_model import (
    PrescribedStateModel,
    PrescribedStatePredictions,
)
from jcm.single_column_model import SCMPredictions, SingleColumnModel
from jcm.utils import (
    create_initial_tracers,
    create_single_column_state,
    load_states_from_xarray,
)

logging.basicConfig(format='%(name)s: %(asctime)s %(levelname)s: %(message)s')

__version__ = "2.0.0b1"

__all__ = [
    "Model",
    "ModelPredictions",
    "PrescribedStateModel",
    "PrescribedStatePredictions",
    "SCMPredictions",
    "SingleColumnModel",
    "create_initial_tracers",
    "create_single_column_state",
    "load_states_from_xarray",
]
