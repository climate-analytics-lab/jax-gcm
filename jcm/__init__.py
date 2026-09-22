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

# No logging configuration here, deliberately. A library emits records and
# leaves handlers and levels to whoever assembles the process; the
# ``basicConfig`` that used to sit here set a format on the *root* logger for
# every importer. It could not even have served the CLI, which is the one
# place jcm is the application: Hydra's ``job_logging`` runs ``dictConfig``
# with a ``root:`` section, which replaces root's handlers outright. The CLI's
# knob lives in ``runners._apply_log_level`` instead. Enforced by
# ``model_test.TestModelLogging``.

__version__ = "3.0.0rc1"

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
