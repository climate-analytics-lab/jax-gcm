"""Emissions data pipeline: regrid an arbitrary source onto the model grid.

Public API:

* :func:`jcm.data.emissions.regrid.build_regridder` / ``model_grid`` — a light,
  first-order conservative remap from any ``(lon, lat, area)`` source (regular or
  unstructured) onto the model's spectral Gaussian grid.
* :func:`jcm.data.emissions.prepare.prepare_emissions` + :class:`Channel` /
  ``cesm_cmip_anthro`` — map source variables to the emissions-file contract and
  regrid them.
* :func:`jcm.data.emissions.downloader.fetch` — host-agnostic resolve-to-local.

The produced dataset is loaded onto ``ForcingData`` via
:func:`jcm.forcing.read_anthropogenic_emissions`. See
``.claude/aerosol_emissions_plan.md`` for the runtime emissions-file contract.
"""

from jcm.data.emissions.prepare import (
    Channel,
    SpeciatedChannel,
    cesm_bb4cmip7,
    cesm_cmip_anthro,
    cesm_mam4_speciated,
    prepare_emissions,
    prepare_speciated_emissions,
)
from jcm.data.emissions.regrid import build_regridder, model_grid

__all__ = [
    "build_regridder",
    "model_grid",
    "prepare_emissions",
    "Channel",
    "cesm_cmip_anthro",
    "cesm_bb4cmip7",
    "prepare_speciated_emissions",
    "SpeciatedChannel",
    "cesm_mam4_speciated",
]
