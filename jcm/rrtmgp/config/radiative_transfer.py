"""
Configuration for the RRTMGP radiation library. 
Adapted from https://github.com/google-research/swirl-jatmos.
"""

import dataclasses
import dataclasses_json


@dataclasses.dataclass(frozen=True, kw_only=True)
class RRTMOptics:
  """Parameters required by the radiation optics library."""
  # Path of NetCDF file containing the longwave lookup tables.
  longwave_nc_filepath: str
  # Path of NetCDF file containing the shortwave lookup tables.
  shortwave_nc_filepath: str
  # Path of NetCDF file containing the cloud longwave lookup table.
  cloud_longwave_nc_filepath: str
  # Path of NetCDF file containing the cloud shortwave lookup table.
  cloud_shortwave_nc_filepath: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class GrayAtmosphereOptics:
  """Parameters for gray atmosphere optics."""
  # Reference surface pressure.
  p0: float = 1e5
  # The ratio of the pressure scale height to the partial-pressure scale height
  # of the infrared absorber.
  alpha: float = 3.5
  # Longwave optical depth of the entire gray atmosphere.
  d0_lw: float = 0.0
  # Shortwave optical depth of the entire gray atmosphere.
  d0_sw: float = 0.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpticsParameters(dataclasses_json.DataClassJsonMixin):
  optics: RRTMOptics | GrayAtmosphereOptics


@dataclasses.dataclass(frozen=True, kw_only=True)
class AtmosphericStateCfg:
  """State of the atmosphere for radiation optics."""

  # Surface emissivity; the same for all bands.
  sfc_emis: float = 0.0
  # Surface albedo; the same for all bands.
  sfc_alb: float = 0.0
  # The solar zenith angle (in radians).
  zenith: float = 0.0
  # The total solar irradiance (in W/m²).
  irrad: float = 0.0
  # The longwave incident flux at the top of the atmosphere (in W/m²).
  toa_flux_lw: float = 0.0
  # Path of a csv file containing volume mixing ratio sounding for particular
  # gas species on a uniform log scale pressure grid. One column should be
  # labeled 'p_ref' for the corresponding pressure profile in Pa that will be
  # used when interpolating to the simulation's pressure field. All other
  # columns should be labeled with the chemical formula of the gas they
  # correspond to. Note that the pressure grid under p_ref must be
  # log-uniformly spaced to ensure accurate interpolation of the profiles.
  vmr_sounding_filepath: str = ''
  # The path of a json file containing volume mixing ratio global means,
  # indexed by the gas chemical formula. This should be populated by gas
  # species that have little variability in the atmosphere. The contents of the
  # file should be a dictionary in json format.
  vmr_global_mean_filepath: str = ''


@dataclasses.dataclass(frozen=True, kw_only=True)
class RadiativeTransfer(dataclasses_json.DataClassJsonMixin):
  """Parameters for the RRTMGP radiative transfer model."""
  # The optics library that calculates optical depth, single-scattering albedo,
  # and other local optical properties of the layered atmosphere.
  optics: OpticsParameters
  # Data about the prevalent atmospheric gases present in the atmosphere as
  # well as boundary conditions for the surface and top of atmosphere.
  atmospheric_state_cfg: AtmosphericStateCfg
  # The time interval between updates of the radiative fluxes (in seconds).
  # Since the RRTMGP model is very computationally expensive, it is highly
  # recommended that the update cycle be no less than 10 minutes. However, too
  # long an update cycle can lead to instabilities. See Pauluis and Emanuel
  # (2004), https://doi.org/10.1175/1520-0493(2004)132<0673:NIRFIC>2.0.CO;2
  update_cycle_seconds: float = 600.0
  # Supercycling time interval between applying the radiative transfer model in
  # the energy equation (in seconds). When it is applied, its amplitude will be
  # scaled larger appropriately. This option exists because the radiation
  # update in the temperature equation can be too small relative to the
  # existing temperature and be lost to floating point truncation in single
  # precision. In practice, the supercycling time interval should be less
  # than or equal to `update_cycle_seconds`. Furthermore, the supercycling time
  # interval should be as small as possible, but large enough such that
  # floating point precision is not an issue. If this field is set to the
  # default value of 0, we do no supercycling and apply the radiative heating at
  # every step.
  apply_cadence_seconds: float = 0.0
  # If True, use jax.lax.scan instead of for loop for scanning through an array.
  use_scan: bool = False

  # ******** Output options ********
  # If True, save the longwave and shortwave heating rates individually in the
  # states dictionary, under names `rad_heat_lw_3d` and `rad_heat_sw_3d`.
  save_lw_sw_heating_rates: bool = False
  do_clear_sky: bool = False  # If true, compute clear sky radiative transfer.
