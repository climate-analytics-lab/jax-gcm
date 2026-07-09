"""CloudSat satellite-simulator diagnostic (COSP warm-rain hook).

Feeds the ECHAM physics state into the external ``jcosp`` package (a JAX
translation of the COSP v2 CloudSat/Quickbeam radar simulator,
https://github.com/climate-analytics-lab/jax-cosp) and surfaces the
Mülmenstädt et al. (2020, doi:10.1126/sciadv.aaz6433) warm-rain occurrence
flags plus CloudSat 2C-PRECIP-COLUMN-style diagnostics as model output. The
climatological warm-rain fraction — the aerosol–cloud-interaction
calibration target of that paper — is ``<warm_rain>/(<warm_rain> +
<cold_rain>)`` computed from the *time-averaged* flags.

Input mapping (and its documented approximations):

* Cloud fraction, gridbox-mean qc/qi and the per-level large-scale rain and
  snow flux profiles come from the ``"clouds"`` CloudData. jcm's Tiedtke
  scheme detrains convective condensate into the same qc/qi fields and no
  separate convective cloud fraction exists, so ALL cloud is presented to
  SCOPS as large-scale (``conv_frac = 0``). This differs from ECHAM-COSP,
  which passes convective cloud separately; revisit if a convective cloud
  fraction diagnostic is ever added.
* Convective precipitation: only the surface flux ``precip_conv`` is
  available from the convection scheme. It is spread uniformly over the
  levels at/below the convective condensate top (where ``qc_conv + qi_conv``
  first becomes nonzero), split rain/snow by the local melting point. This
  is an explicitly-labelled stopgap until Tiedtke's internal per-level flux
  profiles are threaded out; it primarily affects the convective-precip
  radar signal, not the stratiform warm-rain target.
* The physics frame is already TOA-first (level 0 = model top), matching
  jcosp's convention — no vertical flip.
* jcosp's subcolumn sampling is stochastic (SCOPS); the key is derived
  deterministically from the configured seed and the surface-pressure field,
  so a given model state always produces the same subcolumns (mirroring
  COSP's own pressure-based seeding) without any cross-step carry.

The jcosp dependency is imported lazily so jcm works without it; install
with ``pip install jcm[cosp]``.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData


class CloudsatCosp(PhysicsTerm):
    """CloudSat radar simulator + warm-rain classification diagnostic.

    Runs downstream of the cloud microphysics (category ``diagnostics``) and
    writes per-gridpoint occurrence fractions over ``ncolumns`` stochastic
    subcolumns:

    - ``cosp_warm_rain`` / ``cosp_cold_rain``: subcolumn fraction whose max
      reflectivity exceeds 0 dBZe with a liquid / ice-topped source layer;
    - ``cosp_warm_drizzle`` / ``cosp_cold_drizzle``: same for the
      (-15, 0] dBZe drizzle regime;
    - ``cosp_pia``: mean path-integrated attenuation (dB);
    - ``cosp_precip_cover``: (ncols, 10) CloudSat precipitation-class cover.

    All are plain arrays, so ``output_averages`` accumulates them into the
    climatology the Mülmenstädt et al. (2015) satellite target is defined on.
    """

    name: ClassVar[str] = "cloudsat_cosp"
    category: ClassVar[str] = "diagnostics"
    requires: ClassVar[tuple[str, ...]] = (
        "clouds", "convection", "pressure_full", "height_full", "height_half")
    provides: ClassVar[tuple[str, ...]] = ("cosp_warm_rain",)

    def __init__(self, ncolumns: int = 40, overlap: int = 3, seed: int = 0,
                 lut_path=None):
        """Configure the simulator; loads the radar Z-scale LUT eagerly.

        ncolumns trades subcolumn sampling noise against cost (COSP's
        canonical value is 100; the flags are time-averaged climatologies,
        so a smaller in-line value is acceptable and cheaper). ``lut_path``
        overrides the packaged 94-GHz single-moment table.
        """
        try:
            import jcosp  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "CloudsatCosp requires the jax-cosp package "
                "(pip install jcm[cosp])") from err
        import importlib.resources

        from jcosp.radar.optics import load_lut

        if lut_path is None:
            lut_path = (importlib.resources.files("jcosp") / "data"
                        / "quickbeam_lut_94GHz_1M.npz")
        # nnx.data marks the LUT arrays as pytree data on the module.
        self._lut = nnx.data(load_lut(lut_path))
        self.ncolumns = int(ncolumns)
        self.overlap = int(overlap)
        self.seed = int(seed)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Simulate CloudSat reflectivity and classify warm/cold rain."""
        from jcosp import config as jconfig
        from jcosp.simulator import CloudsatInputs, simulate_cloudsat

        clouds = diagnostics["clouds"]
        conv = diagnostics["convection"]
        nlev, ncols = state.temperature.shape

        # Convective surface precip spread below the convective condensate
        # top (stopgap; see module docstring). The mask is 1 from the first
        # level (from the top) with convective condensate down to the
        # surface, and empty for columns without convective condensate.
        conv_cond = conv.qc_conv + conv.qi_conv
        below_top = jnp.cumsum((conv_cond > 0.0).astype(state.temperature.dtype),
                               axis=0) > 0.0
        conv_flux = conv.precip_conv[None, :] * below_top
        frozen = state.temperature < c.tmelt
        fl_ccrain = jnp.where(frozen, 0.0, conv_flux)
        fl_ccsnow = jnp.where(frozen, conv_flux, 0.0)

        # Cloud effective radii (um -> m); zero means "PSD defaults" and the
        # remaining hydrometeor slots always use the PSD defaults.
        reff = jnp.zeros((nlev, ncols, jconfig.N_HYDRO),
                         dtype=state.temperature.dtype)
        reff = reff.at[..., jconfig.I_LSCLIQ].set(clouds.r_eff_liq * 1e-6)
        reff = reff.at[..., jconfig.I_LSCICE].set(clouds.r_eff_ice * 1e-6)

        inputs = CloudsatInputs(
            pressure=diagnostics["pressure_full"],
            temperature=state.temperature,
            specific_humidity=state.specific_humidity,
            zfull=diagnostics["height_full"],
            # jcosp wants each layer's bottom interface; height_half holds
            # the nlev+1 interfaces top-first.
            zhalf=diagnostics["height_half"][1:],
            cloud_frac=clouds.cloud_fraction,
            conv_frac=jnp.zeros_like(clouds.cloud_fraction),
            mr_lsliq=clouds.qc,
            mr_lsice=clouds.qi,
            mr_ccliq=jnp.zeros_like(clouds.qc),
            mr_ccice=jnp.zeros_like(clouds.qc),
            fl_lsrain=clouds.rain_flux,
            fl_lssnow=clouds.snow_flux,
            fl_lsgrpl=jnp.zeros_like(clouds.rain_flux),
            fl_ccrain=fl_ccrain,
            fl_ccsnow=fl_ccsnow,
            land=jnp.reshape(terrain.fmask, (-1,)),
            surfelev=jnp.reshape(terrain.orog, (-1,)),
            # Near-surface air temperature proxy (no 2-m diagnostic in the
            # physics state); only steers the land-point precip-flag path.
            t2m=state.temperature[-1],
            reff=reff,
        )

        # Deterministic subcolumn key from the seed and the surface-pressure
        # field: same state -> same subcolumns, different steps -> fresh
        # draws (COSP itself seeds SCOPS from surface pressure).
        ps_hash = jnp.sum(
            (state.normalized_surface_pressure * 1e5).astype(jnp.int32))
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), ps_hash)

        out = simulate_cloudsat(inputs, self._lut, key=key,
                                ncolumns=self.ncolumns, overlap=self.overlap)

        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {
            **diagnostics,
            "cosp_warm_rain": out.warm_rain.warm_rain,
            "cosp_cold_rain": out.warm_rain.cold_rain,
            "cosp_warm_drizzle": out.warm_rain.warm_drizzle,
            "cosp_cold_drizzle": out.warm_rain.cold_drizzle,
            "cosp_pia": out.pia,
            # (ncols, nclass) so the single ncols axis leads and the class
            # axis expands to cosp_precip_cover.0..9 in the output.
            "cosp_precip_cover": out.precip_cover.T,
        }
