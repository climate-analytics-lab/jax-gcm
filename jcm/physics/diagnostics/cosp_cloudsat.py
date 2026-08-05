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

The CALIPSO and MODIS simulators (``enable_calipso`` / ``enable_modis``)
run on the SAME SCOPS realization as the radar, which is how COSP itself is
wired: one subcolumn draw feeds every simulator, so the instruments see a
mutually consistent cloud field and the (expensive) sampling is paid once.
They provide the CFMIP diagnostics the AeroCom phase-4 ACI and IMO2020
protocols request (jax-gcm#581):

* CALIPSO ``cltcalipso`` / ``cllcalipso`` / ``clmcalipso`` / ``clhcalipso``
  from ``lidar_optics`` -> ``lidar_subcolumn`` -> ``lidar_column``;
* MODIS ``cltmodis`` / ``clwmodis`` / ``climodis`` / ``tauwmodis`` /
  ``tauimodis`` / ``reffclwmodis`` / ``reffclimodis``.

``parasolRefl`` is NOT available: jax-cosp carries the PARASOL optical-depth
inputs (``tau_sfc_liq``/``tau_sfc_ice``) but no reflectance simulator, so
that one AeroCom field needs upstream work.

COST — these are expensive and are OFF BY DEFAULT for that reason.
Measured at T63L47 on an A100 against the same run with no COSP at all:

    radar only (enable_cosp)            +4.5 %
    radar + MODIS                      +12.8 %
    radar + CALIPSO                    +20.1 %
    radar + MODIS + CALIPSO            +20.4 %

Two things worth knowing. The subcolumn sampling is shared, so MODIS is
nearly free once CALIPSO is paying for the lidar optics (+20.1 -> +20.4 %);
enable both together rather than separately. And CALIPSO is the expensive
one — it builds its own 532 nm optics per subcolumn and regrids onto the
40 x 480 m statistical grid, which the radar path does not do. At ~20 %
this exceeds the ~10 % budget for routinely-on diagnostics, so it is
intended for the AeroCom ACI / IMO2020 runs specifically, not for general
production. ``ncolumns`` trades sampling noise against cost roughly
linearly if a cheaper compromise is wanted.

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


def _layer_optical_depth(clouds, pressure_half):
    """Gridbox-mean 0.67 um layer cloud optical depth for the MODIS optics.

    Geometric-optics limit ``tau = 3 W / (2 rho r_eff)`` with ``W`` the layer
    condensate path — the same large-size-parameter form the MODIS simulator
    assumes for its retrievals. Effective radii are floored so a
    condensate-free layer (r_eff = 0) cannot divide by zero; the guard is
    only reached where the numerator is zero too.
    """
    dm = jnp.diff(pressure_half, axis=0) / c.grav
    r_liq = jnp.maximum(clouds.r_eff_liq * 1e-6, 1e-9)
    r_ice = jnp.maximum(clouds.r_eff_ice * 1e-6, 1e-9)
    tau_liq = 1.5 * clouds.qc * dm / (1000.0 * r_liq)
    tau_ice = 1.5 * clouds.qi * dm / (917.0 * r_ice)
    return tau_liq + tau_ice


# Values at or below this are the COSP R_UNDEF missing-data sentinel.
_R_UNDEF_THRESHOLD = -1e20


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
    # ``pressure_half`` is read unconditionally in ``__call__`` — layer mass
    # for the optical-depth integral, and passed straight to the simulators —
    # so it belongs here. An undeclared unconditional read is exactly what
    # ``requires_audit_test`` exists to catch.
    requires: ClassVar[tuple[str, ...]] = (
        "clouds", "convection", "pressure_full", "pressure_half",
        "height_full", "height_half")
    # Static key set: the diagnostics dict is part of the scan carry, so
    # every enabled simulator must publish the same keys on every step.
    provides: ClassVar[tuple[str, ...]] = (
        "cosp_warm_rain",
        # CALIPSO (enable_calipso)
        "cltcalipso", "cllcalipso", "clmcalipso", "clhcalipso",
        # MODIS (enable_modis)
        "cltmodis", "clwmodis", "climodis", "tauwmodis", "tauimodis",
        "reffclwmodis", "reffclimodis", "lwpmodis", "iwpmodis",
    )

    def __init__(self, ncolumns: int = 40, overlap: int = 3, seed: int = 0,
                 lut_path=None, enable_calipso: bool = False,
                 enable_modis: bool = False):
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
        # Static flags (Python branches at trace time), so an unused
        # simulator costs nothing rather than being masked out.
        self.enable_calipso = bool(enable_calipso)
        self.enable_modis = bool(enable_modis)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Simulate CloudSat reflectivity and classify warm/cold rain."""
        from jcosp import config as jconfig
        from jcosp.simulator import (
            CloudsatInputs, simulate_cloudsat, simulate_cloudsat_modis)

        clouds = diagnostics["clouds"]
        conv = diagnostics["convection"]
        # Use the running thermodynamic state (advanced by radiation, vdiff
        # and convection — the state the cloud microphysics actually saw) so
        # the simulated radar is thermodynamically consistent with the cloud
        # and flux fields; fall back to the step-start state when no
        # upstream term seeded it (Codex review on PR #562).
        thermo_run = diagnostics.get("thermo_run")
        if thermo_run is None:
            temperature = state.temperature
            specific_humidity = state.specific_humidity
        else:
            temperature = thermo_run["temperature"]
            specific_humidity = thermo_run["specific_humidity"]
        nlev, ncols = temperature.shape

        # Convective surface precip spread below the convective condensate
        # top (stopgap; see module docstring). The mask is 1 from the first
        # level (from the top) with convective condensate down to the
        # surface, and empty for columns without convective condensate.
        conv_cond = conv.qc_conv + conv.qi_conv
        below_top = jnp.cumsum((conv_cond > 0.0).astype(state.temperature.dtype),
                               axis=0) > 0.0
        conv_flux = conv.precip_conv[None, :] * below_top
        frozen = temperature < c.tmelt
        fl_ccrain = jnp.where(frozen, 0.0, conv_flux)
        fl_ccsnow = jnp.where(frozen, conv_flux, 0.0)

        # Cloud effective radii (um -> m); zero means "PSD defaults" and the
        # remaining hydrometeor slots always use the PSD defaults.
        reff = jnp.zeros((nlev, ncols, jconfig.N_HYDRO),
                         dtype=temperature.dtype)
        reff = reff.at[..., jconfig.I_LSCLIQ].set(clouds.r_eff_liq * 1e-6)
        reff = reff.at[..., jconfig.I_LSCICE].set(clouds.r_eff_ice * 1e-6)

        inputs = CloudsatInputs(
            pressure=diagnostics["pressure_full"],
            temperature=temperature,
            specific_humidity=specific_humidity,
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
            t2m=temperature[-1],
            reff=reff,
        )

        # Deterministic subcolumn key from the seed and the surface-pressure
        # field: same state -> same subcolumns, different steps -> fresh
        # draws (COSP itself seeds SCOPS from surface pressure).
        ps_hash = jnp.sum(
            (state.normalized_surface_pressure * 1e5).astype(jnp.int32))
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), ps_hash)

        p_half = diagnostics["pressure_half"]

        if self.enable_modis:
            # One SCOPS draw feeds radar and imager together (COSP's own
            # wiring), so the subcolumn sampling is paid once. MODIS needs
            # the gridbox-mean 0.67 um layer optical depths of stratiform
            # and convective cloud; jcm detrains convective condensate into
            # the same qc/qi fields and carries no separate convective
            # cloud, so all of it is presented as stratiform (matching the
            # conv_frac = 0 choice above).
            dtau_s = _layer_optical_depth(clouds, p_half)
            dtau_c = jnp.zeros_like(dtau_s)
            joint = simulate_cloudsat_modis(
                inputs, self._lut, dtau_s, dtau_c, p_half, key=key,
                ncolumns=self.ncolumns, overlap=self.overlap)
            # ``ModisOutputs`` wraps optics/subcolumn/column; the column
            # stage holds the gridbox statistics AeroCom asks for.
            out, modis_out = joint.cloudsat, joint.modis.column
        else:
            out = simulate_cloudsat(inputs, self._lut, key=key,
                                    ncolumns=self.ncolumns,
                                    overlap=self.overlap)
            modis_out = None

        extra: dict = {}
        if modis_out is not None:
            # MODIS marks dark (night) and cloud-free retrievals with the
            # COSP R_UNDEF sentinel (a large negative number). Emitting it
            # would poison the time average, so it is mapped to zero: these
            # are grid-box means and the protocol Q/A is explicit that they
            # are NOT divided by cloud cover, so a cloud-free box legitimately
            # contributes zero.
            def _defined(x):
                return jnp.where(x > _R_UNDEF_THRESHOLD, x, 0.0)

            # jcosp reports the MODIS cloud fractions in percent and the
            # AeroCom/CFMIP request is a fraction, so convert here rather
            # than leaving a factor of 100 for the post-processor.
            extra.update({
                "cltmodis": _defined(modis_out.cf_total) * 0.01,
                "clwmodis": _defined(modis_out.cf_liquid) * 0.01,
                "climodis": _defined(modis_out.cf_ice) * 0.01,
                "tauwmodis": _defined(modis_out.tau_liquid),
                "tauimodis": _defined(modis_out.tau_ice),
                "reffclwmodis": _defined(modis_out.size_liquid),
                "reffclimodis": _defined(modis_out.size_ice),
                "lwpmodis": _defined(modis_out.lwp),
                "iwpmodis": _defined(modis_out.iwp),
            })

        if self.enable_calipso:
            extra.update(self._calipso(inputs, out, clouds, p_half,
                                       temperature, diagnostics))

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
            **extra,
        }

    def _calipso(self, inputs, cs, clouds, p_half, temperature, diagnostics):
        """CALIPSO layered cloud cover on the radar's SCOPS realization.

        Reuses ``cs.frac_out``/``cs.prec_frac`` so the lidar sees exactly the
        cloud field the radar saw. ``distribute_hydrometeors`` converts the
        gridbox means into the in-cloud subcolumn mixing ratios the lidar
        optics expect; the four hydrometeor slots the lidar reads are
        large-scale and convective liquid/ice.
        """
        from jcosp import config as jconfig
        from jcosp.inputs import distribute_hydrometeors
        from jcosp.lidar.diagnostics import lidar_column
        from jcosp.lidar.optics import lidar_optics
        from jcosp.lidar.simulator import lidar_subcolumn

        sub = distribute_hydrometeors(
            cs.frac_out, cs.prec_frac, inputs.pressure, inputs.temperature,
            inputs.mr_lsliq, inputs.mr_lsice, inputs.mr_ccliq, inputs.mr_ccice,
            inputs.fl_lsrain, inputs.fl_lssnow, inputs.fl_lsgrpl,
            inputs.fl_ccrain, inputs.fl_ccsnow, reff_in=inputs.reff)

        optics = lidar_optics(
            q_lsliq=sub.mr_hydro[..., jconfig.I_LSCLIQ],
            q_lsice=sub.mr_hydro[..., jconfig.I_LSCICE],
            q_cvliq=sub.mr_hydro[..., jconfig.I_CVCLIQ],
            q_cvice=sub.mr_hydro[..., jconfig.I_CVCICE],
            # NOTE the convention differs from the radar path above: the
            # radar reads reff == 0 as "use the PSD defaults", whereas
            # ``lidar_optics`` reads radius <= 0 as "this class is absent
            # from the volume" and returns no particle backscatter. So a
            # configuration that leaves the effective radii unset reports
            # ZERO lidar cloud cover rather than falling back to defaults.
            ls_radliq=clouds.r_eff_liq * 1e-6,
            ls_radice=clouds.r_eff_ice * 1e-6,
            cv_radliq=jnp.zeros_like(clouds.r_eff_liq),
            cv_radice=jnp.zeros_like(clouds.r_eff_ice),
            pressure=inputs.pressure, pressure_half=p_half,
            temperature=inputs.temperature)

        sc = lidar_subcolumn(
            optics.beta_mol, optics.tau_mol, optics.betatot, optics.tautot,
            optics.betatot_ice, optics.tautot_ice,
            optics.betatot_liq, optics.tautot_liq)

        col = lidar_column(
            sc.pnorm, sc.pmol, sc.pnorm_perp,
            # COSP passes the LOWER interface of each layer as the category
            # pressure (phalf(2:nlev+1)); p_half is TOA-first with nlev+1
            # entries, so that is everything below the model top.
            p_half[1:], inputs.temperature,
            # zhalf is each layer's BOTTOM interface, (nlev,) — the same
            # slice the radar inputs take from the (nlev+1) half-level
            # array, not the full interface profile.
            diagnostics["height_full"], diagnostics["height_half"][1:])

        # cldlayer is (4, *batch) = low / mid / high / total, in percent.
        low, mid, high, total = (col.cldlayer[i] * 0.01 for i in range(4))
        return {"cllcalipso": low, "clmcalipso": mid,
                "clhcalipso": high, "cltcalipso": total}
