"""Post-process JCM output into AeroCom phase-4 submission files.

Reads the netCDF a JCM run writes and emits **one variable per file** under
the AeroCom naming convention, with CMOR variable names and units.

    python tools/aerocom_cmor.py run_day30.nc --model JCM-v2 \
        --experiment AP4-CTRL-PD-NudClim --period 2010 --out submission/

Filename convention (aci-baseline / imo2020 form)::

    aerocom_<ModelName>_<ExperimentName>_<VariableName>_<VertCoord>_<Period>_<Freq>.nc

AP4-CTRL uses a phase-4 variant with ``aerocom4_`` and the simulation name
folded into the experiment field; pass ``--convention aerocom4`` for that.

``<ModelName>`` may not contain underscores (use ``-``) and is capped at 20
characters — enforced here rather than discovered at submission.

Vertical orientation
--------------------
JCM writes output **surface-first** (level index 0 is the surface; see
CLAUDE.md). This script does not reorder levels by default, because the
protocol specifies the *coordinate type* (``ModelLevel``/``Surface``/
``Column``/``TOA``) rather than an ordering, and models submit on their
native grid. Pass ``--flip-levels`` to write TOA-first if your AeroCom
contact asks for it, and record the choice in the submission notes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import xarray as xr

# jcm name -> (CMOR name, units, vertical coordinate type, scale, offset)
#
# ``scale``/``offset`` convert jcm units to the units AeroCom requests.
# Only variables JCM actually writes are listed; the remaining requested
# fields need model-side work (see jax-gcm#581).
NAME_MAP: dict[str, tuple[str, str, str, float, float]] = {
    # --- radiative fluxes (TOA / surface) ---
    "radiation.toa_lw_up": ("rlut", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_lw_up_clear": ("rlutcs", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_sw_up": ("rsut", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_sw_up_clear": ("rsutcs", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_sw_down": ("rsdt", "W m-2", "TOA", 1.0, 0.0),
    "radiation.surface_sw_down": ("rsds", "W m-2", "Surface", 1.0, 0.0),
    "radiation.surface_lw_down": ("rlds", "W m-2", "Surface", 1.0, 0.0),
    "radiation.surface_sw_up": ("rsus", "W m-2", "Surface", 1.0, 0.0),
    "radiation.surface_lw_up": ("rlus", "W m-2", "Surface", 1.0, 0.0),
    # --- turbulent fluxes ---
    "surface.latent_heat_flux": ("hfls", "W m-2", "Surface", 1.0, 0.0),
    "surface.sensible_heat_flux": ("hfss", "W m-2", "Surface", 1.0, 0.0),
    # --- basic state ---
    "surface_pressure": ("ps", "Pa", "Surface", 1.0, 0.0),
    "surface.surface_temperature": ("ts", "K", "Surface", 1.0, 0.0),
    "temperature": ("ta", "K", "ModelLevel", 1.0, 0.0),
    "specific_humidity": ("hus", "kg kg-1", "ModelLevel", 1.0, 0.0),
    "relative_humidity": ("hur", "%", "ModelLevel", 100.0, 0.0),
    "u_wind": ("ua", "m s-1", "ModelLevel", 1.0, 0.0),
    "v_wind": ("va", "m s-1", "ModelLevel", 1.0, 0.0),
    "pressure_full": ("pfull", "Pa", "ModelLevel", 1.0, 0.0),
    # ``height_full`` is already geopotential HEIGHT in metres. jcm also
    # writes ``geopotential`` in m2/s2 (see the dynamics units table); it
    # is mapped with the explicit 1/g conversion so a run that only saved
    # the potential still produces a correct ``zg``. When both are
    # present the height wins (dict order below, resolved in convert()).
    "geopotential": ("zg", "m", "ModelLevel", 1.0 / 9.80665, 0.0),
    "height_full": ("zg", "m", "ModelLevel", 1.0, 0.0),
    "air_density": ("rho", "kg m-3", "ModelLevel", 1.0, 0.0),
    "layer_thickness": ("dzhalf", "m", "ModelLevel", 1.0, 0.0),
    # --- clouds ---
    "radiation.total_cloud_cover": ("clt", "1", "Column", 1.0, 0.0),
    "clouds.cloud_fraction": ("cl", "1", "ModelLevel", 1.0, 0.0),
    "qc": ("clw", "kg kg-1", "ModelLevel", 1.0, 0.0),
    "qi": ("cli", "kg kg-1", "ModelLevel", 1.0, 0.0),
    # CloudData.droplet_number is zero under the 2-moment scheme, which
    # carries qnc instead; AerocomDiagnostics publishes the resolved
    # volumetric profile, so map that.
    "aerocom_cdnc3d": ("cdnc3d", "m-3", "ModelLevel", 1.0, 0.0),
    # jcm carries effective radii in microns; AeroCom asks for metres.
    "clouds.r_eff_liq": ("cdr3d", "m", "ModelLevel", 1e-6, 0.0),
    "clouds.r_eff_ice": ("icr3d", "m", "ModelLevel", 1e-6, 0.0),
    # --- precipitation ---
    "clouds.precip_rain": ("prlr", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "clouds.precip_snow": ("prls", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "convection.precip_conv": ("prc", "kg m-2 s-1", "Surface", 1.0, 0.0),
    # --- aerosol optics ---
    "aerosol.aod_total": ("od550aer", "1", "Column", 1.0, 0.0),
    # --- CFMIP satellite simulators (jax-gcm#581) ---
    # CALIPSO and MODIS cloud products. jcm already emits these under their
    # CMOR names, so the mapping is mostly identity — but without an entry
    # here ``convert`` drops them into ``skipped``, and the advertised
    # echam-jam-aci run would produce the diagnostics and then no submission
    # files for them. Cloud FRACTIONS are stored as 0-1 fractions and CMOR
    # wants percent, hence the x100; the MODIS effective radii are metres in
    # jcosp and CMOR wants metres, so those pass through.
    "cltcalipso": ("cltcalipso", "%", "Column", 100.0, 0.0),
    "cllcalipso": ("cllcalipso", "%", "Column", 100.0, 0.0),
    "clmcalipso": ("clmcalipso", "%", "Column", 100.0, 0.0),
    "clhcalipso": ("clhcalipso", "%", "Column", 100.0, 0.0),
    "cltmodis": ("cltmodis", "%", "Column", 100.0, 0.0),
    "clwmodis": ("clwmodis", "%", "Column", 100.0, 0.0),
    "climodis": ("climodis", "%", "Column", 100.0, 0.0),
    "tauwmodis": ("tauwmodis", "1", "Column", 1.0, 0.0),
    "tauimodis": ("tauimodis", "1", "Column", 1.0, 0.0),
    "reffclwmodis": ("reffclwmodis", "m", "Column", 1.0, 0.0),
    "reffclimodis": ("reffclimodis", "m", "Column", 1.0, 0.0),
    "lwpmodis": ("lwpmodis", "kg m-2", "Column", 1.0, 0.0),
    "iwpmodis": ("iwpmodis", "kg m-2", "Column", 1.0, 0.0),
    # CloudSat warm-rain occurrence: the ACI calibration target of
    # Muelmenstaedt et al. (2020). Not a CMIP variable, kept under its own
    # name so the submission still carries it.
    "cosp_warm_rain": ("cospwarmrain", "1", "Column", 1.0, 0.0),
    # --- spectral aerosol optics (jax-gcm#584) ---
    # These come from the diagnostic Mie pass at the OBSERVATION
    # wavelengths, so ``od550aer`` here is at exactly 550 nm. It is listed
    # after ``aerosol.aod_total`` (the nearest radiation band centre)
    # deliberately: later entries win, so the exact-wavelength field
    # supersedes the band-centre one when a run has both.
    "od550aer": ("od550aer", "1", "Column", 1.0, 0.0),
    "abs550aer": ("abs550aer", "1", "Column", 1.0, 0.0),
    "od355aer": ("od355aer", "1", "Column", 1.0, 0.0),
    "od440aer": ("od440aer", "1", "Column", 1.0, 0.0),
    "od670aer": ("od670aer", "1", "Column", 1.0, 0.0),
    "od865aer": ("od865aer", "1", "Column", 1.0, 0.0),
    "ssa440aer": ("ssa440aer", "1", "Column", 1.0, 0.0),
    "ang4487aer": ("ang4487aer", "1", "Column", 1.0, 0.0),
    "aerindex": ("aerindex", "1", "Column", 1.0, 0.0),
    "ec355aer": ("ec355aer", "m-1", "ModelLevel", 1.0, 0.0),
    "aerosol.angstrom": ("angstrm", "1", "Column", 1.0, 0.0),
    # --- microphysical process rates (jax-gcm#585), column-integrated ---
    "autoconv": ("autoconv", "kg m-2 s-1", "Column", 1.0, 0.0),
    "accretn": ("accretn", "kg m-2 s-1", "Column", 1.0, 0.0),
    "wbf": ("wbf", "kg m-2 s-1", "Column", 1.0, 0.0),
    # --- emission fluxes, summed per species over all emitting terms ---
    "emi_so2": ("emi_so2", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_so4": ("emi_so4", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_bc": ("emi_bc", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_oc": ("emi_oc", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_poa": ("emi_poa", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_soa": ("emi_soa", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_ss": ("emi_ss", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_du": ("emi_du", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_moa": ("emi_moa", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "emi_dms": ("emi_dms", "kg m-2 s-1", "Surface", 1.0, 0.0),
    # --- boundary layer ---
    "vertical_diffusion.pbl_height": ("hdtcbl", "m", "Surface", 1.0, 0.0),
    # --- AerocomDiagnostics term output ---
    "aerocom_clt": ("clt", "1", "Column", 1.0, 0.0),
    "aerocom_ttop": ("ttop", "K", "Column", 1.0, 0.0),
    "aerocom_cdr": ("cdr", "m", "Column", 1.0, 0.0),
    "aerocom_icr": ("icr", "m", "Column", 1.0, 0.0),
    "aerocom_cdnc": ("cdnc", "m-3", "Column", 1.0, 0.0),
    "aerocom_lcc": ("lcc", "1", "Column", 1.0, 0.0),
    "aerocom_icc": ("icc", "1", "Column", 1.0, 0.0),
    "aerocom_cod": ("cod", "1", "Column", 1.0, 0.0),
    "aerocom_codliq": ("codliq", "1", "Column", 1.0, 0.0),
    "aerocom_codice": ("codice", "1", "Column", 1.0, 0.0),
    "aerocom_lwp": ("lwp", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_iwp": ("iwp", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_cllvi": ("cllvi", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_clivi": ("clivi", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_prw": ("prw", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_cdnum": ("cdnum", "m-2", "Column", 1.0, 0.0),
    "aerocom_icnum": ("icnum", "m-2", "Column", 1.0, 0.0),
    "aerocom_albedo": ("albedo", "1", "Column", 1.0, 0.0),
    "aerocom_lts": ("lts", "K", "Surface", 1.0, 0.0),
    "aerocom_u200": ("u200", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_v200": ("v200", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_u700": ("u700", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_v700": ("v700", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_N70": ("N70", "m-3", "ModelLevel", 1.0, 0.0),
    "aerocom_N100": ("N100", "m-3", "ModelLevel", 1.0, 0.0),
    "aerocom_PM1": ("PM1", "kg m-3", "ModelLevel", 1.0, 0.0),
    "aerocom_PM10": ("PM10", "kg m-3", "ModelLevel", 1.0, 0.0),
}

# Column burdens arrive pre-summed per species from AerocomDiagnostics
# (interstitial + cloud-borne + gas), so the writer only renames them.
BURDEN_SPECIES = ("so4", "bc", "oc", "poa", "soa", "ss", "du", "moa",
                  "dms", "so2", "h2so4", "soag")


# CF standard names for the subset where the mapping is unambiguous. The
# short CMOR id is NOT a CF standard name, so anything absent here simply
# gets no ``standard_name`` attribute (which is optional) rather than a
# guessed one that a CF checker would reject.
CF_STANDARD_NAMES = {
    "ta": "air_temperature",
    "ps": "surface_air_pressure",
    "ts": "surface_temperature",
    "hus": "specific_humidity",
    "hur": "relative_humidity",
    "ua": "eastward_wind",
    "va": "northward_wind",
    "zg": "geopotential_height",
    "rho": "air_density",
    "clt": "cloud_area_fraction",
    "cl": "cloud_area_fraction_in_atmosphere_layer",
    "clw": "mass_fraction_of_cloud_liquid_water_in_air",
    "cli": "mass_fraction_of_cloud_ice_in_air",
    "lwp": "atmosphere_mass_content_of_cloud_liquid_water",
    "iwp": "atmosphere_mass_content_of_cloud_ice",
    "prw": "atmosphere_mass_content_of_water_vapor",
    "rlut": "toa_outgoing_longwave_flux",
    "rlutcs": "toa_outgoing_longwave_flux_assuming_clear_sky",
    "rsut": "toa_outgoing_shortwave_flux",
    "rsutcs": "toa_outgoing_shortwave_flux_assuming_clear_sky",
    "rsdt": "toa_incoming_shortwave_flux",
    "rsds": "surface_downwelling_shortwave_flux_in_air",
    "rsus": "surface_upwelling_shortwave_flux_in_air",
    "rlds": "surface_downwelling_longwave_flux_in_air",
    "rlus": "surface_upwelling_longwave_flux_in_air",
    "hfls": "surface_upward_latent_heat_flux",
    "hfss": "surface_upward_sensible_heat_flux",
    "cltcalipso": "cloud_area_fraction",
    "cllcalipso": "cloud_area_fraction_in_atmosphere_layer",
    "clmcalipso": "cloud_area_fraction_in_atmosphere_layer",
    "clhcalipso": "cloud_area_fraction_in_atmosphere_layer",
    "cltmodis": "cloud_area_fraction",
    "clwmodis": "liquid_water_cloud_area_fraction",
    "climodis": "ice_cloud_area_fraction",
    "tauwmodis": "atmosphere_optical_thickness_due_to_cloud",
    "tauimodis": "atmosphere_optical_thickness_due_to_cloud",
    "reffclwmodis": "effective_radius_of_cloud_liquid_water_particles",
    "reffclimodis": "effective_radius_of_cloud_ice_particles",
    "lwpmodis": "atmosphere_mass_content_of_cloud_liquid_water",
    "iwpmodis": "atmosphere_mass_content_of_cloud_ice",
    "od550aer": "atmosphere_optical_thickness_due_to_ambient_aerosol_particles",
    "abs550aer": "atmosphere_absorption_optical_thickness_due_to_ambient_aerosol_particles",
    "od550so4": "atmosphere_optical_thickness_due_to_sulfate_ambient_aerosol_particles",
    "od550bc": "atmosphere_optical_thickness_due_to_black_carbon_ambient_aerosol_particles",
    "od550oa": "atmosphere_optical_thickness_due_to_particulate_organic_matter_ambient_aerosol_particles",
    "od550ss": "atmosphere_optical_thickness_due_to_sea_salt_ambient_aerosol_particles",
    "od550dust": "atmosphere_optical_thickness_due_to_dust_ambient_aerosol_particles",
    "od550aerh2o": "atmosphere_optical_thickness_due_to_water_in_ambient_aerosol_particles",
    "ec355aer": "volume_extinction_coefficient_in_air_due_to_ambient_aerosol_particles",
    "ang4487aer": "angstrom_exponent_of_ambient_aerosol_in_air",
}

VALID_VERT = ("Surface", "TOA", "Column", "ModelLevel")


def _check_model_name(name: str) -> str:
    if "_" in name:
        raise SystemExit(
            f"--model {name!r}: underscores are not allowed in <ModelName> "
            "(AeroCom convention); use '-' instead.")
    if len(name) > 20:
        raise SystemExit(
            f"--model {name!r}: <ModelName> is capped at 20 characters "
            f"(got {len(name)}).")
    return name


def _filename(convention, model, experiment, var, vert, period, freq):
    if convention == "aerocom4":
        # phase-4 form: aerocom4_<Model>_<Exp>-<Sim>_<Var>_<Vert>_<Year>_<Freq>.nc
        return f"aerocom4_{model}_{experiment}_{var}_{vert}_{period}_{freq}.nc"
    return f"aerocom_{model}_{experiment}_{var}_{vert}_{period}_{freq}.nc"


def _collect_burdens(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Rename the per-species column burdens to their CMOR names."""
    out: dict[str, xr.DataArray] = {}
    for spec in BURDEN_SPECIES:
        src = f"aerocom_burden_{spec}"
        if src in ds.data_vars:
            out[f"burden_{spec}"] = ds[src]
    return out


# jcm species -> AeroCom component suffix. The three organic species
# (primary, secondary, marine) are reported as one ``oa`` component, which
# is what the protocol asks for; the raw per-species fields stay in the
# model output if a finer split is wanted.
_OPTICS_COMPONENT = {
    "so4": "so4", "bc": "bc", "du": "dust", "ss": "ss", "wat": "aerh2o",
    "poa": "oa", "soa": "oa", "moa": "oa",
}


def _collect_optics(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Group the per-species optics into AeroCom components (jax-gcm#584).

    Several jcm species map to one AeroCom component, so contributions are
    SUMMED rather than overwritten. Per-mode fields are passed through
    under their own names — they are a JAM extra, not part of the protocol,
    but they are what makes a size-resolved AOD evaluation possible.
    """
    out: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        for prefix in ("od550_", "abs550_"):
            if not str(var).startswith(prefix):
                continue
            rest = str(var)[len(prefix):]
            if rest.startswith("mode_"):
                out[str(var)] = ds[var]
                break
            comp = _OPTICS_COMPONENT.get(rest)
            if comp is None:
                break
            name = f"{prefix[:-1]}{comp}"
            out[name] = out[name] + ds[var] if name in out else ds[var]
            break
    return out


def convert(
    ds: xr.Dataset,
    model: str,
    experiment: str,
    period: str,
    freq: str,
    outdir: pathlib.Path,
    convention: str = "aerocom",
    flip_levels: bool = False,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Write one file per mapped variable; return (written, skipped)."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    candidates: dict[str, tuple[xr.DataArray, str, str]] = {}
    for src, (cmor, units, vert, scale, offset) in NAME_MAP.items():
        if src not in ds.data_vars:
            continue
        # Later entries win, so a directly-available field (e.g.
        # height_full for zg) overrides a converted one (geopotential/g).
        da = ds[src] * scale + offset
        candidates[cmor] = (da, units, vert)
    for name, da in _collect_burdens(ds).items():
        candidates[name] = (da, "kg m-2", "Column")
    for name, da in _collect_optics(ds).items():
        candidates[name] = (da, "1", "Column")

    for cmor, (da, units, vert) in sorted(candidates.items()):
        assert vert in VALID_VERT, vert
        if flip_levels and "level" in da.dims:
            da = da.isel(level=slice(None, None, -1))
        da = da.rename(cmor)
        # ``standard_name`` must be a real CF standard name, not the short
        # CMOR id (``ta`` is not a standard name; ``air_temperature`` is), or
        # a CF checker rejects the file. Omitted where we do not have a
        # verified mapping rather than guessed.
        attrs = {"units": units, "long_name": cmor,
                 "comment": ("Produced by tools/aerocom_cmor.py from JCM "
                             "output; see jax-gcm#581 for coverage.")}
        if cmor in CF_STANDARD_NAMES:
            attrs["standard_name"] = CF_STANDARD_NAMES[cmor]
        da.attrs.update(attrs)
        fname = _filename(convention, model, experiment, cmor, vert, period, freq)
        if not dry_run:
            out = xr.Dataset({cmor: da})
            out.attrs.update(
                model_id=model, experiment_id=experiment, frequency=freq,
                vertical_coordinate_type=vert,
                level_order=("TOA-first" if flip_levels else "surface-first"),
            )
            out.to_netcdf(outdir / fname)
        written.append(fname)

    mapped_srcs = {s for s in NAME_MAP if s in ds.data_vars}
    skipped = sorted(set(ds.data_vars) - mapped_srcs
                     - {f"aerocom_burden_{sp}" for sp in BURDEN_SPECIES})
    return written, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=pathlib.Path, help="JCM output netCDF")
    ap.add_argument("--model", required=True,
                    help="<ModelName>: no underscores, <=20 chars")
    ap.add_argument("--experiment", required=True,
                    help="e.g. all_2000, CTRL, 20shp, AP4-CTRL-PD-NudClim")
    ap.add_argument("--period", required=True, help="e.g. 2010")
    ap.add_argument("--freq", default="monthly",
                    choices=["timeinvariant", "hourly", "3hourly", "daily", "monthly"])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("aerocom_submit"))
    ap.add_argument("--convention", default="aerocom", choices=["aerocom", "aerocom4"])
    ap.add_argument("--flip-levels", action="store_true",
                    help="write TOA-first instead of JCM's native surface-first")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing")
    args = ap.parse_args(argv)

    _check_model_name(args.model)
    ds = xr.open_dataset(args.input, decode_times=False)
    written, skipped = convert(
        ds, args.model, args.experiment, args.period, args.freq, args.out,
        convention=args.convention, flip_levels=args.flip_levels,
        dry_run=args.dry_run)

    print(f"{'would write' if args.dry_run else 'wrote'} {len(written)} "
          f"variable file(s) to {args.out}")
    for f in written:
        print("  ", f)
    if skipped:
        print(f"\n{len(skipped)} JCM variable(s) had no AeroCom mapping "
              "(expected — most are internal diagnostics):")
        print("   " + ", ".join(skipped[:15])
              + (" ..." if len(skipped) > 15 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
