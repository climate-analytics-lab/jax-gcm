"""Turn a source-grid emission file into a model-grid emissions-file.

Bridges an arbitrary emission product to the runtime **emissions-file contract**
(``emis_<sector>_<species>`` bulk surface mass fluxes [kg/m²/s] on the model
grid; see ``.claude/aerosol_emissions_plan.md``). The work is:

1. read the requested source variables (each typically in its own file),
2. convert units to kg/m²/s,
3. conservatively regrid onto the model grid (:mod:`.regrid`), and
4. assemble an :class:`xarray.Dataset` of contract variables ready to write and
   load via :func:`jcm.forcing.read_anthropogenic_emissions`.

The source→contract correspondence is data, not code: a list of :class:`Channel`
records. Two concrete mappings are shipped — :func:`cesm_cmip_anthro` and
:func:`cesm_bb4cmip7`, for the CESM CMIP7 CEDS / biomass-burning files in
``inputdata/atm/cam/chem/emis/cmip7`` — but any product can be described the same
way (this is what keeps the pipeline grid- and product-agnostic).

**Note on the CESM CMIP adapter's approximations** (documented because they are
load-bearing, per the issue): the CESM *bulk surface* files (``SO2-em-anthro``,
``bc_a4-em-anthro``, ``pom_a4-em-anthro``) are summed over all 8 CEDS activity
sectors, so they carry **no per-super-sector split** — this adapter assigns them
all to ``surface_combustion``. Recovering the elevated/shipping injection split
(and the full, pre-primary-split SO₂) needs the raw CEDS *sectored* product
rather than the MAM4-speciated CESM files — that is exactly the work deferred to
the self-hosted-mirror follow-up issue. The differentiable per-super-sector
injection still applies at runtime to whatever channels a richer source
populates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

import jcm.constants as _const
from jcm.data.emissions.regrid import build_regridder, model_grid
from jcm.physics.aerosol.jam.emissions.sectors import OM_OC_RATIO

_AVOGADRO = _const.physical_constants.avogadro  # molec / mol


def molec_flux_to_mass_flux(molar_mass_g_per_mol: float) -> float:
    """Factor converting ``molecules cm⁻² s⁻¹`` → ``kg m⁻² s⁻¹``.

    ``(MW/N_A) g/molec · 1e-3 kg/g · 1e4 cm²/m² = MW·10/N_A``.
    """
    return molar_mass_g_per_mol * 10.0 / _AVOGADRO


@dataclass(frozen=True)
class Channel:
    """One source variable mapped onto a contract variable.

    Attributes:
        contract_var: target name, ``emis_<sector>_<species>``.
        source: path/URL of the source file (resolved via the downloader).
        var: variable name within that file (CESM emission files use ``emiss``).
        scale: extra multiplicative factor applied *after* unit conversion — e.g.
            ``1/OM_OC`` to express an organic-matter source as the OC the term
            re-inflates by OM:OC (so net POA mass is preserved).
        molar_mass: g/mol for the molec→mass conversion; ``None`` reads the
            source variable's ``molecular_weight`` attribute. Set to ``0``/falsy
            via ``mass_units=True`` to skip conversion for already-mass sources.
        mass_units: if ``True`` the source is already a mass flux (kg/m²/s) and
            only ``scale`` is applied.

    """

    contract_var: str
    source: str
    var: str = "emiss"
    scale: float = 1.0
    molar_mass: float | None = None
    mass_units: bool = False


#: CESM CMIP7 CEDS bulk-surface anthropogenic adapter (see module docstring).
#: ``{species}`` files live under e.g.
#: ``…/emis/cmip7/<grid>/CEDS-CMIP-2025-04-18_20251030/<species>-em-anthro_…nc``.
#: ``oc`` is scaled by ``1/OM_OC`` because the CESM ``pom_a4`` field is organic
#: *matter*; the term re-applies OM:OC, so net emitted POA equals ``pom_a4``.
def cesm_cmip_anthro(directory: str, *, suffix: str =
                     "_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_"
                     "175001-202312_c20251030.nc") -> tuple[Channel, ...]:
    """Build the CESM CMIP7 CEDS channel list for a given grid directory."""
    import os

    def path(stem):
        return os.path.join(directory, stem + suffix)

    return (
        Channel("emis_surface_combustion_so2", path("SO2-em-anthro"), molar_mass=64.0),
        Channel("emis_surface_combustion_bc", path("bc_a4-em-anthro"), molar_mass=12.0),
        Channel("emis_surface_combustion_oc", path("pom_a4-em-anthro"),
                molar_mass=12.0, scale=1.0 / OM_OC_RATIO),
    )


def cesm_bb4cmip7(directory: str, *, suffix: str =
                  "_smoothed_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_"
                  "gn_175001-202112_c20251102.nc") -> tuple[Channel, ...]:
    """CESM CMIP7 open-biomass-burning (DRES/van Marle BB4CMIP7) channel list.

    Same bulk-surface MAM4 layout as :func:`cesm_cmip_anthro`, mapped onto the
    ``biomass_burning`` super-sector so it picks up the deep FIRE injection
    profile. ``oc`` is scaled by ``1/OM_OC`` (the ``pom_a4`` field is organic
    *matter*; the term re-applies OM:OC).
    """
    import os

    def path(stem):
        return os.path.join(directory, stem + suffix)

    return (
        Channel("emis_biomass_burning_so2", path("SO2"), molar_mass=64.0),
        Channel("emis_biomass_burning_bc", path("bc_a4"), molar_mass=12.0),
        Channel("emis_biomass_burning_oc", path("pom_a4"),
                molar_mass=12.0, scale=1.0 / OM_OC_RATIO),
    )


def prepare_emissions(
    channels: tuple[Channel, ...],
    coords,
    *,
    time_index=None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    area_name: str = "area",
    known_hash: str | None = None,
) -> xr.Dataset:
    """Assemble a model-grid emissions Dataset from source ``channels``.

    Args:
        channels: source→contract mapping (e.g. ``cesm_cmip_anthro(dir)``).
        coords: the model ``CoordinateSystem`` to regrid onto.
        time_index: optional ``isel(time=…)`` selector (int, slice or list) to
            subset the (often multi-century) source time axis before regridding.
        lon_name, lat_name, area_name: source coordinate/area variable names.
        known_hash: optional sha256 forwarded to the downloader for remote files.

    Returns:
        Dataset with dims ``(time, lon, lat)`` (lon/lat the model grid in
        degrees), one ``emis_<sector>_<species>`` variable per channel, in
        kg/m²/s. Missing contract channels are simply absent (the term treats
        them as zero).

    """
    from jcm.data.emissions.downloader import fetch

    dst_lon, dst_lat, _ = model_grid(coords)
    nlon, nlat = coords.horizontal.nodal_shape

    out = xr.Dataset()
    time_coord = None
    regridder_cache: dict[tuple, object] = {}

    for ch in channels:
        ds = xr.open_dataset(fetch(ch.source, known_hash=known_hash),
                             decode_times=False)
        da = ds[ch.var]
        if time_index is not None and "time" in da.dims:
            da = da.isel(time=time_index)

        # Conservative regridder, reused across channels that share a source
        # grid (keyed by a cheap grid signature so the operator is built once).
        slon = ds[lon_name].values
        slat = ds[lat_name].values
        gkey = (slon.shape, float(slon[0]), float(slat[0]), float(slat[-1]))
        regridder = regridder_cache.get(gkey)
        if regridder is None:
            regridder = build_regridder(slon, slat, ds[area_name].values,
                                        dst_lon, dst_lat)
            regridder_cache[gkey] = regridder

        values = np.asarray(da.values, dtype=np.float64)  # (..., n_source)
        if not ch.mass_units:
            mw = ch.molar_mass
            if mw is None:
                mw = float(da.attrs["molecular_weight"])
            values = values * molec_flux_to_mass_flux(mw)
        values = values * ch.scale

        gridded = regridder(values)  # (..., nlon, nlat)
        dims = (["time", "lon", "lat"] if da.ndim > 1 else ["lon", "lat"])
        out[ch.contract_var] = (dims, gridded)

        if "time" in da.dims and time_coord is None:
            time_coord = ds["time"].isel(
                time=time_index) if time_index is not None else ds["time"]

    out = out.assign_coords(
        lon=("lon", np.rad2deg(dst_lon)),
        lat=("lat", np.rad2deg(dst_lat)),
    )
    if time_coord is not None:
        out = out.assign_coords(time=("time", np.atleast_1d(time_coord.values)))
        for k, v in time_coord.attrs.items():
            out["time"].attrs[k] = v
    out.attrs["title"] = (
        "jax-gcm prescribed anthropogenic emissions (bulk per-super-sector "
        "surface flux, kg/m2/s) — see .claude/aerosol_emissions_plan.md"
    )
    for v in out.data_vars:
        out[v].attrs["units"] = "kg m-2 s-1"
    return out


@dataclass(frozen=True)
class SpeciatedChannel:
    """One already-speciated source file mapped onto a JAM tracer.

    Attributes:
        tracer: target JAM tracer key (e.g. ``m_so4_acc``, ``n_acc``, ``g_so2``)
            — the ``aero_emis_<tracer>`` variable this contributes to. Several
            channels may target the same tracer (they are summed).
        source: path/URL of the source file.
        var: variable name within the file (CESM emission files use ``emiss``).
        elevated: if ``True`` the file is a 3-D ``molecules cm⁻³ s⁻¹`` volume
            source with an ``altitude`` dimension (CAM ``mo_extfrc`` style); it is
            column-integrated over altitude to a surface flux. At GCM vertical
            resolution the CEDS elevated layer (≤ ~400 m) sits within the lowest
            model layer(s), so this is near-exact; the
            :class:`PreSpeciatedEmissions` 3-D path remains for genuinely
            high-resolution elevated injection.

    Number emissions need no special casing: CESM encodes them so the same
    ``molecular_weight``-based molec→mass conversion yields ``#/m²/s`` directly.

    """

    tracer: str
    source: str
    var: str = "emiss"
    elevated: bool = False


def cesm_mam4_speciated(directory: str, *, suffix: str =
                        "-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-"
                        "04-18_gn_175001-202312_c20251030.nc"
                        ) -> tuple[SpeciatedChannel, ...]:
    """CESM CMIP7 CEDS *already-speciated* MAM4 anthropogenic channel list.

    Maps the CESM modal/sector files directly onto JAM MAM4 tracers (a1→accum,
    a2→Aitken, a4→primary-carbon; ``SO2``→gas; ``num_*``→number), summing the
    sector-split sulfate files (and the elevated energy-sector ``ene_vertical``)
    into their mode. This reproduces CAM6's prescribed emissions on the model
    grid — the validation counterpart to the differentiable bulk path.
    """
    import os

    def p(stem):
        return os.path.join(directory, stem + suffix)

    return (
        # Accumulation-mode sulfate: surface (ag+ship+slv) + elevated energy.
        SpeciatedChannel("m_so4_acc", p("so4_a1_ag_ship_slv")),
        SpeciatedChannel("m_so4_acc", p("so4_a1_ene_vertical"), elevated=True),
        SpeciatedChannel("n_acc", p("num_so4_a1_ag")),
        SpeciatedChannel("n_acc", p("num_so4_a1_ship_slv")),
        SpeciatedChannel("n_acc", p("num_so4_a1_ene_vertical"), elevated=True),
        # Aitken-mode sulfate (residential + transport).
        SpeciatedChannel("m_so4_ait", p("so4_a2_res_trs")),
        SpeciatedChannel("n_ait", p("num_so4_a2_res_trs")),
        # Primary-carbon mode: BC + POA (+ their number).
        SpeciatedChannel("m_bc_pcm", p("bc_a4")),
        SpeciatedChannel("m_poa_pcm", p("pom_a4")),
        SpeciatedChannel("n_pcm", p("num_bc_a4")),
        SpeciatedChannel("n_pcm", p("num_pom_a4")),
        # SO2 gas (already net of the 2.5 % primary-sulfate diversion).
        SpeciatedChannel("g_so2", p("SO2")),
    )


def prepare_speciated_emissions(
    channels: tuple[SpeciatedChannel, ...],
    coords,
    *,
    time_index=None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    area_name: str = "area",
    known_hash: str | None = None,
) -> xr.Dataset:
    """Assemble a model-grid *pre-speciated* emissions Dataset (``aero_emis_*``).

    Sister of :func:`prepare_emissions` for the CAM6-faithful path: per-tracer
    surface fluxes (kg/m²/s for mass, #/m²/s for number) on the model grid, ready
    for :func:`jcm.forcing.read_prescribed_aerosol_emissions`. Channels targeting
    the same tracer are summed; ``elevated`` channels are column-integrated over
    their altitude axis first.
    """
    from jcm.data.emissions.downloader import fetch

    dst_lon, dst_lat, _ = model_grid(coords)
    nlon, nlat = coords.horizontal.nodal_shape

    fields: dict[str, np.ndarray] = {}
    time_coord = None
    regridder_cache: dict[tuple, object] = {}

    for ch in channels:
        ds = xr.open_dataset(fetch(ch.source, known_hash=known_hash),
                             decode_times=False)
        da = ds[ch.var]
        if time_index is not None and "time" in da.dims:
            da = da.isel(time=time_index)

        values = np.asarray(da.values, dtype=np.float64)  # (..., [alt,] n_src)
        if ch.elevated:
            # molec/cm³/s volume source → column molec/cm²/s: Σ_alt emiss·Δz.
            dz_cm = np.abs(np.diff(ds["altitude_int"].values)) * 1.0e5  # km→cm
            alt_axis = da.dims.index("altitude")
            values = np.tensordot(values, dz_cm, axes=([alt_axis], [0]))
        # Same molec→mass(/number) conversion CAM applies; reads the file's MW.
        values = values * molec_flux_to_mass_flux(float(da.attrs["molecular_weight"]))

        slon = ds[lon_name].values
        slat = ds[lat_name].values
        gkey = (slon.shape, float(slon[0]), float(slat[0]), float(slat[-1]))
        regridder = regridder_cache.get(gkey)
        if regridder is None:
            regridder = build_regridder(slon, slat, ds[area_name].values,
                                        dst_lon, dst_lat)
            regridder_cache[gkey] = regridder

        gridded = regridder(values)  # (..., nlon, nlat)
        fields[ch.tracer] = fields.get(ch.tracer, 0.0) + gridded

        if "time" in da.dims and time_coord is None:
            time_coord = (ds["time"].isel(time=time_index)
                          if time_index is not None else ds["time"])

    out = xr.Dataset()
    for tracer, arr in fields.items():
        dims = ["time", "lon", "lat"] if arr.ndim > 2 else ["lon", "lat"]
        out[f"aero_emis_{tracer}"] = (dims, arr)
    out = out.assign_coords(lon=("lon", np.rad2deg(dst_lon)),
                            lat=("lat", np.rad2deg(dst_lat)))
    if time_coord is not None:
        out = out.assign_coords(time=("time", np.atleast_1d(time_coord.values)))
    out.attrs["title"] = (
        "jax-gcm prescribed pre-speciated aerosol emissions (per-tracer surface "
        "flux; mass kg/m2/s, number #/m2/s) — CAM6/MAM4-faithful path"
    )
    return out
