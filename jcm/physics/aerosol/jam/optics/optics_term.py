"""``JamOpticsTerm`` — online aerosol optics from the modal population.

Computes per-radiation-band aerosol optical depth, single-scattering albedo
and asymmetry parameter from the JAM population and writes them into the
``aerosol`` diagnostic, giving online aerosol a **direct radiative effect**
(#495). For each mode and band: form the volume-mixed complex refractive index
(dry species + aerosol water), the size parameter ``x = 2π r_wet/λ``, look up
the Mie efficiencies, and accumulate the extinction across modes;
single-scattering albedo and asymmetry are extinction-/scattering-weighted.

The expensive Mie evaluation is paid once at construction (``mie_lut``); the
per-step path is a differentiable table interpolation. The per-band SW/LW
optics are written into the shared ``aerosol`` struct (seeded all-zero each
step by ``AerosolCarrySeeder``) and consumed directly by RRTMGP. For the grey
two-stream scheme, which reads a single broadband 550 nm profile rather than
per-band optics, this term also writes ``aod_profile``/``ssa_profile``/
``asy_profile`` (the SW band nearest 550 nm) and a band-ratio ``angstrom`` so
grey+JAM keeps a direct effect (#640).
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.optics.mie_lut import default_mie_lut, interp_mie
from jcm.physics.aerosol.jam.optics.refractive_index import refractive_index_at
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

_TINY = 1.0e-30

# AeroCom diagnostic wavelengths [nm]. 550 is the reference observable;
# 440/670/865 give the Angstrom exponent and the 440 nm single-scattering
# albedo AERONET reports; 355 is the lidar (ATLID/EarthCARE) wavelength.
# ``refractive_index_at`` interpolates in log10(lambda), so these need not
# coincide with any radiation band.
_DIAG_WAVELENGTHS_NM: tuple[float, ...] = (355.0, 440.0, 550.0, 670.0, 865.0)
_I355, _I440, _I550, _I670, _I865 = 0, 1, 2, 3, 4

# Gauss–Hermite nodes (8) for the lognormal quadrature in ``_band_optics``.
_GH_NODES, _GH_WEIGHTS = (
    tuple(float(v) for v in arr) for arr in np.polynomial.hermite.hermgauss(8)
)
_FOUR_THIRDS_PI = 4.0 / 3.0 * math.pi


#: Reference wavelength for the column AOD diagnostic — 550 nm is the
#: standard visible wavelength for satellite (MODIS/MISR) and AERONET AOD,
#: so it is the natural observable to validate the scheme against.
_AOD_REF_NM = 550.0

#: No aerosol radiative effect above this pressure [Pa] — see the gate in
#: ``_compute_fields``: absorbed flux over the ~1 Pa lid layers gives
#: effectively unbounded heating once real absorbers mix up there, and the
#: aerosol mass above ~2 hPa is radiatively negligible.
_AER_RAD_PMIN = 200.0

#: Cap on the per-layer, per-band aerosol optical depth. A single model
#: layer with tau > 1 is already an extreme plume (whole-column dust-storm
#: AODs are ~5); the winter year-run blow-up (day 212, a Philippine monsoon
#: column at 496 K) showed the aerosol->SW-heating->convection feedback has
#: no other brake once transport ringing plus accumulation push a layer's
#: extinction into pathological territory. The aerosol-optics analogue of
#: ``_MAX_IN_CLOUD_CONDENSATE`` in the RRTMGP wrapper: genuine plumes are
#: untouched, only the runaway tail is clipped.
_MAX_LAYER_TAU = 1.0


@dataclasses.dataclass(frozen=True)
class _OpticsCache:
    """Per-band centers and refractive indices (static; nnx treats as a leaf)."""

    sw_nm: np.ndarray
    lw_nm: np.ndarray
    ri_sw: dict          # species -> (n[bands], k[bands])
    ri_lw: dict
    aod_band_idx: int    # SW band index whose centre is closest to 550 nm
    aod_band_nm: float   # that band's actual centre wavelength [nm]
    ang_band_idx: int    # SW band for the 550/865 Angstrom pair (grey #640)
    ang_band_nm: float   # that band's actual centre wavelength [nm]
    ri_diag: dict | None = None   # RI at _DIAG_WAVELENGTHS_NM (#584), or None


class JamOpticsTerm(PhysicsTerm):
    """Online aerosol SW+LW optics written into the ``aerosol`` diagnostic."""

    name: ClassVar[str] = "jam_optics"
    category: ClassVar[str] = "aerosol_optics"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "aerosol", "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = ("aerosol", "_jam_optics")
    # JAM's column optics diagnostics publish under the explicit ``jam_optics.*``
    # namespace (#640): the ``_jam_optics`` carry dict flattens to
    # ``jam_optics.<field>`` on output. CF/AeroCom metadata for the fields that
    # survive (the per-band arrays are dropped by ``_EXCLUDED_OUTPUT_KEYS``).
    output_attrs: ClassVar[dict[str, dict[str, str]]] = {
        "jam_optics.aod_550": {
            "units": "1",
            "standard_name": (
                "atmosphere_optical_thickness_due_to_ambient_aerosol_particles"
            ),
            "long_name": (
                "JAM total-column aerosol optical depth at the SW band nearest "
                "550 nm (band-centre approximation; the Mie-based od550aer is "
                "the aerocom_optics diagnostic)"
            ),
        },
        "jam_optics.aod_profile": {
            "units": "1",
            "long_name": "JAM 550 nm aerosol optical depth per layer",
        },
        "jam_optics.ssa_profile": {
            "units": "1",
            "long_name": "JAM 550 nm single-scattering albedo per layer",
        },
        "jam_optics.asy_profile": {
            "units": "1",
            "long_name": "JAM 550 nm asymmetry parameter per layer",
        },
        "jam_optics.angstrom": {
            "units": "1",
            "long_name": "JAM column Angstrom exponent (550/865 nm band ratio)",
        },
    }

    def __init__(self, *, spec: ModalAerosolSpec | None = None,
                 optics_diagnostics: bool = False):
        """Build the Mie lookup table and hold the population.

        ``optics_diagnostics`` enables the AeroCom per-species / per-mode /
        spectral optics diagnostics (jax-gcm#584). It is off by default
        because it adds a second Mie pass over
        ``_DIAG_WAVELENGTHS_NM``; enabled, it rides the same radiation gate
        as the radiative optics, so the incremental cost is
        ``len(_DIAG_WAVELENGTHS_NM) / n_sw_band`` of the (already gated)
        aerosol optics rather than a per-step cost.
        """
        self._spec = spec or MAM4_SPEC
        self._lut = default_mie_lut()
        self._cache = None   # set by cache_band_config
        self._radiation_interval_s: float | None = None
        self._optics_diagnostics = bool(optics_diagnostics)

    def optics_diagnostic_keys(self) -> tuple[str, ...]:
        """Diagnostics keys the optics diagnostic publishes (static).

        Derived from the (static) mode/species spec, so the key set cannot
        vary between the initial scan-carry probe and the real steps.
        """
        if not self._optics_diagnostics:
            return ()
        species = sorted({sp for m in self._spec.modes for sp in m.species}) + ["wat"]
        keys = ["od550aer", "abs550aer", "od355aer", "od440aer", "od670aer",
                "od865aer", "ssa440aer", "ang4487aer", "ang550865aer",
                "aerindex", "ec355aer"]
        keys += [f"od550_{sp}" for sp in species]
        keys += [f"abs550_{sp}" for sp in species]
        keys += [f"od550_mode_{m.short}" for m in self._spec.modes]
        keys += [f"abs550_mode_{m.short}" for m in self._spec.modes]
        return tuple(keys)

    def configure_radiation_gate(self, interval_s) -> None:
        """Recompute band optics only on radiation-compute steps.

        The per-band optics are consumed exclusively by the radiation term,
        which replays cached heating rates between its ``radiation_interval``
        compute steps (default 2 h = every 8th step at dt = 900 s) — so the
        30-band × 4-mode Mie evaluation on the intermediate steps is
        discarded work (~8× of the second-largest JAM cost). With the gate
        configured, those steps replay the previous compute's per-band
        fields from the ``_jam_optics`` carry slot instead, using the
        *same* ``radiation.step`` counter the radiation gate reads (this
        term runs earlier in the chain, so both see the identical
        pre-increment value and agree within a step). ``interval_s <= 0``
        disables the gate (compute every step — the pre-gating behaviour).
        The column AOD-550 diagnostic then also updates at the radiation
        cadence.
        """
        v = float(interval_s)
        self._radiation_interval_s = v if v > 0 else None

    def cache_band_config(self, band_config) -> None:
        """Precompute band centers and per-species refractive indices."""
        sw_nm = np.asarray(band_config.sw_band_centers_nm, np.float64)
        lw_nm = np.asarray(band_config.lw_band_centers_nm, np.float64)
        species = {sp for m in self._spec.modes for sp in m.species} | {"h2o"}
        ri_sw, ri_lw = {}, {}
        for sp in species:
            n_sw, k_sw = refractive_index_at(sp, jnp.asarray(sw_nm))
            n_lw, k_lw = refractive_index_at(sp, jnp.asarray(lw_nm))
            ri_sw[sp] = (np.asarray(n_sw), np.asarray(k_sw))
            ri_lw[sp] = (np.asarray(n_lw), np.asarray(k_lw))
        # SW band whose centre is closest to 550 nm — the band the column AOD
        # diagnostic reports (exact 550 nm with RRTMGP's banding; the single
        # broadband centre for grey radiation).
        if sw_nm.size:
            aod_idx = int(np.argmin(np.abs(sw_nm - _AOD_REF_NM)))
            aod_nm = float(sw_nm[aod_idx])
        else:
            aod_idx, aod_nm = 0, float("nan")
        # Second SW band for the column Angstrom exponent fed to the grey
        # two-stream scheme (#640): the band nearest 865 nm, giving the
        # ECHAM-HAM 550/865 pair. With a single broadband SW band (grey's own
        # default RadiationBandConfig) there is no ratio to form, so fall back
        # to the 550 band index and flag it degenerate for the default below.
        if sw_nm.size >= 2:
            ang_idx = int(np.argmin(np.abs(sw_nm - 865.0)))
            if ang_idx == aod_idx:
                ang_idx = int(np.argmax(np.abs(sw_nm - aod_nm)))
            ang_nm = float(sw_nm[ang_idx])
        else:
            ang_idx, ang_nm = aod_idx, aod_nm
        ri_diag = None
        if self._optics_diagnostics:
            diag_nm = jnp.asarray(_DIAG_WAVELENGTHS_NM)
            ri_diag = {}
            for sp in species:
                n_d, k_d = refractive_index_at(sp, diag_nm)
                ri_diag[sp] = (np.asarray(n_d), np.asarray(k_d))
        self._cache = _OpticsCache(sw_nm, lw_nm, ri_sw, ri_lw, aod_idx, aod_nm,
                                   ang_idx, ang_nm, ri_diag)

    def _band_optics(self, state, aer, num_per_area, centers_nm, ri,
                     want_decomposition: bool = False):
        """Per-band ``(aod, ssa, asy)``, each ``(n_band, nlev, ncols)``.

        The bands are independent and share the whole modal geometry
        (volumes, wet radii, number), so the band axis is mapped with a
        single ``jax.vmap`` rather than a Python loop: only the wavelength
        and the per-species refractive index ``(n, k)`` vary across bands.
        The inner loops over modes/species stay explicit — they are ragged
        (each mode carries a different species set) and small.
        """
        n_band = centers_nm.shape[0]
        if n_band == 0:
            empty = jnp.zeros((0,) + state.temperature.shape)
            if not want_decomposition:
                return empty, empty, empty
            n_mode = len(self._spec.modes)
            empty_m = jnp.zeros((0, n_mode) + state.temperature.shape)
            species = {sp for m in self._spec.modes for sp in m.species} | {"wat"}
            return (empty, empty, empty, empty_m, empty_m,
                    {sp: empty for sp in species}, {sp: empty for sp in species})

        zeros = jnp.zeros_like(state.temperature)
        lam_all = jnp.asarray(centers_nm, state.temperature.dtype) * 1.0e-9
        # ri: species -> (n[n_band], k[n_band]); vmap maps the band axis.
        ri_j = {sp: (jnp.asarray(n), jnp.asarray(k)) for sp, (n, k) in ri.items()}

        def one_band(lam_m, ri_band):
            aod = jnp.zeros_like(state.temperature)
            scat = jnp.zeros_like(state.temperature)
            gscat = jnp.zeros_like(state.temperature)
            per_mode_aod: list = []
            per_mode_abs: list = []
            sp_aod: dict = {}
            sp_abs: dict = {}
            for i, mode in enumerate(self._spec.modes):
                r_wet = aer.r_wet[i]
                ln_sig = math.log(mode.geom_std_dev)
                vol_n = jnp.zeros_like(state.temperature)
                vol_k = jnp.zeros_like(state.temperature)
                vol_tot = jnp.zeros_like(state.temperature)
                vol_sp: dict = {}
                for sp in mode.species:
                    mass = state.tracers.get(mass_name(sp, mode.short), zeros)
                    v = mass / self._spec.species_props(sp).density
                    n_sp, k_sp = ri_band[sp]
                    vol_n = vol_n + v * n_sp
                    vol_k = vol_k + v * k_sp
                    vol_tot = vol_tot + v
                    vol_sp[sp] = v
                vol_dry = vol_tot
                v_water = aer.number[i] * _FOUR_THIRDS_PI * jnp.maximum(
                    r_wet ** 3 - aer.r_dry[i] ** 3, 0.0
                )
                n_w, k_w = ri_band["h2o"]
                vol_n = vol_n + v_water * n_w
                vol_k = vol_k + v_water * k_w
                vol_tot = vol_tot + v_water

                safe = jnp.maximum(vol_tot, _TINY)
                m_n = jnp.where(vol_tot > _TINY, vol_n / safe, 1.5)
                m_k = jnp.where(vol_tot > _TINY, vol_k / safe, 1.0e-8)
                # Integrate Mie efficiencies over the mode's lognormal in
                # ln r — ``r_wet`` is the NUMBER-MEDIAN radius, so a single
                # Qext(r_wet)·π·r_wet² misses the r² moment (×e^{2ln²σ})
                # and Qext at the extinction-carrying sizes. Gauss–Hermite:
                # r_k = r_g·e^{√2 lnσ t_k}, weight (w_k/√π)·e^{2√2 lnσ t_k};
                # σ preserved under hygroscopic growth, refractive index
                # size-independent (only x varies per node).
                # ``lax.scan`` so only one node's Mie intermediates are
                # live at a time (unrolling multiplies the working set by
                # n_nodes × n_bands and exceeds GPU memory).
                def _gh_node(carry, t_w):
                    t_k, w_k = t_w
                    growth = jnp.exp(math.sqrt(2.0) * ln_sig * t_k)
                    x_k = 2.0 * math.pi * (r_wet * growth) / lam_m
                    q_k, ssa_k, g_k = interp_mie(self._lut, x_k, m_n, m_k)
                    wgt = (w_k / math.sqrt(math.pi)) * growth ** 2
                    c_sec, c_scat, c_gscat = carry
                    return (
                        c_sec + wgt * q_k,
                        c_scat + wgt * q_k * ssa_k,
                        c_gscat + wgt * q_k * ssa_k * g_k,
                    ), None

                (sec, sec_scat, sec_gscat), _ = jax.lax.scan(
                    _gh_node,
                    (jnp.zeros_like(r_wet),) * 3,
                    (jnp.asarray(_GH_NODES, r_wet.dtype),
                     jnp.asarray(_GH_WEIGHTS, r_wet.dtype)),
                )
                aod_i = num_per_area[i] * sec * math.pi * r_wet ** 2
                # Physical mass gate: tau is EXACTLY zero where the mode
                # carries no material. The number floor above handles the
                # NEGATIVE side of the cold-start Gibbs ringing, but the
                # POSITIVE side pairs a tiny number with a garbage wet
                # radius (r ~ (V/n)^(1/3) of two ringing fields), and
                # n·q_ext·π·r² is then finite at EMPTY levels. At the
                # 1 Pa model top the 1/Δp heating amplification turned
                # that into 13,000 K/day of spurious SW absorption —
                # +90 K in 6 h and a global NaN by day 10 of the first
                # coupled JAM year. 1e-24 m³/kg (≈1e-21 kg/kg of aerosol)
                # is radiatively nothing and far above ringing amplitudes.
                # Gate on the DRY species volume: the hygroscopic water
                # term n·(r_wet³−r_dry³) is itself ringing garbage when
                # there is no dry aerosol to condense on, and it passes a
                # total-volume gate on its own.
                gate = (vol_dry > 1.0e-24)
                area = num_per_area[i] * math.pi * r_wet ** 2
                aod_gated = jnp.where(gate, aod_i, 0.0)
                scat_gated = jnp.where(gate, area * sec_scat, 0.0)
                aod = aod + aod_gated
                scat = scat + scat_gated
                gscat = gscat + jnp.where(gate, area * sec_gscat, 0.0)

                # Diagnostic decomposition (jax-gcm#584). The mode's species
                # are volume-mixed into ONE effective refractive index before
                # the Mie call, so there is no per-species extinction to
                # recover: what follows is an APPORTIONMENT of the mode's
                # extinction, which is what an internally-mixed model can
                # honestly report. Extinction is apportioned by species
                # VOLUME fraction; absorption by k-weighted volume,
                # V_s*k_s / sum(V_s*k_s).
                #
                # The absorption weight is not an ad-hoc choice: under the
                # volume mixing rule used above, sum(V_s*k_s) IS V_tot*k_eff,
                # so the weight is exactly the linear decomposition of the
                # effective imaginary index this mode's optics were computed
                # from. ECHAM-HAM's ham_rad_diag uses the identical pair of
                # weights (mo_ham_rad.f90:1926-1936, "based on volume average
                # for optical thickness, additionally weighted with ni for
                # absorption"), so these fields are directly comparable to
                # HAM's TAU_COMP_*/ABS_COMP_*. Both reduce to the external-
                # mixture answer when a mode carries one species.
                if not want_decomposition:
                    continue
                per_mode_aod.append(aod_gated)
                per_mode_abs.append(aod_gated - scat_gated)
                # ``vol_tot``/``vol_k`` here INCLUDE the hygroscopic water
                # added above, so the fractions sum to one over the mode's
                # species plus water — no extinction is dropped or double
                # counted.
                inv_vol = jnp.where(vol_tot > _TINY, 1.0 / jnp.maximum(vol_tot, _TINY), 0.0)
                inv_volk = jnp.where(vol_k > _TINY, 1.0 / jnp.maximum(vol_k, _TINY), 0.0)
                abs_gated = aod_gated - scat_gated
                for sp, v_sp in vol_sp.items():
                    sp_aod[sp] = sp_aod.get(sp, 0.0) + aod_gated * (v_sp * inv_vol)
                    sp_abs[sp] = sp_abs.get(sp, 0.0) + abs_gated * (
                        v_sp * ri_band[sp][1] * inv_volk)
                # Aerosol water is itself an AeroCom component (TAU_COMP_WAT).
                sp_aod["wat"] = sp_aod.get("wat", 0.0) + aod_gated * (v_water * inv_vol)
                sp_abs["wat"] = sp_abs.get("wat", 0.0) + abs_gated * (
                    v_water * k_w * inv_volk)
            # Clamp the extinction-/scattering-weighted SSA and asymmetry to
            # their physical [0, 1] range. With a non-negative per-mode AOD
            # (number floored at 0 in ``__call__``) these ratios are already
            # bounded, but a tiny Mie-LUT edge overshoot in ``q_ext``/``ssa``
            # could still nudge them out of range, and RRTMGP's two-stream
            # solver NaNs on an SSA outside [0, 1] — so clamp defensively. SSA is
            # physically [0, 1]; the asymmetry parameter is [-1, 1] (negative g =
            # back-scattering), so keep its lower bound at -1 to preserve valid
            # back-scattering aerosol rather than only bounding overshoot.
            ssa_b = jnp.clip(scat / jnp.maximum(aod, _TINY), 0.0, 1.0)
            asy_b = jnp.clip(gscat / jnp.maximum(scat, _TINY), -1.0, 1.0)
            if not want_decomposition:
                return aod, ssa_b, asy_b
            return (aod, ssa_b, asy_b,
                    jnp.stack(per_mode_aod), jnp.stack(per_mode_abs),
                    sp_aod, sp_abs)

        return jax.vmap(one_band)(lam_all, ri_j)

    def _optics_diagnostics_fields(self, state, aer, num_per_area, dz) -> dict:
        """AeroCom per-species / per-mode / spectral optics (jax-gcm#584).

        A second Mie pass at ``_DIAG_WAVELENGTHS_NM``, independent of the
        radiation banding so the reported quantities are at the wavelengths
        the observations are actually at (550 nm for satellite/AERONET AOD,
        440/670/865 nm for the Angstrom exponent and AERONET SSA, 355 nm for
        the ATLID/EarthCARE lidar) rather than at whichever band centre the
        radiation configuration happens to provide.

        Everything except the 355 nm extinction profile is reduced to a
        column integral here rather than downstream: these fields live in the
        ``lax.scan`` carry (they ride the radiation gate with the rest of the
        band optics), and keeping ``n_species x n_wavelength x nlev`` 3-D
        arrays alive there would cost hundreds of MB at T63L47.

        The radiation-stability guards (``_AER_RAD_PMIN`` masking,
        ``_MAX_LAYER_TAU``) are deliberately NOT applied. Those bound a
        heating rate divided by a near-zero lid air mass; the diagnostic is
        meant to be the physical observable a satellite retrieval would see,
        so it reports the column as the model actually holds it.
        """
        c = self._cache
        tau, ssa, _asy, mode_tau, mode_abs, sp_tau, sp_abs = self._band_optics(
            state, aer, num_per_area,
            np.asarray(_DIAG_WAVELENGTHS_NM, np.float64), c.ri_diag,
            want_decomposition=True,
        )
        # (n_wavelength, nlev, *horiz) -> column integral over the vertical.
        # The ``maximum`` is a defensive clamp only: the modal number is
        # already floored at 0 in ``_compute_fields`` and the Mie
        # efficiencies are non-negative, so every per-layer tau here is
        # non-negative by construction. It is kept so a future change
        # upstream cannot silently produce a negative reported AOD.
        def col(x):
            # x is (n_wavelength, nlev, *horiz); axis 1 is the vertical.
            return jnp.maximum(jnp.sum(x, axis=1), 0.0)

        od = col(tau)                                   # (n_wavelength, *horiz)
        absorp = col(tau * (1.0 - ssa))
        out = {
            "od550aer": od[_I550], "abs550aer": absorp[_I550],
            "od355aer": od[_I355], "od440aer": od[_I440],
            "od670aer": od[_I670], "od865aer": od[_I865],
        }
        # Single-scattering albedo needs a non-zero optical depth to be
        # defined; report 1 (purely scattering) in aerosol-free columns
        # rather than 0/0, so a zonal mean is not dragged down by clean air.
        out["ssa440aer"] = jnp.where(
            od[_I440] > _TINY, 1.0 - absorp[_I440] / jnp.maximum(od[_I440], _TINY), 1.0)
        # Angstrom exponent from the 440/865 nm pair (AeroCom's ang4487aer is
        # nominally 440/870; 865 nm is used here because it is a jax-rrtmgp
        # band centre, a 0.6% lever-arm difference). Undefined without
        # aerosol at BOTH wavelengths -> 0 (spectrally flat).
        both = (od[_I440] > _TINY) & (od[_I865] > _TINY)
        ratio = jnp.maximum(od[_I440], _TINY) / jnp.maximum(od[_I865], _TINY)
        ang = jnp.where(both, -jnp.log(ratio) / math.log(440.0 / 865.0), 0.0)
        out["ang4487aer"] = ang
        # ECHAM-HAM reports its Angstrom exponent over 550/865 nm
        # (ANG_550nm_865nm) rather than AeroCom's 440/870 pair, so publish
        # that one too — it is free here (both column AODs already exist)
        # and it is what a direct HAM intercomparison needs.
        both58 = (od[_I550] > _TINY) & (od[_I865] > _TINY)
        ratio58 = jnp.maximum(od[_I550], _TINY) / jnp.maximum(od[_I865], _TINY)
        out["ang550865aer"] = jnp.where(
            both58, -jnp.log(ratio58) / math.log(550.0 / 865.0), 0.0)
        # Aerosol index = AOD x Angstrom exponent: the CCN proxy that
        # correlates with number far better than AOD alone, because the
        # Angstrom factor discounts the coarse mode.
        out["aerindex"] = od[_I550] * ang
        # 3-D extinction coefficient [m-1] at the lidar wavelength. Same
        # defensive clamp as ``col`` so that integrating ec355aer over dz
        # reproduces od355aer exactly under either sign convention.
        out["ec355aer"] = jnp.maximum(tau[_I355], 0.0) / jnp.maximum(dz, _TINY)

        for i, mode in enumerate(self._spec.modes):
            out[f"od550_mode_{mode.short}"] = jnp.maximum(
                jnp.sum(mode_tau[_I550, i], axis=0), 0.0)
            out[f"abs550_mode_{mode.short}"] = jnp.maximum(
                jnp.sum(mode_abs[_I550, i], axis=0), 0.0)
        for sp, v in sp_tau.items():
            out[f"od550_{sp}"] = jnp.maximum(jnp.sum(v[_I550], axis=0), 0.0)
        for sp, v in sp_abs.items():
            out[f"abs550_{sp}"] = jnp.maximum(jnp.sum(v[_I550], axis=0), 0.0)
        return out

    def _compute_fields(self, state, diagnostics) -> dict:
        """Fresh per-band optics + the AOD-550 column diagnostic."""
        aer = diagnostics["_jam_state"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        c = self._cache

        # Number per unit area [m^-2] per mode (number is kg^-1). Floor the
        # modal number at 0 before it enters the optics: spectral advection of
        # the aerosol-number tracers leaves small NEGATIVE number on the
        # near-zero cold-start field (Gibbs ringing). A negative number gives a
        # negative per-mode extinction, which can drive the band AOD ≤ 0 and
        # then the extinction-weighted SSA (= scat / AOD) and asymmetry to
        # ±huge — RRTMGP's two-stream solver NaNs on an out-of-range SSA. As the
        # aerosol burden grows the ringing crosses zero within the first day,
        # so this is a hard stability requirement, not a cosmetic floor. With
        # number ≥ 0 every derived optic is physical (AOD ≥ 0 ⇒ SSA, g ∈
        # [0, 1]); consistent with the AOD-550 diagnostic floor below.
        num_per_area = jnp.maximum(aer.number, 0.0) * (air_density * dz)[jnp.newaxis]

        aod_sw, ssa_sw, asy_sw = self._band_optics(
            state, aer, num_per_area, c.sw_nm, c.ri_sw
        )
        aod_lw, ssa_lw, asy_lw = self._band_optics(
            state, aer, num_per_area, c.lw_nm, c.ri_lw
        )

        # No aerosol radiative effect above _AER_RAD_PMIN. In the thin lid
        # layers (Δp of order 1 Pa on the L47 grid) ANY absorbed flux divides
        # by a near-zero air mass: once real absorbing aerosol mixes up there
        # — which took ~200 simulated days of online emissions — the heating
        # rate is effectively unbounded (day-207 winter blow-up: 453 K at
        # level 2 with the aerosol burden grown through winter; the
        # aerosol-dark validation year could never trigger this). The aerosol
        # mass above ~2 hPa is radiatively negligible, so zeroing tau there
        # is the same pragmatism as the upper-atmosphere temperature sponge
        # (radiation schemes are outside their validity at the lid).
        p_full = diagnostics.get("pressure_full")
        if p_full is not None:
            keep = (p_full > _AER_RAD_PMIN)[jnp.newaxis]
            aod_sw = aod_sw * keep
            aod_lw = aod_lw * keep
        # Bound the per-layer extinction (see _MAX_LAYER_TAU).
        aod_sw = jnp.minimum(aod_sw, _MAX_LAYER_TAU)
        aod_lw = jnp.minimum(aod_lw, _MAX_LAYER_TAU)

        # Column aerosol optical depth at ~550 nm: the total-column extinction
        # optical depth (sum of the per-layer band AOD over the vertical axis 0)
        # in the SW band closest to 550 nm. This is the standard satellite /
        # AERONET observable, so it is the cleanest external check on the scheme.
        # ``aod_sw`` is ``(n_band, nlev, *horiz)``; the column AOD is ``(*horiz)``.
        # Clamp at 0: optical depth is non-negative by definition, but spectral
        # transport leaves small negative aerosol number on the near-zero
        # cold-start field (Gibbs ringing), which can drive a tiny negative
        # per-layer extinction. The floor keeps the reported observable physical.
        if aod_sw.shape[0]:
            aod_550 = jnp.maximum(jnp.sum(aod_sw[c.aod_band_idx], axis=0), 0.0)
        else:
            aod_550 = jnp.zeros_like(state.temperature[0])

        # Broadband 550 nm profile fields for hosts that read a single aerosol
        # profile rather than per-band optics — specifically the grey
        # two-stream scheme (grey_two_stream/radiation_scheme.py reads
        # ``aerosol.aod_profile``/``ssa_profile``/``asy_profile``/``angstrom``
        # and band-scales them itself). With MACv2-SP removed from the JAM path
        # (#640) nothing else writes these, so grey+JAM would otherwise have a
        # silent zero direct effect. Taken from the SW band whose centre is
        # nearest 550 nm — a band-CENTRE approximation to the exact 550 nm
        # optics (the per-species Mie pass at exactly 550 nm is the separate
        # aerocom_optics diagnostic); it reuses the radiation bands already
        # computed here and costs no extra Mie work. ``aod_profile`` inherits
        # the same ``_AER_RAD_PMIN`` mask and ``_MAX_LAYER_TAU`` cap as the
        # per-band SW AOD, so grey and RRTMGP see a consistent burden.
        if aod_sw.shape[0]:
            aod_profile = aod_sw[c.aod_band_idx]
            ssa_profile = ssa_sw[c.aod_band_idx]
            asy_profile = asy_sw[c.aod_band_idx]
        else:
            zeros3 = jnp.zeros_like(state.temperature)
            aod_profile = ssa_profile = asy_profile = zeros3

        # Column Angstrom exponent from the 550/865 nm band pair, which grey
        # uses to scale ``aod_profile`` to its own SW/LW band wavelengths. This
        # is the cheap band-RATIO Angstrom (two radiation-band column AODs) —
        # NOT the Mie-based ``ang550865aer`` of the aerocom_optics pass — and
        # is only well-defined with >=2 SW bands. Defaults to the fine-mode 1.5
        # (the ``AerosolData`` zero default) with a single broadband SW band
        # (grey's own config) or where a column has no aerosol at both bands.
        if aod_sw.shape[0] and c.ang_band_idx != c.aod_band_idx:
            od_a = jnp.maximum(jnp.sum(aod_sw[c.aod_band_idx], axis=0), 0.0)
            od_b = jnp.maximum(jnp.sum(aod_sw[c.ang_band_idx], axis=0), 0.0)
            both = (od_a > _TINY) & (od_b > _TINY)
            ratio = jnp.maximum(od_a, _TINY) / jnp.maximum(od_b, _TINY)
            lam_ratio = math.log(c.aod_band_nm / c.ang_band_nm)
            angstrom = jnp.where(both, -jnp.log(ratio) / lam_ratio, 1.5)
        else:
            angstrom = jnp.full_like(state.temperature[0], 1.5)

        fields = {
            "aod_sw_per_band": aod_sw, "ssa_sw_per_band": ssa_sw,
            "asy_sw_per_band": asy_sw,
            "aod_lw_per_band": aod_lw, "ssa_lw_per_band": ssa_lw,
            "asy_lw_per_band": asy_lw,
            "aod_profile": aod_profile, "ssa_profile": ssa_profile,
            "asy_profile": asy_profile, "angstrom": angstrom,
            "aod_550": aod_550,
        }
        if self._optics_diagnostics:
            # Nested so the AerosolData copy in ``__call__`` (which splats
            # ``fields``) does not see these as struct fields; they are
            # plain diagnostics keys.
            fields["_optics_diag"] = self._optics_diagnostics_fields(
                state, aer, num_per_area, dz)
        return fields

    def __call__(self, state, diagnostics, forcing, terrain):
        cached = diagnostics.get("_jam_optics")
        if (self._radiation_interval_s is None or cached is None
                or "radiation" not in diagnostics):
            # Ungated (every step): gate unconfigured, or no cached fields in
            # the carry yet (the structural-template pass builds them here).
            fields = self._compute_fields(state, diagnostics)
        else:
            # Same gate arithmetic as ``radiation_should_compute``, on the
            # same pre-increment ``radiation.step`` carry counter (see
            # ``configure_radiation_gate``).
            dt = diagnostics["_dt_seconds"]
            steps_per_call = jnp.int32(
                jnp.maximum(jnp.round(self._radiation_interval_s / dt), 1)
            )
            fields = jax.lax.cond(
                jnp.mod(diagnostics["radiation"].step, steps_per_call) == 0,
                # Pin fresh leaves to the carry's dtypes so both cond branches
                # type-check (under x64 hosts some constants promote).
                lambda: jax.tree.map(
                    lambda n, o: n.astype(o.dtype),
                    self._compute_fields(state, diagnostics), cached,
                ),
                lambda: cached,
            )

        new_aerosol = diagnostics["aerosol"].copy(
            **{k: v for k, v in fields.items()
               if k not in ("aod_550", "_optics_diag")}
        )
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        # The column AOD-550 no longer publishes a top-level
        # ``aerosol_optical_depth`` key (#640): that name collided with the
        # unrelated per-band ``RadiationInput`` field, and the value now lives
        # under the JAM namespace as ``jam_optics.aod_550`` (from the
        # ``_jam_optics`` carry dict, which also carries the per-band optics
        # plumbing and the grey profile fields).
        return tendency, {
            **diagnostics,
            **fields.get("_optics_diag", {}),
            "aerosol": new_aerosol,
            "_jam_optics": fields,
        }
