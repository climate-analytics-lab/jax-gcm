"""MACv2 natural aerosol background for the radiation.

MACv2-SP (:mod:`jcm.physics.aerosol.macv2_sp`) describes the anthropogenic
increment of the aerosol column. This term adds what the plumes sit on: the
coarse mode (dust, sea salt) and the pre-industrial fine mode of the MACv2
climatology (Kinne 2019, doi:10.1080/16000889.2019.1623639), read from the
``macv2_natural_aerosol`` bundle (:mod:`jcm.data.mirror.macv2`) into
``ForcingData.natural_aerosol``. ICON makes the same kind of pairing with
``irad_aero = 18`` (``mo_echam_phy_bcs.f90``).

Formulation, following ICON ``mo_bc_aeropt_kinne.f90::set_bc_aeropt_kinne``:

* **Time.** The twelve monthly means are blended between mid-month anchors by
  the forcing's own ``WRAP_YEAR`` selection (ECHAM's ``time_weights``), so the
  term sees fields already valid for the step.
* **Vertical.** Each mode's column optical depth is spread over the model
  layers by the climatology's layer fractions. The bundle gives the fraction
  of the column in each of 20 height layers above sea level, so a model layer
  takes the share of every climatology layer it overlaps in height, and the
  result is renormalised to one over the model column — the reference's own
  renormalisation, which keeps the column optical depth whatever the model
  orography or lid.
* **Shortwave.** The two modes combine with whatever the ``aerosol`` slot
  already holds (the plumes) as sums of τ, τω and τωg. ECHAM6.3's
  ``set_aop_kinne`` carries a second factor of ω in the last sum; that is not
  followed.
* **Longwave.** Coarse mode only, absorption only: τ(1 − ω) of the
  far-infrared optics, added as a non-scattering optical depth.
* **Bands.** The bundle is on RRTMG's bands. Each radiation band takes the
  climatology band whose limits contain its centre (the nearest one
  otherwise), which is one-to-one for RRTMGP and picks the 550 nm band for a
  broadband scheme.

The Twomey factor is untouched: MACv2-SP normalises it with its own background
optical depth (Stevens et al. 2017), and the climatology does not enter it.

Broadcasting-native: the vertical is axis 0 (top first), bands are a leading
axis, and any trailing axes are horizontal.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency
from jcm.terrain import TerrainData

#: Optical depth below which a layer counts as aerosol-free in the completing
#: divisions. As in ``macv2_sp``: the division's reverse-mode derivative forms
#: ``num / den**2``, so the threshold's square must stay a normal float32.
_TINY = 1e-15

#: Nominal wavelength [nm] of the band the 550 nm diagnostics are read from.
_DIAGNOSTIC_NM = 550.0


def band_index(centres_cm, lower_cm, upper_cm):
    """Climatology band for each radiation band, by band-centre wavenumber.

    The band whose ``[lower, upper]`` limits contain the centre; where none
    does, the band with the nearest edge. ``centres_cm`` is ``(n_model,)``,
    the limits ``(n_source,)``; returns integer indices ``(n_model,)``.
    """
    centres = jnp.asarray(centres_cm)[:, None]
    lower = jnp.asarray(lower_cm)[None, :]
    upper = jnp.asarray(upper_cm)[None, :]
    distance = jnp.maximum(jnp.maximum(lower - centres, centres - upper), 0.0)
    return jnp.argmin(distance, axis=1)


def distribute_layer_fractions(fraction, layer_bottom, layer_top, height_half):
    """Spread a column over model layers by height overlap.

    Args:
        fraction: Fraction of the column in each climatology layer,
            ``(nlayer, *horiz)``, summing to one over axis 0.
        layer_bottom: Climatology layer bottoms [m above sea level],
            ``(nlayer,)``.
        layer_top: Climatology layer tops [m above sea level], ``(nlayer,)``.
        height_half: Model interface heights [m above sea level],
            ``(nlev + 1, *horiz)``, top first.

    Returns:
        Fraction of the column in each model layer, ``(nlev, *horiz)``,
        summing to one over axis 0 wherever the model column overlaps the
        climatology at all (zero elsewhere).

    Each climatology layer is uniform inside itself, so a model layer takes
    from it the share of its depth the two overlap. What lies below the model
    surface or above its lid is redistributed by the renormalisation. The
    overlap is formed per layer pair rather than as a difference of cumulative
    shares, which would lose the thin upper layers to float32 cancellation.

    """
    upper, lower = height_half[:-1], height_half[1:]
    profile = jnp.zeros_like(upper)
    for j in range(fraction.shape[0]):
        overlap = jnp.clip(jnp.minimum(layer_top[j], upper)
                           - jnp.maximum(layer_bottom[j], lower), 0.0, None)
        profile = profile + fraction[j] * overlap / (layer_top[j]
                                                     - layer_bottom[j])
    total = jnp.sum(profile, axis=0)
    has = total > _TINY
    return jnp.where(has, profile / jnp.where(has, total, 1.0), 0.0)


def natural_aerosol_optics(climatology, height_half, sw_centres_cm,
                           lw_centres_cm):
    """Per-band layer optics of the natural background, as additive sums.

    Args:
        climatology: The time-selected ``ForcingData.natural_aerosol`` mapping
            (:func:`jcm.forcing.read_natural_aerosol`), horizontal axes
            already in the host layout: ``aod_/ssa_/asy_sw_coarse`` and
            ``..._sw_fine`` ``(n_sw, *horiz)``, ``aod_/ssa_lw_coarse``
            ``(n_lw, *horiz)``, ``layer_fraction_coarse/fine``
            ``(nlayer, *horiz)``, and the band limits and layer bounds.
        height_half: Model interface heights [m above sea level],
            ``(nlev + 1, *horiz)``, top first.
        sw_centres_cm: Radiation shortwave band centres [cm⁻¹], ``(nb_sw,)``.
        lw_centres_cm: Radiation longwave band centres [cm⁻¹], ``(nb_lw,)``
            (may be empty).

    Returns:
        ``dict`` with ``tau_sw``, ``scat_sw`` (τω), ``forward_sw`` (τωg), each
        ``(nb_sw, nlev, *horiz)``; ``absorption_lw`` ``(nb_lw, nlev,
        *horiz)``; and the 550 nm column optical depths ``od550_coarse``,
        ``od550_fine`` ``(*horiz)``.

    """
    c = climatology
    profile_coarse = distribute_layer_fractions(
        c["layer_fraction_coarse"], c["layer_bottom"], c["layer_top"],
        height_half)
    profile_fine = distribute_layer_fractions(
        c["layer_fraction_fine"], c["layer_bottom"], c["layer_top"],
        height_half)

    sw = band_index(sw_centres_cm, c["wavenumber_lower_sw"],
                    c["wavenumber_upper_sw"])

    def on_layers(column, profile):
        return column[:, None] * profile[None]

    tau_c = on_layers(jnp.take(c["aod_sw_coarse"], sw, axis=0), profile_coarse)
    tau_f = on_layers(jnp.take(c["aod_sw_fine"], sw, axis=0), profile_fine)
    scat_c = tau_c * jnp.take(c["ssa_sw_coarse"], sw, axis=0)[:, None]
    scat_f = tau_f * jnp.take(c["ssa_sw_fine"], sw, axis=0)[:, None]
    forward = (scat_c * jnp.take(c["asy_sw_coarse"], sw, axis=0)[:, None]
               + scat_f * jnp.take(c["asy_sw_fine"], sw, axis=0)[:, None])

    n_lw = jnp.asarray(lw_centres_cm).shape[0]
    if n_lw:
        lw = band_index(lw_centres_cm, c["wavenumber_lower_lw"],
                        c["wavenumber_upper_lw"])
        absorbing = (jnp.take(c["aod_lw_coarse"], lw, axis=0)
                     * (1.0 - jnp.take(c["ssa_lw_coarse"], lw, axis=0)))
        absorption_lw = on_layers(absorbing, profile_coarse)
    else:
        absorption_lw = jnp.zeros((0,) + profile_coarse.shape,
                                  dtype=profile_coarse.dtype)

    green = band_index(jnp.asarray([1.0e7 / _DIAGNOSTIC_NM]),
                       c["wavenumber_lower_sw"], c["wavenumber_upper_sw"])[0]
    return {
        "tau_sw": tau_c + tau_f,
        "scat_sw": scat_c + scat_f,
        "forward_sw": forward,
        "absorption_lw": absorption_lw,
        "od550_coarse": jnp.take(c["aod_sw_coarse"], green, axis=0),
        "od550_fine": jnp.take(c["aod_sw_fine"], green, axis=0),
    }


def add_to_aerosol(aerosol, optics):
    """Return ``aerosol`` with the background's per-band optics mixed in.

    Shortwave: sums of τ, τω and τωg with what the slot already holds, then
    the completing divisions with the reference's limits (ω → 1 and g → 0
    without optical depth; g → 0 without scattering). Longwave: the
    background is a pure absorber, so it adds to the optical depth and leaves
    the scattered part, and with it the asymmetry, as they were. Only the
    per-band radiative fields change: the 550 nm plume diagnostics and the
    Twomey fields keep describing MACv2-SP.
    """
    tiny = jnp.asarray(_TINY, dtype=aerosol.aod_sw_per_band.dtype)

    old_scat = aerosol.aod_sw_per_band * aerosol.ssa_sw_per_band
    tau = aerosol.aod_sw_per_band + optics["tau_sw"]
    scat = old_scat + optics["scat_sw"]
    forward = old_scat * aerosol.asy_sw_per_band + optics["forward_sw"]
    has_tau, has_scat = tau > tiny, scat > tiny
    ssa = jnp.where(has_tau, scat / jnp.where(has_tau, tau, 1.0), 1.0)
    asy = jnp.where(has_scat, forward / jnp.where(has_scat, scat, 1.0), 0.0)
    updates = {"aod_sw_per_band": tau, "ssa_sw_per_band": ssa,
               "asy_sw_per_band": asy}

    if aerosol.aod_lw_per_band.shape[0]:
        old_scat_lw = aerosol.aod_lw_per_band * aerosol.ssa_lw_per_band
        tau_lw = aerosol.aod_lw_per_band + optics["absorption_lw"]
        has_lw = tau_lw > tiny
        updates["aod_lw_per_band"] = tau_lw
        updates["ssa_lw_per_band"] = jnp.where(
            has_lw, old_scat_lw / jnp.where(has_lw, tau_lw, 1.0),
            aerosol.ssa_lw_per_band)
    return aerosol.copy(**updates)


class Macv2NaturalAerosol(PhysicsTerm):
    """MACv2 natural aerosol background added to the ``aerosol`` slot.

    Runs after the term that owns the slot (``Macv2SpAerosol``) and before
    radiation. Reads ``forcing.natural_aerosol`` (the time-selected
    ``macv2_natural_aerosol`` bundle); with no climatology on the forcing
    the slot is passed through unchanged and the two diagnostics are zero,
    so an idealised or aquaplanet run composes the same term list.
    """

    name: ClassVar[str] = "macv2_natural_aerosol"
    category: ClassVar[str] = "aerosol"
    requires: ClassVar[tuple[str, ...]] = ("aerosol", "height_half")
    provides: ClassVar[tuple[str, ...]] = (
        "aerosol", "natural_od550_coarse", "natural_od550_fine")
    output_attrs: ClassVar[dict[str, dict[str, str]]] = {
        "natural_od550_coarse": {
            "units": "1",
            "long_name": ("MACv2 coarse-mode (dust, sea salt) column aerosol "
                          "optical depth at 550 nm"),
        },
        "natural_od550_fine": {
            "units": "1",
            "long_name": ("MACv2 pre-industrial fine-mode column aerosol "
                          "optical depth at 550 nm"),
        },
    }

    def consumed_forcing_fields(self) -> tuple[str, ...]:
        """Return the optional forcing field this term reads."""
        return ("natural_aerosol",)

    def __call__(
        self,
        state,
        diagnostics: dict,
        forcing,
        terrain: TerrainData,
    ):
        """Mix the background into the per-band aerosol optics for this step."""
        horiz = state.temperature.shape[1:]
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        climatology = getattr(forcing, "natural_aerosol", None)
        if climatology is None:
            zero = jnp.zeros(horiz, dtype=state.temperature.dtype)
            return tendency, {**diagnostics, "natural_od550_coarse": zero,
                              "natural_od550_fine": zero}

        # Forcing fields arrive grid-shaped ``(..., nlon, nlat)`` in either
        # host; fold the two horizontal axes into the state's own horizontal
        # layout (a no-op on the grid host, the lon-major ravel on the column
        # host). Band limits and layer bounds are one-dimensional metadata.
        def to_host(x):
            return x if x.ndim < 2 else x.reshape(x.shape[:-2] + horiz)

        climatology = {key: to_host(jnp.asarray(value))
                       for key, value in climatology.items()}

        band_config = diagnostics.get("_band_config")
        if band_config is None:
            from jcm.physics.radiation.band_config import RadiationBandConfig
            band_config = RadiationBandConfig.broadband()
        sw_centres = 1.0e7 / jnp.asarray(band_config.sw_band_centers_nm,
                                         dtype=jnp.float32)
        lw_nm = jnp.asarray(band_config.lw_band_centers_nm, dtype=jnp.float32)
        lw_centres = 1.0e7 / lw_nm if lw_nm.shape[0] else lw_nm

        optics = natural_aerosol_optics(
            climatology, diagnostics["height_half"], sw_centres, lw_centres)
        return tendency, {
            **diagnostics,
            "aerosol": add_to_aerosol(diagnostics["aerosol"], optics),
            "natural_od550_coarse": optics["od550_coarse"],
            "natural_od550_fine": optics["od550_fine"],
        }
