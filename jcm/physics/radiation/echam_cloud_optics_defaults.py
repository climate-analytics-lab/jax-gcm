"""Per-truncation defaults of ECHAM's cloud-optics inhomogeneity factors.

ECHAM6.3 sets the sub-grid cloud-optics inhomogeneity factors per spectral
truncation in ``mo_cloud_optics.f90::setup_cloud_optics`` (l.114-134): the
no-convection liquid factor ``zinhoml1`` and the shallow-convection liquid
factor ``zinhoml2`` are 0.80 / 0.40 at every truncation, but the deep/mid-level
convective liquid factor ``zinhoml3`` is 0.40 at T31 and 0.80 at T63/T127/T255,
and the ice factor ``zinhomi`` is 0.80 at T31/T63 and 0.85 at T127/T255. ECHAM
has no row for any other truncation (``CALL finish('mo_cloud_optics',
'Truncation not supported.')``).

:data:`ECHAM_CLOUD_OPTICS_DEFAULTS` is that table, row for row, keyed by the
field names of :class:`~jcm.physics.radiation.radiation_types.RadiationParameters`
(``cloud_inhomogeneity_liquid`` = zinhoml1, ``cloud_inhomogeneity_liquid_shallow``
= zinhoml2, ``cloud_inhomogeneity_liquid_convective`` = zinhoml3,
``cloud_inhomogeneity_ice`` = zinhomi). :func:`echam_cloud_optics_defaults`
reads it through :func:`jcm.physics.resolution_defaults.resolution_defaults`,
the same construction-time mechanism the ECHAM cloud scheme's tunables use
(:mod:`jcm.physics.clouds.echam_cloud_defaults`): the table's values exactly at
T31/T63/T127/T255, and linear interpolation between them elsewhere (T106 lies
43/64 of the way from T63 to T127) — ECHAM itself has no configuration between
its four truncations, so the interpolated values are jcm's choice and are
untuned, exactly as for the cloud-cover tunables.

The ECHAM-HAM prognostic-CDNC pairing (``lcdnc_progn`` with ``ncd_activ = 2``,
which is jcm's JAM aerosol module: the 2M cloud scheme fed by
Abdul-Razzak-Ghan activation) re-tunes the ice factor to ``zinhomi = 0.7`` —
but ECHAM defines that override at T63 only (``nn == 63``, both the L31 and
L47 vertical grids); it has no JAM-ARG row at any other truncation to hold,
interpolate from, or interpolate to. ``jcm`` HOLDS that override at every
truncation rather than interpolating or tapering it back to the base table's
``zinhomi``, because it is a model-PHYSICS re-tuning for a specific aerosol
activation scheme, not a resolution-dependent numerical choice -- there is no
reason to expect the right value at T106 or T255 to revert toward the
non-interactive-aerosol default as truncation increases. This is jcm's
decision, not ECHAM's (ECHAM has literally never run the combination), and is
the one already implemented by
``jcm.physics.echam.echam_terms.default_radiation_parameters`` (unconditional
``0.7`` for ``aerosol_module='jam'``, independent of truncation); this module
only replaces that function's hardcoded 0.8 liquid factors with the
per-truncation table above so the three liquid factors follow ECHAM's table
too.
"""

from __future__ import annotations

from jcm.physics.resolution_defaults import resolution_defaults

__all__ = [
    "ECHAM_CLOUD_OPTICS_DEFAULTS",
    "JAM_ARG_ICE_INHOMOGENEITY_T63",
    "echam_cloud_optics_defaults",
]

#: ``mo_cloud_optics.f90::setup_cloud_optics`` l.114-134, ECHAM's four
#: truncations, keyed by ``RadiationParameters`` field name.
ECHAM_CLOUD_OPTICS_DEFAULTS: dict[int, dict[str, float]] = {
    31: dict(cloud_inhomogeneity_liquid=0.80,
             cloud_inhomogeneity_liquid_shallow=0.40,
             cloud_inhomogeneity_liquid_convective=0.40,
             cloud_inhomogeneity_ice=0.80),
    63: dict(cloud_inhomogeneity_liquid=0.80,
             cloud_inhomogeneity_liquid_shallow=0.40,
             cloud_inhomogeneity_liquid_convective=0.80,
             cloud_inhomogeneity_ice=0.80),
    127: dict(cloud_inhomogeneity_liquid=0.80,
              cloud_inhomogeneity_liquid_shallow=0.40,
              cloud_inhomogeneity_liquid_convective=0.80,
              cloud_inhomogeneity_ice=0.85),
    255: dict(cloud_inhomogeneity_liquid=0.80,
              cloud_inhomogeneity_liquid_shallow=0.40,
              cloud_inhomogeneity_liquid_convective=0.80,
              cloud_inhomogeneity_ice=0.85),
}

#: ECHAM-HAM's ``lcdnc_progn`` / ``ncd_activ == 2`` (ARG activation) ice
#: inhomogeneity override, defined by ECHAM at T63 only (both L31 and L47);
#: see the module docstring for why jcm holds it fixed at every truncation
#: rather than interpolating.
JAM_ARG_ICE_INHOMOGENEITY_T63: float = 0.7


def echam_cloud_optics_defaults(truncation: int | None) -> dict[str, float]:
    """Return the cloud-optics inhomogeneity factors for a truncation.

    Args:
        truncation: the triangular truncation (63 for T63), or ``None`` for a
            grid that is not spectral (which gets the T63 row, with a
            warning).

    Returns:
        ``{field: value}`` for ``cloud_inhomogeneity_liquid``,
        ``cloud_inhomogeneity_liquid_shallow``,
        ``cloud_inhomogeneity_liquid_convective`` and
        ``cloud_inhomogeneity_ice``, from :data:`ECHAM_CLOUD_OPTICS_DEFAULTS`.
        See the module docstring for the rule between and outside the
        tabulated truncations. Does NOT apply the JAM-ARG ice override; the
        caller (``default_radiation_parameters``) applies
        :data:`JAM_ARG_ICE_INHOMOGENEITY_T63` on top for that aerosol module.

    """
    return resolution_defaults(
        ECHAM_CLOUD_OPTICS_DEFAULTS, truncation, nearest=(), fallback=63,
        table_name="cloud-optics inhomogeneity (mo_cloud_optics.f90)")
