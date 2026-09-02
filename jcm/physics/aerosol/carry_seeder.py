"""``AerosolCarrySeeder`` — owner of the shared ``aerosol`` diagnostic slot.

Radiation reads its aerosol optics from the scheme-agnostic
:class:`~jcm.physics.aerosol.aerosol_types.AerosolData` struct threaded under
the ``"aerosol"`` diagnostics key (a hard ``requires`` on every radiation
backend), and the 2M microphysics reads ``aerosol.Nccn``. Exactly one term
must therefore *own* that carry slot — seed it at the right band shape on the
first step and reset it to a well-defined zero base every step, so the optics
terms that run afterwards only have to ``.copy(...)`` their fields in.

In the MACv2-SP path :class:`~jcm.physics.aerosol.macv2_sp.Macv2SpAerosol` is
that owner (it rebuilds the plume optics from scratch each step). In the JAM
path (jax-gcm#640) MACv2-SP is gone, so this minimal seeder takes the role:
it carries no physics, only the all-zero base. With ``jam_optics=True`` the
:class:`~jcm.physics.aerosol.jam.optics.optics_term.JamOpticsTerm` overwrites
the optics fields; with ``jam_optics=False`` the base is left untouched, which
makes the aerosol **radiatively passive** (all-zero optics) — a clean A/B
control for the aerosol direct effect rather than the old MACv2-SP leak-in.
"""

from __future__ import annotations

from typing import ClassVar

from jcm.physics.aerosol.aerosol_types import AerosolData
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm
from jcm.terrain import TerrainData


class AerosolCarrySeeder(PhysicsTerm):
    """Seed and reset the shared ``aerosol`` slot with an all-zero base.

    Sizes the per-band optics arrays to the active radiation backend via
    :meth:`cache_band_config` (mirroring :class:`Macv2SpAerosol`) so the
    cross-step carry agrees with what the optics terms write back.
    """

    name: ClassVar[str] = "aerosol_carry_seeder"
    category: ClassVar[str] = "aerosol"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("aerosol",)
    carry_slots: ClassVar[dict[str, type]] = {"aerosol": AerosolData}

    def __init__(self):
        """Start at the standard RRTMGP band counts (see ``cache_band_config``)."""
        # SW/LW band counts — overridden by ``cache_band_config`` once
        # ``ComposablePhysics`` knows the active radiation backend. Defaults
        # cover the standard RRTMGP gas-optics files for standalone use.
        self._n_bnd_sw: int = 14
        self._n_bnd_lw: int = 16

    def cache_band_config(self, band_config) -> None:
        """Capture SW/LW band counts so the carry slot has the right shape."""
        self._n_bnd_sw = len(band_config.sw_band_centers_nm)
        self._n_bnd_lw = len(band_config.lw_band_centers_nm)

    def initial_carry_state(self, coords):
        """Zero-fill the aerosol carry slot at the active SW/LW band counts."""
        ncols = (
            coords.horizontal.nodal_shape[0]
            * coords.horizontal.nodal_shape[1]
        )
        nlev = coords.nodal_shape[0]
        return {
            "aerosol": AerosolData.zeros(
                (ncols,), nlev,
                n_bnd_sw=self._n_bnd_sw, n_bnd_lw=self._n_bnd_lw,
            )
        }

    def __call__(
        self,
        state,
        diagnostics: dict,
        forcing,
        terrain: TerrainData,
    ):
        """Reset ``aerosol`` to an all-zero base for this step."""
        nlev, ncols = state.temperature.shape

        # Match the band shape of the active radiation backend, exactly as
        # ``Macv2SpAerosol`` does, so the slot the optics terms overwrite has
        # the shape they expect. Falls back to grey broadband defaults for a
        # caller constructing the term outside ``ComposablePhysics``.
        band_config = diagnostics.get("_band_config")
        if band_config is None:
            from jcm.physics.radiation.band_config import RadiationBandConfig
            band_config = RadiationBandConfig.broadband()
        n_bnd_sw = len(band_config.sw_band_centers_nm)
        n_bnd_lw = len(band_config.lw_band_centers_nm)

        base = AerosolData.zeros(
            (ncols,), nlev, n_bnd_sw=n_bnd_sw, n_bnd_lw=n_bnd_lw,
        )
        zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
        return zero_tendencies, {**diagnostics, "aerosol": base}
