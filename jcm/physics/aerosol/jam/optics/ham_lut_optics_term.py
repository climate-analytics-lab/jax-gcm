"""``HamLutOpticsTerm``: HAM-faithful nearest-neighbour Mie-table optics (#1017).

Implements the ``_mode_optics``/``_build_mie_lut`` seam of
:class:`~jcm.physics.aerosol.jam.optics.optics_term.JamOpticsTerm`
(``docs/source/design/jam_optics_mode_seam.md``) with the ECHAM-HAM M7
lookup-table pathway: per mode, a nearest-neighbour read of one of HAM's own
four Mie tables (:mod:`ham_mie_tables`) at this mode's volume-mixed
refractive index and size parameter, rather than the default's on-the-fly
Gauss-Hermite quadrature over jcm's own Bohren-Huffman LUT.

What this backend reuses from the base class, unchanged: the volume mixing
of species (+ hygroscopic water) into one effective refractive index per
mode (``inputs.m_n``/``inputs.m_k``), via JCM'S OWN per-species refractive
index tables (``refractive_index.py``). HAM's own per-species-per-band
refractive index table (``mo_ham_rad_data.f90:198-416``) is NOT used here
-- wiring that in (replacing ``refractive_index_at`` for this backend, or
teaching it HAM's 32-wavelength grid) is a separate, later step; this
backend answers "how does a HAM-style Mie *table lookup* turn a refractive
index and size parameter into optics", not "what is HAM's own refractive
index".
"""

from __future__ import annotations

import logging
import math
import os

import jax
import jax.numpy as jnp
from flax import nnx

from jcm.physics.aerosol.jam.optics.ham_mie_tables import (
    HamRadLUT,
    default_ham_mie_tables,
    ham_rad_fitplus,
    load_ham_mie_tables,
)
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm

logger = logging.getLogger(__name__)


class HamLutOpticsTerm(JamOpticsTerm):
    """HAM M7's own nearest-neighbour Mie-table optics, as a ``_mode_optics`` backend."""

    def __init__(self, *, tables_dir: str | os.PathLike | None = None,
                 tables: dict[str, HamRadLUT] | None = None, **kwargs):
        """Pick HAM's authentic tables, or jcm's own built fallback, now --

        outside any traced function. Held as module data (pytree leaves
        passed into the compiled step), never created lazily inside
        ``__call__``.

        Parameters
        ----------
        tables_dir
            Directory holding HAM's own ``lut_optical_properties_M7.nc`` /
            ``lut_optical_properties_lw_M7.nc``; defaults to the
            ``HAM_INPUT_DIR`` environment variable (see
            :func:`ham_mie_tables.load_ham_mie_tables`). Ignored when
            ``tables`` is given directly. When neither this nor
            ``HAM_INPUT_DIR`` resolves to a directory holding both files,
            this term falls back to :func:`ham_mie_tables.
            default_ham_mie_tables` (jcm's own built approximation -- see
            ``ham_mie_tables.py``'s module docstring for the measured
            differences) rather than raising: the authentic files are a
            nice-to-have, not a hard requirement, for a term that otherwise
            runs identically either way.
        tables
            Pre-built tables, bypassing both the file load and the fallback
            entirely. For tests that exercise this term's OWN logic (mode
            selection, gradients, the nucleation-mode gate) and want a
            specific, deterministic table regardless of ``HAM_INPUT_DIR``'s
            ambient state. No production config passes this.

        ``self.table_source`` records which path was used --
        ``"authentic"``, ``"jcm_built"`` or ``"explicit"`` -- logged once
        here and readable afterwards by tests or a preset's own reporting.

        """
        super().__init__(**kwargs)
        if tables is not None:
            self.table_source = "explicit"
        else:
            try:
                tables = load_ham_mie_tables(tables_dir)
                self.table_source = "authentic"
            except FileNotFoundError as exc:
                logger.info(
                    "HamLutOpticsTerm: HAM's authentic Mie LUT files are not "
                    "available (%s); falling back to jcm's own built "
                    "approximation (ham_mie_tables.default_ham_mie_tables; "
                    "see that module's docstring for the measured "
                    "differences against HAM's real tables).", exc)
                tables = default_ham_mie_tables()
                self.table_source = "jcm_built"
        logger.info("HamLutOpticsTerm: using %s Mie tables.", self.table_source)
        self._ham_tables = nnx.data({
            name: jax.tree_util.tree_map(jnp.asarray, lut)
            for name, lut in tables.items()})

    def _build_mie_lut(self):
        """Skip the default Gauss-Hermite LUT: unused by this backend."""
        return None

    def _mode_optics(self, inputs):
        """Per-mode (tau, tau_scat, tau_scat_g) via HAM's own table lookup.

        Mode activity (``mo_ham.f90``'s ``nrad`` default, ``:583-585``):
        the nucleation mode carries no optics at all, matching HAM's
        ``nrad(1)=0``; every other mode gets SW+LW. Gated on the mode's own
        identity -- ``self._spec.nucleation_mode`` (``inputs.mode.short``,
        static -- an ordinary branch on the seam's own static config), NOT
        ``inputs.mode_index == 0``: mode 0 is HAM's nucleation class only
        for an M7-shaped population (:data:`m7_data.M7_SPEC` sets
        ``nucleation_mode="ns"`` and lists it first); a population with no
        such class at all (every MAM4 population, including the default
        placeholder core) leaves ``nucleation_mode`` at its default
        ``None``, so this gate never fires and MAM4's own mode 0
        (accumulation) keeps its optics. See
        ``population.ModalAerosolSpec.nucleation_mode``'s docstring.
        Table pair (fine/coarse) selected by whichever of HAM's own
        sigma=1.59/2.0 the mode's actual ``geom_std_dev`` is closer to --
        exact for an M7-shaped population (whose modes ARE 1.59/2.0, see
        ``ham_lut_optics_term_test.py``'s ``m7_spec``), an approximation for
        any other (e.g. MAM4's 1.6/1.8).

        ``tau = N_column * Qext * pi * r^2 = num_per_area * q_norm *
        wavelength_m**2`` (``mo_ham_rad.f90:973-976,1070,1085,1219,1234`` --
        see ``ham_mie_tables``'s module docstring for the normalisation
        derivation); ``tau_scat = tau*ssa``, ``tau_scat_g = tau*ssa*g``, so
        the base class's extinction-/scattering-weighted aggregation across
        modes reproduces HAM's own tau-weighted SSA/g reconstruction
        (``mo_ham_rad.f90:1110-1124``) without this method doing it itself.
        """
        zeros = jnp.zeros_like(inputs.r_wet)
        if inputs.mode.short == self._spec.nucleation_mode:
            return zeros, zeros, zeros
        tables = self._ham_tables
        fine = abs(inputs.mode.geom_std_dev - 1.59) <= abs(inputs.mode.geom_std_dev - 2.00)
        if inputs.is_sw:
            lut = tables["sw_fine"] if fine else tables["sw_coarse"]
        else:
            lut = tables["lw_fine"] if fine else tables["lw_coarse"]

        x = 2.0 * math.pi * inputs.r_wet / inputs.wavelength_m
        q_norm, ssa, g = ham_rad_fitplus(lut, x, inputs.m_n, inputs.m_k)

        tau = inputs.num_per_area * q_norm * inputs.wavelength_m ** 2
        return tau, tau * ssa, tau * ssa * g
