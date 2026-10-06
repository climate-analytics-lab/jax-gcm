"""HAM literals ``_aqueous_so4`` (``chemistry/aqueous.py``) can override.

Split out of ``aqueous.py`` into its own, dependency-light module purely so
``microphysics/m7_data.py`` can import :data:`HAM_AQUEOUS_CONSTANTS` without
pulling in ``aqueous.py``'s own import of ``cloud_borne_store`` — which
itself imports ``microphysics.mam4_data``, and so (via the ``microphysics``
package's ``__init__.py`` also importing ``m7_data``) would close a circular
import back onto this module. No behaviour lives here; see ``aqueous.py``
for how these constants are used.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AqueousConstants:
    """Every HAM literal ``_aqueous_so4`` would otherwise read from jcm's own
    constants/species tables, for a population that wants r7492's own
    numbers instead (M7; jax-gcm#1017 task 3, the jax-gcm#1031 defect).
    ``None`` (every MAM4 population, via ``ModalAerosolSpec.aqueous_constants``)
    keeps today's values — for ``zrgas``/``avo``/``avo_xtoc`` that means
    calling ``aqueous._zrgas``/reading ``c.avogadro``/calling
    ``aqueous._avo_xtoc`` dynamically, exactly as before, so the
    anti-staleness behaviour (#772) is unaffected by this object's existence.

    Every field is a number ``mo_ham_chemistry.f90``/``mo_ham_species.f90``/
    ``mo_ham.f90``/``mo_physical_constants.f90`` hardcodes or registers
    DIFFERENTLY from jcm's own value:

    * ``h_so2_0``/``h_so2_act``: SO2 Henry's law (H0 [mol/l/atm], activation
      [K]) — ``speclist(id_so2)%henry``, ``mo_ham_species.f90:181`` (the
      jax-gcm#1031 defect: jcm's own ``_H_SO2_0``/``_H_SO2_ACT`` predate a
      correction HAM made here).
    * ``zrgas``: gas constant [l·atm/(mol·K)] — a ROUNDED literal
      (``mo_ham_chemistry.f90:209``, ``8.2e-2``), not derived from R* the
      way jcm's own ``aqueous._zrgas`` is.
    * ``avo``: Avogadro's number [1/mol] — ``mo_physical_constants.f90:58``
      (``6.02214179e23``), independently sourced from jcm's own
      ``c.avogadro`` (``r_universal/ak``), which does not quite equal it.
    * ``avo_xtoc``: the ``xtoc``/``ctox`` molec·cm⁻³↔mmr factor — a
      SEPARATELY rounded literal (``6.022e+20`` inside ``xtoc``/``ctox``
      themselves), not ``avo·1e-3`` even at full precision.
    * ``mw_so2``: SO2 molar mass [g/mol] — ``mo_ham.f90:310`` (``64.0643``),
      a different rounding from jcm's own ``GAS_SPECIES["so2"]`` value
      (``64.0648``); also the molar mass the SO2↔SO4 mass-conversion ratio
      (``AqueousSulfur._conv_so2_so4``/``_conv_so4_so2``) is built from when
      this object is set.
    """

    h_so2_0: float
    h_so2_act: float
    zrgas: float
    avo: float
    avo_xtoc: float
    mw_so2: float


#: r7492's own values for every field above (``mo_ham_species.f90:181``,
#: ``mo_ham_chemistry.f90:209`` and its ``xtoc``/``ctox`` literal,
#: ``mo_physical_constants.f90:58``, ``mo_ham.f90:310``).
HAM_AQUEOUS_CONSTANTS = AqueousConstants(
    h_so2_0=1.36, h_so2_act=4250.0,
    zrgas=8.2e-2,
    avo=6.02214179e23,
    avo_xtoc=6.022e20,
    mw_so2=64.0643,
)
