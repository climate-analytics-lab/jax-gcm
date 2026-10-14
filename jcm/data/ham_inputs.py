"""Where the ECHAM-HAM M7 preset's nucleation inputs come from (jax-gcm#1017).

``nucleation_scheme=2`` (Kazil-Lovejoy ion-mediated nucleation driven by GCR
ionisation) reads two HAMMOZ inputs: the PARNUC table and the O'Brien
solar-minimum/maximum ion-pair tables. ``HAM_INPUT_DIR``, when set, is a
directory holding them under any of the names their readers accept.
Otherwise they come from the data mirror's Tier A ``products/ham_input/``,
which the mirror's ``ham-input`` stage stages verbatim under the names below.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable naming a local directory of HAM input files.
HAM_INPUT_ENV = "HAM_INPUT_DIR"

#: The Tier A product directory on the data mirror.
HAM_INPUT_PRODUCT = "ham_input"

#: The PARNUC names the M7 adapter accepts, in the order it tries them:
#: ``ham_nucl_initialize`` opens the first, the distributed file is the second.
KAZIL_TABLE_NAMES = ("parnuc.15H2SO4.nc", "parnuc.15H2SO4.A0.total.nc")

#: Published name -> the names a HAMMOZ input tree may carry it under, in the
#: order the stage tries them. Every published name is one the readers accept
#: (:data:`KAZIL_TABLE_NAMES`, ``gcr_ionisation``'s
#: ``_SOLMIN_NAMES``/``_SOLMAX_NAMES``): PARNUC under its distribution name,
#: the GCR tables under ECHAM's own.
HAM_INPUT_FILES: dict[str, tuple[str, ...]] = {
    "parnuc.15H2SO4.A0.total.nc": ("parnuc.15H2SO4.A0.total.nc",
                                   "parnuc.15H2SO4.nc"),
    "gcr_ipr_solmin.txt": ("gcr_ipr_solmin.txt", "solmin.txt", "SOLMIN.txt"),
    "gcr_ipr_solmax.txt": ("gcr_ipr_solmax.txt", "solmax.txt", "SOLMAX.txt"),
}

#: Where those files sit in an ECHAM-HAM emissions tree
#: (``sites.Site.ham_emissions``): its HAMMOZ T63 input directory.
HAM_INPUT_TREE_DIR = "hammoz/T63/hammoz"


def ham_input_dir(fetch=None) -> Path:
    """Return a directory holding the ``nucleation_scheme=2`` inputs.

    ``HAM_INPUT_DIR`` if set, else the directory the mirror's
    ``products/ham_input/`` files are fetched into (all at one revision, so
    one snapshot directory). ``fetch`` defaults to
    :func:`jcm.data.remote.fetch`. Raises ``FileNotFoundError`` naming both
    routes if neither is available.
    """
    env = os.environ.get(HAM_INPUT_ENV)
    if env:
        return Path(env)
    if fetch is None:
        from jcm.data.remote import fetch
    try:
        dirs = {Path(fetch(f"products/{HAM_INPUT_PRODUCT}/{name}")).parent
                for name in HAM_INPUT_FILES}
    except Exception as exc:
        raise FileNotFoundError(
            "nucleation_scheme=2 (Kazil-Lovejoy + GCR ionisation) needs the "
            "PARNUC and O'Brien GCR tables: set HAM_INPUT_DIR to a directory "
            "holding them, or fetch the data mirror's products/ham_input/ "
            f"(on a node with internet, or into a warm cache): {exc}") from exc
    if len(dirs) != 1:
        raise FileNotFoundError(
            f"products/{HAM_INPUT_PRODUCT}/ resolved to several directories "
            f"{sorted(map(str, dirs))}; fetch them at one mirror revision.")
    return dirs.pop()
