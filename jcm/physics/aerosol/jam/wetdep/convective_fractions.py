"""HAMMOZ's convective in-droplet fractions mapped onto the JAM modes.

ECHAM-HAM scavenges every aerosol mode in convective cloud with a per-mode
fraction of the aerosol assumed to sit in the condensate, ``csr_conv``
(``mo_ham_m7ctl.f90``: ``(/0.20, 0.60, 0.99, 0.99, 0.20, 0.40, 0.40/)`` for
the M7 classes NS, KS, AS, CS, KI, AI, CI), used for both phases and at all
temperatures (``mo_ham_wetdep.f90::get_icscavfrac``, convective branch) and
for number and mass alike. JAM carries MAM4's four modes, so each MAM4 mode
takes the value of the M7 class it corresponds to. M7's class boundaries are
dry radii of 5 nm, 50 nm and 500 nm (diameters 10 nm, 100 nm, 1 µm).

=================  ======================  ========  ==========================
JAM (MAM4) mode    M7 class                csr_conv  reason
=================  ======================  ========  ==========================
accumulation       AS (soluble accum.)     0.99      direct: d 53-440 nm,
                                                     soluble
coarse             CS (soluble coarse)     0.99      direct: d 1-4 µm, soluble
aitken             KS (soluble Aitken)     0.60      direct: d 9-52 nm, soluble
primary carbon     KI (insoluble Aitken)   0.20      insoluble, d 10-100 nm
                                                     (median 50 nm): the M7
                                                     Aitken size range; AI's
                                                     0.40 belongs to an
                                                     accumulation-sized
                                                     insoluble mode
=================  ======================  ========  ==========================

MAM4 carries dust (and sea salt) internally mixed in the soluble
accumulation and coarse modes, so dust takes 0.99 here, where HAMMOZ's
insoluble dust classes AI/CI would give 0.40.
"""

from __future__ import annotations

#: Convective in-droplet fraction per JAM mode name, from HAMMOZ's
#: ``csr_conv`` of the corresponding M7 class (see the module docstring).
HAM_CSR_CONV: dict[str, tuple[str, float]] = {
    "accum": ("AS", 0.99),
    "coarse": ("CS", 0.99),
    "aitken": ("KS", 0.60),
    "primary_carbon": ("KI", 0.20),
}


def convective_csr(mode_name: str) -> float:
    """Return HAMMOZ's convective in-droplet fraction for a JAM mode."""
    try:
        return HAM_CSR_CONV[mode_name][1]
    except KeyError:
        raise KeyError(
            f"No HAMMOZ csr_conv mapping for JAM mode {mode_name!r}; add it "
            "to HAM_CSR_CONV with the M7 class it corresponds to."
        ) from None
