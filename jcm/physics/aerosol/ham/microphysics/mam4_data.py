"""MAM4 (4-mode modal) population, as a ``ModalAerosolSpec``.

This builds the population *shape* of the MAM4-MOM 4-mode configuration so
the harness has the real target geometry from day one; the κ-Köhler
``PlaceholderMicrophysics`` uses it, and the eventual MAM4-JAX core wrapper
(#490) will replace only the per-step microphysics, not this contract.

Provenance
----------
Mode order, per-mode species membership, σ_g and reference dry diameters
are physical/configuration constants from the E3SM/MAM4 Fortran sources
``modal_aero_data.F90`` (mode names, ``nspec_amode``, ``lspectype_amode``)
and ``rad_constituents.F90`` (``sigmag``, ``dgnum``/lo/hi), cross-checked
against the MAM4-JAX port ``mam4_jax/data.py`` @ commit
``ccd872b6592d0f1ffa7ee43e51f7952b999fd57b``. These are data, not code.
"""

from __future__ import annotations

from jcm.physics.aerosol.ham.population import AerosolMode, ModalAerosolSpec
from jcm.physics.aerosol.ham.species import SPECIES

#: Upstream MAM4-JAX commit the constants below were cross-checked against.
MAM4_JAX_COMMIT = "ccd872b6592d0f1ffa7ee43e51f7952b999fd57b"

# Per-mode species membership, decoded from ``lspectype_amode`` (indices into
# SPECNAME_AMODE) into our canonical tokens. Order matches the Fortran slots.
#   accum (7):  sulfate, p-organic, s-organic, black-c, dust, seasalt, m-organic
#   aitken (4): sulfate, s-organic, seasalt, m-organic
#   coarse (7): dust, seasalt, sulfate, black-c, p-organic, s-organic, m-organic
#   pcarbon(3): p-organic, black-c, m-organic
_ACCUM_SPECIES = ("so4", "poa", "soa", "bc", "du", "ss", "moa")
_AITKEN_SPECIES = ("so4", "soa", "ss", "moa")
_COARSE_SPECIES = ("du", "ss", "so4", "bc", "poa", "soa", "moa")
_PCARBON_SPECIES = ("poa", "bc", "moa")

MAM4_MODES: tuple[AerosolMode, ...] = (
    AerosolMode(
        name="accum", short="acc", geom_std_dev=1.800,
        dgnum=0.1100e-6, dgnum_lo=0.0535e-6, dgnum_hi=0.4400e-6,
        species=_ACCUM_SPECIES,
        soluble=True, can_activate=True, sediments=True,
    ),
    AerosolMode(
        name="aitken", short="ait", geom_std_dev=1.600,
        dgnum=0.0260e-6, dgnum_lo=0.0087e-6, dgnum_hi=0.0520e-6,
        species=_AITKEN_SPECIES,
        soluble=True, can_activate=True, sediments=True,
    ),
    AerosolMode(
        name="coarse", short="cor", geom_std_dev=1.800,
        dgnum=2.000e-6, dgnum_lo=1.000e-6, dgnum_hi=4.000e-6,
        species=_COARSE_SPECIES,
        soluble=True, can_activate=True, sediments=True,
    ),
    AerosolMode(
        name="primary_carbon", short="pcm", geom_std_dev=1.600,
        dgnum=0.0500e-6, dgnum_lo=0.0100e-6, dgnum_hi=0.1000e-6,
        species=_PCARBON_SPECIES,
        # Freshly emitted primary carbon is hydrophobic until it ages into
        # the accumulation mode; not a CCN source on its own.
        soluble=False, can_activate=False, sediments=True,
    ),
)

# Only the species actually carried by some mode (drop unused nitrate/ammonium
# which MAM4-MOM does not prognose per-mode in this config, plus water).
_USED = tuple(sorted({s for m in MAM4_MODES for s in m.species}))
MAM4_SPECIES = tuple(SPECIES[t] for t in _USED) + (SPECIES["h2o"],)

#: The MAM4 4-mode population.
MAM4_SPEC = ModalAerosolSpec(
    modes=MAM4_MODES,
    species=MAM4_SPECIES,
    family="modal",
)
