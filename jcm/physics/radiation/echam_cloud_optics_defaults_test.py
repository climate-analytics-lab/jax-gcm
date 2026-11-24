"""The cloud-optics inhomogeneity factors' per-truncation defaults (#974).

Pins the table against ``mo_cloud_optics.f90::setup_cloud_optics`` (ECHAM6.3-
HAM2.3 r7492, l.114-134), the interpolation between its truncations, and the
precedence an ``echam_physics()`` run ends up with: an explicit
``RadiationParameters`` object or field override wins, then the JAM-ARG ice
override, then the per-truncation table.
"""

import pytest

from jcm.physics.radiation.echam_cloud_optics_defaults import (
    ECHAM_CLOUD_OPTICS_DEFAULTS,
    JAM_ARG_ICE_INHOMOGENEITY_T63,
    echam_cloud_optics_defaults,
)
from jcm.physics.radiation.radiation_types import RadiationParameters

#: ``mo_cloud_optics.f90`` l.114-134, transcribed independently of the module
#: under test, in the Fortran's ``(zinhoml1, zinhoml2, zinhoml3, zinhomi)``
#: order.
FORTRAN = {
    31: (0.80, 0.40, 0.40, 0.80),
    63: (0.80, 0.40, 0.80, 0.80),
    127: (0.80, 0.40, 0.80, 0.85),
    255: (0.80, 0.40, 0.80, 0.85),
}

FIELDS = ("cloud_inhomogeneity_liquid", "cloud_inhomogeneity_liquid_shallow",
          "cloud_inhomogeneity_liquid_convective", "cloud_inhomogeneity_ice")


@pytest.mark.parametrize("truncation", sorted(FORTRAN))
def test_table_is_the_fortran_table(truncation):
    got = ECHAM_CLOUD_OPTICS_DEFAULTS[truncation]
    assert tuple(got[f] for f in FIELDS) == FORTRAN[truncation]


@pytest.mark.parametrize("truncation", sorted(FORTRAN))
def test_defaults_at_tabulated_truncations_match_fortran(truncation):
    got = echam_cloud_optics_defaults(truncation)
    assert tuple(got[f] for f in FIELDS) == FORTRAN[truncation]


@pytest.mark.parametrize("truncation,expected", [
    (31, FORTRAN[31]), (63, FORTRAN[63]),
    (106, (0.80, 0.40, 0.80, 0.8 + (106 - 63) / (127 - 63) * 0.05)),
    (127, FORTRAN[127]), (255, FORTRAN[255]),
])
def test_radiation_parameters_default_matches_the_table(truncation, expected):
    """``RadiationParameters.default(truncation=...)`` reads the table."""
    p = RadiationParameters.default(truncation=truncation)
    got = tuple(float(getattr(p, f)) for f in FIELDS)
    assert got == pytest.approx(expected, rel=1e-6)
    assert p.defaults_truncation == truncation


def test_t106_interpolates_between_t63_and_t127():
    """Only the ice factor moves between T63 and T127; T106 is 43/64 along."""
    t63 = echam_cloud_optics_defaults(63)
    t127 = echam_cloud_optics_defaults(127)
    t106 = echam_cloud_optics_defaults(106)
    weight = (106 - 63) / (127 - 63)
    for field in FIELDS:
        expected = t63[field] + weight * (t127[field] - t63[field])
        assert t106[field] == pytest.approx(expected, rel=1e-9)
    assert t106["cloud_inhomogeneity_ice"] > t63["cloud_inhomogeneity_ice"]
    assert t106["cloud_inhomogeneity_ice"] < t127["cloud_inhomogeneity_ice"]


def test_non_spectral_grid_takes_the_t63_row():
    got = echam_cloud_optics_defaults(None)
    assert tuple(got[f] for f in FIELDS) == FORTRAN[63]


def test_explicit_override_survives():
    """A field override wins over the per-truncation table at any truncation."""
    p = RadiationParameters.default(truncation=127, cloud_inhomogeneity_ice=0.42)
    assert float(p.cloud_inhomogeneity_ice) == pytest.approx(0.42)
    # The other three factors still come from the T127 row, unaffected.
    assert float(p.cloud_inhomogeneity_liquid) == pytest.approx(0.80)
    assert float(p.cloud_inhomogeneity_liquid_shallow) == pytest.approx(0.40)
    assert float(p.cloud_inhomogeneity_liquid_convective) == pytest.approx(0.80)


def test_jam_arg_ice_override_value():
    """ECHAM-HAM's T63-only JAM-ARG override, as the Fortran has it."""
    assert JAM_ARG_ICE_INHOMOGENEITY_T63 == pytest.approx(0.7)
