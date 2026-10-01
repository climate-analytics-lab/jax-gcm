"""Tests of the radiation parameter defaults (``radiation_types.py``)."""

import pytest

from jcm.physics.radiation.radiation_types import (
    CLOUD_OVERLAP_EXPONENTIAL,
    CLOUD_OVERLAP_MAXIMUM_RANDOM,
    CLOUD_OVERLAP_RANDOM,
    RadiationParameters,
    cloud_overlap_name,
)


def test_default_cloud_overlap_is_echams_maximum_random():
    """ECHAM6.3's ``i_overlap = 1`` (mo_radiation_parameters.f90 l.71)."""
    code = int(RadiationParameters.default().cloud_overlap)
    assert code == CLOUD_OVERLAP_MAXIMUM_RANDOM
    assert cloud_overlap_name(code) == "maximum_random"


def test_overlap_codes_name_the_three_rules():
    assert [cloud_overlap_name(c) for c in (
        CLOUD_OVERLAP_RANDOM, CLOUD_OVERLAP_MAXIMUM_RANDOM,
        CLOUD_OVERLAP_EXPONENTIAL)] == ["random", "maximum_random",
                                        "exponential"]
    with pytest.raises(ValueError, match="Unknown cloud_overlap"):
        cloud_overlap_name(3)
