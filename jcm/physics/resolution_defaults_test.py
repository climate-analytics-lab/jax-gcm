"""The generic resolution-defaults mechanism (``resolution_defaults.py``)."""

import pytest

from jcm.physics import resolution_defaults as rd

TABLE = {10: {"a": 1.0, "n": 1}, 20: {"a": 3.0, "n": 2}, 40: {"a": 7.0, "n": 5}}


@pytest.fixture(autouse=True)
def _fresh_warnings():
    rd._WARNED.clear()
    yield
    rd._WARNED.clear()


def _defaults(truncation):
    return rd.resolution_defaults(TABLE, truncation, nearest=("n",),
                                  fallback=20, table_name="toy")


def test_tabulated_rows_are_exact():
    for t, row in TABLE.items():
        assert _defaults(t) == row


def test_linear_between_rows_and_nearest_for_integers():
    assert _defaults(15) == {"a": 2.0, "n": 2}       # tie -> the finer row
    assert _defaults(14) == {"a": 1.8, "n": 1}
    assert _defaults(35) == {"a": 6.0, "n": 5}


def test_outside_and_non_spectral_warn():
    with pytest.warns(UserWarning, match="T5 is outside"):
        assert _defaults(5) == TABLE[10]
    with pytest.warns(UserWarning, match="no spectral truncation"):
        assert _defaults(None) == TABLE[20]


class _WithDefaults:
    @classmethod
    def default(cls, *, truncation=63):
        return ("grid", truncation)


class _WithoutDefaults:
    @classmethod
    def default(cls):
        return "plain"


class _Term:
    def __init__(self, params=None, *, params_are_defaults=False):
        pass


class _OtherTerm:
    def __init__(self, params=None):
        pass


def test_default_parameters_dispatch():
    assert rd.default_parameters(_WithDefaults, 127) == ("grid", 127)
    assert rd.default_parameters(_WithoutDefaults, 127) == "plain"
    assert rd.has_resolution_defaults(_WithDefaults)
    assert not rd.has_resolution_defaults(_WithoutDefaults)


def test_defaults_flag_kwargs():
    assert rd.defaults_flag_kwargs(_Term, True) == {"params_are_defaults": True}
    assert rd.defaults_flag_kwargs(_OtherTerm, True) == {}
