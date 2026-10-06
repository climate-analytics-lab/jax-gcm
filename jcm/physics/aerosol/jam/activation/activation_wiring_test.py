"""``activation_scheme`` / ``nactivpdf`` through the JAM and ECHAM factories (#1017)."""
import pytest

from jcm.physics.aerosol.jam.activation.arg_term import ArgActivation, ArgParameters
from jcm.physics.aerosol.jam.activation.ham_activation_term import (
    HamActivation,
    HamActivationParameters,
)
from jcm.physics.aerosol.jam.jam_terms import (
    activation_parameter_class,
    jam_aerosol_physics,
)


def _activation(terms):
    (term,) = [t for t in terms if t.category == "aerosol_activation"]
    return term


def test_default_is_cam_arg():
    term = _activation(jam_aerosol_physics(microphysics="placeholder"))
    assert isinstance(term, ArgActivation)
    assert activation_parameter_class("arg") is ArgParameters


@pytest.mark.parametrize("scheme,inner,nactivpdf", [
    ("ham_arg", "arg", 0), ("ham_arg", "arg", 1), ("ham_lin_leaitch", "lin_leaitch", 0)])
def test_ham_schemes_compose_on_m7(scheme, inner, nactivpdf):
    term = _activation(jam_aerosol_physics(
        microphysics="m7_placeholder", cloud_borne=False,
        activation_scheme=scheme, nactivpdf=nactivpdf))
    assert isinstance(term, HamActivation)
    assert term._scheme == inner and term._nactivpdf == nactivpdf
    assert activation_parameter_class(scheme) is HamActivationParameters


def test_invalid_combinations_are_refused():
    with pytest.raises(ValueError, match="nactivpdf"):
        jam_aerosol_physics(microphysics="placeholder", nactivpdf=1)
    with pytest.raises(ValueError, match="Lin & Leaitch"):
        jam_aerosol_physics(microphysics="m7_placeholder", cloud_borne=False,
                            activation_scheme="ham_lin_leaitch", nactivpdf=1)
    with pytest.raises(ValueError, match="activation_scheme"):
        jam_aerosol_physics(microphysics="placeholder", activation_scheme="nope")


def test_echam_factory_maps_activation_overrides_to_the_ham_class():
    from jcm.physics.echam.testing import idealized_echam_physics

    physics = idealized_echam_physics(
        aerosol_module="jam", cloud_scheme="2m", jam_microphysics="m7_placeholder",
        jam_cloud_borne=False, jam_activation_scheme="ham_arg", jam_nactivpdf=1,
        activation={"w_min": 0.2})
    (term,) = [t for t in physics.terms if t.category == "aerosol_activation"]
    assert isinstance(term, HamActivation)
    assert float(term.params.get_value().w_min) == pytest.approx(0.2)
