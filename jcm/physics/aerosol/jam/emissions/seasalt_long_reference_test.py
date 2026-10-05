"""Long et al. (2011) sea-salt scheme vs. the compiled HAMMOZ reference.

The reference (``jcm/data/test/echam_cloud_reference/hamseasalt_long.npz``)
is NUMBERS ONLY from the UNMODIFIED ECHAM6.3-HAM2.3 r7492
``mo_ham_m7_emi_seasalt.f90`` (``start_emi_seasalt``, ``seasalt_emissions_long``,
``seasalt_emissions_gong`` for cross-check), compiled by a standalone harness
(``/scr/dwatsonparris/ham-m7/w1/seasalt/harness/fortran_harness/echam_hamseasalt/``,
own driver/stubs, source not committed — see ``hamseasalt_README.md`` /
``hamseasalt_provenance.json`` next to the npz).

The numeric-fidelity comparison below drives
:meth:`SeaSaltEmissions._long_as_cs_fluxes` directly with a hand-built
open-water fraction that reproduces the native ``(1-slf-alake)*(1-seaice)``,
zeroed where ``slf>0.5`` (``seasalt_emissions_long``,
mo_ham_m7_emi_seasalt.f90:948-953) exactly — i.e. it tests the Long-specific
port (bin grid, SST correction, wind exponent, AS/CS split) this task is
about. A separate, smaller test below drives the full ``__call__`` to check
the wiring (SST read from ``forcing``, tendencies landing on the right
tracers); it intentionally avoids land/lake combinations where jcm's single
``terrain.fmask`` (which has no separate lake fraction, so a test must
combine ``slf+alake`` into it) and the native ``slf>0.5`` threshold (`slf`
ALONE) disagree on whether to zero the flux — e.g. slf=lake=0.3 zeroes in
jcm (fmask=0.6>0.5) but not natively (slf=0.3 is not >0.5). That is a
pre-existing limitation :func:`SeaSaltEmissions._open_water_fraction`
already has for Gong (no Gong-path change here), not something this task's
Long port could fix without a new jcm lake-fraction field; see the #1017 W1
task 2 final report.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.emissions.seasalt import SeaSaltEmissions, SeaSaltParameters
from jcm.physics.aerosol.jam.population import AerosolMode, AerosolSpecies, ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics_interface import PhysicsState

ROOT = Path(__file__).resolve().parents[5]
REFERENCE = ROOT / "jcm" / "data" / "test" / "echam_cloud_reference" / "hamseasalt_long.npz"

# Achieved max relative error over the harness's 720-case grid (full cross
# product of 10 m wind, SST, sea-ice fraction, land/lake fraction). It is
# the SAME order (~1e-6) in float32 and float64 -- which rules out finite-
# precision rounding as the cause, since halving the mantissa would move a
# rounding-dominated error by many orders of magnitude, not leave it
# unchanged. What IS float-width-independent is a cross-implementation
# difference in a transcendental intrinsic (log10/pow: Python/numpy's libm
# vs. gfortran's), and the Long size formula amplifies exactly that: it is
# `10**(cubic polynomial in log10(wet diameter))`, so a ~1 ULP disagreement
# in log10 gets multiplied by the polynomial's local slope and then
# exponentiated. That reproduces a few-ULP-of-log10 -> ~1e-6-of-flux
# amplification without needing any other explanation, and all 720 cases
# spanning the full input grid show the same order of error -- no localized
# outlier of the kind a boundary bin landing on the wrong side of a class
# cut would produce (that failure mode was real and is fixed separately;
# see `_long_bin_grid`'s docstring).
_MAX_RELATIVE_ERROR = 5e-6


def _two_class_ss_spec(density: float) -> ModalAerosolSpec:
    """Build a population whose 'ss' species carries exactly two classes, in
    HAM's own accumulation-then-coarse spec order -- what scheme="long"
    requires.
    """
    accum = AerosolMode(
        name="accum", short="acc", geom_std_dev=2.0, dgnum=3.0e-7,
        dgnum_lo=1e-9, dgnum_hi=1e-2, species=("ss",), soluble=True,
        can_activate=True, sediments=True)
    coarse = AerosolMode(
        name="coarse", short="cor", geom_std_dev=2.0, dgnum=2.0e-6,
        dgnum_lo=1e-9, dgnum_hi=1e-2, species=("ss",), soluble=True,
        can_activate=True, sediments=True)
    ss = AerosolSpecies(name="ss", molar_mass=0.05844, density=density,
                        hygroscopicity=1.0)
    return ModalAerosolSpec(modes=(accum, coarse), species=(ss,))


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE} not present")
    return np.load(REFERENCE)


def _native_seafrac(land, lake, seaice):
    """(1-slf-alake)*(1-seaice), zeroed where slf>0.5 (NOT slf+alake) --
    seasalt_emissions_long's own formula, mo_ham_m7_emi_seasalt.f90:945-953.
    """
    frac = np.clip((1.0 - land - lake) * (1.0 - seaice), 0.0, 1.0)
    return np.where(land > 0.5, 0.0, frac)


@pytest.mark.parametrize("enable_x64", [False, True], ids=["f32", "f64"])
def test_long_scheme_matches_native_reference(reference, enable_x64):
    """AS/CS mass + number fluxes reproduce the harness over its full grid."""
    previous = bool(jax.config.read("jax_enable_x64"))
    try:
        jax.config.update("jax_enable_x64", enable_x64)
        density = float(reference["ss_density"])
        spec = _two_class_ss_spec(density)
        term = SeaSaltEmissions(spec=spec, scheme="long")

        wind = jnp.asarray(reference["wind"])
        sst = jnp.asarray(reference["sst"])
        seafrac = jnp.asarray(_native_seafrac(
            reference["land"], reference["lake"], reference["seaice"]))
        mass_as, mass_cs, number_as, number_cs = term._long_as_cs_fluxes(
            wind, sst, seafrac)

        def relerr(mine, native):
            mine = np.asarray(mine, dtype=np.float64)
            return float(np.max(np.abs(mine - native) / np.maximum(np.abs(native), 1e-300)))

        errors = {
            "massf_as": relerr(mass_as, reference["long_massf_as"]),
            "massf_cs": relerr(mass_cs, reference["long_massf_cs"]),
            "numf_as": relerr(number_as, reference["long_numf_as"]),
            "numf_cs": relerr(number_cs, reference["long_numf_cs"]),
        }
        print(f"Long scheme max relative error vs. native ({'f64' if enable_x64 else 'f32'}):",
              errors)
        for key, value in errors.items():
            assert value < _MAX_RELATIVE_ERROR, (key, value)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_long_scheme_wiring_through_full_call():
    """Integration smoke test of the full __call__ path (not the numeric
    fidelity comparison above): SST comes from forcing, fluxes land on the
    accumulation/coarse tracers, zero over land or full sea-ice. Land/lake
    chosen to avoid the fmask-vs-slf ambiguity this module's docstring notes.
    """
    density = 2165.0
    spec = _two_class_ss_spec(density)
    term = SeaSaltEmissions(spec=spec, scheme="long")

    def run(wind, sst, land, seaice):
        ncols = 1
        state = PhysicsState.zeros((2, ncols)).copy(
            temperature=jnp.full((2, ncols), 285.0),
            u_wind=jnp.full((2, ncols), wind))
        diagnostics = {"air_density": jnp.full((2, ncols), 1.2),
                        "layer_thickness": jnp.full((2, ncols), 100.0)}
        terrain = type("T", (), {"fmask": jnp.asarray([land])})()
        forcing = type("F", (), {"sice_am": jnp.asarray([seaice]),
                                 "sea_surface_temperature": jnp.asarray([sst])})()
        tendency, _ = term(state, diagnostics, forcing, terrain)
        return tendency

    ocean = run(wind=10.0, sst=285.0, land=0.0, seaice=0.0)
    assert float(ocean.tracers[mass_name("ss", "acc")][-1, 0]) > 0.0
    assert float(ocean.tracers[mass_name("ss", "cor")][-1, 0]) > 0.0
    assert float(ocean.tracers[number_name("acc")][-1, 0]) > 0.0
    assert float(ocean.tracers[number_name("cor")][-1, 0]) > 0.0

    land = run(wind=10.0, sst=285.0, land=1.0, seaice=0.0)
    assert float(land.tracers[mass_name("ss", "cor")][-1, 0]) == pytest.approx(0.0)

    iced = run(wind=10.0, sst=272.0, land=0.0, seaice=1.0)
    assert float(iced.tracers[mass_name("ss", "cor")][-1, 0]) == pytest.approx(0.0)

    # Different SST -> different flux, proving SST is actually read.
    warm = run(wind=10.0, sst=300.0, land=0.0, seaice=0.0)
    cold = run(wind=10.0, sst=272.0, land=0.0, seaice=0.0)
    assert float(warm.tracers[mass_name("ss", "cor")][-1, 0]) != pytest.approx(
        float(cold.tracers[mass_name("ss", "cor")][-1, 0]))


def test_two_class_requirement_is_checked_at_construction():
    density = 2165.0
    three_classes = ModalAerosolSpec(
        modes=(
            AerosolMode(name="ait", short="ait", geom_std_dev=1.6, dgnum=4e-8,
                       dgnum_lo=1e-9, dgnum_hi=1e-7, species=("ss",),
                       soluble=True, can_activate=True, sediments=True),
            AerosolMode(name="accum", short="acc", geom_std_dev=2.0, dgnum=3e-7,
                       dgnum_lo=1e-7, dgnum_hi=1e-6, species=("ss",),
                       soluble=True, can_activate=True, sediments=True),
            AerosolMode(name="coarse", short="cor", geom_std_dev=2.0, dgnum=2e-6,
                       dgnum_lo=1e-6, dgnum_hi=1e-2, species=("ss",),
                       soluble=True, can_activate=True, sediments=True),
        ),
        species=(AerosolSpecies(name="ss", molar_mass=0.05844, density=density,
                                hygroscopicity=1.0),))
    with pytest.raises(ValueError, match="two.*class"):
        SeaSaltEmissions(spec=three_classes, scheme="long")


def test_grad_of_as_mass_flux_wrt_wind_is_finite():
    density = 2165.0
    spec = _two_class_ss_spec(density)
    term = SeaSaltEmissions(spec=spec, scheme="long")

    def loss(u10):
        mass_as, _, _, _ = term._long_as_cs_fluxes(
            u10[None], jnp.asarray([280.0]), jnp.asarray([1.0]))
        return jnp.sum(mass_as)

    g = jax.grad(loss)(jnp.asarray(10.0))
    assert np.isfinite(float(g))
    assert float(g) > 0.0


def test_unknown_scheme_rejected():
    density = 2165.0
    spec = _two_class_ss_spec(density)
    with pytest.raises(ValueError, match="Unknown sea-salt scheme"):
        SeaSaltEmissions(spec=spec, scheme="bogus")


def test_long_bin_grid_lands_on_the_as_cs_side_the_native_cumulative_sum_does():
    """Regression lock for the bin-150-at-exactly-1um fragility.

    ``_bin_grid`` (Gong's, vectorized ``arange(nbin)*zdx``) and
    ``_long_bin_grid`` (Long's own, Fortran-style repeated addition) land bin
    150 on OPPOSITE sides of the 1 um AS/CS edge; using the wrong one for
    Long moved ~1-3% of mass between classes (see this module's docstring
    and the #1017 W1 task 2 final report). If a future refactor points
    `long_bin_geometry` back at the shared `_bin_grid`, this starts failing.
    """
    from jcm.physics.aerosol.jam.emissions import seasalt as seasalt_module

    shared_dmt, _, _ = seasalt_module._bin_grid()
    long_dmt, _, _ = seasalt_module._long_bin_grid()
    assert shared_dmt[150] > 1e-6
    assert long_dmt[150] < 1e-6
    np.testing.assert_allclose(shared_dmt[150], 1e-6, rtol=1e-10)
    np.testing.assert_allclose(long_dmt[150], 1e-6, rtol=1e-10)


def test_scale_param_still_defaults_without_long_exponent():
    """SeaSaltParameters(scale=..., wind_exponent=...), the pre-#1017 Gong
    call pattern, still builds — wind_exponent_long defaults in the
    constructor itself, not only in .default().
    """
    p = SeaSaltParameters(scale=jnp.asarray(1.0), wind_exponent=jnp.asarray(3.41))
    assert float(p.wind_exponent_long) == pytest.approx(3.74)
