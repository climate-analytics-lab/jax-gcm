"""Audit: layout-agnostic terms must behave identically in both hosts.

``ComposablePhysics`` runs a term either on the dycore's nodal grid
``(nlev, nlon, nlat)`` or, with ``vectorize_columns=True``, on a flattened
``(nlev, nlon*nlat)`` block. Most ECHAM terms are written column-only and
say so loudly (they unpack ``nlev, ncols = field.shape``, which raises on a
3-D state). But a term used by *both* families — nudging, the sponge,
diagnostics — has to work either way, and nothing checked that.

That gap shipped a broken feature: ``NudgingTerm`` hard-coded
``inv_tau[:, None, None]`` and subtracted a nodal-shaped target straight
from the state, so ``nudging=era5`` died at the first step under every
ECHAM config while the SPEEDY-only unit tests stayed green (#617). The
static ``requires_audit_test`` could not have caught it either: the
offending code was in a module-level helper, not in ``__call__``.

So this audit is dynamic. It drives each layout-agnostic term twice from
one physical state — once shaped ``(nlev, nlon, nlat)``, once flattened —
and requires the tendencies to agree per column. It also refuses to let a
term go unclassified: every concrete :class:`PhysicsTerm` must appear in
exactly one of the two rosters below, so adding a term forces a decision
about which hosts it supports.
"""

import importlib
import pkgutil
import unittest
import warnings

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
import jcm.physics as _physics_pkg
from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData


def _import_all_physics():
    """Import every physics module so ``__subclasses__`` is complete."""
    for mod in pkgutil.walk_packages(
            _physics_pkg.__path__, _physics_pkg.__name__ + "."):
        if mod.name.endswith("_test"):
            continue
        try:
            importlib.import_module(mod.name)
        except Exception:  # optional extras (mam4, cosp) may be absent
            pass
    importlib.import_module("jcm.nudging")


def _all_terms():
    def subs(cls):
        for s in cls.__subclasses__():
            yield s
            yield from subs(s)
    _import_all_physics()
    return sorted({s for s in subs(PhysicsTerm)}, key=lambda x: x.__name__)


#: Terms that must run in BOTH hosts and give the same answer. A term
#: belongs here if any shipped configuration can place it in either — the
#: SPEEDY packages leave ``vectorize_columns`` False, every ECHAM package
#: sets it True, so anything usable with both families qualifies.
LAYOUT_AGNOSTIC = frozenset({
    "AerocomDiagnostics",
    "BettsMillerConvection",
    "FrontalGravityWaveDrag",
    "IceNucleation",
    "LottMillerSso",
    "Mam4JaxMicrophysics",
    "MoistAirColumnState",
    "NudgingTerm",
    "OmegaDiagnostic",
    "PlaceholderMicrophysics",
    "ResetEmissionFluxes",
    "SpeedyFlags",
    "StateSampler",
    "UpperSponge",
})

#: Terms this audit does not drive, and why. Two kinds:
#:
#:   * column-only by construction — they unpack ``nlev, ncols =
#:     field.shape`` and raise on a 3-D state, which is a loud, correct
#:     failure rather than a silent wrong answer;
#:   * not constructible from a synthetic environment here — they need
#:     rich upstream carry state (``_jam_state``, ``oxidants``,
#:     ``convection``), a trained emulator, or per-run tracer lists.
#:
#: Listed explicitly so coverage cannot shrink silently: the roster test
#: below fails if a term is in neither set.
NOT_AUDITED = frozenset({
    "AnthropogenicEmissions", "AqueousSulfur", "ArgActivation",
    "CloudBorneExchange", "CloudsatCosp", "ConvectiveTracerTransport",
    "DmsEmissions", "DustEmissions", "Echam1MMicrophysics",
    "EchamBoundaryConditions", "EchamSurface", "GreyTwoStreamRadiation",
    "HeldSuarez", "HinesGwd", "JamOpticsTerm", "Lohmann2MMicrophysics",
    "Macv2SpAerosol", "ModalMicrophysicsTerm", "NNEmulatorRadiation",
    "PreSpeciatedEmissions", "PrescribedOxidants", "RRTMGPRadiation",
    "SeaSaltEmissions", "SimpleChemistry", "SimpleGwd",
    "SlinnDryDeposition", "SpeedyClouds", "SpeedyConvection",
    "SpeedyDownwardLongwaveRadiation", "SpeedyForcing", "SpeedyHumidity",
    "SpeedyLargeScaleCondensation", "SpeedyShortwaveRadiation",
    "SpeedySurfaceFlux", "SpeedyTermBase", "SpeedyUpwardLongwaveRadiation",
    "SpeedyVerticalDiffusion", "StokesSedimentation", "SulfurGasChemistry",
    "SundqvistCloudFraction", "TiedtkeConvection", "TracerVerticalDiffusion",
    "TteTkeVerticalDiffusion", "UpperTemperatureRelaxation", "WetScavenging",
})


def _nudging_term():
    from jcm.nudging import NudgingConfig, NudgingTerm
    inv = 1.0 / (6 * 3600.0)
    return NudgingTerm(NudgingConfig(
        inv_tau_wind=inv * jnp.ones(_NLEV),
        inv_tau_temperature=0.5 * inv * jnp.ones(_NLEV),
    ))


#: Constructors for terms whose ``__init__`` needs arguments.
TERM_FACTORIES = {"NudgingTerm": _nudging_term}


def _attach_nudging_target(state, diagnostics, forcing, horiz):
    """Give ``NudgingTerm`` a target, else it short-circuits to zeros.

    Without this the audit passed with the #617 bug reinstated: the term
    returns a zero tendency when ``forcing.nudging_target`` is unset, so
    ``nudging_tendency`` — where the bug lived — never ran, and both hosts
    "agreed" on zeros. The target is deliberately built on the NODAL grid
    for both hosts, which is what production does: ERA5 targets are
    regridded to ``(nlev, nlon, nlat)`` regardless of how physics is
    vectorised.
    """
    from jcm.nudging import NudgingTarget

    rng = np.random.default_rng(0)
    grid = lambda: jnp.asarray(  # noqa: E731
        rng.normal(size=(_NLEV, _NLON, _NLAT)))
    target = NudgingTarget(
        u_wind=grid(), v_wind=grid(), temperature=grid() + 280.0)
    return state, diagnostics, forcing.copy(nudging_target=target)


#: Per-term environment tweaks that put a term on its ACTIVE path. A term
#: guarded by "no input -> zero tendency" would otherwise be compared on a
#: branch that exercises none of its arithmetic.
ENV_HOOKS = {"NudgingTerm": _attach_nudging_target}


_COORDS = get_speedy_coords(layers=8, spectral_truncation=21)
_NLEV = _COORDS.nodal_shape[0]
_NLON, _NLAT = _COORDS.horizontal.nodal_shape
_NCOLS = _NLON * _NLAT
_TERRAIN = TerrainData.aquaplanet(_COORDS)


def _carry_classes():
    out = {}
    for term in _all_terms():
        out.update(getattr(term, "carry_slots", {}))
    return out


def _build_env(horiz):
    """Build a plausible, non-degenerate environment on horizontal ``horiz``.

    Deliberately not all-zeros: a zero temperature column breaks downstream
    physics (the #470 lesson recorded in ``PhysicsTerm.initial_carry_state``),
    and a uniform field would hide an axis-ordering error by making every
    column identical.
    """
    def lev(profile):
        arr = jnp.asarray(profile).reshape((-1,) + (1,) * len(horiz))
        return jnp.broadcast_to(arr, (len(profile),) + horiz)

    p_full = lev(np.linspace(2e3, 9.5e4, _NLEV))
    p_half = lev(np.linspace(1e3, 1.0e5, _NLEV + 1))
    temperature = lev(np.linspace(220.0, 288.0, _NLEV))
    rho = p_full / (c.rd * temperature)
    z_half = lev(np.linspace(20000.0, 0.0, _NLEV + 1))

    state = PhysicsState.zeros(
        (_NLEV,) + horiz,
        temperature=temperature,
        specific_humidity=lev(np.geomspace(1e-6, 8e-3, _NLEV)),
        u_wind=lev(np.linspace(5.0, 20.0, _NLEV)),
        v_wind=lev(np.linspace(-3.0, 3.0, _NLEV)),
        normalized_surface_pressure=jnp.ones(horiz),
    )
    diagnostics = {
        "_dt_seconds": 900.0,
        "pressure_full": p_full,
        "pressure_half": p_half,
        "layer_thickness": (p_half[1:] - p_half[:-1]) / (rho * c.grav),
        "air_density": rho,
        "height_full": 0.5 * (z_half[1:] + z_half[:-1]),
        "height_half": z_half,
        "surface_pressure": jnp.full(horiz, 1.0e5),
    }
    for key, cls in _carry_classes().items():
        try:
            diagnostics[key] = cls.zeros(horiz, _NLEV)
        except Exception:
            pass
    return state, diagnostics, ForcingData.zeros(horiz)


def _instantiate(cls):
    term = TERM_FACTORIES.get(cls.__name__, cls)()
    try:
        term.cache_coords(_COORDS)
    except Exception:
        pass
    return term


def _comparable_arrays(tend, diagnostics):
    """Numeric leaves to compare: tendencies AND diagnostics the term wrote.

    Diagnostics matter because several audited terms (``StateSampler``,
    ``OmegaDiagnostic``, ``AerocomDiagnostics``) emit no tendency at all —
    comparing only tendencies would pass them on all-zero arrays.
    """
    out = _tendency_arrays(tend)
    for key, val in sorted(diagnostics.items()):
        if key.startswith("_"):
            continue
        for name, leaf in _numeric_leaves(val):
            out[f"diagnostics[{key}]{name}"] = leaf
    return out


def _numeric_leaves(obj, prefix=""):
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        arr = np.asarray(obj)
        if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 1:
            yield prefix, arr
        return
    fields = getattr(obj, "__dataclass_fields__", None) or getattr(
        obj, "_fields", None)
    if not fields:
        return
    for name in fields:
        yield from _numeric_leaves(getattr(obj, name, None), f"{prefix}.{name}")


def _tendency_arrays(tend):
    out = {}
    for name in ("u_wind", "v_wind", "temperature", "specific_humidity"):
        val = getattr(tend, name, None)
        if val is not None:
            out[name] = np.asarray(val)
    for name, val in (getattr(tend, "tracers", {}) or {}).items():
        out[f"tracers[{name}]"] = np.asarray(val)
    return out


def _flatten_horizontal(arr):
    """Fully ravel so the two hosts are comparable element-for-element.

    A full row-major ravel is the right comparison for EVERY rank here,
    because the column host is produced from the grid host by exactly that
    flatten: ``(nlev, nlon, nlat) -> (nlev, nlon*nlat)`` for level fields and
    ``(nlon, nlat) -> (nlon*nlat,)`` for surface fields. Collapsing only the
    trailing axes instead would mis-handle a horizontal-only field, whose
    leading axis is longitude rather than level.
    """
    return np.asarray(arr).ravel()


class LayoutAgnosticTermsTest(unittest.TestCase):

    def test_grid_and_column_hosts_agree(self):
        for cls in _all_terms():
            if cls.__name__ not in LAYOUT_AGNOSTIC:
                continue
            with self.subTest(term=cls.__name__):
                hook = ENV_HOOKS.get(cls.__name__, lambda *a: a[:3])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    env = _build_env((_NLON, _NLAT))
                    grid_tend, grid_diag = _instantiate(cls)(
                        *hook(*env, (_NLON, _NLAT)), _TERRAIN)
                    env = _build_env((_NCOLS,))
                    col_tend, col_diag = _instantiate(cls)(
                        *hook(*env, (_NCOLS,)), _TERRAIN)

                grid = _comparable_arrays(grid_tend, grid_diag)
                col = _comparable_arrays(col_tend, col_diag)
                self.assertEqual(
                    set(grid), set(col),
                    f"{cls.__name__} emits different tendency fields per host",
                )
                # Guard against a vacuous pass: if every compared array is
                # identically zero the two hosts "agree" without exercising
                # any of the term's arithmetic (see ENV_HOOKS).
                self.assertTrue(
                    any(np.any(v != 0) for v in grid.values()),
                    f"{cls.__name__} produced all-zero tendencies AND "
                    "diagnostics, so this comparison proves nothing — give it "
                    "an ENV_HOOKS entry that puts it on its active path.",
                )
                for key in sorted(grid):
                    a, b = _flatten_horizontal(grid[key]), _flatten_horizontal(col[key])
                    self.assertEqual(
                        a.shape, b.shape,
                        f"{cls.__name__}.{key}: {a.shape} (grid) vs {b.shape} (column)",
                    )
                    np.testing.assert_allclose(
                        np.nan_to_num(a), np.nan_to_num(b),
                        rtol=1e-5, atol=1e-9,
                        err_msg=f"{cls.__name__}.{key} differs between hosts",
                    )

    def test_every_term_is_classified(self):
        """No term may be silently unaudited.

        Adding a ``PhysicsTerm`` forces an explicit choice: either it is
        layout-agnostic (and gets checked above), or it is listed in
        ``NOT_AUDITED`` with the category comment explaining why.
        """
        names = {cls.__name__ for cls in _all_terms()}
        unclassified = sorted(names - LAYOUT_AGNOSTIC - NOT_AUDITED)
        self.assertFalse(
            unclassified,
            "new PhysicsTerm(s) are in neither roster in layout_audit_test.py: "
            f"{unclassified}. Add to LAYOUT_AGNOSTIC if any shipped config can "
            "run them on both the nodal grid and a column-vectorised block "
            "(then this audit checks them), else to NOT_AUDITED.",
        )
        stale = sorted((LAYOUT_AGNOSTIC | NOT_AUDITED) - names)
        self.assertFalse(stale, f"rosters name terms that no longer exist: {stale}")


if __name__ == "__main__":
    unittest.main()
