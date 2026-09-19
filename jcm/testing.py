"""Gradient checks that do not depend on the shape of the pytree they are given.

``jax.test_util.check_vjp``/``check_jvp`` are the natural tools for the
AD-vs-finite-difference checks the SPEEDY schemes carry, but three of their
properties make them brittle here.

**The perturbation direction is a function of the whole pytree.** They draw it
from a single ``np.random.RandomState(0)`` streamed over the leaves in tree
order, so every leaf's direction depends on how many leaves precede it. Adding
one field to ``SWRadiationData`` therefore re-rolled the direction for every
test that passes a ``PhysicsData``, and turned four checks red in longwave,
shortwave and surface-flux code that the new field never touches.
``random_direction`` seeds a generator from each leaf's **field name path**
instead, so a leaf's direction depends only on where it sits by name: adding or
removing a sibling leaves every other direction bit-identical.

Names, not positions, is the whole point, and it needs care: ``tree_math.struct``
flattens through ``astuple()`` and registers with
``register_pytree_node_class``, i.e. *without* key support, so JAX's own
``tree_flatten_with_path`` reports its children as ``[<flat index 3>]`` — still
positional. ``_leaf_names`` walks the tree against the objects themselves and
recovers the declared field name from ``dataclasses.fields``.

**The finite-difference step is fixed.** SPEEDY's schemes are piecewise smooth —
``jnp.where`` on level masks, ``jnp.clip`` in the band-fraction lookup — so a
central difference is only meaningful while the secant stays inside one piece,
and the distance to the nearest branch varies with the direction.
``_finite_difference`` chooses the step by two criteria, neither of which reads
the AD value, because the two ways a secant can go wrong look nothing alike:

* across a *jump* the reference is not noisy but *wrong by* ``jump/eps`` — at
  the step the longwave check used to be tuned to, its reference was 800x the
  true derivative and grew tenfold for each tenfold shrink of the step. Halving
  the step doubles that, so successive rungs disagree. Agreement between
  neighbours is necessary but **not** sufficient: convection's difference sits
  on a plateau near 5.7e7 from 1e-3 down to 4e-6, neighbouring rungs agreeing
  to 0.3%, and only below 2e-6 does the secant clear the branch and drop onto
  the true -2.6e6. So the ladder is walked out rather than stopped at the first
  locally-consistent rung.
* across a *kink* none of that fires at all. A central difference there
  converges, stably and at every rung, to the *mean* of the two one-sided
  derivatives, which is not what AD computes. Only comparing the one-sided
  secants with each other sees it: they differ by O(1) across a kink and
  converge together, as O(eps*f''), on a smooth piece.

**One projection hides a lost gradient — on the output side.** Contracting the
whole output tree onto one random cotangent lets a large-magnitude leaf mask a
small one — these ``f`` return a ``PhysicsData`` spanning ``fsol`` at O(1e3)
down to tendencies at O(1e-5), so a gradient zeroed on any small field would not
move the inner product. ``_cotangent`` therefore scales each leaf's random
cotangent by the inverse RMS of that leaf's **primal** value, which weights
every output field comparably.

**A unit-scale tangent loses a large input outright.** The input side has the
mirror of that problem and a sharper one: the perturbation is not merely
mis-weighted, it can vanish. With a standard-normal direction and an *absolute*
step, a leaf whose values are large is shifted by less than one of its own
float32 ulps and the shifted argument rounds back to the unshifted one bit for
bit — SPEEDY's dry static energy is ``se ~ 3.1e5``, whose ulp is ~0.03, thirty
times the *largest* rung on the ladder. The secant then says nothing whatever
about that leaf, and a ``stop_gradient`` on it would leave the check green.
``_tangent`` therefore scales each input leaf's direction **by** that leaf's
primal RMS, which turns the ladder into one of *relative* steps: rung ``eps``
moves every leaf by a fraction ``eps`` of its own magnitude, resolvable in
float32 whatever that magnitude is, and every leaf contributes to the projection
on comparable terms. Both scalings read only the primal, never either
derivative, so neither can launder a wrong gradient into agreement.

The scale is the RMS and not the standard deviation, though the RMS of an
offset-dominated field — a temperature in kelvin, a dry static energy — is
essentially its offset and so gives a step much larger than the field's own
structure. The standard deviation would size the step to that structure, but it
is *zero* for the uniform fields these fixtures are built from
(``PhysicsData.ones()``), which is exactly the ``se`` case this scaling exists
for: a uniform 3.1e5 would fall back to the absolute step and stay invisible.
Resolvability has to win, and the price is paid at the callers — a step that is
a fraction of 300 K rather than of the few kelvin a column varies by reaches
branch boundaries that a smaller step would not, and where it does the honest
answer is that no central difference exists there.

The checks stay in float32: enabling x64 mid-suite is what issue #729's
``conftest`` pinning exists to prevent, and the arguments these tests build are
float32, so ``jcm.utils.convert_back`` would quietly drop every perturbation to
a float32 leaf if the default float type were promoted underneath it. The one
place float64 appears is ``_inner_prod``, which reduces already-computed float32
arrays on the host; nothing JAX traces is promoted by it.
"""

import dataclasses
import zlib

import jax
import jax.numpy as jnp
import numpy as np

# A halving ladder of *relative* steps: rung ``eps`` displaces each input leaf
# by a fraction ``eps`` of that leaf's own RMS (``_tangent``), so one ladder
# serves an O(1e-5) tendency and an O(1e5) dry static energy alike. Halving, so
# each rung's half-step is the next rung and every difference is computed once.
#
# The span is what a float32 central difference can carry, and both ends are now
# set in units of the leaf rather than in absolute ones. At the top, 1e-3 is a
# 0.1% displacement, where the central secant's quadratic truncation error
# O(eps^2 f'''/f') is ~1e-6 relative — small, and the first thing to grow if the
# ladder started higher. At the bottom, 1e-3*0.5^10 ~ 9.8e-7 is still 8 to 16
# float32 ulps of *any* leaf (float32 spacing is 2^-24 to 2^-23 of the value
# depending on where the mantissa sits), so the shifted primal differs from the
# unshifted one in several bits and the secant is not pure rounding. The ladder
# is unchanged from the absolute-step version because that lower bound was
# always the binding one; making the step relative is what makes it hold for
# every leaf instead of only for the O(100) radiative fields it was derived for.
# Extending it downwards would buy nothing: the rounding floor is relative too,
# so no leaf is better resolved by a smaller rung.
DEFAULT_STEPS = tuple(1e-3 * 0.5**k for k in range(11))


def _child_fields(node):
    """Return ``node``'s fields when they map one-to-one onto its children.

    ``tree_math.struct`` flattens through ``astuple()``, so field order *is*
    child order and the name is recoverable. A dataclass with a static field
    (flax's ``pytree_node=False``, and the positionally-registered nodes in
    ``predictions.py`` and ``band_config.py``) flattens to fewer children than
    it declares, and pairing them off by position would name a real leaf after
    a static field — silently reintroducing the dependence on unrelated fields
    that this module exists to remove. Fall back to positional names there.
    """
    if not dataclasses.is_dataclass(node):
        return None
    fields = dataclasses.fields(node)
    children = jax.tree_util.tree_flatten(
        node, is_leaf=lambda x: x is not node)[0]
    return fields if len(children) == len(fields) else None


def _entry_name(node, entry):
    """Name one step of a tree path, recovering field names where JAX cannot.

    ``tree_math.struct`` registers without key support, so JAX reports its
    children positionally; the declared field order is exactly the flatten
    order (``tree_flatten`` returns ``astuple()``), so the name is recoverable.
    """
    fields = _child_fields(node)
    if isinstance(entry, jax.tree_util.FlattenedIndexKey):
        if fields is not None and entry.key < len(fields):
            return fields[entry.key].name
        return f"#{entry.key}"
    return str(entry)


def _child(node, entry):
    """Descend one path step, or return None when the node is opaque."""
    if isinstance(entry, jax.tree_util.FlattenedIndexKey):
        fields = _child_fields(node)
        if fields is not None and entry.key < len(fields):
            return getattr(node, fields[entry.key].name, None)
        return None
    if isinstance(entry, jax.tree_util.DictKey):
        return node[entry.key] if isinstance(node, dict) else None
    if isinstance(entry, jax.tree_util.SequenceKey):
        try:
            return node[entry.idx]
        except (TypeError, IndexError, KeyError):
            return None
    if isinstance(entry, jax.tree_util.GetAttrKey):
        return getattr(node, entry.name, None)
    return None


def _leaf_names(tree):
    """Names for ``tree``'s leaves in flatten order, by field name where possible."""
    names = []
    for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]:
        node, parts = tree, []
        for entry in path:
            parts.append(_entry_name(node, entry))
            node = _child(node, entry)
        names.append("/".join(parts))
    _check_unique(names)
    return names


def _check_unique(names):
    """Reject a name collision, which would correlate two leaves' directions.

    Unreachable through dataclass, dict and sequence nodes — field names are
    identifiers, dict keys stringify bracketed, and the positional fallback
    carries its index — but a custom pytree that flattens to more children than
    it declares fields could still collide, and silently sharing a direction
    would weaken every check built on it.
    """
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"leaf names are not unique, so directions would be correlated: "
            f"{duplicates}")


def _normal(name, shape, dtype, seed):
    key = zlib.crc32(name.encode()) ^ (seed & 0xFFFFFFFF)
    return jnp.asarray(np.random.default_rng(key).standard_normal(shape), dtype)


def _is_differentiable(leaf):
    return jnp.issubdtype(jnp.result_type(leaf), jnp.floating)


def _scaled_direction(tree, seed, scale_from_rms):
    """Per-leaf normal direction, weighted by a function of the leaf's primal RMS.

    The RMS is taken in float32 whatever the leaf's dtype, so the weight a leaf
    gets does not depend on the precision it happens to be stored in.

    A leaf whose RMS is ~0 has no magnitude to be relative to, and its primal
    says nothing about the scale ``f`` is sensitive on — an identically-zero
    cloud-water field in a fixture could be a variable ``f`` responds to
    steeply or not at all. There is therefore no better scale knowable than
    1.0, which for a tangent is the plain absolute step (and at 0 the
    perturbation *is* the whole value, so it is trivially resolvable) and for a
    cotangent is an unweighted contribution to the projection.

    An *empty* leaf — ``ForcingData`` carries a zero-length ozone climatology
    whenever none is loaded, which is every ECHAM fixture — is that same case
    and takes the same 1.0, but it is read off ``size`` rather than from a mean
    over no elements: ``jnp.mean`` of an empty array is NaN, and although NaN
    fails the ``> 1e-30`` test and so lands on 1.0 anyway, computing it at all
    aborts the whole check under ``jax_debug_nans`` — the flag a caller reaches
    for the moment one of these checks reports a non-finite gradient.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    names = _leaf_names(tree)
    out = []
    for name, leaf in zip(names, leaves):
        dtype = jnp.result_type(leaf)
        if not _is_differentiable(leaf):
            out.append(np.zeros(jnp.shape(leaf), dtype=jax.dtypes.float0))
            continue
        values = jnp.asarray(leaf, jnp.float32)
        rms = float(jnp.sqrt(jnp.mean(values**2))) if values.size else 0.0
        scale = scale_from_rms(rms) if rms > 1e-30 else 1.0
        out.append(_normal(name, jnp.shape(leaf), dtype, seed) * scale)
    return jax.tree_util.tree_unflatten(treedef, out)


def random_direction(tree, seed=0):
    """Build a standard-normal direction for ``tree``, keyed on leaf names.

    This is the *unscaled* building block, and stays that way: it knows only
    the tree's structure and dtypes, so it is the right thing to reason about
    when testing the naming and seeding invariants, and it is what both
    ``_tangent`` and ``_cotangent`` then weight by a leaf's primal magnitude.
    Folding either weighting in here would make one of the two wrong — they
    scale by the RMS and by its reciprocal — and would hide that the scale is a
    property of the *values*, not of the direction.

    Non-float leaves get JAX's ``float0`` empty tangent: perturbing an integer
    rounds away in the primal, which would desynchronise the difference from AD,
    and ``jax.jvp`` rejects any other tangent dtype for them.

    """
    return _scaled_direction(tree, seed, lambda rms: 1.0)


def _tangent(primal, seed):
    """Random tangent, scaled per leaf so the step is relative to the leaf.

    Scaling *up* by the leaf's RMS is what makes ``DEFAULT_STEPS`` a ladder of
    fractional displacements. Without it an absolute step of 1e-3 is below a
    float32 ulp of an O(1e5) leaf and the shifted argument is bit-identical to
    the original, so the finite difference carries no information about that
    leaf at all. The scale is the leaf's primal RMS, independent of both
    derivatives.

    The same object is handed to ``jax.jvp`` and used for the finite-difference
    shift, so automatic differentiation and the secant are taken along exactly
    the same direction; a scaling applied to only one of them would show up as
    a spurious disagreement.

    """
    return _scaled_direction(primal, seed, lambda rms: rms)


def _cotangent(primal, seed):
    """Random cotangent, scaled per leaf so every output field weighs alike.

    Without the scaling a single O(1e3) leaf dominates the projection and a
    gradient zeroed on an O(1e-5) leaf moves it by nothing. The scale is the
    inverse of the leaf's primal RMS, which is independent of both derivatives.

    """
    return _scaled_direction(primal, seed, lambda rms: 1.0 / rms)


def _inner_prod(xs, ys):
    """Sum of elementwise products over the differentiable leaves of two trees.

    The *contraction* is done in numpy float64 while the two trees stay float32:
    a per-leaf tangent scaled to that leaf's own RMS spans the schemes' whole
    dynamic range (an O(1e5) dry static energy beside an O(1e-5) tendency), so
    individual terms of this sum are far larger than the total and a float32
    accumulation loses several digits to cancellation. That showed up as jvp and
    vjp — the same double sum contracted in two different orders, and equal
    exactly in exact arithmetic — disagreeing by 1.2e-4 where float64
    accumulation puts them 1e-5 apart. This is a host-side reduction over
    already-computed float32 arrays, not ``jax_enable_x64``: every function
    evaluation and every derivative stays in float32, so issue #729's conftest
    pinning is untouched.

    Inlined rather than taken from ``jax._src.public_test_util``: this module is
    importable at runtime, and a private JAX path would make ``import
    jcm.testing`` a JAX-upgrade hazard.
    """
    total = 0.0
    for x, y in zip(jax.tree.leaves(xs), jax.tree.leaves(ys)):
        if _is_differentiable(x) and _is_differentiable(y):
            total += float(np.dot(np.asarray(x, np.float64).ravel(),
                                  np.asarray(y, np.float64).ravel()))
    return total


# Two independent ways a secant can be meaningless, and both have to be ruled
# out.
#
# A *jump* inside the step shows up as ``jump/eps``: halving the step doubles
# the apparent slope, so successive rungs disagree. Normalising by the larger
# of the pair puts a clean jump at spread 0.5, so the neighbour threshold has
# to stay well under that whatever the caller's rtol is.
#
# A *kink* (C0 but not C1 — SPEEDY is full of them, ``jnp.maximum`` and
# ``jnp.clip`` on Richardson numbers and cloud fractions) does not show up that
# way at all: a central difference across a kink converges, stably and at every
# rung, to the *mean* of the two one-sided derivatives, which is not what AD
# computes. Neighbour agreement therefore certifies nothing about a kink, and
# the one-sided secants have to be compared with each other as well: they
# differ by O(1) across a kink and converge together, as O(eps*f''), on a
# smooth piece.
MAX_NEIGHBOUR_SPREAD = 0.1
MAX_ONE_SIDED_SPREAD = 0.1

# Below this a rung is converged past any doubt and the walk can stop early.
CONVERGED_SPREAD = 1e-3


def _finite_difference(f, args, tangent, cotangent, steps, rtol, atol):
    """Project ``f``'s directional derivative, choosing the step by consistency.

    Walks the halving ladder and returns ``(projection, eps)`` for the first
    rung that is converged past doubt, or failing that the best-converged rung
    that still clears both thresholds. A rung qualifies only if its central
    secant agrees with the one at twice the step (no jump inside the step) *and*
    its two one-sided secants agree with each other (no kink). Neither test
    reads the automatic-differentiation value, so the step cannot be chosen to
    flatter it.

    Raises when no rung qualifies — the honest outcome for a direction along
    which ``f`` simply is not differentiable, and far more useful than
    comparing a gradient against ``jump/eps``.

    """
    if len(steps) < 2:
        raise ValueError("steps must hold at least two rungs to compare")

    base = f(*args)
    cache = {}

    def secants(eps):
        """(central, left, right) directional derivatives at this step."""
        if eps not in cache:
            shift = lambda s: jax.tree.map(
                lambda x, d: x + s * eps * d if _is_differentiable(x) else x,
                args, tangent)
            plus, minus = f(*shift(1.0)), f(*shift(-1.0))
            project = lambda a, b, scale: _inner_prod(
                jax.tree.map(lambda p, m: (p - m) * scale, a, b), cotangent)
            cache[eps] = (project(plus, minus, 0.5 / eps),
                          project(base, minus, 1.0 / eps),
                          project(plus, base, 1.0 / eps))
        return cache[eps]

    def spread(a, b):
        scale = max(abs(a), abs(b))
        return 0.0 if scale <= atol else abs(a - b) / scale

    tried, qualified = [], []
    for coarse_eps, fine_eps in zip(steps, steps[1:]):
        coarse = secants(coarse_eps)[0]
        central, left, right = secants(fine_eps)
        # A derivative that is legitimately ~0 gives no meaningful relative
        # spread; accept it on the caller's absolute tolerance instead. The
        # *one-sided* secants have to clear atol too, and that is the whole
        # point rather than belt and braces: at a symmetric kink (|x| at 0)
        # every central secant is exactly 0 however small the step gets, so
        # testing only the central pair would accept a zero reference — and
        # with it a wrongly-zero gradient — for a point where no derivative
        # exists. On a smooth zero the one-sided pair shrinks as O(eps*f'') and
        # drops under atol; at a kink it stays pinned at the one-sided slopes.
        if max(abs(coarse), abs(central), abs(left), abs(right)) <= atol:
            return central, fine_eps
        neighbour, one_sided = spread(coarse, central), spread(left, right)
        tried.append((fine_eps, coarse, central, left, right, neighbour, one_sided))
        if neighbour > MAX_NEIGHBOUR_SPREAD or one_sided > MAX_ONE_SIDED_SPREAD:
            continue
        # Only the neighbour spread measures the *central* estimate's error.
        # The one-sided spread is first order in eps even on a perfectly smooth
        # f (it is O(eps*f''/f')), so ranking rungs by it would reject sound
        # references for ordinary curvature; it earns its keep purely as the
        # gate above, where a kink pins it near 1 however small the step gets.
        if neighbour <= CONVERGED_SPREAD:
            return central, fine_eps
        qualified.append((neighbour, fine_eps, central))

    if qualified:
        worst, eps, central = min(qualified)
        if worst <= rtol:
            return central, eps

    report = "\n".join(
        f"    eps={e:<9.2e} D={c:<13.6g} D(2*eps)={co:<13.6g} "
        f"D-={le:<13.6g} D+={ri:<13.6g} neighbour={n:.3g} one_sided={o:.3g}"
        for e, co, c, le, ri, n, o in tried)
    raise AssertionError(
        "no finite-difference step is usable in this direction: no rung has "
        "both a stable neighbour (no jump inside the step) and agreeing "
        "one-sided secants (no kink at the point), so there is no reference to "
        "compare the gradient against.\n" + report)


def _match_leaves(wanted, names):
    """Resolve ``wanted`` to leaf indices, raising if it names none.

    A name matches any leaf whose path contains it as a whole run of segments,
    so ``"[1]/convection/se"``, ``"convection/se"`` and ``"se"`` all reach that
    leaf and an interior name like ``"speedy_coords"`` reaches every leaf of
    that subtree. Naming nothing raises, so a renamed field fails loudly rather
    than quietly checking — or freezing — nothing.
    """
    segments = wanted.split("/")
    matched = [i for i, n in enumerate(names)
               if any(n.split("/")[j:j + len(segments)] == segments
                      for j in range(len(n.split("/"))))]
    if not matched:
        raise ValueError(f"no input leaf named {wanted!r}; the leaves are {names}")
    return matched


def _freeze(tangent, args, fixed_inputs):
    """Zero the tangent on leaves the caller declares are not inputs.

    A *structural* leaf is one the model never differentiates with respect to
    and whose value selects a code path rather than scaling a result: the
    SPEEDY sigma-grid metrics (``speedy_coords``), from which the schemes build
    masks they document as compile-time constants — ``stratosphere_mask(fsg)``,
    ``hsg[k+1] > 0.5`` — and the smoothing widths, which sit at exactly 0 by
    default behind ``jnp.where(w > 0.0, smooth, hard)`` guards where a negative
    width is out of domain. Perturbing either crosses the selector, so no
    two-sided derivative exists along it and the whole direction is wasted:
    with the sigma grid free, ``speedy_longwave``'s check straddles the
    sigma < 0.2 stratosphere mask (``fsg`` has an entry at exactly 0.2) and
    reports a jump instead of a gradient.

    This is about the *domain* of the direction, never about convergence: a
    leaf belongs here only because differentiating with respect to it is
    meaningless, and freezing one to quiet a disagreement would hide exactly
    the defect this module exists to find. Freezing also removes the leaf from
    the reverse projection, since that contracts against this same tangent.
    """
    names = _leaf_names(args)
    frozen = {i for wanted in fixed_inputs for i in _match_leaves(wanted, names)}
    leaves, treedef = jax.tree_util.tree_flatten(tangent)
    return jax.tree_util.tree_unflatten(
        treedef,
        [jnp.zeros_like(x) if (i in frozen and _is_differentiable(x)) else x
         for i, x in enumerate(leaves)])


def _check_live_inputs(live_inputs, args, gradients):
    """Assert each named input leaf carries a finite, non-zero reverse gradient.

    The mirror, on the input side, of the per-output liveness guard the
    ``"adjoint"`` reference applies. Both exist because a single projection can
    only report that *something* moved: an input whose gradient has been lost to
    a ``stop_gradient``, an integer cast or a dropped term can hide behind its
    siblings in the sum. Naming the leaves that must be live turns that into an
    explicit per-leaf assertion, and it is the only input-side check available
    under ``reference="adjoint"``, where no difference is taken at all.

    Names are matched by ``_match_leaves``, and every matching leaf must be
    live.
    """
    names = _leaf_names(args)
    arg_leaves, grad_leaves = jax.tree.leaves(args), jax.tree.leaves(gradients)
    for wanted in live_inputs:
        for i in _match_leaves(wanted, names):
            name, arg, grad = names[i], arg_leaves[i], grad_leaves[i]
            if not _is_differentiable(arg):
                raise ValueError(
                    f"{name} is not a floating-point leaf, so it cannot carry "
                    f"a gradient")
            values = np.asarray(grad)
            assert np.all(np.isfinite(values)), f"{name}: gradient is not finite"
            assert np.any(values != 0.0), (
                f"{name}: gradient is identically zero, so nothing about this "
                f"input is being checked")


def check_gradients(f, args, *, rtol=None, atol=0.0, steps=DEFAULT_STEPS,
                    seed=0, reference="difference", adjoint_rtol=1e-4,
                    live_inputs=(), fixed_inputs=()):
    """Check ``f``'s jvp and vjp at ``args`` in one random direction.

    Args:
        f: function to differentiate, called as ``f(*args)``.
        args: tuple of arguments; any pytree.
        rtol: relative tolerance between the AD and finite-difference
            directional derivatives, and the convergence the reference itself
            must reach. Required unless ``reference="adjoint"``.
        atol: absolute tolerance, for a projection that is legitimately ~0. The
            projection is dimensionless by construction — each output leaf's
            cotangent is divided by that leaf's RMS and each input leaf's
            tangent multiplied by its own — so it measures the *fractional*
            response of the outputs to a fractional displacement of the inputs,
            and the same absolute value means the same thing from one scheme to
            the next.
        steps: candidate finite-difference steps, largest first, each half the
            one before. They are fractions of each input leaf's RMS, not
            absolute displacements; see ``DEFAULT_STEPS``.
        seed: mixes into the per-leaf seeding, to check a second direction.
        reference: ``"difference"`` compares AD against a central difference.
            ``"adjoint"`` drops that comparison and checks only that every
            differentiable output carries a finite, not-identically-zero
            gradient, and that the two AD modes are adjoint. That is
            deliberately weak — it is what remains for a function whose output
            is piecewise constant (a diagnosed level index, say), where a
            difference reports the staircase rather than the derivative. It
            still catches differentiability lost to a ``stop_gradient`` or an
            integer cast, which zeroes the gradient.
        adjoint_rtol: tolerance for jvp against vjp. Its own knob because the
            adjoint identity is exact up to float32 reduction order, far
            tighter than any comparison against a difference.
        live_inputs: names of input leaves that must each carry a finite,
            not-identically-zero reverse gradient, checked per leaf rather than
            through the projection. Opt-in, because which inputs a scheme is
            legitimately insensitive to at a given operating point is the
            caller's knowledge, not this function's.
        fixed_inputs: names of input leaves to hold fixed, because they are
            structural rather than differentiable inputs — see ``_freeze``. The
            justification is always that differentiating with respect to the
            leaf is meaningless, never that the check converges better without
            it.

    Raises:
        AssertionError: if the gradients disagree, if a ``live_inputs`` leaf is
            dead, or if ``reference`` is ``"difference"`` and no step in
            ``steps`` yields a self-consistent reference.
        ValueError: if a ``live_inputs`` or ``fixed_inputs`` name matches no
            input leaf.

    """
    if reference not in ("difference", "adjoint"):
        raise ValueError(f"unknown reference {reference!r}")
    if reference == "difference" and rtol is None:
        raise ValueError("rtol is required when comparing against a difference")

    # One tangent object for both AD and the secant, so they are taken along
    # exactly the same — per-leaf relative — direction.
    tangent = _tangent(args, seed)
    if fixed_inputs:
        tangent = _freeze(tangent, args, fixed_inputs)
    primal_out, vjp_fun = jax.vjp(f, *args)
    _, jvp_out = jax.jvp(f, args, tangent)
    cotangent = _cotangent(primal_out, seed + 1)

    input_grads = vjp_fun(cotangent)
    forward = _inner_prod(jvp_out, cotangent)
    reverse = _inner_prod(tangent, input_grads)

    # jvp and vjp are each other's adjoint exactly, so they get their own tight
    # tolerance. Cheap insurance rather than a strong check: JAX derives both
    # from the same rules, so this only bites on a hand-written pair.
    np.testing.assert_allclose(
        forward, reverse, rtol=adjoint_rtol,
        err_msg="jvp and vjp disagree with each other")

    # Before either reference: a named input is asserted live whether or not a
    # difference is available, and it is the projection's blind spot in both.
    if live_inputs:
        _check_live_inputs(live_inputs, args, input_grads)

    if reference == "adjoint":
        # Per output leaf, not on the reduced projection: a multi-output f
        # (diagnose_convection returns a level index *and* a moisture excess)
        # would otherwise have one live output satisfy the guard on behalf of
        # the dead one it was written to check.
        names = _leaf_names(primal_out)
        checked = 0
        for name, leaf in zip(names, jax.tree.leaves(jvp_out)):
            if not _is_differentiable(leaf):
                continue
            checked += 1
            values = np.asarray(leaf)
            assert np.all(np.isfinite(values)), f"{name}: gradient is not finite"
            assert np.any(values != 0.0), (
                f"{name}: gradient is identically zero, so nothing about this "
                f"output is being checked")
        assert checked, "no differentiable outputs to check"
        return

    value, eps = _finite_difference(
        f, args, tangent, cotangent, steps, rtol, atol)
    for name, ad in (("jvp", forward), ("vjp", reverse)):
        np.testing.assert_allclose(
            ad, value, rtol=rtol, atol=atol,
            err_msg=(f"{name} disagrees with the central difference at "
                     f"eps={eps:.2e}"))
