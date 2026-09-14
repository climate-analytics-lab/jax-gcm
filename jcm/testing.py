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
central difference is only meaningful while the secant stays inside one piece.
The distance to the nearest branch varies with the direction, and when the
secant straddles one the reference is not noisy but *wrong by* ``jump/eps``: at
the step the longwave check used to be tuned to, its reference was 800x the
true derivative and grew tenfold for each tenfold shrink of the step.
``_finite_difference`` picks the step by a criterion that never looks at the AD
value: halve the step until two successive secants agree, which is exactly the
statement that both lie in the same smooth piece.

**One projection hides a lost gradient.** Contracting the whole output tree onto
one random cotangent lets a large-magnitude leaf mask a small one — these ``f``
return a ``PhysicsData`` spanning ``fsol`` at O(1e3) down to tendencies at
O(1e-5), so a gradient zeroed on any small field would not move the inner
product. ``_cotangent`` therefore scales each leaf's random cotangent by the
inverse RMS of that leaf's **primal** value, which weights every output field
comparably. The scale comes from the primal, not from either derivative, so it
cannot launder a wrong gradient into agreement.

The checks stay in float32: enabling x64 mid-suite is what issue #729's
``conftest`` pinning exists to prevent, and the arguments these tests build are
float32, so ``jcm.utils.convert_back`` would quietly drop every perturbation to
a float32 leaf if the default float type were promoted underneath it.
"""

import dataclasses
import zlib

import jax
import jax.numpy as jnp
import numpy as np

# A halving ladder, so each rung's half-step is the next rung and every
# difference is computed once. It spans the range a float32 central difference
# can resolve: above ~1e-3 the secant's quadratic truncation error shows, and
# below ~1e-6 a perturbation is under an ulp of the O(100) radiative fields and
# the difference is pure rounding.
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


def random_direction(tree, seed=0):
    """Build a standard-normal tangent for ``tree``, keyed on leaf names.

    Non-float leaves get JAX's ``float0`` empty tangent: perturbing an integer
    rounds away in the primal, which would desynchronise the difference from AD,
    and ``jax.jvp`` rejects any other tangent dtype for them.

    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    names = _leaf_names(tree)
    out = []
    for name, leaf in zip(names, leaves):
        dtype = jnp.result_type(leaf)
        if jnp.issubdtype(dtype, jnp.floating):
            out.append(_normal(name, jnp.shape(leaf), dtype, seed))
        else:
            out.append(np.zeros(jnp.shape(leaf), dtype=jax.dtypes.float0))
    return jax.tree_util.tree_unflatten(treedef, out)


def _is_differentiable(leaf):
    return jnp.issubdtype(jnp.result_type(leaf), jnp.floating)


def _cotangent(primal, seed):
    """Random cotangent, scaled per leaf so every output field weighs alike.

    Without the scaling a single O(1e3) leaf dominates the projection and a
    gradient zeroed on an O(1e-5) leaf moves it by nothing. The scale is the
    leaf's primal RMS, which is independent of both derivatives.

    """
    leaves, treedef = jax.tree_util.tree_flatten(primal)
    names = _leaf_names(primal)
    out = []
    for name, leaf in zip(names, leaves):
        dtype = jnp.result_type(leaf)
        if not _is_differentiable(leaf):
            out.append(np.zeros(jnp.shape(leaf), dtype=jax.dtypes.float0))
            continue
        rms = float(jnp.sqrt(jnp.mean(jnp.asarray(leaf, jnp.float32)**2)))
        scale = 1.0 / rms if rms > 1e-30 else 1.0
        out.append(_normal(name, jnp.shape(leaf), dtype, seed) * scale)
    return jax.tree_util.tree_unflatten(treedef, out)


def _inner_prod(xs, ys):
    """Sum of elementwise products over the differentiable leaves of two trees.

    Inlined rather than taken from ``jax._src.public_test_util``: this module is
    importable at runtime, and a private JAX path would make ``import
    jcm.testing`` a JAX-upgrade hazard.
    """
    total = 0.0
    for x, y in zip(jax.tree.leaves(xs), jax.tree.leaves(ys)):
        if _is_differentiable(x) and _is_differentiable(y):
            total += float(jnp.vdot(jnp.asarray(x, jnp.float32).ravel(),
                                    jnp.asarray(y, jnp.float32).ravel()))
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
        # spread; accept it on the caller's absolute tolerance instead.
        if max(abs(coarse), abs(central)) <= atol:
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


def check_gradients(f, args, *, rtol=None, atol=0.0, steps=DEFAULT_STEPS,
                    seed=0, reference="difference", adjoint_rtol=1e-4):
    """Check ``f``'s jvp and vjp at ``args`` in one random direction.

    Args:
        f: function to differentiate, called as ``f(*args)``.
        args: tuple of arguments; any pytree.
        rtol: relative tolerance between the AD and finite-difference
            directional derivatives, and the convergence the reference itself
            must reach. Required unless ``reference="adjoint"``.
        atol: absolute tolerance, for a projection that is legitimately ~0.
        steps: candidate finite-difference steps, largest first, each half the
            one before.
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

    Raises:
        AssertionError: if the gradients disagree, or if ``reference`` is
            ``"difference"`` and no step in ``steps`` yields a self-consistent
            reference.

    """
    if reference not in ("difference", "adjoint"):
        raise ValueError(f"unknown reference {reference!r}")
    if reference == "difference" and rtol is None:
        raise ValueError("rtol is required when comparing against a difference")

    tangent = random_direction(args, seed=seed)
    primal_out, vjp_fun = jax.vjp(f, *args)
    _, jvp_out = jax.jvp(f, args, tangent)
    cotangent = _cotangent(primal_out, seed + 1)

    forward = _inner_prod(jvp_out, cotangent)
    reverse = _inner_prod(tangent, vjp_fun(cotangent))

    # jvp and vjp are each other's adjoint exactly, so they get their own tight
    # tolerance. Cheap insurance rather than a strong check: JAX derives both
    # from the same rules, so this only bites on a hand-written pair.
    np.testing.assert_allclose(
        forward, reverse, rtol=adjoint_rtol,
        err_msg="jvp and vjp disagree with each other")

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
