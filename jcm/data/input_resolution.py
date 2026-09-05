"""Typed resolution of a forcing/boundary input value into concrete local paths.

One layer replacing the stringly-typed mini-language that had accreted across
``jcm.runners`` / ``jcm.forcing`` / ``jcm.data.bundle_names`` / ``tools/benchmark``
(issue #751): ``auto`` sentinels, ``{year}`` patterns + coverage clamps, the
``hf://`` scheme, plain paths, lists of any of these, and ``null`` opt-outs. A
user value is parsed once into an :class:`InputSpec`, and :func:`resolve_input`
turns it into a :class:`ResolvedInput` carrying concrete local paths plus the
time-alignment kind and provenance — so every rule (publication gating, year
expansion, per-product coverage, merge compatibility, eager-fetch-loud-offline)
lives in one place and its interactions are reviewable in one diff.

Availability is data-driven: the mirror manifest (:mod:`jcm.data.mirror_manifest`)
is the single source of truth for which products exist on which grids/levels, so
``auto`` on a missing grid → ``None`` and coverage clamping are manifest lookups
rather than conventions spread through code.

Kept free of any intra-package (``jcm``) import at module top — stdlib +
dataclasses only — so it loads without initialising JAX; the manifest,
``fetch`` and year-expansion helpers are imported lazily (or injected), which
also lets ``tools/benchmark``'s pre-GPU enumerator drive it without importing
``jcm`` (see :mod:`jcm.data.mirror_manifest`).
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: ``ResolvedInput.alignment`` values. A product's manifest ``alignment`` maps
#: onto the time-series indexing mode the readers use: a climatology wraps within
#: the year (``WRAP_YEAR``), a transient series indexes by absolute date
#: (``BY_DATE``), a static field has no time axis. ``AUTO`` defers the choice to
#: the reader's span heuristic — used for an explicit path whose kind the
#: resolver cannot know without opening it.
STATIC = "static"
WRAP_YEAR = "wrap_year"
BY_DATE = "by_date"
AUTO = "auto"

_ALIGNMENT_FROM_MANIFEST = {
    "static": STATIC,
    "climatology": WRAP_YEAR,
    "transient": BY_DATE,
}

#: Values that mean "explicitly off" for any forcing key.
_NONE_SENTINELS = (None, "", "null", "none", "None")


class SpecKind(enum.Enum):
    """What a parsed user value denotes."""

    NONE = "none"          # null / "" / "none" -> opt out
    AUTO = "auto"          # follow the grid via the mirror manifest
    EXPLICIT = "explicit"  # concrete path(s) / hf:// URL(s) / {year} pattern(s)


@dataclass(frozen=True)
class InputSpec:
    """A parsed forcing value, before grid/manifest resolution."""

    key: str
    kind: SpecKind
    raw: object = None          # original value (EXPLICIT)
    is_list: bool = False       # raw was a list of products
    has_pattern: bool = False   # any element carries a {year} pattern

    @classmethod
    def parse(cls, key: str, value) -> "InputSpec":
        if isinstance(value, str) and value == "auto":
            return cls(key=key, kind=SpecKind.AUTO)
        if _is_none(value):
            return cls(key=key, kind=SpecKind.NONE)
        is_list = _is_seq(value)
        elems = list(value) if is_list else [value]
        if is_list and not any(not _is_none(e) for e in elems):
            return cls(key=key, kind=SpecKind.NONE)
        has_pattern = any(isinstance(e, str) and "{year}" in e for e in elems)
        return cls(key=key, kind=SpecKind.EXPLICIT, raw=value,
                   is_list=is_list, has_pattern=has_pattern)


@dataclass(frozen=True)
class ResolvedInput:
    """The outcome of resolving one forcing value.

    ``products`` preserves the per-product structure the merge rules need: each
    element is one product (a scalar path, or a list of files — a ``{year}``
    expansion or an explicit yearly set — opened together on one time axis).
    ``paths`` is the flat concrete-path list for consumers that open everything
    at once. ``is_none`` marks an opted-out / unpublished input (no paths).
    """

    key: str
    is_none: bool = False
    products: tuple = ()
    paths: tuple = ()
    alignment: str = AUTO
    source: str = ""                       # "auto:<product>" / "explicit" / "none"
    provenance: tuple = ()


# ---------------------------------------------------------------------------
# small value predicates
# ---------------------------------------------------------------------------

def _is_none(value) -> bool:
    return value in _NONE_SENTINELS


def _is_seq(value) -> bool:
    """Return whether ``value`` is a non-string, non-mapping iterable (a list)."""
    return (not isinstance(value, (str, bytes, Mapping))
            and isinstance(value, Iterable))


# ---------------------------------------------------------------------------
# year expansion + product splitting (delegates to jcm.data.yearly_files)
# ---------------------------------------------------------------------------

def expand_yearly(file_spec, years, available=None, *, expand=None):
    """Expand one ``{year}`` pattern (see ``jcm.data.yearly_files``).

    The science — pattern detection, the ``years`` requirement, the one-year
    by-date bracket, and the coverage clamp — lives in
    :func:`jcm.data.yearly_files.expand_yearly_files`; this only picks that up
    lazily (or an injected ``expand`` for a jcm-free caller) so the resolver
    stays a single home for the rule without re-implementing it.
    """
    fn = expand
    if fn is None:
        from jcm.data.yearly_files import expand_yearly_files as fn
    return fn(file_spec, years, available)


def forcing_products(file_spec, years, available=None, *, expand=None) -> list:
    """Split a spec into independent products, each year-expanded.

    A **list** value names several products (e.g. a biomass-burning file plus an
    anthropogenic one); each element is opened and time-aligned on its own, so a
    transient ``{year}`` product and a 12-month climatology in one list keep
    their distinct time axes instead of being outer-joined. A scalar is a single
    product; a scalar ``{year}`` pattern becomes that product's yearly-file list.
    Port of ``jcm.runners._forcing_products``.
    """
    if _is_seq(file_spec):
        return [expand_yearly(e, years, available, expand=expand)
                for e in file_spec]
    return [expand_yearly(file_spec, years, available, expand=expand)]


# ---------------------------------------------------------------------------
# merge compatibility (#750): products opened together must share one time axis
# ---------------------------------------------------------------------------

def product_time_axis(product, *, open_dataset=None):
    """One product's combined time axis, or ``None`` for a static field.

    Returns ``(dtype, key, label)`` — ``dtype`` is ``"datetime"`` or
    ``"integer-month"``; ``key`` is the hashable frozenset of distinct time
    values (the merge key: two products union cleanly under by-coords iff they
    share one ``(dtype, key)``); ``label`` is a human range for messages. Raises
    when a product's own files straddle both axis kinds. Port of
    ``jcm.runners._product_time_axis``; ``open_dataset`` is injectable for tests.
    """
    import numpy as np
    if open_dataset is None:
        import xarray as xr
        open_dataset = xr.open_dataset
    chunks: list = []
    dtypes: set = set()
    for p in product:
        ds = open_dataset(p)
        try:
            if "time" not in ds.variables and "time" not in ds.dims:
                continue
            vals = np.asarray(ds["time"].values).ravel()
        finally:
            ds.close()
        is_datetime = (np.issubdtype(vals.dtype, np.datetime64)
                       or vals.dtype == np.object_)
        dtypes.add("datetime" if is_datetime else "integer-month")
        chunks.append(vals)
    if not chunks:
        return None
    if len(dtypes) > 1:
        raise ValueError(
            "a single product mixes incompatible time axes — its files "
            f"straddle both an integer-month and a datetime axis "
            f"({sorted(product)}), which cannot share one open_mfdataset "
            "(by-coords) axis.")
    dtype = next(iter(dtypes))
    values = np.concatenate(chunks)
    if dtype == "datetime" and np.issubdtype(values.dtype, np.datetime64):
        order = np.sort(values.astype("datetime64[ns]"))
        key = frozenset(order.astype("int64").tolist())
        label = f"datetime {order[0]}..{order[-1]} ({len(key)} steps)"
    elif dtype == "datetime":
        strs = sorted(str(v) for v in values)
        key = frozenset(strs)
        label = f"datetime {strs[0]}..{strs[-1]} ({len(key)} steps)"
    else:
        months = sorted({int(v) for v in values})
        key = frozenset(months)
        label = f"integer-month {months[0]}..{months[-1]} ({len(key)} steps)"
    return (dtype, key, label)


def assert_uniform_time_axis(products, *, config_key, open_dataset=None) -> None:
    """Reject a product set that cannot share one by-coords time axis.

    Well-posed in exactly two shapes: (a) every time-bearing product shares ONE
    identical axis (a stacked climatology / same-year set — variables union with
    no NaN-fill), or (b) a SINGLE product (its own yearly files tile one
    monotonic transient axis, or a lone climatology). Anything else — e.g. a
    one-year transient beside a climatology dated in another year — is rejected,
    naming each product's range. Port of ``jcm.runners._assert_uniform_time_axis``.
    """
    if sum(len(p) for p in products) <= 1:
        return
    axes = []
    for p in products:
        axis = product_time_axis(p, open_dataset=open_dataset)
        if axis is None:
            continue
        label = p[0] if len(p) == 1 else f"{p[0]} … (+{len(p) - 1} more)"
        axes.append((label, axis))
    if len(axes) <= 1:
        return
    _, (ref_dtype, ref_key, _) = axes[0]
    if all(dtype == ref_dtype and key == ref_key
           for _, (dtype, key, _) in axes[1:]):
        return
    detail = "; ".join(f"{label}: {rng}" for label, (_, _, rng) in axes)
    raise ValueError(
        f"{config_key} mixes incompatible time axes across its products "
        f"({detail}). These products are opened TOGETHER along ONE time axis "
        "(open_mfdataset, by-coords), so they must either all share an "
        "IDENTICAL time axis (a stacked climatology / same-year set) or be a "
        "single product's yearly files (one concatenated transient series). A "
        "one-year transient beside a climatology dated in another year, or an "
        "integer-month climatology beside datetime transients, cannot share one "
        "axis — by-coords would NaN-fill every non-overlapping step. Put them "
        "on one time axis, or drop one.")


# ---------------------------------------------------------------------------
# path fetching (hf:// resolution)
# ---------------------------------------------------------------------------

def _default_fetch(rel_path: str) -> str:
    from jcm.data.remote import fetch
    return fetch(rel_path)


def _fetch_path(p, fetch):
    """Resolve one path element: hf:// fetched to a local cache path, else as-is.

    A list is resolved element-wise; ``None``/sentinels pass through.
    """
    if isinstance(p, str) and p.startswith("hf://"):
        return (fetch or _default_fetch)(p[len("hf://"):])
    if _is_seq(p):
        return [_fetch_path(e, fetch) for e in p]
    return p


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------

def resolve_input(key, value, *, grid_token, nlev=None, vertical="hybrid",
                  years=None, available=None, fetch=None, manifest=None,
                  enabled=True, expand=None) -> ResolvedInput:
    """Resolve one forcing value into concrete local paths + alignment.

    ``value`` is the raw config value (``auto`` / a path / ``hf://`` URL /
    ``{year}`` pattern / a list of these / ``null``). Grid identity
    (``grid_token``, ``nlev``, ``vertical``) drives ``auto`` publication gating
    against the mirror ``manifest`` (loaded lazily if not supplied). ``enabled``
    is the consumer-side gate — a physics package that consumes this input is
    active; when ``False`` an ``auto`` value supplies nothing (the non-JAM /
    pySES-emission case, decided by the caller, not by this layer).

    * ``auto`` → the manifest product for ``key``, fetched *now* so a cold cache
      fails loudly at build time; resolves to a no-op when the mirror does not
      publish it for this grid (silent degrade the caller surfaces as a warning),
      and raises a precise "declared but not yet staged" error for a product like
      ``macv2_sp`` that the mirror has not published yet.
    * explicit path(s) → ``{year}`` patterns expanded (coverage-clamped), each
      list element kept as its own product, ``hf://`` fetched.
    * ``null`` → an opted-out :class:`ResolvedInput` (``is_none``).
    """
    from jcm.data import mirror_manifest as mm
    if manifest is None:
        manifest = mm.load_manifest()

    spec = InputSpec.parse(key, value)

    if spec.kind is SpecKind.NONE:
        return ResolvedInput(key=key, is_none=True, source="none")

    if spec.kind is SpecKind.AUTO:
        name = mm.product_for_key(manifest, key)
        if name is None:
            raise ValueError(
                f"forcing.{key}=auto but no mirror product declares key={key!r}")
        if not enabled:
            return ResolvedInput(key=key, is_none=True, source="none")
        if not mm.is_published(manifest, name, grid_token, nlev, vertical):
            return ResolvedInput(key=key, is_none=True,
                                 source=f"auto:{name}:unpublished")
        if not mm.product(manifest, name)["staged"]:
            raise FileNotFoundError(
                f"forcing.{key}=auto resolves to the {name!r} product, which is "
                "declared in the mirror manifest but NOT yet published on the "
                "data mirror. Provide the file explicitly "
                f"(forcing.{key}=/path/to/file), or set forcing.{key}=null.")
        rel = mm.bundle_path(manifest, name, grid_token, nlev)
        hf = f"hf://{rel}"
        try:
            local = (fetch or _default_fetch)(rel)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(
                f"forcing.{key}=auto resolved to {hf} but it is not in the "
                "local Hugging Face cache and could not be downloaded. Prefetch "
                "on a node with internet, point the key at a local file, or set "
                f"forcing.{key}=null to run without it.") from e
        alignment = _ALIGNMENT_FROM_MANIFEST[
            mm.product(manifest, name)["alignment"]]
        return ResolvedInput(key=key, products=((local,),), paths=(local,),
                             alignment=alignment, source=f"auto:{name}",
                             provenance=(hf,))

    # EXPLICIT
    products = forcing_products(spec.raw, years, available, expand=expand)
    resolved_products = []
    flat = []
    provenance = []
    for p in products:
        for orig in (p if _is_seq(p) else [p]):
            if isinstance(orig, str) and orig.startswith("hf://"):
                provenance.append(orig)
        r = _fetch_path(p, fetch)
        if _is_none(r):
            continue
        if _is_seq(r):
            files = [str(e) for e in r if not _is_none(e)]
            if not files:
                continue
            resolved_products.append(files)
            flat.extend(files)
        else:
            resolved_products.append(str(r))
            flat.append(str(r))
    if not flat:
        return ResolvedInput(key=key, is_none=True, source="none")
    return ResolvedInput(key=key, products=tuple(resolved_products),
                         paths=tuple(flat), alignment=AUTO, source="explicit",
                         provenance=tuple(provenance))
