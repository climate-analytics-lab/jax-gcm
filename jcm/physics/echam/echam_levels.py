"""ECHAM/ICON hybrid sigma-pressure level tables.

The 47-level table here matches both ECHAM6.3 ``L47 lmidatm`` (verified
bit-for-bit against a production ECHAM6.3-HAM T63L47 ``vct_a/vct_b``
log on 2026-05-14) and ICON's standard ``L47`` table — these are
shared heritage tables, not separate. The module was originally named
after ICON because that's where the values came from in this codebase;
the actual numerical values are equally the ECHAM6.3 lmidatm
production grid (top at p=0, three trailing zero ``a`` boundaries
intentionally degenerate at the top). When pairing this grid with
ECHAM-style physics, the model needs ECHAM's ``lmidatm`` stability
stack — upper sponge with T damping on m≠0 modes
(``mo_upper_sponge.f90``) and level-dependent del²→del⁸ horizontal
diffusion at the top 4 levels (``mo_hdiff.f90::sudif``) — otherwise
the thin top layer (~2 Pa mass) runs away under any radiative
imbalance, the failure mode documented in the
``echam_rrtmgp_t63_real_orog_nan`` memory.

ECHAM stores the actual numerical table in initial-condition netCDFs
(``vct_a``/``vct_b`` variables, ``nvclev`` dim) read at runtime by
``mo_io.f90``; the source code itself contains no level numbers.

The ``a_boundaries`` values are in **Pascals** (Pa).
``dinosaur.primitive_equations.PrimitiveEquationsHybrid`` expects ``a``
in its ``hpa_quantity`` unit (default hPa); ``Model`` overrides this
to ``units.pascal`` when constructed with these ``HybridCoordinates``.
"""

import jax.numpy as jnp
import numpy as np
from dinosaur.hybrid_coordinates import HybridCoordinates


def _checked_hybrid(a_boundaries, b_boundaries) -> 'HybridCoordinates':
    """Build ``HybridCoordinates``, rejecting a truncated vct table.

    The tables here are TOA-first, so the *top* interface is index 0: its
    pressure at a 1013.25 hPa surface is ``a_boundaries[0] + b_boundaries[0] *
    101325`` Pa, which for a full-depth grid is essentially zero (a[0]=b[0]=0).
    A table sliced to the bottom N entries of a longer vct (the removed L40
    "grid" was the bottom 40 rows of L47 — a 274 hPa model top, not a real grid)
    keeps a large ``a[0]`` and so a high top pressure. Catch that here rather
    than integrate a decapitated atmosphere.
    """
    p_top_pa = float(a_boundaries[0] + b_boundaries[0] * 101325.0)
    if not p_top_pa < 1000.0:
        raise ValueError(
            f"ECHAM hybrid level table has a top-interface pressure of "
            f"{p_top_pa:.1f} Pa (>= 1000 Pa / 10 hPa) — that is a truncated "
            "vct table, not a full-depth grid whose model top sits at "
            "near-zero pressure. Supply the complete vct_a/vct_b table.")
    return HybridCoordinates(a_boundaries=a_boundaries,
                             b_boundaries=b_boundaries)


def get_echam_levels(nlevels: int) -> 'HybridCoordinates':
    """Get ICON hybrid levels for specified number of levels.

    Returns `a_boundaries` in Pa (ICON native convention).

    Args:
        nlevels: Number of vertical levels (must be available in ICON tables)

    Returns:
        HybridCoordinates with a_boundaries in Pa and dimensionless b_boundaries.

    """
    if nlevels == 47:
        # ICON 47-level standard configuration (a values in Pa)
        a_boundaries = jnp.array([
            0.00000000000, 1.98918528294, 6.57208964360, 15.67390258170,
            30.62427876410, 54.54572041260, 92.55883043370, 150.50469698200,
            235.32745773100, 356.10025910400, 523.91952428200, 751.04294180400,
            1051.13722461000, 1438.98841128000, 1930.17735994000, 2540.69700000000,
            3286.55300000000, 4199.57400000000, 5303.95700000000, 6624.70400000000,
            8187.18500000000, 9976.13700000000, 11820.54000000000, 13431.39000000000,
            14736.36000000000, 15689.21000000000, 16266.61000000000, 16465.00000000000,
            16297.62000000000, 15791.60000000000, 14985.27000000000, 13925.52000000000,
            12665.29000000000, 11261.23000000000, 9771.40600000000, 8253.21100000000,
            6761.34000000000, 5345.91400000000, 4050.71800000000, 2911.56900000000,
            1954.80500000000, 1195.89000000000, 638.14890000000, 271.62650000000,
            72.06360000000, 0.00000000000, 0.00000000000, 0.00000000000
        ])
        
        b_boundaries = jnp.array([
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00040000000, 0.00290000000, 0.00920000000,
            0.02030000000, 0.03700000000, 0.05950000000, 0.08790000000,
            0.12200000000, 0.16140000000, 0.20570000000, 0.25420000000,
            0.30620000000, 0.36110000000, 0.41820000000, 0.47670000000,
            0.53590000000, 0.59510000000, 0.65360000000, 0.71060000000,
            0.76540000000, 0.81720000000, 0.86500000000, 0.90770000000,
            0.94420000000, 0.97300000000, 0.99230000000, 1.00000000000
        ])
        
        return _checked_hybrid(a_boundaries, b_boundaries)

    elif nlevels == 95:
        # ECHAM6 middle-atmosphere ``L95`` table (lid at ~0.01 hPa),
        # transcribed from the ``vct_a``/``vct_b`` of a production
        # ``T63L95_jan_spec.nc`` initial file (2026-07-31). Same TOA-first
        # orientation and Pa units as the L47 table above; supports the
        # stratosphere-resolving MA configuration at any horizontal
        # truncation (the table is vertical-only).
        a_boundaries = jnp.array([
            0.00000000000, 1.98918247223, 2.69261074066, 3.54616451263,
            4.57676124573, 5.81494045258, 7.29508113861, 9.05558681488,
            11.13899898529, 13.59204196930, 16.46557617188, 19.81443786621,
            23.69715881348, 28.17553710938, 33.31410217285, 39.17933654785,
            45.83877563477, 53.36004638672, 61.84652709961, 71.41293334961,
            82.18634033203, 94.30740356445, 107.93159484863, 123.23060607910,
            140.39379882812, 159.62977600098, 181.16809082031, 205.26101684570,
            232.18553161621, 262.24536132812, 295.77294921875, 333.13256835938,
            374.72143554688, 420.97338867188, 472.36132812500, 529.40039062500,
            592.64990234375, 662.71801757812, 740.26416015625, 826.00268554688,
            920.70605468750, 1025.20947265625, 1140.41430664062, 1267.29199218750,
            1406.88818359375, 1560.32666015625, 1728.81445312500, 1913.64550781250,
            2116.40527343750, 2338.83251953125, 2582.83544921875, 2850.50659179688,
            3144.14184570312, 3466.25976562500, 3819.62304687500, 4207.26171875000,
            4632.50390625000, 5098.99218750000, 5610.73046875000, 6172.44531250000,
            6789.26171875000, 7464.85546875000, 8205.07421875000, 9013.73437500000,
            9876.25000000000, 10779.67968750000, 11698.04296875000, 12606.03906250000,
            13479.76171875000, 14289.19140625000, 15005.62109375000, 15604.63671875000,
            16062.08593750000, 16355.96484375000, 16464.95703125000, 16370.24609375000,
            16058.29296875000, 15520.17968750000, 14753.79296875000, 13765.30859375000,
            12573.00000000000, 11218.07421875000, 9756.42187500000, 8253.21093750000,
            6761.33984375000, 5345.91406250000, 4050.71801757812, 2911.56909179688,
            1954.80493164062, 1195.88989257812, 638.14892578125, 271.62646484375,
            72.06359863281, 0.00000000000, 0.00000000000, 0.00000000000
        ])

        b_boundaries = jnp.array([
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00000000000,
            0.00000000000, 0.00000000000, 0.00000000000, 0.00004644049,
            0.00034243893, 0.00110446941, 0.00262146862, 0.00530741736,
            0.00948636234, 0.01555586606, 0.02390019596, 0.03493614122,
            0.04900494963, 0.06649875641, 0.08780068159, 0.11324220896,
            0.14307528734, 0.17757314444, 0.21690040827, 0.26105165482,
            0.30987769365, 0.36276280880, 0.41877347231, 0.47670000792,
            0.53589999676, 0.59509998560, 0.65359997749, 0.71060001850,
            0.76539999247, 0.81720000505, 0.86500000954, 0.90770000219,
            0.94419997931, 0.97299998999, 0.99229997396, 1.00000000000
        ])

        return _checked_hybrid(a_boundaries, b_boundaries)

    else:
        raise ValueError(f"No built-in level definition for {nlevels} levels. "
                         "Available: 47, 95. For any other table supply the "
                         "half-level coefficients through grid.level_table "
                         "(hybrid_levels_from_table / read_vct_file).")


# ---------------------------------------------------------------------------
# User-supplied tables
# ---------------------------------------------------------------------------

#: Built-in level counts (the ``nlevels`` values :func:`get_echam_levels` serves).
BUILTIN_LEVEL_COUNTS = (47, 95)

#: Reference surface pressure [Pa] at which a table's half-level pressures are
#: evaluated for validation and for mapping per-level quantities by pressure
#: (the same 1013.25 hPa the level-resolved boundary files are built at,
#: ``jcm.data.bc.interpolate_ozone.REFERENCE_SURFACE_PRESSURE_PA``).
REFERENCE_SURFACE_PRESSURE_PA = 101325.0

# Surface pressures [Pa] a table's layers must stay positive-mass across: the
# half-level pressure ``a + b*ps`` is linear in ``ps``, so checking the two
# ends of the range that real columns span (the Tibetan Plateau to a deep low)
# covers every value between.
_SURFACE_PRESSURE_RANGE_PA = (50000.0, 110000.0)


def hybrid_levels_from_table(a_boundaries, b_boundaries) -> 'HybridCoordinates':
    """Build ``HybridCoordinates`` from a user-supplied vct table.

    The table has the convention and units of the built-in ones (and of ECHAM's
    ``vct_a``/``vct_b``): half-level (interface) coefficients, ``nlev + 1``
    entries, **top of atmosphere first**, ``a`` in Pa and ``b`` dimensionless,
    so that the interface pressure is ``p = a + b * ps``. The result is built
    through the same top-pressure guard as the built-in tables
    (``_checked_hybrid``), after the checks a hand-made table needs that a
    transcribed one does not. Each failure names the offending property, because
    a table that is silently reinterpreted (a surface-first one flipped, a
    normalised ``a`` read as Pa) integrates a different atmosphere without any
    error:

    * ``a`` and ``b`` are finite, 1-D, of equal length ``>= 3``;
    * ``a[-1] == 0`` and ``b[-1] == 1`` (the lowest interface is the surface,
      ``p = ps``), and ``a >= 0``, ``0 <= b <= 1``, ``b`` non-decreasing;
    * interface pressure increases strictly with index at the reference surface
      pressure and across 50-110 kPa (positive layer mass over mountains and
      deep lows). A table that *decreases* (``b[0] == 1``, ``b[-1] == 0``) is
      surface-first and is rejected rather than flipped: which end is the
      surface is the one fact a caller must state, not one the model guesses.

    Args:
        a_boundaries: ``(nlev + 1,)`` pressure coefficients [Pa], TOA first.
        b_boundaries: ``(nlev + 1,)`` sigma coefficients [1], TOA first.

    Returns:
        ``HybridCoordinates`` with ``a_boundaries`` in Pa, as
        :func:`get_echam_levels` returns.

    Raises:
        ValueError: if the table violates any of the above.

    """
    a = np.asarray(a_boundaries, dtype=np.float64)
    b = np.asarray(b_boundaries, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"hybrid level table: a and b must be 1-D, got shapes {a.shape} "
            f"and {b.shape}.")
    if a.shape != b.shape:
        raise ValueError(
            f"hybrid level table: a has {a.size} entries but b has {b.size}; "
            "both are the nlev + 1 half-level (interface) coefficients.")
    if a.size < 3:
        raise ValueError(
            f"hybrid level table has {a.size} half levels; at least 3 "
            "(2 layers) are needed.")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("hybrid level table: a and b must be finite.")
    if b[0] > b[-1]:
        raise ValueError(
            "hybrid level table looks SURFACE-FIRST (b decreases from "
            f"b[0]={b[0]:g} to b[-1]={b[-1]:g}). The table must be top-of-"
            "atmosphere first, like ECHAM's vct_a/vct_b (a[0]≈0, b[0]=0 at the "
            "top; a[-1]=0, b[-1]=1 at the surface). Reverse both arrays "
            "yourself if that is what you intended.")
    if a[-1] != 0.0 or b[-1] != 1.0:
        raise ValueError(
            "hybrid level table must end at the surface interface (a[-1]=0, "
            f"b[-1]=1 so that p = ps); got a[-1]={a[-1]:g}, b[-1]={b[-1]:g}. "
            "Is the table truncated at the bottom?")
    if np.any(a < 0.0) or np.any(b < 0.0) or np.any(b > 1.0):
        raise ValueError(
            "hybrid level table: a must be >= 0 (Pa) and b within [0, 1]. "
            "ECHAM's vct_a is in Pa; a table with a normalised a (a/p0) must be "
            "multiplied by p0 = 101325 Pa first.")
    if np.any(np.diff(b) < 0.0):
        raise ValueError(
            "hybrid level table: b must be non-decreasing from the top to the "
            "surface.")
    for ps in (REFERENCE_SURFACE_PRESSURE_PA, *_SURFACE_PRESSURE_RANGE_PA):
        dp = np.diff(a + b * ps)
        if not np.all(dp > 0.0):
            k = int(np.argmin(dp))
            raise ValueError(
                "hybrid level table: interface pressure does not increase "
                f"strictly downward at ps={ps:g} Pa (layer {k} has "
                f"dp={dp[k]:g} Pa). A table must have positive layer mass for "
                "every surface pressure the model can reach.")
    return _checked_hybrid(jnp.asarray(a), jnp.asarray(b))


def read_vct_file(path) -> 'HybridCoordinates':
    """Read ECHAM's ``vct_a``/``vct_b`` from a netCDF file into hybrid levels.

    The built-in L47/L95 tables were transcribed from exactly these variables
    (``nvclev`` dimension, ``vct_a`` in Pa, ``vct_b`` dimensionless, TOA first)
    of production ECHAM initial-condition files, so such a file can be named
    directly. The values are validated by :func:`hybrid_levels_from_table`.

    Raises:
        ValueError: if either variable is missing, or the table is invalid.

    """
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:
        missing = [v for v in ("vct_a", "vct_b") if v not in ds.variables]
        if missing:
            raise ValueError(
                f"{path}: no {', '.join(missing)} variable; a level-table file "
                "carries ECHAM's vct_a (Pa) and vct_b (dimensionless) on the "
                f"half levels. Variables present: {sorted(map(str, ds.variables))}.")
        a = np.asarray(ds["vct_a"].values, dtype=np.float64)
        b = np.asarray(ds["vct_b"].values, dtype=np.float64)
    return hybrid_levels_from_table(a, b)


def levels_from_level_table(level_table, layers=None) -> 'HybridCoordinates':
    """Resolve a ``grid.level_table`` value into hybrid levels.

    Args:
        level_table: a mapping with ``a`` and ``b`` lists (the half-level
            coefficients, Pa and dimensionless, TOA first), or the path of a
            netCDF file carrying ``vct_a``/``vct_b`` (:func:`read_vct_file`).
            A path must already be resolved to a local file;
            ``jcm.runners.build_coords`` resolves ``hf://`` and relative paths
            before calling.
        layers: the configured layer count, if any; it must equal the table's
            ``len(a) - 1``.

    Returns:
        The validated ``HybridCoordinates``.

    Raises:
        ValueError: on a malformed table, a layer-count mismatch, or an
            unrecognised ``level_table`` type.

    """
    from collections.abc import Mapping
    if isinstance(level_table, Mapping):
        if set(level_table) != {"a", "b"}:
            raise ValueError(
                "grid.level_table given inline must be {a: [...], b: [...]} "
                f"(the half-level coefficients); got keys {sorted(level_table)}.")
        levels = hybrid_levels_from_table(level_table["a"], level_table["b"])
    elif isinstance(level_table, str) or hasattr(level_table, "__fspath__"):
        levels = read_vct_file(level_table)
    else:
        raise ValueError(
            "grid.level_table must be {a: [...], b: [...]} or the path of a "
            f"netCDF file with vct_a/vct_b; got {type(level_table).__name__}.")
    if layers is not None and int(layers) != levels.layers:
        raise ValueError(
            f"grid.layers={int(layers)} but the level table has {levels.layers} "
            f"layers (len(a) - 1 = {levels.layers}). Set grid.layers to match, "
            "or null to take it from the table.")
    return levels


def is_builtin_table(levels) -> bool:
    """Whether ``levels`` is one of the built-in tables (:func:`get_echam_levels`).

    The built-in tables are the ones the level-resolved boundary data (ozone,
    oxidants) and the ``(truncation, nlev)`` hyperdiffusion profiles are
    published for; any other table that merely shares a level count is a
    different vertical grid. Compared to float32 round-off, since the built-in
    arrays are float32 unless x64 is enabled while a table read from a file is
    float64.
    """
    n = int(levels.layers)
    if n not in BUILTIN_LEVEL_COUNTS:
        return False
    ref = get_echam_levels(n)
    return bool(
        np.allclose(np.asarray(levels.a_boundaries, np.float64),
                    np.asarray(ref.a_boundaries, np.float64),
                    rtol=1e-6, atol=1e-6)
        and np.allclose(np.asarray(levels.b_boundaries, np.float64),
                        np.asarray(ref.b_boundaries, np.float64),
                        rtol=1e-6, atol=1e-6))