# nsnucl=2 end-to-end chain reference (jax-gcm#1017 Kazil/GCR task, W5)

Chains the UNMODIFIED compiled Fortran `read_obrien_gcr_ipr -> gcr_ionization
-> nucl_kazil_lovejoy` against the REAL O'Brien and PARNUC tables: stage 1's
ion-pair-production rate (reused unmodified from `hamgcr.npz`'s own
Fortran-validated `out/pgcripr`) is fed as `nucl_kazil_lovejoy`'s
`ion_pair_rate` input for 16 designed (RH%, [H2SO4], sink) cases spanning the
PARNUC table's ranges, at 2 (column, level) stage-1 pairs (one genuine
mid-table interpolation ~18-45 cm-3 s-1, one below the table's ion-pair-rate
minimum, exercising the clamp branch).

The full PARNUC table (40x40x40x20x40 float32, ~200MB) is NOT embedded; only
the per-case 2-point-per-axis / 32-corner excerpt `kazil_lovejoy`'s own
bracket search actually touches is (`in/kazil_table_axes`,
`in/kazil_table_log_pfr`) -- verified in this build to reproduce the
full-table call bit-for-bit. **Data only**: no ECHAM/HAM or m7-jax source is
part of this repository.

Stage 1 is independently closed in the same build: jcm's own
`gcr_ion_pair_rate`, fed the embedded real O'Brien table plus each case's
(lat, lon, pressure, temperature, date, solar activity), reproduces
`hamgcr.npz`'s Fortran `out/pgcripr` to max relative error
0.000e+00 -- so this file alone exercises BOTH chain stages
end to end, not just stage 2.

Max relative error, jax vs Fortran (stage 2): full table 0.000e+00,
mini table 0.000e+00, mini table at the float32 core
vs Fortran float64 3.693e-06 (see `hamnucl2_provenance.json`).

## Arrays

| name | meaning |
|---|---|
| `meta/column_lat`, `meta/column_lon`, `meta/scenario_name`, `meta/scenario_psolact`, `meta/scenario_date`, `meta/pressure` | stage-1 inputs: which `hamgcr.npz` (scenario, column, level) each case's `ion_pair_rate` came from, and the (lat, lon, pressure, date, solar activity) `gcr_ion_pair_rate` needs to recompute it |
| `in/vertical_cutoff_rigidity`, `in/mass_column_density`, `in/ipr_solmin`, `in/ipr_solmax` | the full REAL O'Brien table (small: (15,110)), reused unmodified from `hamgcr.npz` |
| `in/temperature`, `in/relative_humidity_pct`, `in/h2so4`, `in/total_sink`, `in/ion_pair_rate` | `nucl_kazil_lovejoy`'s own stage-2 inputs, shape (16,) (`in/ion_pair_rate` is stage 1's Fortran output, reused as stage 2's input -- the actual chain) |
| `in/kazil_table_axes` | per-case bracket pair, shape (16, 5, 2), axis order (temperature, RH, H2SO4, ion_pair_rate, condensation_sink) |
| `in/kazil_table_log_pfr` | per-case 32-corner `ln(pfr)` excerpt, shape (16, 2, 2, 2, 2, 2), same axis order |
| `out/rate`, `out/cluster_sulfate` | `nucl_kazil_lovejoy`'s own outputs [cm-3 s-1], [H2SO4 molecules] |

## Regenerate

`python build_hamnucl2_chain.py` from `/scr/dwatsonparris/ham-m7/w5/`
(needs `HAM_INPUT_DIR`'s real PARNUC table, `hamgcr.npz` already committed,
and an m7-jax checkout with Kazil/Lovejoy support on `sys.path`, e.g. the
copy at `/scr/dwatsonparris/ham-m7/w5/m7jax-ref`). jax_enable_x64 MUST be
True before `build_oracle`/`load_kazil_lovejoy_table` run (see the script's
own module docstring for the float32-freeze trap this guards against).
