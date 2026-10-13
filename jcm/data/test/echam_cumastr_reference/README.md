# ECHAM6.3 convection, evaluated on 758 columns

`echam_cumastr.npz` holds what ECHAM6.3-HAM2.3 (r7492)
`mo_cumastr.f90::cucall` returns for 758 single-column states: the convective
decisions (`ldcum`, `ktype`, `kcbot`, `kctop`), the final cloud-base mass
flux, the surface rain and snow, and the temperature and humidity tendencies
per level. `jcm/physics/convection/tiedtke_nordeng/cumastr_reference_test.py`
runs jcm's Tiedtke-Nordeng scheme on the same inputs and compares.

The file contains numbers only. No ECHAM source code is stored in this
repository.

## Columns

Every column is the exact set of arguments jcm's
`tiedtke_nordeng_convection` received at one step of the whole-model
radiative-convective column of `jcm/rce_test.py::TestRceWholeModelTiedtke` as it
was configured at capture, with the idealized grey radiation (80 days,
`dt = 900` s, 47 levels, float32; the test now runs RRTMGP, see
`docs/source/design/rce_testbed.md`), captured from two checkouts:

| `group` | source | columns | what it exercises |
|---|---|---|---|
| `rce_fogged` | `feat/echam-1m-faithful` at 82c2f294 (1M column with a fogged lowest level) | 200 | the `cubase` trigger, the `zlo1` gate and the first ascent test above a cloud base at `klevm1` |
| `rce_warm` | `dev` at 8393799c (the previous 1M; a warmer, fog-free boundary layer) | 200 | the same, and shallow plumes up to 400 hPa |
| `midlevel_rce_fogged`, `midlevel_rce_warm` | 50 captured columns of each, with a synthetic resolved ascent | 100 | the `cubasmc` mid-level trigger and seed |
| `deep_rce_fogged`, `deep_rce_warm` | 50 captured columns of each, with a synthetic moisture convergence | 100 | the deep/shallow test, the deep plume, the Nordeng closure and the downdrafts |
| `zlo1_rce_fogged`, `zlo1_rce_warm` | every captured column where `cubase` finds a cloud base and `zlo1` rejects it | 27 | the `zdqpbl > 0` gate |
| `zlo1_excess_rce_fogged`, `zlo1_excess_rce_warm` | 40 captured columns of each, the two levels above the lowest made nearly as humid as it | 80 | the `zqumqe > zdqmin` gate |
| `zlo1_midlevel_rce_fogged`, `zlo1_midlevel_rce_warm` | 15 captured columns of each, with a synthetic sub-cloud divergence under a resolved ascent | 30 | a surface plume `zlo1` rejects and a mid-level plume in the first ascent |
| `subcloud_evaporation_rce_fogged`, `subcloud_evaporation_rce_warm` | 10 captured columns of each whose convective rain evaporates below cloud base | 20 | the `cevapcu` profile of the sub-cloud evaporation |
| `downdraft_cancel_rce_fogged` | the one captured column whose downdraft `cuflx` cancels (level of free sinking above the final top) | 1 | `IF (kdtop < kctop) lddraf = .FALSE.` |

The RCE columns are stratified by what jcm's scheme did at the time of
capture against what ECHAM does (`sample_class`): 70 columns where both
convect, 70 where only jcm's earlier scheme convected (the disagreement of
jax-gcm#968) and 60 where neither does. `source_step` is the step of the
80-day run. Columns were drawn with `numpy.random.default_rng(968)` and, for
the `zlo1_excess` and `zlo1_midlevel` groups, `numpy.random.default_rng(9681)`.

The column has no dynamical core, so its captured `qte_dynamics` is round-off
(~1e-12 kg/kg/s) and `omega` is zero. The synthetic groups change one captured
argument each, and the stored inputs carry the change:

- Mid-level groups: `omega = −A·sin(π(p − p_top)/(p_bot − p_top))` between
  `p_top` (uniform in 250-450 hPa) and `p_bot` (uniform in 700-900 hPa),
  zero elsewhere, with `A` log-uniform in 0.02-1 Pa/s.
- Deep groups: `qte_dynamics = F·E·s(p)/Σ(s·Δp/g)`, where `s(p) =
  sin(π(p − 400 hPa)/650 hPa)` below 400 hPa and zero above, `E` is the
  column's surface evaporation and `F` is log-uniform in 0.02-3. The
  convergence reaches the sub-cloud layers, so it enters both of `cumastr`'s
  integrals of `pqte`: the deep/shallow test and the sub-cloud supply.
- `zlo1_excess` groups: the humidity of the second and third levels from the
  bottom is set to `F` times the lowest level's, `F` uniform in 0.985-1.005.
- `zlo1_midlevel` groups: `qte_dynamics = −D·E/Σ(Δp/g)` over the lowest four
  levels and zero above, `D` uniform in 1.5-4, and `omega` as in the
  mid-level groups.

## How ECHAM was run

ECHAM's own `mo_cumastr.f90` (md5 `3dbd4fa6c49e9d092bcbb164ddeaa9ed`),
`mo_cuinitialize.f90` (`3353fec5e31f49c9dd87debedc1f83c1`),
`mo_cuascent.f90` (`385ecc2c36f3d920f3d0000878f40b6b`),
`mo_cudescent.f90` (`b1ebe2becb40ec619e8a1645491e8fd7`),
`mo_cufluxdts.f90` (`2c182407d8254b782149538acf33b6bd`),
`mo_cuadjust.f90` (`ee47f6072e3d5446b70b598a23c735ba`) and
`mo_echam_convect_tables.f90` (`ce2f14c2d84e7aa5c30149456671da0c`) were
compiled unmodified with GNU Fortran 11.3.0 (`-O2 -cpp -D__LP64__
-D__ICON__ -fimplicit-none`) on 2026-09-30. `-D__ICON__` selects the
preprocessor branch of the same files that drops ECHAM's own I/O and tracer
plumbing and returns the convective tendencies separately; the science is the
same code. Two stubs stand in for modules that pull in the rest of ECHAM:

- `mo_echam_conv_constants` (md5 `ac9ed751ad1a36850de71c9c1d32ef5f`):
  `cuparam`'s values for the `__ICON__` branch (`entrpen = 1e-4`,
  `entrscv = 3e-3`, `entrmid = 1e-4`, `entrdd = 2e-4`, `centrmax = 3e-4`,
  `cmfctop = 0.2`, `cminbuoy = 0.2`, `cmaxbuoy = 1`, `cbfac = 1`,
  `cmfcmax = 1`, `cmfcmin = 1e-10`, `cmfdeps = 0.3`, `cprcon = 2.5e-4`,
  `cmftau = 7200`) and `lmfmid = lmfdd = lmfdudv = .TRUE.`;
- `mo_echam_cloud_params` (md5 `2e54958f64c1c81da4d861a32170d0b0`):
  `cthomi = tmelt − 35`, `csecfrl = 5e-6`.

`mo_physical_constants` (md5 `601964f57338305b84584de3a8e81188`) is ECHAM's
with one comment line made code (a constant the convection does not use);
`mo_kind`, `mo_math_constants` and `mo_exception` are the ICON versions of
those utility modules. A driver (md5 `0fadb1a333b67d85b958070f5d6024d3`)
reads the columns, sets `cevapcu` from `iniphy.f90`'s profile with
`eta = p/p_s` of the first column and `nmctop` as `cuparam` does, calls
`init_convect_tables` and then `cucall` once on all columns.

The inputs ECHAM receives are the stored float32 values in float64: `pten`
temperature, `pqen` humidity, `pqm1` the step-start humidity, `pxen = qc + qi`,
`puen`, `pven`, `pverv = omega`, `papp1` pressure, `paphp1` pressure_half,
`pqte = moisture_tend_profile + qte_dynamics`, `pqhfla = −moisture_supply`,
`pthvsig = thvsig`, `ldland = land_fraction > 0.5` (always false here) and
the full-level geopotential `pgeo`, built as jcm's
`half_level_environment` builds it, with ECHAM's constants below.

The cloud base, the final cloud-base mass flux and the `ldcum` flag are not
among `cucall`'s arguments. They were read from a second build of the same
routines with assignments to a diagnostics module added and nothing else
changed; the two builds return bit-identical outputs on all 758 columns.

## Arrays

| name | unit | meaning |
|---|---|---|
| `input_<argument>` | as jcm's | the arguments of `tiedtke_nordeng_convection`, top-first, `(758, 47)` or `(758,)`; `input_pressure_half` is `(758, 48)` |
| `dt` | s | the time step, 900 |
| `group`, `sample_class`, `source_step` | – | see above |
| `eta_full` | – | the `eta` of ECHAM's `cevapcu` profile |
| `echam_ldcum`, `echam_ktype` | – | the column convects; its type (0 none, 1 deep, 2 shallow, 3 mid-level) |
| `echam_kcbot`, `echam_kctop` | – | cloud base and top, ECHAM's 1-based top-first level index (jcm's 0-based index + 1); meaningful where `echam_ldcum` |
| `echam_cloud_base_mass_flux` | kg m-2 s-1 | the final cloud-base mass flux `zmfub` |
| `echam_rain`, `echam_snow` | kg m-2 s-1 | surface convective rain and snow (`prsfc`, `pssfc`) |
| `echam_temperature_tendency`, `echam_humidity_tendency` | K s-1, kg kg-1 s-1 | `ptte_cnv`, `pqte_cnv` |
| `echam_grav`, `echam_rd`, `echam_rv`, `echam_cpd`, `echam_cpv`, `echam_alv`, `echam_als`, `echam_tmelt` | SI | ECHAM's constants the routines ran with |
