# ECHAM6.3-HAM2.3 in-cloud impaction tables (#1017 follow-up B)

`cdroprad`/`cplaterad`/`scavdropn`/`scavdropm`/`scaviceplate` are
ECHAM6.3-HAM2.3 r7492's `mo_ham_wetdep_data.f90` in-cloud impaction
look-up tables (Croft et al. 2010), extracted as NUMBERS (never as
Fortran text) by compiling the unmodified data module with a print
driver -- see `dump_ic_tables.f90` and
`/scr/dwatsonparris/ham-m7/w2/imp/harness/` (private scratch, not
pushed). The shared aerosol-radius axis (`caerorad`, 61 points) already
lives in `croft_bc_tables.npz` (the below-cloud slice) and is reused
directly, not duplicated here.

**`cdroprad(6)` reads `0.0`**, not the `30.0` the regular 5-um spacing
(`0,5,10,15,20,25,[30],35,40,45,50`) implies -- confirmed in the COMPILED
output, so a genuine upstream data value, not a transcription slip here.
Flagged to the maintainer as a suspected upstream typo; ported AS
COMPILED (see `jcm/physics/aerosol/jam/wetdep/ham_impaction.py`'s
module docstring for the measured effect on lookups that land on that
node).

## Arrays

| name | shape | meaning |
|---|---|---|
| `cdroprad` | (11,) | cloud-droplet radius axis [um]; index 10 is dead (the index formula clips to 9) |
| `cplaterad` | (35,) | ice-plate radius axis [um] |
| `scavdropn` | (10, 61) | number impaction coefficient, (drop radius, aerosol radius) |
| `scavdropm` | (10, 61) | mass impaction coefficient, (drop radius, aerosol radius) |
| `scaviceplate` | (35, 61) | ice-plate impaction coefficient, (plate radius, aerosol radius) |

## Regenerate

From this directory (`fortran_harness/echam_icscavimp`): rebuild
`dump_ic_tables`, run it, then
`python build_reference_icscavimp_tables.py <jcm>/jcm/data/wetdep`.
