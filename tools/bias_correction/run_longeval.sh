#!/usr/bin/env bash
# Score terms on a long free run cut into consecutive 7-year blocks, to measure
# how much of the spread between terms is sampling noise.
#
# Why this is the first thing to run. Reseeding the same recipe moved annual
# surface temperature by 0.89 K and 500 hPa by 2.79 K, which is larger than the
# differences between every term we have trained. But a reseed changes the
# network AND the realisation, so it cannot say how much of its own spread is
# just a 7-year mean being short. Cutting one run into consecutive 7-year
# blocks holds the network fixed and varies only the realisation, so the spread
# across blocks IS the sampling floor. That is the number every past comparison
# has been missing.
#
# The free run is forced by a climatology, so its calendar year picks which
# ERA5 years the reference averages and nothing else. Running 21 years instead
# of 7 therefore costs three rollouts and buys down the model side of the noise
# without touching the observed side, which is common to every config anyway.
#
# Cost is one forward rollout per config, no training, no backprop.
#
#   GPU=0 tools/bias_correction/run_longeval.sh        # plain + the three 148k terms
#   YEARS=14 TERMS="big_insol_vt" tools/bias_correction/run_longeval.sh
#   PY=/path/to/python overrides the interpreter.
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
YEARS="${YEARS:-21}"
# The terms a ranking claim is actually about: same size, same recipe, one
# deliberate difference each. Plain SPEEDY is run once by evaluate_term itself.
TERMS="${TERMS:-big_clim_vt big_fmask_vt big_insol_vt}"
D=jcm/data/bias_correction
OUT="$PWD/eval_out_long"
HOLD="$PWD/eval_out_holdout"
DAYS=$((90 + YEARS * 365))
PY="${PY:-python}"

mkdir -p "$OUT"

# Reuse the reference the 7-year holdout was scored against rather than
# rebuilding it from WeatherBench2. Same ERA5 years and same grid, so block 0
# of this run is directly comparable to the published 7-year number instead of
# merely similar to it. The stamp is what lets evaluate_term recognise the file: the
# holdout reference may predate the attribute that records its span.
if [ -f "$HOLD/eval_fields_era5_2016_2645d.nc" ] \
   && [ ! -f "$OUT/eval_fields_era5_2016_${DAYS}d.nc" ] \
   && [ ! -f "$OUT/eval_fields_era5_holdout_refs.nc" ]; then
    $PY - "$HOLD/eval_fields_era5_2016_2645d.nc" \
          "$OUT/eval_fields_era5_holdout_refs.nc" <<'PY'
import sys

import xarray as xr

src, dst = sys.argv[1], sys.argv[2]
ds = xr.open_dataset(src).load()
ds.attrs.setdefault("ref_years", "2016-2022")
ds.attrs.setdefault("ref2_year", "none")
ds.to_netcdf(dst)
print("stamped holdout ERA5 reference into", dst)
PY
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export M4_YEAR=2016
export M4_DAYS="$DAYS"
export M4_OUT_DIR="$OUT"
# Segment the rollout so a 21-year run is not held in memory at once, and cut
# the scored span into the same 7-year windows the holdout used.
export M4_CHUNK_DAYS=1825
export M4_BLOCK_YEARS=7
# Pin the observed span. Derived it would land on 2016-2022 anyway, but pinning
# it means a change of run length can never quietly change what we score
# against.
export M4_REF_YEARS=2016-2022
export M4_REF2_YEAR=none

for tag in $TERMS; do
    echo "=== ${YEARS}-year eval: $tag ==="
    M4_TERM="$D/online_term_t31_${tag}.npz" M4_TAG="$tag" $PY tools/bias_correction/evaluate_term.py
done

echo "=== table ==="
$PY tools/bias_correction/holdout_table.py --dir "$OUT" --suffix "_2016_${DAYS}d"
echo "LONG EVAL DONE"
