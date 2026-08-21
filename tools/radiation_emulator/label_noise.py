r"""Measure the irreducible McICA sampling noise in the emulator's labels.

The emulator cannot beat the noise in its own targets, so every RMSE reported
against RRTMGP labels has to be read against this floor. Without it there is no
way to tell a model that has stopped learning from one that has hit the limit
of what the labels can teach.

Method: two INDEPENDENT ``--n-seeds``-draw means of the same columns differ
only by McICA sampling, so the RMS of their difference over sqrt(2) is the
noise on one such mean. Draws are selected through ``model_step`` -- the traced
int32 folded into the PRNG key -- so the two blocks share one XLA trace.

Measured on the 2026-08 T63L47 training set (512 trajectory columns, 8 draws):
SW TOA up 4.70, SW surface down 5.16, LW TOA up 0.40, LW surface down 0.89
W/m^2. Against a model error of 18.4 W/m^2 on SW TOA upward flux, sampling
noise is ~6% of the error VARIANCE -- so at that skill level the labels are not
the limit, and paying for more draws would buy almost nothing. Re-measure
before concluding otherwise once model error approaches ~10 W/m^2.

Usage::

    python tools/radiation_emulator/label_noise.py \
        --data training/trajectory.nc --n-columns 512 --n-seeds 8
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from generate_training_data import (  # noqa: E402
    INTERFACE_FIELDS,
    LW_BAND_FIELDS,
    PROFILE_FIELDS,
    SCALAR_FIELDS,
    SW_BAND_FIELDS,
    make_labeller,
)

# (label, interface index, shortwave?) -- TOA is interface 0 and the surface is
# the last, because the stored columns are TOA-first.
PROBES = (
    ("SW TOA up", "sw_flux_up", 0, True),
    ("SW sfc down", "sw_flux_down", -1, True),
    ("LW TOA up", "lw_flux_up", 0, False),
    ("LW sfc down", "lw_flux_down", -1, False),
)


def seed_block_mean(labeller, batch, start, n_seeds):
    """Mean flux profiles over ``n_seeds`` draws starting at ``start``."""
    keys = [name for _, name, _, _ in PROBES]
    draws = [
        {k: np.asarray(v) for k, v in labeller(batch, s).items()}
        for s in range(start, start + n_seeds)
    ]
    return {k: np.mean([d[k] for d in draws], axis=0) for k in keys}


def main(argv=None):
    import xarray as xr

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="a generated training file")
    p.add_argument("--n-columns", type=int, default=512)
    p.add_argument("--n-seeds", type=int, default=8,
                   help="draws per mean; must match the training set's")
    p.add_argument("--base-seed", type=int, default=0)
    args = p.parse_args(argv)

    fields = (PROFILE_FIELDS + INTERFACE_FIELDS + SCALAR_FIELDS
              + SW_BAND_FIELDS + LW_BAND_FIELDS)
    ds = xr.open_dataset(args.data).isel(column=slice(0, args.n_columns))
    batch = {f: np.asarray(ds[f].values) for f in fields}
    labeller = make_labeller(args.base_seed)

    first = seed_block_mean(labeller, batch, 0, args.n_seeds)
    # Disjoint blocks, so the two means are independent.
    second = seed_block_mean(labeller, batch, 100 * args.n_seeds, args.n_seeds)

    lit = np.asarray(ds["cos_zenith"].values) > 0.0
    ncol = batch["temperature"].shape[0]
    print(f"{ncol} columns ({int(lit.sum())} lit); noise on a "
          f"{args.n_seeds}-draw mean (W/m2):")
    for label, name, index, is_sw in PROBES:
        # Dark columns carry no shortwave signal and would dilute the RMS.
        mask = lit if is_sw else np.ones(ncol, dtype=bool)
        diff = (first[name][:, index] - second[name][:, index])[mask]
        rms = float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(2.0))
        print(f"  {label:12s} RMS {rms:7.3f}   "
              f"max |diff| {float(np.abs(diff).max()):7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
