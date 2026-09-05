"""Make N weight-perturbed copies of a trained term, to measure the eval noise floor.

Why this exists. `online_term_t31_big_clim_vt` and `online_term_t31_big_insol_vt`
differ by only 0.2-0.3% relative RMS in every layer, carry identical static
config, and the `insol` context row they differ by is itself 0.19% of the
trained profile rows, so the input is numerically inert. Yet the two score up
to 0.28 K apart on the holdout. That says the free-run climate metric moves
under weight changes far too small to be doing any physics, so before ranking
any two terms we need to know how much of that gap is just the metric's own
noise.

Each copy gets fresh Gaussian noise scaled to `--rel` relative RMS per array,
matching the observed clim-vs-insol distance. Everything else in the file is
copied through untouched, including keys this script has never heard of, so a
perturbed term is config-identical to its parent by construction.

    python tools/bias_correction/perturb_term.py <in.npz> <out_prefix> --n 5 --rel 0.002

Then score each copy with evaluate_term.py and take the spread of the five metrics.
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("in_npz")
ap.add_argument("out_prefix", help="writes <prefix>_p0.npz ... <prefix>_p<n-1>.npz")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--rel", type=float, default=0.002,
                help="perturbation size as a fraction of each array's own RMS")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src = np.load(a.in_npz)
# Perturb the trainable leaves only. in_mean/in_std/out_scale are frozen
# normalisation buffers, and moving them would change the term's units rather
# than its trajectory, which is not the question being asked.
trainable = [k for k in src.files
             if k.startswith("kernel_") or k.startswith("bias_")]
print(f"{a.in_npz}: perturbing {len(trainable)} arrays at {a.rel:.4%} relative RMS")

rng = np.random.default_rng(a.seed)
for i in range(a.n):
    out = dict(src)
    for k in trainable:
        w = np.asarray(src[k], dtype=np.float64)
        rms = np.sqrt((w ** 2).mean())
        if rms == 0.0:
            continue          # zero-init rows have no scale to perturb relative to
        noise = rng.normal(size=w.shape)
        noise *= a.rel * rms / np.sqrt((noise ** 2).mean())
        out[k] = (w + noise).astype(src[k].dtype)
    path = f"{a.out_prefix}_p{i}.npz"
    np.savez(path, **out)

    # Report the achieved distance on the first layer so the log records what
    # was actually applied, not just what was requested.
    k0 = np.asarray(src["kernel_0"], dtype=np.float64)
    d = np.asarray(out["kernel_0"], dtype=np.float64) - k0
    print(f"  wrote {path}  kernel_0 relRMSdiff "
          f"{np.sqrt((d ** 2).mean()) / np.sqrt((k0 ** 2).mean()):.5f}")
