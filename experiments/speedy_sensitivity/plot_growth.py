"""Plot tangent spatial variance vs time and fit growth rates from growth.py outputs.

  python plot_growth.py runs/growth-*.npz --obs olr --fit 5 40 --out growth.png

Growth rate lambda (1/day, natural log) = 0.5 x the slope of log(variance) over
the fit window; e-folding time = 1/lambda. Printed per file and bump.
"""
import argparse
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("files", nargs="+")
ap.add_argument("--obs", default="olr")
ap.add_argument("--fit", type=float, nargs=2, default=(5.0, 40.0), metavar=("DAY0", "DAY1"))
ap.add_argument("--out", default="growth.png")
args = ap.parse_args()

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for f in sorted(args.files):
    d = np.load(f)
    j = list(d["obs_names"]).index(args.obs)
    label = pathlib.Path(f).stem.removeprefix("growth-")
    for ax, bump in zip(axes, ("tropical", "extratropical")):
        v = d[bump][j]
        day = np.arange(1, v.size + 1)
        m = (day >= args.fit[0]) & (day <= args.fit[1]) & np.isfinite(v) & (v > 0)
        lam = 0.5 * np.polyfit(day[m], np.log(v[m]), 1)[0] if m.sum() > 2 else np.nan
        print(f"{label:28s} {bump:13s} lambda = {lam:+.3f}/day  e-fold = {1 / lam if lam > 0 else np.inf:.1f} d")
        ax.semilogy(day, v, label=f"{label} ({lam:+.2f}/d)")
for ax, bump in zip(axes, ("tropical", "extratropical")):
    ax.set_title(f"{args.obs} tangent spatial variance, {bump} bump", fontsize=10)
    ax.set_xlabel("day")
    ax.axvspan(*args.fit, color="0.92", zorder=0)
    ax.legend(fontsize=7)
axes[0].set_ylabel("area-weighted spatial variance of tangent")
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print(f"saved {args.out}")
