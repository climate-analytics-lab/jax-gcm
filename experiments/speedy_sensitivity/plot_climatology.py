"""Compare climatologies from climatology.py against a reference configuration.

  python plot_climatology.py runs/clim-*.npz --ref runs/clim-T31-L8-eul.npz --out clim.png

Prints global means (difference from the reference in parentheses) and plots
zonal means of OLR, absorbed SW, total precipitation and cloud cover, plus
zonal-mean zonal wind and temperature near sigma = 0.25 and 0.85.
"""
import argparse
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("files", nargs="+")
ap.add_argument("--ref", required=True)
ap.add_argument("--out", default="clim.png")
args = ap.parse_args()

ref = np.load(args.ref)
names = list(ref["obs_names"])
gref = ref["global_mean"].mean(0)
print(f"{'config':26s} " + " ".join(f"{n:>16s}" for n in names))
fig, ax = plt.subplots(2, 4, figsize=(18, 7))
ax = ax.ravel()
for f in [args.ref] + [f for f in sorted(args.files) if pathlib.Path(f).resolve() != pathlib.Path(args.ref).resolve()]:
    d = np.load(f)
    label = pathlib.Path(f).stem.removeprefix("clim-")
    g = d["global_mean"].mean(0)
    print(f"{label:26s} " + " ".join(f"{g[i]:8.3g} ({g[i] - gref[i]:+6.2g})" for i in range(len(names))))
    lat, z = d["lat"], d["zonal"].mean(0)
    for a, i, title in ((ax[0], names.index("olr"), "OLR (W/m2)"), (ax[1], names.index("swtoa"), "absorbed SW (W/m2)"),
                        (ax[3], names.index("cloudc"), "cloud cover")):
        a.plot(lat, z[i], label=label)
        a.set_title(title)
    ax[2].plot(lat, z[names.index("precnv")] + z[names.index("precls")], label=label)
    ax[2].set_title("total precipitation (mm/day)")
    sig = d["sigma"]
    for a, s, fld, title in ((ax[4], 0.25, "u_zonal", "zonal wind"), (ax[5], 0.85, "u_zonal", "zonal wind"),
                             (ax[6], 0.25, "t_zonal", "temperature"), (ax[7], 0.85, "t_zonal", "temperature")):
        k = int(np.argmin(np.abs(sig - s)))
        a.plot(lat, d[fld].mean(0)[k], label=label)
        a.set_title(f"{title}, sigma = {sig[k]:.2f}")
for a in ax:
    a.set_xlabel("latitude")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print(f"saved {args.out}")
