"""De-overlap the GFMIP FD patches into dN/dSST* maps and compare with GFMIP.

Per patch p: d_p = ocean-mean response per K of patch-mean SST (mean over members,
split into LW = -OLR and SW parts; N = SW - OLR). De-overlap (GFMIP protocol):
  dN/dSST*(x) = sum_p d_p dSST_p(x) / sum_p dSST_p(x)
Pattern correlation over ice-free ocean against each GFMIP model and their mean.
Reference maps: the GFMIP preliminary spatial feedbacks (one NetCDF per model with
``spatial_feedbacks_from_both`` or ``..._from_warming``, see
https://github.com/GFMIP/preliminary_spatial_feedbacks).

  python gfmip_analysis.py --truncation 31 --nlev 16 --gfmip_dir /path/to/preliminary_spatial_feedbacks/data --out runs
"""
import argparse
import pathlib

import matplotlib
import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
config.add_config_args(ap)
ap.add_argument("--gfmip_dir", default=None)
ap.add_argument("--out", default="runs")
args = ap.parse_args()

tag = config.config_tag(args)
out = pathlib.Path(args.out)
P = np.load(out / f"gfmip-patches-T{args.truncation}.npz")
du, mdu, w, ice, lats, lons = (P[k] for k in ("dsst_unit", "mean_dsst_unit", "ocean_weight",
                                              "ice_free_ocean", "lats", "lons"))
d = {c: np.full(du.shape[0], np.nan) for c in ("N", "LW", "SW")}
for f in sorted((out / f"gfmip-fd-{tag}").glob("patch_*.npz")):
    p = int(f.stem.split("_")[1])
    z = np.load(f)
    fl = list(z["fields"])
    olr = (z["sens"][:, fl.index("olr")] * w).sum((1, 2)).mean() / mdu[p]
    sw = (z["sens"][:, fl.index("swtoa")] * w).sum((1, 2)).mean() / mdu[p]
    d["N"][p], d["LW"][p], d["SW"][p] = sw - olr, -olr, sw
have = np.isfinite(d["N"])
print(f"{tag}: {have.sum()}/{du.shape[0]} patches")


def deoverlap(v):
    den = du[have].sum(0)
    return np.where(den > 1e-6, (v[have, None, None] * du[have]).sum(0) / np.maximum(den, 1e-12), np.nan)


gf = {c: deoverlap(v) for c, v in d.items()}


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (ice > 0.05)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


ref = {}
if args.gfmip_dir:
    for fn in sorted(pathlib.Path(args.gfmip_dir).glob("*.nc")):
        ds = netCDF4.Dataset(fn)
        v = "spatial_feedbacks_from_both" if "spatial_feedbacks_from_both" in ds.variables else "spatial_feedbacks_from_warming"
        la, lo, f2 = np.array(ds["latitude"][:]), np.array(ds["longitude"][:]), np.ma.filled(ds[v][:].astype(float), np.nan)
        ds.close()
        if la[0] > la[-1]:
            la, f2 = la[::-1], f2[::-1]
        lo2, f2 = np.r_[lo[-1] - 360, lo, lo[0] + 360], np.c_[f2[:, -1:], f2, f2[:, :1]]
        LG, AG = np.meshgrid(lons, lats, indexing="ij")
        ref[fn.stem] = RegularGridInterpolator((la, lo2), f2, bounds_error=False, fill_value=np.nan)(
            np.c_[AG.ravel(), LG.ravel()]).reshape(LG.shape)
    ref["MMM"] = np.nanmean(np.stack(list(ref.values())), 0)
    for k, v in ref.items():
        print(f"  r(N, {k:8s}) = {pearson(gf['N'], v):+.2f}")

panels = [(f"{tag} dN/dSST*", gf["N"]), ("LW part", gf["LW"]), ("SW part", gf["SW"])]
if "MMM" in ref:
    panels.insert(1, (f"GFMIP MMM (r = {pearson(gf['N'], ref['MMM']):+.2f})", ref["MMM"]))
fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4))
for ax, (title, f) in zip(axes, panels):
    im = ax.pcolormesh(lons, lats, np.where(ice > 0.05, f, np.nan).T, cmap="RdBu_r", vmin=-12, vmax=12, shading="nearest")
    ax.set_title(title, fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.8, label="W m$^{-2}$ K$^{-1}$")
fig.savefig(out / f"gfmip-atlas-{tag}.png", dpi=120, bbox_inches="tight")
np.savez(out / f"gfmip-atlas-{tag}.npz", **{f"gf_{k}": v for k, v in gf.items()}, d_p_N=d["N"], lats=lats, lons=lons)
print(f"saved {out}/gfmip-atlas-{tag}.png/.npz")
