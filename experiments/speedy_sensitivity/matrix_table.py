"""Combine growth.py and climatology.py outputs into one markdown table.

  python matrix_table.py runs --ref T31-L8-eul --fit 5 40 > matrix.md

One row per configuration tag found as both growth-<tag>.npz and/or
clim-<tag>.npz. Growth: OLR-tangent rate (1/day) for the tropical /
extratropical bump, fitted as in plot_growth.py. Climatology: global means of
OLR, absorbed SW, net TOA, total precipitation and cloud cover, the 4-seed
standard deviation of net TOA (a noise scale for the differences), and the
peak zonal-mean zonal wind near sigma = 0.25 with its latitude, per
hemisphere. Each climatology column is followed by its difference from
``--ref`` in parentheses.
"""
import argparse
import pathlib
import re
import warnings

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dir")
ap.add_argument("--ref", default="T31-L8-eul")
ap.add_argument("--fit", type=float, nargs=2, default=(5.0, 40.0))
args = ap.parse_args()
warnings.filterwarnings("ignore", category=RuntimeWarning)
d = pathlib.Path(args.dir)


def growth(tag):
    f = d / f"growth-{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    j = list(z["obs_names"]).index("olr")
    out = []
    for bump in ("tropical", "extratropical"):
        v = z[bump][j]
        day = np.arange(1, v.size + 1)
        m = (day >= args.fit[0]) & (day <= args.fit[1]) & np.isfinite(v) & (v > 0)
        out.append(0.5 * np.polyfit(day[m], np.log(v[m]), 1)[0] if m.sum() > 2 else np.nan)
    return out


def clim(tag):
    f = d / f"clim-{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    names = list(z["obs_names"])
    g = z["global_mean"]
    olr, sw = g[:, names.index("olr")], g[:, names.index("swtoa")]
    pr = g[:, names.index("precnv")] + g[:, names.index("precls")]
    sig, lat = z["sigma"], z["lat"]
    k = int(np.argmin(np.abs(sig - 0.25)))
    u = z["u_zonal"].mean(0)[k]
    nh, sh = lat > 0, lat < 0
    return dict(olr=olr.mean(), sw=sw.mean(), net=(sw - olr).mean(), net_sd=(sw - olr).std(),
                pr=pr.mean(), cc=g[:, names.index("cloudc")].mean(),
                jn=u[nh].max(), jn_lat=lat[nh][u[nh].argmax()], js=u[sh].max(), js_lat=lat[sh][u[sh].argmax()],
                finite=int(z["finite"].sum()), n=z["finite"].size)


tags = sorted({re.sub(r"^(growth|clim)-", "", f.stem) for f in d.glob("*.npz") if re.match(r"(growth|clim)-", f.stem)
               and not re.search(r"-(a[0-9.]+|g[0-9.]+)$", f.stem)},
              key=lambda t: (t.split("-")[0], int(t.split("-")[1][1:]), t))
ref = clim(args.ref)
print("| config | growth trop / extra (1/day) | OLR | abs. SW | net TOA (sd) | precip (mm/d) | cloud | NH jet m/s @lat | SH jet m/s @lat | finite |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t in tags:
    g, c = growth(t), clim(t)
    gs = f"{g[0]:.3f} / {g[1]:.3f}" if g and np.isfinite(g).all() else ("blew up" if g else "-")
    if c is not None and c["finite"] == 0:
        print(f"| {t} | {gs} | blew up | | | | | | | 0/{c['n']} |")
        continue
    if c is None:
        print(f"| {t} | {gs} | | | | | | | | |")
        continue

    def col(k, fmt):
        return f"{c[k]:{fmt}} ({c[k] - ref[k]:+{fmt}})" if ref else f"{c[k]:{fmt}}"
    print(f"| {t} | {gs} | {col('olr', '.0f')} | {col('sw', '.0f')} | {col('net', '.1f')} ({c['net_sd']:.1f}) | "
          f"{col('pr', '.2f')} | {col('cc', '.2f')} | {c['jn']:.0f} @{c['jn_lat']:.0f} | {c['js']:.0f} @{c['js_lat']:.0f} | "
          f"{c['finite']}/{c['n']} |")
