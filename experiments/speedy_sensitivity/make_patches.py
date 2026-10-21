"""GFMIP cos^2 SST-patch lattice on a SPEEDY grid (Bloch-Johnson et al. 2024, JAMES).

  dSST_p(i) = A cos^2((pi/2)(phi_i-phi_p)/(dphi/2)) cos^2((pi/2)(theta_i-theta_p)/(dtheta_p/2))
  inside the box, 0 outside; dphi = 20 deg, dtheta_p = 80 deg (|phi_p|<=30) or 80/cos(phi_p).
  Centres at 0, +-10, ..., +-80 deg latitude; longitude spacing dtheta_p/2 (50% overlap);
  start longitude alternates 180/160 by band -> 109 patches.

Unit amplitude is stored; gfmip_fd.py scales by +-A.

  python make_patches.py --truncation 31 --out runs     -> runs/gfmip-patches-T31.npz
"""
import argparse
import pathlib

import numpy as np

import common

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--truncation", type=int, default=31)
ap.add_argument("--out", default="runs")
args = ap.parse_args()

model, coords, terrain = common.build_model(args.truncation, 8)
forcing = common.load_forcing(coords, frozen_bcs=False)
lons = np.rad2deg(np.asarray(coords.horizontal.longitudes))
lats = np.rad2deg(np.asarray(coords.horizontal.latitudes))
w = common.ocean_weights(coords, terrain, forcing)
sic = np.asarray(forcing.sice_am.values).mean(axis=0)
ice_free = (1.0 - np.asarray(terrain.fmask)) * np.clip(1.0 - sic, 0.0, 1.0)

DPHI = 20.0


def dtheta(phi):
    return 80.0 if abs(phi) <= 30.0 else 80.0 / np.cos(np.radians(phi))


centers = []
for phi in [0.0] + [s * p for p in (10, 20, 30, 40, 50, 60, 70, 80) for s in (1, -1)]:
    start = 180.0 if (phi == 0.0 or abs(phi) in (20, 40, 60, 80)) else 160.0
    step = dtheta(phi) / 2.0
    centers += [(phi, (start + k * step) % 360.0) for k in range(round(360.0 / step))]
centers = np.array(centers)


def cos2_patch(phi_p, theta_p):
    dphi = lats[None, :] - phi_p
    dth_i = (lons[:, None] - theta_p + 180.0) % 360.0 - 180.0
    dth = dtheta(phi_p)
    clat = np.where(np.abs(dphi) < DPHI / 2, np.cos((np.pi / 2) * dphi / (DPHI / 2)) ** 2, 0.0)
    clon = np.where(np.abs(dth_i) < dth / 2, np.cos((np.pi / 2) * dth_i / (dth / 2)) ** 2, 0.0)
    return clat * clon


dsst_unit = np.stack([cos2_patch(p, t) for p, t in centers])
mean_dsst_unit = (dsst_unit * w[None]).sum(axis=(1, 2))   # ocean-mean dSST per K of peak
keep = mean_dsst_unit > 1e-4
print(f"T{args.truncation}: {keep.sum()}/{len(centers)} patches touch ice-free ocean")
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"gfmip-patches-T{args.truncation}.npz",
         dsst_unit=dsst_unit[keep].astype(np.float32), centers=centers[keep].astype(np.float32),
         mean_dsst_unit=mean_dsst_unit[keep].astype(np.float32), ocean_weight=w,
         ice_free_ocean=ice_free.astype(np.float32), lats=lats.astype(np.float32), lons=lons.astype(np.float32))
print(f"saved {out}/gfmip-patches-T{args.truncation}.npz")
