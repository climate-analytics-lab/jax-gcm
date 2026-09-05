"""Score every term in eval_out_holdout against the measured noise floor.

Why this exists. The five metrics are read off free-running 7-year climatologies,
and a 7-year mean does not average out internal variability. Five physics-free
0.2% weight perturbations of the shipped term (`perturb_term.py`) spread by
0.06-0.12 K depending on the metric, and consecutive 7-year blocks of one 21-year
run spread by 0.02-0.06 K, so a difference smaller than that is not a result.
Every metric can resolve the gaps between terms at this window length. What none
of them can see through is a reseed, which changes the network itself, so terms
are ranked within a seed (the PAIRED section) and never across one.

There are now three floors, measuring three different things, and they are
reported separately because they are not interchangeable:

  perturbation  jitter the weights, keep the trajectory. A lower bound.
  block         one term, one run, consecutive windows of it (`*_blkN` files
                from a long run). Network held fixed, realisation varied, so
                this is sampling noise alone: how much a metric moves for no
                reason at all at this window length.
  seed          retrain from a different random seed. Network AND realisation
                vary, so it is the largest, and it is the bar an unpaired
                comparison between two terms has to clear.

The gap between the block floor and the seed spread is the part of a reseed
that is genuinely a different network rather than a different realisation.

Every number is cos-lat weighted RMS, the headline statistic. The floors are
measured live from whichever runs are present rather than hardcoded, so they
stay honest as more are added.

    python tools/bias_correction/holdout_table.py [--dir eval_out_holdout] [--suffix _2016_2645d]
    python tools/bias_correction/holdout_table.py --dir eval_out_long --suffix _2016_7755d
"""
import argparse
import glob
import os
import re
import statistics as st

import numpy as np
import xarray as xr

METRICS = [("t_surf_annual", "T ann"), ("t_surf_djf", "T DJF"),
           ("t_surf_jja", "T JJA"), ("t_mid_annual", "T 500"),
           ("q_surf_annual", "q")]
REF = {"t_surf_annual": "era5_t2m_annual", "t_surf_djf": "era5_t2m_djf",
       "t_surf_jja": "era5_t2m_jja", "t_mid_annual": "era5_t500",
       "q_surf_annual": "era5_q925"}

# The 148k terms that share size, activation and recipe and differ only in the
# context feature. These are the ones a ranking claim is actually about.
COMPARABLE = ("big_clim_vt", "big_fmask_vt", "big_insol_vt", "big_sice_vt",
              "big_seas4_vt", "big_seasW_vt")

# (control, experiment) recipes that share a warm start and differ in one
# deliberate change, so they can be differenced within a seed.
PAIRED = (("big_clim_vt", "big_seas4_vt"),
          ("big_clim_vt", "big_seasW_vt"),
          ("big_clim_vt", "big_sice_vt"))

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="eval_out_holdout")
ap.add_argument("--suffix", default="_2016_2645d")
a = ap.parse_args()


def wrms(bias):
    """Cos-lat weighted RMS, matching evaluate_term.stats()."""
    b = np.asarray(bias)
    w = np.broadcast_to(np.cos(np.deg2rad(np.asarray(bias.lat))), b.shape)
    return float(np.sqrt((w * b * b).sum() / w.sum()))


refs = xr.open_dataset(os.path.join(a.dir, f"eval_fields_era5{a.suffix}.nc"))
scores = {}
for path in sorted(glob.glob(os.path.join(a.dir, f"eval_fields_*{a.suffix}.nc"))):
    tag = os.path.basename(path)[len("eval_fields_"):-len(f"{a.suffix}.nc")]
    if tag == "era5":
        continue
    ds = xr.open_dataset(path)
    row = {}
    for key, _ in METRICS:
        if key in ds and REF[key] in refs:
            row[key] = wrms(ds[key] - refs[REF[key]])
    if row:
        scores[tag] = row

# Blocks are windows of one run, not separate configs, so they are pulled out
# of the main table and reported as their own floor below. Left in, they would
# also be swept into the seed grouping, where they would masquerade as extra
# training runs.
BLOCK_RE = re.compile(r"_blk(\d+)$")
block_scores = {}
for tag in [t for t in scores if BLOCK_RE.search(t)]:
    base = BLOCK_RE.sub("", tag)
    block_scores.setdefault(base, {})[tag] = scores.pop(tag)

# Noise floor measured from whatever perturbation runs exist.
noise = {}
pert = [t for t in scores if t.startswith("noisefloor_p")]
for key, _ in METRICS:
    vals = [scores[t][key] for t in pert if key in scores[t]]
    noise[key] = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")

order = [t for t in ("plain", "clim_vt", "20yr_clim_vt", "big_clim_vt",
                     "big_fmask_vt", "big_insol_vt", "big_s7_insol_vt")
         if t in scores]
order += sorted(t for t in scores if t not in order and not t.startswith("noisefloor_p"))

hdr = "".join(f"{lbl:>8s}" for _, lbl in METRICS)
print(f"{'term':26s}{hdr}   beats plain")
print("-" * (26 + len(hdr) + 14))
plain = scores.get("plain", {})
for tag in order:
    row = scores[tag]
    cells = "".join(f"{row.get(k, float('nan')):8.2f}" for k, _ in METRICS)
    n = sum(1 for k, _ in METRICS
            if k in row and k in plain and row[k] < plain[k])
    beats = "" if tag == "plain" else f"   {n}/{len(METRICS)}"
    print(f"{tag:26s}{cells}{beats}")

print("-" * (26 + len(hdr) + 14))
print(f"{f'NOISE FLOOR (n={len(pert)})':26s}" +
      "".join(f"{noise[k]:8.2f}" for k, _ in METRICS))
block_floor = {}
if block_scores:
    print()
    print("SAMPLING FLOOR: one term, one run, consecutive windows of it.")
    print("Same weights throughout, so every difference below is the model's")
    print("own internal variability at this window length.")
    for base, members in sorted(block_scores.items()):
        print()
        print(f"{base:26s}" + "".join(f"{lbl:>8s}" for _, lbl in METRICS))
        for tag in sorted(members,
                          key=lambda t: int(BLOCK_RE.search(t).group(1))):
            row = members[tag]
            print(f"  {tag[len(base):].lstrip('_'):24s}"
                  + "".join(f"{row.get(k, float('nan')):8.2f}"
                            for k, _ in METRICS))
        gap = {}
        for key, _ in METRICS:
            vals = [members[t][key] for t in members if key in members[t]]
            if len(vals) > 1:
                gap[key] = max(vals) - min(vals)
                block_floor[key] = max(block_floor.get(key, 0.0), gap[key])
        print(f"  {'spread':24s}"
              + "".join(f"{gap.get(k, float('nan')):8.2f}" for k, _ in METRICS))
        if base in scores:
            print(f"  {'full window':24s}"
                  + "".join(f"{scores[base].get(k, float('nan')):8.2f}"
                            for k, _ in METRICS))
    if len(block_scores) > 1:
        print()
        print(f"{'WORST BLOCK SPREAD':26s}"
              + "".join(f"{block_floor.get(k, float('nan')):8.2f}"
                        for k, _ in METRICS))

print()
# A SECOND, larger noise estimate: the same recipe trained from a different
# random seed. This is not the same quantity as the perturbation floor above --
# isotropic jitter is a LOWER bound, while a different seed explores a
# genuinely different optimisation path, and it turns out to be far larger.
#
# Group terms that differ ONLY in --seed: `big_clim_vt`, `big_s1_clim_vt` and
# `big_s7_clim_vt` are the same recipe three times. Their spread is
# trajectory-to-trajectory variability, which is the bar any new term has to
# clear. Report each group, then the WORST across groups, because a metric is
# only safe to rank on if it is stable under every group available.
SEED_RE = re.compile(r"^big_s\d+_")
groups: dict = {}
for tag in scores:
    if not tag.startswith("big_"):
        continue
    canon = SEED_RE.sub("big_", tag)
    groups.setdefault(canon, []).append(tag)
groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}

worst = {}
for canon, members in sorted(groups.items()):
    gap = {}
    for key, _ in METRICS:
        vals = [scores[t][key] for t in members if key in scores[t]]
        if len(vals) > 1:
            gap[key] = max(vals) - min(vals)
    label = f"seed n={len(members)} {canon.replace('_vt', '')}"
    print(f"{label:26s}" +
          "".join(f"{gap.get(k, float('nan')):8.2f}" for k, _ in METRICS))
    for k, v in gap.items():
        worst[k] = max(worst.get(k, 0.0), v)
if len(groups) > 1:
    print(f"{'WORST SEED SPREAD':26s}" +
          "".join(f"{worst.get(k, float('nan')):8.2f}" for k, _ in METRICS))

if worst:
    print()
    # Compare the seed gap against the spread within the COMPETITIVE set only:
    # terms of the same size and recipe that differ in one deliberate choice.
    # Using every term instead would fold in the controls -- `notaper` sits at
    # DJF 5.33 because its taper was removed on purpose -- inflating every
    # range until each metric passes trivially. The question is whether the
    # terms we actually want to tell apart differ by more than a reseed.
    peers = [t for t in COMPARABLE if t in scores]
    print(f"Rankable metrics, over the {len(peers)} comparable terms "
          f"({', '.join(peers)}):")
    for key, lbl in METRICS:
        vals = [scores[t][key] for t in peers if key in scores[t]]
        rng = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        ok = rng > worst.get(key, 0.0)
        # The block floor is shown, not used in the verdict. The bar for an
        # unpaired comparison is still the seed spread, because two terms are
        # two training runs; the block column says how much of that bar is
        # sampling and would survive even with the seeds paired.
        floor = (f" [sampling {block_floor[key]:.2f}]"
                 if key in block_floor else "")
        print(f"  {lbl:6s} spread {rng:5.2f} vs seed {worst.get(key, 0.0):5.2f}"
              f"{floor}  -> {'RANKABLE' if ok else 'NOT rankable'}")

# --- paired comparison ------------------------------------------------------
# The unpaired test above is the wrong instrument once seed variance is this
# large: reseeding the same recipe moves annual by 0.89 and 500 hPa by 2.79, so
# almost nothing clears it point-vs-point. But a change can be tested WITHIN a
# seed -- run control and experiment from the same stage-2 term and difference
# them. The seed cancels, and what is left is the effect. A change is real if
# every seed moves the same way and the mean shift exceeds the spread OF THE
# DIFFERENCES, which is a much smaller quantity than the spread of the scores.
def seed_of(tag):
    m = re.match(r"^big_(s\d+)_", tag)
    return m.group(1) if m else "s0"


def paired(control_canon, exp_canon):
    """Per-seed deltas between two recipes that share a warm start."""
    ctl = {seed_of(t): t for t in scores
           if SEED_RE.sub("big_", t) == control_canon}
    exp = {seed_of(t): t for t in scores
           if SEED_RE.sub("big_", t) == exp_canon}
    return sorted(set(ctl) & set(exp)), ctl, exp


for control_canon, exp_canon in PAIRED:
    seeds, ctl, exp = paired(control_canon, exp_canon)
    if len(seeds) < 2:
        continue
    print()
    print(f"PAIRED: {exp_canon} minus {control_canon}, per seed "
          f"(negative = experiment better)")
    print(f"{'seed':26s}" + "".join(f"{lbl:>8s}" for _, lbl in METRICS))
    deltas = {k: [] for k, _ in METRICS}
    for sd in seeds:
        row = ""
        for key, _ in METRICS:
            if key in scores[ctl[sd]] and key in scores[exp[sd]]:
                d = scores[exp[sd]][key] - scores[ctl[sd]][key]
                deltas[key].append(d)
                row += f"{d:+8.2f}"
            else:
                row += "     -  "
        print(f"{sd:26s}{row}")
    print(f"{'mean shift':26s}" +
          "".join(f"{st.fmean(deltas[k]):+8.2f}" if deltas[k] else "     -  "
                  for k, _ in METRICS))
    print(f"{'spread of deltas':26s}" +
          "".join(f"{max(deltas[k]) - min(deltas[k]):8.2f}" if len(deltas[k]) > 1
                  else "     -  " for k, _ in METRICS))
    verdict = ""
    for key, lbl in METRICS:
        d = deltas[key]
        if len(d) < 2:
            continue
        same_sign = all(x < 0 for x in d) or all(x > 0 for x in d)
        beats = abs(st.fmean(d)) > (max(d) - min(d))
        verdict += (f"  {lbl:6s} {'CONSISTENT' if same_sign else 'sign flips'}"
                    f", mean {'>' if beats else '<='} spread"
                    f" -> {'REAL' if same_sign and beats else 'not established'}\n")
    print(verdict, end="")

print()
print("Read the two floors together. Perturbation noise bounds how much")
print("arbitrary weight jitter moves a metric; the seed pair bounds how much a")
print("different training run moves it, and that is the bar a new term must")
print("clear. Where the seed spread exceeds the range between the shipped")
print("terms, those terms cannot be ranked on that metric at all.")
if pert:
    print()
    for key, lbl in METRICS:
        vals = [scores[t][key] for t in pert if key in scores[t]]
        if len(vals) > 1:
            print(f"  {lbl:6s} perturbation sd {st.stdev(vals):.3f}")
