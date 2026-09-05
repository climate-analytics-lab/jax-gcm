"""Post-hoc: load a trained term, add a vertical (surface) T taper, re-save.

Fades the near-surface temperature correction of an ALREADY-trained term
without retraining, so we can quickly test whether removing that (harmful)
piece fixes surface T while keeping the mid-trop and humidity wins. If it
helps, retrain with the same surface_taper for the polished result.

    python tools/bias_correction/add_surface_taper.py <in.npz> <out.npz> <sigma0> <sigma1>
"""
import sys

import numpy as np
import jax.numpy as jnp

from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.bias_correction.nn_bias_correction import (
    load_bias_correction, surface_taper_factor)

src, dst = sys.argv[1], sys.argv[2]
s0, s1 = float(sys.argv[3]), float(sys.argv[4])

# Show what the taper does per level so the profile is visible before saving.
sig = np.asarray(get_speedy_coords().vertical.centers)
w = np.asarray(surface_taper_factor(jnp.asarray(sig), s0, s1))
print("sigma centres :", np.round(sig, 3))
print("T taper weight:", np.round(w, 3), " (1 = full correction, 0 = off)")

base = load_bias_correction(src)
# rebuild() carries every static field across, so this script cannot drop one
# the way hand-listing them did: it silently lost `activation` (a gelu term
# came back as tanh) until that was caught.
tapered = base.rebuild(base.weights.get_value(), surface_taper=(s0, s1))
tapered.save(dst)
print(f"wrote {dst}  (surface_taper=({s0}, {s1}))")
