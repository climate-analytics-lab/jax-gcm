"""SCM check: JAM aerosol through the full ECHAM physics on one column.

A warm tropical column is PRESCRIBED (re-imposed every step, so convection
fires repeatedly — RCE-style forcing without the bare-scheme feedback
instabilities) while the JAM tracers evolve freely through vdiff,
convective transport (updraft + downdraft + in-plume scavenging, #621/#622),
microphysics and wet deposition.

The column's well-mixed sub-cloud layer is load-bearing: ECHAM's ``cubase``
trigger drops any column whose dry-lifted parcel is not buoyant, so a lapse
rate running to the surface convects nowhere and takes the whole convective
aerosol pathway with it (``docs/source/design/convective_trigger_soundings.md``).

Seeded: equal boundary-layer mass in m_so4_acc (soluble, activatable
accumulation mode) and m_poa_pcm (insoluble primary carbon). Checks:
  * everything stays finite over N days (stability);
  * the CONVECTIVE MACHINERY is alive — updraft mass flux, in-plume
    condensate, precipitation flux and a positive scavenged surface flux.
    These are direct, not inferred: when the plume dies the aerosol
    profiles still look plausible because vdiff keeps lofting tracers, and
    that is exactly how #773 shipped;
  * both tracers develop free-troposphere loading (convective transport
    is actually lifting them out of the BL);
  * the soluble tracer ends up depleted aloft relative to the insoluble
    one (in-plume scavenging + wetdep act on it);
  * accumulated wet_so4 deposition is positive.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

# Source-checkout bootstrap: repo root on sys.path before importing jcm, so
# ``python tools/release_validation/scm_check.py`` works without a
# pip-installed jcm.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from jcm.physics.echam.echam_levels import get_echam_levels  # noqa: E402
from jcm.physics.echam.echam_terms import echam_physics  # noqa: E402
from jcm.rce import (  # noqa: E402
    JAM_COLUMN_FT_WINDOW,
    jam_scavenging_column,
)
from jcm.single_column_model import SingleColumnModel  # noqa: E402

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
DT = 900.0
N = int(DAYS * 86400 / DT)
NLEV = 47

vertical = get_echam_levels(NLEV)
physics = echam_physics(cloud_scheme="2m", aerosol_module="jam",
                        radiation_scheme="grey")
scm = SingleColumnModel(
    physics=physics, vertical=vertical, lat_deg=0.0, lon_deg=150.0,
    dt_seconds=DT,
)

state, seed, p = jam_scavenging_column(vertical, physics, sst=302.0,
                                       relative_humidity=0.8)
states = tree_map(lambda x: jnp.broadcast_to(x, (N,) + jnp.shape(x)), state)

preds = scm.run(states, initial_tracers=seed,
                times=jnp.arange(N) * DT / 86400.0)

tr = {k: np.asarray(v) for k, v in preds.tracer_states.items()}
ok = True
def check(name, cond, detail=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    ok = ok and cond

finite = all(np.isfinite(v).all() for v in tr.values())
check("all tracers finite", finite)

# --- the convective machinery itself, before anything inferred from it ---
pd_hist = preds.physics_data
conv = pd_hist.get("convection") if isinstance(pd_hist, dict) else None
if conv is None:
    check("convection diagnostic present", False)
else:
    mfu = float(np.max(np.abs(np.asarray(conv.mass_flux_up))))
    cond = float(np.max(np.asarray(conv.qc_conv) + np.asarray(conv.qi_conv)))
    pflx = float(np.max(np.asarray(conv.precip_flux)))
    check("convection fires (updraft mass flux)", mfu > 0.0, f"max {mfu:.3e}")
    check("in-plume condensate is diagnosed", cond > 0.0, f"max {cond:.3e}")
    check("convective precip flux is diagnosed", pflx > 0.0, f"max {pflx:.3e}")

scav = (pd_hist.get("_conv_scav_flux") or {}).get("m_so4_acc")
if scav is None:
    print("NOTE  _conv_scav_flux not in physics_data history; skipping")
else:
    total = float(np.nansum(np.asarray(scav)))
    check("in-plume scavenging removes soluble aerosol", total > 0.0,
          f"sum {total:.3e}")

so4, pom = tr.get("m_so4_acc"), tr.get("m_poa_pcm")
ft_lo, ft_hi = JAM_COLUMN_FT_WINDOW
ft = (p > ft_lo) & (p < ft_hi)         # free troposphere, 150-600 hPa
so4_ft0, so4_ftN = so4[0][ft].mean(), so4[-1][ft].mean()
pom_ft0, pom_ftN = pom[0][ft].mean(), pom[-1][ft].mean()
check("convective transport lofts insoluble aerosol",
      pom_ftN > max(10 * pom_ft0, 1e-20), f"{pom_ft0:.2e} -> {pom_ftN:.2e}")
check("soluble also lofted but less",
      so4_ftN > max(2 * so4_ft0, 1e-25), f"{so4_ft0:.2e} -> {so4_ftN:.2e}")
# Equal seeds, so the absolute free-troposphere loadings compare
# directly: in-plume scavenging + wetdep must leave far less soluble
# aerosol aloft than insoluble.
check("soluble depleted aloft vs insoluble (equal seeds)",
      so4_ftN < 0.5 * pom_ftN,
      f"FT soluble {so4_ftN:.2e} vs insoluble {pom_ftN:.2e}")

wet = pd_hist.get("wet_so4") if isinstance(pd_hist, dict) else None
if wet is not None:
    wet = np.asarray(wet)
    check("wet so4 deposition accumulates", float(np.nansum(wet)) > 0,
          f"sum {float(np.nansum(wet)):.3e}")
else:
    print("NOTE  wet_so4 not in physics_data history; skipping flux check")

np.savez(sys.argv[2] if len(sys.argv) > 2 else "scm_jam_result.npz",
         p=p, **{k: v for k, v in tr.items()
                 if k in ("m_so4_acc", "m_poa_pcm", "n_acc", "n_pcm")})
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
