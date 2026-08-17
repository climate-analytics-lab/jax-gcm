"""SCM check: JAM aerosol through the full ECHAM physics on one column.

A warm tropical column is PRESCRIBED (re-imposed every step, so convection
fires repeatedly — RCE-style forcing without the bare-scheme feedback
instabilities) while the JAM tracers evolve freely through vdiff,
convective transport (updraft + downdraft + in-plume scavenging, #621/#622),
microphysics and wet deposition.

Seeded: equal boundary-layer mass in m_so4_acc (soluble, activatable
accumulation mode) and m_poa_pcm (insoluble primary carbon). Checks:
  * everything stays finite over N days (stability);
  * both tracers develop free-troposphere loading (convective transport
    is actually lifting them out of the BL);
  * the soluble tracer ends up depleted aloft relative to the insoluble
    one (in-plume scavenging + wetdep act on it);
  * accumulated wet_so4 deposition is positive.
"""
import sys

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.rce import rce_initial_state
from jcm.single_column_model import SingleColumnModel

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

# Warm tropical column via the shared RCE builder (same saturation
# closure the model uses), with a light background wind.
state = rce_initial_state(vertical, sst=302.0, relative_humidity=0.8)
state = state.copy(u_wind=jnp.full(NLEV, 3.0))

# Pressure centers for the level masks below.
ps = 101325.0
a_b = np.asarray(getattr(vertical, "a_boundaries", np.zeros(NLEV + 1)))
b_b = np.asarray(getattr(vertical, "b_boundaries", np.linspace(0, 1, NLEV + 1)))
ph = a_b + b_b * ps
p = 0.5 * (ph[:-1] + ph[1:])

states = tree_map(lambda x: jnp.broadcast_to(x, (N,) + jnp.shape(x)), state)

# Seed every declared JAM tracer at ~0; load the BL (lowest 5 levels)
# of the soluble/insoluble pair plus matching number.
names = []
for term in physics.terms:
    for spec_ in term.required_tracers():
        if spec_.name not in names:
            names.append(spec_.name)
bl = np.zeros(NLEV)
bl[-5:] = 1.0
seed = {nm: jnp.full(NLEV, 1e-30) for nm in names}
for nm, val in (("m_so4_acc", 2.0e-9), ("m_poa_pcm", 2.0e-9),
                ("n_acc", 2.0e8), ("n_pcm", 2.0e8)):
    if nm in seed:
        seed[nm] = jnp.asarray(bl * val + 1e-30, dtype=jnp.float32)
seed["qc"] = jnp.zeros(NLEV)
seed["qi"] = jnp.zeros(NLEV)

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
so4, pom = tr.get("m_so4_acc"), tr.get("m_poa_pcm")
dmw = np.asarray(p) * 0 + 1.0  # equal weights; qualitative profile checks
ft = (p > 150e2) & (p < 600e2)         # free troposphere, 150-600 hPa
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

pd_hist = preds.physics_data
wet = None
if isinstance(pd_hist, dict):
    wet = pd_hist.get("wet_so4")
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
