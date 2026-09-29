"""SCM check: JAM aerosol through the full ECHAM physics on one column.

A warm tropical column is PRESCRIBED (re-imposed every step, so convection
fires repeatedly — RCE-style forcing without the bare-scheme feedback
instabilities), starting under large-scale moisture convergence so the
plume is ECHAM's deep one, while the JAM tracers evolve freely through vdiff,
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
  * the insoluble tracer develops free-troposphere loading (convective
    transport is actually lifting it out of the BL);
  * the soluble tracer is lofted too, and less than the insoluble one, by
    the amounts HAMMOZ's convective scavenging implies: the two bounds
    below are derived from the per-mode in-condensate fractions and the
    precipitation efficiency the convection scheme diagnoses in this
    column;
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

from jcm.physics.aerosol.jam.wetdep.convective_fractions import (  # noqa: E402
    convective_csr,
)
from jcm.physics.echam.echam_levels import get_echam_levels  # noqa: E402
from jcm.physics.echam.echam_terms import echam_physics  # noqa: E402
from jcm.rce import (  # noqa: E402
    JAM_COLUMN_FT_WINDOW,
    convergent_initial_physics_data,
    jam_scavenging_column,
)
from jcm.single_column_model import SingleColumnModel  # noqa: E402
import jcm.constants as c  # noqa: E402

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
DT = 900.0
N = int(DAYS * 86400 / DT)
NLEV = 47

vertical = get_echam_levels(NLEV)
physics = echam_physics(cloud_scheme="2m", aerosol_module="jam")
scm = SingleColumnModel(
    physics=physics, vertical=vertical, lat_deg=0.0, lon_deg=150.0,
    dt_seconds=DT,
)

state, seed, p = jam_scavenging_column(vertical, physics, sst=302.0,
                                       relative_humidity=0.8)
states = tree_map(lambda x: jnp.broadcast_to(x, (N,) + jnp.shape(x)), state)

# The column starts under large-scale moisture convergence, which is what
# selects ECHAM's DEEP plume (see ``convergent_initial_physics_data``).
preds = scm.run(states, initial_tracers=seed,
                initial_physics_data=convergent_initial_physics_data(
                    scm, state),
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

# An absent ledger is a FAILURE, not a skip: losing the key is exactly the
# transport/output-wiring regression this check exists to catch, and the
# profile and wet_so4 checks below cannot stand in for it — stratiform
# scavenging and washout satisfy those on their own.
scav = (pd_hist.get("_conv_scav_flux") or {}).get("m_so4_acc")
if scav is None:
    check("in-plume scavenging removes soluble aerosol", False,
          "_conv_scav_flux[m_so4_acc] absent from the physics_data history")
else:
    total = float(np.nansum(np.asarray(scav)))
    check("in-plume scavenging removes soluble aerosol", total > 0.0,
          f"sum {total:.3e}")

so4, pom = tr.get("m_so4_acc"), tr.get("m_poa_pcm")
ft_lo, ft_hi = JAM_COLUMN_FT_WINDOW
ft = (p > ft_lo) & (p < ft_hi)         # free troposphere, 150-600 hPa
bl = p > 850.0e2                       # boundary layer, where both are seeded
pom_ft0, pom_ftN = pom[0][ft].mean(), pom[-1][ft].mean()
check("convective transport lofts insoluble aerosol",
      pom_ftN > max(10 * pom_ft0, 1e-20), f"{pom_ft0:.2e} -> {pom_ftN:.2e}")

# Soluble vs insoluble lofting, from HAMMOZ's convective scavenging.
#
# Air leaving the boundary layer in the plume carries a fraction ``csr`` of
# each aerosol tracer in the condensate (HAMMOZ ``csr_conv``: 0.99 for the
# accumulation mode, 0.20 for primary carbon); each cloudy layer removes
# the precipitation efficiency ``peff`` of that share from the air that
# continues through its top, and the rest rides up. A layer's detrained
# air leaves at the concentration the plume brought into it (cuasc's flux
# form), before that layer's conversion, so per unit of boundary-layer
# tracer the plume detrains into layer k
#
#     S_k(csr) = (1 − csr) + csr · Π_{cloudy j below k} (1 − peff_j),
#
# the same dilution by entrained air applying to both tracers. The two
# tracers start with equal boundary-layer seeds, and the soluble one never
# has more boundary-layer air to supply (compensating subsidence brings
# back the air it lost aloft), so its free-troposphere loading is at most
#
#     FT_sol / FT_ins ≤ max_FT S(csr_sol) / S(csr_ins)                (1)
#
# ("depleted aloft"). Dividing each loading by the tracer's own current
# boundary-layer value removes the supply difference: the soluble
# boundary layer decays faster, so the division flatters it, and the one
# sink the plume ratio leaves out is the scavenging of free-tropospheric
# air entrained into the cloudy plume, which removes at most ``csr_sol``
# of the soluble tracer in the fraction ``φ`` of the free-tropospheric air
# mass entrained by then. Hence ("lofted too")
#
#     (FT/BL)_sol / (FT/BL)_ins ≥ min_FT S(csr_sol)/S(csr_ins) · (1 − φ·csr_sol)   (2)
#
# Both are evaluated six hours in, while the boundary-layer seeds are
# still resolved (the no-source soluble tracer is gone from the whole
# column within days), with ``peff``, the cloud mask and the entrainment
# the convection scheme published at that step.
SNAP = min(int(6 * 3600 / DT), N) - 1
csr_sol, csr_ins = convective_csr("accum"), convective_csr("primary_carbon")
if conv is not None:
    peff = np.asarray(conv.precip_efficiency)[SNAP].reshape(-1)
    cond_snap = (np.asarray(conv.qc_conv) + np.asarray(conv.qi_conv))[SNAP]
    mfu_snap = np.asarray(conv.mass_flux_up)[SNAP].reshape(-1)
    cloudy = (cond_snap.reshape(-1) > 1e-10) & (mfu_snap > 0.0)
    # Π over the cloudy layers strictly below each layer (top-first axis).
    through = np.cumprod(np.where(cloudy, 1.0 - peff, 1.0)[::-1])[::-1]
    survive = np.append(through[1:], 1.0)
    s_ratio = (((1 - csr_sol) + csr_sol * survive)
               / ((1 - csr_ins) + csr_ins * survive))[ft]
    dm = np.diff(np.asarray(vertical.a_boundaries)
                 + np.asarray(vertical.b_boundaries) * float(c.p0)) / c.grav
    ent = np.asarray(conv.entrain_up)[SNAP].reshape(-1)
    phi = min(float(ent[ft].sum()) * (SNAP + 1) * DT / float(dm[ft].sum()), 1.0)
    so4_s, pom_s = so4[SNAP], pom[SNAP]
    raw = so4_s[ft].mean() / pom_s[ft].mean()
    norm = ((so4_s[ft].mean() / so4_s[bl].mean())
            / (pom_s[ft].mean() / pom_s[bl].mean()))
    lower = s_ratio.min() * (1.0 - phi * csr_sol)
    check("soluble also lofted but less", norm >= lower,
          f"(FT/BL) ratio {norm:.3e} >= {lower:.3e} "
          f"[min S ratio {s_ratio.min():.3e}, phi {phi:.3f}]")
    check("soluble depleted aloft vs insoluble (equal seeds)",
          raw <= s_ratio.max(),
          f"FT soluble/insoluble {raw:.3e} <= max S ratio {s_ratio.max():.3e}")

wet = pd_hist.get("wet_so4") if isinstance(pd_hist, dict) else None
if wet is None:
    check("wet so4 deposition accumulates", False,
          "wet_so4 absent from the physics_data history")
else:
    wet = np.asarray(wet)
    check("wet so4 deposition accumulates", float(np.nansum(wet)) > 0,
          f"sum {float(np.nansum(wet)):.3e}")

np.savez(sys.argv[2] if len(sys.argv) > 2 else "scm_jam_result.npz",
         p=p, **{k: v for k, v in tr.items()
                 if k in ("m_so4_acc", "m_poa_pcm", "n_acc", "n_pcm")})
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
