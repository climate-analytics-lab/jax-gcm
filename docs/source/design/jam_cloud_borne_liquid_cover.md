# JAM cloud-borne aerosol: the liquid cloud cover

`CloudBorneExchange` (`jcm/physics/aerosol/jam/cloud_borne.py`) moves aerosol
between the advected interstitial tracers and the cloud-borne phase that lives
in the physics carry. This page records which cloud cover that exchange acts in,
and why the choice matters far more for this model's upper troposphere than the
size of the cover alone suggests.

## The rule

Droplet activation acts in the liquid part of the cloud:

```
lcf = cf · qc / (qc + qi)          (0 where the cell holds no condensate)
```

with `cf` the post-microphysics cloud fraction and `qc`, `qi` the cloud scheme's
grid-mean condensate. This is CAM's `lcldn` (`microp_aero.F90`), the cover
`ndrop.F90::dropmixnuc` activates into in the CAM5 behaviour CAM6 keeps without
pre-existing ice.

- **Activation.** Under liquid cloud the cloud-borne amount relaxes toward the
  activated partition `f_act · (q_int + q_cb)` on `τ / lcf`. ARG's per-mode
  fractions supply `f_act`.
- **Ice-only cover** (`cf > 0`, `lcf = 0`). The target is zero, and the
  reservoir drains to the interstitial phase on the resuspension timescale.
  This is CAM's resuspension of a shrinking liquid cloud. Without it, aerosol
  activated while the cloud still held liquid would stay in the unadvected,
  unsedimented reservoir under a persistent ice deck with no exit.
- **Cover gone** (`cf = 0`). The evaporation/formation ledger decides
  (unchanged): evaporated droplets release their aerosol, and rained- or
  snowed-out ones leave it to wet deposition.

## Why the ice cover matters

The cloud-borne reservoir is neither advected nor sedimented (CAM keeps `qqcw`
in `pbuf` for the same reason). A reservoir that also filled under ice cloud
would act as a filter fixed in space wherever upper-level ice persists:

- Thin cirrus at the extratropical tropopause pulls the interstitial aerosol
  out of the air moving through it at `cf/τ`. Because ARG's activated mass
  fraction for the accumulation and coarse modes is close to one, the
  reservoir can grow to many times the local interstitial amount.
- Where the ice evaporates, the reservoir is returned as a thin interstitial
  layer at the cloud level. In a T63L47 JAM run this produced single-level
  sea-salt spikes of ~3 mg/kg at 237 hPa over the Southern Ocean, fifty times
  the surface mixing ratio below.
- In the austral-winter polar vortex the 198 K ice cloud at 130-170 hPa held
  50-77 % of the sea salt in the layer core in the reservoir itself.

Thin, sharp layers then interact with the semi-Lagrangian transport. The
quasi-monotone limiter creates mass where it clips the cubic interpolant's
undershoots beside sharp minima. The global proportional fixer returns that
mass in proportion to the field everywhere (#1062). The edges of a sharp
upper-level aerosol layer are therefore where the transport adds mass, taken
from the smooth boundary layer.

One SL step from a JAM state with such layers created 0.6 % of the coarse sea
salt held above 300 hPa at 40-60°S. In the per-term budget of the
40-90°S column above 300 hPa, the dynamics source scaled with the layer's own
burden at about 0.3 per day. With activation confined to liquid cloud the
same budget has a dynamics source of about 0.02 per day of the burden, and the
layer never forms. Over 110 days from the January state, sea salt above
300 hPa stays at 0.1-1.3 mg/m² at 40-60°S and ~0.1 mg/m² over the polar cap,
where the ice-cover rule reached 30-230 and 12-300 mg/m².

A realistic polar vortex is what makes this a trap rather than a transient. In
a circulation whose winter polar cap stays warm (216-223 K at 120-200 hPa) and
cloud-free, the lower stratosphere is flushed within weeks and the layers do
not survive the season. In a cold, isolated vortex with persistent ice cloud
(202-210 K) the layer has no precipitating sink, and the reservoir part of it
does not settle at all.

## Alternatives considered

- **HAM's ice-phase rule** (`mo_ham_wetdep.f90::ic_scav_nuc`, `nwetdep = 3`)
  puts one aerosol particle into each ice crystal, largest first. In ice cloud
  that is a small fraction of the accumulation and coarse modes, so it would
  remove the reservoir's ice-cloud filling much as the liquid cover does. It
  would also need the crystal number inside the exchange. The exchange is
  CAM's construct, so its CAM rule is used.
- **A local SL mass fixer** (the IFS's Bermejo–Conde fixer, Diamantakis &
  Flemming 2014) attacks the transport half of the interaction. It is the
  subject of #1062, because it changes the transport of every nodal tracer
  including cloud ice. With activation confined to liquid cloud, the
  upper-level aerosol does not build up without it.
