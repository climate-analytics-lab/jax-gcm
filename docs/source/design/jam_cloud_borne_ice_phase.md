# JAM cloud-borne aerosol: the ice phase

`CloudBorneExchange` (`jcm/physics/aerosol/jam/cloud_borne.py`) moves aerosol
between the advected interstitial tracers and the cloud-borne phase that lives
in the physics carry. This page records how much of the in-cloud aerosol that
phase holds in ice cloud, and why the answer matters far more for the upper
troposphere than the ice share of the cover suggests.

## The rule

Each interstitial/cloud-borne pair relaxes, on a timescale stretched by the
inverse cloud fraction, toward

```
q_cb* = f · (q_int + q_cb),      f = (1 − p_ice)·f_ARG + p_ice·f_ice
```

This is ECHAM-HAM's treatment of in-cloud aerosol (`mo_ham_wetdep.f90`, HAM2.3,
`nwetdep = 3`) applied to the explicit phase:

- **Phase split.** `ham_wetdep` divides the in-cloud tracer into a water and an
  ice part by the ice share `pice` of the in-cloud condensate
  (`prep_wetdep_hydro`). jcm takes `p_ice` from the same process-time pool its
  wet scavenging uses. Where that ledger is empty in a covered cell it takes the
  share of the grid-mean condensate instead, so ice is never read as liquid.
- **Water part.** `f_ARG` is the per-mode ARG droplet-activation fraction,
  number and mass apart.
- **Ice part.** `ic_scav_nuc` (`kwat_phase = 2`) puts one aerosol particle in
  each ice crystal, filling the soluble modes from the coarse down.
  - The number share is `min(1, max(0, ICNC − N_larger)/N_mode)`.
  - The mass share is the log-normal mass tail beyond the radius whose number
    tail is that share (`ham_m7_invertlogtail`, `ham_m7_logtail`). It is
    computed in closed form as `½·erfc(erfc⁻¹(2 f_n) − 3 ln σ/√2)`; no radius is
    needed.
  - Modes that cannot activate get nothing (HAM's insoluble modes).
- **Crystal number.** ICNC is the two-moment scheme's in-cloud crystal number
  per kg (`qni`, ECHAM's `idt_icnc`). It is advanced by the step's upstream
  tendencies, so it is the post-microphysics value HAM's wet deposition reads.
  Without `qni` the term raises rather than reading an ice cloud as
  crystal-free.

Under persistent ice cloud with fewer crystals than particles, `f` is small and
the reservoir drains to the interstitial phase on the resuspension timescale.
Once the whole cover has gone, the microphysics' evaporation and formation
ledger decides the release.

**Removal by phase.** HAM scavenges each part at its own conversion: the water
part by the liquid conversion `peffwat`, the ice part by `peffice`. The
exchange records each reservoir's droplet-held share, `(1 − p_ice)·f_ARG / f`,
under a step-local key.
- Wet deposition removes that share at the liquid conversion and the rest at
  the ice conversion. A single condensate-weighted fraction would instead rain
  out crystal-held aerosol at the liquid rate.
- Aqueous chemistry forms sulfate on the droplet-held cloud-borne number only.
  The sulfate it forms is added to its reservoir's droplet-held share, so
  wet deposition removes it at the liquid conversion.
- Without the explicit phase, wet deposition applies the same split directly,
  `cf·[(1 − p_ice)·f_ARG·peffwat + p_ice·f_ice·peffice]`.

## Why the ice phase matters

The cloud-borne reservoir is neither advected nor sedimented (CAM keeps `qqcw`
in `pbuf` for the same reason). Wherever the ice phase is taken to hold
aerosol, that reservoir becomes a filter fixed in space.

**Cirrus and the polar-vortex ice cloud.** Filled by droplet activation
(`f_ARG` is close to one for the accumulation and coarse modes), a thin cirrus
deck at the extratropical tropopause pulls the aerosol out of the air moving
through it. It returns the aerosol as a thin interstitial layer where the ice
evaporates.
- In a T63L47 JAM run this produced single-level sea-salt spikes of ~3 mg/kg
  at 237 hPa over the Southern Ocean, fifty times the surface mixing ratio.
- In the austral-winter polar vortex the reservoir held 50-77 % of the sea
  salt in the core of the 198 K ice-cloud layer.
- Because the crystals fill the coarse mode first, what matters is ICNC
  against the coarse number. Those clouds carry 2-4 crystals per litre:
  - 6·10⁵ coarse particles per litre in the vortex layer and 4·10⁴ in the spike
    is a coarse number share of 10⁻⁵-10⁻⁴;
  - the largest particles go first, so that is a mass share of a few per cent
    (3 % at a share of 1.7·10⁻⁴, σ = 1.8);
  - nothing is left for the accumulation mode.

**The interaction with the transport.** Thin, sharp layers are where the
semi-Lagrangian quasi-monotone limiter creates mass. The global proportional
fixer returns it in proportion to the field everywhere (#1062).
- One SL step from a JAM state with such layers created 0.6 % of the coarse
  sea salt above 300 hPa at 40-60°S.
- In the per-term budget of the 40-90°S column above 300 hPa, the dynamics
  source then scaled with the layer's own burden at about 0.3 per day.
- A realistic polar vortex made the layer a trap rather than a transient. The
  cold, isolated cap (202-210 K at 120-200 hPa, with ice cloud) has no
  precipitating sink there. A warm, cloud-free cap (216-223 K) is flushed
  within weeks.

**Convective anvils** carry hundreds to thousands of crystals per litre against
0.1-30 coarse and 10³-10⁴ accumulation particles per litre. The rule therefore
puts the whole coarse mode and much of the accumulation mode into the ice, and
the anvil's snow removes it. In 5-day means of a T63L47 January, the in-ice
mass share in tropical anvils (130-250 hPa) is 0.95 for the coarse mode and
0.52 for the accumulation mode.

**The rejected alternative.** CAM's liquid-only rule (`microp_aero.F90`: the
activation cover `lcldn = cldn·qc/(qc + qi)`) also empties the cirrus. It
leaves the anvils empty too, so convectively detrained aerosol at the anvil
level has no in-cloud sink. Run from the January state with it, sea salt above
150 hPa at 15-30°S reached 6-9 mg/m² within 10 days (12 µg/kg at 91 hPa),
against the 4-5 mg/m² spin-up peak of the crystal-number rule below.

## Spin-up from states built under droplet activation in ice

Warm states made before this rule hold an upper and middle troposphere shaped
by the stronger uptake: droplet activation in every cloud, including
mixed-phase and ice cloud whose crystal number is below the coarse number.
Started from such a state, the crystal-number rule scavenges less of the
subtropical mid-tropospheric aerosol on its way up. Part of it reaches the
tropical tropopause and lower stratosphere before the column adjusts.

From the January state (T63L47 JAM, defaults), sea salt above 150 hPa at
15-30°S, against the same run under droplet activation in ice:

| day | 5 | 10 | 20 | 30 | 50 | 70 | 90 | 125 | 155 |
|---|---|---|---|---|---|---|---|---|---|
| crystal-number rule, mg/m² | 4.0 | 4.9 | 1.4 | 0.63 | 0.17 | 0.07 | 0.04 | 0.02 | 0.01 |
| droplet activation in ice, mg/m² | 0.00 | 0.00 | 0.02 | 0.09 | 0.79 | 0.18 | 0.04 | 0.10 | 0.07 |

- The excess falls by a factor of eight between days 10 and 30, and is at or
  below the old rule's value from day 50 on.
- The 150-300 hPa band at 15-30°S starts at 3 mg/m² and is below the old
  rule's value from day 20.
- North of the equator (5-30°N) the pulse is smaller: 0.29 mg/m² at day 20,
  0.04 by day 70. From the July state it is 0.48 mg/m² at day 10 and gone by
  day 20.
- Sulfate above 100 hPa follows the same course: 1.7 mg/m² at day 15, 0.57 at
  day 45, 0.21 at day 75 and 0.10 at day 155, against the 0.05-0.11 that the
  old rule holds there.

It is an adjustment of the initial state, not a property of the rule. A state
spun up under the rule does not carry it, and 30-day calibration windows
should start from such states.

## Related

- **The transport half of the interaction** is #1062: a local (Bermejo–Conde,
  IFS) fixer in place of the global proportional one. It changes the transport
  of every nodal tracer including cloud ice. With the crystal-number rule the
  upper-level aerosol does not build up without it. Added to the rule, it
  reduced the spin-up transient above by only 15-25 %, so the transient's
  source is the uptake, not the amplifier.
- **HAM's in-cloud impaction** (`ic_scav_imp`: collision of interstitial
  aerosol with cloud droplets and ice plates) is the other half of HAM's
  in-cloud fraction. jcm applies it to the interstitial aerosol of every
  stratiform cloud (see the in-cloud impaction section of
  {doc}`../science/aerosol`).
  It does not take over the uptake this rule leaves out of crystal-poor ice
  cloud. At the crystal numbers and radii of the January and July T63 L47
  states the crystals collect a burden-weighted 1-2·10⁻⁴ of the interstitial
  coarse dust per step, and impaction alone removes it on a timescale of
  decades.
