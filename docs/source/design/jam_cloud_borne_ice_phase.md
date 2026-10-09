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
  wet scavenging uses.
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
ledger decides the release, unchanged.

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
- Those clouds carry 2-4 crystals per litre against 10⁴-10⁶ particles per
  litre, so the one-particle-per-crystal rule leaves them empty.

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
the anvil's snow removes it.

**The rejected alternative.** CAM's liquid-only rule (`microp_aero.F90`: the
activation cover `lcldn = cldn·qc/(qc + qi)`) also empties the cirrus. It
leaves the anvils empty too, so convectively detrained aerosol at the anvil
level has no in-cloud sink. Run from the January state with it, sea salt above
150 hPa at 15-30°S reached 6-9 mg/m² within 10 days (12 µg/kg at 91 hPa), where
the anvil uptake keeps it below 0.1.

## Related

The transport half of the interaction, a local (Bermejo–Conde) fixer in place
of the global proportional one, is #1062.
