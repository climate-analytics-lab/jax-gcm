# Open questions — ARMBE single-column experiment

Things I found but couldn't settle. Written 2026-07-16, before any real ARM data
existed, so read every number below as "measured on a synthetic fixture I made
up." That's a big caveat and it applies to all of it.

Related: `../../SCM_FORCING_PATCH_NOTE.md` at the repo root covers the patch I
made to `jcm/single_column_model.py` and what's still unfinished there.

---

## 1. The model rains constantly and I don't know why

**What I see.** On the fixture the column rains ~12 mm/hr, every hour, for the
whole week. The observations are wet 4% of the time.

**Facts, measured:**

- Convection fired on 168 of 168 steps. 100%.
- The rate barely moves: 10.3–12.8 mm/hr, std 0.73.
- The fixture's convective instability over the same week swings a lot: surface
  theta_e minus saturated theta_e at 500 hPa runs −1.3 to +26.9 K, unstable in
  94% of hours.

**Why that's odd.** A rain rate that sits flat while instability swings ~20x
isn't the scheme tracking CAPE. Something else is setting that number.

**Candidates, none verified:**

- The mass-flux closure saturating against the fixture's very moist surface layer
  (~19 g/kg the whole time).
- The static surface forcing (see `run_scm.py` — surface temp, soil moisture and
  albedo are fixed at record means, they don't follow the obs hour to hour).
- Diagnostic mode throwing away convection's stabilizing tendency each step, so
  CAPE never gets consumed. This was my first theory. It doesn't fit the flat
  rate on its own.
- The fixture just being unphysical.

**What I got wrong.** I asserted the diagnostic-mode explanation confidently
before testing it. Mitchell pushed back — if we feed new data every step, why
would it always rain? — and he was right. I'd gone looking for confirmation
instead of trying to break the story. The stability numbers above are what
falsified it.

**How to settle it.** Real ARMBE profiles. The fixture is unstable 94% of hours
*by construction*, which real soundings won't be, so some unknown fraction of
this is my formulas and not the method. Synthetic data can't separate the two.
When real data lands, check: does precip become intermittent, and does the rate
start tracking instability? If it needs the convective feedback to score fairly,
the SCM already supports `relaxation_timescales={"temperature": tau,
"specific_humidity": tau}` — the column evolves under its own physics while
staying tied to the obs.

---

## 2. Downward longwave at the surface looks low

`rlds` = 196 W/m². For a column with a 299 K surface and ~19 g/kg of water vapour
in the boundary layer I'd expect something like 350–400.

Cross-check: net LW at the surface comes out at 443.8 − 195.8 = 248 W/m². Real
net LW is more like 50–100. So it's not just that I'm misreading the field —
the number really is out of range.

Could be the fixture (my profile is very cold aloft — I capped it at 215 K).
Could be SPEEDY's longwave being simplified — it's a 4-band scheme. Could be
radiation sub-stepping: the SW term carries `compute_shortwave` and `step` slots,
so radiation may not run every step, and `single_column_model.py` references
issue #470 about that carry's step counter being off by one under `nstrad > 1`.
Haven't checked any of these.

Checkable against SPEEDY's radiation code without real data, if it's worth the
time.

---

## 3. Sensible heat flux is small and negative

`shf` on the land tile = −2.59 W/m² mean, range −21 to +16. For June afternoons
at SGP I'd expect daily-mean sensible heat well positive, order 50 W/m².

Latent heat is fine, which makes this stranger: `evap` = 0.0585 g/m²/s → 146
W/m², which is about right for daily-mean June. So the surface scheme isn't
broken across the board — just this field.

Prime suspect is the static forcing. Surface temperature is pinned at the record
mean (298.9 K), so the land-air temperature difference that drives sensible heat
never develops a diurnal swing. Untested.

---

## 3b. The real tell: the model is flat in *every* field

Look at `outputs/compare.png`. The model line is flat in all five panels — precip,
SW down, LW down, sensible heat, latent heat — across the whole week, while the
obs vary in each.

That reframes items 1–3. They're probably not five separate problems. The model
isn't responding to day-to-day variation in *anything*, which points at one
common cause upstream of all of them. Best guess is the static forcing: surface
temperature, soil moisture and albedo are pinned at record means in
`run_scm.py`, so every surface-driven field is nailed down, and SPEEDY's
insolation only moves with season (not day to day within a week).

Caveat that cuts the other way: the fixture's obs are generated independently of
its profiles, so there's genuinely nothing for the model to track. A flat model
line against uncorrelated noise is what you'd expect even if everything worked.

So: don't chase items 1–3 separately yet. Check the common cause first. Cheapest
test that doesn't need real data — make the forcing follow the record
(`ForcingData` takes `TimeSeries` leaves and `select()` already slices them per
step, see the patch note) and see whether the flat lines start moving.

---

## 4. The fixture's "observations" are meaningless as a target

`make_synthetic_armbe.py` generates the obs series (SW, LW, precip, fluxes) from
formulas that have nothing to do with the profiles it generates. There is no
physical consistency between the two.

So every bias/RMSE number `evaluate.py` currently prints is noise. The fixture
tests plumbing — does the loader resolve names, does the interpolation land on
the right levels, does the model run, do the units line up. It cannot test
whether the model is any good. Don't let those numbers into a slide.

Also: n = 7 daily values. The 0.88 precip correlation is nothing.

---

## 5. ARMBE variable names are still guesses

ARM's docs describe ARMBE's contents in prose ("Dry Bulb Temperature", "Eastward
Wind Component") but don't publish the netCDF variable names, and they differ
across sites and versions. I couldn't find them anywhere.

So `armbe_io.CANDIDATES` is an ordered list of plausible names per field, and the
loader picks whichever exists. It'll probably need extending on first contact
with a real file. Run `python armbe_io.py <file.nc>` — it prints every variable
in the file and what each canonical field resolved to.

Confirmed by hand from the ADC metadata pages:

- Both datastreams exist: `sgparmbeatmC1.c1`, `sgparmbecldradC1.c1`.
- SGP C1 is at 36.607322 N, −97.487643 W.
- ARMBE ships **dewpoint**, not specific humidity. The loader derives q.

One thing I got wrong here too: the `sgparmbeatmC1.c1` metadata page says its
record ends 2016-12-31, and I briefly treated that as fact and nearly hardcoded
it as a date-range check. ARM's own news says SGP ARMBE has been extended through
2019/2020 and beyond, so the page is just stale. Don't hardcode coverage — ask
the API. `download.py` already does, and reports what it actually finds.

---

## 6. Known limitations that aren't bugs

- **No diurnal cycle, ever.** SPEEDY's shortwave reads `forcing.solar.tyear` —
  fraction of year — and nothing else. Daily-mean, zonally-averaged insolation by
  design. That's why `evaluate.py` scores daily means. If you want hourly, that
  means ECHAM, whose radiation computes a real cos(zenith).
- **Forcing is static apart from the sun.** Surface temperature, soil moisture,
  albedo are set once from record means. `ForcingData` supports `TimeSeries`
  leaves and `select()` slices them per step, so following the obs hour by hour
  is possible — just not done.
- **Blocked:** no ARM data yet. Login is failing; support has been emailed. The
  likely cause is that the ARM username isn't the email address — there's a
  username-reminder tool at https://adc.arm.gov/armuserreg/#/forgot.
