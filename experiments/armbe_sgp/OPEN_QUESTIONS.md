# Open questions — ARMBE single-column experiment

Things I found. Written 2026-07-16, before any real ARM data existed.

Related: `../../SCM_FORCING_PATCH_NOTE.md` at the repo root covers the patch I
made to `jcm/single_column_model.py` and what's still unfinished there.

## Triage — read this first

Mitchell asked the right question: is any of this real, or is it all just the
synthetic fixture? Answer: mostly fixture, but two things survive. Sorted by
whether they depend on the data at all.

**Real, provable from code — nothing to do with the fixture:**

- The season bug, already fixed. `forcing.select` was never called so `tyear`
  stayed 0 and every run used January insolation. See the patch note.
- SPEEDY has no diurnal cycle. `speedy_shortwave.py:256` reads `tyear` only.
- **Two setup bugs in `run_scm.py`, now fixed** (item 3): `lfluxland` defaulted
  to False so land fluxes were never computed while `fmask=1` selected exactly
  that tile — the atmosphere saw zero surface fluxes, silently. And I had the
  tile index backwards: land is tile **0**, not 1.
- **`rlds` is nearly insensitive to water vapour** — item 2. Demonstrated by a
  humidity sweep, no fixture involved. **This is the only unexplained model
  finding still standing**, and the surface-flux fixes above didn't move it
  (still 195.8).
- Variable names are still guesses (item 5). Patch has no tests yet.

**Fixture artifacts — don't chase these:**

- Constant rain (item 1). The fixture is convectively unstable in 94% of hours
  *by construction*. Real soundings won't be.
- Flat lines day to day (item 3b). Every hourly profile comes from the same
  formula with only a diurnal temperature cycle and small noise, so day to day
  the atmosphere barely changes. A flat model response is arithmetic, not a
  finding. I over-read this one.
- Every bias/RMSE number (item 4). The obs are generated independently of the
  profiles.

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

## 2. REAL: downward longwave barely responds to water vapour

This is the one genuine model finding in this doc. It does not involve the
fixture.

**The test.** Hand-built columns, surface air at 300 K, sweeping surface specific
humidity across a 30x range:

| q_sfc | rlds |
|---|---|
| 1 g/kg (desert) | 190.0 W/m² |
| 5 | 191.9 |
| 10 | 194.2 |
| 19 | 200.8 |
| 30 g/kg (tropical) | 211.5 |

**Why that's wrong.** Downward LW at the surface is dominated by emission from
near-surface water vapour. Going desert -> tropical should swing it roughly
250 -> 450 W/m². Here 30x the vapour buys 11%. The absolute value is also low:
~196 for a warm moist column where ~400 is expected. Net LW at the surface comes
out at 443.8 − 195.8 = 248 W/m² against a realistic 50–100.

**Ruled out:**

- Not the fixture — the humidities above were chosen directly.
- Not spin-up or radiation sub-stepping — the 168-step run gives 195.8 and a
  3-step run gives ~200. Consistent regardless of length.

**Still open.** Either SPEEDY's longwave is genuinely this crude (it's a 4-band
scheme, and it *is* an intermediate-complexity model — but 11% for 30x vapour
seems too crude even so), or something in the SCM's LW path isn't wired up. Note
`_longwave_rad` only ever exposes `dfabs` and `ftop` in physics_data, which is a
much thinner set than the SW term carries. Worth a look at
`jcm/physics/radiation/speedy_longwave.py` next.

If it's the former, it just bounds what this experiment can claim about LW. If
it's the latter, it's a bug worth reporting.

---

## 3. FIXED: two real setup bugs, and a chain of wrong conclusions

Chasing `shf ~ 0` turned up two genuine bugs in `run_scm.py`. Both are fixed.
Worth reading how wrong I got this on the way, because the failure mode is quiet.

**Bug A — land fluxes were never computed.**
`TerrainData.single_column` defaults to `lfluxland=False`, and I didn't set it.
The entire land-flux branch is behind
`jax.lax.cond(lfluxland, land_fluxes, pass_fn)` (`speedy_surface_flux.py:228`),
so the land tile stayed zeros. Meanwhile `fmask=1` *selects* the land tile
(line 264). Net effect: **the atmosphere saw zero surface fluxes, silently.**
Nothing errors. Fix: `lfluxland=True`.

**Bug B — I had the tile index backwards.**
`speedy_surface_flux.py:264` blends `var[:,:,1] + fmask*(var[:,:,0] - var[:,:,1])`.
So tile 0 is **land**, tile 1 is **sea**, and `fmask` is the land fraction. I'd
been reading tile 1 — the sea tile — and calling it land.

**How I fooled myself.** I "verified" tile 1 was land because `rlus[1]` = 443.8
matched sigma*T^4 = 452.9 (emissivity 0.98). That check is worthless for the
question I asked it: Stefan-Boltzmann holds for *any* surface, so it cannot
distinguish tiles. It felt like confirmation and wasn't.

Then I "proved" the land scheme responds to land temperature by sweeping
`stl_am` and watching shf move −25.6 -> +21.1. But that sweep set `stl_am` *and*
`sea_surface_temperature` to the same value, so I was watching the **sea** scheme
respond to **SST**. Two confident conclusions, both wrong, from one bug.

**What actually caught it.** Making `stl_am` a `TimeSeries` changed `tsfc`
(std 0 -> 4.2) but left `shf` *byte-identical* (`np.array_equal` -> True). A
13 K surface swing that moves the flux by exactly zero isn't a subtle
discrepancy — it's proof the flux never looked at that number. Identical output
is a much better alarm than an implausible one.

**After both fixes:**

| field | tile 0 (land) | tile 1 (sea) |
|---|---|---|
| `rlus` | 432.1, std 18.2, range 404–461 | 443.8, std 0.0001 (frozen) |
| `shf` | −5.3, std 3.9 | −2.6, std 11.6 |

Land `rlus` now tracks the surface temperature, as it should.

**What's left is the fixture.** `shf` is still ~0 because I drive `stl_am` from
the fixture's `temp_sfc`, which is labelled *surface **air** temperature*. So
land temperature == air temperature -> zero gradient -> zero flux. That's my
synthetic data, not the model.

Real ARMBE has a distinct skin temperature, and if there's no explicit variable,
derive it from ARMBECLDRAD's upwelling longwave: `T_skin = (LWup / (eps*sigma))**0.25`.
That's the right fix when data lands — don't feed air temperature in as ground
temperature.

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
