# Note on the single-column model forcing patch

**Changed:** `jcm/single_column_model.py`
**When:** 2026-07-16
**Why you're reading this:** I changed shared code that other people's runs depend on. Here's what and why.

## The bug

`SingleColumnModel` built its `ForcingData` once and reused that same object for every step of the scan. It never called `forcing.select(date)`.

`Model` does call it, every single step — see `model.py:436`. That call is what fills in `SolarGeometry` and slices any `TimeSeries` leaves.

A `ForcingData` you build yourself starts with a null `SolarGeometry`: `tyear=0`, `orbital_phase=0`, `synodic_phase=0`. The SCM never refreshed it, so `tyear` stayed 0 for the whole run. `tyear=0` means January 1.

So every SCM run used January insolation, no matter what date you thought you were simulating. At SGP (36.6N) that's 184 W/m² at TOA instead of 483 in late June — off by 2.6x. Nothing crashes. The numbers just come out wrong and look completely reasonable while doing it.

## What I changed

1. `__init__` now takes `start_date` and `calendar`, with the same defaults as `Model` (2000-01-01, `365_day`).
2. Added `_date_at_step()`, which builds a `DateData` for step i. It copies `Model._date_from_sim_time` deliberately — same floor/round split into days and seconds, same `stop_gradient`. The stop_gradient matters: date math uses floor/round/int casts, and leaving it in the AD graph would break `jax.grad` through a run.
3. The step function now calls `forcing.select(self._date_at_step(time_idx), calendar=self.calendar)` and hands that to `compute_tendencies`, instead of the frozen forcing from the closure.

## Why I think it's right

Daily-mean TOA insolation at SGP after the patch:

| date | fsol (TOA) |
|---|---|
| Jan 1 | 184 W/m² |
| Mar 21 | 352 |
| Jun 21 | 483 |
| Sep 21 | 354 |
| Dec 21 | 181 |

Solstice max and min land where they should. The two equinoxes agree to 0.6% (352 vs 354) — they have to, and that's a real check that the date/orbit math isn't skewed. Before the patch, all five of those rows read 184.

## What this does NOT fix

SPEEDY still has no diurnal cycle. That isn't a bug and I didn't try to change it.

`speedy_shortwave.py:256` reads `forcing.solar.tyear` and nothing else — fraction of *year*, not fraction of day. It computes daily-mean, zonally-averaged insolation. `synodic_phase` is ignored entirely. That's just how SPEEDY works.

If you want a diurnal cycle out of an SCM run, use ECHAM — its radiation computes a real `cos_zenith` from solar altitude.

Short version: this patch fixes the season. It does not add a diurnal cycle, and it can't.

## Open issues — read before trusting this

- **The default is still wrong.** If you don't pass `start_date`, you get 2000-01-01, which means January insolation — the same bug as before. I kept the default to stay backwards compatible, but anyone who doesn't set a date still gets the old behavior. Worth arguing about.
- **I haven't run the existing SCM test suite against this yet.** Do that first.
- **No test for the new behavior yet.** There should be one asserting insolation actually changes with `start_date`.
- **Only exercised with SPEEDY.** ECHAM goes through the same `select` path so it should be fine, but I haven't checked.
- Not reported upstream. This is a local patch for now.
