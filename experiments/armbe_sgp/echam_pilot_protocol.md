# ECHAM Layer-Cloud Pilot Protocol

## Scientific Question

Can a compact, condensate-aware equation improve hourly ECHAM layer cloud
fraction over native Sundqvist cloud fraction when evaluated against ARM SGP
observations?

This is Grundner-inspired feature exploration, not a reproduction of the
DYAMOND training experiment or a claim of global validity.

## Frozen Pilot Scope

- Site: ARM Southern Great Plains central facility, `sgp C1`.
- Period: 1-30 June 2018 UTC.
- Pipeline inspection day: 1 June 2018. Inspection may diagnose data handling,
  but it must not be used to select equations or tune performance thresholds.
- Online host: ECHAM 1-moment physics with RRTMGP.
- Candidate insertion point: replace Sundqvist cloud fraction before ECHAM
  1-moment microphysics and RRTMGP.
- One initial model sample: `(site, observed ARMBEATM profile time, ECHAM L47
  layer)`. Use every populated atmospheric profile at its actual timestamp.
  Cadence is nominally six-hourly but may contain three to five profiles per day.
  A profile requires observed temperature, humidity, dewpoint, and surface
  pressure at that timestamp. Incomplete profiles are excluded rather than
  filled through time.
- Primary target: layer cloud fraction in `[0, 1]`, never column-total cloud.

The source products are production `sgpmicrobaseC1.c1`, annual
`sgparmbeatmC1.c1`, and annual `sgparmbecldradC1.c1`. Source filenames,
checksums, versions, and exclusions must be retained in the dataset manifest.

## Split Policy

Whole UTC days are indivisible groups. Every height, ECHAM layer, feature
variant, and averaging-window variant derived from a day stays in one split.

- Training: 1-16 June 2018.
- Validation: 17-21 June 2018.
- Outer holdout: 22-30 June 2018.

The outer holdout must not be read by feature selection, symbolic search,
coefficient tuning, threshold selection, or candidate selection. The June 1
inspection is part of the training period. A later multi-site campaign must
reserve a separate leave-site-out test; this pilot cannot establish spatial
generalization.

## Observational Operators

The primary comparison window is 60 minutes centered on each ARMBE timestamp.
Sensitivity operators use centered 15-, 30-, 120-, and 360-minute windows.
The final cache must retain valid high-frequency sample counts and must not turn
missing instrument coverage into clear sky.

Two MICROBASE occurrence definitions are retained until comparison with ARMBE
and manual cases is complete:

- Strict primary: `retrieval_flag == 1` is cloudy, `retrieval_flag == 0` is
  clear, and all other values are excluded.
- Inclusive sensitivity: flags 1 and 2 are cloudy, flag 0 is clear, and all
  other values are excluded. Flag 2 admits possible cloud/clutter contribution.

MICROBASE and ARMBE cloud fields are prepared for all 24 hourly windows so the
same observational cache can support a later hourly INTERPSONDE expansion. The
initial training dataset selects only windows whose center exactly matches a
populated ARMBEATM profile; it does not interpolate atmospheric profiles in
time. ARMBE `cld_frac / 100` is the archived hourly reference. Primary ARMBE rows
require `qc_cld_frac == 0`; QC value 1 is retained only as a sensitivity case,
and values 2-4 are excluded. Temporal occurrence is only a proxy for ECHAM
horizontal grid-cell fraction, so averaging-window and wind-distance
sensitivity must be reported.

## Condensate Policy

MICROBASE liquid and ice fields are concentrations in `g m-3`. Clear cells
(`retrieval_flag == 0`) are physical zero condensate. For a cloudy cell in the
primary condensate view, `retrieval_flag == 1` and both relevant bit-packed QC
values must be zero. Flag 2, bad bit 6, missing values, and unsupported retrieval
flags are excluded rather than set to zero.

The 2025 MICROBASE technical report, DOE/SC-ARM-TR-095 Sections 1.2, 2.2, and
4.2, defines the output as instantaneous microphysical profiles calculated in
each ARSCL time-height bin. It is therefore neither an hourly mean nor an ECHAM
grid-cell mean. A detected cloudy bin contains the retrieved cloud-water
concentration for that instrument volume; a valid flag-0 clear bin contributes
physical zero. Convert each valid four-second concentration using its
contemporaneous density before averaging mixing ratio over the target window.
Do not divide an hourly mean concentration by an hourly mean density in the
final adapter.

After that gate is resolved, conversion to ECHAM mass mixing ratio is:

```text
qc = 1e-3 * liquid_water_concentration / air_density
qi = 1e-3 * ice_water_concentration / air_density
```

Air density must use collocated pressure, temperature, and moisture. For the
initial dataset, temperature and humidity come only from populated ARMBEATM
profiles; linear interpolation from their labelled 45 m height grid to the 30 m
cloud grid is allowed, but temporal interpolation is not. ARMBEATM has surface
pressure but no pressure profile on its height grid, so pressure is reconstructed
hydrostatically and later checked against INTERPSONDE. INTERPSONDE is an optional
hourly expansion and validation source, not a prerequisite for the observed-profile
baseline.

The baseline hydrostatic operator integrates on ARMBEATM's native 45 m height
grid from observed surface pressure. Actual vapor pressure is calculated from
sounding dewpoint using liquid-water saturation, avoiding an undocumented
ice-versus-water interpretation of the MWR-scaled relative-humidity field.
Log-pressure is then interpolated vertically to the 30 m cloud grid, where moist
ideal-gas density and mixing ratios are calculated. For a fixed observed
atmospheric snapshot, dividing the hourly mean concentration by snapshot density
is algebraically identical to converting each four-second concentration with
that same density and then averaging.

The June 1 MICROBASE `precip_flag` is entirely missing. Precipitation therefore
remains unknown for that day, not false. No precipitation-conditioned training
or claim is allowed until the monthly behavior and the product convention are
resolved.

## Permitted Inputs

The core feature group is incoming-state relative humidity, temperature,
vertical relative-humidity gradient, cloud liquid `qc`, and cloud ice `qi`.
Pressure or hybrid-layer location may be used by declared ablations. Every
feature must exist before cloud fraction executes online. Variables diagnosed
later by microphysics, target cloud fraction, precipitation outcomes, and
radiative outputs are prohibited inputs.

The derivative coordinate, smoothing, standardization, bounds, and feature
ablations are frozen later in the feature contract. Standardization is fitted
on training days only.

## Metrics

The primary offline score is layer cloud-fraction RMSE in fraction units, with
equal weight per valid profile before averaging profiles. Required companion
metrics are bias, mean absolute error, correlation, and calibration by predicted
cloud-fraction bin. Report low/mid/high layers, clear/cloudy regimes, averaging
windows, and strict/inclusive occurrence definitions separately.

Raw predictions and predictions after the predeclared physical bounds and
condensate gate are reported separately. Online promotion additionally requires
finite gradients, conservation checks, and no unacceptable TOA/surface radiation
regression.

## Stage Gates

The one-day native-height audit may proceed now. Month-scale caching is blocked
until precipitation handling and a pressure-profile source are resolved. ECHAM
L47 remapping is additionally blocked until condensate mass can be checked
before and after remapping.
