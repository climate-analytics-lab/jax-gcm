# Volcanic SO₂

JAM emits volcanic sulfur from two inventories, both on in the `echam-jam` and
`echam-jam-m7` presets (`jam_volcanic`):

- **continuous degassing** (CMIP6/CESM2 `contvolcano`), constant in time;
- **explosive eruptions** (CMIP7 `utsvolcemis`), each injected on its own day.

The science-register entry is the *Volcanic SO₂* section of
{doc}`../science/aerosol`. This page records the inputs, how they are placed
in the model, and what they change.

## Why it is there

Volcanic degassing is a large part of the pre-industrial sulfur background,
and that background sets how susceptible clouds are to the anthropogenic
increment. In the AeroCom AP4-CTRL climatological nudged pair at T63
(`echam-jam-m7`, 2010 against the 1850–1859 emissions):
- the pre-industrial SO₂ source is 4.7 Tg SO₂/yr (anthropogenic and biomass
  burning);
- DMS gives 24.5 Tg S/yr;
- degassing at the CMIP6 rate is 25.9 Tg SO₂/yr, 5.5 times that SO₂ source.

Without volcanic sulfur the pair's aerosol ERF was −2.04 W m⁻², with ERFaci
−1.86 W m⁻², beyond the AR6 assessed range. Clouds sitting near the
10 cm⁻³ CDNC floor signal missing aerosol and droplet sources, not a floor to
retune; the floor stays at 10 (maintainer decision, 2026-11-02).

The AP4-CTRL protocol also asks models to inject SO₂ from Carn (2022) rather
than prescribe stratospheric aerosol. The CMIP7 eruption list's satellite era
is Carn's MSVOLSO2L4.

## Inputs: one grid-free product

`products/volcanic_so2_cmip.nc` (`forcing.volcanic_file`, `auto` by default)
holds both sources as point lists. `jcm/data/mirror/volcanic.py` builds it from
two files: the CMIP6 degassing file from the CESM inputdata tree, and the CMIP7
eruption list from input4MIPs. The mirror's `volcanic` stage runs that build.

**Degassing.** `emissions-cmip6_SO2_contvolcano_vertical_850-5000_0.9x1.25_c20170724.nc`:
- The AeroCom `volc_continuous.nc` (Diehl, after Andres & Kasgnoc 1998),
  scaled ×1.5 to match Carn et al. (2017). The file is constant from 850 to
  5000 CE; the builder refuses one that is not.
- It is given in molecules cm⁻³ s⁻¹ on 20 hybrid layers whose surface
  pressure is fixed at 1013.25 hPa, i.e. fixed pressures. The builder reads
  them as ICAO standard-atmosphere altitudes (0–9.8 km a.s.l.), the column
  those pressures describe.
- Each non-zero 0.9°×1.25° cell becomes one source: 43 cells, 25.89 Tg SO₂/yr,
  mass-weighted mean altitude 2.3 km.
- The file's own description assumes 2.5 % goes directly to SO₄, the value
  JAM already uses for primary sulfate (`SO4_PRIMARY_FRACTION`, HAM's
  `zfacso2`).

**Eruptions.** `utsvolcemis_input4MIPs_emissions_CMIP_UOEXETER-CMIP-2-2-1_gn_17500101-20231120.nc`:
- 463 eruptions from 1750 to 2023, each with a day, position, injection
  altitude `H` (a.s.l.), injection depth `D` and SO₂ mass.
- Mean emission 1.14 Tg SO₂/yr over 1979–2023.
- Documented by Aubry et al. (2026, GMD preprint egusphere-2026-546).

**Choice of inventory.** Degassing uses CMIP6's Carn-scaled inventory, not
ECHAM-HAM's unscaled AeroCom one (Dentener et al. 2006), which is about
two-thirds of it (maintainer decision, 2026-11-02). It is the CMIP inventory
beside the CEDS and BB4CMIP7 emissions JAM reads, and the one CESM2 emits.

## Placement in the model

`jcm.forcing.read_volcanic_emissions` puts each source in the column that
contains it and divides its mass rate by that column's area, so the model's
own global integral returns the inventory's mass on any grid:
- **Gaussian grids:** cell boundaries and areas come from the quadrature
  weights, the areas the tracer mass fixer uses.
- **pySES:** each source goes to its nearest column; `FVPhysicsGrid.cell_areas`
  gives the column areas from the FV metric.

Sources sharing a column are summed. The forcing carries them as small static
arrays (`ForcingData.volcanic_emissions`), not daily fields.

`VolcanicEmissions` places them vertically by the column's geopotential height
above sea level (`height_half`):
- **Degassing.** Each inventory layer is split over the model layers it
  overlaps. A share below the model's smoothed surface goes into the lowest
  layer, and any share above the lid into the top one, so no mass is lost
  where a vent stands below the T63 orography.
- **Eruptions** follow Aubry et al. (2026, §V.2):
  - **Time:** the SO₂ is injected over the 24 hours of the eruption's UTC day,
    selected from `forcing.solar.calendar_year` and `day_of_year`.
  - **Place:** the column containing the volcano.
  - **Profile:** `exp(−((z − H)/D)²)`, integrated exactly over each layer
    with `erf`. A plume thin against the layers keeps both its mass and a
    smooth derivative in `H`. The tails below the surface and above the lid
    go to the end layers, as for degassing.

**Calendar.** The model calendar picks the eruptions, so a transient run emits
each year's own. `forcing.volcanic_eruption_year` replays one year's eruptions
every model year, for a climatological run whose calendar is not the year it
should see. The dataset's own guidance has no pre-industrial recommendation
yet; a PD/PI pair that shares a calendar gets the same eruptions in both runs.

**Primary sulfate.** Degassing's 2.5 % takes the population's own primary
sulfate targets:
- **MAM4:** `primary_split`, half Aitken and half accumulation. This is the
  split CESM gives the same inventory's sulfate (`so4_a2`/`so4_a1`).
- **M7:** HAM's default sector class, half soluble Aitken at 30 nm and half
  soluble accumulation at 75 nm. Every sector HAM's emission routine does not
  special-case takes this class.

Eruptions emit SO₂ only, as the CMIP7 list gives SO₂ mass.

The degassing and eruption scales and the primary-sulfate fraction are
differentiable (`volcanic_params`), as are the injection heights in the
forcing.

## Measured effect

The measurements use April 2010 arms of the AP4-CTRL NudClim pair (T63
`echam-jam-m7`, ERA5 winds nudged). Each arm resumes the production run's
1 March state, on the same code tree with and without the volcanic source.
Values are PD − PI in W m⁻² unless marked; the noise between nudged arms is
about 0.03 W m⁻².

| | without | with |
|---|---|---|
| ERF (March / April) | −1.57 / −1.88 | −1.46 / −1.83 |
| ERFaci (March / April) | −1.43 / −1.79 | −1.33 / −1.71 |
| ERFari (March / April) | −0.15 / −0.12 | −0.15 / −0.14 |
| PI SO₄ burden, April (mg m⁻²) | 1.67 | 2.79 |
| PD SO₄ burden, April (mg m⁻²) | 4.61 | 5.78 |
| PD net TOA (March / April) | 5.30 / 2.26 | 5.01 / 1.67 |

- **Forcing.** ERFaci weakens by 0.08–0.10 W m⁻². Volcanic sulfur is a real
  but small part of this pair's strong aerosol forcing: on its own it does not
  bring the ERF into the assessed range.
- **Vertical placement matters.** An earlier approximation emitted the same
  degassing through the biomass-burning channel, at 1 ± 1.5 km a.s.l. and so
  near the surface over elevated vents. It weakened April's ERFaci by 0.19,
  twice as much. That difference is why the inventory's own profile is kept.
- **Calibration.** The present-day TOA is 0.3–0.6 W m⁻² more reflective. The
  v3.1 M7 calibration (annual net TOA +2.2 W m⁻² against CERES +1.0) was
  fitted without volcanic sulfur.
- **MAM4 check.** A two-day `echam-jam` run from 14 April 2010 reproduces the
  inventory exactly:
  - On a day with no eruption, the emitted SO₂ rises by 25.24 Tg/yr: the
    97.5 % of 25.89 Tg/yr that stays gas.
  - On 14 April it rises by 27.07 Tg/yr, adding Eyjafjallajökull's 0.005 Tg
    emitted over that day.
  - The primary SO₄ rises by 1.16 Tg/yr.

## Limitations

- Degassing is constant in time; individual volcanoes' variability is not
  represented.
- The CMIP7 list ends in November 2023. A later run emits degassing only;
  scenario eruptions come from the ScenarioMIP dataset, which is not mirrored.
- Neither source carries ash, halogens or water vapour.
