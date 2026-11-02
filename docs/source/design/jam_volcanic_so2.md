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
- AeroCom continuous degassing is 12.6 Tg S/yr (25.2 Tg SO₂/yr), 5.3 times that
  SO₂ source.

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

**Degassing.** `emissions-cmip6_SO2_contvolcano_vertical_850-5000_0.9x1.25_c20170724.nc`,
built by NCAR's `create_files.ncl` (on Glade under
`/glade/u/home/lamar/CESM/CESM2/Volcanoes/`):
- **Inventory.** Diehl's `volc_continuous.nc` (after Andres & Kasgnoc 1998)
  scaled ×2.6325 to the AeroCom continuous total, 12.6 Tg S/yr = 25.2 Tg
  SO₂/yr (Dentener et al. 2006). The factor adds the non-SO₂ sulfur and
  Textor's ×1.5 correction for an underestimate; that ×1.5 is the "scaled by
  1.5 … Carn et al. (2017)" in the file's description, not a further factor.
- **Layers.** Each volcano is spread evenly over the 500 m layers from
  two-thirds of its vent altitude up to the vent, 0–10 km a.s.l. The file
  stores the layers as pressures `1013.25·exp(−z/7600)` hPa over a fixed
  surface, which the builder turns back into those altitudes.
- **Units and split.** The file is constant from 850 to 5000 CE (the builder
  refuses one that is not) and gives molecules cm⁻³ s⁻¹ over each layer's
  volume. It holds 97.5 % of the sulfur as SO₂; CESM's separate SO₄ files hold
  the other 2.5 %.
- **The product** inverts that construction with the script's own cell area,
  radius and molecular weight. Each non-zero 0.9°×1.25° cell becomes one
  source carrying the inventory's total sulfur as SO₂: 43 cells, 25.22 Tg
  SO₂/yr, mass-weighted mean altitude 2.2 km. The model's
  `so4_primary_fraction` (2.5 %, also HAM's `zfacso2`) restores CESM's split.

**Eruptions.** `utsvolcemis_input4MIPs_emissions_CMIP_UOEXETER-CMIP-2-2-1_gn_17500101-20231120.nc`:
- 463 eruptions from 1750 to 2023, each with a day, position, injection
  altitude `H` (a.s.l.), injection depth `D` and SO₂ mass.
- Mean emission 1.14 Tg SO₂/yr over 1979–2023.
- Documented by Aubry et al. (2026, GMD preprint egusphere-2026-546).

**Choice of inventory.** Degassing reads CESM2's CMIP6 build of the AeroCom
continuous inventory (maintainer decision, 2026-11-02), the CMIP input beside
the CEDS and BB4CMIP7 emissions JAM reads. Its total is AeroCom's, the
inventory ECHAM-HAM also draws its volcanic emissions from; the vertical spread
below each vent is CESM's.

**Fetching.** `auto` resolves this one file on every grid, so a JAM run that
used to need no download on a grid without mirror bundles (pySES, T42) now
fetches it at build time: prefetch it on a node with network, as for the
other inputs.

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
- **MAM4:** `primary_split`, half Aitken and half accumulation by mass, the
  mass split CESM gives the same inventory's sulfate (`so4_a2`/`so4_a1`). The
  number follows each mode's own size, as for the anthropogenic sulfate;
  CESM's number files assume one 0.134 µm size for both modes, so its Aitken
  number is far lower.
- **M7:** HAM's default sector class, half soluble Aitken at 30 nm and half
  soluble accumulation at 75 nm. Every sector HAM's emission routine does not
  special-case takes this class.

Eruptions emit SO₂ only, as the CMIP7 list gives SO₂ mass.

The degassing and eruption scales and the primary-sulfate fraction are
differentiable (`volcanic_params`), as are the injection heights in the
forcing.

## Measured effect

The measurements use March and April 2010 arms of the AP4-CTRL NudClim pair
(T63 `echam-jam-m7`, ERA5 winds nudged). Each arm resumes the production run's
1 March state, on the same code tree with and without the volcanic source.
Values are PD − PI in W m⁻² unless marked.

| | without | with |
|---|---|---|
| ERF (March / April) | −1.57 / −1.88 | −1.54 / −1.74 |
| ERFaci (March / April) | −1.43 / −1.79 | −1.40 / −1.60 |
| ERFari (March / April) | −0.15 / −0.12 | −0.15 / −0.15 |
| PI SO₄ burden, April (mg m⁻²) | 1.67 | 2.75 |
| PD SO₄ burden, April (mg m⁻²) | 4.61 | 5.75 |
| PD net TOA (March / April) | 5.30 / 2.26 | 4.97 / 1.76 |

- **Noise.** One month of a nudged arm resolves the forcing to about
  0.1 W m⁻². Two volcanic arms whose products differ by 3 % in mass, run from
  the same state, differ in ERFaci by 0.07 in March and 0.11 in April, in
  opposite directions.
- **Forcing.** ERFaci weakens by 0.03 in March and 0.19 in April: about
  0.1 W m⁻² across the volcanic arms, the same size as the noise. Volcanic
  sulfur is a real but small part of this pair's strong aerosol forcing: on its
  own it does not bring the ERF into the assessed range.
- **Vertical placement.** An approximation that emitted the same degassing
  through the biomass-burning channel, at 1 ± 1.5 km a.s.l., weakened ERFaci by
  0.08 in March and 0.19 in April. These arms cannot tell the two profiles
  apart; the inventory's own profile is used because it is the inventory's.
- **Calibration.** The present-day TOA is 0.3–0.5 W m⁻² more reflective. The
  v3.1 M7 calibration (annual net TOA +2.2 W m⁻² against CERES +1.0) was
  fitted without volcanic sulfur.
- **MAM4 check.** A two-day `echam-jam` run from 14 April 2010 reproduces
  CESM's build exactly:
  - On a day with no eruption, the emitted SO₂ rises by 24.59 Tg/yr, the
    97.5 % of 25.22 Tg/yr that CESM's SO₂ file carries.
  - On 14 April it rises by 26.41 Tg/yr, adding Eyjafjallajökull's 0.005 Tg
    emitted over that day.
  - The primary SO₄ rises by 1.13 Tg/yr: the other 2.5 %, as MAM4's sulfate.

## Limitations

- Degassing is constant in time; individual volcanoes' variability is not
  represented.
- The CMIP7 list runs from 1 January 1750 to 20 November 2023. A run outside
  it emits degassing only and is warned at start; scenario eruptions come from
  the ScenarioMIP dataset, which is not mirrored. A replayed
  `volcanic_eruption_year` with no eruption is warned when the forcing loads.
- A replayed leap year's eruptions after February fall a day later in a
  non-leap model year, and its 31 December eruption never fires.
- Neither source carries ash, halogens or water vapour.
