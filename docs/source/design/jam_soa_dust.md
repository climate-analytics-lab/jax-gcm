# SOA production and dust removal

JAM's MAM4 core already partitions SOAG reversibly into aerosol SOA. A
zero SOAG source therefore gives zero SOA regardless of optical parameters.
Default JAM loads the published CAM6 single-bin source from the same HF
emissions bundle as its existing bulk sources. This is the original
CAM6 formulation described in [Jo et al. (2023), section 2.2](https://gmd.copernicus.org/articles/16/3893/2023/).
It uses prescribed VOC-derived SOAG, reversible gas–particle exchange,
and aerosol wet/dry removal. It does not implement the distinct CAM6.3
oxidation-delay tracer, gas deposition or aerosol photolysis described in
that paper's section 2.3, or a volatility basis set.
SOAG exchange is enabled for Aitken and accumulation particles, with
transient primary-carbon coating subsequently transferred by ageing.
Coarse-mode SOA uptake is disabled through a per-call core parameter:
the MAM4-MOM box-model default includes that reservoir, but CAM6 does not.

The default JAM configuration runs the fixed-substep condensation backend
(`mam4_jax`), the one reverse-mode differentiation goes through; CAM's
Fortran ASTEM semi-implicit backend is selectable
(`physics.jam_microphysics=mam4_jax_astem`). ASTEM's corrector conserves gas
plus aerosol during both condensation and evaporation. A warm, aerosol-rich
cell exposed a 1.27% organic excess in the fixed-substep backend: its frozen
equilibrium flux can exhaust a modal reservoir, after which clipping the
aerosol breaks the budget (#1064). ASTEM avoids that defect, but its adaptive
loop (a `lax.while_loop` with data-dependent bounds) cannot be differentiated
in reverse mode, so it is not the default of a model whose contract is
`jax.grad`; the matched runs below were made with it.

## Emissions and reproduction

The official NCAR anthropogenic, biogenic and biomass-burning SOAG surface
inventories are averaged over the bundle's bulk source period: 2005–2014
for present-day and 1850–1859 for preindustrial. The sensitivity experiments
below used a separately prepared 2014 inventory or the 1995–2005 climatology,
as labelled. Their VOC yields and CAM's
1.5 source multiplier are already applied. Fluxes are carbon-equivalent
molecules per square centimetre per second: CAM's surface-emission routine
converts them using the destination tracer molecular weight, 12.011 g/mol.
The 250 g/mol defining the equilibrium saturation concentration must not be used
to convert these emissions. That substitution would inflate the source by
about 20.8 times.

The captured MAM4 exchange uses slightly different conversion factors for
SOAG (12.011/150) and aerosol SOA (12/150). A per-call equilibrium
molecular weight of 250 g/mol reproduces CAM6's 1.02 µg/m³ saturation
concentration at 298 K (reference pressure 10⁻¹⁰ atm, enthalpy 156 kJ/mol).
It scales equilibrium gas by 250/150 on the unchanged local molecular basis,
equivalent to converting gas, SOA and absorbing POA to 250 before exchange
and converting back afterwards. The 10% absorbing POA fraction is retained,
except in primary carbon where coating is transient. Other microphysical
volume conversions and CAM's 0.81 uptake ratio relative to sulphuric acid
remain unchanged. Gas-plus-aerosol budgets must
convert both to the common molecular basis: on the aerosol mass basis the
gas receives a factor 12.011/12. The resulting 0.09% convention difference
is reversible and must not be diagnosed as an additional organic source.
`emi_soag` reports the gas source separately from particulate `emi_soa`;
combine both phases, their wet/dry losses and the corrected gas basis when
closing the organic budget.

`jcm.data.emissions.cam6_soa` pins the three upstream SHA256 hashes, validates
nonnegative finite fluxes and matching calendars, sums their converted
sources, and conservatively remaps them through the shared emissions pipeline.
The mirror builder adds `aero_emis_g_soag` to `emissions_pd.nc` and
`emissions_pi.nc` on T63, T106, T127 and T255. It aligns rates by calendar
month, keeps every bulk field unchanged, and records SOAG source hashes,
period and units in bundle metadata. JAM's existing automatic emission
resolver reads both bulk and pre-speciated channels from this single file.
The runtime snapshot is [the SOAG bundle commit](https://huggingface.co/datasets/climate-analytics-lab/jax-gcm-data/commit/a6a3d075cd8f32c9ade195dccf8630f0d9e5fb6c).
MAM4-JAX is installed from PyPI as `mam4-jax==0.5.1`; no dependency SHA
or packaged SOAG inventory is required.

```bash
python -m jcm.main +configuration=t63-echam-jam
```

Explicit transient inventories still need matching SOAG; the existing
automatic AMIP/ERA5 ancillary choice remains the present-day climatology.
To reproduce the earlier single-year sensitivity inventory:

```bash
python -m jcm.data.emissions.cam6_soa --truncation 63 --year 2014 --output soag.nc
```

Five days from a zero-SOA donor measures the initial
response, not the equilibrium SOA burden or annual AOD. Jo et al. document
upper-tropospheric and high-latitude SOA biases in the original CAM6 scheme;
matching its formulation does not establish a validated JCM climatology.
An annual assessment must include the vertical distribution and removal,
in addition to total AOD.

## The aerosol working population

The physics host supplies step-start tracers and a running tendency ledger.
Each sequential aerosol process must read the population left by its
predecessors and return only its own change. Previously removal consumed
current-step emissions while the MAM4 core diagnosed sizes from step-start
mass and number. Optics also combined step-start species mass with core
geometry. The shared `split_view` reconstructs the working population.

| Process | Input population |
| --- | --- |
| Turbulent tracer mixing | Earlier surface emissions included in the implicit solve |
| Convective tracer transport and scavenging | Earlier emissions and turbulent mixing included |
| Sulfur oxidation | Earlier emissions and transport included |
| MAM4 and placeholder microphysics | Earlier emissions, transport and chemistry included |
| Cloud-borne exchange and aqueous chemistry | Existing working view and sequential cloud-borne store |
| Sedimentation, surface deposition and wet removal | Existing working view; each sees previous removal |
| Optics and ice nucleation | Working view matching their position in the chain |

After condensation, ageing and coagulation, MAM4's returned mass/number
can differ from the population used for its initial size diagnosis.
The adapter diagnoses the updated lognormal size relation and recomputes
equilibrium water before removal and optics. It does not repeat `calcsize`,
which would advance number adjustment and mode transfer twice.

## Dust lifetime tests

Coarse-mode drag includes CAM's 0.8 asphericity correction, motivated by
[Huang et al. (2020)](https://doi.org/10.1029/2019GL086592) and present in
`modal_aero_depvel_part` in CAM's October 2025 implementation. Both number
and mass velocities retain their lognormal moment weighting. The correction
also reaches the Stokes number used for turbulent impaction. CAM applies it
to the internally mixed coarse mode, so it affects sea salt as well as dust;
their burdens must be assessed together. It changes drag, not optical shape.
Its magnitude is insufficient by itself to explain a several-fold lifetime
deficit. The coarse-mode width and emission parameters are unchanged.

The `dry_du` diagnostic combines gravitational surface loss and turbulent
surface deposition. `sed_du` and `turb_dry_du` separately report those
losses in kg/m²/s; their sum must equal `dry_du`. Together with `wet_du`,
`emi_du` and the change in both aerosol phases they expose the dust budget.
The CAM distribution factors in Stokes settling are retained: number and
mass represent different moments of a lognormal mode. Reducing that factor
to increase the dust burden would violate the reference formulation.

Matched five-day control and corrected runs use the same spun-up donor,
meteorological forcing, emission calibration and precision. Separate
experiments add coarse drag and land-cover collection to the corrected SOA
and working-population coupling.
Source and sink budgets, dust size and vertical distribution, and climate
health must support any lifetime change before it becomes a release fix.


## Surface collection on land

The previous JAM surface term applied Slinn & Slinn's ocean collection law
on every surface. Dust source regions therefore had no surface-dependent
impaction or rebound. JAM now uses CAM's eleven-class Zhang (2001) collection
law (`aero_model.F90::modal_aero_depvel_part`), with the official
`regrid_vegetation.nc` inventory reduced by CAM's PFT-to-Wesely mapping.
The source URL and SHA256 accompany `jcm/data/bc/cam_landuse.nc`;
`python -m jcm.data.bc.cam_landuse SOURCE.nc OUTPUT.nc` reproduces that asset.

The inventory is conservatively remapped to a spectral model grid and
normalized **after** remapping, as in CAM: some source PFT and urban/lake
fractions overlap. Point-grid hosts use nearest inventory mixtures. An
all-ocean terrain selects only the water class. Surface collector radius,
Brownian exponent, impaction parameter and dry-surface sticking fraction
follow CAM for both number and mass moments. The turbulent resistance retains
the settling cross term, while the separate sedimentation term owns the
gravitational sink. The host's neutral aerodynamic resistance is retained;
this is not a port of CAM's entire surface-layer calculation. Cloud-borne
surface collection still uses the internally mixed aerosol mode rather than
CAM's explicit droplet velocities.

Reference tests compare every class and both moments to velocities from the
unmodified CAM routine compiled with minimal module stubs (CAM commit
`21a782945d122785ccb5d78e27ea59d80fb73396`). This is a structural correction
with fixed literature coefficients, not an emission or lifetime multiplier.
Its global effect must be assessed together with sea salt and SOA removal.

## Matched five-day evidence

The January and July comparisons use separate T63L47 warm donors, each
spun up for 185 days on `49c0724c`. Both arms start from the same donor and
reset its clock to the stated calendar date. Dust emission parameters and the coarse
mode width are fixed. The control is `35dc1997` with diagnostic-only additions
for the separated dust dry sinks. The corrected production code is
`102d936f`, with MAM4 production code `25924f0` (included in the 0.5.1 release source) and the
2014 CAM6 inventory. These historical comparisons predate the default
period-matched bundle update. Runs use native `jcm.main`, float32, and five-day health gates.
The donors contain no SOA.

The table uses global means of native daily averages over **days 3–5**.
Dust burden includes interstitial and cloud-borne mass. The diagnosed
lifetime is **mean burden divided by mean dry-plus-wet loss**, not the mean
of daily burden/loss ratios, an equilibrium residence time, or an annual
estimate. SOA's optical diagnostic is its volume-allocated share of wet
extinction; water has its own share.

| Season / case | Total AOD | Dust lifetime, days | Dust burden, mg/m² | Sea salt, mg/m² | SOA AOD share |
| --- | ---: | ---: | ---: | ---: | ---: |
| January control | 0.07678 | 1.12 | 16.11 | 16.76 | 0.00000 |
| January corrected | 0.08878 | 2.04 | 24.99 | 21.33 | 0.00206 |
| July control | 0.08236 | 0.89 | 40.54 | 12.94 | 0.00000 |
| July corrected | 0.10185 | 1.70 | 73.09 | 17.23 | 0.00277 |

AOD increases by 16% in January and 24% in July. Diagnosed dust lifetime
increases by 83% and 92%, respectively. Sea-salt burden also increases by
27% and 33%, a material response of the internally mixed coarse mode.
Both corrected runs complete their health gates.


The isolated January ablations used the retained 2000 climatology, keeping
that source fixed across corrected cases. Working-population coupling plus
SOA gave AOD 0.08465 and diagnosed dust lifetime 1.24 days; adding coarse
asphericity gave 0.08667 and 1.42 days; adding CAM surface collection gave
0.08873 and 2.05 days. These are sensitivity experiments, distinct from the
final 2014-inventory comparison above. They identify surface collection as
the largest of the tested dust-lifetime corrections.

## Matched twenty-day continuations

Both arms continued from their own day-five checkpoints for another fifteen
days, retaining native five-day health gates. All continuation health gates
passed. January retains daily output; July uses native partial-month means
(`run.monthly_means=true run.save_chunks=false`) covering exactly July 6–20,
with coverage 15/31. The July rows therefore describe the entire continuation,
not its last three days. Each comparison uses the same window in both arms.

| Window / case | Total AOD | Dust lifetime, days | Dust burden, mg/m² | Sea salt, mg/m² | SOA AOD share |
| --- | ---: | ---: | ---: | ---: | ---: |
| January days 18–20 control | 0.09476 | 0.95 | 43.10 | 15.45 | 0.00000 |
| January days 18–20 corrected | 0.10829 | 1.45 | 60.79 | 19.44 | 0.00435 |
| July days 6–20 control | 0.12860 | 0.99 | 124.66 | 12.48 | 0.00000 |
| July days 6–20 corrected | 0.16642 | 1.98 | 251.01 | 17.40 | 0.00497 |

The January late-window gain is 14% in AOD and 54% in diagnosed dust
lifetime; July's continuation-mean gains are 29% and 101%. Sea-salt burden
increases by 26% and 39%, respectively. SOA burden reaches 1.04 mg/m² in
the January late window and averages 1.18 mg/m² over the July continuation.
The control's substantial seasonal and weather evolution demonstrates why
these values cannot be compared directly with the observed annual AOD 0.145.
SOA remains a modest contribution; these tests do not establish a missing
0.02–0.03 of annual fine-mode AOD.

To reproduce a continuation, use each arm's own day-five checkpoint with
`init.file`, set `run.start_time=2000-01-06` (or `2000-07-06`) and
`run.total_time=15`, and retain its original physics and forcing settings.
For January, average native daily outputs for days 18–20. For July, add the
monthly-output overrides above and reduce the partial-month file. Compute
lifetime from window-mean burden and window-mean dry-plus-wet loss.

## Default-bundle validation

A native five-day January run on `218a15d5` uses the new HF snapshot and
a locally built MAM4-JAX wheel declaring 0.5.0. Its numerical source is
identical to the 0.5.1 correction in
[MAM4-JAX #82](https://github.com/reflective-org/MAM4-JAX/pull/82); only the
version metadata differs.
It uses the same January warm donor and `t63-jam-aod-5day`, with no physics,
emissions-file or alignment overrides. Provenance records the pinned SOAG
bundle snapshot, and the native health gate passes.

Means over days 3–5 are total AOD **0.08843**, diagnosed dust lifetime
**2.02 days**, SOA burden **0.5016 mg/m²**, and SOA's allocated AOD share
**0.00206**. This verifies default delivery of the matched-period inventory;
it does not replace the historical paired comparisons above or annual
release calibration.

The changes improve short-run AOD but do not establish the annual AOD,
equilibrium SOA burden, or an acceptable sea-salt climatology. Nitrate remains
absent, and the dust size-to-extinction relationship needs annual evaluation.
The box tests conserve organics on the common molecular basis; native
float32 budget diagnostics do not establish whole-model closure below their
reported precision floor.

To reproduce the historical corrected short-run setup, prepare the 2014
`soag.nc` above and pin the old bulk-only mirror snapshot. Supply the appropriate donor:

```bash
JCM_MIRROR_REVISION=2ec867b36ea7acf0b71180faeec1a9c0aae9a629 \
python -m jcm.main +configuration=t63-jam-aod-5day \
  physics.jam_microphysics=mam4_jax_astem \
  'forcing.emissions_file=[hf://bundles/t63/emissions_pd.nc,/absolute/path/soag.nc]' \
  'forcing.emissions_align=[auto,wrap_year]' \
  init.file=warm_january.msgpack run.start_time=2000-01-01 \
  run.total_time=5 run.output_prefix=corrected_january \
  +run.checkpoint_path=corrected_january.ckpt
```

For July, use the July donor and `run.start_time=2000-07-01`. Run the control
on `35dc1997` with the same donor, date and output settings, omitting the
SOA source and selecting `physics.jam_microphysics=mam4_jax` explicitly
when reproducing its fixed-substep backend from the updated configuration. Diagnostic-only additions are needed to
split its dry sinks; its native `dry_du` already reports their total.
Use `jcm.analysis.global_mean` on `jam_optics.aod_550`,
`aerocom_burden_du`, `aerocom_burden_ss`, `dry_du` and `wet_du` to reproduce
the table from the daily output file.
