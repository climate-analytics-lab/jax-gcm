# SOA production and dust removal

JAM's MAM4 core already partitions SOAG reversibly into aerosol SOA. A
zero SOAG source therefore gives zero SOA regardless of optical parameters.
The present-day `t63-echam-jam-soa` configuration supplies the published CAM6
single-bin source alongside the existing emissions. This is the original
CAM6 formulation described in [Jo et al. (2023), section 2.2](https://gmd.copernicus.org/articles/16/3893/2023/).
It uses prescribed VOC-derived SOAG, reversible gas–particle exchange,
and aerosol wet/dry removal. It does not implement the distinct CAM6.3
oxidation-delay tracer, gas deposition or aerosol photolysis described in
that paper's section 2.3, or a volatility basis set.
SOAG exchange is enabled for Aitken and accumulation particles, with
transient primary-carbon coating subsequently transferred by ageing.
Coarse-mode SOA uptake is disabled through a per-call core parameter:
the MAM4-MOM box-model default includes that reservoir, but CAM6 does not.

The SOA preset selects the Fortran ASTEM semi-implicit condensation backend.
Its corrector conserves gas plus aerosol during both condensation and
evaporation. A warm, aerosol-rich cell exposed a 1.27% organic excess in the
fixed-substep backend: its frozen equilibrium flux can exhaust a modal
reservoir, after which clipping the aerosol breaks the budget. The ASTEM
choice avoids that defect; it has an adaptive loop and is unsuitable for
reverse-mode differentiation through the condensation solve.

## Emissions and reproduction

The official NCAR anthropogenic, biogenic and biomass-burning SOAG surface
inventories are sampled at 2014 to match the existing present-day emissions
bundle. The initial sensitivity experiments used the separately retained
1995–2005 climatology; they are explicitly labelled below. Their VOC yields and CAM's
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
The prepared T63 inventory is packaged in `data/bc/t63/soag_cam6_2014.nc`;
its metadata records source files and hashes. The `pkg://data/...` path
resolves inside the installed package independently of the working directory.

Reproduce a native-grid inventory with:

```bash
python -m jcm.data.emissions.cam6_soa --truncation 63 --year 2014 --output soag.nc
```

Use the prepared file alongside the existing mass and number sources:

```bash
python -m jcm.main +configuration=t63-echam-jam-soa
```

This is an explicit present-day inventory choice, not a default substitution
for preindustrial or transient forcing. Those simulations need matching
SOAG inventories. Five days from a zero-SOA donor measures the initial
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
reset its clock to the stated calendar date. Dust emissions and the coarse
mode width are fixed. The control is `35dc1997` with diagnostic-only additions
for the separated dust dry sinks. The corrected production code is
`102d936f`, with the immutable MAM4 dependency pin `25924f0` and the 2014
CAM6 inventory. Runs use native `jcm.main`, float32, and five-day health gates.
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

The changes improve short-run AOD but do not establish the annual AOD,
equilibrium SOA burden, or an acceptable sea-salt climatology. Nitrate remains
absent, and the dust size-to-extinction relationship needs annual evaluation.
The box tests conserve organics on the common molecular basis; native
float32 budget diagnostics do not establish whole-model closure below their
reported precision floor.

To reproduce the corrected short-run setup, supply the appropriate donor:

```bash
python -m jcm.main +configuration=t63-jam-aod-5day \
  physics.jam_microphysics=mam4_jax_astem \
  'forcing.emissions_file=[hf://bundles/t63/emissions_pd.nc,pkg://data/bc/t63/soag_cam6_2014.nc]' \
  'forcing.emissions_align=[auto,wrap_year]' \
  init.file=warm_january.msgpack run.start_time=2000-01-01 \
  run.total_time=5 run.output_prefix=corrected_january \
  +run.checkpoint_path=corrected_january.ckpt
```

For July, use the July donor and `run.start_time=2000-07-01`. Run the control
on `35dc1997` with the same donor, date and output settings, omitting the
SOA source and ASTEM overrides. Diagnostic-only additions are needed to
split its dry sinks; its native `dry_du` already reports their total.
Use `jcm.analysis.global_mean` on `jam_optics.aod_550`,
`aerocom_burden_du`, `aerocom_burden_ss`, `dry_du` and `wet_du` to reproduce
the table from the daily output file.
