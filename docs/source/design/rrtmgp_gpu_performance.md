# RRTMGP radiation on a GPU: structure, cost and what XLA does with it

The ECHAM+RRTMGP presets (`t63-echam-rrtmgp`) spend most of a model day in the
radiation call. This page records why that call costs what it does on an
accelerator, which structural choices in jcm and `jax-rrtmgp` (>= 0.7.0) follow
from that, and the measurements behind them. The longwave solver option it
discusses is described, with its reference provenance, in
{doc}`../science/radiation`.

All timings are A100-80GB PCIe, `t63-echam-rrtmgp`, float32 radiation,
`compute_cre=True`, one radiation call = the per-column-vmapped solve of the
whole T63 grid (18,432 columns, 47 levels). "Launches" is the static count of
kernel launches per call read from the compiled HLO.

## Call structure

`RRTMGPRadiation` vmaps `radiation_scheme_rrtmgp` over columns. Inside the
library, a solve loops over the 128 longwave (112 shortwave) g-points, and each
g-point does a gas-optics gather, the cloud/aerosol combination, per-cell
two-stream (or no-scatter) coefficients, and the vertical recurrences, which are
unrolled over levels and written as a chain of dynamic-update-slices. Every one
of those steps is small: a g-point's work on an 18,432-column block is a few
elementwise kernels, a table gather, and ~47-deep sequential recurrences. The
arithmetic is cheap and the call is **launch- and gather-bound**, not
flop-bound: a change that adds launches costs time even if it removes flops, and
the launch count, not the flop count, tracks wall time (`jax-rrtmgp`'s
`performance_test.py` pins both).

## What the library does about it

Four changes, with three levels of exactness. The recurrence fences are
bit-identical on CPU. The shared clear-sky solve moves the shortwave clear-sky
fluxes by at most two float32 ulp (2.4e-4 W/m2) and nothing else (the longwave
is bit-identical). Evaluating the gas optics of all g-points at once is
op-by-op identical to the per-g-point functions, but the compiled programs
differ by up to 5 ulp on the fluxes (35 ulp on the longwave net flux, a
cancellation of two large terms) from XLA fusion and FMA contraction. Scanning
the vertical recurrences at a large chunk differs from the unrolled program by
at most 12 float32 ulp on every flux (shortwave <= 4, longwave down <= 7, longwave
up <= 11-12; 99.9th percentile <= 5; largest absolute difference 4.9e-4 W/m2 on
fluxes up to ~1400 W/m2, largest relative 9.8e-7; one cloudy full-model call
after a 10-day spin-up), and the same program run twice is bit-identical. The
heating-rate differences are what that flux rounding gives through the thin top
layers: up to 0.2 K/day at the 1 Pa level, at most 1e-3 K/day below 1000 Pa.

Measured one after another on the same card class:

| configuration | s / sim-day | ms / radiation call | launches |
|---|---|---|---|
| separate all-sky and clear-sky solves | 16.05 | 739 | ~52k |
| + clear-sky solved inside the all-sky g-point loop (`with_clear_sky`) | 12.88 | 478 | |
| + fence at the end of every recurrence (`recurrent_op_with_halos`) | 11.22 | 350 | 14.9k |
| + gas optics of all g-points ahead of the loop (`all_gpts`, `gpt_chunk=4`) | ~9.7 | ~202 | |
| + level recurrences as `lax.scan` inside 64-g-point chunks (`use_scan`, `gpt_chunk=64`) | ~9.1 | 139 | 3.5k kernels |

About 43% off the baseline in s/sim-day, and 739 -> 139 ms per call. The last
two rows were each their own A/B on the same card class with the previous row as
the control: all-g-point gas optics 337 -> 202 ms per call and 11.71 -> 9.71
s/sim-day against `per_gpt` (its best chunk, 8), with the fence in both arms;
the scan 202 -> 139 ms per call, end to end 9.06-9.15 against 9.61-10.01
s/sim-day (noisy host; the per-call numbers are the ones to trust). They are
chained onto the rows above, not re-measured as five steps of one sequence.
These are the jcm defaults; the library's own defaults are unchanged.

**Shared clear-sky solve.** The clear-sky fluxes needed for the cloud radiative
effect differ from the all-sky ones only in the cloud optical properties. The
gas optics (the gather from the k-tables, the Planck sources) is the bulk of a
g-point's cost and is identical, so the library computes it once and runs the
second transport on it (`clear_sky_fluxes=True` on the all-sky call). jcm asks
for it exactly when `compute_cre` is set; aerosols stay in the clear-sky branch
(the CMIP convention, "clear" meaning cloud-free).

**Shared aerosol-free solve.** The AeroCom `*noa` fluxes (the all-sky and
clear-sky problems with no aerosol optics mixed in; `aerosol_free_interval`,
`aerocom_erfari_sampling.md`) are the same pattern one step further:
`aerosol_free_fluxes=True` (jax-rrtmgp 0.7.0) solves the aerosol-free twin
of every problem in the same g-point loop, sharing the gas optics, Planck
sources and cloud mix-in and repeating only the cell properties and the
transport, where the earlier jcm implementation ran the whole call a second
time with the aerosol optics zeroed. At T63L47 (`t63-echam-jam-aerocom`,
JAM aerosol in both bands, clear sky on) the radiation call is 142.7 ms
without the diagnostic, 211.7 ms with it (+48 %) and was
268.3 ms as a second solve (+88 %); with radiation every tenth
step the model step goes from 327.5 to 335.2 ms (+2.4 %). The twins equal a separate
call without the aerosol arguments to float rounding (the earlier zeroed
mix-in differed from "no aerosol" by its `max(tau, eps)` guards); without
aerosol arguments they are the originals at no cost, so a MACv2-SP run
(shortwave aerosol only) pays for the shortwave twins alone. With a companion
interval N > 1 jcm runs the solve with the twins on every Nth radiation call
and the plain solve on the others under a `lax.cond`, so both programs are
compiled.

**All-g-point gas optics.** In the spectral loop the gas optics is a scalar
gather per (cell, interpolation corner, g-point): thousands of small gathers, the
dominant cost of a g-point. `gas_optics_mode="all_gpts"` computes the
g-independent interpolation (temperature, pressure and mixing-fraction indices
and weights, the troposphere mask, the minor absorbers' scalings) once per cell,
then reads every table as a row gather over the g-point axis, laid out as the
RRTMGP GPU kernels do it, for all g-points in one pass ahead of the loop. The
loop takes its g-points' slices from the result. The corner order and the
left-to-right sum of the per-g-point interpolation are kept. The gas optics
phase goes from 196 to 27 ms per call; transport is now 127 of the 202 ms and is
the next bottleneck. The price is the memory of `n_gpt` precomputed fields, flat
in practice (below). The library default stays `per_gpt` (additive API), so a caller that does not
opt in is unchanged. The precomputed stage always reads the tables by direct
indexing (a gather), whatever `RRTMGP_LOOKUP_IMPL` selects, and jcm's default is
`all_gpts`: a TPU user of the matmul lookup must set `gas_optics_mode=per_gpt`.

**The fusion rule.** XLA must never be allowed to fuse a consumer into an
unrolled dynamic-update-slice (DUS) chain. A chain runs in place only while it
is the root of its fusion; fused into a consumer (the spectral-sum `add`, the
zero-pad of the halo, the next recurrence reading `output[:, :, 1]`) it is
re-evaluated per output element. The signature in a profile is one multi-ms
kernel per g-point containing ~90 DUS ops. It appeared three ways: with a
`jnp.stack` variant of the recurrence, with the halo pad, and in the no-scatter
solver (below). The cure is `jax.lax.optimization_barrier` on the chain's result,
which changes no value and makes the result opaque to fusion.

**The best placement of the barrier depends on the solver and the mode.** Two
placements were measured: one at the end of `recurrent_op_with_halos` (after the
pad), and one inside `recurrent_op` straight after the level loop
(`fence_chain=True`).

| solver, gas optics | end barrier only | + inner barrier |
|---|---|---|
| two-stream | 350 ms | 561 ms |
| no-scatter, `per_gpt` | 3138 ms | 502 ms |
| no-scatter, `all_gpts` (G = 4, unrolled) | 163.5 ms | 210 ms |

So the end barrier is unconditional, and the inner one is opt-in: the
no-scatter solver sets it only while the gas optics is inside the g-point loop
(`solve_lw` passes `fence_chain=(gas_optics_mode == 'per_gpt')`). With
`use_scan=True` there is no update chain to fence. The reason for the split was
not established beyond the measurement; it should be re-measured, not assumed,
when a solver or the XLA version changes.

**The g-point loop: unrolled or scanned.** `use_scan` replaces the unrolled
level recurrences by a `lax.scan` over levels, inside each chunk of `gpt_chunk`
g-points (the loop over chunks stays a `fori_loop`). A scan step is one kernel
over `gpt_chunk` x columns elements. At small chunks every step is a launch and
the unrolled loop wins; with the gas optics hoisted and a large chunk the step
amortises its launch and the transport of a call drops from 126 to 37 ms.
Sweep, `all_gpts`, `use_scan=True`:

| `gpt_chunk` | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| ms / radiation call | 156.6 | 163.8 | 139.0 | 157.6 |
| peak memory (GiB) | 8.5 | 8.5 | 12.5 | 20.5 |

The sweep is non-monotonic but deterministic (kernel counts are identical from
run to run), so the G = 64 minimum is a property of the compiled program and
should be re-measured when XLA or the library changes. The unrolled loop does
not profit from a large chunk (its fused level chains are the cost): with the
gas optics inside the loop and the end barrier in, G = 1, 4, 8, 16 give 11.22,
11.46, 11.44, 11.40 s/day, and with `all_gpts` unrolled G = 4-8 is the optimum.

**Chunking and memory.** Peak memory scales with columns x levels x G. At
T63L47 the defaults peak at 12.5 GiB (+4.0 over the unrolled G = 4 form) and at
T63L95 at 24.6 GiB (+8.0; 570.5 -> 291.8 ms per call there), both completing
under a ~39 GiB allocator ceiling (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.49` on an
80 GB card), so a 40 GB A100 holds them. G = 128 costs +12 GiB at T63L47 and is
not an option at T63L95. `gpt_chunk=16` is the setting that adds no memory
(156.6 ms) for larger grids. T106-class grids were not measured, and a linear
extrapolation puts the model itself beyond 40 GB there regardless of this
option. jcm sets `gas_optics_mode="all_gpts"`, `gpt_chunk=64` and
`use_scan=True` (`RRTMGPRadiation`, `physics.terms.rrtmgp_radiation.*` on term-list presets, or
`physics.radiation_{gas_optics_mode,gpt_chunk,use_scan}` on the factory-built
JAM presets); the
library's own defaults stay `per_gpt`, 1 and `use_scan=False`.

**Reverse mode.** The backward pass of the radiation term holds per-g-point
profiles of every column live, so its scratch grows linearly with columns and
no placement fits the full T63L47 grid on one 80 GB card without column
chunking (`JCM_RRTMGP_COL_CHUNKS`, `rrtmgp.py::_maybe_chunked_vmap`). The
placement decides how much, and on the GPU the ordering is the reverse of the
forward-only intuition that hoisting the gas optics must cost memory. XLA's
static scratch for the term's backward at T63L47 (18432 columns, f32,
`jax.grad` of the term alone, measured per column and checked linear in the
column count):

| placement | backward scratch, full grid | per column | with `JCM_RRTMGP_COL_CHUNKS=8`: peak / s per call |
|---|---|---|---|
| `per_gpt`, G = 1, unrolled | 334 GiB | 18.6 MiB | 25.7 GiB / 7.5 s |
| `all_gpts`, G = 4, unrolled | 183 GiB | 10.2 MiB | 11.6 GiB / 2.4 s |
| `all_gpts`, G = 64, scanned (default) | 271 GiB | 15.1 MiB | 15.1 GiB / 1.2 s |

On 2304 columns (an eighth of the grid, the largest fraction that fits
unchunked for the default) the default's backward costs 33x its forward's peak
memory and 5.4x its time (34.1 GiB, 173 ms against 1.03 GiB, 32 ms); `per_gpt`
costs 84x and 17x (41.8 GiB, 1336 ms against 0.50 GiB, 78 ms). The default is
therefore also the cheapest placement to differentiate on the GPU, and the
unrolled G = 4 form is the one to pick when backward memory is the binding
constraint. All gradients were finite. On the CPU the ordering inverts, for a
reason that has nothing to do with the arrays: XLA's CPU backend fuses the
roughly one thousand weighted table reads of the hoisted gas optics into a few
enormous backward kernels and the LLVM codegen of those takes tens of GiB
independent of the grid size, which is why the slow-suite gradient harnesses
pin the per-g-point placement (`test_suite_memory.md`). A library variant that
batches the corner and minor-interval reads into one gather per group removes
that CPU cost (47.6 to 11.2 GiB on the one-column harness, values op-by-op
identical) but costs the GPU forward 11% at the default and saves nothing in
GPU backward scratch, so it is not adopted (table below).

The forward output of the term on the GPU is bit-identical within a process
but takes one of two distinct values across processes for the same inputs and
library (maximum difference 1.8e-6 K/s, about 1e-3 of the field's maximum, at
the top levels, 69% of elements bit-equal between the two classes). The choice
is made at compile time and is the same with and without the batched-gather
variant, so it is a property of XLA's GPU compilation of this program (its
autotuner is the usual source), not of the placement options; it was not
chased further and the exactness statements above are for the CPU.

## The no-scatter longwave solver

`lw_solver="noscat"` (`RRTMGPRadiation(lw_solver=...)`) selects RTE's
absorption-only longwave transport, which is what ECHAM6's `mo_lrtm_solver.f90`
does (no longwave scattering by cloud or aerosol). It is selectable and **not
the default**: adopting it shifts the tuned energy balance.

Both layers of each cell share one Planck source per level, the geometric mean
of the two adjacent layers' spectral mappings that the two-stream solver also
uses, as current RTE does (RRTMG, RTE <= 1.6 and ECHAM6's `rtrnmc` give each
layer its own face values instead; the two conventions differ by at most 0.04
W/m2 on the RFMIP clear-sky profiles), so the two solvers differ only in the
transport. Against upstream's RFMIP clear-sky reference (computed with this
solver, g256) the top-of-atmosphere upward flux is within 0.05 W/m2 on every
site and the largest deviation anywhere is 1.7 W/m2 (the surface downward flux
of the moistest column); `longwave_noscat_test.py` pins both.

Cost, with the inner fence above and the gas optics inside the loop: 502 ms per
radiation call and 12.62 s/sim-day at `gpt_chunk=1`.

Climate effect, a 30-day January A/B against the two-stream default at T63L47,
global means:

| quantity | change |
|---|---|
| OLR | +1.7 W/m2 |
| longwave cloud radiative effect | -1.55 W/m2 |
| TOA net | -2.3 W/m2 |
| clear-sky OLR | +0.16 W/m2 |
| cloud cover, LWP/IWP, precipitation | within the noise of a perturbed twin |

The clear-sky OLR hardly moves, so the shift is the treatment of cloudy
columns (two-stream scatters longwave off cloud, no-scatter does not), which is
the physics, not a bug. Because the model is tuned with the two-stream solver,
a user who wants ECHAM's own treatment should expect to retune the TOA balance
with it.

## Directions measured and rejected

Each row is the number that decided it, so the experiment is not repeated.

| direction | measured | decision |
|---|---|---|
| `jnp.stack` of the level outputs instead of the `.at[].set` chain | 2-3x slower (the DUS-fusion signature above) | kept `.at[].set` plus the fence |
| `optimization_barrier` inside `recurrent_op` for the two-stream solver | 350 -> 561 ms per call; launches 14.9k -> 90.8k | opt-in (`fence_chain`), used by the no-scatter solver only, and only with the gas optics in the loop |
| associative parallel-prefix scan for the affine / albedo recurrences | 183-298 ms per call against 139-202 for the sequential form, dominated by ~3000 strided slice and concatenate kernels | sequential recurrences kept |
| unrolled loop on z-leading copies of the arrays | 269-336 ms per call, and a `ptxas` crash at G = 64 | not adopted |
| bfloat16 / float16 anywhere in the solve | errors above 1 W/m2 and 0.1 K/day; the recurrences break first, below tau ~ 1e-3 | float32 throughout (see below) |
| `lax.scan` recurrences at `gpt_chunk=128` | 157.6 ms per call and +12 GiB (G = 64: 139.0 ms) | not adopted; G = 64 |
| batched corner / minor-interval table gathers in the hoisted gas optics (one gather per interpolation group instead of one per corner) | CPU backward 47.6 -> 11.2 GiB and compile 82 -> 52 s, but GPU forward 139 -> 154 ms end to end (+11%) and GPU backward scratch unchanged (271 -> 275 GiB full grid) | not adopted; the CPU-only cost is handled by the test harnesses' placement pin |
| no-scatter longwave as the default | speed parity only, and a climate change (OLR +1.7 W/m2, longwave CRE -1.55 W/m2, about -9%) | selectable option; the default is a science decision |

## Reduced precision is not an option

A CPU study of the solve in bfloat16 and float16 against a float64 reference
(100 RFMIP columns, 60 layers, with synthetic clouds; production float32 is the
yardstick, with a worst broadband flux error of 0.004 W/m2) found no viable
arithmetic stage at either precision. Casting the two-stream cell kernels, the
vertical recurrences, the g-point accumulation, or the gas-optics arithmetic
gives worst-case errors of order 1-40 W/m2 and 0.4-200 K/day, and plain f16 overflows
or flushes to zero (the k-tables sit at 1e-41 to 1e-14, below f16's smallest
subnormal). The recurrences break first: they consume the transmittance `T` and
reflectance `R` themselves, and `T = 1 - 1e-3` does not survive 8 (bf16) or 11
(f16) significant bits even when the closed forms for `1 - R - T` do.

Only table **storage** survives: gas and Rayleigh tables stored in bf16 stay
within 0.04 W/m2 and f16 with a per-slice power-of-two prescale is
indistinguishable from float32. That saves table bytes, not bandwidth in a
launch- and gather-bound call, so it is not pursued. The Planck tables, the
shortwave cloud single-scattering-albedo tables (which need a `1 - ssa`
reformulation) and `vmr_ref` must not be stored in plain half precision.

## Provenance

Measurements: November 2026, on the shared development workstation's
A100-80GB PCIe, with `tools/benchmark.py` following the `jcm-benchmark` method
(timings per radiation call and per simulated day) and `tools/profile_terms.py`
(unchunked per-phase profiling). The reduced-precision study was a CPU
comparison of each variant against the same solve in float64. Library behaviour is pinned
by `jax-rrtmgp`'s `rrtmgp/rte/performance_test.py` (launch and flop budgets),
`rte_utils_test.py` (the barrier adds one op and no value change) and
`longwave_noscat_test.py` (the no-scatter solver against upstream RTE's RFMIP clear-sky reference and the two-stream solver).
