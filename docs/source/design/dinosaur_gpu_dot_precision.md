# dinosaur's GPU matmul precision

jcm runs every float32 matmul in the dinosaur dycore at `Precision.HIGHEST` on
GPU, overriding the bfloat16-emulation defaults that dinosaur 1.5.0
introduced. The override lives in `jcm/dycore/dinosaur/dot_precision.py`, is
applied on `import jcm`, and leaves alone any algorithm a caller set
beforehand.

## Why: two defaults, two failures

dinosaur resolves float32 dots through two module-level defaults. Both
produce wrong answers on GPU in jcm's configurations, for different reasons.

| dinosaur default | used by | problem on GPU | jcm override |
|---|---|---|---|
| `jax_numpy_utils.FLOAT32_DOT_ALGORITHM` = `BF16_BF16_F32_X6` | vertical operators; `RealSphericalHarmonics` transforms (single-device runs) | XLA miscompile in jaxlib < 0.11.2 | `HIGHEST` while jaxlib < 0.11.2 |
| `spherical_harmonic.FAST_TRANSFORM_DOT_ALGORITHM` = `BF16_BF16_F32_X3` | `FastSphericalHarmonics` transforms (`spmd_mesh` runs) | 3-pass transforms visibly alter the solution | `HIGHEST`, always |

### The X6 miscompile (single-device)

The XLA bundled with jaxlib 0.10.2 to 0.11.1 miscompiles a bf16-emulated
contraction under a specific combination: one operand is a compile-time
constant (dinosaur's numpy Legendre basis), and the other has a size-1 free
dimension. The dot is strength-reduced to multiply-and-reduce, and constant
folding then evaluates a layout-changing `bitcast-convert` of the constant as a
raw buffer reinterpretation. That permutes the constant's bf16 hi/mid/lo
split, so the result carries ~2e-5 relative error instead of ~1e-7.

In the dycore this is exactly the inverse transform of a single-level field,
`log_surface_pressure`, inside the semi-Lagrangian transport. The same error
pattern is added every step, and in the hybrid-coordinate core it integrates
into a hemispheric mass drift. A T63L47 Held-Suarez aquaplanet, which is
symmetric about the equator, develops a −116 hPa NH−SH surface-pressure
asymmetry in 30 days under the default, against +0.03 hPa at `HIGHEST`. With
ECHAM physics and real orography the same defect shows as a 40–60°N surface
pressure of ~938 hPa within a month. Sigma-coordinate runs were not affected in
testing, and CPU never uses this default.

jaxlib 0.11.2 compiles the contraction correctly, so this override is gated on
the running jaxlib version and lifts on its own when jcm's jax pin moves past
it. Upstream report: neuralgcm/dinosaur#147.

### The X3 transforms (SPMD)

`FastSphericalHarmonics` has its own 3-pass default. This is not the
miscompile, because disabling XLA constant folding leaves the result
unchanged. On GPU it measurably changes the model: a T63L47 Held-Suarez
aquaplanet loses 0.1 hPa of global-mean surface pressure on the first day and
spins up about 4 hPa too weakly at 40–60°N over 10 days, and `HIGHEST`
restores the single-device reference. On CPU, X3 gives results identical to
`HIGHEST`, so the override costs nothing there and is not version-gated.

## Cost

Steady-state time per simulated day, Held-Suarez T63L47 on one A100. This is
the dycore-only worst case; physics dilutes it in full runs.

| path | dinosaur default | `HIGHEST` |
|---|---|---|
| single-device | 0.61 s | 0.62 s (+2 %) |
| SPMD mesh (1,1,1) | 0.59 s | 0.64 s (+8 %) |

## Checking for regressions

`jcm/dycore/dinosaur/dot_precision_test.py` includes a GPU-only test of the
triggering shape: an inverse transform of a `(1, M, L)` field checked against
a float64 evaluation of the same basis. It fails if the override is removed on
an affected jaxlib. CI runs on CPU, where the test is skipped, so run it on a
GPU host when changing the jax pin:

```bash
python -m pytest jcm/dycore/dinosaur/dot_precision_test.py
```
