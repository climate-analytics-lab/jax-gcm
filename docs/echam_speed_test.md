# ECHAM speed test

For L4/RTX reproduction commands, pinned dependencies, results and cleanup,
see [the benchmark handoff](echam_speed_test_handoff.md).

From a jax-gcm checkout with its Python dependencies installed:

```bash
python echam_speed_test.py --total_time 5 --save_interval 1 --n_repeats 3
```

Same warmup, timing loop and JSON output as `speed_test.py`, using T31/L47
ECHAM with a 15-minute timestep, stock Sundqvist cloud fraction and RRTMGP.
Inputs are built-in dry JW initialization, flat aquaplanet and constant forcing.
Cloud fraction runs every timestep; radiation uses its native cadence.
`save_interval` controls output only. Durations are in days. No local data paths.
Set `JAX_PLATFORMS=cpu` for CPU, or `CUDA_VISIBLE_DEVICES` to select a GPU.
