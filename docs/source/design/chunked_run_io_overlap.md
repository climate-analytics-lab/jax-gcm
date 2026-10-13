# Chunked runs: host work overlaps the next chunk

`jcm.runners.run_chunked` integrates a long run in chunks of
`run.chunk_days`. Between two integrations each chunk has host work to do:
convert the chunk's predictions to xarray (the device-to-host copy), feed
the monthly-mean stream, run the health check and the aerosol-budget report,
write the chunk and monthly netCDF files, and stage the monthly stream and
the checkpoint. The loop runs that work on a single I/O thread
(`jcm-chunk-io`) while the main thread integrates the next chunk, so on a GPU
it costs no wall time as long as it is shorter than one chunk's integration.

## Why the work is overlapped

The monitor and release-validation recipe is a year of daily means in 5-day
chunks streamed into monthly files (`run=longrun`; `run.chunk_days=5
run.save_interval=1 run.monthly_means=true run.save_chunks=false`). At T63L47
with JAM aerosol (`+configuration=ma-t63-l47`) on one A100, measured
2026-10-05 on a card with no other running job:

| per 5-day chunk | integration | host work | wall clock | sim-days/hour |
| --- | --- | --- | --- | --- |
| host work in series | 127–128 s | 34–38 s | ~165 s | 109 |
| host work overlapped | 127–129 s | 18–33 s, hidden | ~128 s | 140 |

Run serially, the host work was 22% of the wall clock. Overlapped, the time
between one chunk's end and the next chunk's start is under 0.1 s, the
integration itself is unchanged to within 1%, and the monthly files and the
checkpoint are byte-identical to the serial loop's.

Of the serial host time (profiled with `py-spy`), the monthly accumulator's per-variable
xarray bookkeeping was ~26 s, writing the pending month's float64 sums
(~1.8 GB, `.monthly.new`) ~6 s and the checkpoint (~1.3 GB) ~7 s. Monthly
files and the health check were a few seconds; the device-to-host copy
itself about one.

Two changes follow. The accumulator does its arithmetic on the NumPy
buffers in place (the alignment checks it relied on xarray for are done once
per chunk, before any statistic changes), which cuts its share from ~15 s to
~2 s per chunk at this size with bit-identical sums. And the remaining work
runs beside the next integration rather than between integrations, which
removes all of it from the wall clock. Writing less was not the lever: the
checkpoint and the staged month are what make a 5-day-chunk run resumable
bit for bit, and they are kept every chunk.

## What the overlap preserves

* **Order.** One worker, so chunks' host work runs one at a time in chunk
  order; the monthly stream, the per-chunk reports and the checkpoint
  sequence advance exactly as in a serial loop. The main thread waits for
  chunk *n*'s job before queueing chunk *n + 1*'s.
* **What is written.** Each job reads only what was captured at its chunk
  boundary: the predictions and the model's `RunState`. The checkpoint is
  written from that state (`save_checkpoint(..., run_state=...)`) and the
  staged month is stamped with its clock, never from the model's live state,
  which by then belongs to the next chunk. Output files are identical to a
  serial loop's: the arithmetic is unchanged and runs in the same order.
* **The health gate.** A chunk is still checkpointed only after it passes
  its own health check. What changes is when a failure is acted on: the
  verdict on chunk *n* arrives while chunk *n + 1* integrates, so with
  `run.bail_on_unhealthy` the run stops after integrating one more chunk,
  which is discarded unwritten (no chunk file, no monthly file, no
  checkpoint). The cost is one chunk of compute on a run that is failing
  anyway.
* **Failures.** An exception in a job (a full disk, a refused write)
  surfaces in the main thread at the next chunk boundary. An exception in
  the integration still lets the queued job — a passing chunk's outputs and
  checkpoint — finish before it propagates.

The work needs its own thread rather than JAX's asynchronous dispatch: the
compiled chunk call does not return until the integration has finished
(measured: the call itself takes the whole ~128 s), so nothing queued after
it on the same thread could overlap. The call holds no GIL while it waits,
which is what lets the I/O thread's NumPy, xarray and netCDF work proceed
beside it (measured: the integration time is unchanged while that work
runs). No model code uses host callbacks that would need the GIL during the
integration.

## Reading the log

Each chunk's report ends with

```text
  Wall: 127.7s this chunk, 1222s total (118 sim days/hr)
  Host: 18.0s output and checkpoint work for this chunk
```

`Wall` is the integration alone, as before (the running sim-days/hour
includes the compile in chunk 0); `Host` is that chunk's host work, which
overlapped the following chunk's integration. A `Host` time
approaching `Wall` means the I/O thread is about to become the critical path
(for example on a host whose CPUs are oversubscribed). Because a chunk's
report is printed by the I/O thread, it appears after the next chunk's
`Model starting` line.
