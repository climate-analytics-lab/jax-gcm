#!/usr/bin/env python
"""Where a JCM timestep's GPU time goes: dynamics, bridge, per physics term.

Usage::

    python tools/profile_terms.py --preset ma-t63-l47 --gpu 3
    python tools/profile_terms.py --preset ma-t63-l47 --gpu 3 --cycles 4
    python tools/profile_terms.py --preset speedy-t31 --gpu 0 --outdir /tmp/p

Writes ``report.md``, ``result.json`` and the raw trace to
``<outdir>/<label>/`` and prints the report.

Unlike ``tools/benchmark.py``, which spawns the model as a subprocess and so
takes a ``--pythonpath``, this runs in-process: to profile a worktree rather
than the installed package, put it on ``PYTHONPATH`` yourself. The report names
the ``jcm`` that was actually imported, so check that line before quoting an
A/B.

Why this exists rather than a stopwatch around each term
--------------------------------------------------------
A JCM step is compiled as one XLA module. Dynamics, the spectral/gridpoint
bridge and all ~60 physics terms are fused into a single stream of kernels
launched asynchronously, so there is no runtime boundary to time: a
``perf_counter`` around a term measures Python tracing, not device work, and
inserting ``block_until_ready`` to force a boundary changes the program being
measured (it defeats the overlap that makes the real step fast).

What does survive compilation is instruction metadata. ``jcm.profiling`` opens
a :func:`jax.named_scope` around the dycore call, each bridge direction and
each physics term, which XLA records as an ``op_name`` prefix on every
instruction traced inside. A profiler trace reports, per GPU kernel, the HLO
instruction that produced it. Joining the two attributes measured device time
back to the component that emitted it, with no instrumentation in the physics
and without perturbing the compiled program.

Two consequences of measuring a *fused* program, both reported rather than
hidden:

- A fusion that merges instructions from two scopes is charged whole to its
  root instruction's scope. The report counts how much device time sits in
  such mixed kernels; that number is the attribution's error bar, and a run
  where it is large should not be quoted per-term.
- Kernels that belong to no scope (the scan plumbing, output assembly, device
  copies) are reported as ``unattributed`` rather than being spread pro-rata
  over the components, which would flatter every one of them.

A third consequence is a hard ceiling on the window. The profiler's event
buffer holds ~1e6 events and a T63L47 JAM step emits ~19,000 kernels, so ~20
steps is all that fits at that resolution; past it the profiler stops recording
and the averages silently come out low. The default window is two radiation
sub-cycles for that reason, and the tool fails rather than reports if the
buffer overflows. Nothing is lost by the short window: kernel shapes are static
and the model takes no data-dependent branches, so a step's cost does not vary
with the state.

Methodology follows the ``jcm-benchmark`` skill: run only on a verified-free
GPU, discard the compile pass, and quote steady state only.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gzip
import json
import os
import pathlib
import re
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from benchmark import PRESETS  # noqa: E402
from gpu_util import describe as describe_gpu  # noqa: E402
from gpu_util import free_indices, gpu_table, is_free  # noqa: E402

DEFAULT_OUTDIR = pathlib.Path("/scr/dwatsonparris/profiles")

# XLA captures the step into a CUDA graph by default, which is faster but
# collapses every kernel's reported provenance to a single ``command_buffer_N``
# instruction -- the attribution join then has nothing to join on. Raising the
# capture threshold above any real module size keeps kernels individually
# attributable. This is the one way the profiled program differs from the
# production one: it pays per-kernel launch overhead that a captured graph
# amortises, so the TOTAL step time here reads slightly high and is not a
# throughput benchmark (use ``tools/benchmark.py`` for that). The per-component
# SPLIT, which is what this tool is for, is unaffected.
_NO_CUDA_GRAPHS = "--xla_gpu_graph_min_graph_size=2147483647"


# --------------------------------------------------------------------------
# HLO parsing
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Instruction:
    """One HLO instruction: its own scope label and the computations it calls."""

    op_name: str
    calls: tuple[str, ...]


def parse_hlo_module(text: str) -> tuple[str, dict[str, Instruction]]:
    """Index one dumped HLO module by instruction name.

    Parameters
    ----------
    text
        Contents of an XLA ``*_after_optimizations.txt`` dump.

    Returns
    -------
    (str, dict)
        The module name (from the ``HloModule`` header) and a map from
        instruction name to :class:`Instruction`. Instruction names are unique
        across a module, so computations are flattened into one namespace; the
        ``calls`` edges are what preserve the nesting that matters here.

    """
    module = ""
    m = re.search(r"^HloModule ([^\s,]+)", text, re.M)
    if m:
        module = m.group(1)

    instructions: dict[str, Instruction] = {}
    for line in text.splitlines():
        # Instruction lines are indented inside a computation body; the
        # computation headers and closing braces sit at column 0.
        if not line.startswith(" "):
            continue
        m = re.match(r"\s+(?:ROOT )?%?([\w.\-]+) = ", line)
        if not m:
            continue
        op_name = ""
        meta = re.search(r'op_name="([^"]*)"', line)
        if meta:
            op_name = meta.group(1)
        # Called computations are always ``%``-prefixed. Anchoring on that is
        # what stops the match running on into the ``metadata=`` field that
        # follows, which has no space before it when the metadata is a bare
        # op_name.
        calls: tuple[str, ...] = ()
        callsm = re.search(r"calls=(%[\w.\-]+(?:,\s*%[\w.\-]+)*)", line)
        if callsm:
            calls = tuple(
                c.strip().lstrip("%") for c in callsm.group(1).split(",")
            )
        instructions[m.group(1)] = Instruction(op_name=op_name, calls=calls)
    return module, instructions


def _dumped_modules(dump_dir: pathlib.Path) -> set[str]:
    """Names of the optimized-HLO dumps present, for recompile detection."""
    return {p.name for p in dump_dir.glob("*after_optimizations.txt")}


def prune_dump(dump_dir: pathlib.Path) -> int:
    """Delete everything in an XLA dump except the optimized HLO text.

    ``--xla_dump_to`` also writes LLVM IR and PTX for every kernel, which for a
    JAM configuration runs to hundreds of megabytes that nothing here reads.
    Returns the number of bytes reclaimed.
    """
    freed = 0
    for path in dump_dir.iterdir():
        if path.is_file() and not path.name.endswith("after_optimizations.txt"):
            freed += path.stat().st_size
            path.unlink()
        elif path.is_dir():
            freed += sum(f.stat().st_size for f in path.rglob("*")
                         if f.is_file())
            shutil.rmtree(path, ignore_errors=True)
    return freed


def index_hlo_dump(
    dump_dir: pathlib.Path,
) -> tuple[
    dict[str, dict[str, Instruction]],
    dict[str, dict[str, list[str]]],
    list[str],
]:
    """Index every optimized HLO module in an ``--xla_dump_to`` directory.

    Returns the instruction index, the computation-membership index and a list
    of informational notes. Both indices come from a single read of each file:
    a JAM step's dump is ~32 MB and there are two of them, so reading twice to
    build them separately is noticeable.

    A trace identifies a kernel's module by name alone, and the warm-up runs
    legitimately compile the step more than once, so several dumps can share a
    name. Files are therefore read in ``module_NNNN`` order and later ones win,
    which resolves an instruction to the most recent compilation -- the one the
    traced execution used, given ``profile_run`` proves no compile happened
    inside the trace.
    """
    def module_id(path: pathlib.Path) -> int:
        m = re.match(r"module_(\d+)", path.name)
        return int(m.group(1)) if m else -1

    index: dict[str, dict[str, Instruction]] = {}
    members: dict[str, dict[str, list[str]]] = {}
    seen: dict[str, str] = {}
    notes: list[str] = []
    for path in sorted(dump_dir.glob("*after_optimizations.txt"), key=module_id):
        text = path.read_text()
        module, instructions = parse_hlo_module(text)
        if not module:
            continue
        if module in seen:
            notes.append(
                f"module {module!r} was compiled more than once "
                f"({seen[module]} then {path.name}); using the later"
            )
        seen[module] = path.name
        index.setdefault(module, {}).update(instructions)
        members.setdefault(module, {}).update(
            parse_hlo_module_computations(text)
        )
    return index, members, notes


def parse_hlo_module_computations(text: str) -> dict[str, list[str]]:
    """Map each computation in a dumped module to its instruction names.

    Needed because a GPU kernel is charged to the fusion instruction that
    launched it, while the scopes that explain the work sit on the instructions
    *inside* that fusion's computation.
    """
    members: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if not line.startswith(" ") and line.rstrip().endswith("{"):
            m = re.match(r"(?:ENTRY )?%?([\w.\-]+)", line)
            current = m.group(1) if m else None
            if current is not None:
                members.setdefault(current, [])
            continue
        if line.startswith("}"):
            current = None
            continue
        if current is None or not line.startswith(" "):
            continue
        m = re.match(r"\s+(?:ROOT )?%?([\w.\-]+) = ", line)
        if m:
            members[current].append(m.group(1))
    return members


# --------------------------------------------------------------------------
# Trace parsing and attribution
# --------------------------------------------------------------------------


def load_trace_kernels(trace_dir: pathlib.Path) -> list[dict]:
    """Read GPU kernel events from a ``jax.profiler.trace`` output directory.

    Returns one dict per kernel execution with ``ts``/``dur`` (microseconds),
    ``module`` and ``op`` (the HLO instruction that produced it).
    """
    paths = sorted(trace_dir.glob("**/*.trace.json.gz"))
    if not paths:
        raise FileNotFoundError(f"no *.trace.json.gz under {trace_dir}")
    events = json.load(gzip.open(paths[-1]))["traceEvents"]
    kernels = []
    for e in events:
        args = e.get("args") or {}
        # ``kernel_details`` is present only on device-side kernel events,
        # which is what separates them from host-side Python/XLA spans.
        if e.get("ph") != "X" or "kernel_details" not in args:
            continue
        kernels.append({
            "ts": float(e.get("ts", 0.0)),
            "dur": float(e.get("dur", 0.0)),
            "module": args.get("hlo_module", ""),
            "op": args.get("hlo_op", ""),
        })
    if not kernels:
        # Distinguish this from "the model did no work": the usual cause is the
        # trace's event budget being consumed by host events before any device
        # event lands, which otherwise surfaces as a division by zero in the
        # report rather than as the collection failure it is.
        raise RuntimeError(
            f"{paths[-1]} contains {len(events)} events but no GPU kernels. "
            "The device plane was not collected -- check that host/python "
            "tracer levels are 0 and that the run really used a GPU."
        )
    return kernels


def device_span_us(kernels: list[dict]) -> float:
    """Elapsed device time from the first kernel's start to the last one's end.

    The denominator for occupancy. Preferred over a host ``perf_counter``
    because it excludes the profiler's own start/stop cost and any host-side
    setup that happens to fall inside the traced block.
    """
    if not kernels:
        return 0.0
    return max(k["ts"] + k["dur"] for k in kernels) - min(
        k["ts"] for k in kernels
    )


@dataclasses.dataclass
class Attribution:
    """Device time per component, plus the honesty metrics for the split."""

    per_label: dict[str, float]
    kernels_per_label: dict[str, int]
    steps_seen: dict[str, int]
    mixed_us: float
    unjoinable_us: float
    total_us: float

    def as_dict(self) -> dict:
        """Return a JSON-serialisable view."""
        return {
            "per_label_us": self.per_label,
            "kernels_per_label": self.kernels_per_label,
            "steps_seen": self.steps_seen,
            "mixed_us": self.mixed_us,
            "unjoinable_us": self.unjoinable_us,
            "total_us": self.total_us,
        }


def attribute(
    kernels: list[dict],
    hlo_index: dict[str, dict[str, Instruction]],
    members: dict[str, dict[str, list[str]]],
) -> Attribution:
    """Charge each kernel's device time to the scope that emitted it.

    Parameters
    ----------
    kernels
        Output of :func:`load_trace_kernels`.
    hlo_index
        Module name to instruction map, from :func:`index_hlo_dump`.
    members
        Module name to computation-membership map, from
        :func:`parse_hlo_module_computations`.

    Returns
    -------
    Attribution
        ``per_label`` holds the summed microseconds per component, with
        ``"unattributed"`` collecting kernels traced outside any JCM scope.
        ``mixed_us`` is the device time in kernels whose fused instructions
        span more than one component -- the error bar on the split -- and
        ``unjoinable_us`` the time in kernels whose instruction was not found
        in the dump at all (a join failure, not a component).
        ``steps_seen`` is the minimum per-instruction execution count for each
        component. For the dynamics and bridge scopes it equals the number of
        steps the trace captured, which is how a truncated trace is detected.

    """
    from jcm import profiling

    per_label: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    # Executions per (label, instruction), reduced to a MINIMUM per label. For
    # the dynamics and bridge scopes -- which sit directly in Model's step,
    # outside every loop and branch -- no instruction can run fewer than once
    # per step, so that minimum is exactly the number of steps the trace
    # captured. It is a completeness check, not a cadence measurement: internal
    # loops and vmaps make the counts within one physics term span three orders
    # of magnitude, so no summary of them recovers "how often was this term
    # invoked". Cadence comes from the configuration (see subcycle_steps).
    op_count: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int))
    mixed_us = 0.0
    unjoinable_us = 0.0
    total_us = 0.0

    for k in kernels:
        total_us += k["dur"]
        instrs = hlo_index.get(k["module"])
        instr = instrs.get(k["op"]) if instrs else None
        if instr is None:
            unjoinable_us += k["dur"]
            per_label["unjoinable"] += k["dur"]
            counts["unjoinable"] += 1
            continue

        label = profiling.label_from_op_name(instr.op_name)
        # A fusion's own metadata is its root's; the constituent instructions
        # can come from more than one scope. Flag that rather than pretend the
        # root's label covers the whole kernel.
        inner = set()
        mod_members = members.get(k["module"], {})
        for comp in instr.calls:
            for name in mod_members.get(comp, ()):
                child = instrs.get(name)
                if child is not None:
                    inner.add(profiling.label_from_op_name(child.op_name))
        inner.discard(None)
        if len(inner | ({label} if label else set())) > 1:
            mixed_us += k["dur"]
        if label is None and len(inner) == 1:
            # The root is plumbing (a copy, a bitcast) but everything it fuses
            # came from one component: that component did the work.
            label = next(iter(inner))

        key = label or "unattributed"
        per_label[key] += k["dur"]
        counts[key] += 1
        op_count[key][k["op"]] += 1

    steps_seen = {key: min(per_op.values())
                  for key, per_op in op_count.items()}

    return Attribution(
        per_label=dict(per_label),
        kernels_per_label=dict(counts),
        steps_seen=steps_seen,
        mixed_us=mixed_us,
        unjoinable_us=unjoinable_us,
        total_us=total_us,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def render_report(result: dict) -> str:
    """Render the markdown report shown to the user and written to disk."""
    att = result["attribution"]
    nsteps = result["steps"]
    total = att["total_us"]
    span = result["device_span_us"]
    per_cycle = result.get("radiation_subcycle_steps", 1)
    subcycled = set(result.get("subcycled_terms") or ())
    lines = [
        f"# Step cost breakdown — {result['preset']}",
        "",
        f"- config: `{result['preset']}` — {result['config']}",
        f"- jcm: `{result.get('jcm_path', '?')}`",
        f"- GPU: {result['gpu_name']} (index {result['gpu_index']})",
        f"- steady-state steps profiled: {nsteps} "
        f"(after two discarded warm-up runs)",
        f"- radiation sub-cycle: every "
        f"{result.get('radiation_subcycle_steps', 1)} steps, gating "
        f"{', '.join(sorted(subcycled)) or 'nothing'}",
        f"- device elapsed per step: {span / nsteps / 1000:.2f} ms",
        f"- device busy per step: {total / nsteps / 1000:.2f} ms "
        f"({100 * total / span:.0f}% occupancy; the rest is launch gaps, "
        "which shrink with resolution)",
        "",
        "``ms/step`` amortises a component over every step of the window and",
        "is the throughput-relevant number. ``ms/call`` is what it costs on a",
        "step where it actually runs; the two differ for the terms gated to",
        "the radiation cadence, whose membership and interval are read from",
        "the configuration rather than inferred from the trace.",
        "",
        "| component | ms/step | % of device | ms/call | kernels/step |",
        "|---|---:|---:|---:|---:|",
    ]
    # Every term is listed, including those that cost nothing measurable: a
    # term absent from the table would read as "not measured" when what it
    # actually means is "free", which is the more interesting fact.
    per_label = dict(att["per_label_us"])
    for name in result.get("terms", ()):
        per_label.setdefault(name, 0.0)
    ordered = sorted(per_label.items(), key=lambda kv: -kv[1])
    for label, us in ordered:
        pct = 100.0 * us / total if total else 0.0
        cadence = per_cycle if label in subcycled else 1
        per_call = us * cadence / nsteps / 1000
        kernels = att["kernels_per_label"].get(label, 0) / nsteps
        lines.append(
            f"| {label} | {us / nsteps / 1000:.2f} | {pct:.1f} "
            f"| {per_call:.2f} | {kernels:.0f} |"
        )
    lines += [
        "",
        "## Attribution quality",
        "",
        f"- mixed kernels (fusions spanning components, charged whole to "
        f"their root's component): {100 * att['mixed_us'] / total:.1f}% "
        "of device time",
        f"- unjoinable kernels (HLO instruction absent from the dump): "
        f"{100 * att['unjoinable_us'] / total:.1f}%",
        "",
        "A large mixed fraction means the per-component split is soft and "
        "should not be quoted term by term; the totals remain sound.",
        "",
        "Note: CUDA graph capture is disabled so that kernels stay "
        "individually attributable, so the total step time here reads high "
        "against `tools/benchmark.py`. Use that tool for throughput.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The profiled run
# --------------------------------------------------------------------------


def profile_run(
    overrides: list[str], steps: int | None, days: float | None, cycles: int,
    dump_dir: pathlib.Path,
    trace_dir: pathlib.Path,
) -> dict:
    """Build the model, run it three times, and trace the third.

    1. ``jcm.runners.run`` — establishes the initial state from the preset's
       own ``init`` group and compiles. Discarded.
    2. ``Model.resume`` with exactly the arguments the traced call will use.
       Discarded. This second warm-up is not redundant: ``runners.run``
       dispatches through ``resume`` with its OWN argument set, and any
       difference (the snapshot arguments, a different ``ForcingData``
       instance) is a cache miss whose compilation would otherwise happen
       inside the traced block.
    3. The same call again, traced. A cache hit, hence steady state.

    Step 3 is ``resume`` rather than a repeat of ``runners.run`` because
    ``run`` rebuilds the forcing and re-injects the initial profile every time.
    Both do device work that no timestep pays for, and inside the traced block
    it would be profiled as step cost.

    A recompile inside the traced block would put XLA's autotuning kernels in
    the profile, silently inflating whichever components own the fusions being
    tuned, so the guard below turns that into a hard failure rather than a
    plausible-looking table.
    """
    import hydra
    import jax
    from hydra import compose, initialize_config_dir

    import jcm
    from jcm import runners

    # Locate the config groups through the imported package rather than this
    # file's own path, so profiling a worktree on PYTHONPATH reads that
    # worktree's configs and not the installed copy's. (``jcm/config`` is a
    # data directory with no ``__init__.py``, hence the dir form.)
    config_dir = pathlib.Path(jcm.__file__).resolve().parent / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # One output frame over the whole window and no chunking, so the traced
    # region is dominated by timestepping rather than by output assembly.
    time_step_min = float(cfg.run.time_step)
    steps_were_explicit = steps is not None

    model = runners.build_model(cfg)
    per_cycle, subcycled_terms = subcycle_steps(model, time_step_min)

    if steps is None:
        steps = (max(1, round(days * 1440.0 / time_step_min))
                 if days is not None else cycles * per_cycle)

    # The window must span a whole number of radiation sub-cycles, or it
    # averages in a different number of radiation calls per step than the model
    # pays -- at the default 2 h cadence, a factor-of-two error on the largest
    # row. An explicit --steps is honoured or rejected; a duration is rounded
    # up, since the user asked for a duration, not an exact count.
    if steps % per_cycle:
        if steps_were_explicit:
            raise SystemExit(
                f"--steps {steps} does not span a whole number of radiation "
                f"sub-cycles ({per_cycle} steps at radiation_interval / dt). "
                "The window would average the wrong number of radiation calls "
                f"per step. Use a multiple of {per_cycle}."
            )
        steps += per_cycle - (steps % per_cycle)

    window_days = steps * time_step_min / 1440.0
    cfg.run.total_time = window_days
    cfg.run.save_interval = window_days
    cfg.run.chunk_days = 0
    cfg.run.output_averages = False

    # Built once, outside the trace, exactly as jcm.runners._run_full does.
    forcing = runners.build_forcing(
        cfg, model.coords, dycore=getattr(model, "dycore", None),
    )
    forcing = runners._maybe_attach_nudging_target(forcing, cfg, model)

    resume_kwargs = dict(
        forcing=forcing,
        save_interval=window_days,
        total_time=window_days,
        output_averages=False,
    )

    # Warm-up 1: initial state + first compile. Warm-up 2: the traced call's
    # exact signature, so that its compilation (if any) happens here.
    runners.run(cfg, model=model)
    jax.block_until_ready(model._final_dycore_state)
    jax.block_until_ready(model.resume(**resume_kwargs))

    before = _dumped_modules(dump_dir)
    shutil.rmtree(trace_dir, ignore_errors=True)
    # Host and Python tracing off. The trace has a fixed event budget, and at
    # JAM resolution the host-side events exhaust it before a single device
    # event is recorded -- a T63 JAM trace came back with 1,000,000 host events
    # and an empty GPU plane. Only device events are read here anyway.
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    t0 = time.perf_counter()
    with jax.profiler.trace(str(trace_dir), profiler_options=options):
        predictions = model.resume(**resume_kwargs)
        jax.block_until_ready(predictions)
    wall_s = time.perf_counter() - t0

    recompiled = _dumped_modules(dump_dir) - before
    if recompiled:
        raise RuntimeError(
            "XLA compiled "
            f"{', '.join(sorted(recompiled))} inside the traced block, so the "
            "trace contains autotuning kernels and the breakdown would be "
            "wrong. The traced call was expected to hit the compilation "
            "cache warmed immediately before it."
        )

    terms = [t.name for t in getattr(model.physics, "terms", ())]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return {
        "steps": steps,
        "time_step_min": time_step_min,
        # The traced block as a whole. Deliberately NOT divided by steps: it
        # also covers the profiler's start/stop and the output assembly, which
        # for a short window dwarf the timestepping. Per-step cost comes from
        # the device timeline instead.
        "traced_block_wall_s": wall_s,
        "radiation_subcycle_steps": per_cycle,
        "subcycled_terms": sorted(subcycled_terms),
        "terms": terms,
        # Which checkout was profiled. An editable install shadowing a
        # worktree is silent otherwise, and the numbers would describe the
        # wrong code.
        "jcm_path": str(pathlib.Path(jcm.__file__).resolve().parent),
        "config": _config_summary(cfg, overrides),
        "dump_dir": str(dump_dir),
    }


def subcycle_steps(model, time_step_min: float) -> tuple[int, frozenset[str]]:
    """Return (steps between radiation calls, the names of terms on that gate).

    Radiation runs on a ``radiation_interval`` cadence (7200 s by default) and
    is usually the largest component. Two things depend on knowing that:

    - the profiled window must span a whole number of sub-cycles, or it
      averages in a different number of radiation calls per step than the model
      pays -- at the default 2 h cadence, a factor-of-two error on the biggest
      row;
    - a gated term's cost *on a step where it runs* is its per-step average
      times this factor.

    More than one term rides the gate. ``JamOpticsTerm`` is gated too, because
    its per-band Mie optics are consumed only by radiation, so recomputing them
    on intermediate steps would be discarded work. Membership is therefore
    taken from the codebase's own marker for it -- the ``configure_radiation_gate``
    method that ``echam_physics`` calls on exactly these terms -- rather than
    from a hardcoded name, so a future gated term is picked up automatically.

    Cadence is read from the configuration rather than inferred from the trace
    on purpose. Kernel execution counts cannot recover it: internal scans and
    vmaps make the per-instruction counts within a single term span three
    orders of magnitude (RRTMGP's range from 2 to 2816 over a 20-step window in
    which it ran exactly twice), so no summary statistic of them is the
    invocation count.

    Returns ``(1, frozenset())`` when nothing is sub-cycled.
    """
    per_cycle = 1
    gated: set[str] = set()
    for term in getattr(model.physics, "terms", ()):
        params = getattr(term, "params", None)
        value = params.get_value() if params is not None else None
        interval = getattr(value, "radiation_interval", None)
        # ``_radiation_interval_s`` is None when the gate is configured off
        # (interval <= 0), which is what distinguishes "gated" from "merely
        # gate-aware".
        if interval is None and hasattr(term, "configure_radiation_gate"):
            interval = getattr(term, "_radiation_interval_s", None)
        if interval:
            per_cycle = max(1, round(float(interval) / (time_step_min * 60.0)))
            name = getattr(term, "name", None)
            if name:
                gated.add(name)
    return per_cycle, frozenset(gated)


def _config_summary(cfg, overrides: list[str]) -> str:
    """One-line description of what was profiled, for the report header."""
    groups = [o for o in overrides
              if re.match(r"^(physics|grid|dycore|run|init)=", o)]
    dt = cfg.run.time_step
    return f"{' '.join(groups)} dt={dt}min"


def main(argv=None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preset", required=True, choices=sorted(PRESETS),
                   help="configuration to profile (shared with benchmark.py)")
    p.add_argument("--gpu", type=int, required=True,
                   help="GPU index; must be free (see tools/gpu_util.py)")
    # Default: two radiation sub-cycles (20 steps at dt=12 min). Not a whole
    # simulated day, though that is the intuitive choice, for two reasons.
    #
    # It is not needed: every kernel in a step has a static shape and the model
    # takes no data-dependent branches, so a step's cost does not vary with the
    # state or the time of day. Once the window spans a whole number of
    # radiation sub-cycles the per-step average is already exact, and further
    # steps only reduce timing jitter.
    #
    # And it does not fit: the profiler's event buffer holds ~1e6 events, and a
    # T63L47 JAM step emits ~19,000 kernels, so ~20 steps is the ceiling at
    # that resolution. A one-day (120-step) window silently recorded 20 steps
    # and reported a 4x undercount. --days is still available for cheaper
    # configurations; the run fails loudly if the buffer overflows.
    p.add_argument("--cycles", type=int, default=2,
                   help="radiation sub-cycles to trace (default 2)")
    p.add_argument("--days", type=float, default=None,
                   help="simulated days to trace; overrides --cycles")
    p.add_argument("--steps", type=int, default=None,
                   help="steps to trace; overrides --days and --cycles")
    p.add_argument("--label", default=None,
                   help="output subdirectory name (default: the preset)")
    p.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    p.add_argument("--extra", nargs="*", default=[],
                   help="additional Hydra overrides")
    p.add_argument("--allow-busy-gpu", action="store_true",
                   help="skip the free-GPU gate (results will be noisy)")
    args = p.parse_args(argv)

    # Same gate as tools/benchmark.py: a co-tenant on the card perturbs kernel
    # durations, and this tool reads those durations directly.
    g = next((x for x in gpu_table() if x["index"] == args.gpu), None)
    if g is None or not is_free(g):
        why = describe_gpu(args.gpu)
        if not args.allow_busy_gpu:
            print(f"{why}\nProfile on an idle card, or pass --allow-busy-gpu "
                  f"to proceed anyway. Free now: {free_indices()}",
                  file=sys.stderr)
            return 2
        print(f"warning: {why} -- proceeding because --allow-busy-gpu was "
              "passed. The resulting split is NOT trustworthy.",
              file=sys.stderr)

    outdir = args.outdir / (args.label or args.preset)
    outdir.mkdir(parents=True, exist_ok=True)
    dump_dir = outdir / "hlo"
    shutil.rmtree(dump_dir, ignore_errors=True)
    dump_dir.mkdir()

    # Set before jax is imported anywhere: XLA reads these when the backend is
    # first initialised, and both are needed for the attribution join to have
    # anything to join on.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_FLAGS"] = " ".join(filter(None, [
        os.environ.get("XLA_FLAGS", ""),
        f"--xla_dump_to={dump_dir}",
        "--xla_dump_hlo_pass_re=^$",
        _NO_CUDA_GRAPHS,
    ]))

    overrides = [*PRESETS[args.preset], *args.extra]
    trace_dir = outdir / "trace"
    result = profile_run(
        overrides, args.steps, args.days, args.cycles, dump_dir, trace_dir,
    )

    freed = prune_dump(dump_dir)
    print(f"pruned {freed / 2**20:.0f} MiB of PTX/LLVM IR from the HLO dump",
          file=sys.stderr)
    hlo_index, members, notes = index_hlo_dump(dump_dir)
    for n in notes:
        print(f"note: {n}", file=sys.stderr)

    kernels = load_trace_kernels(trace_dir)
    att = attribute(kernels, hlo_index, members)

    # The profiler's event buffer holds ~1e6 events and a JAM step emits
    # ~19,000 kernels, so a long enough window silently stops recording partway
    # instead of erroring -- a 120-step window captured only 20 steps and
    # reported a 4x undercount that looked entirely plausible.
    #
    # The dynamics and bridge scopes sit directly in Model's step function,
    # outside every loop and conditional, so each of their instructions runs
    # exactly once per step BY CONSTRUCTION. That makes them the trace's
    # completeness check. (A physics term is no good for this: internal scans
    # and vmaps put its per-instruction counts all over the place.)
    from jcm import profiling

    probes = {profiling.DYNAMICS, profiling.BRIDGE_TO_PHYSICS,
              profiling.BRIDGE_TO_DYNAMICS}
    # EVERY probe must be complete, not just the best of them. The probes are
    # ordered within a step (bridge_to_physics runs before dynamics), so a
    # buffer that fills partway through the final step leaves the earlier probe
    # at the full count while a later one is short; taking the max would accept
    # that trace and then divide short totals by the full step count.
    cycle = result.get("radiation_subcycle_steps", 1)
    seen = {label: n for label, n in att.steps_seen.items() if label in probes}
    if seen and min(seen.values()) < result["steps"]:
        worst = min(seen.values())
        fits = max(cycle, worst // cycle * cycle)
        raise SystemExit(
            f"the trace covers only {worst} of the {result['steps']} steps run "
            f"({len(kernels)} kernels captured; per-probe {seen}), so the "
            "profiler's event buffer overflowed and every number would be an "
            f"undercount. Profile a shorter window: --steps {fits} or fewer."
        )

    result.update({
        "preset": args.preset,
        "overrides": overrides,
        "gpu_index": args.gpu,
        "gpu_name": describe_gpu(args.gpu),
        "device_span_us": device_span_us(kernels),
        "hlo_index_notes": notes,
        "attribution": att.as_dict(),
    })
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    report = render_report(result)
    (outdir / "report.md").write_text(report)
    print(report)
    print(f"\nWritten to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
