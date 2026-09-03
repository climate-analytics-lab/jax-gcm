r"""Reproducible jcm throughput benchmark.

A thin wrapper around ``python -m jcm.main``: it builds a Hydra command line,
runs it, samples GPU telemetry alongside, and parses the per-chunk wall times
into a throughput number with an explicit convergence criterion. It does
**not** reimplement any part of the run loop — chunking, health gates and
checkpointing all stay in ``jcm.runners`` (see the "No bespoke run scripts"
rule in CLAUDE.md).

Why this exists rather than eyeballing the log:

* jcm's ``N sim days/hr`` log line is **cumulative including compile time**.
  Reading it as the throughput understates a run by 2-5x early on, and it
  keeps drifting upward for the whole run. A 5.3x "regression" was once filed
  against jax-rrtmgp on the strength of chunk 1 of a run that settled 22x
  faster. This tool ignores that line entirely and uses ``Wall: Xs this
  chunk``.
* Chunk times settle over 3-4 chunks (XLA autotuning, cache warming, host
  allocator). This tool requires two consecutive chunks agreeing within a
  tolerance before it reports a number, and says so when they never do.
* **High GPU utilisation does not prove steady state.** XLA's autotuner keeps
  the device at 95%+ while still picking kernels, so "the GPU is busy" is not
  evidence that a chunk time is converged. Only chunk-to-chunk agreement is.

Usage::

    python tools/benchmark.py --preset t63-echam-rrtmgp --months 1 --gpu 1
    python tools/benchmark.py --preset t63-echam-jam --months 12 --gpu 3 \\
        --label jam-baseline --pythonpath /path/to/lib/worktree

Results land in ``<outdir>/<label>/`` as ``report.md``, ``result.json``,
``run.log`` and ``gpu.csv``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import os
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from chunk_timing import analyse as analyse_chunks  # noqa: E402
from chunk_timing import DEFAULT_TOL, parse_walls  # noqa: E402
from gpu_util import describe as describe_gpu  # noqa: E402
from gpu_util import free_indices, is_free, gpu_table  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
# The interpreter running this script, not a hardcoded dev-box path: in a
# container (Nautilus) that path does not exist, and hardcoding it made the
# harness die instantly there. $JCM_PYTHON overrides when the model must run
# under a different interpreter from the wrapper.
DEFAULT_PY = os.environ.get("JCM_PYTHON") or sys.executable
DEFAULT_OUTDIR = pathlib.Path("/scr/dwatsonparris/benchmarks")
# A benchmark only needs the timings, the telemetry and the health verdict --
# all of which are read from the log while the run is happening. The netCDF it
# produces is dead weight: a 30-day T63L47 run writes several GB per arm, and
# a 3-pair A/B fills a scratch disk for nothing. So model output goes to a
# scratch directory that is deleted afterwards by default.
#
# NOT the same call as for a science run: there the output IS the point. This
# is why the removal lives in the benchmark harness and not in jcm-run.
DEFAULT_SCRATCH_ROOT = pathlib.Path(
    os.environ.get("JCM_BENCH_SCRATCH", "/tmp/jcm-bench"))

# Chunk-to-chunk agreement required before a rate is called converged.
CONVERGENCE_TOL = 0.03      # 3 % between consecutive chunks
GPU_SAMPLE_SECONDS = 10.0
# Utilisation above this counts as "the GPU is doing the run",
# separating integration from the idle stretches during compile.
_ACTIVE_UTIL_PCT = 5.0

_NAN_RE = re.compile(r"NaN vars:\s*(\d+)\s*/\s*(\d+)")
# check_health trips on more than NaN -- q_max, temperature range -- and
# run_chunked then stops early while jcm.main still exits 0. Without this a
# truncated run prints a converged-looking rate and returns success.
_UNHEALTHY_RE = re.compile(r"atmosphere unhealthy|FAILED: T_min|FAILED: T_max"
                           r"|q_max=", re.I)
_SAVED_RE = re.compile(r"Saved .*_day(\d+)\.nc")

# The experiment-group Hydra compositions (jcm/config/experiment/*.yaml) are the
# single home of the validated override sets -- physics x grid x radiation
# pairing x init x forcing -- each carrying a comment for WHY every setting is
# what it is (an isothermal cold start with no sponge goes NaN within days at
# L47, so these are not interchangeable with a bare ``grid=`` override). This
# table is only a thin shim: it maps each benchmark id to ``+experiment=<name>``
# plus the benchmark-ONLY overrides the yaml deliberately leaves out. Keeping
# the science in the yamls means a hand-composed ``python -m jcm.main
# +experiment=<name>`` and a benchmark run share one validated definition rather
# than drifting apart.
#
# The only benchmark-only overrides are machine-local data with no data-mirror
# bundle (T119 native terrain + level-matched ozone under $JCM_BC_DIR) and
# disposable output plumbing (the multi-GB pySES checkpoint, sent to scratch).
_BC = pathlib.Path(os.environ.get("JCM_BC_DIR", "/scr/dwatsonparris/bc_l95"))


def _exp(name: str, *extra: str) -> list[str]:
    """``+experiment=<name>`` plus any benchmark-only overrides."""
    return [f"+experiment={name}", *extra]


PRESETS: dict[str, list[str]] = {
    "speedy-t31": _exp("speedy-t31"),
    # Release-matrix MACv2-SP members (#638): echam-1m/2m at t63/t106.
    **{f"{t}-echam-{v}": _exp(f"{t}-echam-{v}")
       for t in ("t63", "t106") for v in ("1m", "2m")},
    # T63 RRTMGP / JAM family (historical benchmark ids).
    "t63-echam-rrtmgp": _exp("t63-echam-rrtmgp"),
    "t63-echam-rrtmgp-2m": _exp("t63-echam-rrtmgp-2m"),
    "t63-echam-emulated-2m": _exp("t63-echam-emulated-2m"),
    "t63-echam-jam": _exp("t63-echam-jam"),
    "t63-echam-jam-aerocom": _exp("t63-echam-jam-aerocom"),
    "t63-echam-jam-aerocom-optics": _exp("t63-echam-jam-aerocom-optics"),
    # Middle-atmosphere JAM sweep. t63/t106 are fully on the mirror; t119 has
    # no bundle, so its terrain + level-matched ozone stay machine-local.
    **{f"ma-{t}-l{lv}": _exp(f"ma-{t}-l{lv}")
       for t in ("t63", "t106") for lv in (47, 95)},
    **{f"ma-t119-l{lv}": _exp(
        f"ma-t119-l{lv}",
        f"terrain.file={_BC}/T119_terrain.nc",
        f"forcing.ozone_file={_BC}/t119_ozone_l{lv}.nc") for lv in (47, 95)},
    # pySES CAM-SE ne30 (dycore comparison); run=pyses_year drops a relative
    # checkpoint into cwd, so redirect it to the disposable scratch dir.
    **{f"ma-ne30-l{lv}": _exp(
        f"ma-ne30-l{lv}",
        f"run.checkpoint_path={DEFAULT_SCRATCH_ROOT}/pyses.ckpt")
       for lv in (47, 95)},
}


def _compose_preset(overrides: list[str]):
    """Compose the effective config for a preset's override list.

    Cheap (pure Hydra composition -- no model build, no jcm import), so it is
    safe to call before a GPU is claimed. Used to enumerate the prescribed-input
    files a preset resolves to, which since the PRESETS shim now live inside the
    ``+experiment`` yaml rather than in the override strings.
    """
    from hydra import compose, initialize_config_dir
    cfgdir = str(REPO / "jcm" / "config")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        return compose(config_name="config", overrides=overrides)


def _load_bundle_names():
    """Load ``jcm.data.bundle_names`` WITHOUT importing the ``jcm`` package.

    Same rationale (and mechanism) as :func:`_hf_fetch`: reaching the module as
    ``from jcm.data.bundle_names import ...`` executes ``jcm/__init__.py``,
    which initialises a JAX backend and preallocates ~75 % of the device the
    instant it is touched — before the free-GPU gate. ``bundle_names.py`` has
    no intra-package imports (that is a maintained invariant), so loading it by
    file path is safe and keeps the auto-bundle naming convention a single
    source of truth shared with ``jcm.runners``.
    """
    import importlib.util
    src = REPO / "jcm" / "data" / "bundle_names.py"
    spec = importlib.util.spec_from_file_location("_jcm_bundle_names", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _auto_emission_files(cfg) -> list[str]:
    """``hf://`` bundles the JAM ``auto`` emission default resolves to.

    :func:`_preset_data_files` collects only the LITERAL paths in the composed
    config, but the four prescribed-emission keys default to ``auto`` and are
    resolved lazily by ``jcm.runners`` during model construction — i.e. AFTER
    the GPU is claimed and the telemetry sampler is running. Enumerate them here
    (from the same ``jcm.data.bundle_names`` convention the runner uses) so they
    join the pre-GPU prefetch: an unreachable or non-existent bundle then
    refuses the run up front instead of stalling on a held card.

    Mirrors ``jcm.runners._resolve_one_emission_input``: ``auto`` supplies a
    bundle only when a prognostic-aerosol (JAM) *spectral* package is active. A
    non-JAM package consumes no emissions, and the pySES backend's native grids
    are not the spectral-token bundles — both resolve ``auto`` to nothing. A
    grid outside the mirror's published-grid whitelist (``PUBLISHED_GRIDS``)
    likewise has no bundle to fetch, so ``auto`` nulls there too. A key
    explicitly set to a path/``null`` in the preset is honoured (the literal
    path is already picked up by ``_preset_data_files``; ``null`` opts out).
    """
    phys = cfg.get("physics") or {}
    if str(phys.get("aerosol_module", "")) != "jam":
        return []
    if str((cfg.get("dycore") or {}).get("name", "")) == "pyses":
        return []
    grid = cfg.get("grid") or {}
    trunc, nlev = grid.get("spectral_truncation"), grid.get("layers")
    if trunc is None or nlev is None:
        return []
    names = _load_bundle_names()
    forcing = cfg.get("forcing") or {}
    token = names.grid_token(trunc)
    # Mirror the runner's published-grid whitelist: a non-mirrored grid has no
    # bundles to prefetch (``auto`` resolves to None there), so enumerate none.
    if token not in names.PUBLISHED_GRIDS:
        return []
    return [path
            for key, path in names.auto_emission_bundle_paths(
                token, nlev).items()
            if str(forcing.get(key, "auto")) == "auto"]


def _preset_data_files(overrides: list[str]) -> list[str]:
    """Prescribed-input paths (hf:// or local) a preset resolves to.

    Walks the COMPOSED forcing/terrain/dycore config for keys ending in
    ``file`` and returns the concrete paths, skipping ``auto``/``null``/``none``
    (resolved lazily at build time) and unset ``???`` values — then ADDS the
    ``auto`` emission bundles a JAM preset resolves lazily (see
    :func:`_auto_emission_files`), which the ``file``-key walk cannot see.
    """
    from omegaconf import OmegaConf
    cfg = _compose_preset(overrides)
    out: list[str] = []

    def _add(v):
        if isinstance(v, str) and v not in ("auto", "null", "none", "???"):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _add(x)

    for group in ("forcing", "terrain", "dycore"):
        node = cfg.get(group, None)
        if node is None:
            continue
        cont = OmegaConf.to_container(node, resolve=False,
                                      throw_on_missing=False)
        if not isinstance(cont, dict):
            continue
        for k, v in cont.items():
            if str(k).endswith("file"):
                _add(v)
    out += _auto_emission_files(cfg)
    return out


def _gpu_sampler(gpu: int, out_path: pathlib.Path, stop: threading.Event):
    """Sample memory/utilisation until ``stop`` is set."""
    query = ("--query-gpu=timestamp,memory.used,memory.total,"
             "utilization.gpu,utilization.memory,power.draw")
    with out_path.open("w") as fh:
        fh.write("timestamp,mem_used_mib,mem_total_mib,util_gpu_pct,"
                 "util_mem_pct,power_w\n")
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={gpu}", query,
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20, check=False)
                if out.stdout.strip():
                    fh.write(out.stdout.strip() + "\n")
                    fh.flush()
            except (OSError, subprocess.SubprocessError):
                pass    # a transient nvidia-smi failure must not kill the run
            stop.wait(GPU_SAMPLE_SECONDS)


def _summarize_gpu(csv_path: pathlib.Path) -> dict:
    """Peak memory and steady-state utilisation from the telemetry."""
    if not csv_path.exists():
        return {}
    mem, util, power, total = [], [], [], []
    for line in csv_path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            mem.append(float(parts[1]))
            total.append(float(parts[2]))
            util.append(float(parts[3]))
            power.append(float(parts[5]))
        except ValueError:
            continue
    if not mem:
        return {}
    # Utilisation/power are summarised over ACTIVE samples only -- those
    # above _ACTIVE_UTIL_PCT. A positional heuristic ("drop the first 20 %")
    # does not work: XLA compilation leaves the GPU genuinely idle, and on a
    # short run compile is most of the wall clock, so a positional cut still
    # averages mostly zeros and reports ~0 % for a run that was pegged at 85 %
    # whenever it was actually integrating.
    #
    # Peak MEMORY keeps the whole series -- an allocation spike during
    # compile is still a real provisioning requirement.
    #
    # ``active_fraction`` is reported because it is informative in its own
    # right: it is the share of wall time the GPU was busy, so a short run
    # shows how much went to compile, and a long run that does NOT approach
    # 1.0 is host-bound (dispatch, I/O, chunk writes) rather than
    # compute-bound.
    active = [(u, p) for u, p in zip(util, power) if u > _ACTIVE_UTIL_PCT]
    au = [u for u, _ in active] or util
    ap = [p for _, p in active] or power
    return {
        "peak_mem_mib": max(mem),
        "peak_mem_gib": round(max(mem) / 1024, 2),
        "mem_total_gib": round(max(total) / 1024, 2) if total else None,
        "median_util_active_pct": round(statistics.median(au), 1),
        "max_util_pct": round(max(util), 1),
        "median_power_active_w": round(statistics.median(ap), 1),
        "active_fraction": round(len(active) / len(util), 3) if util else 0.0,
        "n_samples": len(mem),
        "n_active_samples": len(active),
    }


# Libraries whose version materially changes what a benchmark measures. All
# are editable installs, so the working tree IS the running code -- a report
# that does not say which tree it used cannot be reproduced or trusted.
_PROVENANCE_MODULES = ("jcm", "rrtmgp", "dinosaur", "mam4_jax")


def _provenance(env: dict, python: str = None) -> dict:
    """Resolve each key library to a path + git SHA under the run's env.

    Runs a probe subprocess with the SAME interpreter and environment as the
    benchmark, so a ``PYTHONPATH`` override is reflected exactly as the run saw
    it. Both halves matter: the environment gives the override that was
    actually in force, and the interpreter matters because ``--python``
    routinely points at a different environment from the one running this
    wrapper (the micromamba env is not on PATH here). Probing
    ``sys.executable`` would happily attribute a benchmark to whichever
    ``jcm``/``rrtmgp`` the *wrapper* imports -- a provenance record that is
    confidently wrong is worse than none.
    """
    # Record a version alongside the path: a packaged (non-editable) install
    # has no git SHA, and "site-packages" alone does not identify what ran.
    probe = (
        "import importlib,importlib.metadata as md,os,json\n"
        "out={}\n"
        "for m in %r:\n"
        "    e={}\n"
        "    try:\n"
        "        e['path']=os.path.dirname(importlib.import_module(m).__file__)\n"
        "    except Exception as ex:\n"
        "        e['path']='unavailable: %%s' %% type(ex).__name__\n"
        "    for dist in (m, m.replace('_','-')):\n"
        "        try:\n"
        "            e['version']=md.version(dist); break\n"
        "        except Exception:\n"
        "            pass\n"
        "    out[m]=e\n"
        "print(json.dumps(out))" % (_PROVENANCE_MODULES,)
    )
    try:
        r = subprocess.run([python or sys.executable, "-c", probe],
                           capture_output=True, text=True, timeout=180,
                           env=env, cwd=str(REPO), check=False)
        found = json.loads(r.stdout.strip() or "{}")
    except (subprocess.SubprocessError, OSError, ValueError):
        return {}

    out = {}
    for mod, entry in found.items():
        path = entry.get("path", "")
        if isinstance(path, str) and os.path.isdir(path):
            for key, args in (("sha", ["rev-parse", "--short", "HEAD"]),
                              ("branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
                g = subprocess.run(["git", "-C", path, *args],
                                   capture_output=True, text=True, timeout=30,
                                   check=False)
                if g.returncode == 0:
                    entry[key] = g.stdout.strip()
            d = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30,
                               check=False)
            if d.returncode == 0:
                entry["dirty"] = bool(d.stdout.strip())
        out[mod] = entry
    return out


def _require_free_gpu(idx: int, wait_s: float, allow_busy: bool) -> None:
    """Refuse to start unless the target GPU is genuinely idle.

    Waits rather than failing immediately (the documented remedy is "wait for
    a free card"), which also absorbs the second or two the driver takes to
    release memory after a previous run exits.
    """
    deadline = wait_s
    while True:
        g = next((x for x in gpu_table() if x["index"] == idx), None)
        if g is not None and is_free(g):
            return
        why = describe_gpu(idx)
        if allow_busy:
            print(f"warning: {why} -- proceeding because --allow-busy-gpu was "
                  "passed. The resulting timing is NOT trustworthy.",
                  file=sys.stderr)
            return
        if deadline <= 0:
            free = free_indices()
            raise SystemExit(
                f"{why}\n"
                "Benchmarks must run on a genuinely idle card: a shared GPU "
                "yields a wrong number that still looks plausible.\n"
                + (f"Free right now: {free}. Use --gpu {free[0]}."
                   if free else "No GPU is free right now.")
                + " Or --wait-for-gpu SECONDS, or --allow-busy-gpu to accept "
                  "an untrustworthy result."
            )
        step = min(30.0, deadline)
        print(f"{why} -- waiting {deadline:.0f}s more", file=sys.stderr)
        time.sleep(step)
        deadline -= step


def _hf_fetch(path: str) -> str:
    """Prefetch one mirror file, WITHOUT importing the ``jcm`` package.

    ``jcm.data.remote.fetch`` is the function we want, but reaching it as
    ``from jcm.data.remote import fetch`` executes ``jcm/__init__.py``,
    which initialises a JAX backend -- and JAX preallocates ~75 % of the
    device the instant it is touched. Doing that here, before the free-GPU
    gate, makes the harness look like a 61 GiB tenant to its own gate; a
    six-job sweep died that way. So load the module from its file with no
    package context: ``remote.py`` has no intra-package imports, which is
    what makes this safe, and it stays the single source of truth for the
    dataset id rather than being copied in here.
    """
    import importlib.util
    src = REPO / "jcm" / "data" / "remote.py"
    spec = importlib.util.spec_from_file_location("_jcm_remote", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fetch(path)


def run(args) -> dict:
    preset = PRESETS[args.preset]
    days = args.days if args.days else args.months * 30
    chunk = args.chunk_days
    # Every file the preset names must be in hand BEFORE a GPU is claimed. A
    # preset referencing prepared boundary data is unusable wherever that
    # data is absent (a container, another machine), and finding out 20
    # minutes into a pod — after the image pull and the clone — wastes the
    # slot and the quota.
    #
    # ``hf://`` paths are DOWNLOADED here rather than merely checked. The
    # mirror bundles run to ~2 GB (t106_l95 oxidants), and jcm resolves them
    # lazily during model construction — which on this path is after the GPU
    # is claimed and the telemetry sampler is running. Pulling them first
    # keeps the download out of the timed region and turns an unreachable
    # mirror into an immediate refusal instead of a stall on a held card.
    #
    # The validated file references now live inside the ``+experiment`` yaml,
    # not in the override strings, so enumerate them from the COMPOSED config
    # (``auto``/``null`` inputs resolve lazily at build time and are skipped).
    #
    # ``args.extra`` is folded in FIRST so the preflight enumerates the SAME
    # inputs the real command (``[*preset, ..., *args.extra]`` below) will build
    # from: an ``--extra`` that nulls an auto input, or repoints a ``file`` key
    # at a local path, changes which bundles the run actually needs. Composing
    # the preset alone would prefetch (or fail offline on) bundles the effective
    # config never uses.
    files = _preset_data_files([*preset, *args.extra])
    missing = []
    for f in files:
        if f.startswith("hf://"):
            try:
                print(f"prefetching {f}", file=sys.stderr)
                _hf_fetch(f[len("hf://"):])
            except Exception as e:              # unreachable or absent
                missing.append(f"{f}  ({type(e).__name__}: {e})")
        elif not pathlib.Path(f).exists():
            missing.append(f)
    if missing:
        raise SystemExit(
            "preset references files that are not available here:\n  "
            + "\n  ".join(missing)
            + "\nSet $JCM_BC_DIR, or use a preset whose data is present.")

    # Gate first: a refused run must not leave directories behind.
    #
    # ``--gpu auto`` selects AND claims in one loop. Selecting with a separate
    # tool and then gating here leaves a window: on a busy shared box a card
    # reported free is routinely taken in the seconds before the gate runs,
    # and the run then dies having waited out its whole timeout on a card
    # that is no longer available. Re-picking on each poll fixes that.
    if str(args.gpu) == "auto":
        deadline = time.monotonic() + max(args.wait_for_gpu, 60.0)
        while True:
            free = free_indices()
            if free:
                args.gpu = free[0]
                break
            if time.monotonic() >= deadline:
                raise SystemExit(
                    "no GPU became free within --wait-for-gpu; nothing was "
                    "written. Re-run later or pass an explicit --gpu with "
                    "--allow-busy-gpu.")
            time.sleep(20)
    else:
        args.gpu = int(args.gpu)
    _require_free_gpu(args.gpu, wait_s=args.wait_for_gpu,
                      allow_busy=args.allow_busy_gpu)

    outdir = pathlib.Path(args.outdir) / (args.label or args.preset)
    outdir.mkdir(parents=True, exist_ok=True)

    # Model output (netCDF, checkpoints) goes somewhere disposable; the report,
    # log and telemetry stay in outdir. --keep-output writes it into outdir
    # instead and leaves it, for when a run needs inspecting after the fact.
    if args.keep_output:
        data_dir = outdir
    else:
        data_dir = (pathlib.Path(args.scratch_root)
                    / f"{args.label or args.preset}")
        if data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        *preset,
        f"run.total_time={days}",
        f"run.chunk_days={chunk}",
        # save_interval must be <= chunk_days or the chunk write dies with an
        # IndexError from to_xarray() on an empty time axis.
        f"run.save_interval={min(args.save_interval, chunk)}",
        # With --allow-unhealthy the driver keeps integrating past a health
        # gate trip. Timing stays valid when it does: XLA runs the same
        # compiled program over the same shapes regardless of the values in
        # them, so NaN arithmetic costs what finite arithmetic costs. What is
        # NOT valid is the science, so the report says so loudly.
        # ``++`` not ``+``/bare: run/longrun.yaml does NOT define this key
        # while run/pyses_year.yaml does, so a bare override dies with
        # "Key 'bail_on_unhealthy' is not in struct" on one and a bare ``+``
        # would die on the other. ``++`` adds-or-overrides either way.
        f"++run.bail_on_unhealthy={'false' if args.allow_unhealthy else 'true'}",
        f"run.output_prefix={data_dir}/state",
        f"hydra.run.dir={data_dir}",
        *args.extra,
    ]
    cmd = [args.python, "-m", "jcm.main", *overrides]

    env_note = {}
    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath
        env_note["PYTHONPATH"] = args.pythonpath
    # Precision belongs in the report, not in an invisible shell export: it
    # changes BOTH memory and speed, so a sweep that mixes it is not
    # internally comparable, and a number quoted without it is unreadable.
    # f32 is required above T63 (the f64 model saturates an 80 GiB card at
    # ~62 GiB and OOMs at T106/T119 L95) and is forward-only — MAM4
    # microphysics gradients are non-finite in f32, which is fine for a
    # benchmark and not for a training run.
    env["MAM4_JAX_ENABLE_X64"] = "0" if args.f32 else "1"
    env_note["MAM4_JAX_ENABLE_X64"] = env["MAM4_JAX_ENABLE_X64"]

    log_path = outdir / "run.log"
    gpu_path = outdir / "gpu.csv"
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler,
                               args=(args.gpu, gpu_path, stop), daemon=True)
    sampler.start()

    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, check=False)
    wall_total = time.time() - t0
    stop.set()
    sampler.join(timeout=30)

    log = log_path.read_text()
    walls = parse_walls(log)
    nan_hits = [(int(a), int(b)) for a, b in _NAN_RE.findall(log)]
    last_day = max((int(d) for d in _SAVED_RE.findall(log)), default=0)

    result = {
        "label": args.label or args.preset,
        "preset": args.preset,
        "months": args.months,
        "requested_days": days,
        "completed_days": last_day,
        "chunk_days": chunk,
        "gpu_index": args.gpu,
        "gpu_name": _gpu_name(args.gpu),
        "exit_code": proc.returncode,
        "total_wall_s": round(wall_total, 1),
        "unhealthy": bool(_UNHEALTHY_RE.search(log)),
        # A run that stopped early is not a benchmark of the requested
        # workload even if every chunk it did finish looked healthy.
        "truncated": last_day < days,
        "allow_unhealthy": bool(args.allow_unhealthy),
        "nan_any": any(n > 0 for n, _ in nan_hits),
        "nan_max_vars": max((n for n, _ in nan_hits), default=0),
        "nan_total_vars": nan_hits[0][1] if nan_hits else None,
        "overrides": overrides,
        "env": env_note,
        "provenance": _provenance(env, args.python),
        **analyse_chunks(walls, chunk, tol=args.tol),
        "gpu": _summarize_gpu(gpu_path),
    }
    # Reclaim the model output. Deliberately AFTER the log has been parsed and
    # the report written, and skipped when the run was unhealthy so a failure
    # can still be investigated -- deleting the evidence of a bad run is how
    # you end up unable to explain it.
    if not args.keep_output and data_dir != outdir:
        if should_keep_output(result):
            result["output_kept_at"] = str(data_dir)
            print(f"run did not complete cleanly — model output kept at "
                  f"{data_dir} for investigation", file=sys.stderr)
        else:
            freed = _dir_size_mib(data_dir)
            shutil.rmtree(data_dir, ignore_errors=True)
            result["output_removed_mib"] = round(freed, 1)
    # Written last, so the report can state what happened to the output.
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    (outdir / "report.md").write_text(_report(result))
    return result


def should_keep_output(result: dict) -> bool:  # noqa: D401
    """Whether a benchmark's model output is worth keeping.

    Only a run that completed cleanly is safe to discard: its numbers are in
    the report and the fields were never the point. Anything that failed,
    NaN'd, tripped the health gate or stopped short keeps its output, because
    deleting the evidence of a failure is how you end up unable to explain it.
    """
    return bool(result.get("nan_any") or result.get("unhealthy")
                or result.get("truncated") or result.get("exit_code", 0) != 0)


def _dir_size_mib(d: pathlib.Path) -> float:
    total = 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total / (1024 * 1024)


def _gpu_name(idx: int) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _report(r: dict) -> str:
    g = r.get("gpu") or {}
    lines = [
        f"# jcm benchmark — {r['label']}",
        "",
        f"- preset: `{r['preset']}`",
        f"- GPU {r['gpu_index']}: {r['gpu_name']}",
        f"- precision: {'f32' if r.get('env', {}).get('MAM4_JAX_ENABLE_X64') == '0' else 'f64'}"
        " (MAM4 core)",
        f"- requested {r['requested_days']} d, "
        f"completed {r['completed_days']} d",
        f"- exit code: {r['exit_code']}",
        "",
        "## Throughput",
        "",
    ]
    if r.get("converged") is None:
        lines.append("No chunk timings parsed — the run did not get that far.")
    elif r.get("s_per_sim_day"):
        status = "converged" if r["converged"] else "**NOT CONVERGED**"
        lines += [
            f"**{r['sim_days_per_hour']} sim days/hr** "
            f"({r['s_per_sim_day']} s per sim day, "
            f"{r['sim_years_per_day']} sim years/day) — {status}",
            "",
            f"- {r['reason']}",
            f"- compile chunk: {r['compile_chunk_s']} s (discarded)",
            f"- per-chunk walls: {r['chunk_walls_s']}",
        ]
    else:
        lines.append(f"Not enough chunks: {r.get('reason')}")
    prov = r.get("provenance") or {}
    if prov:
        lines += ["", "## Code measured", ""]
        for mod, e in sorted(prov.items()):
            dirty = " **+uncommitted**" if e.get("dirty") else ""
            if e.get("sha"):
                what = f"{e['sha']} ({e.get('branch', '?')}){dirty}"
            else:
                what = f"v{e.get('version', '?')} (packaged)"
            lines.append(f"- `{mod}` {what} — `{e.get('path')}`")
    lines += ["", "## GPU", ""]
    if g:
        lines += [
            f"- peak memory: {g['peak_mem_gib']} GiB of "
            f"{g['mem_total_gib']} GiB",
            f"- median utilisation while active: "
            f"{g['median_util_active_pct']} % (max {g['max_util_pct']} %)",
            f"- median power while active: {g['median_power_active_w']} W",
            f"- GPU busy for {g['active_fraction']:.0%} of wall time "
            f"({g['n_active_samples']}/{g['n_samples']} samples)",
        ]
    else:
        lines.append("No GPU telemetry collected.")
    if r.get("output_removed_mib") is not None:
        lines += ["", "## Output", "",
                  f"- model netCDF discarded after the report was written "
                  f"({r['output_removed_mib']:.0f} MiB reclaimed) — a "
                  f"benchmark needs the timings, not the fields. "
                  f"`--keep-output` to retain them."]
    elif r.get("output_kept_at"):
        lines += ["", "## Output", "",
                  f"- **kept** at `{r['output_kept_at']}` because the run was "
                  f"unhealthy — investigate before deleting."]
    lines += ["", "## Health", ""]
    bad = []
    if r["nan_any"]:
        bad.append(f"**NaN detected** — up to {r['nan_max_vars']}/"
                   f"{r['nan_total_vars']} variables.")
    if r.get("unhealthy"):
        bad.append("**Health gate tripped** (temperature range / q_max, not "
                   "necessarily NaN) — the run was stopped early.")
    if r.get("truncated"):
        bad.append(f"**Truncated** — completed {r['completed_days']} of "
                   f"{r['requested_days']} requested days.")
    if bad and r.get("allow_unhealthy"):
        lines += bad + [
            "", "**Run with `--allow-unhealthy`: the timing below is a "
            "deliberate COMPUTE-COST measurement of a configuration already "
            "known to be unstable.** XLA executes the same compiled program "
            "over the same shapes whatever the values, so the throughput is "
            "valid; the simulated fields are not. Do not use this run for "
            "anything scientific."]
    elif bad:
        lines += bad + [
            "", "Timing from a run that did not complete the requested "
            "workload healthily is not a valid benchmark: fix the "
            "configuration and re-run."]
    else:
        lines.append("Completed the full request with no NaN and no health "
                     "gate trip.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--months", type=int, default=1,
                   help="1 for the short benchmark, 12 for the long one "
                        "(a 'month' is 30 days here)")
    p.add_argument("--days", type=int, default=None,
                   help="explicit sim-day count; overrides --months")
    p.add_argument("--gpu", required=True,
                   help="GPU index, or 'auto' to select and claim a free card "
                        "in one operation (avoids the select-then-gate race "
                        "on a busy shared box)")
    p.add_argument("--label", default=None)
    p.add_argument("--chunk-days", type=int, default=5,
                   help="5 gives several post-compile chunks in a 30-day run")
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--pythonpath", default=None,
                   help="prepend a library worktree (editable-install A/B)")
    p.add_argument("--allow-unhealthy", action="store_true",
                   help="keep integrating past NaN / health-gate trips and "
                        "report the throughput anyway. For measuring COMPUTE "
                        "COST of a configuration already known to be "
                        "unstable; the fields are meaningless.")
    p.add_argument("--f32", action="store_true",
                   help="MAM4_JAX_ENABLE_X64=0 — required above T63; "
                        "forward-only (MAM4 gradients are non-finite in f32)")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   help="chunk-to-chunk agreement required to call a rate "
                        # %% : argparse %-expands help strings
                        f"converged (default {DEFAULT_TOL:.0%})".replace(
                            "%", "%%"))
    p.add_argument("--keep-output", action="store_true",
                   help="write model netCDF into the result directory and "
                        "keep it; default is a disposable scratch dir that is "
                        "removed once the report is written")
    p.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_ROOT),
                   help=f"disposable model-output root "
                        f"(default {DEFAULT_SCRATCH_ROOT}; "
                        "override with $JCM_BENCH_SCRATCH)")
    p.add_argument("--wait-for-gpu", type=float, default=120.0,
                   help="seconds to wait for the target GPU to become free "
                        "before giving up (default 120)")
    p.add_argument("--allow-busy-gpu", action="store_true",
                   help="run even if the GPU is in use; the timing will NOT "
                        "be trustworthy")
    p.add_argument("--extra", nargs="*", default=[],
                   help="additional raw Hydra overrides")
    args = p.parse_args(argv)

    if not shutil.which("nvidia-smi"):
        print("warning: nvidia-smi not found; no GPU telemetry",
              file=sys.stderr)
    r = run(args)
    print((pathlib.Path(args.outdir) /
           (args.label or args.preset) / "report.md").read_text())
    ok = r["exit_code"] == 0 and (
        r.get("allow_unhealthy")
        or not (r["nan_any"] or r.get("unhealthy") or r.get("truncated")))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
