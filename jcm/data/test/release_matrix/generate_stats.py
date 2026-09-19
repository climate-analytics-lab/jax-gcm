"""Climatology-band fixtures for the supported-configuration matrix.

One band file per member of ``tools/release_validation/matrix.yaml``, built
through that member's *validated preset* (``jcm/config/configuration/*.yaml``,
the same recipes ``tools/benchmark.py`` and the release-validation launcher
use) rather than a composition assembled here. A fixture that hand-rolls its
physics tests something nobody runs; going through the preset means the
regression covers exactly what the project claims to support.

What is where
-------------
* **Bands** — ``<member>_statistics.nc`` in this directory, a few KB each and
  checked in, so a change in what the model produces shows up as a reviewable
  diff.
* **Initial states** — *not* in git. Each member resumes from a stamped
  checkpoint on the Hugging Face data mirror under
  ``bundles/<grid>_<levels>/init_states/``, fetched cache-first by
  :func:`jcm.data.remote.fetch`. They are tens of MB and would otherwise be
  re-committed in full on every regeneration.

Each member's band file and its init state are a matched pair: the bands
describe the window that *follows* that exact state, so regenerating one
without the other compares a trajectory against bands drawn from a different
starting point.

Regenerating
------------
One command per member, on a GPU::

    CUDA_VISIBLE_DEVICES=<idx> python -c "from jcm.data.test.release_matrix.generate_stats import generate; generate('echam-1m-t63', out_dir='/scr/$USER/fixtures')"

``generate`` writes the band file and, unless ``write_state=False``, the
member's init state to ``out_dir`` for upload to the mirror (see
``docs/source/design/data_mirror.md`` for the publish path — uploads are
deliberately explicit and additive).

Where each member starts
------------------------
Every member spins up for ``SPIN_UP_DAYS`` from its **preset's own init**, and
the bands cover the ``STATS_DAYS`` that follow. The mirror does host older
equilibrated year-2 states from the #638 campaign, and starting from those
would give better-conditioned bands than any short spin-up, but current jcm
cannot read them: they are unstamped *and* structurally stale, storing 118
physics-carry arrays where an ECHAM T63L47 model now expects 146 (51 vs 56 for
SPEEDY). An unstamped file carries no field names, so #834 refuses the
structural difference rather than guessing at it — correctly. Re-equilibrating
them is #762's deliverable, not this module's.

Consequence, and it is a real limitation rather than a detail: a five-day
window out of a from-cold transient gives bands that are narrow, fast-moving
and unrepresentative of the model's climate. They are a *regression* signal —
"this member still produces what it produced" — and not a climatology. Treat a
failure as "something changed", not as "the physics is wrong".

The JAM members carry a further caveat: their bands describe the aerosol
climate before the in-flight dust retune (#787/#808) and must be regenerated
when it lands. That is recorded in the band file's own ``provisional``
attribute, not just here.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent
# Resolved from this file rather than the cwd: ``generate`` spawns its
# reproducibility repeats as subprocesses, and a fixture set that silently
# depends on where it was launched from is exactly the drift this module
# exists to remove.
_MATRIX = _HERE.parents[3] / "tools" / "release_validation" / "matrix.yaml"

#: Variables a band file carries when the member produces them. The list is the
#: union across packages — SPEEDY and ECHAM name their scheme outputs
#: differently — and is intersected with what the run actually emits, so one
#: list serves every member and a member that gains a scheme picks it up on the
#: next regeneration. The band file records the names it ended up with, and the
#: test reads them back from the file rather than re-deriving them here.
CANDIDATE_STAT_VARS = (
    # Prognostic state, every package.
    "u_wind",
    "v_wind",
    "temperature",
    "specific_humidity",
    "normalized_surface_pressure",
    # Moist-air diagnostics (MoistAirColumnState; ECHAM compositions).
    "pressure_full",
    "air_density",
    "layer_thickness",
    "relative_humidity",
    # ECHAM scheme outputs.
    "radiation.toa_lw_up",
    "radiation.surface_sw_down",
    "clouds.cloud_fraction",
    "clouds.precip_rain",
    "convection.precip_conv",
    # SPEEDY scheme outputs.
    "longwave_rad.ftop",
    "shortwave_rad.ftop",
    "humidity.rh",
    "condensation.precls",
    "convection.precnv",
)

#: ``member -> bundle``: where on the mirror this member's init state lives.
MEMBER_BUNDLE = {
    "speedy-t31": "t31_l8",
    "echam-1m-t63": "t63_l47",
    "echam-2m-t63": "t63_l47",
    "echam-1m-t106": "t106_l47",
    "echam-2m-t106": "t106_l47",
    "echam-jam-t63-l47": "t63_l47",
    "echam-jam-t63-l95": "t63_l95",
}

#: Members whose bands describe an aerosol climate the in-flight dust retune
#: (#787/#808) will move, and which therefore have to be regenerated when it
#: lands. Recorded in the band file so the fixture itself says so.
PRE_DUST_RETUNE = ("echam-jam-t63-l47", "echam-jam-t63-l95")

SPIN_UP_DAYS = 5.0
STATS_DAYS = 5.0
SAVE_INTERVAL_DAYS = 1.0


def members() -> dict[str, str]:
    """``member -> preset`` from the release-validation matrix.

    Read from ``matrix.yaml`` rather than duplicated here so the fixture set
    and the validation matrix cannot drift apart.
    """
    import yaml

    with open(_MATRIX) as f:
        matrix = yaml.safe_load(f)
    return {name: spec["preset"] for name, spec in matrix["members"].items()}


def band_path(member: str) -> Path:
    """In-repo band file for ``member``."""
    return _HERE / f"{member}_statistics.nc"


def state_mirror_path(member: str) -> str:
    """Mirror path of ``member``'s init state, as ``fetch`` takes it."""
    return f"bundles/{MEMBER_BUNDLE[member]}/init_states/{member}_fixture.msgpack"


def resolve_state(member: str) -> str:
    """Local path to ``member``'s init state, fetching it if necessary.

    Normally the mirror, resolved cache-first, so a warm cache needs no
    network and a cold one on an internet-less node fails with the prefetch
    instructions rather than a bare error.

    ``JCM_FIXTURE_STATE_DIR`` overrides that with a directory of freshly
    generated states. This exists because regenerating a fixture and
    publishing it are necessarily two steps — the state has to be validated
    against its own bands *before* anyone uploads it, and without the override
    that check could only be done after publishing, which is the wrong order.
    It is deliberately not a silent fallback: a member missing from the
    override directory raises rather than quietly reaching for the mirror,
    since the whole point of setting it is to test the local files.
    """
    import os

    override = os.environ.get("JCM_FIXTURE_STATE_DIR")
    if override:
        local = Path(override) / Path(state_mirror_path(member)).name
        if not local.exists():
            raise FileNotFoundError(
                f"JCM_FIXTURE_STATE_DIR={override} is set but {local} does "
                f"not exist. Generate {member}'s state there first, or unset "
                "the variable to use the published state.")
        return str(local)

    from jcm.data.remote import fetch
    return fetch(state_mirror_path(member))


def _load_member(member: str, init_overrides: dict, days: float):
    """Compose ``member``'s preset for a ``days``-long daily-snapshot run.

    Daily *snapshots* rather than interval averages: ``output_averages=True``
    on hybrid coords trips a shape-broadcast bug in
    ``compute_diagnostic_state_hybrid``, and the mean of daily snapshots is a
    close approximation of the true mean for the slow-varying global
    statistics these bands compare.
    """
    from jcm.configurations import load

    presets = members()
    if member not in presets:
        raise ValueError(
            f"Unknown matrix member {member!r}. Known: {sorted(presets)}")
    overrides = {
        "run.total_time": days,
        "run.save_interval": SAVE_INTERVAL_DAYS,
        "run.output_averages": False,
        **init_overrides,
    }
    return load(presets[member], **overrides)


def _from_state_overrides(file_path: str) -> dict:
    """``init=from_state`` overrides for ``file_path``.

    No ``unstamped_scale``: the states this module writes are stamped, and a
    stamped file needs no unit assertion — offering one is rejected as misuse,
    correctly, since the file already records its own convention.
    """
    return {"init": "from_state", "init.file": file_path}


def _global_mean(predictions):
    """``(time, lon, lat)``-mean of the candidate variables a run produced."""
    ds = predictions.to_xarray()
    means = ds.mean(dim={"time", "lon", "lat"})
    present = [v for v in CANDIDATE_STAT_VARS if v in means]
    return means[present]


def stats_window_global_mean(member: str, state_path: str):
    """Run ``member``'s stats window from ``state_path`` and reduce it.

    The quantity the regression compares, for one run — also the subprocess
    entry point for a reproducibility repeat.
    """
    exp = _load_member(
        member, _from_state_overrides(state_path), STATS_DAYS)
    predictions = exp.model.run(**{**exp.run_kwargs, "forcing": exp.forcing})
    return _global_mean(predictions)


def write_stats_window_global_mean(member: str, state_path: str, out: str):
    """Subprocess entry point for one stats window.

    Writes the *daily* global mean, keeping the time axis, because the run
    that seeds the bands needs the per-day values for ``std`` while the
    reproducibility ensemble needs only their mean. One worker serves both.
    """
    exp = _load_member(
        member, _from_state_overrides(state_path), STATS_DAYS)
    predictions = exp.model.run(**{**exp.run_kwargs, "forcing": exp.forcing})
    ds = predictions.to_xarray()
    present = [v for v in CANDIDATE_STAT_VARS if v in ds]
    ds[present].mean(dim={"lon", "lat"}).to_netcdf(out)


def write_spinup_state(member: str, out_path: str):
    """Subprocess entry point for the spin-up stage."""
    from jcm.checkpoint import save_checkpoint

    exp = _load_member(member, {}, SPIN_UP_DAYS)
    exp.model.run(**{**exp.run_kwargs, "forcing": exp.forcing})
    save_checkpoint(exp.model, out_path, elapsed_days=SPIN_UP_DAYS)


def _run_worker(call: str) -> None:
    """Run one module entry point in a fresh interpreter.

    Every stage that integrates the model goes through here, so the
    orchestrating process never holds a device pool of its own. That is not
    tidiness: JAX does not return pool memory, so a parent that had just run
    a T63 L95 JAM spin-up left too little of an 80 GB card for its own child
    and the generation died in the first repeat. Keeping the parent free of
    device memory also makes the ensemble homogeneous — every member of it is
    produced the same way, rather than one in-process and the rest not.
    """
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c",
         "from jcm.data.test.release_matrix.generate_stats import "
         f"{call}"],
        check=True,
    )


def stats_window_global_mean_isolated(member: str, state_path: str,
                                      tmp_dir) -> "object":
    """One stats window in a fresh interpreter, reduced as the bands are.

    The regression compares ``(time, lon, lat)``-mean values, and this returns
    exactly that — but computed in a child process, for the same reason
    :func:`_run_worker` exists. A single process that walks the whole matrix
    accumulates a device pool JAX never gives back, and the members are not
    equal: T63 L95 with JAM is several times the footprint of T31 L8, so it is
    always the last member that dies of the memory the earlier ones are still
    holding. Running each member's window in its own process makes the test
    independent of how many members precede it, and of their order.
    """
    import xarray as xr

    out = Path(tmp_dir) / f"{member}_window.nc"
    _run_worker(
        "write_stats_window_global_mean as w; "
        f"w({member!r}, {state_path!r}, {str(out)!r})")
    return xr.open_dataset(out).load().mean(dim="time")


def _stats_windows(member, state_path, n_runs, tmp_dir):
    """Run the stats window ``n_runs`` times, each in a fresh process.

    Separate processes because that is the configuration the floor has to
    cover: the regression test runs in its own process, never inside the one
    that wrote the bands, so a repeat taken in-process would be measuring a
    situation that never arises. No claim is made about how much larger a
    cross-process spread is than an in-process one.

    All repeats run the same source tree, so this bounds run-to-run
    reproducibility and says nothing about a code change — which is the
    intent. A code change moving a band is the signal, not the noise. Pin
    ``PYTHONPATH`` when measuring anything like this from a scratch
    directory: ``sys.path[0]`` for ``python /path/script.py`` is the script's
    own directory, so ``jcm`` can silently resolve through an editable
    install to a different checkout.
    """
    import xarray as xr

    runs = []
    for i in range(n_runs):
        out = Path(tmp_dir) / f"window_{i}.nc"
        print(f"  stats window {i + 1}/{n_runs} …", flush=True)
        _run_worker(
            "write_stats_window_global_mean as w; "
            f"w({member!r}, {state_path!r}, {str(out)!r})")
        runs.append(xr.open_dataset(out).load())
    return runs


def _prepare_state(member: str, out_dir: Path) -> tuple[str, str]:
    """Produce ``member``'s fixture init state; return ``(path, provenance)``.

    Spun up for ``SPIN_UP_DAYS`` from the preset's own init and written with
    the current checkpoint schema, so the stats window — and the regression
    test — can resume it with no migration escape hatch.
    """
    out_path = out_dir / Path(state_mirror_path(member)).name
    print(f"  {SPIN_UP_DAYS:g}-day spin-up from the preset's own init …",
          flush=True)
    _run_worker(
        f"write_spinup_state as w; w({member!r}, {str(out_path)!r})")
    return str(out_path), (
        f"{SPIN_UP_DAYS:g}-day spin-up from the preset's own init "
        f"({members()[member]})")


def generate(member: str, out_dir=None, n_reproducibility_repeats=3,
             write_state=True):
    """Generate ``member``'s fixture: its init state and its bands.

    Args:
        member: A ``tools/release_validation/matrix.yaml`` member name.
        out_dir: Where the state file is written for upload. Defaults to the
            current directory; keep it off ``/data`` for the larger grids.
        n_reproducibility_repeats: Stats-window repeats, in their own
            processes, used to size ``<var>.noise``. 0 writes no ``noise``,
            which the regression test then rejects.
        write_state: When False, reuse the state already at ``out_dir`` rather
            than producing one — for re-deriving bands without re-migrating or
            re-spinning.

    Returns:
        ``(state_path, band_path)``.

    """
    import tempfile

    import jax
    import xarray as xr

    print(f"member {member}: backend {jax.default_backend()} "
          f"on {jax.devices()}", flush=True)
    out_dir = Path(out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_state:
        state_path, provenance = _prepare_state(member, out_dir)
    else:
        state_path = str(out_dir / Path(state_mirror_path(member)).name)
        # Still a spin-up state — ``write_state=False`` only ever reuses one
        # this module wrote — so describe it as such rather than as an opaque
        # "reused file", which would leave the fixture unable to say where its
        # own initial condition came from.
        provenance = (
            f"{SPIN_UP_DAYS:g}-day spin-up from the preset's own init "
            f"({members()[member]}); state reused from an earlier generate() "
            "call rather than re-spun")
    print(f"  state: {state_path}", flush=True)

    # One window seeds the bands; the rest size ``noise``. All of them run
    # in their own process, including the first — see :func:`_run_worker`.
    n_runs = 1 + n_reproducibility_repeats
    print(f"  {n_runs} x {STATS_DAYS:g}-day stats window …", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        runs = _stats_windows(member, state_path, n_runs, tmp)

    daily_global = runs[0]
    present = list(daily_global.data_vars)
    pred_mean = daily_global.mean(dim="time")
    pred_std = daily_global.std(dim="time")

    noise = None
    if n_reproducibility_repeats:
        stacked = xr.concat([r.mean(dim="time") for r in runs],
                            dim="_repeat")
        noise = stacked.max(dim="_repeat") - stacked.min(dim="_repeat")

    out = {}
    for var in present:
        out[f"{var}.mean"] = pred_mean[var]
        out[f"{var}.std"] = pred_std[var]
        if noise is not None:
            out[f"{var}.noise"] = noise[var]
    stats_ds = xr.Dataset(out)
    stats_ds.attrs["member"] = member
    stats_ds.attrs["preset"] = members()[member]
    stats_ds.attrs["init_state"] = state_mirror_path(member)
    stats_ds.attrs["init_state_provenance"] = provenance
    stats_ds.attrs["stats_days"] = STATS_DAYS
    if member in PRE_DUST_RETUNE:
        stats_ds.attrs["provisional"] = (
            "PRE-DUST-RETUNE: these bands describe the aerosol climate before "
            "the dust retune (#787/#808) and must be regenerated when it "
            "lands; the equilibrated JAM state is #762's deliverable")
    bands = band_path(member)
    stats_ds.to_netcdf(bands)
    print(f"  wrote {bands} ({bands.stat().st_size} bytes) "
          f"with {len(present)} variables", flush=True)
    return state_path, str(bands)
