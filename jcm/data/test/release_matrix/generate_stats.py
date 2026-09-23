"""Climatology-band fixtures for the supported-configuration matrix.

One band file per member of ``tools/release_validation/matrix.yaml``, built
through that member's *validated preset* (``jcm/config/configuration/*.yaml``,
the same recipes ``tools/benchmark.py`` and the release-validation launcher
use) rather than a composition assembled here. A fixture that hand-rolls its
physics tests something nobody runs; going through the preset means the
regression covers exactly what the project claims to support.

What is where
-------------
* **Bands** — ``<member>_statistics.nc`` in this directory, tens to a few
  hundred KB each and checked in, so a change in what the model produces
  shows up as a reviewable diff.
* **Initial states** — *not* in git. Each member resumes from a stamped
  checkpoint on the Hugging Face data mirror under
  ``bundles/<grid>_<levels>/init_states/``, fetched cache-first by
  :func:`jcm.data.remote.fetch`. They run from a few MB to several GB and
  would otherwise be
  re-committed in full on every regeneration.

Each member's band file and its init state are a matched pair: the bands
describe the window that *follows* that exact state, so regenerating one
without the other compares a trajectory against bands drawn from a different
starting point.

Regenerating
------------
One command per member, on a GPU::

    CUDA_VISIBLE_DEVICES=<idx> python -c "import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'; from jcm.data.test.release_matrix.generate_stats import generate; generate('echam-1m-t63', out_dir='/scr/$USER/fixtures')"

The ``os.environ`` assignment must come *before* the jcm import — merely
importing jcm initialises the JAX CUDA backend (the SPEEDY lookup tables are
built at import time), and the default claims 75 % of the card for this
orchestrating process, which holds no device work of its own but would starve
every worker it spawns. ``generate`` refuses to run without it rather than
OOM-ing an hour in.

Run it in a CI-parity environment — a fresh venv with
``pip install -e ".[mam4]"`` and the pinned CUDA jax — never a shared or
long-lived one: the bands are only valid under the dependencies the test later
runs with, and bands drawn under a different jax-rrtmgp release fail a correct
model across the whole column. Each band file records what it was drawn under
(``bands_environment``), what its init state was spun up under
(``init_state_environment``) and its ``reproducibility_repeats``; see
:func:`generate`.

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

The JAM members' bands describe the aerosol climate on the dust retune
(#787/#808/#840): the ECHAM-like relative-soil-wetness saltation gate and the
``nduscale_reg`` recalibration for jcm's wind climate. They were regenerated
against that code and the rebuilt forcing bundle (which carries the
``soilw_rel`` channel the gate reads), so a failure is a regression, not the
known-provisional state the pre-#840 bands were.
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
#:
#: The list must cover each member's **defining prognostic state**, not just
#: core meteorology: a matrix member exists to validate its distinguishing
#: physics, and bands blind to that physics would pass a regression confined
#: to it — five days of broad meteorological means barely move when an
#: aerosol or number-concentration pathway breaks. Hence the cloud tracers
#: for the 1M/2M members, the full JAM tracer mass state (interstitial,
#: cloud-borne, gases) plus the AOD the radiation sees, and — for the number
#: concentrations, which cannot be banded level by level — their column
#: integrals (see :data:`COLUMN_NUMBER_BURDENS`).
#: Emission / deposition *flux* diagnostics are deliberately not banded:
#: starting from an identical initial state they act directly on the banded
#: tracers, so a flux regression surfaces in the tracer bands within the
#: window without doubling the variable count.
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
    # Cloud condensate tracers (1M/2M/JAM). NUMBER concentrations are
    # deliberately NOT banded — not the hydrometeor numbers (qnc/qni), not
    # the cloud-borne droplet mirror (nc_*), and not the interstitial
    # aerosol mode numbers (n_*): every one of them is set by a thresholdy
    # source (droplet/ice activation, Aitken-mode nucleation), so at levels
    # where the source is intermittent the per-level global mean jumps
    # between runs on identical code. Measured on echam-jam-t63-l47 across
    # up to seven independent windows: level-23 qnc was 2.3e-13 on four
    # generation draws and 33.4 on a fifth; n_ait sat inside its band on six
    # windows and outside it on a seventh; nc_ait reached 3.2x its band. No
    # finite-repeat noise floor can band that per level. The numbers ARE
    # banded as mass-weighted column integrals instead (the ``*_column``
    # entries below, derived by :func:`_with_column_number_burdens`): a
    # near-threshold cell carries a negligible share of the column, so the
    # integral is robust where the level is not. What is banded per level is
    # every MASS:
    # condensate (qc/qi here), interstitial aerosol (m_* below) and
    # cloud-borne aerosol (mc_*), plus the precursor gases (g_*) and AOD.
    # Those are the conserved burdens an emission/chemistry/deposition
    # regression actually moves, and across those same windows they
    # reproduce to <=0.01x of their bands — so the aerosol pathway is
    # covered where a band is meaningful, and not asserted where it is noise.
    "qc",
    "qi",
    # Column-integrated number concentrations [m-2]: the 2M members' droplet
    # and ice numbers, and the JAM total particle number. Derived from the
    # same window (:data:`COLUMN_NUMBER_BURDENS`), not model outputs.
    "qnc_column",
    "qni_column",
    "n_total_column",
    # ECHAM scheme outputs.
    "radiation.toa_lw_up",
    "radiation.surface_sw_down",
    "clouds.cloud_fraction",
    "clouds.precip_rain",
    "convection.precip_conv",
    # MACv2-SP simple plumes: the non-JAM ECHAM members' aerosol pathway.
    # The term publishes under the explicit ``macsp.*`` output namespace
    # (#640), with the CF/AeroCom ``od550aer`` name for the total-column AOD.
    "macsp.od550aer",
    # JAM modal aerosol: the interstitial per-species-per-mode MASSES and
    # the precursor gases (the mode numbers n_* are excluded with the other
    # number concentrations, see the cloud-tracer note above).
    "m_so4_ait", "m_so4_acc", "m_so4_cor",
    "m_bc_acc", "m_bc_cor", "m_bc_pcm",
    "m_du_acc", "m_du_cor",
    "m_ss_ait", "m_ss_acc", "m_ss_cor",
    "m_moa_ait", "m_moa_acc", "m_moa_cor", "m_moa_pcm",
    "m_poa_acc", "m_poa_cor", "m_poa_pcm",
    "m_soa_ait", "m_soa_acc", "m_soa_cor",
    "g_dms", "g_so2", "g_h2so4", "g_soag",
    # — and the cloud-borne phase's MASSES, transported and scavenged in
    # their own right (#602/#708). The cloud-borne numbers (nc_*) are
    # excluded with the other number concentrations (see above).
    "jam_cloud_borne.mc_so4_ait", "jam_cloud_borne.mc_so4_acc",
    "jam_cloud_borne.mc_so4_cor",
    "jam_cloud_borne.mc_bc_acc", "jam_cloud_borne.mc_bc_cor",
    "jam_cloud_borne.mc_bc_pcm",
    "jam_cloud_borne.mc_du_acc", "jam_cloud_borne.mc_du_cor",
    "jam_cloud_borne.mc_ss_ait", "jam_cloud_borne.mc_ss_acc",
    "jam_cloud_borne.mc_ss_cor",
    "jam_cloud_borne.mc_moa_ait", "jam_cloud_borne.mc_moa_acc",
    "jam_cloud_borne.mc_moa_cor", "jam_cloud_borne.mc_moa_pcm",
    "jam_cloud_borne.mc_poa_acc", "jam_cloud_borne.mc_poa_cor",
    "jam_cloud_borne.mc_poa_pcm",
    "jam_cloud_borne.mc_soa_ait", "jam_cloud_borne.mc_soa_acc",
    "jam_cloud_borne.mc_soa_cor",
    # JAM optics: the integrated aerosol state the radiation actually sees
    # (the ``jam_optics.*`` output namespace, #640).
    "jam_optics.aod_550",
    # SPEEDY scheme outputs.
    "longwave_rad.ftop",
    "shortwave_rad.ftop",
    "humidity.rh",
    "condensation.precls",
    "convection.precnv",
)

#: Derived band variables: ``name -> predicate`` selecting the per-level
#: number-concentration outputs [kg-1] whose mass-weighted column integral
#: [m-2] the variable is. Per-level numbers are too threshold-driven to band
#: (see the note in :data:`CANDIDATE_STAT_VARS`); their column integrals are
#: not, because an intermittent activation or nucleation cell holds a
#: negligible share of the column total.
#:
#: ``n_total_column`` sums every JAM mode's number over BOTH phases — the
#: interstitial ``n_<mode>`` tracers and the cloud-borne
#: ``jam_cloud_borne.nc_<mode>`` store. Activation moves particles between
#: those two phases at exactly the thresholds that make the per-level
#: numbers jumpy, so the phase sum is invariant to the process that makes
#: the individual numbers unbandable, while an emission, nucleation,
#: coagulation or deposition regression still moves it.
COLUMN_NUMBER_BURDENS = {
    "qnc_column": lambda name: name == "qnc",
    "qni_column": lambda name: name == "qni",
    "n_total_column": lambda name: (
        name.startswith("n_") or name.startswith("jam_cloud_borne.nc_")),
}


def _with_column_number_burdens(ds):
    """``ds`` plus the :data:`COLUMN_NUMBER_BURDENS` it can supply.

    Each column integral is ``sum_k N_k * dp_k / g`` over the mid-level axis:
    the layer air mass [kg m-2] from the run's own layer pressure thickness,
    taken per column before any horizontal mean, so the band compares the
    global mean of a burden and not a burden of global means. ``dp`` comes
    from :func:`jcm.analysis.layer_pressure_thickness`, which reads the
    ``pressure_thickness`` diagnostic and falls back to differencing
    ``pressure_half``.

    It is deliberately NOT ``air_density * layer_thickness``: the
    ``layer_thickness`` diagnostic is floored at 10 m for the physics that
    divides by it (see :mod:`jcm.physics.diagnostics.moist_air_state`), so
    wherever the floor binds, that product overstates the layer mass. No path
    falls back to it. A member that carries number tracers but no pressure
    thickness raises instead, because a burden it cannot weigh correctly must
    not be banded.

    A variable whose sources the member does not carry (no 2M scheme, no JAM)
    is simply not added, and so is not banded for that member.
    """
    import jcm.constants as c
    from jcm.analysis import layer_pressure_thickness

    wanted = {}
    for out_name, selects in COLUMN_NUMBER_BURDENS.items():
        sources = [v for v in ds.data_vars
                   if selects(v) and "level" in ds[v].dims]
        if sources:
            wanted[out_name] = sources
    if not wanted:
        return ds
    if "pressure_thickness" not in ds and "pressure_half" not in ds:
        raise ValueError(
            "column number burdens need the layer pressure thickness "
            "(pressure_thickness or pressure_half) to weight "
            f"{sorted(v for s in wanted.values() for v in s)}; the run "
            "carries neither, and the floored layer_thickness is not a mass "
            "weight")
    air_mass = layer_pressure_thickness(ds) / c.grav            # kg m-2
    derived = {}
    for out_name, sources in wanted.items():
        number = sum(ds[v] for v in sources)
        derived[out_name] = (number * air_mass).sum("level", skipna=False)
        derived[out_name].attrs = {
            "units": "m-2",
            "long_name": "column-integrated number: "
                         + " + ".join(sorted(sources)),
        }
    return ds.assign(derived)


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

#: Members whose bands describe an aerosol climate a not-yet-landed dust retune
#: would move, and which therefore carry a ``provisional`` marker until it does.
#: Empty since the retune (#787/#808/#840) landed and the JAM bands were
#: regenerated against it; kept as the hook a future in-flight retune uses.
PRE_DUST_RETUNE: tuple[str, ...] = ()

#: Members whose init state is deliberately NOT published yet, and why.
#:
#: A fixture in this state is a real gap, not a passing test: the regression
#: skips the member and says so. It is recorded in the band file rather than
#: only here so the *fixture* carries its own status — a checkout with an
#: older band file skips for the reason that band file was written with,
#: instead of inheriting a judgement from whatever this module says today.
#:
#: Publishing is not a silent transition either way. Until a member's state
#: is on the mirror, hosting a fixture that fetches it would turn the whole
#: GPU gate red for as long as it takes; once published, the entry is removed
#: and a 404 for that member goes back to being a hard failure.
#: Empty: every matrix member's init state is published on the mirror. The JAM
#: pair was held pre-#840 because the dust retune (#787/#808/#840) was about to
#: move their aerosol climate; it landed and they were regenerated and published
#: against it, so their 404-is-a-hard-failure contract is back in force like
#: every other member's. Kept as the declared-gap hook for a future member.
HELD_STATES: dict[str, str] = {}

SPIN_UP_DAYS = 5.0
STATS_DAYS = 5.0
SAVE_INTERVAL_DAYS = 1.0

#: Stats-window repeats used to size ``<var>.noise``, per member; members not
#: listed use :data:`DEFAULT_REPRODUCIBILITY_REPEATS`. The JAM members carry
#: more because their band set is the largest and the most intermittent
#: (cloud cover and cloud-borne aerosol respond to activation thresholds), so
#: a four-window peak-to-peak spread under-samples their run-to-run
#: reproducibility: an echam-jam-t63-l47 ``clouds.cloud_fraction`` level fell
#: outside a band drawn from four windows in an independent run of the same
#: code and state that a re-run then passed. Both JAM members use the same
#: count so they are banded alike.
DEFAULT_REPRODUCIBILITY_REPEATS = 3
REPRODUCIBILITY_REPEATS = {
    "echam-jam-t63-l47": 6,
    "echam-jam-t63-l95": 6,
}

#: Distributions whose versions a band file records (see
#: :func:`generation_environment`): the ones that change what a member
#: produces. jax-rrtmgp is the reason this exists — a band drawn under one
#: jax-rrtmgp release and checked under another fails with a whole-column
#: radiative shift, which is indistinguishable from a physics regression
#: unless the file says what it was drawn under.
_ENV_DISTRIBUTIONS = ("jax", "jaxlib", "jax-rrtmgp", "dinosaur", "flax",
                      "mam4-jax", "numpy", "xarray")


def generation_environment() -> str:
    """``name==version`` of every :data:`_ENV_DISTRIBUTIONS` entry, and Python.

    Read from installed-distribution metadata rather than by importing, so
    it costs nothing and cannot initialise a device. A distribution that is
    not installed is recorded as such (``mam4-jax`` on a core install).
    """
    import platform
    from importlib import metadata

    parts = [f"python=={platform.python_version()}"]
    for dist in _ENV_DISTRIBUTIONS:
        try:
            parts.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{dist}: not installed")
    return "; ".join(parts)


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


def state_digest(path) -> str:
    """Short content digest used to version a published state file."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def state_mirror_path(member: str, digest: str) -> str:
    """Mirror path of ``member``'s init state, as ``fetch`` takes it.

    The content digest is in the **filename**, not merely recorded beside it,
    because a stable name cannot be republished safely:
    :func:`jcm.data.remote.fetch` resolves cache-first and never revalidates a
    hit, so any host that had already fetched the old state would keep using
    it against newly committed bands — a mismatched pair failing for as long
    as that cache survived, with nothing to indicate why. A new state is a new
    path, so a stale cache entry simply goes unused.

    The band file records the exact path it was generated against; nothing
    reconstructs this name in order to read a state back.
    """
    return (f"bundles/{MEMBER_BUNDLE[member]}/init_states/"
            f"{member}_fixture_{digest}.msgpack")


def _assert_state_digest(path, mirror_path: str) -> None:
    """Assert the state at ``path`` is the one ``mirror_path`` names.

    The mirror filename carries a content digest (see
    :func:`state_mirror_path`), so re-hashing the resolved file and comparing
    turns the state<->bands pairing from a naming *convention* into a checked
    invariant. Combined with the digest being *in* the path — a new state is a
    new filename, which ``fetch``'s cache-first resolution can never confuse
    with an old one — this is what version-locks a fixture's bands to the exact
    state they were generated against: a mismatched or corrupted state fails
    loudly against the bands that expect it rather than being validated as if
    it matched.
    """
    expected = Path(mirror_path).stem.rsplit("_", 1)[-1]
    actual = state_digest(path)
    if actual != expected:
        raise ValueError(
            f"state file {path} hashes to {actual!r} but the band file asks "
            f"for {expected!r} ({mirror_path}): a mismatched state/bands pair. "
            "Regenerate them together with "
            "jcm.data.test.release_matrix.generate_stats.generate().")


def resolve_state(mirror_path: str) -> str | None:
    """Local path to the state at ``mirror_path``, fetching it if necessary.

    ``mirror_path`` comes from the band file's own ``init_state`` attribute,
    digest and all, and the resolved file's content is verified against that
    digest (:func:`_assert_state_digest`), so a fixture is always read against
    the exact state it was generated against.

    Normally the mirror, resolved cache-first: a warm cache needs no network,
    and a cold one on an internet-less node raises ``fetch``'s message naming
    the prefetch command rather than a bare error.

    ``JCM_FIXTURE_STATE_DIR`` overrides that with a directory of locally
    generated states, because regenerating a fixture and publishing it are
    necessarily two steps and the pair has to be checkable in between.
    Generating states one member at a time is the documented workflow, so a
    member absent from the override directory returns ``None`` — the caller
    skips it, naming it — rather than raising and making one freshly generated
    member unverifiable until all seven had been gathered. Nothing falls back
    to the mirror while the override is set: mixing a published state into a
    run meant to validate local ones is how the wrong pair gets blessed.
    """
    import os

    override = os.environ.get("JCM_FIXTURE_STATE_DIR")
    if override:
        local = Path(override) / Path(mirror_path).name
        if not local.exists():
            return None
        _assert_state_digest(local, mirror_path)
        return str(local)

    from jcm.data.remote import fetch
    local = fetch(mirror_path)
    _assert_state_digest(local, mirror_path)
    return local


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


def _area_mean(predictions):
    """Area-weighted global mean, per output time, of the band variables.

    The ONE horizontal reduction behind every band: the stats-window worker
    applies it for both generation and the regression's own check, so the two
    cannot drift apart.

    The weights are the grid's own Gauss-Legendre quadrature weights
    (:func:`jcm.analysis.global_mean` via :func:`jcm.analysis.area_weights`,
    which falls back to ``cos(lat)`` only off a Gaussian grid). An arithmetic
    ``lat`` mean would give every latitude ring equal weight, over-counting
    the small polar rings by up to ~1/cos(lat), so the bands would describe
    a polar-weighted statistic rather than the global mean or global-mean
    column burden they are named for.

    ``skipna=False`` throughout: one non-finite cell must make the mean
    non-finite, not be averaged away (see :func:`generate`).
    """
    import xarray as xr

    from jcm.analysis import area_weights, global_mean

    ds = _with_column_number_burdens(predictions.to_xarray())
    present = [v for v in CANDIDATE_STAT_VARS if v in ds]
    weights = area_weights(ds)
    return xr.Dataset({v: global_mean(ds[v], weights, skipna=False)
                       for v in present})


def _global_mean(predictions):
    """``(time, area)``-mean of the candidate variables a run produced."""
    return _area_mean(predictions).mean(dim="time", skipna=False)


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
    _area_mean(predictions).to_netcdf(out)


def report_backend() -> None:
    """Subprocess entry point: print the accelerator the workers will use.

    Run in a child so :func:`generate` itself never imports JAX and so never
    holds a device context while the workers run — see :func:`_run_worker` for
    why the orchestrating process must stay free of device memory.
    """
    import jax

    print(f"  backend {jax.default_backend()} on {jax.devices()}", flush=True)


def write_spinup_state(member: str, out_path: str):
    """Subprocess entry point for the spin-up stage."""
    from jcm.checkpoint import save_checkpoint

    exp = _load_member(member, {}, SPIN_UP_DAYS)
    exp.model.run(**{**exp.run_kwargs, "forcing": exp.forcing})
    save_checkpoint(exp.model, out_path, elapsed_days=SPIN_UP_DAYS)


def _run_worker(call: str, env=None) -> None:
    """Run one module entry point in a fresh interpreter.

    Every stage that integrates the model goes through here, so the
    orchestrating process never holds a device pool of its own. That is not
    tidiness: JAX does not return pool memory, so a parent that had just run
    a T63 L95 JAM spin-up left too little of an 80 GB card for its own child
    and the generation died in the first repeat. Keeping the parent free of
    device memory also makes the ensemble homogeneous — every member of it is
    produced the same way, rather than one in-process and the rest not.

    The child always gets ``XLA_PYTHON_CLIENT_PREALLOCATE=false``: the default
    claims 75 % of the card up front, which on a shared box is antisocial and,
    when the parent has done the same, leaves the child a quarter of a card to
    run a model in.

    ``env`` is the environment handed to the child, defaulting to this
    process's. A caller that has pinned *itself* to CPU — as the regression
    test does, needing no device of its own — passes the environment it
    captured beforehand, so the child still reaches the accelerator.

    Raises:
        RuntimeError: the worker exited non-zero, carrying the tail of its
            stderr. Without that the failure surfaces as a bare
            ``CalledProcessError`` naming only the command, and the reason —
            an out-of-memory, a missing input — dies with the child.

    """
    import os
    import subprocess
    import sys

    child_env = dict(os.environ if env is None else env)
    child_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    result = subprocess.run(
        [sys.executable, "-c",
         "from jcm.data.test.release_matrix.generate_stats import "
         f"{call}"],
        env=child_env, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        tail = "\n".join((result.stderr or "").strip().splitlines()[-25:])
        raise RuntimeError(
            f"fixture worker exited {result.returncode} running {call!r}\n"
            f"--- last 25 lines of its stderr ---\n{tail}")


def stats_window_global_mean_isolated(member: str, state_path: str,
                                      tmp_dir, env=None) -> "object":
    """One stats window in a fresh interpreter, reduced as the bands are.

    The regression compares time-mean, area-weighted global-mean values,
    and this returns exactly that — but computed in a child process, for the same reason
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
        f"w({member!r}, {state_path!r}, {str(out)!r})", env=env)
    return xr.open_dataset(out).load().mean(dim="time", skipna=False)


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
    print(f"  {SPIN_UP_DAYS:g}-day spin-up from the preset's own init …",
          flush=True)
    tmp_path = out_dir / f"{member}_fixture.partial"
    _run_worker(
        f"write_spinup_state as w; w({member!r}, {str(tmp_path)!r})")
    out_path = out_dir / Path(
        state_mirror_path(member, state_digest(tmp_path))).name
    tmp_path.replace(out_path)
    return str(out_path), (
        f"{SPIN_UP_DAYS:g}-day spin-up from the preset's own init "
        f"({members()[member]})")


def generate(member: str, out_dir=None, n_reproducibility_repeats=None,
             write_state=True, state_environment=None):
    """Generate ``member``'s fixture: its init state and its bands.

    Args:
        member: A ``tools/release_validation/matrix.yaml`` member name.
        out_dir: Where the state file is written for upload. Defaults to the
            current directory; keep it off ``/data`` for the larger grids.
        n_reproducibility_repeats: Stats-window repeats, in their own
            processes, used to size ``<var>.noise``. ``None`` takes the
            member's entry in :data:`REPRODUCIBILITY_REPEATS`. 0 writes no
            ``noise``, which the regression test then rejects.
        write_state: When False, reuse the state already at ``out_dir`` rather
            than producing one — for re-deriving bands without re-migrating or
            re-spinning.
        state_environment: With ``write_state=False``, the environment the
            reused state was spun up under (as :func:`generation_environment`
            would describe it), recorded in the band file. The state carries
            no record of its own, and bands may legitimately be drawn under
            different dependencies from the ones that spun up their initial
            condition, so the file has to say both. Ignored when
            ``write_state`` is True: the state is then spun up here, under
            the same environment as the bands.

    Returns:
        ``(state_path, band_path)``.

    """
    import os
    import tempfile

    import numpy as np
    import xarray as xr

    # The orchestrator must never hold a device pool (see :func:`_run_worker`)
    # — but it cannot avoid initialising JAX, because merely importing jcm
    # builds device lookup tables (measured: ``import jcm`` alone claims
    # 61,214 MiB of an 80 GB A100 under JAX's default 75 % preallocation, in
    # a process that then does no device work at all; #859 tracks making the
    # library import lazy). The pool is grabbed at *import* time, before this
    # function can do anything about it, so the environment variable must be
    # set before the jcm import — and rather than hope, refuse to run
    # without it: with preallocation on, the orchestrator's pool starves the
    # workers and the largest member OOMs an hour in, which is a far worse
    # failure than this one. No code here can set it retroactively; only the
    # invocation can.
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
        raise RuntimeError(
            "generate() must run with XLA_PYTHON_CLIENT_PREALLOCATE=false "
            "set BEFORE jcm is imported: importing jcm initialises the JAX "
            "CUDA backend, whose default preallocates 75% of the card to "
            "this orchestrating process and starves the worker subprocesses "
            "that integrate the model. Invoke as:\n"
            "  python -c \"import os; "
            "os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'; "
            "from jcm.data.test.release_matrix.generate_stats import "
            f"generate; generate({member!r}, ...)\"")
    # The backend confirmation still runs in its own throwaway child: the
    # parent's on-demand footprint stays a few hundred MB and, more
    # importantly, every process that integrates the model reports the same
    # way (see :func:`_run_worker`).
    print(f"member {member}:", flush=True)
    _run_worker("report_backend as w; w()")
    out_dir = Path(out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_state:
        state_path, provenance = _prepare_state(member, out_dir)
    else:
        # Reuse the one state this module has already written for the
        # member. The name carries a content digest, so glob rather than
        # reconstruct it, and refuse an ambiguous directory instead of
        # picking one arbitrarily — the bands would then describe a state
        # nobody could identify afterwards.
        found = sorted(out_dir.glob(f"{member}_fixture_*.msgpack"))
        if len(found) != 1:
            raise FileNotFoundError(
                f"write_state=False needs exactly one "
                f"{member}_fixture_*.msgpack in {out_dir}; found "
                f"{[f.name for f in found]}")
        state_path = str(found[0])
        # Still a spin-up state — ``write_state=False`` only ever reuses one
        # this module wrote — so describe it as such rather than as an opaque
        # "reused file", which would leave the fixture unable to say where its
        # own initial condition came from.
        provenance = (
            f"{SPIN_UP_DAYS:g}-day spin-up from the preset's own init "
            f"({members()[member]}); state reused from an earlier generate() "
            "call rather than re-spun")
    # The band file names its state by the digest in the state's filename,
    # and the regression later checks the fetched state against that digest.
    # So the digest recorded here must be the file's actual content hash,
    # not merely what its name claims: a reused state renamed, copied over or
    # truncated under a matching name would otherwise yield bands that point
    # at a state they were never drawn from, and that mismatch would only
    # surface when someone published the pair. Re-hash before any stats
    # window runs, and record only the verified digest.
    claimed = Path(state_path).stem.rsplit("_", 1)[-1]
    digest = state_digest(state_path)
    if digest != claimed:
        raise ValueError(
            f"{member}: state {state_path} hashes to {digest!r} but its "
            f"filename claims {claimed!r}; refusing to draw bands against a "
            "state whose name does not identify its contents. Regenerate the "
            "state (write_state=True) or restore the file that name belongs "
            "to.")
    print(f"  state: {state_path} (digest {digest} verified)", flush=True)

    # One window seeds the bands; the rest size ``noise``. All of them run
    # in their own process, including the first — see :func:`_run_worker`.
    if n_reproducibility_repeats is None:
        n_reproducibility_repeats = REPRODUCIBILITY_REPEATS.get(
            member, DEFAULT_REPRODUCIBILITY_REPEATS)
    n_runs = 1 + n_reproducibility_repeats
    print(f"  {n_runs} x {STATS_DAYS:g}-day stats window …", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        runs = _stats_windows(member, state_path, n_runs, tmp)

    daily_global = runs[0]
    present = list(daily_global.data_vars)
    pred_mean = daily_global.mean(dim="time", skipna=False)
    pred_std = daily_global.std(dim="time", skipna=False)

    noise = None
    if n_reproducibility_repeats:
        stacked = xr.concat([r.mean(dim="time", skipna=False) for r in runs],
                            dim="_repeat")
        noise = (stacked.max(dim="_repeat", skipna=False)
                 - stacked.min(dim="_repeat", skipna=False))

    # Every reduction above propagates NaN (``skipna=False``): xarray's
    # default would average a partially-NaN window into a finite number, so a
    # run that had blown up in some cells could be written as a band. Refuse
    # to write one instead.
    nonfinite = sorted(
        v for v in present
        if not np.isfinite(pred_mean[v].values).all()
        or not np.isfinite(pred_std[v].values).all()
        or (noise is not None and not np.isfinite(noise[v].values).all()))
    if nonfinite:
        raise ValueError(
            f"{member}: non-finite values in the stats window for "
            f"{nonfinite}; refusing to write bands from a run that is not "
            "finite everywhere")

    out = {}
    for var in present:
        out[f"{var}.mean"] = pred_mean[var]
        out[f"{var}.std"] = pred_std[var]
        if noise is not None:
            out[f"{var}.noise"] = noise[var]
    stats_ds = xr.Dataset(out)
    stats_ds.attrs["member"] = member
    stats_ds.attrs["preset"] = members()[member]
    # The exact path this fixture was generated against, digest and all, so
    # the regression reads back the state these bands describe rather than
    # whatever currently sits at a reconstructed name.
    stats_ds.attrs["init_state"] = state_mirror_path(member, digest)
    stats_ds.attrs["init_state_provenance"] = provenance
    stats_ds.attrs["stats_days"] = STATS_DAYS
    stats_ds.attrs["reproducibility_repeats"] = n_reproducibility_repeats
    # What the bands were drawn under, and what their initial state was spun
    # up under — separately, because a reused state can predate the current
    # dependencies. A band that fails with a whole-column shift is first
    # checked against this before being read as a physics regression.
    band_env = generation_environment()
    stats_ds.attrs["bands_environment"] = band_env
    stats_ds.attrs["init_state_environment"] = (
        band_env if write_state else (state_environment or "unrecorded"))
    held = HELD_STATES.get(member)
    stats_ds.attrs["hosted_state"] = "pending" if held else "published"
    if held:
        stats_ds.attrs["hosted_state_reason"] = held
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
