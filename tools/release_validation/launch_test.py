"""Tests for the release-validation launcher.

The guarded property is provenance: every member of a matrix launch must
integrate a *fresh* year in a directory named after the commit under test. A
fixed rundir let a second launch silently resume the first one's checkpoint
and report a healthy year for an integration that never ran (#701).

Nothing here submits anything -- ``qsub`` is monkeypatched out, and the
Kubernetes path never reaches a cluster or GitHub (see its section below).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import launch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
MEMBER = "speedy-t31"          # the cheapest member: no JAM aux-input lookup


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keep the input preflight off the network in every test.

    The preflight's own behaviour is tested below by re-patching
    ``_preset_data_files`` with a known input list.
    """
    monkeypatch.setattr(launch, "_hf_fetch", lambda rel: rel)
    monkeypatch.setattr(launch, "_preset_data_files", lambda ovs: [])
    # main() exports each member's commit into its own environment; setenv
    # first so monkeypatch restores the variable afterwards.
    monkeypatch.setenv("JCM_MIRROR_REVISION", "0" * 40)
    monkeypatch.delenv("JCM_MIRROR_REVISION")


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Return a throwaway SCRATCH, so rundirs never touch the real one."""
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    return tmp_path / "scratch"


@pytest.fixture
def repo(tmp_path):
    """Return a stand-in repo to write ``runs/*.pbs`` into."""
    (tmp_path / "repo").mkdir()
    return tmp_path / "repo"


def _launch(repo, *args):
    return launch.main(["--repo", str(repo), "--members", MEMBER, *args])


def test_repo_tag_is_the_head_short_sha():
    """The default tag names the commit the outputs will claim provenance for."""
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert launch.repo_tag(REPO) == sha


def test_repo_tag_falls_back_outside_a_checkout(tmp_path):
    """A non-git tree still gets a tag that separates one day's launch."""
    tag = launch.repo_tag(tmp_path)
    assert tag.startswith("nogit_")
    assert tag.replace("_", "").isalnum()


def test_default_tag_namespaces_the_rundir(scratch, repo):
    """With no --tag, the rundir carries the launched repo's own HEAD SHA."""
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git[:3], "init", "-q"], check=True)
    (repo / "f").write_text("x")
    subprocess.run([*git, "add", "f"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    sha = subprocess.run([*git[:3], "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         check=True).stdout.strip()

    _launch(repo)
    job = (repo / "runs" / f"mx_speedy_t31_{sha}.pbs").read_text()
    assert f"{scratch}/jam_runs/mx_speedy_t31_{sha}" in job


def test_explicit_tag_gives_each_launch_its_own_rundir(scratch, repo):
    """Two tags of the same member write two jobs and two checkpoint paths."""
    _launch(repo, "--tag", "aaa1111")
    _launch(repo, "--tag", "bbb2222")
    ckpts = set()
    for tag in ("aaa1111", "bbb2222"):
        job = (repo / "runs" / f"mx_speedy_t31_{tag}.pbs").read_text()
        rundir = f"{scratch}/jam_runs/mx_speedy_t31_{tag}"
        assert f"run.checkpoint_path={rundir}/checkpoint.msgpack" in job
        assert f"run.output_prefix={rundir}/mx_speedy_t31_{tag}" in job
        assert f"hydra.run.dir={rundir}" in job
        ckpts.add(rundir)
    assert len(ckpts) == 2


@pytest.mark.parametrize("tag", ["feature/foo", "x y", "a;echo hi", ""])
def test_unsafe_explicit_tag_is_refused(scratch, repo, tag):
    """A tag that is not both a path segment and a job name is refused.

    Unchecked, ``feature/foo`` wrote into a directory that does not exist and
    ``x y`` produced a malformed ``#PBS -N`` directive.
    """
    with pytest.raises(SystemExit) as e:
        _launch(repo, "--tag", tag)
    assert "--tag" in str(e.value)
    assert not list((repo / "runs").glob("*.pbs"))


def _seed_checkpoint(scratch, tag):
    rundir = scratch / "jam_runs" / f"mx_speedy_t31_{tag}"
    rundir.mkdir(parents=True)
    (rundir / "checkpoint.msgpack").write_bytes(b"stale")
    return rundir


def test_existing_checkpoint_is_refused(scratch, repo):
    """Relaunching into a populated rundir would resume the previous run."""
    rundir = _seed_checkpoint(scratch, "aaa1111")
    with pytest.raises(SystemExit) as e:
        _launch(repo, "--tag", "aaa1111")
    msg = str(e.value)
    assert str(rundir / "checkpoint.msgpack") in msg
    assert "--resume" in msg and "--tag" in msg
    # Refused before writing anything, so nothing can be submitted by hand.
    assert not (repo / "runs" / "mx_speedy_t31_aaa1111.pbs").exists()


def test_resume_bypasses_the_refusal(scratch, repo):
    """--resume is the explicit way to continue an interrupted member."""
    _seed_checkpoint(scratch, "aaa1111")
    _launch(repo, "--tag", "aaa1111", "--resume")
    assert (repo / "runs" / "mx_speedy_t31_aaa1111.pbs").exists()


def test_a_dirty_member_blocks_the_whole_matrix(scratch, repo, monkeypatch):
    """One populated rundir fails the launch whole, not half-submitted."""
    _seed_checkpoint(scratch, "aaa1111")
    calls = []
    monkeypatch.setattr(launch.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd))
    with pytest.raises(SystemExit):
        launch.main(["--repo", str(repo), "--tag", "aaa1111", "--submit"])
    assert calls == []
    assert not list((repo / "runs").glob("*.pbs"))


def test_submit_qsubs_the_written_job(scratch, repo, monkeypatch):
    """--submit hands the generated script to qsub, unchanged."""
    calls = []
    monkeypatch.setattr(launch.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd))
    _launch(repo, "--tag", "aaa1111", "--submit")
    path = repo / "runs" / "mx_speedy_t31_aaa1111.pbs"
    assert calls == [["qsub", str(path)]]


def test_jam_members_archive_a_pre_onset_checkpoint(monkeypatch):
    """JAM members must keep permanent archives, not only the rotating pair.

    A JAM aerosol runaway develops over weeks, so by the time it is visible
    both ``checkpoint.msgpack`` and its ``.prev`` have been written from
    poisoned state and there is nothing left to restart from before the onset.

    ``jam_aux`` is stubbed out: it globs ``$JAM_INPUTS`` and exits when the
    staged oxidant/emission files are absent, which is every machine but a
    prepared one. The override under test does not come from it.
    """
    monkeypatch.setattr(launch, "jam_aux",
                        lambda grid, levels, fetch_here=True: [])
    cfg = yaml.safe_load((pathlib.Path(launch.HERE) / "matrix.yaml").read_text())
    for name, member in cfg["members"].items():
        ovs = launch.overrides(name, member, cfg["defaults"], "/tmp/rundir")
        setting = [o for o in ovs if o.startswith("run.archive_ckpt_every=")]
        assert len(setting) == 1, name
        expected = 30 if "jam" in name else 0
        assert setting[0] == f"run.archive_ckpt_every={expected}", name


def test_archive_setting_is_a_real_run_key():
    """Guards against a silently-ignored Hydra override."""
    default = yaml.safe_load(
        (REPO / "jcm" / "config" / "run" / "default.yaml").read_text())
    assert "archive_ckpt_every" in default


def test_prefetch_downloads_every_hf_input(monkeypatch):
    """Every ``hf://`` input a member resolves to is pulled before submitting.

    A matrix member is a ten-hour GPU job; an input that cannot be resolved
    must fail on the submitting node, which has network, rather than inside
    model construction on a compute node that does not (#774).
    """
    fetched = []
    monkeypatch.setattr(launch, "_hf_fetch", lambda rel: fetched.append(rel))
    monkeypatch.setattr(
        launch, "_preset_data_files",
        lambda ovs: ["hf://bundles/t63/terrain.nc",
                     "hf://bundles/t63_l47/ozone_pd.nc"])
    assert launch.prefetch(["physics=echam"]) == []
    assert fetched == ["bundles/t63/terrain.nc", "bundles/t63_l47/ozone_pd.nc"]


def test_prefetch_reports_an_unavailable_input(monkeypatch):
    def boom(rel):
        raise FileNotFoundError(f"hf://{rel} is not in the local cache")

    monkeypatch.setattr(launch, "_hf_fetch", boom)
    monkeypatch.setattr(launch, "_preset_data_files",
                        lambda ovs: ["hf://bundles/t63_l47/ozone_pd.nc"])
    missing = launch.prefetch([])
    assert len(missing) == 1 and "ozone_pd.nc" in missing[0]


def test_prefetch_reports_a_missing_local_input(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "_preset_data_files",
                        lambda ovs: [str(tmp_path / "nope.nc")])
    assert launch.prefetch([]) == [str(tmp_path / "nope.nc")]


def test_launch_refuses_when_an_input_is_unavailable(scratch, repo, monkeypatch):
    """No job file is written when the preflight fails."""
    monkeypatch.setattr(launch, "prefetch",
                        lambda ovs: ["hf://bundles/t63_l47/ozone_pd.nc"])
    with pytest.raises(SystemExit) as exc:
        _launch(repo, "--tag", "prefetchfail")
    assert "ozone_pd.nc" in str(exc.value)
    assert not list((repo / "runs").glob("*.pbs"))


def test_no_prefetch_generates_jobs_offline(scratch, repo, monkeypatch):
    called = []
    monkeypatch.setattr(launch, "prefetch",
                        lambda ovs: called.append(ovs) or [])
    _launch(repo, "--tag", "offlinegen", "--no-prefetch")
    assert not called
    assert list((repo / "runs").glob("*.pbs"))


def _job_env(repo, tag):
    """Return the ``export``s of a generated PBS script."""
    text = (repo / "runs" / f"mx_speedy_t31_{tag}.pbs").read_text()
    return dict(re.findall(r"^export (\w+)=(\S*)$", text, re.M))


def _checkpoint(scratch, tag):
    (scratch / "jam_runs" / f"mx_speedy_t31_{tag}"
     / "checkpoint.msgpack").write_bytes(b"x")


def test_job_exports_the_pinned_commit(scratch, repo):
    import json

    from jcm.data import remote
    _launch(repo, "--tag", "p")
    env = _job_env(repo, "p")
    assert env["JCM_MIRROR_REVISION"] == remote.MIRROR_REVISION
    assert "JCM_ALLOW_MIRROR_REVISION_CHANGE" not in env
    record = scratch / "jam_runs" / "mx_speedy_t31_p" / launch.MIRROR_RECORD
    assert json.loads(record.read_text())["source"] == "pinned"


def test_resume_keeps_the_recorded_commit_when_the_pin_moves(
        scratch, repo, monkeypatch):
    from jcm.data import remote
    first = remote.MIRROR_REVISION
    _launch(repo, "--tag", "rp")
    _checkpoint(scratch, "rp")
    monkeypatch.setattr(remote, "MIRROR_REVISION", "d" * 40)
    monkeypatch.delenv("JCM_MIRROR_REVISION")
    _launch(repo, "--tag", "rp", "--resume")
    assert _job_env(repo, "rp")["JCM_MIRROR_REVISION"] == first


def test_resume_refuses_a_different_explicit_sha_unless_forced(
        scratch, repo, monkeypatch):
    import json
    a, c = "a" * 40, "c" * 40
    monkeypatch.setenv("JCM_MIRROR_REVISION", a)
    _launch(repo, "--tag", "rx")
    _checkpoint(scratch, "rx")
    monkeypatch.setenv("JCM_MIRROR_REVISION", c)
    with pytest.raises(SystemExit, match="force-mirror-revision"):
        _launch(repo, "--tag", "rx", "--resume")
    assert _job_env(repo, "rx")["JCM_MIRROR_REVISION"] == a

    monkeypatch.setenv("JCM_MIRROR_REVISION", c)
    _launch(repo, "--tag", "rx", "--resume", "--force-mirror-revision")
    env = _job_env(repo, "rx")
    assert env["JCM_MIRROR_REVISION"] == c
    assert env["JCM_ALLOW_MIRROR_REVISION_CHANGE"] == "1"
    record = scratch / "jam_runs" / "mx_speedy_t31_rx" / launch.MIRROR_RECORD
    assert json.loads(record.read_text())["commit"] == c


def test_failed_preflight_leaves_the_record_untouched(scratch, repo,
                                                      monkeypatch):
    record = scratch / "jam_runs" / "mx_speedy_t31_fp" / launch.MIRROR_RECORD
    monkeypatch.setenv("JCM_MIRROR_REVISION", "a" * 40)
    _launch(repo, "--tag", "fp")
    _checkpoint(scratch, "fp")
    before = record.read_text()
    monkeypatch.setenv("JCM_MIRROR_REVISION", "c" * 40)
    monkeypatch.setattr(launch, "prefetch", lambda ovs: ["hf://missing.nc"])
    with pytest.raises(SystemExit, match="not available"):
        _launch(repo, "--tag", "fp", "--resume", "--force-mirror-revision")
    assert record.read_text() == before


def test_members_at_different_commits_are_launched_separately(
        scratch, repo, monkeypatch):
    monkeypatch.setenv("JCM_MIRROR_REVISION", "a" * 40)
    launch.main(["--repo", str(repo), "--tag", "mx", "--members", "speedy-t31"])
    monkeypatch.setenv("JCM_MIRROR_REVISION", "c" * 40)
    launch.main(["--repo", str(repo), "--tag", "mx",
                 "--members", "echam-1m-t63"])
    monkeypatch.delenv("JCM_MIRROR_REVISION")       # each reuses its record
    with pytest.raises(SystemExit, match="launch them separately"):
        launch.main(["--repo", str(repo), "--tag", "mx", "--resume",
                     "--members", "speedy-t31,echam-1m-t63"])



# ---------------------------------------------------------------------------
# --site nautilus: Kubernetes Jobs from the production-run engine (mkrun.py).
# Nothing here reaches a cluster or GitHub: ``remote_refs`` (git ls-remote)
# and ``_kubectl`` are replaced, and the fetch test runs against a stand-in
# ``kubectl`` executable whose "volume" is a temporary directory.


DIGEST = "sha256:" + "d" * 64
DEFAULTS = yaml.safe_load(
    (pathlib.Path(launch.HERE) / "matrix.yaml").read_text())["defaults"]


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, name):
    (repo / name).write_text(name)
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", name)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def gitrepo(tmp_path):
    """Return a checkout with one commit, which the fake remote carries."""
    repo = tmp_path / "gitrepo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


@pytest.fixture
def remote(monkeypatch):
    """Return the branches/tags a fake GitHub remote carries (edit to taste)."""
    refs = {}
    monkeypatch.setattr(launch, "remote_refs", lambda url: dict(refs))
    monkeypatch.setattr(launch, "image_ref",
                        lambda image: image.rsplit(":", 1)[0] + "@" + DIGEST)
    return refs


@pytest.fixture
def cluster(monkeypatch):
    """Return a fake cluster: ``jobs`` maps Job name to status; calls recorded."""
    state = {"jobs": {}, "calls": []}

    def kubectl(site, *args, input=None, timeout=300):
        state["calls"].append((args, input))
        verb = args[0]
        if verb == "get":
            status = state["jobs"].get(args[2])
            if status is None:
                return subprocess.CompletedProcess(
                    args, 1, "", f'jobs.batch "{args[2]}" NotFound')
            conds = {"complete": [{"type": "Complete", "status": "True"}],
                     "failed": [{"type": "Failed", "status": "True"}],
                     "active": []}[status]
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"status": {"conditions": conds}}), "")
        if verb == "delete":
            state["jobs"].pop(args[2], None)
        if verb == "apply":
            state["jobs"][json.loads(input)["metadata"]["name"]] = "active"
        return subprocess.CompletedProcess(args, 0, f"{verb} ok", "")

    monkeypatch.setattr(launch, "_kubectl", kubectl)
    return state


def _k8s(repo, capsys, *args, members=MEMBER):
    """Run the Nautilus door; return the printed manifests."""
    capsys.readouterr()
    launch.main(["--site", "nautilus", "--repo", str(repo),
                 "--members", members, *args])
    out = capsys.readouterr().out
    return [json.loads(doc) for doc in out.split("\n---\n") if doc.strip()]


def _script(job):
    return job["spec"]["template"]["spec"]["containers"][0]["command"][2]


def _jcm_main_overrides(job):
    """Return the override list the Job's ``python -m jcm.main`` line passes."""
    line = next(ln for ln in _script(job).splitlines()
                if ln.strip().startswith("python -m jcm.main "))
    words = shlex.split(line.split("python -m jcm.main ", 1)[1])
    return words[:words.index("2>&1")]


def _env(job):
    return {e["name"]: e.get("value") for e in
            job["spec"]["template"]["spec"]["containers"][0]["env"]}


def test_nautilus_job_runs_the_pbs_overrides_at_the_pinned_commit(
        scratch, repo, gitrepo, remote, capsys):
    """One Job per member, running exactly the override list the PBS job does.

    Only the rundir differs (the cluster volume instead of $SCRATCH), and the
    code is the full SHA of the launched checkout's HEAD.
    """
    sha = _commit(gitrepo, "a")
    remote["refs/heads/dev"] = sha
    [job] = _k8s(gitrepo, capsys, "--tag", "t1")
    run = "mx_speedy_t31_t1"
    assert job["kind"] == "Job"
    assert job["metadata"]["name"] == "jcm-run-mx-speedy-t31-t1"
    assert job["metadata"]["labels"] == {"jcm-run": run}
    assert job["spec"]["backoffLimit"] == 20       # eviction budget
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "OnFailure"
    assert {"name": "runs", "persistentVolumeClaim": {
        "claimName": "jcm-runs"}} in pod["volumes"]
    assert pod["nodeSelector"] == {"nvidia.com/gpu.memory": "81920"}
    # The image by digest, so every retry and resume runs the same one.
    assert pod["containers"][0]["image"] == \
        f"ghcr.io/climate-analytics-lab/jcm@{DIGEST}"
    script = _script(job)
    assert (f"rm -rf /work/jcm\ngit clone --filter=blob:none --no-checkout "
            f"{launch.JCM_URL} /work/jcm\n"
            f"git -C /work/jcm fetch --depth 1 origin {sha}\n"
            f"git -C /work/jcm checkout --detach {sha}\n") in script
    assert ("pip install --no-cache-dir --disable-pip-version-check "
            "-c /tmp/cuda-jax.txt -e '/work/jcm[mam4]'") in script
    assert launch.pinned_setup(f"/runs/{run}") in script
    assert "PYTHONPATH=" not in script      # the pins, not side-loaded clones

    ovs = _jcm_main_overrides(job)
    member = yaml.safe_load(
        (pathlib.Path(launch.HERE) / "matrix.yaml").read_text())["members"][MEMBER]
    assert ovs == launch.member_overrides(run, member, DEFAULTS, f"/runs/{run}",
                                          fetch_here=False)
    launch.main(["--repo", str(repo), "--members", MEMBER, "--tag", "t1"])
    pbs = (repo / "runs" / f"{run}.pbs").read_text()
    pbs_ovs = shlex.split(pbs.split("python -u -m jcm.main", 1)[1]
                          .split("\necho ")[0].replace("\\\n", " "))
    assert [o.replace(f"{scratch}/jam_runs/", "/runs/") for o in pbs_ovs] == ovs
    assert f"run.checkpoint_path=/runs/{run}/checkpoint.msgpack" in ovs
    assert f"hydra.run.dir=/runs/{run}" in ovs
    assert f'if [ -f "/runs/{run}/checkpoint.msgpack" ]' in script
    assert '[ "$LAST" -lt 365 ]' in script


def test_nautilus_exports_the_recorded_mirror_commit(
        scratch, gitrepo, remote, capsys):
    from jcm.data import remote as mirror
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    [job] = _k8s(gitrepo, capsys, "--tag", "mc")
    env = _env(job)
    assert env["JCM_MIRROR_REVISION"] == mirror.MIRROR_REVISION
    assert "JCM_ALLOW_MIRROR_REVISION_CHANGE" not in env
    record = scratch / "nautilus_runs" / "mx_speedy_t31_mc" / launch.MIRROR_RECORD
    assert json.loads(record.read_text())["commit"] == mirror.MIRROR_REVISION


def test_nautilus_forced_mirror_switch_opts_the_job_in(
        scratch, gitrepo, remote, capsys, monkeypatch):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    monkeypatch.setenv("JCM_MIRROR_REVISION", "a" * 40)
    _k8s(gitrepo, capsys, "--tag", "fm")
    monkeypatch.setenv("JCM_MIRROR_REVISION", "c" * 40)
    with pytest.raises(SystemExit, match="force-mirror-revision"):
        _k8s(gitrepo, capsys, "--tag", "fm", "--resume")
    [job] = _k8s(gitrepo, capsys, "--tag", "fm", "--resume",
                 "--force-mirror-revision")
    env = _env(job)
    assert env["JCM_MIRROR_REVISION"] == "c" * 40
    assert env["JCM_ALLOW_MIRROR_REVISION_CHANGE"] == "1"


def test_nautilus_jam_inputs_stay_mirror_urls(monkeypatch):
    """A pod resolves the JAM aux inputs itself: no local cache path leaks in.

    A path into this node's Hugging Face cache would not exist in the pod.
    """
    from jcm.data import remote as mirror

    def no_fetch(rel):
        raise AssertionError(f"fetched {rel} for a Kubernetes job")

    monkeypatch.setattr(mirror, "fetch", no_fetch)
    ovs = launch.jam_aux("echam_t63_l95_hybrid", "l95", fetch_here=False)
    keys = [o.split("=", 1)[0] for o in ovs]
    assert keys == [f"forcing.{k}" for k in
                    launch._JAM_PD_KEYS + launch._DUST_KEYS]
    assert all(o.split("=", 1)[1].startswith("hf://bundles/t63") for o in ovs)
    assert "forcing.oxidants_file=hf://bundles/t63_l95/oxidants_pd.nc" in ovs


def test_default_pin_is_head_and_names_the_tag(scratch, gitrepo, remote,
                                               capsys):
    sha = _commit(gitrepo, "a")
    remote["refs/heads/dev"] = sha
    [job] = _k8s(gitrepo, capsys)
    short = _git(gitrepo, "rev-parse", "--short", "HEAD")
    assert job["metadata"]["labels"]["jcm-run"] == f"mx_speedy_t31_{short}"
    assert f"checkout --detach {sha}" in _script(job)


def test_pin_refuses_an_unpushed_commit(scratch, gitrepo, remote, capsys):
    """The pod clones from GitHub: a local-only commit must fail here, not there."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    local_only = _commit(gitrepo, "b")
    with pytest.raises(SystemExit) as e:
        _k8s(gitrepo, capsys, "--tag", "u")
    assert local_only[:12] in str(e.value) and "push it" in str(e.value)
    assert capsys.readouterr().out == ""
    assert not (scratch / "nautilus_runs").exists()      # nothing recorded


def test_pin_takes_a_pushed_ancestor_and_a_remote_branch(
        scratch, gitrepo, remote, capsys):
    first = _commit(gitrepo, "a")
    second = _commit(gitrepo, "b")
    remote["refs/heads/dev"] = second
    # An older commit is reachable from the pushed branch, so a pod can have it.
    [job] = _k8s(gitrepo, capsys, "--pin", f"jcm={first}")
    assert f"checkout --detach {first}" in _script(job)
    # The pinned commit, not the checkout's HEAD, names the default tag.
    assert job["metadata"]["labels"]["jcm-run"].endswith(
        _git(gitrepo, "rev-parse", "--short", first))
    # A branch name means the remote's branch, as mkrun.py resolves it.
    remote["refs/heads/feature"] = first
    [job] = _k8s(gitrepo, capsys, "--pin", "jcm=feature", "--tag", "br")
    assert f"checkout --detach {first}" in _script(job)


def test_pin_takes_only_jcm(scratch, gitrepo, remote, capsys):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    with pytest.raises(SystemExit, match="only jcm"):
        _k8s(gitrepo, capsys, "--pin", "jax-rrtmgp=main")


def test_suffix_and_tag_namespace_rundir_job_and_label(
        scratch, gitrepo, remote, capsys):
    """An arm carries its suffix in every artefact, so arms never share a run."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    [a, b] = [_k8s(gitrepo, capsys, "--tag", "rt", "--suffix", s)[0]
              for s in ("cape100", "cape150")]
    for job, s in ((a, "cape100"), (b, "cape150")):
        run = f"mx_speedy_t31_rt_{s}"
        assert job["metadata"]["name"] == f"jcm-run-mx-speedy-t31-rt-{s}"
        assert job["metadata"]["labels"]["jcm-run"] == run
        ovs = _jcm_main_overrides(job)
        assert f"run.output_prefix=/runs/{run}/{run}" in ovs
        assert f"run.checkpoint_path=/runs/{run}/checkpoint.msgpack" in ovs
        assert (scratch / "nautilus_runs" / run / launch.LAUNCH_RECORD).exists()


def test_arm_overrides_land_last(scratch, repo, gitrepo, remote, capsys):
    """An arm's settings win: warm start, then the raw overrides, last of all."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    arm = ["--days", "30", "--init", "/runs/spin/checkpoint_day90.ckpt",
           "--extra", "+physics.convection.trigger_cape=150.0", "run.foo=1"]
    [job] = _k8s(gitrepo, capsys, "--tag", "arm", "--suffix", "c150", *arm)
    ovs = _jcm_main_overrides(job)
    assert ovs[-4:] == ["init=from_state",
                        "init.file=/runs/spin/checkpoint_day90.ckpt",
                        "+physics.convection.trigger_cape=150.0", "run.foo=1"]
    assert "run.total_time=30.0" in ovs
    assert '[ "$LAST" -lt 30 ]' in _script(job)    # the gate follows --days
    # The PBS door layers the same arm the same way (a local state path).
    state = scratch / "state.ckpt"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(b"x")
    launch.main(["--repo", str(repo), "--members", MEMBER, "--tag", "arm",
                 "--suffix", "c150", "--days", "30", "--init", str(state),
                 "--extra", "+physics.convection.trigger_cape=150.0"])
    pbs = (repo / "runs" / "mx_speedy_t31_arm_c150.pbs").read_text()
    body = pbs.split("python -u -m jcm.main", 1)[1].split("\necho ")[0]
    assert shlex.split(body.replace("\\\n", " "))[-3:] == [
        "init=from_state", f"init.file={state}",
        "+physics.convection.trigger_cape=150.0"]


def test_extra_overrides_are_shell_quoted(scratch, repo, gitrepo, remote,
                                          capsys):
    """A Hydra list or quoted value reaches jcm.main as written."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    odd = 'run.snapshot_variables=[aerocom_clt,"radiation.toa_sw_up"]'
    [job] = _k8s(gitrepo, capsys, "--tag", "q", "--extra", odd)
    assert _jcm_main_overrides(job)[-1] == odd
    launch.main(["--repo", str(repo), "--members", MEMBER, "--tag", "q",
                 "--extra", odd])
    pbs = (repo / "runs" / "mx_speedy_t31_q.pbs").read_text()
    assert shlex.quote(odd) in pbs


def test_resume_reuses_the_recorded_launch(scratch, gitrepo, remote, capsys):
    """--resume re-emits the recorded definition even after HEAD has moved.

    A resumed arm must continue on the same code with the same overrides; the
    tag, not the command line, says which launch that is.
    """
    first = _commit(gitrepo, "a")
    remote["refs/heads/dev"] = first
    [fresh] = _k8s(gitrepo, capsys, "--tag", "rs", "--suffix", "arm1",
                   "--days", "10", "--extra", "+physics.convection.tau=3600.0")
    remote["refs/heads/dev"] = _commit(gitrepo, "b")          # HEAD moves on
    [resumed] = _k8s(gitrepo, capsys, "--tag", "rs", "--suffix", "arm1",
                     "--resume")
    # The same Job, except that it may continue a checkpoint another Job wrote.
    assert "[ 0 -eq 0 ]" in _script(fresh)
    assert _script(resumed) == _script(fresh).replace("[ 0 -eq 0 ]",
                                                      "[ 1 -eq 0 ]")
    assert f"checkout --detach {first}" in _script(resumed)
    assert _jcm_main_overrides(resumed)[-1] == "+physics.convection.tau=3600.0"


def test_resume_cannot_change_the_definition(scratch, gitrepo, remote,
                                             capsys):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    _k8s(gitrepo, capsys, "--tag", "rd")
    for change in (["--days", "3"], ["--extra", "a=1"],
                   ["--init", "/runs/x.ckpt"], ["--pin", "jcm=HEAD"]):
        with pytest.raises(SystemExit, match="new --tag or --suffix"):
            _k8s(gitrepo, capsys, "--tag", "rd", "--resume", *change)


def test_resume_needs_a_recorded_launch(scratch, gitrepo, remote, capsys):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    with pytest.raises(SystemExit, match="no recorded launch"):
        _k8s(gitrepo, capsys, "--tag", "never", "--resume")


def test_relaunch_is_idempotent_but_a_new_definition_is_refused(
        scratch, gitrepo, remote, capsys):
    """Inspect-then-submit regenerates the same launch; a changed one is refused."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    [one] = _k8s(gitrepo, capsys, "--tag", "id")
    [two] = _k8s(gitrepo, capsys, "--tag", "id")
    assert one == two
    with pytest.raises(SystemExit, match="different definition"):
        _k8s(gitrepo, capsys, "--tag", "id", "--days", "7")


def _guard(tmp_path, digest, resume=False):
    rundir = tmp_path / "vol" / "run"
    rundir.mkdir(parents=True, exist_ok=True)
    defn = {"digest": digest, "run": "run"}
    return rundir, launch.rundir_guard(str(rundir), f"{rundir}/ckpt", defn,
                                       resume)


def _bash(script, **env):
    return subprocess.run(["bash", "-c", "set -euo pipefail\n" + script],
                          capture_output=True, text=True,
                          env={**os.environ, **env})


def test_rundir_guard_ties_the_volume_to_one_launch_and_one_start(
        tmp_path, scratch, gitrepo, remote, capsys):
    """The pod refuses what check_fresh refuses on a PBS rundir (#701).

    The generating node cannot see the volume, so the guard is the only thing
    that stops a reused tag, or a fresh relaunch after its Job was deleted,
    from silently resuming an existing run and reporting it as its own.
    """
    rundir, fresh = _guard(tmp_path, "aaaa")
    _, resume = _guard(tmp_path, "aaaa", resume=True)
    no_uid = _bash(fresh, JOB_UID="")
    assert no_uid.returncode == 1 and "JOB_UID is empty" in no_uid.stdout
    assert not (rundir / "launch.json").exists()
    assert _bash(fresh, JOB_UID="u1").returncode == 0      # first attempt
    assert (rundir / "launch.json").read_text() == launch.launch_record_text(
        {"digest": "aaaa", "run": "run"})
    (rundir / "ckpt").write_bytes(b"x")                    # it integrates
    assert _bash(fresh, JOB_UID="u1").returncode == 0      # its own retry
    other = _bash(fresh, JOB_UID="u2")                     # a fresh relaunch
    assert other.returncode == 1 and "written by another Job" in other.stdout
    assert _bash(resume, JOB_UID="u2").returncode == 0     # --resume
    assert (rundir / "JOBS").read_text().split() == ["u1", "u2"]
    foreign = _bash(_guard(tmp_path, "bbbb", resume=True)[1], JOB_UID="u3")
    assert foreign.returncode == 1 and "different launch" in foreign.stdout
    (rundir / "launch.json").unlink()
    orphan = _bash(resume, JOB_UID="u2")
    assert orphan.returncode == 1 and "unknown origin" in orphan.stdout
    # And the Job runs exactly this guard, before it clones anything, with its
    # uid from the label the Job controller puts on every pod it creates.
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    [job] = _k8s(gitrepo, capsys, "--tag", "g")
    defn = json.loads((scratch / "nautilus_runs" / "mx_speedy_t31_g"
                       / launch.LAUNCH_RECORD).read_text())
    shipped = launch.rundir_guard("/runs/mx_speedy_t31_g",
                                  "/runs/mx_speedy_t31_g/checkpoint.msgpack",
                                  defn, resume=False)
    script = _script(job)
    assert shipped in script
    assert script.index(shipped) < script.index("git clone")
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    assert {"name": "JOB_UID", "valueFrom": {"fieldRef": {"fieldPath": (
        "metadata.labels['batch.kubernetes.io/controller-uid']")}}} in env


def test_job_name_must_be_a_dns_label():
    """Refused, not folded: truncating or lowercasing merges two runs' Jobs."""
    assert launch.job_name("jcm-run", "mx_a_b1") == "jcm-run-mx-a-b1"
    with pytest.raises(SystemExit, match="63"):
        launch.job_name("jcm-run", "mx_echam_jam_t63_l95_31d9f6ff_" + "x" * 30)
    with pytest.raises(SystemExit, match="alphanumeric"):
        launch.job_name("jcm-run", "mx_a_tag_")
    with pytest.raises(SystemExit, match="lower case"):
        launch.job_name("jcm-run", "mx_a_B1")       # --tag B1 vs --tag b1


def test_upper_case_tag_is_refused_on_kubernetes(scratch, gitrepo, remote,
                                                 capsys):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    with pytest.raises(SystemExit, match="lower case"):
        _k8s(gitrepo, capsys, "--tag", "T1")
    assert not (scratch / "nautilus_runs").exists()


def test_submit_applies_every_job_after_vetting_all(
        scratch, gitrepo, remote, capsys, cluster):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    _k8s(gitrepo, capsys, "--tag", "s", "--submit",
         members="speedy-t31,echam-1m-t63")
    applied = [json.loads(i)["metadata"]["name"]
               for (args, i) in cluster["calls"] if args[0] == "apply"]
    assert applied == ["jcm-run-mx-speedy-t31-s", "jcm-run-mx-echam-1m-t63-s"]
    # The applied manifest is the one recorded next to the launch.
    rec = json.loads((scratch / "nautilus_runs" / "mx_speedy_t31_s" / "job.json")
                     .read_text())
    first = next(json.loads(i) for (args, i) in cluster["calls"]
                 if args[0] == "apply")
    assert first == rec


def test_submit_never_touches_a_running_job(scratch, gitrepo, remote, capsys,
                                            cluster):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    cluster["jobs"]["jcm-run-mx-speedy-t31-r"] = "active"
    for extra in ([], ["--resume"]):
        with pytest.raises(SystemExit, match="still running"):
            _k8s(gitrepo, capsys, "--tag", "r", "--submit", *extra)
    assert [a for a, _ in cluster["calls"] if a[0] in ("apply", "delete")] == []


def test_resume_replaces_the_finished_job_and_fresh_refuses(
        scratch, gitrepo, remote, capsys, cluster):
    """Jobs are immutable: a resume replaces the finished one; a fresh launch never does."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    _k8s(gitrepo, capsys, "--tag", "f", "--submit")
    name = "jcm-run-mx-speedy-t31-f"
    cluster["jobs"][name] = "failed"            # e.g. evictions exhausted
    with pytest.raises(SystemExit, match="already exists"):
        _k8s(gitrepo, capsys, "--tag", "f", "--submit")
    cluster["calls"].clear()
    _k8s(gitrepo, capsys, "--tag", "f", "--submit", "--resume")
    verbs = [(a[0], a[2] if a[0] == "delete" else None)
             for a, _ in cluster["calls"] if a[0] in ("apply", "delete")]
    assert verbs == [("delete", name), ("apply", None)]


def test_k8s_only_flags_are_refused_on_pbs(scratch, repo):
    for flag in (["--pin", "jcm=HEAD"], ["--fetch"], ["--with-checkpoints"]):
        with pytest.raises(SystemExit, match="Kubernetes"):
            _launch(repo, *flag)


def test_engine_gate_reads_the_checkpoint_jcm_writes():
    """The last run.checkpoint_path wins in Hydra, so the gate reads that one."""
    ovs = ["run.checkpoint_path=/runs/r/checkpoint.msgpack",
           "++run.checkpoint_path=/runs/r/arm.msgpack"]
    assert launch.mkrun.checkpoint_of(ovs) == "/runs/r/arm.msgpack"
    job = launch.mkrun.job_manifest(
        site=launch.sites.get("nautilus"), job_name="j", label="l",
        rundir="/runs/r", overrides=ovs, days=1, resolved={}, setup="",
        python_env="", retries=0, gpus=1, cpu=1, memory="1Gi")
    assert 'if [ -f "/runs/r/arm.msgpack" ]' in _script(job)
    with pytest.raises(ValueError, match="no run.checkpoint_path"):
        launch.mkrun.checkpoint_of(["run.total_time=1"])


def test_regenerating_a_launch_keeps_its_recorded_mirror_commit(
        scratch, gitrepo, remote, capsys, monkeypatch):
    """Inspect-then-submit must not re-point a running Job's mirror record.

    The digest leaves the mirror commit out (jcm's own resume check guards
    it), so without this a regenerated launch would record whatever commit
    this process reads now, and a later --resume would export it.
    """
    from jcm.data import remote as mirror
    first = mirror.MIRROR_REVISION
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    _k8s(gitrepo, capsys, "--tag", "rg")
    monkeypatch.setattr(mirror, "MIRROR_REVISION", "d" * 40)   # pin moves
    monkeypatch.delenv("JCM_MIRROR_REVISION")
    [job] = _k8s(gitrepo, capsys, "--tag", "rg")
    assert _env(job)["JCM_MIRROR_REVISION"] == first
    monkeypatch.setenv("JCM_MIRROR_REVISION", "c" * 40)       # explicit
    with pytest.raises(SystemExit, match="force-mirror-revision"):
        _k8s(gitrepo, capsys, "--tag", "rg")


FAKE_KUBECTL = r"""#!/usr/bin/env bash
# Stand-in kubectl: pods are no-ops, the "volume" /runs is $FAKE_VOLUME, and
# every call is logged to $FAKE_VOLUME.log.
while [ "$1" = "-n" ]; do shift 2; done
echo "$*" >> "$FAKE_VOLUME.log"
case "$1" in
  apply) cat > /dev/null; echo "pod created" ;;
  delete) echo deleted ;;
  get)
    if [ "$2" = job ]; then echo "jobs.batch \"$3\" NotFound" >&2; exit 1; fi
    echo -n "${FAKE_PHASE:-Running}" ;;
  exec)
    shift 3   # exec POD --
    if [ "$1" = sh ]; then
      exec sh -c "${3//\/runs/$FAKE_VOLUME}"
    fi
    if [ -n "${FAKE_TAR_FAIL:-}" ]; then exit 2; fi
    # A run still being written: its log grows after the listing.
    if [ -n "${FAKE_GROW:-}" ]; then echo more >> "$FAKE_VOLUME/$FAKE_GROW"; fi
    args=(); for x in "$@"; do args+=("${x//\/runs/$FAKE_VOLUME}"); done
    exec "${args[@]}" ;;
esac
"""


@pytest.fixture
def volume(tmp_path, monkeypatch):
    """Return a fake cluster volume holding one finished speedy run."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "kubectl").write_text(FAKE_KUBECTL)
    (bindir / "kubectl").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    vol = tmp_path / "volume"
    monkeypatch.setenv("FAKE_VOLUME", str(vol))
    run = vol / "mx_speedy_t31_ft"
    (run / ".hydra").mkdir(parents=True)
    files = {"mx_speedy_t31_ft_day5.nc": b"n" * 1000,
             "mx_speedy_t31_ft_day5.nc.provenance.json": b"{}",
             ".hydra/overrides.yaml": b"- a=1\n", "run.log": b"| Day 5 (\n",
             "launch.json": b"{}", "checkpoint.msgpack": b"c" * 5000,
             "checkpoint.msgpack.prev": b"c", "mx_speedy_t31_ft_day30.ckpt": b"c"}
    for name, data in files.items():
        (run / name).write_bytes(data)
    return run



def test_fetch_copies_the_run_but_not_its_checkpoints(scratch, volume,
                                                      gitrepo, capsys):
    launch.main(["--site", "nautilus", "--repo", str(gitrepo), "--fetch",
                 "--members", MEMBER, "--tag", "ft"])
    local = scratch / "nautilus_runs" / "mx_speedy_t31_ft"
    got = sorted(str(p.relative_to(local)) for p in local.rglob("*")
                 if p.is_file())
    assert got == sorted([".hydra/overrides.yaml", "launch.json", "run.log",
                          "mx_speedy_t31_ft_day5.nc",
                          "mx_speedy_t31_ft_day5.nc.provenance.json"])
    assert (local / "mx_speedy_t31_ft_day5.nc").read_bytes() == b"n" * 1000
    # A later fetch never replaces the local launch record --resume reads,
    # and refuses to copy another launch's run next to it.
    (local / "launch.json").write_text('{"digest": "local"}\n')
    with pytest.raises(SystemExit, match="different launch; nothing copied"):
        launch.main(["--site", "nautilus", "--repo", str(gitrepo), "--fetch",
                     "--members", MEMBER, "--tag", "ft"])
    assert (local / "launch.json").read_text() == '{"digest": "local"}\n'


def test_fetch_is_incremental_and_reports_a_short_copy(scratch, volume,
                                                       monkeypatch, capsys):
    import fetch_run
    site = launch.sites.get("nautilus")
    dest = scratch / "copy"
    assert fetch_run.fetch_run("mx_speedy_t31_ft", dest, site=site,
                               pod="p", with_checkpoints=True) == []
    assert (dest / "checkpoint.msgpack").stat().st_size == 5000
    # The run grew; this stream dies, so the grown file stays short.
    with open(volume / "mx_speedy_t31_ft_day5.nc", "ab") as fh:
        fh.write(b"more")
    monkeypatch.setenv("FAKE_TAR_FAIL", "1")
    capsys.readouterr()
    bad = fetch_run.fetch_run("mx_speedy_t31_ft", dest, site=site, pod="p")
    assert bad == ["mx_speedy_t31_ft_day5.nc (short)"]
    assert "1 of 5 files to copy" in capsys.readouterr().err
    monkeypatch.delenv("FAKE_TAR_FAIL")
    assert fetch_run.fetch_run("mx_speedy_t31_ft", dest, site=site,
                               pod="p") == []


def test_fetch_of_a_missing_run_says_so(scratch, volume):
    import fetch_run
    with pytest.raises(SystemExit, match="cannot list /runs/nope"):
        fetch_run.fetch_run("nope", scratch / "x",
                            site=launch.sites.get("nautilus"), pod="p")


def test_reader_pod_mounts_the_volume_read_only():
    import fetch_run
    pod = fetch_run.reader_pod(launch.sites.get("nautilus"), "p", "r")
    spec = pod["spec"]
    assert spec["containers"][0]["volumeMounts"][0]["readOnly"] is True
    assert spec["volumes"][0]["persistentVolumeClaim"] == {
        "claimName": "jcm-runs", "readOnly": True}
    assert "nvidia.com/a100" not in json.dumps(spec)   # no GPU quota


def test_pod_records_the_resolved_dependency_versions(tmp_path):
    """The shipped setup writes one deps line per attempt, and tolerates a gap.

    It runs under the Job's ``set -euo pipefail``, so a broken line here would
    fail every attempt of every member; run it, do not just read it.
    """
    rundir = tmp_path / "run"
    rundir.mkdir()
    setup = launch.pinned_setup(str(rundir))
    snippet = setup[setup.index("python - <<'PYDEPS'"):]
    snippet = snippet.replace("python - ", f"{sys.executable} - ", 1)
    for _ in range(2):                                   # two attempts
        r = _bash(snippet)
        assert r.returncode == 0, r.stderr
    lines = (rundir / "PROVENANCE").read_text().splitlines()
    assert len(lines) == 2 and lines[0] == lines[1]
    assert lines[0].startswith("deps: jcm==")
    for dist in launch._RECORDED_DISTS:
        assert f" {dist}==" in f" {lines[0][6:]}"


def test_fetch_keeps_the_local_launch_record(scratch, volume, capsys):
    """The local record is what --resume re-emits; the volume's never replaces it."""
    import fetch_run
    dest = scratch / "keep"
    dest.mkdir(parents=True)
    (dest / "launch.json").write_bytes((volume / "launch.json").read_bytes())
    stamp = (volume / "launch.json").stat().st_mtime + 500
    os.utime(dest / "launch.json", (stamp, stamp))     # ours, same content
    assert fetch_run.fetch_run("mx_speedy_t31_ft", dest,
                               site=launch.sites.get("nautilus"), pod="p",
                               keep=("launch.json",)) == []
    assert (dest / "launch.json").stat().st_mtime == stamp   # not rewritten
    assert (dest / "run.log").exists()


def test_fetch_of_a_run_still_being_written_is_not_short(scratch, volume,
                                                          monkeypatch):
    """A file that grows between the listing and the copy is complete, not short."""
    import fetch_run
    monkeypatch.setenv("FAKE_GROW", "mx_speedy_t31_ft/run.log")
    dest = scratch / "grow"
    listed = (volume / "run.log").stat().st_size
    assert fetch_run.fetch_run("mx_speedy_t31_ft", dest,
                               site=launch.sites.get("nautilus"), pod="p") == []
    assert (dest / "run.log").stat().st_size > listed


def test_fetch_deletes_a_reader_pod_that_never_ran(scratch, volume,
                                                   monkeypatch):
    """A leaked reader keeps the volume attached: delete it on every path."""
    import fetch_run
    monkeypatch.setenv("FAKE_PHASE", "Failed")
    with pytest.raises(SystemExit, match="never reached Running"):
        fetch_run.fetch_run("mx_speedy_t31_ft", scratch / "x",
                            site=launch.sites.get("nautilus"), pod="p")
    calls = pathlib.Path(os.environ["FAKE_VOLUME"] + ".log").read_text().splitlines()
    assert calls[-1].startswith("delete pod p")


def test_image_is_resolved_to_its_digest_with_the_registry_token_flow(
        monkeypatch):
    """The moving tag is pinned the way a pull resolves it (401, token, HEAD)."""
    import io
    import urllib.error
    import urllib.request
    seen = []

    class Response(io.BytesIO):
        def __init__(self, body=b"", headers=None):
            super().__init__(body)
            self.headers = headers or {}

    def urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        auth = None if isinstance(req, str) else req.get_header("Authorization")
        seen.append((url, auth))
        if url.startswith("https://ghcr.io/token"):
            return Response(json.dumps({"token": "tkn"}).encode())
        if auth != "Bearer tkn":
            raise urllib.error.HTTPError(url, 401, "no", {
                "WWW-Authenticate": 'Bearer realm="https://ghcr.io/token",'
                'service="ghcr.io",scope="repository:org/img:pull"'}, None)
        return Response(headers={"Docker-Content-Digest": DIGEST})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert launch.image_ref("ghcr.io/org/img:latest") == \
        f"ghcr.io/org/img@{DIGEST}"
    assert seen[1] == ("https://ghcr.io/token?service=ghcr.io&"
                       "scope=repository:org/img:pull", None)
    assert seen[2] == ("https://ghcr.io/v2/org/img/manifests/latest",
                       "Bearer tkn")
    assert launch.image_ref(f"ghcr.io/org/img@{DIGEST}") == \
        f"ghcr.io/org/img@{DIGEST}"                    # already pinned

    def down(req, timeout=None):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    with pytest.raises(SystemExit, match="cannot resolve"):
        launch.image_ref("ghcr.io/org/img:latest")


def test_resume_keeps_the_recorded_image(scratch, gitrepo, remote, capsys,
                                         monkeypatch):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    [fresh] = _k8s(gitrepo, capsys, "--tag", "im")
    monkeypatch.setattr(launch, "image_ref",
                        lambda image: "ghcr.io/x@sha256:" + "e" * 64)
    [resumed] = _k8s(gitrepo, capsys, "--tag", "im", "--resume")
    images = [j["spec"]["template"]["spec"]["containers"][0]["image"]
              for j in (fresh, resumed)]
    assert images == [f"ghcr.io/climate-analytics-lab/jcm@{DIGEST}"] * 2
    # ...while a fresh relaunch against a republished image is a new launch.
    with pytest.raises(SystemExit, match="different definition"):
        _k8s(gitrepo, capsys, "--tag", "im")


def test_warm_start_state_is_checked_before_submitting(
        scratch, gitrepo, remote, capsys, monkeypatch):
    """A missing hf:// state refuses here; a volume path, in the pod's guard."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    asked = []
    monkeypatch.setattr(launch, "hf_exists",
                        lambda url: asked.append(url) or False)
    state = "hf://bundles/t31_l8/init_states/typo.msgpack"
    with pytest.raises(SystemExit, match="typo.msgpack"):
        _k8s(gitrepo, capsys, "--tag", "ws", "--init", state)
    assert asked == [state]
    assert capsys.readouterr().out == ""

    def offline(url):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(launch, "hf_exists", offline)
    with pytest.raises(SystemExit, match="hub unreachable"):
        _k8s(gitrepo, capsys, "--tag", "ws", "--init", state)
    assert not (scratch / "nautilus_runs" / "mx_speedy_t31_ws").exists()
    # A path on the volume passes here and is checked where it lives.
    [job] = _k8s(gitrepo, capsys, "--tag", "wv", "--init", "/runs/d/s.ckpt")
    script = _script(job)
    assert "if [ ! -f /runs/d/s.ckpt ]; then" in script
    assert script.index("/runs/d/s.ckpt ]") < script.index("git clone")


def test_guard_refuses_a_missing_warm_start_state(tmp_path):
    rundir = tmp_path / "run"
    rundir.mkdir()
    state = tmp_path / "donor.ckpt"
    defn = {"digest": "a", "overrides": ["init=from_state",
                                         f"init.file={state}"]}
    guard = launch.rundir_guard(str(rundir), f"{rundir}/ckpt", defn, False)
    missing = _bash(guard, JOB_UID="u")
    assert missing.returncode == 1 and "does not exist" in missing.stdout
    state.write_bytes(b"x")
    assert _bash(guard, JOB_UID="u").returncode == 0


def test_force_mirror_revision_needs_resume_on_kubernetes(
        scratch, gitrepo, remote, capsys):
    """A fresh Job never carries the opt-in to resume across mirror commits."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    with pytest.raises(SystemExit, match="needs --resume"):
        _k8s(gitrepo, capsys, "--tag", "fr", "--force-mirror-revision")


def test_kubernetes_records_live_apart_from_pbs_rundirs(
        scratch, repo, gitrepo, remote, capsys):
    """A PBS run and a Kubernetes run under one name never share a directory."""
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    _k8s(gitrepo, capsys, "--tag", "sep")
    launch.main(["--repo", str(repo), "--members", MEMBER, "--tag", "sep"])
    k8s = scratch / "nautilus_runs" / "mx_speedy_t31_sep"
    pbs = scratch / "jam_runs" / "mx_speedy_t31_sep"
    assert (k8s / launch.LAUNCH_RECORD).exists()
    assert not (pbs / launch.LAUNCH_RECORD).exists()
    assert (pbs / launch.MIRROR_RECORD).exists()


def test_fetch_carries_on_past_a_member_that_fails(scratch, volume, gitrepo,
                                                   capsys):
    """A member never launched is reported with the rest, not fatal to them."""
    with pytest.raises(SystemExit) as e:
        launch.main(["--site", "nautilus", "--repo", str(gitrepo), "--fetch",
                     "--members", "echam-1m-t63,speedy-t31", "--tag", "ft"])
    assert "incomplete copies" in str(e.value)
    assert "echam-1m-t63" in str(e.value) and "cannot list" in str(e.value)
    assert "speedy-t31" not in str(e.value).split("incomplete copies")[1]
    assert (scratch / "nautilus_runs" / "mx_speedy_t31_ft" / "run.log").exists()


def test_fetch_refreshes_a_file_rewritten_at_the_same_size(scratch, volume):
    """The rotating checkpoint is rewritten in place at a fixed size."""
    import fetch_run
    site = launch.sites.get("nautilus")
    dest = scratch / "same"
    fetch_run.fetch_run("mx_speedy_t31_ft", dest, site=site, pod="p",
                        with_checkpoints=True)
    ckpt = volume / "checkpoint.msgpack"
    ckpt.write_bytes(b"D" * 5000)                    # a later state, same size
    stamp = ckpt.stat().st_mtime + 100
    os.utime(ckpt, (stamp, stamp))
    assert fetch_run.fetch_run("mx_speedy_t31_ft", dest, site=site, pod="p",
                               with_checkpoints=True) == []
    assert (dest / "checkpoint.msgpack").read_bytes() == b"D" * 5000

    # ...and when that refresh's stream fails, the stale same-size copy is
    # reported, not accepted on its size.
    ckpt.write_bytes(b"E" * 5000)
    os.utime(ckpt, (stamp + 100, stamp + 100))
    os.environ["FAKE_TAR_FAIL"] = "1"
    try:
        bad = fetch_run.fetch_run("mx_speedy_t31_ft", dest, site=site,
                                  pod="p", with_checkpoints=True)
    finally:
        del os.environ["FAKE_TAR_FAIL"]
    assert bad == ["checkpoint.msgpack (stale: the copy of the rewritten "
                   "file did not complete)"]


def test_fetch_replaces_a_truncated_local_record(scratch, volume, gitrepo):
    """An interrupted first fetch leaves a cut launch.json; the next fetch heals it.

    The truncated record is neither defended as this machine's record nor
    read (a JSON error would block every later --fetch): it is short and
    unparsable, so it is copied again like any other incomplete file.
    """
    import fetch_run
    (volume / "launch.json").write_text('{"digest": "aaaa", "job": "j"}\n')
    dest = scratch / "cut"
    dest.mkdir(parents=True)
    (dest / "launch.json").write_text('{"digest": "aa')        # tar cut mid-file
    assert fetch_run.fetch_run("mx_speedy_t31_ft", dest,
                               site=launch.sites.get("nautilus"), pod="p",
                               keep=("launch.json",)) == []
    assert (dest / "launch.json").read_text() == '{"digest": "aaaa", "job": "j"}\n'
    # The launcher's own --fetch door reads that record before copying and
    # must not fail on the cut one either.
    local = scratch / "nautilus_runs" / "mx_speedy_t31_ft"
    local.mkdir(parents=True)
    (local / "launch.json").write_text('{"digest": "aa')
    launch.main(["--site", "nautilus", "--repo", str(gitrepo), "--fetch",
                 "--members", MEMBER, "--tag", "ft"])
    assert json.loads((local / "launch.json").read_text())["digest"] == "aaaa"


def test_resume_does_not_read_a_truncated_record_as_absent(scratch, gitrepo):
    """--resume re-emits the record: a cut one is an error, never silently new."""
    local = scratch / "nautilus_runs" / "mx_speedy_t31_ft"
    local.mkdir(parents=True)
    (local / "launch.json").write_text('{"digest": "aa')
    with pytest.raises(ValueError):
        launch.read_launch(local)


def test_fetch_refuses_a_foreign_record_of_the_same_size(scratch, volume):
    """A different launch on the volume is not copied in under this record.

    Compared by content: two records normally have one length (fixed-width
    digest and SHA).
    """
    import fetch_run
    dest = scratch / "foreign"
    dest.mkdir(parents=True)
    (volume / "launch.json").write_text('{"digest": "aaaa"}\n')
    (dest / "launch.json").write_text('{"digest": "bbbb"}\n')
    bad = fetch_run.fetch_run("mx_speedy_t31_ft", dest,
                              site=launch.sites.get("nautilus"), pod="p",
                              keep=("launch.json",))
    assert len(bad) == 1 and "different launch; nothing copied" in bad[0]
    assert sorted(p.name for p in dest.iterdir()) == ["launch.json"]
    assert (dest / "launch.json").read_text() == '{"digest": "bbbb"}\n'


def test_clone_restarts_cleanly_and_stops_on_a_failed_step(tmp_path):
    """A container restarted in its Pod keeps /work; the clone must cope.

    The checkout is removed first, and each git step is its own command, so
    under the Job's ``set -e`` a failed step stops the attempt — in an
    ``a && b`` list a failed ``a`` would not, and the attempt would carry on
    with whatever a killed earlier attempt left in /work.
    """
    job = launch.mkrun.job_manifest(
        site=launch.sites.get("nautilus"), job_name="j", label="l",
        rundir="/runs/r", overrides=["run.checkpoint_path=/runs/r/c"],
        days=1, resolved={"jcm": ("https://example/x", "a" * 40)},
        setup="", python_env="", retries=0, gpus=1, cpu=1, memory="1Gi")
    script = _script(job)
    clone = script[script.index("rm -rf /work/jcm"):script.index("cd /work/jcm")]
    work = tmp_path / "work"
    (work / "jcm").mkdir(parents=True)
    (work / "jcm" / "left-by-a-killed-attempt").write_text("x")
    log = tmp_path / "git.log"
    fake = f"""git() {{ echo "$*" >> {log}; case "$1" in clone) mkdir -p "$5";;
  -C) [ "$3" != fetch ];; esac; }}
"""
    r = _bash(fake + clone.replace("/work/", f"{work}/") + "echo REACHED\n")
    assert r.returncode != 0 and "REACHED" not in r.stdout   # fetch failed
    calls = log.read_text().splitlines()
    assert calls[0].startswith("clone ") and calls[1].startswith(f"-C {work}/jcm fetch")
    assert len(calls) == 2                        # no checkout after it
    assert not (work / "jcm" / "left-by-a-killed-attempt").exists()


def test_dependencies_are_resolved_once_per_launch(tmp_path):
    """The first attempt writes the lock; every later one installs under it.

    The pins are floors, so resolving again on a retry or a resume could pick
    up a release made mid-run and change the model across a checkpoint.
    """
    rundir = tmp_path / "run"
    rundir.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "pip.log"
    (bindir / "pip").write_text(f"""#!/usr/bin/env bash
echo "$*" >> {log}
if [ "$1" = freeze ]; then
  printf 'jax==0.10.1\njaxlib==0.10.1\n-e git+https://x@y#egg=jcm\njcm @ file:///app\nfoo==1.2\n'
fi
""")
    (bindir / "pip").chmod(0o755)
    setup = launch.pinned_setup(str(rundir))
    install = setup[setup.index("if [ -f "):setup.index("python - <<'PYDEPS'")]
    install = install.replace("/tmp/cuda-jax.txt", f"{tmp_path}/cuda-jax.txt")
    env = {"PATH": f"{bindir}:{os.environ['PATH']}"}
    assert _bash(install, **env).returncode == 0          # first attempt
    lock = rundir / launch.LOCK_FILE
    assert lock.read_text().split() == ["jax==0.10.1", "jaxlib==0.10.1",
                                        "foo==1.2"]
    first = log.read_text().splitlines()
    assert any(f"-c {tmp_path}/cuda-jax.txt -e /work/jcm[mam4]" in c
               for c in first)
    log.write_text("")
    assert _bash(install, **env).returncode == 0          # a later attempt
    later = log.read_text().splitlines()
    assert later == [f"install --no-cache-dir --disable-pip-version-check "
                     f"-c {lock} -e /work/jcm[mam4]"]


def test_fetch_refuses_a_run_with_no_launch_record(scratch, volume):
    """Outputs no launch recorded are not copied in under this machine's record."""
    import fetch_run
    dest = scratch / "orphan"
    dest.mkdir(parents=True)
    (volume / "launch.json").unlink()                # a legacy / cleaned run
    (dest / "launch.json").write_text('{"digest": "mine"}\n')
    bad = fetch_run.fetch_run("mx_speedy_t31_ft", dest,
                              site=launch.sites.get("nautilus"), pod="p",
                              keep=("launch.json",))
    assert len(bad) == 1 and "is absent" in bad[0] and "nothing copied" in bad[0]
    assert sorted(p.name for p in dest.iterdir()) == ["launch.json"]
    # With no local record either (a run launched elsewhere), it is copied.
    assert fetch_run.fetch_run("mx_speedy_t31_ft", scratch / "elsewhere",
                               site=launch.sites.get("nautilus"), pod="p",
                               keep=("launch.json",)) == []


@pytest.mark.parametrize("extra", ["run.total_time=3", "+run.total_time=3",
                                   "++run.end_time=2001-01-01"])
def test_extra_cannot_set_the_run_length(scratch, repo, gitrepo, remote,
                                         capsys, extra):
    """The length is --days, which the Kubernetes day-count gate follows.

    Set through --extra, jcm would stop at the new length while the gate kept
    the old target, and every retry would fail a finished run.
    """
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    with pytest.raises(SystemExit, match="--days"):
        _k8s(gitrepo, capsys, "--tag", "len", "--extra", extra)
    assert not (scratch / "nautilus_runs").exists()
    with pytest.raises(SystemExit, match="--days"):
        _launch(repo, "--tag", "len", "--extra", extra)
    assert not list((repo / "runs").glob("*.pbs"))


def test_pod_writes_the_mirror_record_beside_the_launch_record(tmp_path):
    rundir = tmp_path / "run"
    rundir.mkdir()
    record = json.dumps({"commit": "c" * 40, "source": "pinned"}, indent=1)
    guard = launch.rundir_guard(str(rundir), f"{rundir}/ckpt",
                                {"digest": "a"}, False, record)
    assert _bash(guard, JOB_UID="u").returncode == 0
    assert json.loads((rundir / launch.MIRROR_RECORD).read_text()) == \
        json.loads(record)


def test_a_fetched_launch_resumes_at_its_own_mirror_commit(
        tmp_path, scratch, volume, gitrepo, remote, capsys, monkeypatch):
    """Machine B fetches machine A's launch and resumes it on A's mirror commit.

    The pod writes both records into the rundir; without the mirror record B
    would resume at its own checkout's pin, which may have moved.
    """
    from jcm.data import remote as mirror
    first = mirror.MIRROR_REVISION
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    [job] = _k8s(gitrepo, capsys, "--tag", "ft")               # machine A
    a_local = scratch / "nautilus_runs" / "mx_speedy_t31_ft"
    script = _script(job)
    mirror_text = (a_local / launch.MIRROR_RECORD).read_text()
    assert mirror_text.rstrip("\n") in script                  # the pod's copy
    for name in (launch.LAUNCH_RECORD, launch.MIRROR_RECORD):   # what it wrote
        (volume / name).write_text((a_local / name).read_text())
    monkeypatch.setenv("SCRATCH", str(tmp_path / "machine_b"))
    launch.main(["--site", "nautilus", "--repo", str(gitrepo), "--fetch",
                 "--members", MEMBER, "--tag", "ft"])
    monkeypatch.setattr(mirror, "MIRROR_REVISION", "d" * 40)   # B's pin moved
    monkeypatch.delenv("JCM_MIRROR_REVISION")
    [resumed] = _k8s(gitrepo, capsys, "--tag", "ft", "--resume")
    assert _env(resumed)["JCM_MIRROR_REVISION"] == first


def test_resume_needs_a_mirror_record(scratch, gitrepo, remote, capsys):
    remote["refs/heads/dev"] = _commit(gitrepo, "a")
    _k8s(gitrepo, capsys, "--tag", "nm")
    (scratch / "nautilus_runs" / "mx_speedy_t31_nm"
     / launch.MIRROR_RECORD).unlink()
    with pytest.raises(SystemExit, match="mirror commit the run reads"):
        _k8s(gitrepo, capsys, "--tag", "nm", "--resume")
