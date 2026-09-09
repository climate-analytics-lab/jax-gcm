"""Tests for the release-validation launcher.

The guarded property is provenance: every member of a matrix launch must
integrate a *fresh* year in a directory named after the commit under test. A
fixed rundir let a second launch silently resume the first one's checkpoint
and report a healthy year for an integration that never ran (#701).

Nothing here submits anything -- ``qsub`` is monkeypatched out.
"""

from __future__ import annotations

import pathlib
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
    monkeypatch.setattr(launch, "jam_aux", lambda grid, levels: [])
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
