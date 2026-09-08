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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import launch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
MEMBER = "speedy-t31"          # the cheapest member: no JAM aux-input lookup


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
