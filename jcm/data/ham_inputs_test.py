"""Where ``nucleation_scheme=2``'s HAM inputs come from (``jcm.data.ham_inputs``)."""

from pathlib import Path

import pytest

from jcm.data import ham_inputs
from jcm.data.ham_inputs import HAM_INPUT_FILES, ham_input_dir


def test_ham_input_dir_env_wins_and_skips_the_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("HAM_INPUT_DIR", str(tmp_path))

    def fetch(rel):
        raise AssertionError("HAM_INPUT_DIR set: the mirror must not be read")

    assert ham_input_dir(fetch=fetch) == tmp_path


def test_without_env_the_mirror_files_share_one_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("HAM_INPUT_DIR", raising=False)
    fetched = []

    def fetch(rel):
        fetched.append(rel)
        return str(tmp_path / "snapshot" / rel)

    assert ham_input_dir(fetch=fetch) == tmp_path / "snapshot/products/ham_input"
    assert sorted(fetched) == sorted(f"products/ham_input/{n}"
                                     for n in HAM_INPUT_FILES)


def test_the_default_fetch_is_the_mirror_fetch(tmp_path, monkeypatch):
    monkeypatch.delenv("HAM_INPUT_DIR", raising=False)
    monkeypatch.setattr("jcm.data.remote.fetch",
                        lambda rel: str(tmp_path / rel))
    assert ham_input_dir() == tmp_path / "products/ham_input"


def test_neither_route_names_both(monkeypatch):
    monkeypatch.delenv("HAM_INPUT_DIR", raising=False)

    def fetch(rel):
        raise OSError("offline")

    with pytest.raises(FileNotFoundError,
                       match="HAM_INPUT_DIR.*products/ham_input.*offline"):
        ham_input_dir(fetch=fetch)


def test_files_split_across_revisions_are_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("HAM_INPUT_DIR", raising=False)
    names = iter(("a", "b", "c"))

    def fetch(rel):
        return str(tmp_path / next(names) / Path(rel).name)

    with pytest.raises(FileNotFoundError, match="one mirror revision"):
        ham_input_dir(fetch=fetch)


def test_every_published_name_is_one_its_reader_accepts():
    from jcm.physics.aerosol.jam.chemistry import gcr_ionisation

    published = list(HAM_INPUT_FILES)
    assert published[0] in ham_inputs.KAZIL_TABLE_NAMES
    assert published[1] in gcr_ionisation._SOLMIN_NAMES
    assert published[2] in gcr_ionisation._SOLMAX_NAMES
    assert ham_inputs.HAM_INPUT_PRODUCT == "ham_input"
