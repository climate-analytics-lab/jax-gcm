"""Tests for the CI slow-suite shard definition."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slow_shards  # noqa: E402


def test_rest_is_the_complement_of_radiation():
    rest = slow_shards.pytest_args("rest")
    assert rest == [f"--ignore={p}" for p in slow_shards.RADIATION_PATHS]
    assert slow_shards.pytest_args("radiation") == list(
        slow_shards.RADIATION_PATHS)


def test_listed_paths_exist():
    root = Path(__file__).resolve().parents[2]
    for path in slow_shards.RADIATION_PATHS:
        assert (root / path).exists(), path


def test_partition_errors_names_each_failure():
    whole = {"a::1", "b::1", "c::1"}
    assert slow_shards.partition_errors(
        whole, {"radiation": {"a::1"}, "rest": {"b::1", "c::1"}}) == []
    errors = slow_shards.partition_errors(
        whole, {"radiation": {"a::1", "b::1"}, "rest": {"b::1"}})
    assert any("both" in e for e in errors)
    assert any("no shard" in e for e in errors)
    assert any("no slow tests" in e for e in slow_shards.partition_errors(
        whole, {"radiation": set(), "rest": whole}))


def test_unknown_shard_is_rejected():
    with pytest.raises(ValueError):
        slow_shards.pytest_args("bogus")
