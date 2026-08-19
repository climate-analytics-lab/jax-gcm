"""Tests for the unified paired-ARMBE cache split policy."""

import numpy as np
import pytest

from unified_cache import chronological_splits, randomized_month_block_splits


def test_chronological_splits_follow_timestamp_order_and_sample_count():
    times = np.array(
        ["2020-01-01", "2020-04-01", "2020-08-01", "2020-10-01", "2020-12-31"],
        dtype="datetime64[ns]",
    )

    split = chronological_splits(times)

    np.testing.assert_array_equal(split, ["train", "train", "train", "validation", "test"])


def test_chronological_splits_reject_record_without_every_split():
    with pytest.raises(ValueError, match="at least three timestamps"):
        chronological_splits(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"))


def test_random_month_blocks_are_deterministic_and_never_split_a_month():
    times = np.array(
        [
            "2018-01-01", "2018-01-02", "2019-01-01", "2019-01-02", "2020-01-01",
            "2020-01-02", "2021-01-01", "2021-01-02", "2022-01-01", "2022-01-02",
        ],
        dtype="datetime64[ns]",
    )

    first = randomized_month_block_splits(times, seed=17)
    second = randomized_month_block_splits(times, seed=17)

    np.testing.assert_array_equal(first, second)
    for block in np.unique(times.astype("datetime64[M]")):
        assert len(set(first[times.astype("datetime64[M]") == block])) == 1
    assert set(first) == {"train", "validation", "test"}
