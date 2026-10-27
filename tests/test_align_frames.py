"""Public alignment retains the union of sparse coordinates and their values."""
from datetime import date

import polars as pl
import pytest

from bagelquant_core import Node


def panel(rows):
    return pl.DataFrame(rows, schema={"time": pl.Date, "asset_id": pl.String,
        "value": pl.Float64}, orient="row")


@pytest.mark.parametrize("empty_first", [False, True])
def test_outer_alignment_preserves_disjoint_and_overlapping_coordinates(empty_first):
    first = panel([] if empty_first else [(date(2020, 1, 1), "A", 1.0)])
    second = panel([(date(2020, 1, 2), "B", 2.0)])
    third = panel([(date(2020, 1, 1), "A", 3.0), (date(2020, 1, 3), "C", 4.0)])
    aligned = Node.align_frames(first, second, third, join="outer")
    coordinates = [(date(2020, 1, 1), "A"), (date(2020, 1, 2), "B"), (date(2020, 1, 3), "C")]
    for actual, values in zip(aligned,
            ([None if empty_first else 1.0, None, None], [None, 2.0, None], [3.0, None, 4.0])):
        assert actual.columns == ["time", "asset_id", "value"]
        assert actual.select("time", "asset_id").rows() == coordinates
        assert actual["value"].to_list() == values


def test_inner_alignment_keeps_only_common_coordinates_and_empty_union_stays_typed():
    first = panel([(date(2020, 1, 1), "A", 1.0), (date(2020, 1, 2), "B", 2.0)])
    second = panel([(date(2020, 1, 2), "B", 3.0)])
    aligned = Node.align_frames(first, second)
    assert [frame.rows() for frame in aligned] == [
        [(date(2020, 1, 2), "B", 2.0)], [(date(2020, 1, 2), "B", 3.0)]]
    empty = panel([])
    assert all(frame.schema == empty.schema and frame.is_empty()
        for frame in Node.align_frames(empty, empty, join="outer"))
