import pandas as pd
import pytest

from bagelquant_core import Panel


def test_panel_rejects_non_numeric_data() -> None:
    data = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(TypeError):
        Panel("bad", data)


def test_panel_align_inner() -> None:
    left = pd.DataFrame(
        {"a": [1, 2], "b": [3, 4]},
        index=pd.Index([1, 2], name="date"),
    )
    right = pd.DataFrame(
        {"b": [5, 6], "c": [7, 8]},
        index=pd.Index([2, 3], name="date"),
    )

    aligned_left, aligned_right = Panel.align_frames(left, right, join="inner")

    assert list(aligned_left.index) == [2]
    assert list(aligned_left.columns) == ["b"]
    assert list(aligned_right.index) == [2]
    assert list(aligned_right.columns) == ["b"]
