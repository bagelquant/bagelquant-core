import pandas as pd
import pytest

from bagelquant_core import Panel
from bagelquant_core.composer import add, weighted_sum


def test_panel_rejects_non_numeric_data() -> None:
    data = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(TypeError):
        Panel(data, name="bad")


def test_panel_data_is_a_defensive_copy() -> None:
    panel = Panel(pd.DataFrame({"a": [1, 2]}), name="panel")

    data = panel.data
    data.loc[0, "a"] = 99

    assert panel.data.loc[0, "a"] == 1


def test_composer_aligns_panel_inputs_on_intersection() -> None:
    left = Panel(
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[1, 2]),
        name="left",
    )
    right = Panel(
        pd.DataFrame({"b": [5, 6], "c": [7, 8]}, index=[2, 3]),
        name="right",
    )

    result = add(left, right).compute()

    assert list(result.data.index) == [2]
    assert list(result.data.columns) == ["b"]
    assert result.data.loc[2, "b"] == 9


def test_weighted_sum_combines_frames_without_mutating_inputs() -> None:
    left = Panel(pd.DataFrame({"a": [1.0, 2.0]}), name="left")
    right = Panel(pd.DataFrame({"a": [3.0, 4.0]}), name="right")

    result = weighted_sum(left, right, weights=[0.25, 0.75]).compute()

    assert result.data["a"].tolist() == [2.5, 3.5]
    assert left.data["a"].tolist() == [1.0, 2.0]
