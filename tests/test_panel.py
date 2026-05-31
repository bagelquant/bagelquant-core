import pandas as pd
import pytest

from bagelquant_core import Panel
from bagelquant_core.composer import add


def test_panel_rejects_non_numeric_data() -> None:
    data = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(TypeError):
        Panel(data, name="bad")


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
