import pandas as pd
import pytest

from bagelquant_core import Panel
from bagelquant_core.composer import (
    add,
    div,
    maximum,
    mean,
    minimum,
    mul,
    product,
    sub,
    sum_frames,
    weighted_mean,
    weighted_sum,
)


def panel(values: list[float], *, name: str) -> Panel:
    return Panel(pd.DataFrame({"a": values}), name=name)


def test_arithmetic_composers() -> None:
    left = panel([6.0, 8.0], name="left")
    right = panel([2.0, 4.0], name="right")

    assert add(left, right).compute().data["a"].tolist() == [8.0, 12.0]
    assert sub(left, right).compute().data["a"].tolist() == [4.0, 4.0]
    assert mul(left, right).compute().data["a"].tolist() == [12.0, 32.0]
    assert div(left, right).compute().data["a"].tolist() == [3.0, 2.0]


def test_aggregation_composers() -> None:
    first = panel([1.0, 4.0], name="first")
    second = panel([2.0, 3.0], name="second")
    third = panel([3.0, 2.0], name="third")

    assert sum_frames(first, second, third).compute().data["a"].tolist() == [
        6.0,
        9.0,
    ]
    assert mean(first, second, third).compute().data["a"].tolist() == [2.0, 3.0]
    assert product(first, second, third).compute().data["a"].tolist() == [
        6.0,
        24.0,
    ]
    assert minimum(first, second, third).compute().data["a"].tolist() == [
        1.0,
        2.0,
    ]
    assert maximum(first, second, third).compute().data["a"].tolist() == [
        3.0,
        4.0,
    ]


def test_weighted_composers() -> None:
    left = panel([1.0, 2.0], name="left")
    right = panel([3.0, 4.0], name="right")

    total = weighted_sum(left, right, weights=[1.0, 3.0])
    average = weighted_mean(left, right, weights=[1.0, 3.0])

    assert total.compute().data["a"].tolist() == [10.0, 14.0]
    assert average.compute().data["a"].tolist() == [2.5, 3.5]
    assert average.spec().nodes[-1].config["weights"] == [1.0, 3.0]


def test_aggregation_composers_do_not_mutate_inputs() -> None:
    left = panel([1.0, 2.0], name="left")
    right = panel([3.0, 4.0], name="right")

    sum_frames(left, right).compute()
    product(left, right).compute()

    assert left.data["a"].tolist() == [1.0, 2.0]
    assert right.data["a"].tolist() == [3.0, 4.0]


def test_weighted_composers_require_one_numeric_weight_per_frame() -> None:
    left = panel([1.0], name="left")
    right = panel([2.0], name="right")

    with pytest.raises(ValueError, match="one weight per frame"):
        weighted_sum(left, right, weights=[1.0]).compute()

    with pytest.raises(TypeError, match="real numbers"):
        weighted_sum(left, right, weights=[1.0, "invalid"]).compute()


def test_weighted_mean_rejects_zero_total_weight() -> None:
    left = panel([1.0], name="left")
    right = panel([2.0], name="right")

    with pytest.raises(ValueError, match="must not sum to zero"):
        weighted_mean(left, right, weights=[-1.0, 1.0]).compute()
