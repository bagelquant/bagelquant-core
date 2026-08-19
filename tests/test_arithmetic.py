from __future__ import annotations

from bagelquant_core.composer import div

from helpers import panel, values


def test_division_by_zero_is_missing_instead_of_infinite() -> None:
    numerator = panel(
        [
            ("2024-01-01", "a", 6.0),
            ("2024-01-01", "b", 2.0),
            ("2024-01-01", "c", -3.0),
        ],
        name="numerator",
    )
    denominator = panel(
        [
            ("2024-01-01", "a", 3.0),
            ("2024-01-01", "b", 0.0),
            ("2024-01-01", "c", -0.0),
        ],
        name="denominator",
    )

    result = div(numerator, denominator)
    result.compute()

    expected = [
        2.0,
        None,
        None,
    ]
    assert list(values(result.output.collect(dense=True)).values()) == expected
    eager = div.operation(
        numerator.collect(dense=False),
        denominator.collect(dense=False),
    )
    assert list(values(eager).values()) == expected
