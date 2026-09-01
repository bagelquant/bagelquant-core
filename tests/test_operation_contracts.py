from __future__ import annotations

import inspect
import math

import pytest

from bagelquant_core.composer import (
    COMPOSER_REGISTRY,
    weighted_sum,
    xand,
)
from bagelquant_core.transformer import (
    TRANSFORMER_REGISTRY,
    anscombe,
    boxcox,
    date_age_constraint,
    ffill,
    group_percentile,
    group_rankpct,
    inv_log_sqrt_rank,
    fillna_zero,
    log,
    log1p,
    log_rank,
    negonly,
    net_scale,
    non_nan_to_one,
    non_nan_to_zero,
    normalize,
    notnan,
    posonly,
    mask,
    orthogonalize,
    project,
    rankpct,
    rolling_kurt,
    rolling_ols,
    rolling_percentile,
    rolling_rank,
    rolling_skew,
    sign,
    trig,
)

from helpers import panel, values


def test_registered_operations_expose_reference_docstrings() -> None:
    for registry in (TRANSFORMER_REGISTRY, COMPOSER_REGISTRY):
        for name in registry.names():
            documentation = inspect.getdoc(registry.get(name).operation)
            assert documentation, name
            assert documentation.endswith("."), name


def _single_time(values_: list[float | None], *, name: str = "x"):
    return panel(
        [
            ("2024-01-01", chr(ord("a") + index), value)
            for index, value in enumerate(values_)
        ],
        name=name,
    )


def test_missing_value_and_sign_filters_keep_distinct_contracts() -> None:
    source = _single_time([None, float("nan"), -1.0, 0.0, 2.0])

    filled = fillna_zero(source)
    present = notnan(source)
    positive = posonly(source)
    negative = negonly(source)
    signed = sign(source)
    for graph in (filled, present, positive, negative, signed):
        graph.compute()

    assert list(values(filled.output.collect(dense=True)).values()) == [0.0, 0.0, -1.0, 0.0, 2.0]
    assert list(values(present.output.collect(dense=True)).values()) == [0.0, 0.0, 1.0, 1.0, 1.0]
    positive_values = list(values(positive.output.collect(dense=True)).values())
    negative_values = list(values(negative.output.collect(dense=True)).values())
    assert positive_values[0] is None
    assert positive_values[1] is None
    assert positive_values[2:] == [None, 0.0, 2.0]
    assert negative_values[0] is None
    assert negative_values[1] is None
    assert negative_values[2:] == [-1.0, 0.0, None]
    assert list(values(signed.output.collect(dense=True)).values()) == [
        None,
        None,
        -1.0,
        0.0,
        1.0,
    ]


def test_missing_values_are_canonical_and_forward_fill_is_bounded() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", float("nan")),
            ("2024-01-03", "a", None),
            ("2024-01-04", "a", 4.0),
        ]
    )

    graph = ffill(source, limit=1)
    graph.compute()

    assert list(values(graph.output.collect(dense=True)).values()) == [1.0, 1.0, None, 4.0]


def test_date_age_constraint_uses_trailing_valid_observations() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", None),
            ("2024-01-03", "a", 3.0),
            ("2024-01-04", "a", 4.0),
            ("2024-01-05", "a", None),
        ]
    )

    graph = date_age_constraint(source, window=3, min_valid=2)
    graph.compute()

    assert list(values(graph.output.collect(dense=True)).values()) == [
        None,
        None,
        3.0,
        4.0,
        None,
    ]


def test_cross_sectional_scaling_and_rank_contracts() -> None:
    source = _single_time([-2.0, -1.0, 1.0, 1.0, 3.0])

    normalized = normalize(source)
    net = net_scale(source)
    dense = rankpct(source)
    average_log = log_rank(source)
    inverse = inv_log_sqrt_rank(source)
    for graph in (normalized, net, dense, average_log, inverse):
        graph.compute()

    assert list(values(normalized.output.collect(dense=True)).values()) == pytest.approx([
        -1.0,
        -0.6,
        0.2,
        0.2,
        1.0,
    ])
    assert list(values(net.output.collect(dense=True)).values()) == pytest.approx(
        [-2 / 3, -1 / 3, 1 / 5, 1 / 5, 3 / 5]
    )
    assert list(values(dense.output.collect(dense=True)).values()) == pytest.approx(
        [0.25, 0.5, 0.75, 0.75, 1.0]
    )
    assert list(values(average_log.output.collect(dense=True)).values()) == pytest.approx(
        [math.log(0.2), math.log(0.4), math.log(0.7), math.log(0.7), 0.0]
    )
    assert values(inverse.output.collect(dense=True))[("2024-01-01", "e")] == 0.0


def test_log_boxcox_trig_and_anscombe_respect_math_domains() -> None:
    source = _single_time([-2.0, -1.0, 0.0, 1.0])
    logged = log(source)
    logged1p = log1p(source)
    boxed = boxcox(source)
    trigged = trig(source)
    stabilized = anscombe(source)
    for graph in (logged, logged1p, boxed, trigged, stabilized):
        graph.compute()

    assert list(values(logged.output.collect(dense=True)).values())[:3] == [None, None, None]
    assert list(values(logged1p.output.collect(dense=True)).values())[:2] == [None, None]
    assert list(values(boxed.output.collect(dense=True)).values())[:3] == [None, None, None]
    assert values(trigged.output.collect(dense=True))[("2024-01-01", "a")] is None
    assert values(stabilized.output.collect(dense=True))[("2024-01-01", "a")] == pytest.approx(
        2 * math.sqrt(3 / 8)
    )


def test_xand_is_logical_equivalence() -> None:
    lhs = _single_time([0.0, 0.0, 1.0, 1.0], name="lhs")
    rhs = _single_time([0.0, 1.0, 0.0, 1.0], name="rhs")

    graph = xand(lhs, rhs)
    graph.compute()

    assert list(values(graph.output.collect(dense=True)).values()) == [1.0, 0.0, 0.0, 1.0]


def test_project_mask_and_replace_helpers_keep_their_public_semantics() -> None:
    source = _single_time([1.0, 2.0, 3.0], name="source")
    selector = _single_time([1.0, 2.0, 0.0], name="selector")

    projected = project(source, binary=selector)
    masked = mask(source, mask_frame=selector, replace_value=-1.0)
    ones = non_nan_to_one(source)
    zeros = non_nan_to_zero(source)
    for graph in (projected, masked, ones, zeros):
        graph.compute()

    assert list(values(projected.output.collect(dense=True)).values()) == [1.0, None, None]
    assert list(values(masked.output.collect(dense=True)).values()) == [1.0, 2.0, -1.0]
    assert list(values(ones.output.collect(dense=True)).values()) == [1.0, 1.0, 1.0]
    assert list(values(zeros.output.collect(dense=True)).values()) == [0.0, 0.0, 0.0]


def test_group_dense_rank_and_average_percentile_are_distinct() -> None:
    source = _single_time([1.0, 1.0, 2.0, 3.0], name="x")
    groups = _single_time([1.0, 1.0, 1.0, 1.0], name="group")

    dense = group_rankpct(source, group=groups)
    average = group_percentile(source, group=groups)
    dense.compute()
    average.compute()

    assert list(values(dense.output.collect(dense=True)).values()) == pytest.approx(
        [1 / 3, 1 / 3, 2 / 3, 1.0]
    )
    assert list(values(average.output.collect(dense=True)).values()) == pytest.approx(
        [0.375, 0.375, 0.75, 1.0]
    )


def test_rolling_ols_supports_multiple_factors_and_excludes_current_target() -> None:
    target = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 4.0),
            ("2024-01-03", "a", 7.0),
            ("2024-01-04", "a", 10.0),
        ],
        name="target",
    )
    first = panel(
        [
            ("2024-01-01", "a", 0.0),
            ("2024-01-02", "a", 1.0),
            ("2024-01-03", "a", 2.0),
            ("2024-01-04", "a", 3.0),
        ],
        name="first",
    )
    second = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 1.0),
            ("2024-01-03", "a", 1.0),
            ("2024-01-04", "a", 1.0),
        ],
        name="second",
    )

    graph = rolling_ols(target, factors=(first, second), window=3)
    graph.compute()

    assert values(graph.output.collect(dense=True))[("2024-01-04", "a")] == pytest.approx(10.0)


def test_rolling_rank_ties_and_higher_moments_match_reference_formulas() -> None:
    source = panel(
        [
            ("2024-01-01", "a", 1.0),
            ("2024-01-02", "a", 2.0),
            ("2024-01-03", "a", 2.0),
            ("2024-01-04", "a", 8.0),
        ]
    )
    ranked = rolling_rank(source, window=3)
    percentile = rolling_percentile(source, window=3)
    skewed = rolling_skew(source, window=4)
    kurtosis = rolling_kurt(source, window=4)
    for graph in (ranked, percentile, skewed, kurtosis):
        graph.compute()

    assert values(ranked.output.collect(dense=True))[("2024-01-03", "a")] == 2.5
    assert values(percentile.output.collect(dense=True))[("2024-01-03", "a")] == pytest.approx(
        2.5 / 3
    )

    sample = [1.0, 2.0, 2.0, 8.0]
    mean = sum(sample) / len(sample)
    centered = [value - mean for value in sample]
    variance = sum(value**2 for value in centered) / (len(sample) - 1)
    std = math.sqrt(variance)
    expected_skew = (
        len(sample)
        / ((len(sample) - 1) * (len(sample) - 2))
        * sum((value / std) ** 3 for value in centered)
    )
    n = len(sample)
    fourth = sum((value / std) ** 4 for value in centered)
    expected_kurtosis = (
        n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * fourth
        - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    )
    assert values(skewed.output.collect(dense=True))[("2024-01-04", "a")] == pytest.approx(
        expected_skew
    )
    assert values(kurtosis.output.collect(dense=True))[("2024-01-04", "a")] == pytest.approx(
        expected_kurtosis
    )


def test_public_boundaries_reject_ambiguous_inputs() -> None:
    source = _single_time([1.0])

    with pytest.raises(ValueError, match="requires at least one Panel"):
        orthogonalize(source, factors=()).compute()
    with pytest.raises(ValueError, match="at least two"):
        weighted_sum(weights=[])
    with pytest.raises(TypeError, match="real numbers"):
        weighted_sum(source, source, weights=[True, 1.0]).compute()
    with pytest.raises(TypeError, match="real number"):
        boxcox(source, lambda_="bad").compute()
