from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from bagelquant_core import (
    ElasticNetConfig,
    ElasticNetModel,
    ElasticNetPredictionComposer,
    LabelBoundary,
    WeightedRegressionMoments,
    WalkForwardConfig,
    ZeroPreservingRmsScaler,
    build_expanding_walk_forward,
    elastic_net_candidates,
    elastic_net_candidates_from_moments,
    equal_period_sample_weights,
    fit_elastic_net,
    fit_elastic_net_from_moments,
    zero_preserving_rms_scaler_from_moments,
)


def _months(count: int) -> list[date]:
    return [date(2000 + index // 12, index % 12 + 1, 1) for index in range(count)]


def test_expanding_walk_forward_is_monthly_and_embargoes_sample_tail() -> None:
    folds = build_expanding_walk_forward(
        _months(36),
        WalkForwardConfig(
            initial_sample_months=12,
            validation_months=6,
            oos_months=6,
            embargo_months=2,
        ),
    )

    assert len(folds) == 3
    assert len(folds[0].sample_periods) == 12
    assert len(folds[0].training_periods) == 10
    assert folds[0].embargoed_periods == tuple(_months(12)[-2:])
    assert len(folds[1].sample_periods) == 18
    assert folds[0].oos_periods[-1] < folds[1].oos_periods[0]


def test_expanding_walk_forward_can_publish_a_partial_trailing_oos_fold() -> None:
    config = WalkForwardConfig(
        initial_sample_months=12,
        validation_months=6,
        oos_months=6,
    )

    assert build_expanding_walk_forward(_months(19), config) == ()
    folds = build_expanding_walk_forward(
        _months(19),
        config,
        include_incomplete_oos=True,
    )

    assert len(folds) == 1
    assert folds[0].oos_periods == (date(2001, 7, 1),)
    completed = build_expanding_walk_forward(
        _months(24),
        config,
        include_incomplete_oos=True,
    )
    assert completed[0].sample_periods == folds[0].sample_periods
    assert completed[0].validation_periods == folds[0].validation_periods
    assert completed[0].oos_periods[:1] == folds[0].oos_periods


def test_validation_month_floor_cannot_be_weakened() -> None:
    assert (
        WalkForwardConfig(validation_months=7).effective_minimum_valid_validation_months
        == 6
    )
    with pytest.raises(ValueError, match="75%"):
        WalkForwardConfig(validation_months=24, minimum_valid_validation_months=17)


def test_label_boundary_requires_strict_availability_before_fit() -> None:
    valid = LabelBoundary(
        feature_date=date(2024, 1, 30),
        prediction_date=date(2024, 1, 31),
        execution_date=date(2024, 2, 1),
        label_start_date=date(2024, 2, 1),
        label_end_date=date(2024, 3, 1),
        label_available_date=date(2024, 3, 1),
        model_fit_cutoff=date(2024, 3, 2),
    )
    valid.validate()

    with pytest.raises(ValueError, match="strictly before"):
        LabelBoundary(
            feature_date=valid.feature_date,
            prediction_date=valid.prediction_date,
            execution_date=valid.execution_date,
            label_start_date=valid.label_start_date,
            label_end_date=valid.label_end_date,
            label_available_date=valid.label_available_date,
            model_fit_cutoff=valid.label_available_date,
        ).validate()


def test_rms_scaler_preserves_gated_zeros_and_drops_constant_zero_column() -> None:
    sample = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scaler = ZeroPreservingRmsScaler.fit(
        sample,
        feature_names=("alpha", "out_of_scope", "missing::alpha"),
        sample_weight=np.array([0.5, 1.0, 0.5]),
    )

    transformed = scaler.transform(sample)
    assert scaler.active_feature_names == ("alpha", "missing::alpha")
    assert scaler.dropped_columns == {"out_of_scope": "zero_rms"}
    assert transformed[0, 0] == 0.0
    assert transformed[1, 1] == 0.0
    assert np.isfinite(transformed).all()
    restored = ZeroPreservingRmsScaler.from_dict(scaler.to_dict())
    assert restored == scaler
    assert np.array_equal(restored.transform(sample), transformed)


def test_period_weights_assign_equal_total_weight_to_each_month() -> None:
    weights = equal_period_sample_weights(["jan", "jan", "feb", "feb", "feb"])

    assert weights[:2].sum() == pytest.approx(1.0)
    assert weights[2:].sum() == pytest.approx(1.0)


def test_relative_alpha_path_is_recomputed_from_fold_target_scale() -> None:
    values = np.array([[-1.0], [0.0], [1.0]])
    config = ElasticNetConfig(alpha_ratios=(1.0, 0.1), l1_ratio_values=(0.5,))

    first = elastic_net_candidates(config, values, np.array([-1.0, 0.0, 1.0]))
    second = elastic_net_candidates(config, values, np.array([-2.0, 0.0, 2.0]))

    assert [item.alpha_ratio for item in first] == [1.0, 0.1]
    assert second[0].alpha == pytest.approx(first[0].alpha * 2.0)


def test_elastic_net_uses_unpenalized_intercept_without_centering_features() -> None:
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = 4.0 + 2.0 * values[:, 0]

    model = fit_elastic_net(
        values,
        target,
        alpha=0.0,
        l1_ratio=0.5,
        tolerance=1e-10,
    )

    assert model.converged
    assert model.intercept == pytest.approx(4.0, abs=1e-7)
    assert model.coefficients == pytest.approx((2.0,), abs=1e-7)
    assert model.predict(np.array([[0.0]])).item() == pytest.approx(4.0, abs=1e-7)
    restored = ElasticNetModel.from_dict(model.to_dict())
    assert restored == model
    assert np.array_equal(restored.predict(values), model.predict(values))


def test_weighted_moment_elastic_net_matches_row_matrix_fit_across_batches() -> None:
    values = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [2.0, 0.5, 0.0],
            [3.0, 4.0, 1.0],
            [4.0, 2.5, 0.0],
            [5.0, 6.0, 1.0],
        ]
    )
    target = 3.0 + values @ np.array([0.8, -0.3, 0.5])
    weights = np.array([0.5, 0.5, 1.0, 0.25, 0.25, 1.0])
    names = ("first", "second", "missing::first")
    scaler = ZeroPreservingRmsScaler.fit(
        values, feature_names=names, sample_weight=weights
    )
    scaled = scaler.transform(values)
    config = ElasticNetConfig(alpha_ratios=(0.1,), l1_ratio_values=(0.5,))
    candidate = elastic_net_candidates(
        config, scaled, target, sample_weight=weights
    )[0]
    expected = fit_elastic_net(
        scaled,
        target,
        alpha=candidate.alpha,
        l1_ratio=candidate.l1_ratio,
        sample_weight=weights,
        tolerance=1e-10,
    )

    moments = WeightedRegressionMoments.empty(values.shape[1])
    moments.update(values[:2], target[:2], sample_weight=weights[:2])
    moments.update(values[2:], target[2:], sample_weight=weights[2:])
    moment_scaler = zero_preserving_rms_scaler_from_moments(
        moments, feature_names=names
    )
    moment_candidate = elastic_net_candidates_from_moments(
        config, moments, moment_scaler
    )[0]
    actual = fit_elastic_net_from_moments(
        moments,
        moment_scaler,
        alpha=moment_candidate.alpha,
        l1_ratio=moment_candidate.l1_ratio,
        tolerance=1e-10,
    )

    assert moment_scaler.scales == pytest.approx(scaler.scales)
    assert moment_candidate.alpha == pytest.approx(candidate.alpha)
    assert actual.intercept == pytest.approx(expected.intercept, abs=1e-8)
    assert actual.coefficients == pytest.approx(expected.coefficients, abs=1e-8)
    assert actual.predict(moment_scaler.transform(values)) == pytest.approx(
        expected.predict(scaled), abs=1e-8
    )


def test_moment_weight_rescaling_matches_direct_weights_and_preserves_snapshot() -> None:
    values = np.array([[1., 2.], [3., 1.], [2., 4.], [5., 3.]])
    target = np.array([0.1, 0.3, -0.2, 0.5])
    actual = WeightedRegressionMoments.empty(2)
    actual.update(values[:2], target[:2], sample_weight=np.array([0.5, 1.]))
    saved = actual.copy()
    actual.scale_weights(0.25)
    actual.update(values[2:], target[2:], sample_weight=np.array([0.5, 1.]))
    expected = WeightedRegressionMoments.empty(2)
    expected.update(values, target, sample_weight=np.array([0.125, 0.25, 0.5, 1.]))
    assert actual.observation_count == expected.observation_count == 4
    assert actual.weight_sum == pytest.approx(expected.weight_sum)
    assert actual.target_sum == pytest.approx(expected.target_sum)
    np.testing.assert_allclose(actual.feature_sum, expected.feature_sum)
    np.testing.assert_allclose(actual.feature_cross, expected.feature_cross)
    np.testing.assert_allclose(actual.feature_target, expected.feature_target)
    assert saved.observation_count == 2
    assert saved.weight_sum == 1.5


@pytest.mark.parametrize("factor", [0., -1., float("nan"), float("inf")])
def test_moment_weight_rescaling_rejects_invalid_scale(factor: float) -> None:
    moments = WeightedRegressionMoments.empty(1)
    with pytest.raises(ValueError, match="positive and finite"):
        moments.scale_weights(factor)


def test_elastic_net_composer_round_trips_complete_configuration() -> None:
    composer = ElasticNetPredictionComposer(
        walk_forward=WalkForwardConfig(),
        coverage={
            "minimum_all_market_observations": 2,
            "minimum_applicable_coverage": 0.7,
            "required_marker_unknown_policy": "reject_if_required_marker_unknown",
        },
        target={
            "uuid": "target-1",
            "revision": 2,
            "revision_hash": "a" * 64,
            "definition": {"frequency": "monthly"},
        },
        elastic_net=ElasticNetConfig(),
        validation={"objective": "mean_ic", "minimum_valid_months": 18},
    )

    restored = ElasticNetPredictionComposer.from_dict(composer.to_dict())

    assert restored.to_dict() == composer.to_dict()
    assert restored.to_dict()["scaling"]["method"] == (
        "weighted_rms_without_centering"
    )
