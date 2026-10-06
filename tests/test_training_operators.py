from datetime import date, timedelta
import polars as pl
import pytest
from bagelquant_core import (
    Domain,
    Node,
    rolling_elastic_net_prediction,
    exposure_constrained_weights,
    capture_training_audits,
    date_balanced_training_keys,
)
from bagelquant_core import capture_operator_checkpoints
from polars.testing import assert_frame_equal


def test_balanced_sampling_is_order_independent_and_redistributes_sparse_dates():
    days = [date(2024, 1, i) for i in range(1, 5)]
    keys = pl.DataFrame({"time": [day for day, size in zip(days, [1, 9, 9, 9]) for _ in range(size)],
                         "asset_id": [str(i) for size in [1, 9, 9, 9] for i in range(size)]})
    sample = date_balanced_training_keys(keys, budget=13)
    assert_frame_equal(sample, date_balanced_training_keys(keys.reverse(), budget=13))
    assert sample.group_by("time").len().sort("time")["len"].to_list() == [1, 4, 4, 4]
    assert sample.height == 13
    assert sample.filter(pl.col("time") == days[1])["asset_id"].to_list() != ["5", "6", "7", "8"]
    assert not sample.equals(date_balanced_training_keys(keys, budget=13, seed=99))


def test_training_audit_and_row_label_boundaries_are_causal():
    frame, domain = panels()
    late = frame.with_columns(pl.lit(date(2024, 1, 10).toordinal()).alias("value"))
    with capture_training_audits() as audits:
        rolling_elastic_net_prediction(
            Node.from_domain(frame, domain), labels=Node.from_domain(frame, domain),
            label_available=Node.from_domain(late, domain), label_end=Node.from_domain(late, domain),
            window=5, fit_every=2, min_samples=4, max_samples=7, label_maturity=2,
        ).compute().collect()
    assert all(row["sample_count"] == 0 for row in audits if row["fit_date"] < date(2024, 1, 10))
    fitted = [row for row in audits if row["status"] == "fitted"]
    assert fitted
    assert all(row["maximum_label_available"] <= row["fit_date"] for row in fitted)
    assert all(sum(day["count"] for day in row["daily_counts"]) == row["sample_count"] for row in audits)
    assert all(len(row["training_key_hash"]) == 64 and row["seed"] == 1729 for row in audits)


def panels():
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(12)]
    rows = [
        {"time": day, "asset_id": asset, "value": float(i + j + 1)}
        for i, day in enumerate(days)
        for j, asset in enumerate("ABCD")
    ]
    frame = pl.DataFrame(rows)
    domain = Domain(calendar=days, universe=list("ABCD"))
    return frame, domain


def test_training_cannot_observe_unmatured_or_future_labels():
    frame, domain = panels()
    source = Node.from_domain(frame, domain)
    labels = frame.with_columns((pl.col("value") * 2).alias("value"))

    def compute(label):
        with capture_training_audits() as audit:
            output = (
                rolling_elastic_net_prediction(
                    source,
                    labels=Node.from_domain(label, domain),
                    window=5,
                    fit_every=2,
                    min_samples=4,
                    max_samples=20,
                    label_maturity=3,
                )
                .compute()
                .collect()
            )
        return output, audit

    first, audits = compute(labels)
    modified = labels.with_columns(
        pl.when(pl.col("time") >= date(2024, 1, 8))
        .then(pl.lit(999.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    second, _ = compute(modified)
    assert first.filter(pl.col("time") < date(2024, 1, 11)).equals(
        second.filter(pl.col("time") < date(2024, 1, 11))
    )
    assert all(
        audit["training_end"] is None
        or audit["training_end"] <= audit["fit_date"] - timedelta(days=3)
        for audit in audits
    )
    assert first["value"].null_count() > 0


def test_exposure_constraints_and_missing_coordinates_are_explicit():
    frame, domain = panels()
    frame = frame.filter(pl.col("time") == date(2024, 1, 2))
    domain = Domain(calendar=[date(2024, 1, 2)], universe=list("ABCD"))
    source = Node.from_domain(frame, domain)
    exposure = frame.with_columns(
        pl.col("asset_id")
        .replace_strict({"A": -1.0, "B": -1.0, "C": 1.0, "D": 1.0})
        .alias("value")
    )

    def operator(panel):
        return exposure_constrained_weights(
            source,
            exposures=(panel,),
            lower_bounds=[-0.1],
            upper_bounds=[0.1],
            max_weight=0.5,
        )

    result = operator(Node.from_domain(exposure, domain)).compute().collect()
    actual = (
        result.join(exposure.rename({"value": "exposure"}), on=["time", "asset_id"])
        .select((pl.col("value") * pl.col("exposure")).sum())
        .item()
    )
    assert abs(actual) <= 0.1000001
    with pytest.raises(ValueError, match="required exposure missing"):
        operator(
            Node.from_domain(exposure.filter(pl.col("asset_id") != "A"), domain)
        ).compute().collect()


def test_model_checkpoint_continuation_matches_full_and_does_not_refit_prefix():
    frame, domain = panels()
    days = list(domain.times)

    def compute(selected, restored=None, offset=0):
        selected_domain = Domain(
            calendar=selected["time"].unique().sort(), universe=list("ABCD")
        )
        source = Node.from_domain(selected, selected_domain, source_key="features")
        labels = Node.from_domain(
            selected.with_columns((pl.col("value") * 2).alias("value")), selected_domain, source_key="labels"
        )
        with (
            capture_operator_checkpoints(restored) as checkpoints,
            capture_training_audits() as audits,
        ):
            result = (
                rolling_elastic_net_prediction(
                    source,
                    labels=labels,
                    window=5,
                    fit_every=2,
                    min_samples=4,
                    max_samples=20,
                    label_maturity=3,
                    anchor_offset=offset,
                    name="forecast",
                )
                .compute()
                .collect()
            )
        return result, checkpoints.captured, audits

    full, _, _ = compute(frame)
    prefix, states, _ = compute(frame.filter(pl.col("time") <= days[6]))
    suffix, _, audits = compute(frame.filter(pl.col("time") >= days[1]), states, 1)
    joined = pl.concat([prefix, suffix.filter(pl.col("time") > days[6])])
    assert_frame_equal(joined, full, check_exact=False, abs_tol=1e-12, rel_tol=1e-12)
    assert all(audit["fit_date"] > days[6] for audit in audits)
    corrupted = {name: {**value, "signature": {}} for name, value in states.items()}
    with pytest.raises(ValueError, match="signature changed"):
        compute(frame.filter(pl.col("time") >= days[1]), corrupted, 1)


def test_training_phase_and_label_maturity_include_empty_universe_sessions():
    frame, original = panels()
    days = list(original.times)
    frame = frame.filter(~pl.col("time").is_in([days[0], days[1], days[5]]))

    def compute(first, last, restored=None):
        calendar = days[first:last]
        local = frame.filter(pl.col("time").is_in(calendar))
        domain = Domain(
            calendar=calendar,
            universe=local.select("time", "asset_id").with_columns(
                pl.lit(True).alias("active")
            ),
        )
        source = Node.from_domain(local, domain, source_key="features")
        labels = Node.from_domain(
            local.with_columns((pl.col("value") * 2).alias("value")), domain, source_key="labels"
        )
        with (
            capture_operator_checkpoints(restored, calendar=calendar) as checkpoints,
            capture_training_audits() as audits,
        ):
            result = (
                rolling_elastic_net_prediction(
                    source,
                    labels=labels,
                    name="forecast",
                    window=5,
                    fit_every=4,
                    min_samples=4,
                    max_samples=20,
                    label_maturity=2,
                    anchor_offset=first,
                )
                .compute()
                .collect()
            )
        return result, checkpoints.captured, audits

    full, _, audits = compute(0, len(days))
    assert [audit["fit_date"] for audit in audits] == [days[0], days[4], days[8]]
    assert audits[1]["training_end"] == days[2]
    prefix, state, _ = compute(0, 7)
    suffix, _, resumed_audits = compute(1, len(days), state)
    assert_frame_equal(
        pl.concat([prefix, suffix.filter(pl.col("time") > days[6])]),
        full,
        check_exact=False,
        abs_tol=1e-12,
        rel_tol=1e-12,
    )
    assert [audit["fit_date"] for audit in resumed_audits] == [days[8]]
