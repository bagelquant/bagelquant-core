from __future__ import annotations

from bagelquant_core._documentation import OPERATION_DESCRIPTIONS, operation_category
from bagelquant_core.composer import COMPOSER_REGISTRY
from bagelquant_core.operation_examples import operation_example
from bagelquant_core.transformer import TRANSFORMER_REGISTRY


MOVED_TRANSFORMERS = {
    "group_demean",
    "group_max",
    "group_mean",
    "group_median",
    "group_min",
    "group_percentile",
    "group_rank",
    "group_rankpct",
    "group_std",
    "group_zscore",
    "mask",
    "not_",
    "orthogonalize",
    "project",
    "rolling_elastic_net",
    "rolling_lasso",
    "rolling_ols",
    "rolling_ridge",
    "vol_scale",
}


def test_all_registered_operations_have_curated_authoritative_summaries() -> None:
    transformers = [
        TRANSFORMER_REGISTRY.get(name)
        for name in TRANSFORMER_REGISTRY.names()
        if TRANSFORMER_REGISTRY.get(name).operation.__module__.startswith(
            "bagelquant_core"
        )
    ]
    composers = [
        COMPOSER_REGISTRY.get(name)
        for name in COMPOSER_REGISTRY.names()
        if COMPOSER_REGISTRY.get(name).operation.__module__.startswith(
            "bagelquant_core"
        )
    ]

    assert len(transformers) == 97
    assert len(composers) == 25
    for operation in [*transformers, *composers]:
        name = operation.operation.__name__
        assert name in OPERATION_DESCRIPTIONS
        summary = (operation.__doc__ or "").strip().splitlines()[0]
        assert summary
        assert summary != f"Apply `{name}` to long-form panel inputs."


def test_operation_classification_and_examples_cover_the_public_catalog() -> None:
    transformers = {
        TRANSFORMER_REGISTRY.get(runtime).operation.__name__
        for runtime in TRANSFORMER_REGISTRY.names()
    }
    composers = {
        COMPOSER_REGISTRY.get(runtime).operation.__name__
        for runtime in COMPOSER_REGISTRY.names()
    }

    assert MOVED_TRANSFORMERS <= transformers
    assert MOVED_TRANSFORMERS.isdisjoint(composers)
    assert not any(name.startswith("category_") for name in transformers | composers)
    for kind, names in (("transformer", transformers), ("composer", composers)):
        for name in names:
            example = operation_example(name, kind=kind)
            assert example.inputs
            assert not example.output.data.is_empty()
            assert operation_category(name, kind=kind)


def test_duplicate_transformer_names_are_not_registered() -> None:
    removed = {
        "abs_value",
        "delta",
        "fisher",
        "logrank",
        "nonnans",
        "rate_of_change",
    }
    names = {
        TRANSFORMER_REGISTRY.get(runtime).operation.__name__
        for runtime in TRANSFORMER_REGISTRY.names()
    }

    assert removed.isdisjoint(names)
