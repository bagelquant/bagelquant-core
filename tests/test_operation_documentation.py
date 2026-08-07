from __future__ import annotations

from bagelquant_core._documentation import OPERATION_DESCRIPTIONS
from bagelquant_core.composer import COMPOSER_REGISTRY
from bagelquant_core.transformer import TRANSFORMER_REGISTRY


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

    assert len(transformers) == 79
    assert len(composers) == 47
    for operation in [*transformers, *composers]:
        name = operation.operation.__name__
        assert name in OPERATION_DESCRIPTIONS
        summary = (operation.__doc__ or "").strip().splitlines()[0]
        assert summary
        assert summary != f"Apply `{name}` to long-form panel inputs."
