"""Executable, shared examples for generated operation documentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import polars as pl

from .composer import COMPOSER_REGISTRY
from .panel import CategoryPanel, Domain, Panel
from .transformer import TRANSFORMER_REGISTRY


@dataclass(frozen=True, slots=True)
class ExamplePanel:
    label: str
    data: pl.DataFrame


@dataclass(frozen=True, slots=True)
class OperationExample:
    name: str
    kind: str
    call: str
    inputs: tuple[ExamplePanel, ...]
    panel_parameters: Mapping[str, tuple[ExamplePanel, ...]]
    output: ExamplePanel


def operation_example(name: str, *, kind: str) -> OperationExample:
    """Execute one deterministic, hand-checkable operation example."""

    operation = _registry_item(name, kind=kind)
    source, auxiliary, binary, group = _panels(name)
    config = _config(name)
    if kind == "composer":
        graph = operation(source, auxiliary, **config)
        inputs = (
            ExamplePanel("input_1", source.data),
            ExamplePanel("input_2", auxiliary.data),
        )
        panel_parameters: dict[str, tuple[ExamplePanel, ...]] = {}
        call = f"{name}(input_1, input_2{_config_call(config)})"
    else:
        panel_arguments: dict[str, Any] = {}
        parameter_examples: dict[str, tuple[ExamplePanel, ...]] = {}
        for parameter, multiple in operation.panel_parameter_kinds.items():
            panel = (
                group
                if parameter == "group"
                else binary
                if parameter in {"binary", "mask_frame"}
                else auxiliary
            )
            panel_arguments[parameter] = (panel,) if multiple else panel
            parameter_examples[parameter] = (
                ExamplePanel(parameter, panel.data),
            )
        graph = operation(source, **panel_arguments, **config)
        inputs = (ExamplePanel("source", source.data),)
        panel_parameters = parameter_examples
        panel_call = "".join(
            f", {parameter}="
            + (f"({parameter},)" if multiple else parameter)
            for parameter, multiple in operation.panel_parameter_kinds.items()
        )
        call = f"{name}(source{panel_call}{_config_call(config)})"
    return OperationExample(
        name=name,
        kind=kind,
        call=call,
        inputs=inputs,
        panel_parameters=panel_parameters,
        output=ExamplePanel("output", graph.compute().data),
    )


def _registry_item(name: str, *, kind: str) -> Any:
    registry = TRANSFORMER_REGISTRY if kind == "transformer" else COMPOSER_REGISTRY
    matches = [
        registry.get(value)
        for value in registry.names()
        if registry.get(value).operation.__name__ == name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous {kind} example operation: {name}")
    return matches[0]


def _panels(name: str) -> tuple[Panel, Panel, Panel, CategoryPanel]:
    if name.startswith("group_") or name == "orthogonalize":
        times = [date(2024, 1, 2)]
        assets = ["a", "b", "c", "d"]
        source_values = [1.0, 3.0, 6.0, 10.0]
        auxiliary_values = [1.0, 1.0, 2.0, 3.0]
        groups = ["tech", "tech", "bank", "bank"]
    elif name.startswith("rolling_") or name.startswith("ewm_") or name in {
        "lag",
        "diff",
        "delta",
        "pct_change",
        "rate_of_change",
        "remove_repeated",
        "kelly",
        "kelly_nonan_standardize",
        "kelly_rank_boxcox",
        "kelly_rescaling_weight",
    }:
        times = [date(2024, 1, day) for day in (2, 3, 4, 5)]
        assets = ["a", "b"]
        source_values = [1.0, 2.0, 4.0, 7.0, 2.0, 3.0, 5.0, 8.0]
        auxiliary_values = [1.0, 1.5, 2.0, 3.0, 1.0, 2.0, 2.5, 4.0]
        groups = ["one"] * 4 + ["two"] * 4
    else:
        times = [date(2024, 1, 2), date(2024, 1, 3)]
        assets = ["a", "b", "c"]
        source_values = [1.0, None, 4.0, 2.0, 3.0, 8.0]
        auxiliary_values = [1.0, 2.0, 2.0, 1.0, 2.0, 4.0]
        groups = ["tech", "tech", "bank", "tech", "tech", "bank"]
    rows = [(time, asset) for asset in assets for time in times]
    domain = Domain(calendar=times, universe=assets)

    def frame(values: list[Any]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [row[0] for row in rows],
                "asset_id": [row[1] for row in rows],
                "value": values,
            }
        )

    source = Panel.from_domain(frame(source_values), domain, name="source")
    auxiliary = Panel.from_domain(
        frame(auxiliary_values), domain, name="auxiliary"
    )
    binary_values = [float(index % 2 == 0) for index in range(len(rows))]
    binary = Panel.from_domain(frame(binary_values), domain, name="binary")
    group = CategoryPanel.from_domain(frame(groups), domain, name="group")
    return source, auxiliary, binary, group


def _config(name: str) -> dict[str, Any]:
    if name in {"trim", "truncate"}:
        return {"lower": 0.0, "upper": 5.0}
    if name in {"power", "signed_power"}:
        return {"exponent": 2.0}
    if name == "replace_non_nan":
        return {"value": 1.0}
    if name in {"weighted_mean", "weighted_sum"}:
        return {"weights": [0.25, 0.75]}
    if name in {"ewm_mean", "ewm_std", "ewm_var"}:
        return {"span": 2.0}
    if name == "rolling_ewm_fw":
        return {"halflife": 2.0}
    if name.startswith("rolling_") or name.startswith("kelly") or name == (
        "date_age_constraint"
    ):
        return {"window": 2}
    if name in {"ffill", "bfill"}:
        return {"limit": 1}
    if name == "mask":
        return {"replace_value": 0.0}
    return {}


def _config_call(config: Mapping[str, Any]) -> str:
    return "".join(f", {key}={value!r}" for key, value in config.items())


__all__ = ["ExamplePanel", "OperationExample", "operation_example"]
