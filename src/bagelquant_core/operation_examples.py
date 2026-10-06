"""Executable, shared examples for generated operation documentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import polars as pl

from bagelquant_core.operator import OPERATOR_REGISTRY
from bagelquant_core.node import Node, Domain


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
    if operation.factory is not None:
        import inspect
        config = {key: {"window": 2, "half_life": 2, "quantiles": 2}[key]
                  for key in inspect.signature(operation.factory).parameters}
        auxiliary_inputs = {}
        if operation.factory.supervised:
            auxiliary_inputs = {"targets": auxiliary, "availability": Node.from_domain(
                auxiliary.collect().with_columns(pl.lit(date(2024, 1, 1).toordinal()).alias("value")), auxiliary.domain)}
        peers = (source,) if name == "identity_prediction" else (source, auxiliary)
        result = operation(*peers, **auxiliary_inputs, **config).compute()
        peer_names = ("source",) if len(peers) == 1 else ("source", "second")
        arguments = ", ".join(peer_names) + "".join(f", {key}={key}" for key in auxiliary_inputs)
        return OperationExample(name, kind, f"{name}({arguments}{_config_call(config)})",
            tuple(ExamplePanel(key, value.collect()) for key, value in zip(peer_names, peers, strict=True)),
            {key: (ExamplePanel(key, value.collect()),) for key, value in auxiliary_inputs.items()},
            ExamplePanel("output", result.collect()))
    if name == "equal_weight":
        source = binary

    if operation.minimum_inputs > 1 or operation.maximum_inputs is None:
        if name in {"rolling_elastic_net_prediction", "rolling_lightgbm_prediction"}:
            config.update(window=3, fit_every=1, min_samples=2, max_samples=100, label_maturity=1)
            if name == "rolling_lightgbm_prediction":
                config.update(num_boost_round=2, min_data_in_leaf=1, min_samples=100)
            graph = operation(source, labels=auxiliary, **config)
            return OperationExample(name, kind, f"{name}(source, labels=labels{_config_call(config)})",
                (ExamplePanel("source", source.collect(dense=True)),),
                {"labels": (ExamplePanel("labels", auxiliary.collect(dense=True)),)},
                ExamplePanel("output", graph.compute().collect(dense=True)))
        if name == "broadcast_by_time":
            source = Node.from_domain(
                auxiliary.collect(dense=False)
                .filter(pl.col("asset_id") == "a")
                .select("time", "asset_id", "value"),
                source.domain,
                name="source",
            )
        graph = operation(source, auxiliary, **config)
        inputs = (
            ExamplePanel(
                "input_1",
                source.collect(dense=name != "broadcast_by_time"),
            ),
            ExamplePanel("input_2", auxiliary.collect(dense=True)),
        )
        panel_parameters: dict[str, tuple[ExamplePanel, ...]] = {}
        call = f"{name}(input_1, input_2{_config_call(config)})"
    else:
        panel_arguments: dict[str, Any] = {}
        parameter_examples: dict[str, tuple[ExamplePanel, ...]] = {}
        for parameter, multiple in operation.panel_parameter_kinds.items():
            if parameter == "reference":
                continue
            panel = (
                group
                if parameter == "group"
                else binary
                if parameter in {"binary", "mask_frame"}
                else auxiliary
            )
            panel_arguments[parameter] = (panel,) if multiple else panel
            parameter_examples[parameter] = (
                ExamplePanel(parameter, panel.collect(dense=True)),
            )
        graph = operation(source, **panel_arguments, **config)
        inputs = (ExamplePanel("source", source.collect(dense=True)),)
        panel_parameters = parameter_examples
        panel_call = "".join(
            f", {parameter}="
            + (f"({parameter},)" if multiple else parameter)
            for parameter, multiple in operation.panel_parameter_kinds.items() if parameter != "reference"
        )
        call = f"{name}(source{panel_call}{_config_call(config)})"
    return OperationExample(
        name=name,
        kind=kind,
        call=call,
        inputs=inputs,
        panel_parameters=panel_parameters,
        output=ExamplePanel("output", graph.compute().collect(dense=True)),
    )


def _registry_item(name: str, *, kind: str) -> Any:
    registry = OPERATOR_REGISTRY
    matches = [
        registry.get(value)
        for value in registry.names()
        if (value.split(":", 1)[1]+"_prediction" if value.startswith("prediction:") else registry.get(value).operation.__name__) == name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous {kind} example operation: {name}")
    return matches[0]


def _panels(name: str) -> tuple[Node, Node]:
    if name.startswith("group_") or name == "orthogonalize":
        times = [date(2024, 1, 2)]
        assets = ["a", "b", "c", "d"]
        source_values = [1.0, 3.0, 6.0, 10.0]
        auxiliary_values = [1.0, 1.0, 2.0, 3.0]
        groups = ["tech", "tech", "bank", "bank"]
    elif name in {
        "diff_from_last_change",
        "pct_change_from_last_change",
        "repeat_count",
        "streak_count",
    }:
        times = [date(2024, 1, day) for day in (2, 3, 4, 5, 8)]
        assets = ["a", "b"]
        source_values = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0]
        auxiliary_values = [1.0] * 10
        groups = ["one"] * 5 + ["two"] * 5
    elif name == "smooth":
        times = [
            date(2024, 1, day)
            for day in (2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16)
        ]
        assets = ["a", "b"]
        source_values = [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            2.0,
            4.0,
            None,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
        ]
        auxiliary_values = [1.0] * 22
        groups = ["one"] * 11 + ["two"] * 11
    elif name.startswith("kelly"):
        times = [date(2024, 1, day) for day in (2, 3, 4, 5)]
        assets = ["a", "b", "c"]
        source_values = [
            2.0,
            None,
            5.0,
            2.0,
            3.0,
            2.0,
            1.0,
            6.0,
            1.0,
            4.0,
            3.0,
            1.0,
        ]
        auxiliary_values = [1.0] * 12
        groups = ["one"] * 4 + ["two"] * 4 + ["three"] * 4
    elif name.startswith("rolling_") or name.startswith("ewm_") or name in {
        "lag",
        "diff",
        "pct_change",
        "remove_repeated",
    }:
        times = [date(2024, 1, day) for day in (2, 3, 4, 5)]
        assets = ["a", "b"]
        source_values = [1.0, 2.0, 4.0, 7.0, 2.0, 3.0, 5.0, 8.0]
        auxiliary_values = [1.0, 1.5, 2.0, 3.0, 1.0, 2.0, 2.5, 4.0]
        groups = ["one"] * 4 + ["two"] * 4
    elif name in {
        "anscombe",
        "arccos",
        "arcsin",
        "arctanh",
        "boxcox",
        "freeman",
        "log",
        "log1p",
        "sqrt",
    }:
        times = [date(2024, 1, 2), date(2024, 1, 3)]
        assets = ["a", "b", "c"]
        source_values = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
        auxiliary_values = [1.0, 2.0, 4.0, 0.0, 1.0, 2.0]
        groups = ["left", "left", "right", "left", "left", "right"]
    elif name in {"abs", "ceil", "negonly", "posonly", "sign"}:
        times = [date(2024, 1, 2), date(2024, 1, 3)]
        assets = ["a", "b", "c"]
        source_values = [-2.0, -0.5, 0.0, 0.2, 1.2, 2.7]
        auxiliary_values = [1.0] * 6
        groups = ["left", "left", "middle", "middle", "right", "right"]
    elif name in {
        "and_",
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "not_",
        "or_",
        "xand",
        "xor",
    }:
        times = [date(2024, 1, 2), date(2024, 1, 3)]
        assets = ["a", "b", "c"]
        source_values = [0.0, 1.0, None, 1.0, 0.0, 2.0]
        auxiliary_values = [0.0, 0.0, 1.0, 1.0, 2.0, None]
        groups = ["left", "left", "right", "left", "left", "right"]
    elif name in {
        "bfill",
        "coalesce",
        "ffill",
        "fillna_zero",
        "notnan",
        "replace_inf",
    }:
        times = [date(2024, 1, 2), date(2024, 1, 3)]
        assets = ["a", "b", "c"]
        source_values = [1.0, None, float("nan"), None, 3.0, float("inf")]
        auxiliary_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        groups = ["left", "left", "right", "left", "left", "right"]
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

    source = Node.from_domain(frame(source_values), domain, name="source")
    auxiliary = Node.from_domain(
        frame(auxiliary_values), domain, name="auxiliary"
    )
    binary_values = [float(index % 2 == 0) for index in range(len(rows))]
    binary = Node.from_domain(frame(binary_values), domain, name="binary")
    group = Node.from_domain(frame(groups), domain, name="group", value_type="category")
    return source, auxiliary, binary, group


def _config(name: str) -> dict[str, Any]:
    if name == "exposure_constrained_weights":
        return {"max_weight": 1.0, "lower_bounds": [-100.0], "upper_bounds": [100.0]}
    if name == "top_n":
        return {"count": 2}
    if name == "rebalance":
        return {"every": 1, "data_start": "2024-01-02"}
    if name == "regularized_weights":
        return {"max_weight": 0.5}

    if name in {"trim", "truncate"}:
        return {"lower": 0.0, "upper": 5.0}
    if name in {"power", "signed_power"}:
        return {"exponent": 2.0}
    if name == "boxcox":
        return {"lambda_": 0.5}
    if name == "replace_non_nan":
        return {"value": 1.0}
    if name in {"weighted_mean", "weighted_sum"}:
        return {"weights": [0.25, 0.75]}
    if name in {"ewm_mean", "ewm_std", "ewm_var"}:
        return {"span": 2.0}
    if name == "rolling_ewm_fw":
        return {"window": 2, "halflife": 2.0}
    if name == "denoise":
        return {"threshold": 2.5}
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
