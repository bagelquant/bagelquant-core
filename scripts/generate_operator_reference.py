"""Generate one reference page for each public transformer and composer."""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bagelquant_core import composer as composer_api  # noqa: E402
from bagelquant_core import transformer as transformer_api  # noqa: E402
from bagelquant_core._documentation import (  # noqa: E402
    OPERATION_DESCRIPTIONS,
    operation_category,
    operation_description,
)
from bagelquant_core.operation_examples import operation_example  # noqa: E402

REFERENCE = ROOT / "docs" / "en" / "reference"

EXCLUDED = {
    "COMPOSER_REGISTRY",
    "ComposerFunction",
    "TRANSFORMER_REGISTRY",
    "TransformerFunction",
    "composer",
    "pct_change_frame",
    "transformer",
}

PARAMETER_DESCRIPTIONS = {
    "source": "Input numeric `Panel` or single-output `Graph`.",
    "like": "Stock-domain `Panel` or single-output `Graph` whose dated keys define the broadcast output.",
    "lhs": "Left-hand numeric `Panel` or single-output `Graph`.",
    "rhs": "Right-hand numeric `Panel` or single-output `Graph`.",
    "frame": "Input numeric `Panel` or single-output `Graph`.",
    "frames": "One or more numeric `Panel` or single-output `Graph` inputs.",
    "binary": "Binary projection input. Cells equal to `1` are retained; other cells are masked.",
    "mask_frame": "Mask input. Truthy cells retain values; false or missing cells are replaced.",
    "categories": "Matching `CategoryPanel` containing row-wise group labels.",
    "group": "Matching `CategoryPanel` containing row-wise group labels.",
    "factors": "One or more factor `Panel` or single-output `Graph` inputs.",
    "y": "Dependent-variable `Panel` or single-output `Graph`.",
    "volatility": "Volatility `Panel` or single-output `Graph` used as the divisor.",
    "power": "Exponent `Panel` or single-output `Graph`.",
    "name": "Optional graph-node name. A generated name is used when omitted.",
    "metadata": "Optional metadata stored on the graph node.",
    "periods": "Number of prior rows to shift or compare. Must be a positive integer.",
    "interval": "Number of prior rows between observations. Must be a positive integer.",
    "window": "Positive trailing-window length in rows.",
    "min_periods": "Minimum number of observations required to produce a value.",
    "ddof": "Delta degrees of freedom used by variance or standard-deviation calculations.",
    "limit": "Maximum number of consecutive missing values to fill.",
    "value": "Numeric replacement or constant value.",
    "threshold": "Non-negative magnitude below which values are replaced with zero.",
    "lower": "Lower fixed bound or lower quantile, depending on the operation.",
    "upper": "Upper fixed bound or upper quantile, depending on the operation.",
    "lambda_": "Box-Cox lambda parameter. Use `0` for the logarithmic limit.",
    "exponent": "Numeric exponent.",
    "weights": "One numeric weight for each input frame.",
    "replace_value": "Value inserted where the mask is false or missing.",
    "min_valid": "Minimum valid observations required within the trailing window.",
    "com": "Center-of-mass decay parameter. Supply exactly one decay parameter.",
    "span": "Span decay parameter. Supply exactly one decay parameter.",
    "halflife": "Half-life decay parameter. Supply exactly one decay parameter.",
    "alpha": "Smoothing or regularization parameter, depending on the operation.",
    "adjust": "Whether to divide by the decaying adjustment factor.",
    "reset_on_equal": "Whether an unchanged value resets the directional streak to zero.",
    "ignore_na": "Whether missing values are ignored when calculating weights.",
    "bias": "Whether to use the biased exponentially weighted variance calculation.",
    "l1_ratio": "Elastic-net mixing parameter in `[0, 1]`.",
    "max_iter": "Maximum coordinate-descent iterations.",
    "tolerance": "Convergence tolerance for coordinate descent.",
    "target": "Dependent-variable `Panel` predicted from the trailing factor window.",
    "fit_intercept": "Whether the cross-sectional or rolling regression includes an intercept.",
}

EXAMPLE_CONFIG = {
    "boxcox": "lambda_=0.5",
    "constant": "value=2",
    "date_age_constraint": "window=2",
    "denoise": "threshold=1e-6",
    "diff": "periods=1",
    "fillna": "value=0",
    "kelly": "window=2",
    "kelly_nonan_standardize": "window=2",
    "kelly_rank_boxcox": "window=2",
    "kelly_rescaling_weight": "window=2",
    "lag": "periods=1",
    "pct_change": "periods=1",
    "power": "exponent=2",
    "replace_non_nan": "value=1",
    "rolling_ewm_fw": "window=2, halflife=2",
    "signed_power": "exponent=0.5",
    "trim": "lower=-1, upper=1",
    "trim_quantile": "lower=0.1, upper=0.9",
    "truncate": "lower=-1, upper=1",
}

ROLLING_TRANSFORMERS = {
    "rolling_kurt",
    "rolling_max",
    "rolling_mean",
    "rolling_median",
    "rolling_min",
    "rolling_percentile",
    "rolling_rank",
    "rolling_skew",
    "rolling_std",
    "rolling_sum",
    "rolling_var",
    "rolling_zscore",
}

EWM_TRANSFORMERS = {"ewm_mean", "ewm_std", "ewm_var"}
GROUP_COMPOSERS = {
    name for name in OPERATION_DESCRIPTIONS if name.startswith("group_")
}
ROLLING_RELATIONSHIPS = {"rolling_corr", "rolling_cov"}
ROLLING_REGRESSIONS = {
    "rolling_elastic_net",
    "rolling_lasso",
    "rolling_ols",
    "rolling_ridge",
}

COMPOSER_CONFIG = {
    "mask": "replace_value=0",
    "rolling_corr": "window=2",
    "rolling_cov": "window=2",
    "rolling_elastic_net": "window=2",
    "rolling_lasso": "window=2",
    "rolling_ols": "window=2",
    "rolling_ridge": "window=2",
    "weighted_mean": "weights=[0.25, 0.75]",
    "weighted_sum": "weights=[0.25, 0.75]",
}


def _operation(item: Any) -> Any:
    return getattr(item, "operation", item)


def _description(name: str, item: Any) -> str:
    if name in OPERATION_DESCRIPTIONS:
        description = OPERATION_DESCRIPTIONS[name]
        links = {
            "`div`": "[`div`](./div.md)",
            "`maximum`": "[`maximum`](./maximum.md)",
            "`minimum`": "[`minimum`](./minimum.md)",
            "`mul`": "[`mul`](./mul.md)",
            "`sub`": "[`sub`](./sub.md)",
            "`ewm_std`": "[`ewm_std`](./ewm_std.md)",
            "`ewm_mean`": "[`ewm_mean`](./ewm_mean.md)",
        }
        for label, link in links.items():
            description = description.replace(label, link)
        return description
    doc = inspect.getdoc(_operation(item))
    if doc:
        return doc.splitlines()[0]
    return operation_description(name)


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    text = str(annotation).replace("typing.", "")
    return text.replace("<class '", "").replace("'>", "")


def _public_parameters(item: Any, *, kind: str) -> list[tuple[str, str, str]]:
    signature = inspect.signature(_operation(item))
    parameters: list[tuple[str, str, str]] = []
    for parameter in signature.parameters.values():
        name = parameter.name
        annotation = _format_annotation(parameter.annotation)
        if name == "frame" and kind == "transformer":
            name = "source"
            annotation = "Panel | Graph"
        elif annotation in {"pd.DataFrame", "pl.DataFrame"}:
            annotation = "Panel | Graph"
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            annotation = "Panel | Graph"
        default = ""
        if parameter.default is not inspect.Parameter.empty:
            default = f", default `{parameter.default!r}`"
        suffix = f"{annotation}{default}"
        if name not in PARAMETER_DESCRIPTIONS:
            raise RuntimeError(f"missing public parameter documentation: {name}")
        parameters.append((name, suffix, PARAMETER_DESCRIPTIONS[name]))
    parameters.extend(
        [
            ("name", "str | None, default `None`", PARAMETER_DESCRIPTIONS["name"]),
            (
                "metadata",
                "Mapping[str, Any] | None, default `None`",
                PARAMETER_DESCRIPTIONS["metadata"],
            ),
        ]
    )
    return parameters


def _signature(name: str, item: Any, *, kind: str) -> str:
    parts: list[str] = []
    keyword_boundary = False
    for parameter in inspect.signature(_operation(item)).parameters.values():
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not keyword_boundary:
            parts.append("*")
            keyword_boundary = True
        parameter_name = (
            "source"
            if kind == "transformer" and parameter.name in {"frame", "target"}
            else parameter.name
        )
        default = (
            ""
            if parameter.default is inspect.Parameter.empty
            else f"={parameter.default!r}"
        )
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{parameter_name}")
            keyword_boundary = True
        else:
            parts.append(f"{parameter_name}{default}")
    if not keyword_boundary:
        parts.append("*")
    parts.extend(("name=None", "metadata=None"))
    return f"{name}({', '.join(parts)})"


def _transformer_example(name: str) -> str:
    if name.startswith("category_"):
        return f"""import polars as pl

from bagelquant_core import CategoryPanel, Domain, Panel
from bagelquant_core.transformer import {name}

domain = Domain(calendar=["2024-01-02"], universe=["a", "b", "c"])
factor = Panel.from_domain(
    pl.DataFrame({{
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": [1.0, 3.0, 8.0],
    }}),
    domain,
)
industry = CategoryPanel.from_domain(
    pl.DataFrame({{
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": ["tech", "tech", "bank"],
    }}),
    domain,
)

result = {name}(factor, industry).compute().data
print(result)"""
    config = EXAMPLE_CONFIG.get(name)
    if name in ROLLING_TRANSFORMERS:
        config = "window=2"
    elif name in EWM_TRANSFORMERS:
        config = "span=2"
    arguments = f"source, {config}" if config else "source"
    return f"""import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.transformer import {name}

domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
source = Panel.from_domain(
    pl.DataFrame({{
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }}),
    domain,
)

result = {name}({arguments}).compute().data
print(result)"""


def _composer_example(name: str) -> str:
    if name in GROUP_COMPOSERS:
        return f"""import polars as pl

from bagelquant_core import CategoryPanel, Domain, Panel
from bagelquant_core.composer import {name}

domain = Domain(calendar=["2024-01-02"], universe=["a", "b", "c"])
factor = Panel.from_domain(
    pl.DataFrame({{
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": [1.0, 3.0, 8.0],
    }}),
    domain,
)
industry = CategoryPanel.from_domain(
    pl.DataFrame({{
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": ["tech", "tech", "bank"],
    }}),
    domain,
)

result = {name}(factor, industry).compute().data
print(result)"""
    if name == "orthogonalize":
        call = "orthogonalize(factor, size)"
    elif name in ROLLING_REGRESSIONS:
        call = f"{name}(left, right, window=2)"
    elif name in ROLLING_RELATIONSHIPS:
        call = f"{name}(left, right, window=2)"
    elif name in {
        "coalesce",
        "mean",
        "maximum",
        "minimum",
        "product",
        "sum_frames",
    }:
        call = f"{name}(left, right)"
    elif name in {"weighted_mean", "weighted_sum"}:
        call = f"{name}(left, right, weights=[0.25, 0.75])"
    elif name == "not_":
        call = "not_(left)"
    elif name == "project":
        call = "project(left, right)"
    elif name == "mask":
        call = "mask(left, right, replace_value=0.0)"
    else:
        call = f"{name}(left, right)"
    if name == "orthogonalize":
        setup = """domain = Domain(calendar=["2024-01-02"], universe=["a", "b", "c"])
factor = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": [1.0, 3.0, 5.0],
    }),
    domain,
)
size = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": [0.0, 1.0, 2.0],
    }),
    domain,
)"""
    else:
        setup = """domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
left = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }),
    domain,
)
right = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 1.0, 2.0, 1.0, 2.0, 4.0],
    }),
    domain,
)"""
    return f"""import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import {name}

{setup}

result = {call}.compute().data
print(result)"""


def _notes(name: str, *, kind: str) -> str:
    category = operation_category(name, kind=kind)
    notes = []
    if kind == "transformer":
        notes.append(
            "The primary input is one sparse long-form Panel keyed by "
            "`(time, asset_id)`; absent keys remain absent."
        )
    else:
        notes.append(
            "Peer inputs are aligned on `(time, asset_id)` before the operation "
            "runs; alignment does not invent Universe membership."
        )
    if category in {"Cross-sectional", "Group & neutralization"}:
        notes.append(
            "Each date cross-section is calculated independently, so values from "
            "other dates cannot influence the result."
        )
    if name in {
        "and_",
        "or_",
        "not_",
        "xand",
        "xor",
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
    }:
        notes.append(
            "Logical and comparison results are numeric panels containing `1.0` and `0.0`."
        )
    if category in {"Rolling regression", "Rolling statistics"}:
        notes.append(
            "History is grouped by `asset_id` and ordered by `time`; rows with "
            "insufficient observations remain missing."
        )
    if name in ROLLING_REGRESSIONS:
        notes.append(
            "The model is fit on prior rows only and predicts the current row."
        )
    if name.startswith("group_"):
        notes.append(
            "The keyword-only `group` Panel participates in graph topology and "
            "availability tracing; missing group labels are excluded."
        )
    if name == "bfill":
        notes.append(
            "Back-fill reads later observations and is non-causal for research "
            "definitions unless a caller explicitly permits that use."
        )
    if name == "ffill":
        notes.append(
            "Forward-fill is causal only with the explicit finite `limit`; it never "
            "fills a missing dynamic-Universe row."
        )
    return "\n\n".join(notes)


def _page(name: str, item: Any, *, kind: str) -> str:
    parameters = _public_parameters(item, kind=kind)
    parameter_lines = []
    for parameter_name, type_text, description in parameters:
        parameter_lines.append(f"**{parameter_name}** : {type_text}\n: {description}")
    example = operation_example(name, kind=kind)
    input_sections = []
    for panel in example.inputs:
        input_sections.append(
            f"### {panel.label}\n\n{_markdown_table(panel.data)}"
        )
    for parameter, panels in example.panel_parameters.items():
        for index, panel in enumerate(panels, start=1):
            suffix = f" {index}" if len(panels) > 1 else ""
            input_sections.append(
                f"### Panel parameter: {parameter}{suffix}\n\n"
                f"{_markdown_table(panel.data)}"
            )
    return f"""# `{name}`

{_description(name, item)}

## Signature

```python
{_signature(name, item, kind=kind)}
```

## Parameters

{chr(10).join(parameter_lines)}

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
{example.call}
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

{chr(10).join(input_sections)}

### Output

{_markdown_table(example.output.data)}

## Panel and temporal semantics

{_notes(name, kind=kind)}
"""


def _markdown_table(frame: Any) -> str:
    data = frame.select("time", "asset_id", "value").sort(
        "time", "asset_id"
    )
    assets = sorted(str(value) for value in data.get_column("asset_id").unique())
    values = {
        (str(row["time"]), str(row["asset_id"])): row["value"]
        for row in data.to_dicts()
    }
    times = sorted({str(value) for value in data.get_column("time")})
    lines = [
        f"| time | {' | '.join(assets)} |",
        f"|---|{'|'.join('---:' for _ in assets)}|",
    ]
    for time in times:
        rendered = [
            _markdown_value(values.get((time, asset)))
            for asset in assets
        ]
        lines.append(f"| {time} | {' | '.join(rendered)} |")
    return "\n".join(lines)


def _markdown_value(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "missing"
    if isinstance(value, float):
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _generate_kind(module: Any, *, kind: str, check: bool) -> list[str]:
    directory = REFERENCE / f"{kind}s"
    registry = (
        transformer_api.TRANSFORMER_REGISTRY
        if kind == "transformer"
        else composer_api.COMPOSER_REGISTRY
    )
    items = {
        registry.get(runtime_name).operation.__name__: registry.get(runtime_name)
        for runtime_name in registry.names()
    }
    names = sorted(items)
    contents: dict[Path, str] = {}
    for name in names:
        item = items[name]
        contents[directory / f"{name}.md"] = _page(name, item, kind=kind)
    categories: dict[str, list[str]] = {}
    for name in names:
        categories.setdefault(operation_category(name, kind=kind), []).append(name)
    links = "\n\n".join(
        f"## {category}\n\n"
        + "\n".join(f"- [`{name}`](./{name}.md)" for name in category_names)
        for category, category_names in categories.items()
    )
    contents[directory / "index.md"] = f"""# {kind.title()} reference

Each public {kind} has a generated reference page with an exact signature,
parameter contract, executable Panel example, and temporal semantics.

{links}
"""
    expected = set(contents)
    actual = set(directory.glob("*.md")) if directory.is_dir() else set()
    if check:
        if actual != expected:
            raise RuntimeError(
                f"generated {kind} reference inventory is stale: {directory}"
            )
        for path, content in contents.items():
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(f"generated documentation is stale: {path}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        for stale in actual - expected:
            stale.unlink()
        for path, content in contents.items():
            path.write_text(content, encoding="utf-8", newline="\n")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    transformers = _generate_kind(
        transformer_api, kind="transformer", check=args.check
    )
    composers = _generate_kind(
        composer_api, kind="composer", check=args.check
    )
    index = f"""# Operation reference

BagelQuant operations build deterministic lazy graphs from sparse long-form
`Panel` inputs.

- [Transformer reference](./transformers/index.md): {len(transformers)} public operations
- [Composer reference](./composers/index.md): {len(composers)} public operations

The reference pages are generated from the exported API and curated
documentation metadata. Regenerate them after changing the operation catalog:

```bash
uv run python scripts/generate_operator_reference.py
```
"""
    index_path = REFERENCE / "index.md"
    if args.check:
        if not index_path.is_file() or index_path.read_text(encoding="utf-8") != index:
            raise RuntimeError(f"generated documentation is stale: {index_path}")
    else:
        index_path.write_text(index, encoding="utf-8", newline="\n")
        print(
            f"Generated {len(transformers)} transformer pages and "
            f"{len(composers)} composer pages"
        )


if __name__ == "__main__":
    main()
