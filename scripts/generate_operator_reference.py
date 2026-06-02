"""Generate one reference page for each public transformer and composer."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bagelquant_core import composer as composer_api
from bagelquant_core import transformer as transformer_api

REFERENCE = ROOT / "docs" / "reference"

EXCLUDED = {
    "COMPOSER_REGISTRY",
    "ComposerFunction",
    "TRANSFORMER_REGISTRY",
    "TransformerFunction",
    "composer",
    "transformer",
}

DESCRIPTIONS = {
    "abs": "Return the absolute value of each element.",
    "and_": "Return one where both corresponding elements are truthy and zero elsewhere.",
    "arccos": "Return the inverse cosine of each element, masking values outside `[-1, 1]`.",
    "arcsin": "Return the inverse sine of each element, masking values outside `[-1, 1]`.",
    "arctan": "Return the inverse tangent of each element.",
    "arctanh": "Return the inverse hyperbolic tangent of each element, masking values outside `(-1, 1)`.",
    "coalesce": "Return the first non-missing value from the supplied inputs for each cell.",
    "cos": "Return the cosine of each element.",
    "divide": "Alias for [`div`](./div.md). Divide the first input by the second element-wise.",
    "equal": "Return one where corresponding elements are equal and zero elsewhere.",
    "greater": "Return one where the first input is greater than the second and zero elsewhere.",
    "greater_equal": "Return one where the first input is greater than or equal to the second and zero elsewhere.",
    "group_demean": "Subtract the row-wise group mean from each element.",
    "group_max": "Replace each element with its row-wise group maximum.",
    "group_mean": "Replace each element with its row-wise group mean.",
    "group_median": "Replace each element with its row-wise group median.",
    "group_min": "Replace each element with its row-wise group minimum.",
    "group_percentile": "Return row-wise percentile ranks within each group.",
    "group_rank": "Return row-wise ranks within each group.",
    "group_rankpct": "Return row-wise dense percentile ranks within each group.",
    "group_std": "Replace each element with its row-wise group standard deviation.",
    "group_zscore": "Return row-wise z-scores within each group.",
    "less": "Return one where the first input is less than the second and zero elsewhere.",
    "less_equal": "Return one where the first input is less than or equal to the second and zero elsewhere.",
    "max": "Alias for [`maximum`](./maximum.md). Return element-wise maximum values.",
    "min": "Alias for [`minimum`](./minimum.md). Return element-wise minimum values.",
    "multiply": "Alias for [`mul`](./mul.md). Multiply two inputs element-wise.",
    "not_": "Return one where elements are falsy and zero where they are truthy.",
    "or_": "Return one where either corresponding element is truthy and zero elsewhere.",
    "power": "Raise each element of the first input to the corresponding element of the second input.",
    "power_df": "Raise each element of the first input to the corresponding element of the second input.",
    "rolling_ew_std": "Alias for [`ewm_std`](./ewm_std.md). Return exponentially weighted standard deviations.",
    "rolling_ewm": "Alias for [`ewm_mean`](./ewm_mean.md). Return exponentially weighted means.",
    "sin": "Return the sine of each element.",
    "subtract": "Alias for [`sub`](./sub.md). Subtract the second input from the first element-wise.",
    "xand": "Return one where corresponding truth values are equivalent and zero elsewhere.",
    "xor": "Return one where exactly one corresponding element is truthy and zero elsewhere.",
}

PARAMETER_DESCRIPTIONS = {
    "source": "Input numeric `Panel` or single-output `Graph`.",
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
    "periods": "Number of rows to shift or compare. Must be a non-zero integer.",
    "interval": "Number of rows between observations. Must be a non-zero integer.",
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
    "ignore_na": "Whether missing values are ignored when calculating weights.",
    "bias": "Whether to use the biased exponentially weighted variance calculation.",
    "l1_ratio": "Elastic-net mixing parameter in `[0, 1]`.",
    "max_iter": "Maximum coordinate-descent iterations.",
    "tolerance": "Convergence tolerance for coordinate descent.",
}

EXAMPLE_CONFIG = {
    "boxcox": "lambda_=0.5",
    "constant": "value=2",
    "date_age_constraint": "window=2",
    "delta": "interval=1",
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
    "rate_of_change": "interval=1",
    "replace_non_nan": "value=1",
    "rolling_ew_std": "span=2",
    "rolling_ewm": "span=2",
    "rolling_ewm_fw": "halflife=2",
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
GROUP_COMPOSERS = {name for name in DESCRIPTIONS if name.startswith("group_")}
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
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    doc = inspect.getdoc(_operation(item))
    if doc:
        return doc.splitlines()[0]
    raise ValueError(f"Missing description for {name}")


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
        elif annotation == "pd.DataFrame":
            annotation = "Panel | Graph"
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            annotation = "Panel | Graph"
        default = ""
        if parameter.default is not inspect.Parameter.empty:
            default = f", default `{parameter.default!r}`"
        suffix = f"{annotation}{default}"
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
    parts = []
    for parameter_name, type_text, _ in _public_parameters(item, kind=kind):
        default = ""
        if ", default `" in type_text:
            default = "=" + type_text.split(", default `", 1)[1][:-1]
        if parameter_name in {"name", "metadata"}:
            parts.append(f"{parameter_name}{default}")
            continue
        original = inspect.signature(_operation(item)).parameters.get(parameter_name)
        if original and original.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{parameter_name}")
        else:
            parts.append(f"{parameter_name}{default}")
    return f"{name}({', '.join(parts)})"


def _transformer_example(name: str) -> str:
    if name.startswith("category_"):
        return f"""import pandas as pd

from bagelquant_core import CategoryPanel, Panel
from bagelquant_core.transformer import {name}

factor = Panel(pd.DataFrame({{"a": [1.0], "b": [3.0], "c": [8.0]}}))
industry = CategoryPanel(pd.DataFrame({{"a": ["tech"], "b": ["tech"], "c": ["bank"]}}))

result = {name}(factor, industry).compute().data
print(result)"""
    config = EXAMPLE_CONFIG.get(name)
    if name in ROLLING_TRANSFORMERS:
        config = "window=2"
    elif name in EWM_TRANSFORMERS:
        config = "span=2"
    arguments = f"source, {config}" if config else "source"
    return f"""import pandas as pd

from bagelquant_core import Panel
from bagelquant_core.transformer import {name}

source = Panel(pd.DataFrame({{"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}}))

result = {name}({arguments}).compute().data
print(result)"""


def _composer_example(name: str) -> str:
    if name in GROUP_COMPOSERS:
        return f"""import pandas as pd

from bagelquant_core import CategoryPanel, Panel
from bagelquant_core.composer import {name}

factor = Panel(pd.DataFrame({{"a": [1.0], "b": [3.0], "c": [8.0]}}))
industry = CategoryPanel(pd.DataFrame({{"a": ["tech"], "b": ["tech"], "c": ["bank"]}}))

result = {name}(factor, industry).compute().data
print(result)"""
    if name == "orthogonalize":
        call = "orthogonalize(factor, size)"
    elif name in ROLLING_REGRESSIONS:
        call = f"{name}(left, right, window=2)"
    elif name in ROLLING_RELATIONSHIPS:
        call = f"{name}(left, right, window=2)"
    elif name in {"coalesce", "mean", "maximum", "minimum", "max", "min", "product", "sum_frames"}:
        call = f"{name}(left, right)"
    elif name in {"weighted_mean", "weighted_sum"}:
        call = f"{name}(left, right, weights=[0.25, 0.75])"
    elif name == "not_":
        call = "not_(left)"
    elif name == "project":
        call = "project(left, right)"
    elif name == "mask":
        call = "mask(left, right, replace_value=0)"
    else:
        call = f"{name}(left, right)"
    if name == "orthogonalize":
        setup = """factor = Panel(pd.DataFrame({"a": [1.0], "b": [3.0], "c": [5.0]}))
size = Panel(pd.DataFrame({"a": [0.0], "b": [1.0], "c": [2.0]}))"""
    else:
        setup = """left = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))
right = Panel(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}))"""
    return f"""import pandas as pd

from bagelquant_core import Panel
from bagelquant_core.composer import {name}

{setup}

result = {call}.compute().data
print(result)"""


def _notes(name: str, *, kind: str) -> str:
    notes = []
    if kind == "transformer":
        notes.append("Rows represent time and columns represent assets.")
    else:
        notes.append("Inputs are aligned by index and columns before the operation runs.")
    if name in {"and_", "or_", "not_", "xand", "xor", "equal", "greater", "greater_equal", "less", "less_equal"}:
        notes.append("Logical and comparison results are numeric panels containing `1.0` and `0.0`.")
    if name.startswith("rolling_") or name.startswith("ewm_"):
        notes.append("Rolling calculations run independently down each asset column.")
    if name in ROLLING_REGRESSIONS:
        notes.append("The model is fit on prior rows only and predicts the current row.")
    if name.startswith("category_") or name.startswith("group_"):
        notes.append("Missing group labels are excluded from the group calculation.")
    return "\n\n".join(notes)


def _page(name: str, item: Any, *, kind: str) -> str:
    parameters = _public_parameters(item, kind=kind)
    parameter_lines = []
    for parameter_name, type_text, description in parameters:
        parameter_lines.append(f"**{parameter_name}** : {type_text}\n: {description}")
    example = _transformer_example(name) if kind == "transformer" else _composer_example(name)
    return f"""# {name}

```python
{_signature(name, item, kind=kind)}
```

{_description(name, item)}

## Parameters

{chr(10).join(parameter_lines)}

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Examples

```python
{example}
```

## Notes

{_notes(name, kind=kind)}
"""


def _generate_kind(module: Any, *, kind: str) -> list[str]:
    directory = REFERENCE / f"{kind}s"
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(name for name in module.__all__ if name not in EXCLUDED)
    for name in names:
        item = getattr(module, name)
        (directory / f"{name}.md").write_text(_page(name, item, kind=kind))
    links = "\n".join(f"- [`{name}`](./{name}.md)" for name in names)
    (directory / "index.md").write_text(
        f"""# {kind.title()} Reference

Each public {kind} has a dedicated reference page with its signature,
parameters, return value, notes, and an example.

{links}
"""
    )
    return names


def main() -> None:
    transformers = _generate_kind(transformer_api, kind="transformer")
    composers = _generate_kind(composer_api, kind="composer")
    (REFERENCE / "index.md").write_text(
        f"""# API Reference

BagelQuant operations build lazy graphs from `Panel` inputs.

- [Transformer reference](./transformers/index.md): {len(transformers)} public operations
- [Composer reference](./composers/index.md): {len(composers)} public operations

The reference pages are generated from the exported API and curated
documentation metadata. Regenerate them after changing the operation catalog:

```bash
uv run python scripts/generate_operator_reference.py
```
"""
    )
    print(f"Generated {len(transformers)} transformer pages and {len(composers)} composer pages")


if __name__ == "__main__":
    main()
