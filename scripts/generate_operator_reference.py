"""Generate the authoritative unified Operator catalog from executable examples."""
from __future__ import annotations
import argparse
import inspect
import math
from pathlib import Path
from typing import Any
from bagelquant_core.operator import OPERATOR_REGISTRY
from bagelquant_core._documentation import operation_category, operation_description
from bagelquant_core.operation_examples import operation_example
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "en" / "reference"

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



def generate(check=False):
    contents = {}
    categories = {}
    for runtime in OPERATOR_REGISTRY.names():
        operation = OPERATOR_REGISTRY.get(runtime)
        name = runtime.split(":", 1)[1]+"_prediction" if operation.factory else operation.operation.__name__
        category = "Prediction models" if operation.factory else operation_category(name, kind="operator")
        categories.setdefault(category, []).append(name)
        example = operation_example(name, kind="operator")
        callable_ = operation.factory or operation.operation
        signature = str(inspect.signature(callable_))
        description = inspect.getdoc(callable_).splitlines()[0] if operation.factory else operation_description(name)
        sections = [f"# `{name}`", description, "## Contract", f"Registry: `{runtime}`. Version: `{operation.version}`.",
            f"Peer inputs: {operation.minimum_inputs} to {operation.maximum_inputs or 'N'} typed Nodes. Returns a deferred Node.",
            f"Execution: `{operation.contract.execution.value}`; density: `{operation.contract.density.value}`; traces: `{operation.contract.trace_rule.value}`.",
            "## Numerical signature", f"```python\n{name}{signature}\n```",
            "DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.",
            "## Executable example", f"```python\n{example.call}\n```"]
        for value in example.inputs:
            sections.extend([f"### {value.label}", _markdown_table(value.data)])
        for role, values in example.panel_parameters.items():
            for value in values:
                sections.extend([f"### Auxiliary: {role}", _markdown_table(value.data)])
        sections.extend(["### Output", _markdown_table(example.output.data)])
        contents[REFERENCE/"operators"/f"{name}.md"] = "\n\n".join(sections)+"\n"
    index = "# Operator reference\n\nAll calls accept typed Nodes and return a deferred Node. Scalar parameters never become peer inputs.\n\n"
    index += "\n\n".join(f"## {category}\n\n"+"\n".join(f"- [`{name}`](./{name}.md)" for name in sorted(names)) for category,names in sorted(categories.items()))+"\n"
    contents[REFERENCE/"operators"/"index.md"] = index
    contents[REFERENCE/"index.md"] = "# Core reference\n\n- [Unified Operator catalog](./operators/index.md)\n"
    actual = set(REFERENCE.glob("**/*.md"))
    generated = {path for path in actual if any(part in {"operators"} for part in path.parts)} | {REFERENCE/"index.md"}
    if check:
        if generated != set(contents) or any(not path.exists() or path.read_text() != content for path,content in contents.items()):
            raise RuntimeError("generated Operator reference is stale")
    else:
        for path in generated-set(contents):
            path.unlink(missing_ok=True)
        for path,content in contents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f"Verified {len(contents)-2} Operator pages")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    generate(parser.parse_args().check)
