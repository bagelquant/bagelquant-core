"""Safe numerical DSL compiler shared by Python packages and application editors."""
from __future__ import annotations
import ast
import keyword
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal
from inspect import signature, Parameter
from types import SimpleNamespace


@dataclass(frozen=True)
class DslDiagnostic:
    line: int
    column: int
    message: str
    severity: str = "error"


class DslError(ValueError):
    def __init__(self, message, *, line, column=1):
        self.diagnostic = DslDiagnostic(line, column, message)
        super().__init__(f"line {line}, column {column}: {message}")


@dataclass(frozen=True)
class _CompiledExpression:
    kind: Literal["panel", "scalar"]
    value: Any


def compile_dsl(
    source: str,
    *,
    source_names: set[str],
    parameter_names: set[str],
    allow_parameters: bool,
    operation_specs: Mapping[str, Any],
    validate_parameters,
    positions=None,
    macro_dependencies=lambda name: (),
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    operation_specs = dict(operation_specs)
    referenced: list[str] = []
    nodes: list[dict[str, Any]] = []
    resolved: set[str] = set()
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise DslError(
            error.msg,
            line=int(error.lineno or 1),
            column=int(error.offset or 1),
        ) from error

    assignments: list[ast.Assign] = []
    explicit_names: set[str] = set()
    for statement in module.body:
        line_number = int(getattr(statement, "lineno", 1))
        if not isinstance(statement, ast.Assign):
            raise DslError(
                "each top-level statement must be one node assignment",
                line=line_number,
                column=int(getattr(statement, "col_offset", 0)) + 1,
            )
        if (
            len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or keyword.iskeyword(statement.targets[0].id)
        ):
            raise DslError(
                "the assignment target must be a node name",
                line=line_number,
                column=int(getattr(statement, "col_offset", 0)) + 1,
            )
        node_name = statement.targets[0].id
        if node_name in explicit_names or node_name in source_names:
            raise DslError(
                f"duplicate node or source name {node_name!r}",
                line=line_number,
            )
        explicit_names.add(node_name)
        assignments.append(statement)

    reserved_names = {*source_names, *explicit_names}

    def synthetic_name(expression: ast.AST, *, suffix: str = "expr") -> str:
        base = (
            f"__dsl_{int(getattr(expression, 'lineno', 1))}_"
            f"{int(getattr(expression, 'col_offset', 0)) + 1}_{suffix}"
        )
        candidate = base
        number = 2
        while candidate in reserved_names or candidate in resolved:
            candidate = f"{base}_{number}"
            number += 1
        return candidate

    def add_reference(name: str) -> None:
        if name in source_names and name not in referenced:
            referenced.append(name)

    def panel_reference(expression: ast.expr) -> str:
        compiled = compile_expression(expression)
        if compiled.kind != "panel":
            raise DslError(
                "Node inputs must be sources, earlier nodes, or Node expressions",
                line=int(getattr(expression, "lineno", 1)),
                column=int(getattr(expression, "col_offset", 0)) + 1,
            )
        return str(compiled.value)

    def panel_parameter_references(expression: ast.expr) -> list[str]:
        values = (
            list(expression.elts)
            if isinstance(expression, (ast.List, ast.Tuple))
            else [expression]
        )
        return [panel_reference(value) for value in values]

    def append_node(
        *,
        node_id: str,
        specification: Any,
        inputs: list[str],
        parameters: dict[str, Any],
        panel_parameters: dict[str, list[str]] | None = None,
        expression: ast.AST,
    ) -> _CompiledExpression:
        supplied_panels = dict(panel_parameters or {})
        try:
            validate_parameters(
                specification.name,
                inputs,
                parameters,
                panel_parameters=supplied_panels,
            )
        except (TypeError, ValueError) as error:
            raise DslError(
                str(error),
                line=int(getattr(expression, "lineno", 1)),
                column=int(getattr(expression, "col_offset", 0)) + 1,
            ) from error
        if positions is not None:
            positions[node_id] = (int(getattr(expression, "lineno", 1)), int(getattr(expression, "col_offset", 0)) + 1)
        nodes.append(
            {
                "id": node_id,
                "operation": specification.name,
                "inputs": inputs,
                "panel_parameters": supplied_panels,
                "parameters": parameters,
            }
        )
        resolved.add(node_id)
        return _CompiledExpression("panel", node_id)

    def compile_call(
        call: ast.Call,
        *,
        output_id: str | None,
    ) -> _CompiledExpression:
        if not isinstance(call.func, ast.Name):
            raise DslError(
                "operation calls must use a governed operation name",
                line=int(getattr(call, "lineno", 1)),
                column=int(getattr(call, "col_offset", 0)) + 1,
            )
        operation_name = call.func.id
        specification = operation_specs.get(operation_name)
        if specification is None:
            raise DslError(
                f"unknown operation {operation_name!r}",
                line=int(getattr(call.func, "lineno", 1)),
                column=int(getattr(call.func, "col_offset", 0)) + 1,
            )

        inputs: list[str] = []
        positional_parameters: list[ast.expr] = []
        if specification.maximum_inputs is None:
            parameter_phase = False
            for argument in call.args:
                if specification.parameters and _is_scalar_syntax(argument):
                    parameter_phase = True
                if parameter_phase:
                    if not _is_scalar_syntax(argument):
                        raise DslError(
                            "Node input cannot follow a positional operation parameter",
                            line=int(getattr(argument, "lineno", 1)),
                            column=int(getattr(argument, "col_offset", 0)) + 1,
                        )
                    positional_parameters.append(argument)
                else:
                    inputs.append(panel_reference(argument))
        else:
            input_count = min(len(call.args), specification.maximum_inputs)
            inputs.extend(panel_reference(value) for value in call.args[:input_count])
            positional_parameters.extend(call.args[input_count:])

        if len(positional_parameters) > len(specification.parameters):
            raise DslError(
                f"{specification.label} received too many positional parameters",
                line=int(getattr(positional_parameters[len(specification.parameters)], "lineno", 1)),
                column=int(getattr(positional_parameters[len(specification.parameters)], "col_offset", 0)) + 1,
            )
        parameters = {
            specification.parameters[index].name: _literal(
                argument,
                line_number=int(getattr(argument, "lineno", 1)),
                allow_parameters=allow_parameters,
                parameter_names=parameter_names,
            )
            for index, argument in enumerate(positional_parameters)
        }
        panel_parameters: dict[str, list[str]] = {}
        declared_panel_parameters = {
            item.name for item in specification.panel_parameters
        }
        for keyword_argument in call.keywords:
            if keyword_argument.arg is None:
                raise DslError(
                    "expanded keyword arguments are not supported",
                    line=int(getattr(keyword_argument.value, "lineno", 1)),
                    column=int(getattr(keyword_argument.value, "col_offset", 0)) + 1,
                )
            name = keyword_argument.arg
            if name in parameters or name in panel_parameters:
                raise DslError(
                    f"duplicate operation parameter {name!r}",
                    line=int(getattr(keyword_argument.value, "lineno", 1)),
                    column=int(getattr(keyword_argument.value, "col_offset", 0)) + 1,
                )
            if name in declared_panel_parameters:
                panel_parameters[name] = panel_parameter_references(
                    keyword_argument.value
                )
            else:
                parameters[name] = _literal(
                    keyword_argument.value,
                    line_number=int(getattr(keyword_argument.value, "lineno", 1)),
                    allow_parameters=allow_parameters,
                    parameter_names=parameter_names,
                )

        for fixed_source in macro_dependencies(specification.name):
            if fixed_source not in source_names:
                raise DslError(
                    f"{operation_name} requires active system DataItem "
                    f"{fixed_source!r}",
                    line=int(getattr(call.func, "lineno", 1)),
                    column=int(getattr(call.func, "col_offset", 0)) + 1,
                )
            add_reference(fixed_source)
        node_id = output_id or synthetic_name(call, suffix=operation_name)
        return append_node(
            node_id=node_id,
            specification=specification,
            inputs=inputs,
            parameters=parameters,
            panel_parameters=panel_parameters,
            expression=call.func,
        )

    def lift_scalar(
        scalar: _CompiledExpression,
        *,
        seed: str,
        expression: ast.expr,
    ) -> str:
        specification = operation_specs["constant"]
        node_id = synthetic_name(expression, suffix="constant")
        return str(
            append_node(
                node_id=node_id,
                specification=specification,
                inputs=[seed],
                parameters={"value": scalar.value},
                expression=expression,
            ).value
        )

    def compile_binary(
        expression: ast.BinOp,
        *,
        output_id: str | None,
    ) -> _CompiledExpression:
        operation_name = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
        }.get(type(expression.op))
        if operation_name is None:
            raise DslError(
                "only +, -, *, and / arithmetic operators are supported",
                line=int(getattr(expression, "lineno", 1)),
                column=int(getattr(expression, "col_offset", 0)) + 1,
            )
        left = compile_expression(expression.left)
        right = compile_expression(expression.right)
        if left.kind == right.kind == "scalar":
            raise DslError(
                "DSL arithmetic must include at least one Node expression",
                line=int(getattr(expression, "lineno", 1)),
                column=int(getattr(expression, "col_offset", 0)) + 1,
            )
        if left.kind == "scalar":
            right_panel = str(right.value)
            left = _CompiledExpression(
                "panel",
                lift_scalar(left, seed=right_panel, expression=expression.left),
            )
        if right.kind == "scalar":
            left_panel = str(left.value)
            right = _CompiledExpression(
                "panel",
                lift_scalar(right, seed=left_panel, expression=expression.right),
            )
        specification = operation_specs.get(operation_name)
        if specification is None:
            raise DslError(
                "arithmetic operators are not available for this DSL kind",
                line=int(getattr(expression, "lineno", 1)),
                column=int(getattr(expression, "col_offset", 0)) + 1,
            )
        return append_node(
            node_id=output_id or synthetic_name(expression, suffix=operation_name),
            specification=specification,
            inputs=[str(left.value), str(right.value)],
            parameters={},
            expression=expression,
        )

    def compile_unary(
        expression: ast.UnaryOp,
        *,
        output_id: str | None,
    ) -> _CompiledExpression:
        if _is_scalar_syntax(expression):
            return _CompiledExpression(
                "scalar",
                _literal(
                    expression,
                    line_number=int(getattr(expression, "lineno", 1)),
                    allow_parameters=allow_parameters,
                    parameter_names=parameter_names,
                ),
            )
        operand = compile_expression(expression.operand)
        if operand.kind != "panel":
            raise DslError(
                "unary arithmetic requires a Node or numeric value",
                line=int(getattr(expression, "lineno", 1)),
                column=int(getattr(expression, "col_offset", 0)) + 1,
            )
        if isinstance(expression.op, ast.UAdd):
            if output_id is None:
                return operand
            return append_node(
                node_id=output_id,
                specification=_required_operation_spec(
                    operation_specs, "identity", expression
                ),
                inputs=[str(operand.value)],
                parameters={},
                expression=expression,
            )
        if isinstance(expression.op, ast.USub):
            return append_node(
                node_id=output_id or synthetic_name(expression, suffix="negate"),
                specification=_required_operation_spec(
                    operation_specs, "negate", expression
                ),
                inputs=[str(operand.value)],
                parameters={},
                expression=expression,
            )
        raise DslError(
            "only unary + and - are supported",
            line=int(getattr(expression, "lineno", 1)),
            column=int(getattr(expression, "col_offset", 0)) + 1,
        )

    def compile_expression(
        expression: ast.expr,
        *,
        output_id: str | None = None,
    ) -> _CompiledExpression:
        if isinstance(expression, ast.Name):
            if expression.id not in source_names and expression.id not in resolved:
                raise DslError(
                    f"unknown or forward-referenced input {expression.id!r}",
                    line=int(getattr(expression, "lineno", 1)),
                    column=int(getattr(expression, "col_offset", 0)) + 1,
                )
            add_reference(expression.id)
            return _CompiledExpression("panel", expression.id)
        if isinstance(expression, ast.Call):
            return compile_call(expression, output_id=output_id)
        if isinstance(expression, ast.BinOp):
            return compile_binary(expression, output_id=output_id)
        if isinstance(expression, ast.UnaryOp):
            return compile_unary(expression, output_id=output_id)
        if _is_scalar_syntax(expression):
            return _CompiledExpression(
                "scalar",
                _literal(
                    expression,
                    line_number=int(getattr(expression, "lineno", 1)),
                    allow_parameters=allow_parameters,
                    parameter_names=parameter_names,
                ),
            )
        raise DslError(
            "unsupported DSL expression",
            line=int(getattr(expression, "lineno", 1)),
            column=int(getattr(expression, "col_offset", 0)) + 1,
        )

    for assignment in assignments:
        line_number = int(getattr(assignment, "lineno", 1))
        node_name = assignment.targets[0].id
        if isinstance(assignment.value, ast.Name) or _is_scalar_syntax(assignment.value):
            raise DslError(
                "the right side must produce a Node operation",
                line=line_number,
                column=int(getattr(assignment.value, "col_offset", 0)) + 1,
            )
        compiled = compile_expression(assignment.value, output_id=node_name)
        if compiled.kind != "panel" or compiled.value != node_name:
            raise DslError(
                "the right side must produce a named Node operation",
                line=line_number,
                column=int(getattr(assignment.value, "col_offset", 0)) + 1,
            )
    if not nodes:
        raise DslError("formula must contain at least one node assignment", line=1)
    return nodes, tuple(referenced)


def _required_operation_spec(
    operation_specs: Mapping[str, Any],
    name: str,
    expression: ast.AST,
) -> Any:
    specification = operation_specs.get(name)
    if specification is None:
        raise DslError(
            f"operation {name!r} is not available for this DSL kind",
            line=int(getattr(expression, "lineno", 1)),
            column=int(getattr(expression, "col_offset", 0)) + 1,
        )
    return specification


def _is_scalar_syntax(node: ast.expr) -> bool:
    if isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return True
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
    ):
        return True
    return isinstance(node, ast.UnaryOp) and _is_scalar_syntax(node.operand)


def _literal(
    node: ast.expr,
    *,
    line_number: int,
    allow_parameters: bool,
    parameter_names: set[str],
) -> Any:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "params":
        if not allow_parameters:
            raise DslError(
                "dynamic parameters are not supported for this definition kind",
                line=line_number,
                column=int(node.col_offset) + 1,
            )
        if node.attr not in parameter_names:
            raise DslError(
                f"unknown declared parameter {node.attr!r}",
                line=line_number,
                column=int(node.col_offset) + 1,
            )
        return {"$parameter": node.attr}
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise DslError(
            "parameter values must be literals or declared params.<name>",
            line=line_number,
            column=int(getattr(node, "col_offset", 0)) + 1,
        ) from error
    if isinstance(value, (tuple, dict, set, bytes, complex)):
        raise DslError("unsupported parameter literal", line=line_number)
    return value



def parse_dsl(source, *, inputs, parameters=None, operators=None):
    """Construct a local Graph with explicitly bound or symbolic source Nodes."""
    from .operator import OPERATOR_REGISTRY
    from .node import Node
    from .graph import Graph
    namespace = {}
    for key in OPERATOR_REGISTRY.names():
        operation = OPERATOR_REGISTRY.get(key)
        name = key.split(":", 1)[1] + "_prediction" if key.startswith("prediction:") else operation.display_name
        if name in namespace:
            raise ValueError(f"ambiguous DSL operation: {name}")
        namespace[name] = operation
    namespace.update(operators or {})
    specs = {}
    for name, operation in namespace.items():
        callable_ = operation.factory or operation.operation
        scalar = [SimpleNamespace(name=p.name) for p in signature(callable_).parameters.values()
                  if (operation.factory is not None or p.kind == Parameter.KEYWORD_ONLY) and p.name not in operation.panel_parameter_kinds]
        specs[name] = SimpleNamespace(name=name, label=name, parameters=scalar,
            panel_parameters=[SimpleNamespace(name=role) for role in operation.panel_parameter_kinds],
            maximum_inputs=operation.maximum_inputs)
    def validate(name, peers, config, *, panel_parameters):
        operation = namespace[name]
        operation.validate_input_count(len(peers))
        if operation.factory is None:
            signature(operation.operation).bind(*[object() for _ in peers], **config,
                **{role: object() for role in panel_parameters})
    positions = {}
    compiled, references = compile_dsl(source, source_names=set(inputs), positions=positions,
        parameter_names=set(parameters or {}), allow_parameters=True,
        operation_specs=specs, validate_parameters=validate)
    values = {name: value if isinstance(value, Node) else Node.symbolic(name, value_type=value) for name, value in inputs.items()}
    def resolve(value):
        if isinstance(value, dict) and set(value) == {"$parameter"}:
            return parameters[value["$parameter"]]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [resolve(item) for item in value]
        return value
    for item in compiled:
        operation = namespace[item["operation"]]
        config = resolve(item["parameters"])
        auxiliary = {role: tuple(values[key] for key in keys) if operation.panel_parameter_kinds[role] else values[keys[0]]
                     for role, keys in item.get("panel_parameters", {}).items()}
        peers = [values[key] for key in item["inputs"]]
        try:
            values[item["id"]] = operation(*peers, name=item["id"], **auxiliary, **config)
        except (ValueError, TypeError) as error:
            raise DslError(str(error), line=positions[item["id"]][0], column=positions[item["id"]][1]) from error
    return Graph(outputs=(values[compiled[-1]["id"]],))
