from bagelquant_core._documentation import OPERATION_DESCRIPTIONS, operation_category
from bagelquant_core.operator import OPERATOR_REGISTRY
from bagelquant_core.operation_examples import operation_example


def test_unified_catalog_has_functional_categories_and_executable_examples():
    names = set()
    for runtime in OPERATOR_REGISTRY.names():
        operation = OPERATOR_REGISTRY.get(runtime)
        if not operation.operation.__module__.startswith("bagelquant_core."):
            continue
        name = runtime.split(":", 1)[1]+"_prediction" if operation.factory else operation.operation.__name__
        assert name not in names
        names.add(name)
        if not operation.factory:
            assert name in OPERATION_DESCRIPTIONS
        example = operation_example(name, kind="operator")
        assert example.inputs
        assert not example.output.data.is_empty()
        assert operation_category(name, kind="operator")
    assert {"rolling_mean", "add", "group_zscore", "identity_prediction"} <= names
