from pathlib import Path

import bagelquant_core


def test_package_imports_from_src_layout() -> None:
    package_path = Path(bagelquant_core.__file__).resolve()

    assert package_path.parts[-3:] == ("src", "bagelquant_core", "__init__.py")
