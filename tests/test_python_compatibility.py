from pathlib import Path
import ast


def test_runtime_code_parses_with_python310_grammar() -> None:
    roots = [Path("continuum_robot"), Path("scripts")]
    paths = [path for root in roots for path in root.rglob("*.py")]
    assert paths

    for path in paths:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path), feature_version=(3, 10))
