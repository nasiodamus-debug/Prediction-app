"""predictions/core may import only the standard library."""
import ast
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "predictions" / "core"


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield "predictions.core"  # relative import stays inside core
            else:
                yield node.module or ""


def test_core_imports_only_standard_library_or_core():
    offenders = []
    for path in CORE.rglob("*.py"):
        for name in _imports(path):
            top = name.split(".")[0]
            if name == "predictions.core" or name.startswith("predictions.core."):
                continue
            if top not in sys.stdlib_module_names:
                offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders


def test_core_files_exist():
    assert {p.name for p in CORE.glob("*.py")} >= {"types.py", "ports.py", "dates.py", "schedule.py"}
