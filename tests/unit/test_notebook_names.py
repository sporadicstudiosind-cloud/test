"""Every notebook name must be defined before it is used.

The failure this prevents: `DEVICE` was assigned only inside the *optional* TPU
cell, so running the cells in any order that skipped it — or simply running the
one you wanted — killed everything downstream with ``NameError: name 'DEVICE'
is not defined``, pointing at a line that was perfectly correct.

An optional cell must never own a name the rest of the notebook needs. This
walks each notebook cell in order, tracking what has been assigned, and fails on
the first load of a name nothing has defined yet. It is static: it cannot catch
a name that only exists on some runtime branch, but it catches the whole class
of ordering mistakes that this notebook format invites.
"""

import ast
import builtins
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "notebooks").glob("*.ipynb"))
#: The studio notebooks share a generator and a DEVICE convention; the older
#: fixed-rung trainer predates both and selects its device differently.
STUDIO = [p for p in NOTEBOOKS if p.name.startswith("iridium_studio")]

#: Names the notebook environment provides that no cell assigns.
AMBIENT = {"get_ipython", "_", "__", "___", "In", "Out", "exit", "quit", "display"}


def _assigned_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    out.add(target.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.Global):
            out.update(node.names)
    return out


def _python_source(cell: dict) -> str:
    """Cell source with IPython magics and Colab #@param markers removed."""
    src = "".join(cell["source"])
    src = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
    )
    return re.sub(r"\s*#@param.*$", "", src, flags=re.M)


def test_there_are_notebooks_to_check():
    assert NOTEBOOKS, "no notebooks found — this test would pass vacuously"
    assert STUDIO, "no studio notebooks found — the DEVICE check would be vacuous"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_with_cells(path):
    nb = json.loads(path.read_text())
    assert nb.get("cells"), f"{path.name} has no cells"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_cell_parses(path):
    nb = json.loads(path.read_text())
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(_python_source(cell))
        except SyntaxError as exc:
            pytest.fail(f"{path.name} cell {index} does not parse: {exc}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_name_is_used_before_it_is_defined(path):
    nb = json.loads(path.read_text())
    defined = set(dir(builtins)) | AMBIENT
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(_python_source(cell))
        used = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assigned = _assigned_names(tree)
        missing = used - assigned - defined
        assert not missing, (
            f"{path.name} cell {index} uses {sorted(missing)} before anything "
            f"defines it — an earlier cell owns the name, and skipping or "
            f"reordering that cell breaks this one"
        )
        defined |= assigned


@pytest.mark.parametrize("path", STUDIO, ids=lambda p: p.name)
def test_device_is_defined_before_the_optional_accelerator_cell(path):
    """The specific regression, pinned by name.

    ``DEVICE`` has to come from a cell that always runs. The TPU cell may
    override it; it must not be the only thing that sets it.
    """
    nb = json.loads(path.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    setters = [
        i for i, c in enumerate(code)
        if re.search(r"^\s*DEVICE\s*=", _python_source(c), re.M)
    ]
    assert setters, f"{path.name} never assigns DEVICE"
    first = code[setters[0]]
    source = _python_source(first)
    assert "FORCE_XLA" not in source, (
        f"{path.name}: the first cell to set DEVICE is the optional accelerator "
        f"cell. Skip it and every later cell fails with NameError."
    )
