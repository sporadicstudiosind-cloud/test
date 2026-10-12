"""Static validation of the notebooks that cannot rot silently.

Nothing here executes a notebook (no network, no training, no GPU needed to
run the test suite). Every check is static: nbformat shape, Python syntax
after stripping magics, that every ``iridium`` import actually resolves
against the *current* package, that every preset name and CLI subcommand a
notebook mentions still exists, and that ``notebooks/build_notebooks.py``
still produces byte-identical output for every notebook it owns -- so a
hand-edit to a generated ``.ipynb`` (a divergence from its own source) fails
loudly instead of quietly drifting.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
BUILD_SCRIPT = NOTEBOOKS_DIR / "build_notebooks.py"


def _python_source(cell: dict) -> str:
    """Cell source with IPython magics and shell lines stripped."""
    src = "".join(cell["source"])
    src = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
    )
    return re.sub(r"\s*#@param.*$", "", src, flags=re.M)


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c.get("cell_type") == "code"]


def test_there_are_notebooks_to_check():
    assert NOTEBOOKS, "no notebooks found under notebooks/ -- this suite would pass vacuously"


# --------------------------------------------------------------------------
# nbformat shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_nbformat_json(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, f"{path.name}: expected nbformat 4"
    assert "nbformat_minor" in nb
    assert isinstance(nb.get("metadata"), dict)
    cells = nb.get("cells")
    assert isinstance(cells, list) and cells, f"{path.name} has no cells"
    for index, cell in enumerate(cells):
        assert cell.get("cell_type") in ("code", "markdown"), \
            f"{path.name} cell {index} has an unknown cell_type"
        assert isinstance(cell.get("source"), list), \
            f"{path.name} cell {index}'s source must be a list of lines"
        if cell["cell_type"] == "code":
            assert "outputs" in cell and cell["outputs"] == [], \
                f"{path.name} cell {index} is a code cell with baked-in outputs"
            assert cell.get("execution_count") is None, \
                f"{path.name} cell {index} has a stale execution_count"


# --------------------------------------------------------------------------
# Every code cell compiles
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_compiles(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(_code_cells(nb)):
        source = _python_source(cell)
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{path.name} cell {index} does not compile: {exc}\n---\n{source}")


# --------------------------------------------------------------------------
# Every iridium import resolves against the current package
# --------------------------------------------------------------------------

def _iridium_imports(tree: ast.AST) -> list[tuple[str, list[str]]]:
    """[(module, [imported names or empty for a bare `import module`])]."""
    out: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "iridium":
            out.append((node.module, [a.name for a in node.names if a.name != "*"]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "iridium":
                    out.append((alias.name, []))
    return out


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_iridium_imports_resolve(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(_code_cells(nb)):
        source = _python_source(cell)
        tree = ast.parse(source)
        for module_name, names in _iridium_imports(tree):
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                pytest.fail(f"{path.name} cell {index}: `import {module_name}` fails: {exc}")
            for name in names:
                assert hasattr(module, name), (
                    f"{path.name} cell {index}: `from {module_name} import {name}` -- "
                    f"{module_name} has no attribute {name!r}"
                )


# --------------------------------------------------------------------------
# Every referenced preset name exists
# --------------------------------------------------------------------------

_PRESET_REF = re.compile(
    r"""(?:get_preset\(\s*['"]([\w-]+)['"]|
        \b[A-Z_]*PRESET[A-Z_]*\s*=\s*['"]([\w-]+)['"])""",
    re.VERBOSE,
)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_referenced_presets_exist(path):
    from iridium.presets import PRESETS

    nb = json.loads(path.read_text(encoding="utf-8"))
    found = set()
    for cell in _code_cells(nb):
        source = _python_source(cell)
        for m in _PRESET_REF.finditer(source):
            name = m.group(1) or m.group(2)
            if name:
                found.add(name)
    unknown = found - set(PRESETS)
    assert not unknown, f"{path.name} references unknown preset(s) {sorted(unknown)}"


def test_at_least_one_notebook_references_a_real_preset():
    """The regex above would pass vacuously on a notebook with no presets in it."""
    from iridium.presets import PRESETS

    total = set()
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in _code_cells(nb):
            for m in _PRESET_REF.finditer(_python_source(cell)):
                name = m.group(1) or m.group(2)
                if name:
                    total.add(name)
    assert total & set(PRESETS), "no notebook references any known preset by name"


# --------------------------------------------------------------------------
# Every referenced CLI subcommand exists
# --------------------------------------------------------------------------

_CLI_INVOCATION = re.compile(r"python\s+-m\s+iridium\s+([a-zA-Z][\w-]*)")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_referenced_cli_subcommands_exist(path):
    from iridium.cli import build_parser

    parser = build_parser()
    subparsers_action = next(
        a for a in parser._subparsers._group_actions  # type: ignore[attr-defined]
        if hasattr(a, "choices")
    )
    known = set(subparsers_action.choices)

    nb = json.loads(path.read_text(encoding="utf-8"))
    referenced = set()
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        referenced.update(_CLI_INVOCATION.findall(text))
    unknown = referenced - known
    assert not unknown, f"{path.name} references unknown `iridium` subcommand(s) {sorted(unknown)}"


def test_at_least_one_notebook_references_a_real_subcommand():
    from iridium.cli import build_parser

    parser = build_parser()
    subparsers_action = next(
        a for a in parser._subparsers._group_actions  # type: ignore[attr-defined]
        if hasattr(a, "choices")
    )
    known = set(subparsers_action.choices)
    total = set()
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            total.update(_CLI_INVOCATION.findall("".join(cell["source"])))
    assert total & known, "no notebook references any real `iridium` CLI subcommand"


# --------------------------------------------------------------------------
# build_notebooks.py is the one source of truth
# --------------------------------------------------------------------------

def test_build_notebooks_regenerates_byte_identically(tmp_path):
    """Every generated notebook must equal what the builder would write now.

    Runs the real builder against a temp copy of the ``notebooks`` directory
    name mapping (by monkeypatching its ``HERE``), so this test never writes
    into the working tree, and fails if a ``.ipynb`` was hand-edited after
    being generated, or if the builder's output has drifted from what is
    checked in.
    """
    module_name = "notebooks.build_notebooks"
    spec_path = BUILD_SCRIPT
    assert spec_path.is_file(), "notebooks/build_notebooks.py is missing"

    globs = runpy.run_path(str(spec_path), run_name="__not_main__")
    builder_here = globs["HERE"]
    assert builder_here == NOTEBOOKS_DIR

    # Regenerate into tmp_path by pointing the module's HERE there, then diff.
    src = spec_path.read_text(encoding="utf-8")
    patched = src.replace(
        "HERE = Path(__file__).resolve().parent",
        f"HERE = Path({str(tmp_path)!r})",
        1,
    )
    assert patched != src, "could not patch HERE in build_notebooks.py; check its source layout"
    tmp_script = tmp_path / "build_notebooks.py"
    tmp_script.write_text(patched, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=tmp_path)

    generated = sorted(p.name for p in tmp_path.glob("*.ipynb"))
    checked_in = sorted(p.name for p in NOTEBOOKS_DIR.glob("*.ipynb"))
    assert generated == checked_in, (
        "build_notebooks.py's targets no longer match the notebooks in the "
        f"directory: generated={generated} checked_in={checked_in}"
    )
    for name in generated:
        expected = (tmp_path / name).read_bytes()
        actual = (NOTEBOOKS_DIR / name).read_bytes()
        assert actual == expected, (
            f"{name} does not match what build_notebooks.py generates now -- "
            "run `python notebooks/build_notebooks.py` and commit the result, "
            "or the notebook was hand-edited and has drifted from its source"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
