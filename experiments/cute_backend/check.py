"""Check the compiler's style, types, structural limits, and CPU contracts."""

import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path


def nesting(node: ast.AST, level: int = 0) -> int:
    """Count control-flow blocks, treating elif as the same decision level."""
    controls = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Match)
    current = level + int(isinstance(node, controls))
    depths = [current]
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        same_decision = isinstance(node, ast.If) and child in node.orelse
        depths.append(nesting(child, level if same_decision else current))
    return max(depths)


def violations(source: str) -> list[str]:
    """Keep modules within one concern and functions within one screen."""
    errors: list[str] = []
    if len(source.splitlines()) > 300:
        errors.append("module exceeds 300 lines; split its responsibilities")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in ("Any", "object"):
            errors.append(f"line {node.lineno}: replace {node.id} with the concrete contract")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assert node.end_lineno is not None
        if node.end_lineno - node.lineno + 1 > 70:
            errors.append(f"{node.name}: function exceeds 70 lines")
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        count = len([arg for arg in arguments if arg.arg not in ("self", "cls")])
        count += int(node.args.vararg is not None) + int(node.args.kwarg is not None)
        if count > 5:
            errors.append(f"{node.name}: more than 5 arguments")
        if nesting(node) > 3:
            errors.append(f"{node.name}: more than 3 nested control-flow blocks")
    return errors


@dataclass(frozen=True)
class CheckIo:
    root: Path

    def run(self, module: str, arguments: list[str]) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            str(self.root / path) for path in ("python", "experiments/cute_backend")
        )
        subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=self.root,
            env=environment,
            check=True,
        )


def main() -> None:
    io = CheckIo(root=Path(__file__).resolve().parents[2])
    directory = io.root / "experiments/cute_backend"
    compiler = sorted((directory / "tiki_compiler").rglob("*.py"))
    compiler.extend(directory / name for name in ("tiki.py", "associative_scan.py"))
    for path in compiler:
        errors = violations(path.read_text())
        if errors:
            raise AssertionError(f"{path.relative_to(io.root)}: {'; '.join(errors)}")
    if version("ty") != "0.0.79":
        raise RuntimeError("install requirements-check.txt; ty 0.0.79 is required")
    clients = [*sorted(directory.glob("test_*.py")), *sorted(directory.glob("demo_*.py"))]
    targets = [str(path.relative_to(io.root)) for path in (*compiler, *clients, Path(__file__))]
    stubs = str((directory / "typings").relative_to(io.root))
    io.run("ruff", ["check", *targets, stubs])
    io.run("ruff", ["format", "--check", *targets, stubs])
    io.run("ty", ["check", "--config-file", str(directory / "ty.toml"), *targets])
    io.run("unittest", ["discover", "-s", str(directory), "-p", "test_*.py"])


if __name__ == "__main__":
    main()
