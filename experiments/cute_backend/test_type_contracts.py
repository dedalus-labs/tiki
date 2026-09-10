"""Prove that the checker rejects invalid compiler states at their call sites."""

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

DECLARATIONS = """
from mlx import core
from tiki_compiler.graph import Node, Operation, Symbol, Value
from tiki_compiler.native import Collective, CollectiveOperation, GroupIndex, Matmul
from tiki_compiler.schedule import Schedule

def construct(stream: core.Stream, group: core.distributed.Group) -> None:
    name = Symbol("value")
    output = Value.dense(name=name, shape=(4,))
"""

VALID = """
    Schedule(threads=128)
    Node(operation=Operation.ADD, inputs=(name, name), output=output)
    Matmul(left=name, right=name, output=output, stream=stream)
    Collective(operation=CollectiveOperation.SUM, input=name, output=output,
               stream=stream, group=group, group_index=GroupIndex(0))
"""

INVALID = """
    Schedule(threads="128")
    Node(operation="Add", inputs=(name, name), output=output)
    Matmul(left=name, right=name, output=output, stream=stream, group_index=None)
    Collective(operation=CollectiveOperation.SUM, input=name, output=output,
               stream=stream, group=group, group_index=None)
"""


class TypeContractTests(unittest.TestCase):
    def test_invalid_states_fail_where_valid_states_compile(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with TemporaryDirectory(prefix="tiki-type-contract-") as directory:
            path = Path(directory) / "contract.py"
            for source, errors in ((VALID, 0), (INVALID, 4)):
                path.write_text(DECLARATIONS + source)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ty",
                        "check",
                        "--config-file",
                        "experiments/cute_backend/ty.toml",
                        "--python",
                        sys.executable,
                        "--output-format",
                        "concise",
                        str(path),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, int(errors > 0), result.stdout + result.stderr)
                self.assertEqual(result.stdout.count("error["), errors, result.stdout)
                self.assertNotIn("unresolved", result.stdout)


if __name__ == "__main__":
    unittest.main()
