"""Dependency-light tests for PNG sequence discovery and ordering."""

import ast
import re
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "inference_png_sequence_semantic.py"


def load_discovery_functions():
    source = ast.parse(SCRIPT.read_text())
    selected = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"natural_sort_key", "find_png_frames"}
    ]
    namespace = {"Path": Path, "re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


class PngSequenceDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.functions = load_discovery_functions()

    def test_numeric_frame_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
                (root / name).touch()
            frames = self.functions["find_png_frames"](root)
            self.assertEqual(
                [path.name for path in frames],
                ["frame_1.png", "frame_2.png", "frame_10.png"],
            )

    def test_case_insensitive_png_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "000001.PNG").touch()
            frames = self.functions["find_png_frames"](root)
            self.assertEqual([path.name for path in frames], ["000001.PNG"])

    def test_rejects_multiple_frame_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("sequence_a", "sequence_b"):
                path = root / folder / "000001.png"
                path.parent.mkdir()
                path.touch()
            with self.assertRaisesRegex(RuntimeError, "Multiple PNG frame directories"):
                self.functions["find_png_frames"](root)


if __name__ == "__main__":
    unittest.main()
