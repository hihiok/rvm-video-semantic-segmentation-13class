"""Dependency-light tests for PNG sequence discovery and ordering."""

import ast
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None


SCRIPT = Path(__file__).resolve().parents[1] / "inference_png_sequence_semantic.py"


def load_functions(names):
    source = ast.parse(SCRIPT.read_text())
    selected = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]
    namespace = {
        "Path": Path,
        "re": re,
        "np": np,
        "cv2": cv2,
        "torch": torch,
        "OTHER_CLASS_NAME": "other",
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


class PngSequenceDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.functions = load_functions({"natural_sort_key", "find_png_frames"})

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


class OutputClassResolutionTests(unittest.TestCase):
    def setUp(self):
        self.resolve = load_functions({"resolve_output_classes"})[
            "resolve_output_classes"
        ]
        self.class_names = [
            "background", "sky", "person", "plant", "building", "flower",
            "food", "water", "desert", "ice_or_snow", "text", "ball",
            "mountain",
        ]

    def test_sky_water_mountain_other_mapping(self):
        spec = self.resolve(
            ["sky", "water", "mountain", "other"], self.class_names
        )
        self.assertEqual(spec["selected_indices"], [1, 7, 12])
        self.assertTrue(spec["include_other"])

    def test_unknown_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown output classes"):
            self.resolve(["sky", "ocean"], self.class_names)

    def test_background_and_other_are_rejected_together(self):
        with self.assertRaisesRegex(ValueError, "cannot be requested together"):
            self.resolve(["background", "sky", "other"], self.class_names)


@unittest.skipIf(torch is None, "torch is not installed in the lightweight test env")
class RestrictedPredictionTests(unittest.TestCase):
    def setUp(self):
        names = {
            "restricted_mask_from_probabilities",
            "update_probability_ema",
        }
        self.functions = load_functions(names)
        self.spec = {
            "selected_indices": [1, 7, 12],
            "include_other": True,
        }

    def probabilities(self, sky=0.0, water=0.0, mountain=0.0, person=0.0):
        values = torch.zeros((1, 13, 1, 1), dtype=torch.float32)
        values[0, 1, 0, 0] = sky
        values[0, 2, 0, 0] = person
        values[0, 7, 0, 0] = water
        values[0, 12, 0, 0] = mountain
        return values

    def test_excluded_top1_maps_by_requested_class_confidence(self):
        probabilities = self.probabilities(sky=0.25, water=0.30, person=0.40)
        mask = self.functions["restricted_mask_from_probabilities"](
            probabilities, self.spec, 0.20
        )
        self.assertEqual(mask.item(), 7)

    def test_low_requested_confidence_becomes_other(self):
        probabilities = self.probabilities(sky=0.12, water=0.18, person=0.60)
        mask = self.functions["restricted_mask_from_probabilities"](
            probabilities, self.spec, 0.20
        )
        self.assertEqual(mask.item(), 0)

    def test_hysteresis_holds_then_allows_class_change(self):
        function = self.functions["restricted_mask_from_probabilities"]
        previous = torch.tensor([[[1]]], dtype=torch.long)
        near_tie = self.probabilities(sky=0.38, water=0.40)
        held = function(near_tie, self.spec, 0.20, previous, 0.05)
        self.assertEqual(held.item(), 1)

        decisive = self.probabilities(sky=0.20, water=0.50)
        changed = function(decisive, self.spec, 0.20, previous, 0.05)
        self.assertEqual(changed.item(), 7)

    def test_ema_uses_current_frame_alpha(self):
        current = self.probabilities(sky=1.0)
        previous = self.probabilities(water=1.0)
        result = self.functions["update_probability_ema"](
            current, previous, 0.25
        )
        self.assertAlmostEqual(result[0, 1, 0, 0].item(), 0.25)
        self.assertAlmostEqual(result[0, 7, 0, 0].item(), 0.75)


@unittest.skipIf(cv2 is None, "opencv is not installed in the lightweight test env")
class HistogramSceneCutTests(unittest.TestCase):
    def setUp(self):
        self.score = load_functions({"histogram_scene_cut_score"})[
            "histogram_scene_cut_score"
        ]

    def test_identical_frames_have_zero_distance(self):
        frame = np.full((90, 160, 3), (20, 80, 200), dtype=np.uint8)
        _, histogram = self.score(None, frame)
        score, _ = self.score(histogram, frame.copy())
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_black_to_white_is_a_hard_cut(self):
        black = np.zeros((90, 160, 3), dtype=np.uint8)
        white = np.full((90, 160, 3), 255, dtype=np.uint8)
        _, histogram = self.score(None, black)
        score, _ = self.score(histogram, white)
        self.assertGreater(score, 0.90)


if __name__ == "__main__":
    unittest.main()
