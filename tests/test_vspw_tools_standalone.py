"""Dependency-light integration coverage runnable without PyTorch or pytest."""

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
spec = importlib.util.spec_from_file_location("resolve_static_dataset_for_tests", TOOLS / "resolve_static_dataset.py")
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def load_pure_training_functions():
    source = ast.parse((REPO / "train_vspw_mixed.py").read_text())
    selected = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"mixed_batch_sources", "stage_for_epoch", "balanced_score"}
    ]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "train_vspw_mixed.py", "exec"), namespace)
    return namespace


def write_pair(root, split, filename, label, video_name=None):
    relative = Path(video_name) / filename if video_name else Path(filename)
    image = root / "images" / split / relative.with_suffix(".jpg")
    mask = root / "annotations" / split / relative.with_suffix(".png")
    image.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 18, 3), 128, dtype=np.uint8)).save(image)
    Image.fromarray(np.full((12, 18), label, dtype=np.uint8)).save(mask)
    return image, mask


class StaticResolverTests(unittest.TestCase):
    def test_prefers_13class_coco_ade_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            preferred = parent / "coco_ADE_13class"
            previous = parent / "coco_ADE_12class"
            for root in (preferred, previous):
                write_pair(root, "train", "train", 1)
                write_pair(root, "val", "val", 12)
            result = resolver.discover_candidates([parent])
            self.assertEqual(result[0]["root"], str(preferred))

    def test_fails_loudly_for_tied_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for name in ("coco_ADE_13class_a", "coco_ADE_13class_b"):
                root = parent / name
                write_pair(root, "train", "train", 1)
                write_pair(root, "val", "val", 1)
            process = subprocess.run(
                [sys.executable, str(TOOLS / "resolve_static_dataset.py"), "--search-root", str(parent)],
                text=True, capture_output=True,
            )
            self.assertEqual(process.returncode, 3)
            self.assertIn("STATIC_ROOT explicitly", process.stderr)


class CurriculumTests(unittest.TestCase):
    def test_every_video_group_receives_static_replay(self):
        functions = load_pure_training_functions()
        self.assertEqual(
            list(functions["mixed_batch_sources"](5, 2, 1)),
            ["video", "video", "static", "video", "video", "static", "video", "static"],
        )

    def test_stage_transition_switches_clip_and_ratio(self):
        functions = load_pure_training_functions()
        args = SimpleNamespace(
            stage2_epochs=2,
            stage2_clip_length=5,
            stage2_video_batches=1,
            stage2_static_batches=1,
            stage3_clip_length=8,
            stage3_video_batches=2,
            stage3_static_batches=1,
        )
        self.assertEqual(functions["stage_for_epoch"](args, 1)["clip_length"], 5)
        stage3 = functions["stage_for_epoch"](args, 2)
        self.assertEqual((stage3["clip_length"], stage3["video_batches"]), (8, 2))

    def test_balanced_score_combines_both_domains(self):
        functions = load_pure_training_functions()
        self.assertAlmostEqual(functions["balanced_score"]({"miou": 0.8}, {"miou": 0.6}, 0.5), 0.7)


class MixedAuditTests(unittest.TestCase):
    def create_datasets(self, parent, leak=False, invalid=False):
        vspw, static = parent / "VSPW_13cls", parent / "coco_ADE_13class"
        write_pair(vspw, "train", "0001", 13 if invalid else 1, "video_a")
        write_pair(vspw, "val", "0001", 12, "video_a" if leak else "video_b")
        write_pair(static, "train", "static_train", 6)
        write_pair(static, "val", "static_val", 12)
        return vspw, static

    def run_audit(self, parent, vspw, static):
        report = parent / "report.json"
        process = subprocess.run(
            [
                sys.executable, str(TOOLS / "audit_vspw_mixed.py"),
                "--vspw-root", str(vspw), "--static-root", str(static),
                "--output-json", str(report),
            ],
            text=True, capture_output=True,
        )
        return process, report

    def test_audit_reports_complementary_class_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            process, report = self.run_audit(parent, *self.create_datasets(parent))
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result["train_val_video_overlap_count"], 0)
            self.assertGreater(result["vspw"]["train"]["pixel_counts"]["sky"], 0)
            self.assertGreater(result["static"]["train"]["pixel_counts"]["food"], 0)

    def test_audit_rejects_video_split_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            process, _ = self.run_audit(parent, *self.create_datasets(parent, leak=True))
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("video leakage", process.stderr)

    def test_audit_rejects_unknown_mask_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            process, _ = self.run_audit(parent, *self.create_datasets(parent, invalid=True))
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Invalid IDs", process.stderr)


if __name__ == "__main__":
    unittest.main()
