"""Training preflight tests: generated tiny images, no YOLO/GPU downloads."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from training.dataset import check_dataset, read_dataset
from training.train import main


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "data.yaml"
        self.config.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: wire\n", encoding="utf-8")
        for index, split in enumerate(("train", "val")):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(index * 80, 0, 0)).save(self.root / "images" / split / "sample.png")
            (self.root / "labels" / split / "sample.txt").write_text("0 .5 .5 .2 .2\n")

    def check(self, **kwargs):
        return check_dataset(read_dataset(self.config), **kwargs)

    def test_relative_root_and_valid_dataset(self):
        data = read_dataset(self.config)
        self.assertEqual(Path(data["path"]), self.root.resolve())
        report, errors = check_dataset(data)
        self.assertEqual(errors, [])
        self.assertEqual(report["train"]["objects"]["wire"], 1)

    def test_empty_dataset_and_missing_label(self):
        (self.root / "labels/train/sample.txt").unlink()
        self.assertTrue(any("Anotacao ausente" in error for error in self.check()[1]))
        (self.root / "images/train/sample.png").unlink()
        self.assertTrue(any("nenhuma imagem" in error for error in self.check()[1]))

    def test_invalid_class_nonfinite_and_bad_dimensions(self):
        label = self.root / "labels/train/sample.txt"
        for text in ("5 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .5 .5 -.1 .2", "0 .5 .5 .2", "0 .5 .5 1.5 .2"):
            with self.subTest(text=text):
                label.write_text(text)
                self.assertTrue(any("sample.txt:1" in error for error in self.check()[1]))

    def test_identical_images_across_splits_are_rejected(self):
        shutil.copyfile(self.root / "images/train/sample.png", self.root / "images/val/sample.png")
        self.assertTrue(any("duplicada" in error for error in self.check()[1]))

    def test_test_split_required_only_for_evaluation(self):
        self.assertFalse(self.check()[1])
        self.assertTrue(any("test: nenhuma imagem" in error for error in self.check(require_test=True)[1]))

    def test_empty_labels_are_valid_for_extra_negative_images(self):
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(self.root / "images/train/negative.png")
        (self.root / "labels/train/negative.txt").touch()
        self.assertFalse(self.check()[1])

    def test_orphan_annotations_are_reported(self):
        (self.root / "labels/train/orphan.txt").write_text("0 .5 .5 .2 .2")
        self.assertTrue(any("sem imagem" in error for error in self.check()[1]))

    def test_check_does_not_select_device_or_start_training(self):
        with patch("training.train.select_device") as device:
            self.assertEqual(main(["--data", str(self.config), "--check"]), 0)
            device.assert_not_called()
