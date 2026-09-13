import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_validator import (
    create_output_dataset,
    create_sample,
    scan_dataset,
    validate_labels,
    validate_yolo_label,
    write_reports,
)


class DatasetValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name):
        (self.images / name).write_bytes(b"unchanged-image")

    def label(self, name, content="0 0.5 0.5 0.2 0.2\n"):
        (self.labels / name).write_text(content, encoding="utf-8")

    def test_synthetic_matching_expected_metrics(self):
        for name in ("A.jpg", "B.jpg", "C.png", "D.jpg", "F.jpg"):
            self.image(name)
        for name in ("A.txt", "B.txt", "D.txt", "E.txt", "F.txt"):
            self.label(name)
        result = scan_dataset(self.images, self.labels)
        self.assertEqual(result.summary()["total_images"], 5)
        self.assertEqual(result.summary()["total_labels"], 5)
        self.assertEqual(result.summary()["name_matched_pairs"], 4)
        self.assertEqual([row["image_stem"] for row in result.incorrect_images], ["C"])
        self.assertEqual([row["label_stem"] for row in result.incorrect_labels], ["E"])

    def test_duplicate_case_mismatch_and_label_statuses(self):
        self.image("A.jpg")
        self.image("A.png")
        self.image("B.jpg")
        self.image("C.jpg")
        self.image("Drone001.jpg")
        self.label("A.txt")
        self.label("B.txt", "")
        self.label("C.txt", "0 4.5 abc 0.2\n")
        self.label("drone001.txt")
        result = scan_dataset(self.images, self.labels)
        validate_labels(result)
        self.assertEqual(len(result.duplicates), 3)
        self.assertEqual(len(result.case_mismatches), 1)
        self.assertEqual({row["label_status"] for row in result.invalid_labels}, {"EMPTY", "INVALID"})
        insensitive = scan_dataset(self.images, self.labels, case_insensitive=True)
        self.assertIn("Drone001", [row["image_stem"] for row in insensitive.correct_pairs])

    def test_yolo_validation(self):
        valid = self.labels / "valid.txt"
        empty = self.labels / "empty.txt"
        invalid = self.labels / "invalid.txt"
        valid.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        empty.write_text("", encoding="utf-8")
        invalid.write_text("0 4.5 abc 0.2\n", encoding="utf-8")
        self.assertEqual(validate_yolo_label(valid)[0], "VALID")
        self.assertEqual(validate_yolo_label(empty)[0], "EMPTY")
        self.assertEqual(validate_yolo_label(invalid)[0], "INVALID")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            self.assertEqual(validate_yolo_label(valid)[0], "INVALID")

    def test_copy_reports_and_deterministic_sampling_preserve_sources(self):
        for name in ("A.jpg", "B.png", "C.webp"):
            self.image(name)
            self.label(Path(name).stem + ".txt")
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (*self.images.iterdir(), *self.labels.iterdir())}
        result = scan_dataset(self.images, self.labels)
        validate_labels(result)
        output = self.root / "output"
        self.assertEqual(create_output_dataset(result, output), [])
        sample = create_sample(result, output, 2, 42)
        self.assertEqual(len(list((sample / "images").iterdir())), 2)
        self.assertEqual(len(list((sample / "labels").iterdir())), 2)
        self.assertTrue((sample / "sample_2.csv").is_file())
        write_reports(result, output / "reports")
        self.assertTrue((output / "reports" / "dataset_summary.csv").is_file())
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
        self.assertEqual(before, after)

    def test_sample_refuses_shortfall(self):
        self.image("A.jpg")
        self.label("A.txt")
        result = scan_dataset(self.images, self.labels)
        validate_labels(result)
        with self.assertRaisesRegex(ValueError, "Available valid pairs: 1; requested pairs: 2"):
            create_sample(result, self.root / "output", 2, 42)


if __name__ == "__main__":
    unittest.main()
