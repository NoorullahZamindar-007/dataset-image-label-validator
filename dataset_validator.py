"""Core dataset matching, YOLO validation, reporting, and safe copy operations."""

from __future__ import annotations

import csv
import logging
import math
import random
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTENSION = ".txt"
Progress = Callable[[str, int, int], None]
LOGGER = logging.getLogger("dataset_validator")


@dataclass
class ScanResult:
    image_dir: str
    label_dir: str
    case_insensitive: bool
    images: list[Path] = field(default_factory=list)
    labels: list[Path] = field(default_factory=list)
    correct_pairs: list[dict] = field(default_factory=list)
    incorrect_images: list[dict] = field(default_factory=list)
    incorrect_labels: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    case_mismatches: list[dict] = field(default_factory=list)
    invalid_labels: list[dict] = field(default_factory=list)
    scan_errors: list[dict] = field(default_factory=list)
    labels_validated: bool = False

    @property
    def valid_pairs(self) -> list[dict]:
        return [row for row in self.correct_pairs if row["label_status"] == "VALID"]

    @property
    def empty_label_count(self) -> int:
        return sum(row["label_status"] == "EMPTY" for row in self.correct_pairs)

    @property
    def invalid_label_count(self) -> int:
        return sum(row["label_status"] == "INVALID" for row in self.correct_pairs)

    def summary(self) -> dict[str, int | bool]:
        return {
            "total_images": len(self.images),
            "total_labels": len(self.labels),
            "name_matched_pairs": len(self.correct_pairs),
            "valid_matched_pairs": len(self.valid_pairs),
            "images_without_labels": len(self.incorrect_images),
            "labels_without_images": len(self.incorrect_labels),
            "duplicate_ambiguous_stems": len({row["stem"] for row in self.duplicates}),
            "case_mismatches": len(self.case_mismatches),
            "empty_labels": self.empty_label_count,
            "invalid_yolo_labels": self.invalid_label_count,
            "scan_errors": len(self.scan_errors),
            "labels_validated": self.labels_validated,
        }


def configure_logging(output_dir: Path) -> Path:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / "app.log"
    if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path.resolve()
               for handler in LOGGER.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return log_path


def _notify(progress: Progress | None, phase: str, current: int = 0, total: int = 1) -> None:
    if progress:
        progress(phase, current, total)


def _scan_folder(folder: Path, extensions: set[str], kind: str, errors: list[dict],
                 progress: Progress | None) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"{kind.title()} folder does not exist: {folder}")
    records: list[Path] = []
    try:
        entries = list(folder.iterdir())
    except OSError as exc:
        raise OSError(f"Cannot read {kind} folder {folder}: {exc}") from exc
    total = len(entries)
    for index, path in enumerate(entries, 1):
        try:
            if path.is_file() and path.suffix.lower() in extensions:
                records.append(path.resolve())
        except OSError as exc:
            errors.append({"operation": "scan", "source": str(path), "destination": "", "error": str(exc)})
        if index == total or index % 1000 == 0:
            _notify(progress, f"Scanning {kind}...", index, max(total, 1))
    records.sort(key=lambda item: (item.stem, item.name))
    return records


def _index(records: Iterable[Path], case_insensitive: bool) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for record in records:
        index[record.stem.casefold() if case_insensitive else record.stem].append(record)
    return dict(index)


def _find_case_mismatches(images: list[Path], labels: list[Path]) -> list[dict]:
    image_index = _index(images, True)
    label_index = _index(labels, True)
    rows = []
    for key in sorted(image_index.keys() & label_index.keys()):
        image_stems = {item.stem for item in image_index[key]}
        label_stems = {item.stem for item in label_index[key]}
        if image_stems != label_stems:
            rows.append({
                "case_key": key,
                "image_stems": "; ".join(sorted(image_stems)),
                "label_stems": "; ".join(sorted(label_stems)),
                "image_files": "; ".join(map(str, image_index[key])),
                "label_files": "; ".join(map(str, label_index[key])),
                "reason": "POSSIBLE CASE MISMATCH",
            })
    return rows


def scan_dataset(image_dir: Path | str, label_dir: Path | str, case_insensitive: bool = False,
                 progress: Progress | None = None) -> ScanResult:
    """Index and match supported files in O(n) expected time without decoding images."""
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    result = ScanResult(str(image_dir.resolve()), str(label_dir.resolve()), case_insensitive)
    LOGGER.info("Scan started: images=%s labels=%s case_insensitive=%s", image_dir, label_dir, case_insensitive)
    result.images = _scan_folder(image_dir, IMAGE_EXTENSIONS, "images", result.scan_errors, progress)
    result.labels = _scan_folder(label_dir, {LABEL_EXTENSION}, "labels", result.scan_errors, progress)
    _notify(progress, "Building filename index...", 0, 1)
    images_by_stem = _index(result.images, case_insensitive)
    labels_by_stem = _index(result.labels, case_insensitive)
    all_stems = sorted(images_by_stem.keys() | labels_by_stem.keys())

    for position, stem in enumerate(all_stems, 1):
        images = images_by_stem.get(stem, [])
        labels = labels_by_stem.get(stem, [])
        ambiguous = len(images) > 1 or len(labels) > 1
        if ambiguous:
            for file_type, records in (("image", images), ("label", labels)):
                for record in records:
                    result.duplicates.append({
                        "stem": stem,
                        "file_type": file_type,
                            "filename": record.name,
                            "full_path": str(record),
                        "reason": f"AMBIGUOUS / DUPLICATE: {len(images)} image(s), {len(labels)} label(s) share this matching stem",
                    })
        elif images and labels:
            image, label = images[0], labels[0]
            result.correct_pairs.append({
                "image_stem": image.stem,
                "image_filename": image.name,
                "label_filename": label.name,
                "filename_status": "MATCHED",
                "label_status": "NOT CHECKED",
                "label_errors": "",
                "image_path": str(image),
                "label_path": str(label),
            })
        elif images:
            image = images[0]
            result.incorrect_images.append({
                "image_stem": image.stem,
                "image_filename": image.name,
                "expected_label": f"{image.stem}.txt",
                "image_path": str(image),
                "reason": "NO CORRESPONDING TXT LABEL",
            })
        elif labels:
            label = labels[0]
            result.incorrect_labels.append({
                "label_stem": label.stem,
                "label_filename": label.name,
                "expected_image_stem": label.stem,
                "label_path": str(label),
                "reason": "NO CORRESPONDING IMAGE",
            })
        if position == len(all_stems) or position % 5000 == 0:
            _notify(progress, "Matching pairs and checking duplicates...", position, max(len(all_stems), 1))

    _notify(progress, "Checking case mismatches...", 0, 1)
    result.case_mismatches = _find_case_mismatches(result.images, result.labels)
    LOGGER.info(
        "Scan completed: images=%d labels=%d matches=%d image_only=%d label_only=%d duplicates=%d case_mismatches=%d errors=%d",
        len(result.images), len(result.labels), len(result.correct_pairs), len(result.incorrect_images),
        len(result.incorrect_labels), len({row['stem'] for row in result.duplicates}),
        len(result.case_mismatches), len(result.scan_errors),
    )
    _notify(progress, "Scan finished.", 1, 1)
    return result


def validate_yolo_label(path: Path | str) -> tuple[str, list[str]]:
    """Return VALID, EMPTY, or INVALID and human-readable validation errors."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return "INVALID", [f"Unreadable label: {exc}"]
    if not text.strip():
        return "EMPTY", []

    errors = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"Line {line_number}: expected 5 fields, found {len(fields)}")
            continue
        try:
            class_id, x_center, y_center, width, height = map(float, fields)
        except ValueError:
            errors.append(f"Line {line_number}: fields must be numeric")
            continue
        if not all(math.isfinite(value) for value in (class_id, x_center, y_center, width, height)):
            errors.append(f"Line {line_number}: values must be finite numbers")
        if not class_id.is_integer() or class_id < 0:
            errors.append(f"Line {line_number}: class ID must be a non-negative integer")
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            errors.append(f"Line {line_number}: coordinates must be between 0 and 1")
        if width <= 0 or height <= 0:
            errors.append(f"Line {line_number}: width and height must be greater than 0")
    return ("INVALID", errors) if errors else ("VALID", [])


def validate_labels(result: ScanResult, progress: Progress | None = None) -> None:
    result.invalid_labels.clear()
    total = len(result.correct_pairs)
    for index, pair in enumerate(result.correct_pairs, 1):
        status, errors = validate_yolo_label(pair["label_path"])
        pair["label_status"] = status
        pair["label_errors"] = "; ".join(errors)
        if status != "VALID":
            result.invalid_labels.append({
                "image_stem": pair["image_stem"],
                "label_filename": pair["label_filename"],
                "label_path": pair["label_path"],
                "label_status": status,
                "reason": pair["label_errors"] or "Label file is empty",
            })
            if status == "INVALID":
                LOGGER.warning("Label validation failed: %s: %s", pair["label_path"], pair["label_errors"])
        if index == total or index % 1000 == 0:
            _notify(progress, "Validating YOLO labels...", index, max(total, 1))
    result.labels_validated = True
    LOGGER.info("Label validation completed: valid=%d empty=%d invalid=%d", len(result.valid_pairs),
                result.empty_label_count, result.invalid_label_count)


REPORT_FIELDS = {
    "correct_pairs.csv": ["image_stem", "image_filename", "label_filename", "filename_status", "label_status", "label_errors", "image_path", "label_path"],
    "incorrect_images.csv": ["image_stem", "image_filename", "expected_label", "image_path", "reason"],
    "incorrect_labels.csv": ["label_stem", "label_filename", "expected_image_stem", "label_path", "reason"],
    "invalid_labels.csv": ["image_stem", "label_filename", "label_path", "label_status", "reason"],
    "duplicate_names.csv": ["stem", "file_type", "filename", "full_path", "reason"],
    "case_mismatches.csv": ["case_key", "image_stems", "label_stems", "image_files", "label_files", "reason"],
    "copy_errors.csv": ["operation", "source", "destination", "error"],
}


def _write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(result: ScanResult, reports_dir: Path | str, copy_errors: list[dict] | None = None) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "correct_pairs.csv": result.correct_pairs,
        "incorrect_images.csv": result.incorrect_images,
        "incorrect_labels.csv": result.incorrect_labels,
        "invalid_labels.csv": result.invalid_labels,
        "duplicate_names.csv": result.duplicates,
        "case_mismatches.csv": result.case_mismatches,
        "copy_errors.csv": copy_errors if copy_errors is not None else result.scan_errors,
    }
    for filename, rows in sources.items():
        _write_csv(reports_dir / filename, rows, REPORT_FIELDS[filename])
    summary = result.summary()
    _write_csv(reports_dir / "dataset_summary.csv", ({"metric": key, "value": value} for key, value in summary.items()), ["metric", "value"])
    LOGGER.info("Reports generated in %s", reports_dir)


def _safe_copy(source: str, destination_dir: Path, operation: str, errors: list[dict]) -> bool:
    destination = destination_dir / Path(source).name
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True
    except OSError as exc:
        error = {"operation": operation, "source": source, "destination": str(destination), "error": str(exc)}
        errors.append(error)
        LOGGER.error("Copy failed: %s", error)
        return False


def create_output_dataset(result: ScanResult, output_dir: Path | str, include_clean_dataset: bool = True,
                          progress: Progress | None = None) -> list[dict]:
    """Copy categorized files. Source paths are never renamed, moved, deleted, or edited."""
    output_dir = Path(output_dir)
    for relative in ("correct_images", "correct_labels", "incorrect_images", "incorrect_labels",
                     "reports", "correct_dataset/images", "correct_dataset/labels"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    errors: list[dict] = list(result.scan_errors)
    work = len(result.correct_pairs) * 2 + len(result.incorrect_images) + len(result.incorrect_labels)
    done = 0
    for pair in result.correct_pairs:
        _safe_copy(pair["image_path"], output_dir / "correct_images", "copy correct image", errors)
        _safe_copy(pair["label_path"], output_dir / "correct_labels", "copy correct label", errors)
        if include_clean_dataset and pair["label_status"] == "VALID":
            _safe_copy(pair["image_path"], output_dir / "correct_dataset" / "images", "copy clean image", errors)
            _safe_copy(pair["label_path"], output_dir / "correct_dataset" / "labels", "copy clean label", errors)
        done += 2
        if done % 1000 == 0:
            _notify(progress, "Copying files...", done, max(work, 1))
    for row in result.incorrect_images:
        _safe_copy(row["image_path"], output_dir / "incorrect_images", "copy incorrect image", errors)
        done += 1
    for row in result.incorrect_labels:
        _safe_copy(row["label_path"], output_dir / "incorrect_labels", "copy incorrect label", errors)
        done += 1
    write_reports(result, output_dir / "reports", errors)
    LOGGER.info("Output dataset completed: files=%d errors=%d", done, len(errors))
    _notify(progress, "Finished.", max(work, 1), max(work, 1))
    return errors


def create_sample(result: ScanResult, output_dir: Path | str, pair_count: int, seed: int,
                  progress: Progress | None = None) -> Path:
    if pair_count <= 0:
        raise ValueError("Number of pairs must be greater than zero")
    available = result.valid_pairs
    if not result.labels_validated:
        raise ValueError("Validate labels before creating a sample")
    if len(available) < pair_count:
        raise ValueError(f"Available valid pairs: {len(available)}; requested pairs: {pair_count}")
    output_dir = Path(output_dir)
    sample_dir = output_dir / f"sample_{pair_count}"
    if sample_dir.exists():
        raise FileExistsError(f"{sample_dir} already exists; clear previous output before replacing it")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".sample_{pair_count}_", dir=output_dir))
    selected = random.Random(seed).sample(available, pair_count)
    errors: list[dict] = []
    rows = []
    for index, pair in enumerate(selected, 1):
        image_ok = _safe_copy(pair["image_path"], staging / "images", "sample image", errors)
        label_ok = _safe_copy(pair["label_path"], staging / "labels", "sample label", errors)
        if image_ok and label_ok:
            rows.append({"image_stem": pair["image_stem"], "image_filename": pair["image_filename"], "label_filename": pair["label_filename"], "seed": seed})
        if index == pair_count or index % 500 == 0:
            _notify(progress, "Creating sample...", index, pair_count)
    if errors or len(rows) != pair_count:
        write_reports(result, Path(output_dir) / "reports", errors)
        shutil.rmtree(staging, ignore_errors=True)
        raise OSError(f"Sample incomplete: copied {len(rows)} of {pair_count} pairs; see copy_errors.csv")
    _write_csv(staging / f"sample_{pair_count}.csv", rows, ["image_stem", "image_filename", "label_filename", "seed"])
    staging.rename(sample_dir)
    LOGGER.info("Sample created: pairs=%d seed=%d path=%s", pair_count, seed, sample_dir)
    return sample_dir


def create_zip(source_dir: Path | str, destination: Path | str) -> Path:
    source_dir, destination = Path(source_dir), Path(destination)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Nothing to archive: {source_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.resolve() != destination.resolve():
                archive.write(path, path.relative_to(source_dir))
    LOGGER.info("ZIP created: source=%s destination=%s", source_dir, destination)
    return destination


def search_result(result: ScanResult, query: str) -> dict:
    query = query.strip()
    normalize = str.casefold if result.case_insensitive else lambda value: value
    target = normalize(query)
    images = [record for record in result.images if normalize(record.stem) == target]
    labels = [record for record in result.labels if normalize(record.stem) == target]
    pairs = [pair for pair in result.correct_pairs if normalize(pair["image_stem"]) == target]
    duplicate = any(normalize(row["stem"]) == target for row in result.duplicates)
    return {
        "Image found": "YES" if images else "NO",
        "Label found": "YES" if labels else "NO",
        "Filename match": "YES" if pairs else "NO",
        "Duplicate": "YES" if duplicate else "NO",
        "Label validation": pairs[0]["label_status"] if pairs else "NOT CHECKED",
        "Image paths": "; ".join(map(str, images)),
        "Label paths": "; ".join(map(str, labels)),
    }
