"""Streamlit interface for the Dataset Image ↔ Label Validator."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_validator import (
    IMAGE_EXTENSIONS,
    configure_logging,
    create_output_dataset,
    create_sample,
    create_zip,
    scan_dataset,
    search_result,
    validate_labels,
    write_reports,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
REPORTS = OUTPUT / "reports"

st.set_page_config(page_title="Dataset Image ↔ Label Validator", page_icon="🔎", layout="wide")
configure_logging(OUTPUT)


def progress_callback(progress_bar, status_box):
    def update(phase: str, current: int, total: int) -> None:
        status_box.write(phase)
        progress_bar.progress(min(current / max(total, 1), 1.0))
    return update


def run_action(action, success_message: str):
    progress_bar = st.progress(0)
    status_box = st.empty()
    try:
        value = action(progress_callback(progress_bar, status_box))
    except Exception as exc:
        logging.getLogger("dataset_validator").exception("Application action failed")
        st.error(str(exc))
        return None
    progress_bar.progress(1.0)
    status_box.success(success_message)
    return value


def save_uploads(images, labels) -> tuple[Path, Path]:
    root = Path(tempfile.mkdtemp(prefix="dataset_validator_"))
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    for uploaded in images:
        (image_dir / Path(uploaded.name).name).write_bytes(uploaded.getbuffer())
    for uploaded in labels:
        (label_dir / Path(uploaded.name).name).write_bytes(uploaded.getbuffer())
    return image_dir, label_dir


def show_table(rows: list[dict], key: str) -> None:
    if not rows:
        st.info("No records in this category.")
        return
    left, right = st.columns(2)
    limit = left.selectbox("Rows per page", [100, 500, 1000], key=f"{key}_limit")
    pages = max(1, (len(rows) + limit - 1) // limit)
    page = right.number_input("Page", 1, pages, 1, key=f"{key}_page")
    start = (page - 1) * limit
    st.caption(f"Showing {start + 1:,}–{min(start + limit, len(rows)):,} of {len(rows):,}")
    st.dataframe(pd.DataFrame(rows[start:start + limit]), width="stretch", hide_index=True)


def report_download(filename: str) -> None:
    path = REPORTS / filename
    if path.is_file():
        st.download_button(f"Download {filename}", path.read_bytes(), filename, "text/csv", key=f"download_{filename}")


st.title("Dataset Image ↔ Label Validator")
st.caption("Validate, separate and export computer-vision image/label pairs.")
st.info("For very large datasets, Local Folder mode is recommended.")

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Mode", ["Local folders", "Streamlit uploads"])
    case_insensitive = st.checkbox("Case-insensitive matching", value=False, help="Default is exact, case-sensitive matching. Files are never renamed.")
    image_dir = label_dir = None
    uploaded_images = uploaded_labels = []
    if mode == "Local folders":
        image_dir = Path(st.text_input("Images folder", str(ROOT / "train" / "images"))).expanduser()
        label_dir = Path(st.text_input("Labels folder", str(ROOT / "train" / "labels"))).expanduser()
        scan_label = "Scan Local Dataset"
    else:
        uploaded_images = st.file_uploader("Upload Images", type=sorted(extension.lstrip(".") for extension in IMAGE_EXTENSIONS), accept_multiple_files=True)
        uploaded_labels = st.file_uploader("Upload Labels", type=["txt"], accept_multiple_files=True)
        scan_label = "Scan Dataset"

    if st.button(scan_label, type="primary", width="stretch"):
        if mode == "Streamlit uploads":
            if not uploaded_images and not uploaded_labels:
                st.error("Select image or label files first.")
                st.stop()
            image_dir, label_dir = save_uploads(uploaded_images, uploaded_labels)
        result = run_action(
            lambda progress: scan_dataset(image_dir, label_dir, case_insensitive, progress),
            "Dataset scan finished.",
        )
        if result is not None:
            st.session_state.scan_result = result
            write_reports(result, REPORTS)

    result = st.session_state.get("scan_result")
    st.divider()
    st.subheader("Output control")
    if st.button("Validate Labels", disabled=result is None, width="stretch"):
        completed = run_action(lambda progress: validate_labels(result, progress), "Label validation finished.")
        if completed is not None or result.labels_validated:
            write_reports(result, REPORTS)
    if st.button("Create Output Dataset", disabled=result is None, width="stretch"):
        run_action(lambda progress: create_output_dataset(result, OUTPUT, True, progress), "Output dataset created.")
    confirm_clear = st.checkbox("I confirm clearing generated output", value=False)
    if st.button("Clear Previous Output", disabled=not confirm_clear, width="stretch"):
        resolved_output = OUTPUT.resolve()
        if resolved_output == ROOT.resolve() or resolved_output in {(ROOT / "train" / "images").resolve(), (ROOT / "train" / "labels").resolve()}:
            st.error("Refusing to clear an unsafe path.")
        else:
            logger = logging.getLogger("dataset_validator")
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
            shutil.rmtree(resolved_output, ignore_errors=False)
            configure_logging(OUTPUT)
            st.success(f"Cleared generated files in {resolved_output}")

result = st.session_state.get("scan_result")
if result is None:
    st.write("Choose a data source and scan the dataset to begin.")
    st.stop()

summary = result.summary()
metrics = [
    ("Total Images", summary["total_images"]),
    ("Total Labels", summary["total_labels"]),
    ("Valid Matched Pairs", summary["valid_matched_pairs"]),
    ("Images Without Labels", summary["images_without_labels"]),
    ("Labels Without Images", summary["labels_without_images"]),
    ("Duplicate/Ambiguous Stems", summary["duplicate_ambiguous_stems"]),
    ("Case Mismatches", summary["case_mismatches"]),
    ("Empty Labels", summary["empty_labels"]),
    ("Invalid YOLO Labels", summary["invalid_yolo_labels"]),
]
for start in range(0, len(metrics), 5):
    for column, (label, value) in zip(st.columns(min(5, len(metrics) - start)), metrics[start:start + 5]):
        column.metric(label, f"{value:,}")
if not result.labels_validated:
    st.caption(f"{len(result.correct_pairs):,} filename pairs found. Validate labels to populate valid, empty, and invalid label metrics.")
if not result.images and not result.labels:
    st.warning("No supported image or TXT files were found directly in the selected folders. ZIP archives are not unpacked automatically.")

tabs = st.tabs(["Overview", "Correct Pairs", "Incorrect Images", "Incorrect Labels", "Invalid Labels", "Duplicates", "Search", "Sampling", "Export"])

with tabs[0]:
    st.subheader("Dataset summary")
    summary_rows = [
        {
            "metric": key,
            "value": "Yes" if value is True else "No" if value is False else f"{value:,}",
        }
        for key, value in summary.items()
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
    if result.case_mismatches:
        st.subheader("Possible case mismatches")
        show_table(result.case_mismatches, "case_mismatches")
    if result.scan_errors:
        st.subheader("Scan errors")
        show_table(result.scan_errors, "scan_errors")
    report_download("dataset_summary.csv")

with tabs[1]:
    show_table(result.correct_pairs, "correct")
    report_download("correct_pairs.csv")

with tabs[2]:
    show_table(result.incorrect_images, "incorrect_images")
    report_download("incorrect_images.csv")

with tabs[3]:
    show_table(result.incorrect_labels, "incorrect_labels")
    report_download("incorrect_labels.csv")

with tabs[4]:
    show_table(result.invalid_labels, "invalid_labels")
    report_download("invalid_labels.csv")

with tabs[5]:
    show_table(result.duplicates, "duplicates")
    report_download("duplicate_names.csv")
    report_download("case_mismatches.csv")

with tabs[6]:
    query = st.text_input("Search by exact filename stem", placeholder="drone_00451")
    if query:
        found = search_result(result, query)
        st.dataframe(pd.DataFrame([found]), width="stretch", hide_index=True)

with tabs[7]:
    st.write(f"Available valid pairs: **{len(result.valid_pairs):,}**")
    sample_count = st.number_input("Number of pairs", min_value=1, value=10000, step=1)
    seed = st.number_input("Random seed", value=42, step=1)
    st.write(f"Requested pairs: **{sample_count:,}**")
    if st.button(f"Create {sample_count:,} Sample", disabled=not result.labels_validated):
        run_action(lambda progress: create_sample(result, OUTPUT, int(sample_count), int(seed), progress), f"Created exactly {sample_count:,} image/label pairs.")

with tabs[8]:
    st.write("ZIP files are created only when requested.")
    archive_jobs = {
        "Prepare Correct Dataset ZIP": (OUTPUT / "correct_dataset", OUTPUT / "downloads" / "correct_dataset.zip"),
        "Prepare Incorrect Dataset ZIP": (OUTPUT, OUTPUT / "downloads" / "incorrect_dataset.zip"),
        "Prepare Reports ZIP": (REPORTS, OUTPUT / "downloads" / "reports.zip"),
    }
    for label, (source, destination) in archive_jobs.items():
        if st.button(label):
            if label == "Prepare Incorrect Dataset ZIP":
                staging = Path(tempfile.mkdtemp(prefix="incorrect_dataset_"))
                for name in ("incorrect_images", "incorrect_labels"):
                    if (OUTPUT / name).is_dir():
                        shutil.copytree(OUTPUT / name, staging / name)
                source = staging
            created = run_action(lambda _progress, source=source, destination=destination: create_zip(source, destination), f"Prepared {destination.name}.")
            if created:
                st.session_state[f"archive_{destination.name}"] = str(created)
    for path_text in [value for key, value in st.session_state.items() if key.startswith("archive_")]:
        path = Path(path_text)
        if path.is_file():
            st.download_button(f"Download {path.name}", path.open("rb"), path.name, "application/zip", key=f"zip_{path.name}")
