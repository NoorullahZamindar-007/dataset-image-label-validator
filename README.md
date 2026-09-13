# Dataset Image ↔ Label Validator

A local Streamlit utility for matching computer-vision images with TXT labels, validating basic YOLO rows, reviewing problems, and copying reviewed datasets without changing source files.

## Setup and run

```powershell
cd "D:\Projects\1st Try"
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Local mode reads files directly inside `train/images` and `train/labels`. Supported images are JPG, JPEG, PNG, BMP, WEBP, TIF, and TIFF; labels must be TXT. ZIP archives are not unpacked automatically. For very large datasets, use Local Folder mode.

Exact, case-sensitive stem matching is the default. Duplicate stems are excluded from clean pairs. YOLO validation is a separate action, and samples are created only from pairs whose labels validate successfully.

Generated files are placed under `output/`. Source files are only read; copies use `shutil.copy2()`.

## Tests

```powershell
py -m unittest discover -s tests -v
py -m compileall -q app.py dataset_validator.py tests
```
