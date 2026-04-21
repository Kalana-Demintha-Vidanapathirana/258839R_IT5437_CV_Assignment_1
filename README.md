# IT5437 — Image Processing and Machine Vision
### Assignment 1: Intensity Transformations and Neighborhood Filtering

**Student:** K.D. Vidanapathirana | **Index:** 258839R

---

## Overview

This repository contains the implementation for Assignment 1 of IT5437 (Computer Vision), covering fundamental image processing techniques:

| Q | Topic |
|---|---|
| Q1 | Gamma correction and contrast stretching |
| Q2 | Gamma correction in L\*a\*b\* colour space |
| Q3 | Manual histogram equalization |
| Q4 | Otsu thresholding and foreground-only equalization |
| Q5 | Gaussian kernel construction and smoothing |
| Q6 | Derivative-of-Gaussian edge detection vs. Sobel |
| Q7 | Nearest-neighbour and bilinear image interpolation |
| Q8 | Salt-and-pepper noise removal |
| Q9 | Image sharpening via unsharp masking |
| Q10 | Bilateral filtering (manual and OpenCV) |
| Q11 | Spatial filtering and frequency response (theory) |
| Q12 | Homomorphic filtering for illumination correction |

The submission report is `258839R_a01.pdf`. The editable source is `258839R_a01.docx`.

---

## Project Structure

```
├── data/
│   └── raw/                  # Input images
│       └── a1q8images/       # Interpolation test image pairs (Q7)
├── notebooks/                # Jupyter notebooks (q01.ipynb – q12.ipynb)
├── src/                      # Python scripts (q01.py – q10.py, q12.py)
├── results/                  # Generated output images and CSV
├── 258839R_a01.pdf           # Submission report
├── 258839R_a01.docx          # Report source
├── pyproject.toml            # Project dependencies
└── uv.lock                   # Locked dependency versions
```

---

## Setup

### Requirements
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd 258839R_IT5437_CV_Assignment_1

# 2. Create virtual environment and install all dependencies
uv sync

# 3. Activate the environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

That's it — no manual pip installs or conda setup required. `uv sync` reads `pyproject.toml` and `uv.lock` to reproduce the exact environment.

---

## Running the Code

### Python scripts
```bash
cd src
python q01.py   # run any question script individually
```

### Jupyter notebooks
```bash
jupyter notebook
# then open notebooks/q01.ipynb through q12.ipynb
```

Results (images and CSV) are saved to the `results/` directory, organized by question (`results/q01/`, `results/q02/`, etc.).

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Core image processing |
| `numpy` | Array operations and kernel computation |
| `matplotlib` | Visualization and result plots |
| `scipy` / `scikit-image` | Supporting image utilities |
| `pandas` | SSD results table (Q7) |
| `jupyter` | Interactive notebooks |
| `pillow` | Auxiliary image I/O |
