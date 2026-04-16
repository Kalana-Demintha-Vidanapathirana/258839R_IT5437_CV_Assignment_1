# Assignment 1

MSc in AI - Year 1Q3 - CV Assignment 1

## Getting Started

This repository contains the code and documentation for Assignment 1.

## Project Structure

```
├── README.md
├── .gitignore
└── Assignment_1.code-workspace
```

## Installation

# Create virtual environment
uv venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies (once you create requirements.txt or pyproject.toml)
uv pip install -r requirements.txt
# OR
uv pip install opencv-python numpy matplotlib scikit-image jupyter pandas

# Sync dependencies from pyproject.toml
uv sync

## Usage

### Navigate to /src for python files
cd src

### Navigate to /notebooks for jupyter notebook files
cd notebooks
