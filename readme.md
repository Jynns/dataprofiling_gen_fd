# Project Setup and Training Data Documentation

This repository contains training data and supporting assets for model development.
Due to file size constraints, some files (including ZIP archives) are managed using Git Large File Storage (LFS).

## Dataset: training_data.csv

The training_data.csv file contains labeled rules used for training and evaluation.

### Column Definitions

**LHS**
All left-hand-side attributes, stored as a single string.
- Attributes are enclosed in quotation marks
- Attributes are separated by commas (,)
- The complete list is wrapped in brackets

Example:
["attr1","attr2","attr3"]

**RHS**
The right-hand-side attribute associated with the LHS attributes.

**genuine**
Binary label indicating rule validity:
- 0 — Not genuine
- 1 — Genuine

## Python Environment Setup

Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib sentence-transformers python-Levenshtein

### Dependency Overview
- pandas / numpy — data manipulation and numerical computation
- scikit-learn — modeling and evaluation
- matplotlib — visualization
- sentence-transformers — semantic text embeddings
- python-Levenshtein — string similarity metrics

## Ollama Installation

Ollama is required for local model inference.

Install using the official installer:

curl -fsSL https://ollama.com/install.sh | sh

Verify the installation:

ollama --version

## Git LFS (Mandatory)

Large files (including ZIP archives) are tracked using Git Large File Storage (LFS).
If Git LFS is not installed or initialized, these files will appear as small text pointer files and cannot be unzipped.

### Install Git LFS

git lfs version || sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install

### Download LFS Files

After cloning the repository, fetch the actual large files:

git lfs pull

This step is required before using any ZIP or large data files.

## Common Issues and Fixes

ZIP extraction fails with:
End-of-central-directory signature not found

Cause: The ZIP file is still an LFS pointer file.
Fix: Run git lfs pull

ZIP file size is only a few kilobytes:
This indicates that the real file has not been downloaded from Git LFS.

## Notes

- ZIP files are fully compatible across macOS and Linux
- Always run git lfs pull after switching branches
- Do not transfer ZIP files via text-based channels (email, chat copy/paste)
