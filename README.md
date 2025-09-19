# Image Repair Toolkit

A reference project for restoring vintage photographs through automatic defect repair and multi-scale super-resolution. It bundles data loading, defect simulation, color enhancement, denoising, and quality assessment utilities.

## Key Features
- **Multi-scale super-resolution**: `MultiScaleSRNet` produces x2/x3/x4 outputs to match different resolution targets.
- **Defect detection and repair**: `DefectRepairModule` combines segmentation and inpainting networks to address scratches, stains, and creases.
- **Color restoration and denoising**: Lightweight gray-world balancing and Gaussian smoothing utilities improve color fidelity and suppress noise.
- **Quality evaluation**: Built-in PSNR and SSIM metrics help validate restoration quality.

## Project Layout
```
src/
  data/                # Datasets and defect simulation utilities
  models/              # Super-resolution and defect repair networks
  pipelines/           # End-to-end inference pipeline
  utils/               # Image I/O, metrics, logging, and color helpers
scripts/
  run_inference.py     # Command-line inference entry point
assets/                # Small demo images and documentation
assets/README.md       # Provenance notes for sample assets
tests/                 # Unit tests mirroring the source tree
```

## Environment Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```
Install the pre-commit hooks once via `pre-commit install` to keep formatting consistent.

## Run Tests
```bash
pytest
```

## Inference Example
```bash
python scripts/run_inference.py --input assets/samples --output outputs/demo --scale 4
```
Add `--save-intermediate` to persist intermediate repair stages.

## Data Notes
Large training datasets are not checked into the repository. Document acquisition steps or preprocessing scripts in `data/README.md`.
