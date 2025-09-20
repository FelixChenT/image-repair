# Dataset and Training Guide

This document explains how to collect data, prepare the environment, and train the image repair models. It is written for newcomers, so each step is spelled out in detail.

## 1. Hardware Checklist
- **Minimum viable setup**: 8 CPU cores, 16 GB RAM, and at least 50 GB of free disk space. Training will run on CPU but will be very slow (multiple days for medium datasets).
- **Recommended setup**: NVIDIA GPU with 8 GB VRAM (e.g., RTX 3060 or better), 32 GB system RAM, SSD storage. A stronger GPU shortens training to hours instead of days.
- **Optional extras**: An external drive for raw scans, and a calibrated monitor for inspecting visual results.

## 2. Software Prerequisites
- Windows 10/11, macOS 12+, or a recent Linux distribution.
- [Python 3.10+](https://www.python.org/downloads/) with `pip` installed.
- [Git](https://git-scm.com/) for downloading the repository (or download the ZIP archive manually).
- Up-to-date GPU drivers and CUDA toolkit if you plan to train on an NVIDIA GPU.

## 3. Project Setup (once per machine)
1. Clone or extract the repository.
   ```bash
   git clone https://github.com/your-org/image-repair.git
   cd image-repair
   ```
2. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   # PowerShell
   . .venv/Scripts/Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```
3. Install dependencies.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. (Optional but recommended) enable formatting and linting hooks.
   ```bash
   pre-commit install
   ```

## 4. Organise Your Dataset
1. Collect original photos (PNG, JPG, or TIFF). Higher resolution images (>1024 px on the short side) produce better results.
2. Place all images inside a dedicated directory, for example:
   ```
   data/
     raw_photos/
       family_001.jpg
       family_002.png
       ...
   ```
3. Keep a simple naming scheme (no spaces, ASCII characters) to avoid path issues.
4. Ensure every image is in RGB. For black-and-white scans, convert them to RGB using any image editor or the script below:
   ```bash
   python - <<'PY'
   from pathlib import Path
   from PIL import Image

   root = Path('data/raw_photos')
   for path in root.rglob('*'):
       if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
           image = Image.open(path).convert('RGB')
           image.save(path)
   PY
   ```

## 5. Optional Pre-processing Tips
- Crop away borders or scanner artefacts before training.
- If your scans vary wildly in size, consider resizing them so the shortest side is between 720 and 1080 pixels.
- Store a validation subset (e.g., 10% of the images) in a separate folder if you plan to monitor generalisation.

## 6. Start Training
The repository includes `scripts/train.py`, which trains both the defect repair module and the multi-scale super-resolution model using synthetic degradations.

### Basic training run
```bash
python scripts/train.py \
  --data-root data/raw_photos \
  --epochs 10 \
  --batch-size 4 \
  --patch-size 128 \
  --scale 4 \
  --device cuda \
  --output-dir outputs/checkpoints
```
- Set `--device cpu` if you do not have a GPU (expect extremely long runtimes).
- `--patch-size` must be divisible by the upscale `--scale`. Increase to 192 or 256 if your GPU has enough memory.
- Checkpoints are written to `outputs/checkpoints`, one file per epoch.

### Recommended monitoring
- Watch the console for **SR Loss**, **Repair Loss**, **PSNR**, and **SSIM**. Rising PSNR/SSIM indicate better reconstructions.
- Use a logging tool (e.g., [Weights & Biases](https://wandb.ai/), TensorBoard) by extending `train.py` if you need richer charts.

### Resuming training
To resume from a checkpoint, pass `--resume outputs/checkpoints/checkpoint_epoch_010.pth` (add this flag after extending the script; a simple resume helper is easy to implement if needed).

## 7. Validate the Result
1. Copy a handful of untouched photos into `assets/samples` (or any folder you prefer).
2. Run inference with your trained models (update paths if you stored checkpoints elsewhere):
   ```bash
   python scripts/run_inference.py \
     --input assets/samples \
     --output outputs/demo \
     --scale 4 \
     --device cuda \
     --save-intermediate
   ```
3. Inspect the generated files:
   - `_enhanced.png`: final result after colour and noise adjustments.
   - `_repaired.png`, `_x4.png`, `_color.png`: intermediate outputs that help debug issues.

## 8. Practical Advice
- **Start small**: run 1-2 epochs on a dozen photos to verify everything works before launching a long training run.
- **Backup**: keep raw scans and checkpoints in a safe location (cloud storage or external drive).
- **Version control**: commit configuration changes (learning rate, patch size) so you can reproduce results later.
- **Quality checks**: compare PSNR/SSIM against a validation subset; large drops often signal overfitting or data issues.

## 9. Troubleshooting
- `CUDA out of memory`: lower `--batch-size`, reduce `--patch-size`, or switch to `--device cpu`.
- `No image files discovered`: double-check the `--data-root` path and file extensions.
- `Training is extremely slow`: confirm the script is running on GPU (`nvidia-smi` should show activity) or upgrade hardware.
- `Results look blurry`: keep training for more epochs, gather higher-resolution data, or experiment with larger patch sizes.

With these steps, even a newcomer can move from raw photos to a trained restoration model. Continue refining the dataset and hyperparameters to reach production-ready quality.
