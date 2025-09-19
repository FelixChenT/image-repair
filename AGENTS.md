# Repository Guidelines

## Project Structure & Module Organization
Keep production code under src/, separated into src/data/ for loaders and augmentations, src/models/ for the super-resolution and inpainting networks, and src/pipelines/ for end-to-end orchestration. Lightweight utilities (metrics, logging) belong in src/utils/. Command-line entry points and experiment scripts live in scripts/ (e.g., scripts/run_inference.py). Tests mirror the tree inside 	ests/, while Jupyter experiments stay in 
otebooks/ checked in with stripped outputs. Store small demo assets in ssets/ with a README describing provenance; large datasets stay out of the repo and are referenced through data/README.md.

## Build, Test, and Development Commands
Bootstrap once with python -m venv .venv followed by ./.venv/Scripts/activate and pip install -r requirements.txt. Use pre-commit install to enable linting hooks. Run pre-commit run --all-files before any push. Execute the main test suite via pytest (unit tests) and pytest tests/integration for full pipeline checks. Quickly preview the model with python scripts/run_inference.py --input assets/samples --output outputs/demo.

## Coding Style & Naming Conventions
Adopt PEP 8 with 4-space indentation and line length ¡Ü 100. Modules and functions use snake_case, classes use PascalCase, experiment configs use kebab-case (e.g., configs/sr-x4.yaml). Format code with lack and lint with uff; enforce typing hints on public functions and prefer 
umpy.typing or 	orch annotations where relevant.

## Testing Guidelines
Write unit tests alongside each module (	ests/models/test_srnet.py). Favor deterministic fixtures using cached sample images. For restoration quality, include perceptual metric assertions (PSNR, SSIM thresholds). Update or regenerate golden outputs with scripts/build_golden.py and document changes in 	ests/README.md. Aim for ¡Ý85% coverage per pytest --cov=src.

## Commit & Pull Request Guidelines
No git history exists yet; start with Conventional Commits (e.g., eat: add x4 super-resolution pipeline). Limit each pull request to one feature or fix, describe datasets and configs touched, and attach before/after thumbnails for visual changes. Link issue IDs when available and note any files larger than 20 MB kept out of git.

## Data & Security Notes
Respect privacy: do not commit personal photos without anonymization or consent. Store API keys or model weights in .env files tracked via .env.example, and rotate shared credentials quarterly.
