"""Run the image repair pipeline over a directory of images."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.pipelines import ImageRepairPipeline
from src.utils import load_image, save_image, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image repair and super-resolution inference")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing input images")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store enhanced images")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4], help="Upscale factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on (cpu or cuda)")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Persist intermediate outputs alongside the final result",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("scripts.run_inference")

    input_dir = args.input
    output_dir = args.output
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    device = torch.device(args.device)
    pipeline = ImageRepairPipeline(device=device)

    image_paths = sorted(
        [p for p in input_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    if not image_paths:
        logger.warning("No images discovered under %s", input_dir)
        return

    for path in image_paths:
        logger.info("Processing %s", path)
        image = load_image(path, device=device)
        result = pipeline(image, scale=args.scale)

        relative = path.relative_to(input_dir)
        destination_root = output_dir / relative.parent
        destination_root.mkdir(parents=True, exist_ok=True)
        base_name = relative.stem

        save_image(result.denoised, destination_root / f"{base_name}_enhanced.png")
        if args.save_intermediate:
            save_image(result.repaired_low_res, destination_root / f"{base_name}_repaired.png")
            save_image(result.super_resolved, destination_root / f"{base_name}_x{args.scale}.png")
            save_image(result.color_restored, destination_root / f"{base_name}_color.png")
    logger.info("Processing complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
