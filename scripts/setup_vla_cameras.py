#!/usr/bin/env python3
"""Standalone CLI for injecting VLA cameras into a MuJoCo scene.

Usage:
    python scripts/setup_vla_cameras.py \\
        outputs/.../scene_with_arm.xml \\
        --output outputs/.../scene_with_cameras.xml \\
        --behind 0.45 --above 0.55 --fovy 70 \\
        --preview --preview-dir renders/vla_preview/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject VLA cameras into a MuJoCo scene with an arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "scene_xml",
        type=Path,
        help="Path to scene XML (must already contain the arm).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output XML path. Defaults to scene_with_cameras.xml in the same dir.",
    )

    # Third-person camera params.
    tp = parser.add_argument_group("Third-person camera")
    tp.add_argument("--behind", type=float, default=0.45, help="Distance behind arm (m).")
    tp.add_argument("--above", type=float, default=0.55, help="Height above surface (m).")
    tp.add_argument("--fovy", type=float, default=70.0, help="Vertical FOV (degrees).")

    # Wrist camera params.
    wrist = parser.add_argument_group("Wrist camera")
    wrist.add_argument("--wrist-body", type=str, default=None, help="Body to mount wrist camera on.")
    wrist.add_argument("--wrist-fovy", type=float, default=None, help="Wrist camera FOV.")

    # Preview.
    prev = parser.add_argument_group("Preview rendering")
    prev.add_argument("--preview", action="store_true", help="Render preview images.")
    prev.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Directory for preview PNGs. Defaults to renders/vla_preview/ next to output.",
    )
    prev.add_argument("--preview-width", type=int, default=1280)
    prev.add_argument("--preview-height", type=int, default=720)

    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.scene_xml.exists():
        print(f"ERROR: Scene XML not found: {args.scene_xml}")
        sys.exit(1)

    from scenesmith.arm_placement.vla_camera import (
        VLACameraConfig,
        inject_vla_cameras,
        render_vla_camera_previews,
    )

    # Build config from CLI args.
    config_kwargs: dict = {
        "third_person_behind": args.behind,
        "third_person_above": args.above,
        "third_person_fovy": args.fovy,
    }
    if args.wrist_body is not None:
        config_kwargs["wrist_body"] = args.wrist_body
    if args.wrist_fovy is not None:
        config_kwargs["wrist_fovy"] = args.wrist_fovy

    config = VLACameraConfig(**config_kwargs)

    # Determine output path.
    output_path = args.output
    if output_path is None:
        output_path = args.scene_xml.parent / "scene_with_cameras.xml"

    # Inject cameras.
    print(f"Injecting VLA cameras into: {args.scene_xml}")
    result_path = inject_vla_cameras(args.scene_xml, config=config, output_path=output_path)
    print(f"  Output: {result_path}")
    print(f"  Third-person: behind={config.third_person_behind}m, above={config.third_person_above}m, fovy={config.third_person_fovy}°")
    print(f"  Wrist: body={config.wrist_body}, fovy={config.wrist_fovy}°")

    # Optional preview rendering.
    if args.preview:
        preview_dir = args.preview_dir
        if preview_dir is None:
            preview_dir = result_path.parent / "renders" / "vla_preview"

        print(f"\nRendering previews to: {preview_dir}")
        previews = render_vla_camera_previews(
            result_path,
            output_dir=preview_dir,
            config=config,
            width=args.preview_width,
            height=args.preview_height,
        )
        for p in previews:
            print(f"  {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
