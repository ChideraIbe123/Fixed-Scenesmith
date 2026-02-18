#!/usr/bin/env python3
"""CLI entry point for placing a robot arm in a MuJoCo scene.

Uses a hierarchical Planner/Designer/Critic agent system to intelligently
place an OpenManipulator-X arm in an exported MuJoCo scene.

Usage:
    python scripts/place_arm_in_scene.py \\
        outputs/.../scene_000/mujoco/scene.xml \\
        --arm OMX-Files-/open_manipulator_description/urdf/omx_f/omx_f.urdf \\
        --output outputs/.../scene_000/mujoco_with_arm/ \\
        --max-critique-rounds 3

    # Manual placement (no agent):
    python scripts/place_arm_in_scene.py \\
        outputs/.../scene_000/mujoco/scene.xml \\
        --arm OMX-Files-/open_manipulator_description/urdf/omx_f/omx_f.urdf \\
        --output outputs/.../scene_000/mujoco_with_arm/ \\
        --no-agent --surface kitchen_island --x-offset 0.1 --z-rotation 90
"""

import argparse
import asyncio
import logging
import sys

from pathlib import Path

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Place a robot arm in a MuJoCo scene using AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "scene_xml",
        type=Path,
        help="Path to the MuJoCo scene XML file.",
    )
    parser.add_argument(
        "--arm",
        type=Path,
        default=Path(
            "OMX-Files-/open_manipulator_description/urdf/omx_f/omx_f.urdf"
        ),
        help="Path to the arm URDF file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to scene_dir/mujoco_with_arm/.",
    )

    # Agent configuration.
    agent_group = parser.add_argument_group("Agent configuration")
    agent_group.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2).",
    )
    agent_group.add_argument(
        "--max-critique-rounds",
        type=int,
        default=3,
        help="Maximum critique/design cycles (default: 3).",
    )
    agent_group.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum planner turns (default: 30).",
    )
    agent_group.add_argument(
        "--early-finish-score",
        type=int,
        default=9,
        help="Min score for early finish (default: 9).",
    )

    # Manual placement (no agent).
    manual_group = parser.add_argument_group("Manual placement (--no-agent)")
    manual_group.add_argument(
        "--no-agent",
        action="store_true",
        help="Skip AI agent, place arm manually.",
    )
    manual_group.add_argument(
        "--surface",
        type=str,
        help="Surface body name for manual placement.",
    )
    manual_group.add_argument(
        "--x-offset",
        type=float,
        default=0.0,
        help="X offset from surface center.",
    )
    manual_group.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        help="Y offset from surface center.",
    )
    manual_group.add_argument(
        "--z-rotation",
        type=float,
        default=0.0,
        help="Z rotation in degrees.",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


async def run_agent_placement(args: argparse.Namespace) -> None:
    """Run the AI agent pipeline for arm placement."""
    from scenesmith.arm_placement.arm_placement_agent import (
        ArmPlacementAgent,
        ArmPlacementConfig,
    )

    cfg = ArmPlacementConfig(
        model=args.model,
        max_turns=args.max_turns,
        max_critique_rounds=args.max_critique_rounds,
        early_finish_min_score=args.early_finish_score,
    )

    output_dir = args.output or args.scene_xml.parent / "mujoco_with_arm"

    agent = ArmPlacementAgent(
        scene_xml_path=args.scene_xml.resolve(),
        arm_urdf_path=args.arm.resolve(),
        output_dir=output_dir.resolve(),
        cfg=cfg,
    )

    result = await agent.place_arm()

    # Print summary.
    print("\n" + "=" * 60)
    print("ARM PLACEMENT RESULT")
    print("=" * 60)
    if result.success:
        print(f"  Status: SUCCESS")
        print(f"  Surface: {result.arm_surface}")
        print(f"  Position: {result.arm_position}")
        print(f"  Rotation: {result.arm_rotation:.1f} degrees")
        print(f"  Output: {result.scene_xml_path}")
        if result.final_scores:
            print("\n  Final Scores:")
            for score in result.final_scores.get_scores():
                print(f"    {score.name}: {score.grade}/10 - {score.comment}")
    else:
        print(f"  Status: FAILED")
        print(f"  Error: {result.error}")
    print("=" * 60)


def run_manual_placement(args: argparse.Namespace) -> None:
    """Place the arm manually without AI agents."""
    import math

    from scenesmith.arm_placement.scene_analyzer import analyze_scene
    from scenesmith.arm_placement.tools.designer_tools import DesignerTools
    from scenesmith.arm_placement.urdf_to_mjcf import convert_urdf_to_mjcf

    output_dir = args.output or args.scene_xml.parent / "mujoco_with_arm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert URDF.
    print("Converting URDF to MJCF...")
    arm_mjcf_path = convert_urdf_to_mjcf(
        urdf_path=args.arm.resolve(),
        output_dir=output_dir.resolve(),
    )
    print(f"  Arm MJCF: {arm_mjcf_path}")

    # Analyze scene.
    print("Analyzing scene...")
    scene_desc = analyze_scene(args.scene_xml.resolve())
    print(f"  Room: {scene_desc.room_type}")
    print(f"  Surfaces: {len(scene_desc.surfaces)}")

    if not args.surface:
        print("\nAvailable surfaces:")
        for s in scene_desc.surfaces:
            print(f"  {s.to_text()}")
        print("\nSpecify --surface <name> to place the arm.")
        return

    # Create designer tools for the merge functionality.
    tools_obj = DesignerTools(
        scene_xml_path=args.scene_xml.resolve(),
        arm_mjcf_path=arm_mjcf_path,
        output_dir=output_dir.resolve(),
        scene_description=scene_desc,
    )

    result = tools_obj._place_arm_impl(
        args.surface, args.x_offset, args.y_offset, args.z_rotation
    )
    print(f"\n{result}")

    # Render final scene.
    if tools_obj.arm_placed:
        try:
            from scenesmith.arm_placement.mujoco_renderer import render_scene

            render_dir = output_dir / "renders" / "final"
            print(f"\nRendering final scene to {render_dir}...")
            render_scene(
                scene_xml_path=tools_obj.current_scene_xml,
                output_dir=render_dir,
            )
            print("Done.")
        except Exception as e:
            print(f"\nRendering skipped (no display available): {e}")
            print("The merged scene XML is still valid and can be loaded in MuJoCo.")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Validate inputs.
    if not args.scene_xml.exists():
        print(f"ERROR: Scene XML not found: {args.scene_xml}")
        sys.exit(1)
    if not args.arm.exists():
        print(f"ERROR: Arm URDF not found: {args.arm}")
        sys.exit(1)

    if args.no_agent:
        run_manual_placement(args)
    else:
        asyncio.run(run_agent_placement(args))


if __name__ == "__main__":
    main()
