#!/usr/bin/env python3
"""Run a Pi0.5 VLA control episode on a MuJoCo scene with an OMX-F arm.

Usage:
    python scripts/run_vla_control.py \
        outputs/.../scene_with_cameras.xml \
        --server-host 136.113.180.52 --server-port 8000 \
        --prompt "pick up the cup" \
        --action-scale 0.1 --max-steps 600 \
        --record --record-dir renders/vla_episode/
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure MUJOCO_GL is set before any mujoco import.
import os
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Pi0.5 VLA control loop on a MuJoCo arm scene.",
    )
    parser.add_argument(
        "scene_xml",
        type=Path,
        help="Path to scene XML with arm and VLA cameras injected.",
    )
    parser.add_argument(
        "--server-host",
        default="136.113.180.52",
        help="Pi0.5 policy server hostname/IP (default: 136.113.180.52).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Pi0.5 policy server port (default: 8000).",
    )
    parser.add_argument(
        "--prompt",
        default="pick up the object",
        help="Task instruction for the VLA model.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum episode steps (default: 600).",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.1,
        help="Velocity-to-qpos scaling factor (default: 0.1).",
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=15.0,
        help="Control loop frequency in Hz (default: 15).",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=8,
        help="Steps between server queries (default: 8).",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Gripper binarization threshold (default: 0.5).",
    )
    parser.add_argument(
        "--gripper-close-qpos",
        type=float,
        default=0.5,
        help="Gripper closed position (default: 0.5).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record episode frames and save video.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path("renders/vla_episode"),
        help="Directory for recorded frames/video.",
    )
    parser.add_argument(
        "--add-test-object",
        action="store_true",
        help="Inject a small red cup near the arm for pickup testing.",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Use mj_step with position actuators for real contacts/grasping.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Auto-enable physics when test object is present (no point without it).
    if args.add_test_object and not args.physics:
        args.physics = True

    # Configure logging.
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate scene XML exists.
    if not args.scene_xml.exists():
        print(f"ERROR: Scene XML not found: {args.scene_xml}", file=sys.stderr)
        sys.exit(1)

    from scenesmith.arm_placement.vla_control import (
        VLAControlConfig,
        VLAControlLoop,
    )

    config = VLAControlConfig(
        server_host=args.server_host,
        server_port=args.server_port,
        prompt=args.prompt,
        control_freq=args.control_freq,
        open_loop_horizon=args.open_loop_horizon,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        gripper_threshold=args.gripper_threshold,
        gripper_close_qpos=args.gripper_close_qpos,
        add_test_object=args.add_test_object,
        physics=args.physics,
        record=args.record,
        record_dir=args.record_dir,
    )

    print(f"Scene:        {args.scene_xml}")
    print(f"Server:       {config.server_host}:{config.server_port}")
    print(f"Prompt:       '{config.prompt}'")
    print(f"Max steps:    {config.max_steps}")
    print(f"Action scale: {config.action_scale}")
    print(f"Control freq: {config.control_freq} Hz")
    print(f"Horizon:      {config.open_loop_horizon}")
    print(f"Physics:      {config.physics}")
    print(f"Record:       {config.record}")
    print()

    with VLAControlLoop(args.scene_xml, config) as loop:
        result = loop.run()

    print()
    print("=== Episode Results ===")
    print(f"Steps:        {result['steps']}")
    print(f"Duration:     {result['duration']:.1f}s")
    print(f"Effective Hz: {result['effective_hz']:.1f}")
    if "recording_path" in result:
        print(f"Recording:    {result['recording_path']}")


if __name__ == "__main__":
    main()
