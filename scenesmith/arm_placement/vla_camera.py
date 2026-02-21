"""VLA camera placement for Pi model integration.

Provides deterministic camera placement for Vision-Language-Action models:
- A fixed third-person camera behind/above the arm
- A wrist-mounted camera on the arm's link4 body

Both cameras are injected as <camera> elements directly into the scene XML,
making them available to any downstream MuJoCo consumer (Pi VLA, Blender, etc.).
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from math import atan2, cos, sin
from pathlib import Path

import mujoco
import numpy as np

from scenesmith.arm_placement.robot_camera import CAMERA_MOUNT_POINTS

console_logger = logging.getLogger(__name__)

# Wrist mount defaults pulled from the existing end_effector config.
_EE_MOUNT = CAMERA_MOUNT_POINTS["end_effector"]


@dataclass
class VLACameraConfig:
    """Configuration for VLA camera placement."""

    # Third-person camera.
    third_person_behind: float = 0.45   # Distance behind arm (metres).
    third_person_above: float = 0.55    # Height above the surface (metres).
    third_person_fovy: float = 70.0     # Vertical FOV (degrees).

    # Wrist camera — defaults from robot_camera.py end_effector mount.
    wrist_body: str = field(default_factory=lambda: _EE_MOUNT["body"])
    wrist_pos: list[float] = field(default_factory=lambda: list(_EE_MOUNT["pos"]))
    wrist_quat: list[float] = field(default_factory=lambda: list(_EE_MOUNT["quat"]))
    wrist_fovy: float = field(default_factory=lambda: float(_EE_MOUNT["fovy"]))


# ---------------------------------------------------------------------------
# Third-person camera heuristic
# ---------------------------------------------------------------------------

def compute_third_person_camera(
    arm_pos: np.ndarray,
    arm_quat: np.ndarray,
    config: VLACameraConfig | None = None,
) -> dict:
    """Compute third-person camera parameters from arm pose.

    The camera is placed behind the arm (relative to the arm's facing
    direction) and above the surface level, looking towards a point
    slightly in front of the arm.

    Args:
        arm_pos: (3,) world-frame position of the arm base.
        arm_quat: (4,) quaternion [w, x, y, z] of the arm base.
        config: Optional VLACameraConfig overrides.

    Returns:
        Dict with keys ``pos``, ``xyaxes``, ``fovy`` suitable for a
        MuJoCo ``<camera>`` element.
    """
    if config is None:
        config = VLACameraConfig()

    arm_pos = np.asarray(arm_pos, dtype=float)
    arm_quat = np.asarray(arm_quat, dtype=float)

    # Extract yaw from quaternion (rotation around Z).
    qw, _qx, _qy, qz = arm_quat
    theta = 2.0 * atan2(qz, qw)
    forward = np.array([cos(theta), sin(theta), 0.0])

    # Camera position: behind arm + above surface.
    cam_pos = (
        arm_pos
        - forward * config.third_person_behind
        + np.array([0.0, 0.0, config.third_person_above])
    )

    # Look-at point: slightly ahead of arm at surface level.
    lookat = arm_pos + forward * 0.2 + np.array([0.0, 0.0, 0.05])

    # Build camera frame.
    look_dir = lookat - cam_pos
    look_dir /= np.linalg.norm(look_dir)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(look_dir, world_up)
    right /= np.linalg.norm(right)

    up = np.cross(right, look_dir)
    up /= np.linalg.norm(up)

    # MuJoCo xyaxes: x-axis (right) then y-axis (up) of the camera frame.
    xyaxes = np.concatenate([right, up])

    return {
        "pos": cam_pos,
        "xyaxes": xyaxes,
        "fovy": config.third_person_fovy,
    }


# ---------------------------------------------------------------------------
# XML injection
# ---------------------------------------------------------------------------

def inject_vla_cameras(
    scene_xml_path: str | Path,
    config: VLACameraConfig | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Inject VLA cameras into a scene XML.

    Adds two ``<camera>`` elements:
    - ``vla_third_person`` under ``<worldbody>``
    - ``vla_wrist`` under the wrist body (default ``link4``)

    The operation is idempotent — existing VLA cameras are removed first.

    Args:
        scene_xml_path: Path to scene XML that already contains the arm.
        config: Camera configuration.  Defaults used if *None*.
        output_path: Where to write the modified XML.  Defaults to
            overwriting the input file.

    Returns:
        Path to the written XML file.
    """
    if config is None:
        config = VLACameraConfig()

    scene_xml_path = Path(scene_xml_path)
    if output_path is None:
        output_path = scene_xml_path
    output_path = Path(output_path)

    tree = ET.parse(scene_xml_path)
    root = tree.getroot()

    # --- Locate arm base body (omx_f_base) for position/quat. ---
    arm_body = None
    for elem in root.iter("body"):
        if elem.get("name") == "omx_f_base":
            arm_body = elem
            break

    if arm_body is None:
        raise ValueError(
            "Body 'omx_f_base' not found in scene XML. "
            "Is the arm merged into this scene?"
        )

    arm_pos_str = arm_body.get("pos", "0 0 0")
    arm_pos = np.array([float(v) for v in arm_pos_str.split()])

    arm_quat_str = arm_body.get("quat", "1 0 0 0")
    arm_quat = np.array([float(v) for v in arm_quat_str.split()])

    # --- Compute third-person camera. ---
    tp = compute_third_person_camera(arm_pos, arm_quat, config)

    # --- Remove existing VLA cameras (idempotency). ---
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene XML has no <worldbody>.")

    for cam in worldbody.findall("camera"):
        if cam.get("name") in ("vla_third_person",):
            worldbody.remove(cam)

    # Also search recursively for vla_wrist inside any body.
    for body in root.iter("body"):
        for cam in body.findall("camera"):
            if cam.get("name") == "vla_wrist":
                body.remove(cam)

    # --- Inject third-person camera under worldbody. ---
    tp_elem = ET.SubElement(worldbody, "camera")
    tp_elem.set("name", "vla_third_person")
    tp_elem.set(
        "pos",
        f"{tp['pos'][0]:.6f} {tp['pos'][1]:.6f} {tp['pos'][2]:.6f}",
    )
    tp_elem.set(
        "xyaxes",
        " ".join(f"{v:.6f}" for v in tp["xyaxes"]),
    )
    tp_elem.set("fovy", str(int(tp["fovy"])))

    # --- Inject wrist camera under the wrist body. ---
    wrist_body = None
    for elem in root.iter("body"):
        if elem.get("name") == config.wrist_body:
            wrist_body = elem
            break

    if wrist_body is None:
        raise ValueError(
            f"Wrist body '{config.wrist_body}' not found in scene XML."
        )

    wrist_elem = ET.SubElement(wrist_body, "camera")
    wrist_elem.set("name", "vla_wrist")
    wrist_elem.set(
        "pos",
        f"{config.wrist_pos[0]:.6f} {config.wrist_pos[1]:.6f} "
        f"{config.wrist_pos[2]:.6f}",
    )
    wrist_elem.set(
        "quat",
        f"{config.wrist_quat[0]:.6f} {config.wrist_quat[1]:.6f} "
        f"{config.wrist_quat[2]:.6f} {config.wrist_quat[3]:.6f}",
    )
    wrist_elem.set("fovy", str(int(config.wrist_fovy)))

    # --- Write output. ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), xml_declaration=True)
    console_logger.info(f"Injected VLA cameras → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------

def render_vla_camera_previews(
    scene_xml_path: str | Path,
    output_dir: str | Path,
    config: VLACameraConfig | None = None,
    width: int = 1280,
    height: int = 720,
) -> list[Path]:
    """Render preview images from the VLA cameras.

    Loads the scene (with rendering enhancements), resets to the home
    keyframe if available, and renders from both ``vla_third_person``
    and ``vla_wrist`` cameras.

    Args:
        scene_xml_path: Path to scene XML (must already have VLA cameras).
        output_dir: Directory to write preview PNGs.
        config: Unused for rendering but kept for API symmetry.
        width: Image width.
        height: Image height.

    Returns:
        List of paths to rendered PNG files.
    """
    from PIL import Image

    from scenesmith.arm_placement.mujoco_renderer import (
        _apply_render_flags,
        _create_scene_option,
        _enhance_scene_xml,
    )

    scene_xml_path = Path(scene_xml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _enhance_scene_xml(scene_xml_path, quality="high")
    data = mujoco.MjData(model)

    # Reset to home keyframe if available (compact arm pose).
    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        pass

    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = _create_scene_option()

    camera_names = ["vla_third_person", "vla_wrist"]
    output_paths: list[Path] = []

    for cam_name in camera_names:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            console_logger.warning(f"Camera '{cam_name}' not found, skipping preview.")
            continue

        renderer.update_scene(data, camera=cam_name, scene_option=scene_option)
        _apply_render_flags(renderer, shadows=True, reflections=True)
        pixels = renderer.render()

        out_path = output_dir / f"{cam_name}.png"
        Image.fromarray(pixels).save(str(out_path))
        output_paths.append(out_path)
        console_logger.info(f"Saved VLA preview: {out_path}")

    renderer.close()
    return output_paths
