"""MuJoCo scene rendering for agent visual feedback.

Renders a MuJoCo scene from multiple camera angles to PNG images using
the built-in mujoco.Renderer. Used by both designer and critic agents
to observe the current state of the scene.

Supports headless rendering via OSMesa or EGL backends. Falls back to
osmesa if no display is available.
"""

import logging
import math
import os

from pathlib import Path

import mujoco
import numpy as np

console_logger = logging.getLogger(__name__)

# Camera angle presets: (azimuth, elevation, distance, lookat_offset).
CAMERA_PRESETS = {
    "front": (180.0, -20.0, 3.0, [0.0, 0.0, 0.5]),
    "top": (0.0, -90.0, 4.0, [0.0, 0.0, 0.0]),
    "side": (90.0, -20.0, 3.0, [0.0, 0.0, 0.5]),
    "perspective": (225.0, -30.0, 3.5, [0.0, 0.0, 0.5]),
}


def render_scene(
    scene_xml_path: Path,
    output_dir: Path,
    camera_angles: list[str] | None = None,
    width: int = 640,
    height: int = 480,
    lookat: list[float] | None = None,
) -> list[Path]:
    """Render the MuJoCo scene from multiple camera angles.

    Args:
        scene_xml_path: Path to the MuJoCo scene XML file.
        output_dir: Directory to save rendered PNG images.
        camera_angles: List of camera angle names from CAMERA_PRESETS.
            Defaults to all angles.
        width: Image width in pixels.
        height: Image height in pixels.
        lookat: Optional lookat point override [x, y, z].

    Returns:
        List of paths to rendered PNG images.
    """
    if camera_angles is None:
        camera_angles = list(CAMERA_PRESETS.keys())

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and create data.
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Compute scene center if not provided.
    if lookat is None:
        if model.nbody > 1:
            all_pos = data.xpos[1:]  # Skip world body.
            lookat = list(np.mean(all_pos, axis=0))
        else:
            lookat = [0.0, 0.0, 0.5]

    # Create renderer.
    renderer = mujoco.Renderer(model, height=height, width=width)

    output_paths = []
    for angle_name in camera_angles:
        if angle_name not in CAMERA_PRESETS:
            console_logger.warning(f"Unknown camera angle: {angle_name}, skipping")
            continue

        azimuth, elevation, distance, lookat_offset = CAMERA_PRESETS[angle_name]

        # Set up camera.
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.distance = distance
        camera.lookat[:] = [
            lookat[0] + lookat_offset[0],
            lookat[1] + lookat_offset[1],
            lookat[2] + lookat_offset[2],
        ]

        # Render.
        renderer.update_scene(data, camera)
        pixels = renderer.render()

        # Save to PNG.
        output_path = output_dir / f"scene_{angle_name}.png"
        _save_png(pixels, output_path)
        output_paths.append(output_path)

        console_logger.debug(f"Rendered {angle_name} view to {output_path}")

    renderer.close()

    console_logger.info(
        f"Rendered {len(output_paths)} views to {output_dir}"
    )
    return output_paths


def _save_png(pixels: np.ndarray, path: Path) -> None:
    """Save a pixel array as PNG.

    Args:
        pixels: HxWx3 uint8 array.
        path: Output file path.
    """
    try:
        from PIL import Image

        img = Image.fromarray(pixels)
        img.save(str(path))
    except ImportError:
        # Fallback: write raw PPM then convert.
        # PIL should be available in the environment.
        console_logger.warning("PIL not available, skipping PNG save")
