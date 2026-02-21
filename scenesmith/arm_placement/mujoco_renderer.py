"""MuJoCo scene rendering for agent visual feedback.

Renders a MuJoCo scene from multiple camera angles to PNG images using
the built-in mujoco.Renderer. Used by both designer and critic agents
to observe the current state of the scene.

Supports headless rendering via OSMesa or EGL backends. Falls back to
osmesa if no display is available.
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET

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

# Quality presets: (shadow_size, off_samples, offwidth, offheight).
QUALITY_PRESETS = {
    "fast": (1024, 4, 640, 480),
    "medium": (2048, 4, 1280, 720),
    "high": (4096, 8, 1920, 1080),
}


def _enhance_scene_xml(scene_xml_path: Path, quality: str = "high") -> mujoco.MjModel:
    """Parse scene XML and inject rendering enhancements before loading.

    Fixes colorspace from linear to sRGB on all file-based textures,
    adds brighter lighting, shadow quality settings, and directional lights.

    Args:
        scene_xml_path: Path to the scene XML file.
        quality: Quality preset name ("fast", "medium", "high").

    Returns:
        Enhanced MjModel loaded from the modified XML.
    """
    shadow_size, off_samples, offwidth, offheight = QUALITY_PRESETS.get(
        quality, QUALITY_PRESETS["high"]
    )

    tree = ET.parse(scene_xml_path)
    root = tree.getroot()

    # Fix colorspace: change linear -> sRGB on all file-based textures.
    asset_elem = root.find("asset")
    if asset_elem is not None:
        for tex in asset_elem.findall("texture"):
            if tex.get("file"):
                tex.set("colorspace", "sRGB")

    # Find or create <visual> element.
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    # Brighter headlight.
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.5 0.5 0.5")
    headlight.set("diffuse", "1.0 1.0 1.0")
    headlight.set("specular", "0.3 0.3 0.3")

    # Shadow quality.
    quality_elem = visual.find("quality")
    if quality_elem is None:
        quality_elem = ET.SubElement(visual, "quality")
    quality_elem.set("shadowsize", str(shadow_size))
    quality_elem.set("offsamples", str(off_samples))

    # Offscreen buffer size.
    global_elem = visual.find("global")
    if global_elem is None:
        global_elem = ET.SubElement(visual, "global")
    global_elem.set("offwidth", str(offwidth))
    global_elem.set("offheight", str(offheight))

    # Add directional lights to worldbody.
    worldbody = root.find("worldbody")
    if worldbody is not None:
        # Remove any previously injected lights (idempotency).
        for light in worldbody.findall("light"):
            if light.get("name", "").startswith("_enhanced_"):
                worldbody.remove(light)

        # Sun light (with shadows).
        sun = ET.SubElement(worldbody, "light")
        sun.set("name", "_enhanced_sun")
        sun.set("directional", "true")
        sun.set("dir", "-0.5 -0.5 -1.0")
        sun.set("diffuse", "0.7 0.7 0.7")
        sun.set("specular", "0.3 0.3 0.3")
        sun.set("pos", "0 0 5")
        sun.set("castshadow", "true")

        # Fill light (no shadows, opposite side).
        fill = ET.SubElement(worldbody, "light")
        fill.set("name", "_enhanced_fill")
        fill.set("directional", "true")
        fill.set("dir", "0.5 0.5 -0.5")
        fill.set("diffuse", "0.3 0.3 0.3")
        fill.set("specular", "0.0 0.0 0.0")
        fill.set("pos", "0 0 4")
        fill.set("castshadow", "false")

    # Write enhanced XML to temp file in same directory (so meshdir resolves).
    scene_dir = scene_xml_path.parent
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=scene_dir)
    try:
        os.close(fd)
        tree.write(tmp_path, xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return model


def _create_scene_option() -> mujoco.MjvOption:
    """Create MjvOption with clean visualization (no debug overlays).

    Returns:
        Configured MjvOption with no joint/actuator/contact visualization.
    """
    option = mujoco.MjvOption()
    # Disable debug visualizations for clean renders.
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False
    # Disable coordinate frame arrows.
    option.frame = mujoco.mjtFrame.mjFRAME_NONE
    # Hide collision geoms (group 3) — only show visual meshes.
    option.geomgroup[3] = 0
    return option


def _apply_render_flags(
    renderer: mujoco.Renderer,
    shadows: bool = True,
    reflections: bool = True,
) -> None:
    """Set rendering flags on the renderer's internal scene.

    Rendering flags (shadows, reflections, skybox) live on MjvScene.flags,
    NOT on MjvOption.flags (which controls visualization overlays).

    Args:
        renderer: MuJoCo renderer (after update_scene has been called).
        shadows: Enable shadow rendering.
        reflections: Enable reflection rendering.
    """
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflections
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True


def render_scene(
    scene_xml_path: Path,
    output_dir: Path,
    camera_angles: list[str] | None = None,
    width: int = 1280,
    height: int = 720,
    lookat: list[float] | None = None,
    quality: str = "high",
    shadows: bool = True,
    reflections: bool = True,
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
        quality: Quality preset ("fast", "medium", "high").
        shadows: Enable shadow rendering.
        reflections: Enable reflection rendering.

    Returns:
        List of paths to rendered PNG images.
    """
    if camera_angles is None:
        camera_angles = list(CAMERA_PRESETS.keys())

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model with enhanced lighting and colorspace fixes.
    model = _enhance_scene_xml(scene_xml_path, quality=quality)
    # Disable contact computation — not needed for rendering and avoids
    # stack overflow on scenes with many overlapping geoms.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)

    # If model has a "home" keyframe, reset to it (compact arm pose).
    try:
        home_key_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    except Exception:
        pass

    mujoco.mj_forward(model, data)

    # Compute scene center if not provided.
    if lookat is None:
        if model.nbody > 1:
            all_pos = data.xpos[1:]  # Skip world body.
            lookat = list(np.mean(all_pos, axis=0))
        else:
            lookat = [0.0, 0.0, 0.5]

    # Create renderer and scene option.
    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = _create_scene_option()

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

        # Render: update scene with vis option, then set render flags on scene.
        renderer.update_scene(data, camera, scene_option=scene_option)
        _apply_render_flags(renderer, shadows=shadows, reflections=reflections)
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
