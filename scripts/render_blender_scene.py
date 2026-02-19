#!/usr/bin/env python3
"""
Render the kitchen scene with robot arm in Blender.

Loads the existing house.blend (preserving PBR materials, textures, lighting),
imports OMX-F arm STL meshes at the MuJoCo-computed placement, and renders
high-quality stills + an orbit video using EEVEE.

Usage:
    python scripts/render_blender_scene.py
"""

import bpy
import math
import mathutils
import os
import subprocess

# ─── Configuration ───────────────────────────────────────────────────────────

BLEND_FILE = "/workspace/scenesmith/outputs/2026-02-19/05-01-54/scene_000/combined_house/house.blend"
STL_DIR = "/workspace/scenesmith/OMX-Files-/open_manipulator_description/meshes/omx_f"
OUTPUT_DIR = "/workspace/scenesmith/renders/blender"

# Arm base placement from MuJoCo result
ARM_POS = (2.179693, 2.620423, 0.788227)
ARM_QUAT_WXYZ = (0.382683, 0.0, 0.0, 0.923880)  # 135° around Z

# Arm base world transform
ARM_BASE_MATRIX = mathutils.Matrix.LocRotScale(
    mathutils.Vector(ARM_POS),
    mathutils.Quaternion(ARM_QUAT_WXYZ),
    mathutils.Vector((1, 1, 1)),
)

# URDF kinematic chain: (stl_filename, joint_offset_xyz, joint_axis, default_angle)
# All STLs are in millimeters, scale 0.001
KINEMATIC_CHAIN = [
    ("follower_01_base.stl",              (0.0, 0.0, 0.0),        None, 0.0),
    ("follower_02_base_tilt_Revised.stl", (-0.01125, 0.0, 0.034), "Z",  0.0),
    ("follower_03_middle_verticle.stl",   (0.0, 0.0, 0.0635),     "Y",  0.0),
    ("follower_04_middle_horizontal.stl", (0.0415, 0.0, 0.11315), "Y",  0.0),
    ("follower_05_tip.stl",               (0.162, 0.0, 0.0),      "Y",  0.0),
    ("follower_06_pan_Revised.stl",       (0.0287, 0.0, 0.0),     "X",  0.0),
    ("follower_07_gripper_motorized.stl", (0.0295, 0.0075, 0.0),  None, 0.0),
    ("follower_08_gripper_gear.stl",      (0.0295, -0.0108, 0.0), None, 0.0),
]

STL_SCALE = 0.001  # mm to meters

# Render settings
RESOLUTION = (1920, 1080)
SAMPLES = 64
VIDEO_FRAMES = 180
VIDEO_FPS = 30

# Camera orbit — room is ~4.6x4.5m, arm at center
ORBIT_RADIUS = 1.2
ORBIT_HEIGHT = 1.4
ORBIT_TARGET = mathutils.Vector(ARM_POS) + mathutils.Vector((0, 0, 0.15))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def import_stl(filepath):
    """Import an STL file and return the newly created object."""
    existing = set(bpy.data.objects)
    bpy.ops.wm.stl_import(filepath=filepath)
    new_objs = set(bpy.data.objects) - existing
    if not new_objs:
        raise RuntimeError(f"No object created from {filepath}")
    return new_objs.pop()


def create_arm_material():
    """Create a dark grey metallic material for the arm."""
    mat = bpy.data.materials.new(name="ArmMetal")
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.6
    bsdf.inputs["Roughness"].default_value = 0.4

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def build_and_place_arm():
    """Import arm STLs, position via kinematic chain, place at arm base.

    Each part's world matrix = ARM_BASE_MATRIX @ cumulative_chain_matrix.
    This avoids Blender parenting issues.
    """
    arm_material = create_arm_material()
    arm_objects = []
    cumulative = mathutils.Matrix.Identity(4)

    for i, (stl_name, offset, axis, angle) in enumerate(KINEMATIC_CHAIN):
        stl_path = os.path.join(STL_DIR, stl_name)
        if not os.path.exists(stl_path):
            print(f"  WARNING: STL not found: {stl_path}")
            continue

        # Accumulate joint offset
        joint_mat = mathutils.Matrix.Translation(offset)
        if axis and angle != 0.0:
            axis_vec = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis]
            joint_mat = joint_mat @ mathutils.Matrix.Rotation(angle, 4, axis_vec)
        cumulative = cumulative @ joint_mat

        # Import and scale STL (mm → m)
        obj = import_stl(stl_path)
        obj.name = f"arm_{i:02d}_{stl_name.replace('.stl', '')}"
        obj.scale = (STL_SCALE, STL_SCALE, STL_SCALE)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(scale=True)
        obj.select_set(False)

        # Place in world: arm base transform @ local kinematic chain
        obj.matrix_world = ARM_BASE_MATRIX @ cumulative

        # Assign material
        if obj.data.materials:
            obj.data.materials[0] = arm_material
        else:
            obj.data.materials.append(arm_material)

        arm_objects.append(obj)
        loc = obj.matrix_world.translation
        print(f"  {stl_name}: world ({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f})")

    return arm_objects


def setup_camera():
    """Create a camera for rendering."""
    cam_data = bpy.data.cameras.new(name="RenderCam")
    cam_data.lens = 35
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def set_camera_orbit(cam_obj, angle_deg, radius=ORBIT_RADIUS, height=ORBIT_HEIGHT,
                     target=None):
    """Position camera at given orbit angle, looking at target."""
    if target is None:
        target = ORBIT_TARGET
    angle_rad = math.radians(angle_deg)
    cam_obj.location = (
        target.x + radius * math.cos(angle_rad),
        target.y + radius * math.sin(angle_rad),
        height,
    )
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_render():
    """Set up EEVEE render settings for stills."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.eevee.taa_render_samples = SAMPLES
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15


def add_extra_lighting():
    """Add supplemental lighting to complement the existing scene light."""
    # Key light above the arm
    key = bpy.data.lights.new(name="ArmKeyLight", type="AREA")
    key.energy = 200
    key.size = 2.0
    key.color = (1.0, 0.95, 0.9)
    key_obj = bpy.data.objects.new("ArmKeyLight", key)
    bpy.context.scene.collection.objects.link(key_obj)
    key_obj.location = (ARM_POS[0] + 1.0, ARM_POS[1] - 1.0, ARM_POS[2] + 2.0)
    d = mathutils.Vector(ARM_POS) - key_obj.location
    key_obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    # Fill light from opposite side
    fill = bpy.data.lights.new(name="ArmFillLight", type="AREA")
    fill.energy = 80
    fill.size = 3.0
    fill.color = (0.9, 0.95, 1.0)
    fill_obj = bpy.data.objects.new("ArmFillLight", fill)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (ARM_POS[0] - 1.5, ARM_POS[1] + 1.5, ARM_POS[2] + 1.5)
    d = mathutils.Vector(ARM_POS) - fill_obj.location
    fill_obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def render_still(cam_obj, name, angle_deg, radius, height):
    """Render a single still frame at full resolution."""
    set_camera_orbit(cam_obj, angle_deg, radius, height)
    filepath = os.path.join(OUTPUT_DIR, f"{name}.png")
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered: {name}.png")
    return filepath


def render_orbit_frames(cam_obj):
    """Render orbit video frames at 720p for speed."""
    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.eevee.taa_render_samples = 32

    for i in range(VIDEO_FRAMES):
        angle = (i / VIDEO_FRAMES) * 360.0
        set_camera_orbit(cam_obj, angle)
        filepath = os.path.join(frames_dir, f"frame_{i:04d}.png")
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i+1}/{VIDEO_FRAMES}")

    # Restore full resolution
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.eevee.taa_render_samples = SAMPLES


def stitch_video():
    """Use ffmpeg to combine frames into MP4."""
    frames_pattern = os.path.join(OUTPUT_DIR, "frames", "frame_%04d.png")
    video_path = os.path.join(OUTPUT_DIR, "orbit_video.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(VIDEO_FPS),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        video_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Video saved: {video_path}")
    return video_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the house blend file
    print("Loading house.blend...")
    bpy.ops.wm.open_mainfile(filepath=BLEND_FILE)
    print(f"  {len(bpy.data.objects)} objects, {len(bpy.data.materials)} materials")

    # 2. Import arm and place directly at world position
    print("Importing and placing arm...")
    arm_objects = build_and_place_arm()
    print(f"  Placed {len(arm_objects)} arm parts")

    # 3. Add supplemental lighting
    print("Adding lighting...")
    add_extra_lighting()

    # 4. Set up camera and render settings
    print("Configuring render...")
    cam_obj = setup_camera()
    configure_render()

    # 5. Render still views
    print("\n=== Stills (1920x1080, 64 spp) ===")
    still_views = [
        ("view_front",       -30,  1.2, 1.3),
        ("view_side",        90,   1.3, 1.2),
        ("view_top",         0,    0.8, 2.4),
        ("view_perspective", 210,  1.5, 1.6),
        ("view_closeup",     150,  0.5, 1.0),  # close-up of the arm
    ]
    for name, angle, radius, height in still_views:
        render_still(cam_obj, name, angle, radius, height)

    # 6. Render orbit video frames
    print(f"\n=== Orbit Video ({VIDEO_FRAMES} frames, 1280x720, 32 spp) ===")
    render_orbit_frames(cam_obj)

    # 7. Stitch frames into MP4
    print("\n=== Stitching Video ===")
    video_path = stitch_video()

    print("\n=== Done ===")
    print(f"Stills:  {OUTPUT_DIR}/view_*.png")
    print(f"Video:   {video_path}")


if __name__ == "__main__":
    main()
