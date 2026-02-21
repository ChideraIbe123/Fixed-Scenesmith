#!/usr/bin/env python3
"""
Render animated VLA episode in Blender from saved trajectory.

Loads house.blend, imports OMX-F arm STL meshes, replays joint trajectory
from a VLA control episode, and renders each frame with EEVEE. Optionally
adds a cup mesh if cup_pose data is present in the trajectory.

Usage (run inside Blender):
    blender --background --python scripts/render_blender_vla.py -- \
        --trajectory renders/vla_episode/trajectory.npz \
        --blend-file outputs/.../house.blend \
        --output-dir renders/blender_vla/ \
        --resolution 1280 720 \
        --samples 32 \
        --camera-angle 210 \
        --camera-radius 1.5 \
        --camera-height 1.6 \
        --fps 15
"""

import argparse
import math
import os
import subprocess
import sys

import bpy
import mathutils
import numpy as np

# ─── Arm Constants (shared with render_blender_scene.py) ─────────────────────

STL_DIR = "/workspace/scenesmith/OMX-Files-/open_manipulator_description/meshes/omx_f"

# Arm base placement defaults (overridden by trajectory data or CLI args).
ARM_POS = (2.179693, 2.620423, 0.788227)
ARM_QUAT_WXYZ = (0.382683, 0.0, 0.0, 0.923880)  # 135° around Z

# Will be computed at runtime from actual arm pose.
ARM_BASE_MATRIX = None

# URDF kinematic chain: (stl_filename, joint_offset_xyz, joint_axis, default_angle)
# All STLs are in millimeters, scale 0.001.
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

STL_SCALE = 0.001  # mm → meters

ORBIT_TARGET = None  # Computed at runtime from ARM_POS.


def init_arm_globals(pos, quat_wxyz):
    """Set ARM_POS, ARM_QUAT_WXYZ, ARM_BASE_MATRIX, ORBIT_TARGET from values."""
    global ARM_POS, ARM_QUAT_WXYZ, ARM_BASE_MATRIX, ORBIT_TARGET
    ARM_POS = tuple(pos)
    ARM_QUAT_WXYZ = tuple(quat_wxyz)
    ARM_BASE_MATRIX = mathutils.Matrix.LocRotScale(
        mathutils.Vector(ARM_POS),
        mathutils.Quaternion(ARM_QUAT_WXYZ),
        mathutils.Vector((1, 1, 1)),
    )
    ORBIT_TARGET = mathutils.Vector(ARM_POS) + mathutils.Vector((0, 0, 0.15))
    print(f"  Arm base: pos={ARM_POS}, quat={ARM_QUAT_WXYZ}")


# ─── Helpers (reused from render_blender_scene.py) ───────────────────────────

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


def import_arm_parts():
    """Import arm STL meshes, apply scale, assign material.

    Returns list of Blender objects in kinematic chain order.
    Does NOT position them — that's done by pose_arm().
    """
    arm_material = create_arm_material()
    arm_objects = []

    for i, (stl_name, _offset, _axis, _angle) in enumerate(KINEMATIC_CHAIN):
        stl_path = os.path.join(STL_DIR, stl_name)
        if not os.path.exists(stl_path):
            print(f"  WARNING: STL not found: {stl_path}")
            arm_objects.append(None)
            continue

        obj = import_stl(stl_path)
        obj.name = f"arm_{i:02d}_{stl_name.replace('.stl', '')}"
        obj.scale = (STL_SCALE, STL_SCALE, STL_SCALE)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(scale=True)
        obj.select_set(False)

        if obj.data.materials:
            obj.data.materials[0] = arm_material
        else:
            obj.data.materials.append(arm_material)

        arm_objects.append(obj)

    return arm_objects


def pose_arm(arm_objects, joint_angles, gripper_angle):
    """Reposition arm parts for given joint angles.

    Args:
        arm_objects: List of 8 Blender objects from import_arm_parts().
        joint_angles: Array of 5 arm joint angles [j1, j2, j3, j4, j5].
        gripper_angle: Scalar gripper_joint_1 angle (joint_2 = -joint_1).
    """
    # Map joints to chain indices:
    #   0: base (fixed), 1: j1(Z), 2: j2(Y), 3: j3(Y),
    #   4: j4(Y), 5: j5(X), 6: gripper1(no axis), 7: gripper2(no axis)
    angles = [
        0.0,
        joint_angles[0],
        joint_angles[1],
        joint_angles[2],
        joint_angles[3],
        joint_angles[4],
        gripper_angle,
        -gripper_angle,
    ]

    cumulative = mathutils.Matrix.Identity(4)
    for i, (_, offset, axis, _) in enumerate(KINEMATIC_CHAIN):
        joint_mat = mathutils.Matrix.Translation(offset)
        if axis:
            axis_vec = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis]
            joint_mat = joint_mat @ mathutils.Matrix.Rotation(angles[i], 4, axis_vec)
        cumulative = cumulative @ joint_mat

        if arm_objects[i] is not None:
            arm_objects[i].matrix_world = ARM_BASE_MATRIX @ cumulative


def create_cup_mesh():
    """Create a simple red cylinder matching the MuJoCo test cup geom.

    Radius=0.025m, half-height=0.035m → full height=0.07m.
    Returns the Blender object.
    """
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.025,
        depth=0.07,
        location=(0, 0, 0),
    )
    cup = bpy.context.active_object
    cup.name = "test_cup"

    # Red material.
    mat = bpy.data.materials.new(name="CupRed")
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.85, 0.15, 0.15, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    cup.data.materials.append(mat)
    return cup


def pose_cup(cup_obj, cup_pose):
    """Set cup position and orientation from freejoint DOFs.

    Args:
        cup_obj: Blender object for the cup.
        cup_pose: Array of 7 floats [x, y, z, qw, qx, qy, qz].
    """
    pos = mathutils.Vector(cup_pose[:3])
    quat = mathutils.Quaternion((cup_pose[3], cup_pose[4], cup_pose[5], cup_pose[6]))
    cup_obj.matrix_world = mathutils.Matrix.LocRotScale(
        pos, quat, mathutils.Vector((1, 1, 1)),
    )


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


def set_camera_orbit(cam_obj, angle_deg, radius, height, target=None):
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


def configure_render(resolution, samples):
    """Set up EEVEE render settings."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.eevee.taa_render_samples = samples
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15


def add_extra_lighting():
    """Add supplemental lighting to complement the existing scene light."""
    key = bpy.data.lights.new(name="ArmKeyLight", type="AREA")
    key.energy = 200
    key.size = 2.0
    key.color = (1.0, 0.95, 0.9)
    key_obj = bpy.data.objects.new("ArmKeyLight", key)
    bpy.context.scene.collection.objects.link(key_obj)
    key_obj.location = (ARM_POS[0] + 1.0, ARM_POS[1] - 1.0, ARM_POS[2] + 2.0)
    d = mathutils.Vector(ARM_POS) - key_obj.location
    key_obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    fill = bpy.data.lights.new(name="ArmFillLight", type="AREA")
    fill.energy = 80
    fill.size = 3.0
    fill.color = (0.9, 0.95, 1.0)
    fill_obj = bpy.data.objects.new("ArmFillLight", fill)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (ARM_POS[0] - 1.5, ARM_POS[1] + 1.5, ARM_POS[2] + 1.5)
    d = mathutils.Vector(ARM_POS) - fill_obj.location
    fill_obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def stitch_video(frames_dir, output_path, fps):
    """Use ffmpeg to combine frames into MP4."""
    frames_pattern = os.path.join(frames_dir, "frame_%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        output_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Video saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse CLI arguments after Blender's '--' separator.

    When run via `blender --background --python script.py -- --args`,
    our args come after '--'. When run directly with `python script.py --args`,
    use sys.argv[1:] as usual.
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Render animated VLA episode in Blender."
    )
    parser.add_argument(
        "--trajectory", required=True,
        help="Path to trajectory.npz from VLA episode.",
    )
    parser.add_argument(
        "--blend-file", required=True,
        help="Path to house.blend scene file.",
    )
    parser.add_argument(
        "--output-dir", default="renders/blender_vla",
        help="Output directory for frames and video.",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[1280, 720],
        metavar=("W", "H"),
        help="Render resolution (default: 1280 720).",
    )
    parser.add_argument(
        "--samples", type=int, default=32,
        help="EEVEE TAA samples (default: 32).",
    )
    parser.add_argument(
        "--camera-angle", type=float, default=210,
        help="Camera orbit angle in degrees (default: 210).",
    )
    parser.add_argument(
        "--camera-radius", type=float, default=1.5,
        help="Camera orbit radius (default: 1.5).",
    )
    parser.add_argument(
        "--camera-height", type=float, default=1.6,
        help="Camera height (default: 1.6).",
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Output video FPS (default: 15, matching VLA control freq).",
    )
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Render every Nth frame (default: 1 = all frames).",
    )
    parser.add_argument(
        "--arm-pos", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Arm base position (overrides trajectory/default).",
    )
    parser.add_argument(
        "--arm-quat", type=float, nargs=4, default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Arm base quaternion WXYZ (overrides trajectory/default).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # Load trajectory.
    print(f"Loading trajectory from {args.trajectory}...")
    traj = np.load(args.trajectory)
    joint_angles = traj["joint_angles"]      # (N, 5)
    gripper_angles = traj["gripper_angles"]  # (N,)
    has_cup = "cup_pose" in traj
    cup_poses = traj["cup_pose"] if has_cup else None
    traj_fps = float(traj["fps"]) if "fps" in traj else args.fps

    n_frames = len(joint_angles)
    print(f"  {n_frames} frames at {traj_fps} Hz")

    # Determine arm base pose: CLI args > trajectory data > defaults.
    if args.arm_pos is not None:
        arm_pos = args.arm_pos
    elif "arm_pos" in traj:
        arm_pos = traj["arm_pos"]
    else:
        arm_pos = ARM_POS

    if args.arm_quat is not None:
        arm_quat = args.arm_quat
    elif "arm_quat" in traj:
        arm_quat = traj["arm_quat"]
    else:
        arm_quat = ARM_QUAT_WXYZ

    init_arm_globals(arm_pos, arm_quat)
    if has_cup:
        print(f"  Cup trajectory included ({len(cup_poses)} frames)")

    # Determine which frames to render.
    frame_indices = list(range(0, n_frames, args.frame_step))
    print(f"  Rendering {len(frame_indices)} frames (step={args.frame_step})")

    # Setup output directories.
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 1. Load the house blend file.
    print(f"Loading {args.blend_file}...")
    bpy.ops.wm.open_mainfile(filepath=args.blend_file)
    print(f"  {len(bpy.data.objects)} objects, {len(bpy.data.materials)} materials")

    # 2. Import arm STL meshes.
    print("Importing arm parts...")
    arm_objects = import_arm_parts()
    print(f"  Imported {sum(1 for o in arm_objects if o is not None)} arm parts")

    # 3. Create cup mesh if trajectory has cup data.
    cup_obj = None
    if has_cup:
        print("Creating cup mesh...")
        cup_obj = create_cup_mesh()

    # 4. Add lighting.
    print("Adding lighting...")
    add_extra_lighting()

    # 5. Setup camera and render.
    print("Configuring render...")
    cam_obj = setup_camera()
    set_camera_orbit(
        cam_obj, args.camera_angle,
        args.camera_radius, args.camera_height,
    )
    configure_render(args.resolution, args.samples)

    # 6. Render each frame.
    print(f"\n=== Rendering {len(frame_indices)} frames "
          f"({args.resolution[0]}x{args.resolution[1]}, "
          f"{args.samples} spp) ===")

    for render_idx, frame_idx in enumerate(frame_indices):
        # Pose arm at this frame's joint angles.
        pose_arm(arm_objects, joint_angles[frame_idx], gripper_angles[frame_idx])

        # Pose cup if present.
        if cup_obj is not None and cup_poses is not None:
            pose_cup(cup_obj, cup_poses[frame_idx])

        # Render frame.
        filepath = os.path.join(frames_dir, f"frame_{render_idx:04d}.png")
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

        if (render_idx + 1) % 10 == 0 or render_idx == 0:
            print(f"  Frame {render_idx + 1}/{len(frame_indices)} "
                  f"(traj frame {frame_idx})")

    # 7. Stitch frames into MP4.
    output_fps = args.fps if args.frame_step == 1 else max(1, args.fps // args.frame_step)
    video_path = os.path.join(args.output_dir, "vla_episode.mp4")
    print(f"\n=== Stitching video at {output_fps} fps ===")
    stitch_video(frames_dir, video_path, output_fps)

    print(f"\n=== Done ===")
    print(f"Frames: {frames_dir}/ ({len(frame_indices)} PNGs)")
    print(f"Video:  {video_path}")


if __name__ == "__main__":
    main()
