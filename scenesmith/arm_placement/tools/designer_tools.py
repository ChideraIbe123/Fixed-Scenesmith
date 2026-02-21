"""Designer tool factories for arm placement.

Provides tools for the designer agent to observe the scene, analyze surfaces,
place the robot arm, and adjust its position. Follows the closure-based tool
factory pattern used throughout SceneSmith.
"""

import json
import logging
import math
import shutil
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from agents import FunctionTool, ToolOutputImage, ToolOutputText, function_tool

from scenesmith.arm_placement.mujoco_renderer import render_scene
from scenesmith.arm_placement.scene_analyzer import SceneDescription
from scenesmith.utils.openai import encode_image_to_base64

console_logger = logging.getLogger(__name__)

# Compact upright home position for the arm joints.
# This folds the forearm upward so the arm has a small horizontal footprint.
# Order: joint1, joint2, joint3, joint4, joint5, gripper_joint_1, gripper_joint_2
ARM_HOME_QPOS = [0.0, 0.0, -1.5, 0.7, 0.0, 0.0, 0.0]
ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5",
    "gripper_joint_1", "gripper_joint_2",
]


def _get_arm_body_ids(model: mujoco.MjModel) -> set[int]:
    """Get the set of body IDs that belong to the robot arm."""
    arm_ids: set[int] = set()
    # Find the arm root body (omx_f_base).
    root_id = None
    for i in range(model.nbody):
        if model.body(i).name == "omx_f_base":
            root_id = i
            break
    if root_id is None:
        return arm_ids
    # BFS to collect all descendant bodies.
    queue = [root_id]
    while queue:
        bid = queue.pop(0)
        arm_ids.add(bid)
        for i in range(model.nbody):
            if model.body_parentid[i] == bid and i not in arm_ids:
                queue.append(i)
    return arm_ids


def _set_arm_home_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set arm joints to the compact home position."""
    for jname, qval in zip(ARM_JOINT_NAMES, ARM_HOME_QPOS):
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                qadr = model.jnt_qposadr[jid]
                data.qpos[qadr] = qval
        except Exception:
            pass


class DesignerTools:
    """Tool factory for arm placement designer agent.

    Creates closure-based tools that capture scene state and provide
    the designer with capabilities to observe, analyze, and modify
    the arm placement.
    """

    def __init__(
        self,
        scene_xml_path: Path,
        arm_mjcf_path: Path,
        output_dir: Path,
        scene_description: SceneDescription,
    ):
        self.scene_xml_path = scene_xml_path
        self.arm_mjcf_path = arm_mjcf_path
        self.output_dir = output_dir
        self.scene_description = scene_description
        self.render_count = 0
        self.arm_placed = False
        self.arm_position: list[float] | None = None
        self.arm_rotation: float = 0.0
        self.arm_surface: str | None = None
        self.current_scene_xml: Path = scene_xml_path
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create tool closures capturing self for state access."""

        @function_tool
        async def observe_scene() -> list[ToolOutputImage | ToolOutputText]:
            """Take visual snapshots of the current scene from multiple angles.

            After calling this, you'll see images of the scene from front, top,
            side, and perspective views. Use this to understand the room layout
            and evaluate arm placement.

            Returns:
                Images of the scene from multiple viewpoints.
            """
            return self._observe_scene_impl()

        @function_tool
        def get_scene_info() -> str:
            """Get a text description of the room, furniture, and available surfaces.

            Returns:
                Structured text description of the scene contents.
            """
            return self._get_scene_info_impl()

        @function_tool
        def list_placement_surfaces() -> str:
            """List all candidate surfaces where the arm could be placed.

            Returns surfaces with their height, dimensions, nearby objects,
            and clearance measurements. Use this to choose where to place the arm.

            Returns:
                Formatted list of available surfaces.
            """
            return self._list_placement_surfaces_impl()

        @function_tool
        def place_arm(
            surface_name: str,
            x_offset: float = 0.0,
            y_offset: float = 0.0,
            z_rotation_deg: float = 0.0,
        ) -> str:
            """Place the robot arm on a specified surface.

            The arm is placed at the surface center plus the given offsets.
            The z_rotation_deg controls the arm's facing direction.

            Args:
                surface_name: Name of the surface body to place the arm on.
                x_offset: X offset from surface center in meters (-0.3 to 0.3).
                y_offset: Y offset from surface center in meters (-0.3 to 0.3).
                z_rotation_deg: Rotation around Z axis in degrees (0-360).

            Returns:
                Success/failure message with placement details.
            """
            return self._place_arm_impl(
                surface_name, x_offset, y_offset, z_rotation_deg
            )

        @function_tool
        def adjust_arm_position(
            dx: float = 0.0,
            dy: float = 0.0,
            dz_rotation_deg: float = 0.0,
        ) -> str:
            """Fine-tune the current arm placement.

            Adjusts the arm position relative to its current location.

            Args:
                dx: X adjustment in meters.
                dy: Y adjustment in meters.
                dz_rotation_deg: Rotation adjustment in degrees.

            Returns:
                Updated placement details.
            """
            return self._adjust_arm_position_impl(dx, dy, dz_rotation_deg)

        @function_tool
        def get_arm_placement_status() -> str:
            """Get the current arm placement status.

            Returns position, orientation, surface, and nearby objects.

            Returns:
                Current arm placement details or 'not placed' message.
            """
            return self._get_arm_placement_status_impl()

        return {
            "observe_scene": observe_scene,
            "get_scene_info": get_scene_info,
            "list_placement_surfaces": list_placement_surfaces,
            "place_arm": place_arm,
            "adjust_arm_position": adjust_arm_position,
            "get_arm_placement_status": get_arm_placement_status,
        }

    def _observe_scene_impl(self) -> list[ToolOutputImage | ToolOutputText]:
        """Render scene and return images."""
        console_logger.info("Tool called: observe_scene (designer)")

        self.render_count += 1
        render_dir = self.output_dir / "renders" / f"designer_{self.render_count:03d}"

        image_paths = render_scene(
            scene_xml_path=self.current_scene_xml,
            output_dir=render_dir,
        )

        if not image_paths:
            return [
                ToolOutputText(
                    text="Unable to observe scene - rendering failed."
                )
            ]

        outputs: list[ToolOutputImage | ToolOutputText] = []
        for img_path in image_paths:
            img_base64 = encode_image_to_base64(img_path)
            outputs.append(
                ToolOutputImage(image_url=f"data:image/png;base64,{img_base64}")
            )

        status = "with arm" if self.arm_placed else "without arm"
        outputs.append(
            ToolOutputText(
                text=f"Scene observed from {len(image_paths)} viewpoints ({status}). "
                "Visual feedback is now available for analysis."
            )
        )

        console_logger.info(
            f"Returning {len(image_paths)} images via ToolOutputImage"
        )
        return outputs

    def _get_scene_info_impl(self) -> str:
        """Return scene description text."""
        console_logger.info("Tool called: get_scene_info (designer)")
        return self.scene_description.to_text()

    def _list_placement_surfaces_impl(self) -> str:
        """List available placement surfaces."""
        console_logger.info("Tool called: list_placement_surfaces")

        surfaces = self.scene_description.surfaces
        if not surfaces:
            return "No suitable placement surfaces found in the scene."

        lines = [f"Found {len(surfaces)} candidate surfaces:\n"]
        for i, s in enumerate(surfaces, 1):
            lines.append(f"{i}. {s.to_text()}")
        lines.append(
            "\nArm base footprint: ~0.15m x 0.15m. "
            "Arm reach radius: ~0.4m. "
            "Prefer surfaces at counter height (~0.85m) with good clearance."
        )
        return "\n".join(lines)

    def _place_arm_impl(
        self,
        surface_name: str,
        x_offset: float,
        y_offset: float,
        z_rotation_deg: float,
    ) -> str:
        """Place the arm on the specified surface with ground-clamping."""
        console_logger.info(f"Tool called: place_arm on '{surface_name}'")

        # Find the surface.
        surface = None
        for s in self.scene_description.surfaces:
            if s.body_name == surface_name or s.name == surface_name:
                surface = s
                break

        if surface is None:
            available = [s.name for s in self.scene_description.surfaces]
            return (
                f"ERROR: Surface '{surface_name}' not found. "
                f"Available surfaces: {available}"
            )

        # Compute arm placement position.
        # The arm's base bottom is at Z=0 in its local frame, so placing
        # the wrapper body at surface.height puts the base flush on the surface.
        arm_x = surface.position[0] + x_offset
        arm_y = surface.position[1] + y_offset
        arm_z = surface.height
        z_rotation_rad = math.radians(z_rotation_deg)

        try:
            merged_path = self._merge_arm_into_scene(
                arm_x, arm_y, arm_z, z_rotation_rad
            )
        except Exception as e:
            console_logger.error(f"Failed to place arm: {e}")
            return f"ERROR: Failed to place arm: {e}"

        # Validate simulation stability (with arm in home position).
        try:
            model = mujoco.MjModel.from_xml_path(str(merged_path))
            data = mujoco.MjData(model)
            _set_arm_home_qpos(model, data)
            mujoco.mj_forward(model, data)

            for _ in range(10):
                mujoco.mj_step(model, data)

            if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                return "ERROR: Arm placement caused simulation instability (NaN)."

        except Exception as e:
            return f"ERROR: Merged scene failed validation: {e}"

        # Check collisions (with arm in home position).
        surface_contacts, object_collisions = self._check_collisions(
            merged_path, surface_name
        )

        # Update state.
        self.arm_placed = True
        self.arm_position = [arm_x, arm_y, arm_z]
        self.arm_rotation = z_rotation_deg
        self.arm_surface = surface_name
        self.current_scene_xml = merged_path

        # Build result message.
        msg = (
            f"Arm placed successfully on '{surface_name}'.\n"
            f"Position: ({arm_x:.3f}, {arm_y:.3f}, {arm_z:.3f})\n"
            f"Rotation: {z_rotation_deg:.1f} degrees\n"
            f"Surface height: {surface.height:.3f}m\n"
            f"Surface dimensions: {surface.dimensions[0]:.3f} x "
            f"{surface.dimensions[1]:.3f}m\n"
            f"Nearby objects: {surface.nearby_objects}\n"
        )

        msg += f"Grounded: YES (base placed at surface height {surface.height:.3f}m)\n"

        if object_collisions:
            msg += (
                f"COLLISION WARNING: Arm is colliding with: {object_collisions}\n"
                "You MUST adjust the arm position to avoid these collisions.\n"
            )
        else:
            msg += "Collisions: None (arm is clear of all objects)\n"

        msg += f"Scene saved to: {merged_path}"
        return msg

    def _adjust_arm_position_impl(
        self, dx: float, dy: float, dz_rotation_deg: float
    ) -> str:
        """Adjust the current arm position with collision checking."""
        console_logger.info("Tool called: adjust_arm_position")

        if not self.arm_placed or self.arm_position is None:
            return "ERROR: No arm has been placed yet. Use place_arm() first."

        new_x = self.arm_position[0] + dx
        new_y = self.arm_position[1] + dy
        new_z = self.arm_position[2]
        new_rotation = self.arm_rotation + dz_rotation_deg
        new_rotation_rad = math.radians(new_rotation)

        try:
            merged_path = self._merge_arm_into_scene(
                new_x, new_y, new_z, new_rotation_rad
            )
        except Exception as e:
            return f"ERROR: Failed to adjust arm: {e}"

        # Validate (with arm in home position).
        try:
            model = mujoco.MjModel.from_xml_path(str(merged_path))
            data = mujoco.MjData(model)
            _set_arm_home_qpos(model, data)
            mujoco.mj_forward(model, data)
            for _ in range(10):
                mujoco.mj_step(model, data)
            if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                return "ERROR: Adjustment caused simulation instability."
        except Exception as e:
            return f"ERROR: Adjusted scene failed validation: {e}"

        # Check collisions.
        surface_contacts, object_collisions = self._check_collisions(
            merged_path, self.arm_surface or ""
        )

        self.arm_position = [new_x, new_y, new_z]
        self.arm_rotation = new_rotation
        self.current_scene_xml = merged_path

        msg = (
            f"Arm position adjusted.\n"
            f"New position: ({new_x:.3f}, {new_y:.3f}, {new_z:.3f})\n"
            f"New rotation: {new_rotation:.1f} degrees\n"
            f"Adjustments applied: dx={dx:.3f}, dy={dy:.3f}, "
            f"dz_rotation={dz_rotation_deg:.1f}\n"
        )

        if object_collisions:
            msg += (
                f"COLLISION WARNING: Arm is colliding with: {object_collisions}\n"
                "You MUST adjust the arm position to avoid these collisions.\n"
            )
        else:
            msg += "Collisions: None (arm is clear of all objects)\n"

        return msg

    def _get_arm_placement_status_impl(self) -> str:
        """Return current arm status."""
        console_logger.info("Tool called: get_arm_placement_status")

        if not self.arm_placed:
            return "No arm has been placed yet."

        # Find nearby objects.
        nearby = []
        if self.arm_position is not None:
            arm_xy = np.array(self.arm_position[:2])
            for f in self.scene_description.furniture:
                f_xy = np.array(f.position[:2])
                dist = np.linalg.norm(arm_xy - f_xy)
                if dist < 0.8:  # Within ~2x reach radius.
                    nearby.append(f"{f.name} (dist={dist:.2f}m)")

        return (
            f"Arm placement status:\n"
            f"  Surface: {self.arm_surface}\n"
            f"  Position: ({self.arm_position[0]:.3f}, "
            f"{self.arm_position[1]:.3f}, {self.arm_position[2]:.3f})\n"
            f"  Rotation: {self.arm_rotation:.1f} degrees\n"
            f"  Reach radius: ~0.4m\n"
            f"  Nearby objects within reach: {nearby}"
        )

    def _check_collisions(
        self, merged_path: Path, surface_body_name: str
    ) -> tuple[list[str], list[str]]:
        """Check for arm collisions with scene objects.

        Uses the arm home position and proper body ID detection (not
        substring matching) to avoid false positives with scene bodies
        whose names contain 'link'.

        Returns two lists:
        - surface_contacts: contacts with the placement surface (expected/good)
        - object_collisions: contacts with other objects (bad/penetration)
        """
        surface_contacts: list[str] = []
        object_collisions: list[str] = []

        try:
            model = mujoco.MjModel.from_xml_path(str(merged_path))
            data = mujoco.MjData(model)
            _set_arm_home_qpos(model, data)
            mujoco.mj_forward(model, data)

            arm_body_ids = _get_arm_body_ids(model)

            for c in range(data.ncon):
                contact = data.contact[c]
                geom1_body = model.geom_bodyid[contact.geom1]
                geom2_body = model.geom_bodyid[contact.geom2]

                is_arm1 = geom1_body in arm_body_ids
                is_arm2 = geom2_body in arm_body_ids

                if not (is_arm1 or is_arm2):
                    continue

                # Skip arm self-collisions.
                if is_arm1 and is_arm2:
                    continue

                other_body = geom2_body if is_arm1 else geom1_body
                other_name = model.body(other_body).name

                # Check if contact is with the placement surface.
                if other_name.lower() == surface_body_name.lower():
                    if other_name not in surface_contacts:
                        surface_contacts.append(other_name)
                else:
                    if other_name not in object_collisions:
                        object_collisions.append(other_name)

        except Exception as e:
            console_logger.warning(f"Collision check failed: {e}")

        return surface_contacts, object_collisions

    def _merge_arm_into_scene(
        self, x: float, y: float, z: float, z_rotation_rad: float
    ) -> Path:
        """Merge the arm MJCF into the scene XML at the specified position.

        Creates a new XML file that includes the arm at the given transform.
        Handles mesh path resolution by copying arm meshes into the scene's
        mesh directory and using the scene's existing meshdir.

        Args:
            x: X position in world frame.
            y: Y position in world frame.
            z: Z position in world frame.
            z_rotation_rad: Rotation around Z axis in radians.

        Returns:
            Path to the merged scene XML.
        """
        # Load the original scene XML (always start from base scene).
        scene_tree = ET.parse(self.scene_xml_path)
        scene_root = scene_tree.getroot()

        # Load the arm XML.
        arm_tree = ET.parse(self.arm_mjcf_path)
        arm_root = arm_tree.getroot()

        # Determine the scene's meshdir from compiler element.
        scene_compiler = scene_root.find("compiler")
        scene_dir = self.scene_xml_path.parent
        if scene_compiler is not None:
            meshdir_rel = scene_compiler.get("meshdir", ".")
            scene_meshdir = scene_dir / meshdir_rel
        else:
            scene_meshdir = scene_dir

        # Copy arm meshes into the scene's mesh directory.
        arm_dir = self.arm_mjcf_path.parent
        arm_meshes_src = arm_dir / "arm_meshes"
        if arm_meshes_src.exists():
            for stl_file in arm_meshes_src.iterdir():
                dst = scene_meshdir / stl_file.name
                if not dst.exists():
                    shutil.copy2(stl_file, dst)

        # Find or create worldbody in scene.
        scene_worldbody = scene_root.find("worldbody")
        if scene_worldbody is None:
            scene_worldbody = ET.SubElement(scene_root, "worldbody")

        # Get arm's worldbody children (the arm body tree).
        arm_worldbody = arm_root.find("worldbody")
        if arm_worldbody is None:
            raise ValueError("Arm MJCF has no worldbody")

        # Compute quaternion for Z-rotation: [cos(θ/2), 0, 0, sin(θ/2)].
        qw = math.cos(z_rotation_rad / 2)
        qz = math.sin(z_rotation_rad / 2)

        # Create a wrapper body for the arm at the placement position.
        arm_wrapper = ET.SubElement(scene_worldbody, "body")
        arm_wrapper.set("name", "omx_f_base")
        arm_wrapper.set("pos", f"{x:.6f} {y:.6f} {z:.6f}")
        arm_wrapper.set("quat", f"{qw:.6f} 0 0 {qz:.6f}")

        # Copy arm bodies into the wrapper.
        # The arm's <default> block (which sets type="mesh") doesn't get copied,
        # so we must explicitly set type="mesh" on all arm geoms that reference
        # a mesh but lack an explicit type attribute.
        for child in list(arm_worldbody):
            arm_wrapper.append(child)
        for geom in arm_wrapper.iter("geom"):
            if geom.get("mesh") and not geom.get("type"):
                geom.set("type", "mesh")

        # Copy arm meshes to scene asset section.
        # Update mesh file paths to be relative to scene's meshdir
        # (arm meshes were copied into scene meshdir above).
        scene_asset = scene_root.find("asset")
        if scene_asset is None:
            scene_asset = ET.SubElement(scene_root, "asset")

        arm_asset = arm_root.find("asset")
        if arm_asset is not None:
            for mesh in arm_asset.findall("mesh"):
                # Update file path: arm_meshes/foo.stl -> foo.stl
                # (since we copied files directly into scene meshdir).
                file_attr = mesh.get("file", "")
                if "/" in file_attr:
                    mesh.set("file", file_attr.split("/")[-1])
                scene_asset.append(mesh)

        # Copy equality constraints (mimic joints).
        arm_equality = arm_root.find("equality")
        if arm_equality is not None:
            scene_equality = scene_root.find("equality")
            if scene_equality is None:
                scene_equality = ET.SubElement(scene_root, "equality")
            for child in arm_equality:
                scene_equality.append(child)

        # Add keyframe with the compact home position for the arm.
        # Build a full qpos vector: all zeros for scene joints, then arm home.
        # We write the joint names and values so it works regardless of order.
        keyframe_el = scene_root.find("keyframe")
        if keyframe_el is None:
            keyframe_el = ET.SubElement(scene_root, "keyframe")
        key_el = ET.SubElement(keyframe_el, "key")
        key_el.set("name", "home")
        # Build qpos string: count total joints from scene + arm.
        # Load temporarily to find joint count and arm joint addresses.
        output_path = scene_dir / "scene_with_arm.xml"
        scene_tree.write(str(output_path), xml_declaration=True)
        try:
            tmp_model = mujoco.MjModel.from_xml_path(str(output_path))
            qpos = np.zeros(tmp_model.nq)
            for jname, qval in zip(ARM_JOINT_NAMES, ARM_HOME_QPOS):
                jid = mujoco.mj_name2id(
                    tmp_model, mujoco.mjtObj.mjOBJ_JOINT, jname
                )
                if jid >= 0:
                    qpos[tmp_model.jnt_qposadr[jid]] = qval
            key_el.set("qpos", " ".join(f"{v:.6f}" for v in qpos))
            # Re-write with keyframe populated.
            scene_tree.write(str(output_path), xml_declaration=True)
        except Exception as e:
            console_logger.warning(f"Could not add keyframe: {e}")

        console_logger.info(f"Merged scene saved to: {output_path}")
        return output_path

    def get_current_scene_xml(self) -> str:
        """Get the current scene XML as a string (for checkpointing).

        Returns:
            XML content string.
        """
        with open(self.current_scene_xml) as f:
            return f.read()

    def restore_scene_xml(self, xml_content: str) -> None:
        """Restore scene from an XML string (for checkpoint reset).

        Args:
            xml_content: XML content to restore.
        """
        output_path = self.output_dir / "scene_with_arm.xml"
        with open(output_path, "w") as f:
            f.write(xml_content)
        self.current_scene_xml = output_path


def create_designer_tools(
    scene_xml_path: Path,
    arm_mjcf_path: Path,
    output_dir: Path,
    scene_description: SceneDescription,
) -> tuple[list[FunctionTool], "DesignerTools"]:
    """Create designer tools for arm placement.

    Args:
        scene_xml_path: Path to the base scene XML.
        arm_mjcf_path: Path to the arm MJCF file.
        output_dir: Working output directory.
        scene_description: Analyzed scene description.

    Returns:
        Tuple of (list of FunctionTools, DesignerTools instance for state access).
    """
    tools_obj = DesignerTools(
        scene_xml_path=scene_xml_path,
        arm_mjcf_path=arm_mjcf_path,
        output_dir=output_dir,
        scene_description=scene_description,
    )
    return list(tools_obj.tools.values()), tools_obj
