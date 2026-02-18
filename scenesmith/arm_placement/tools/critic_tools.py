"""Critic tool factories for arm placement evaluation.

Provides read-only tools for the critic agent to observe the scene,
inspect furniture, and evaluate the arm placement quality.
"""

import logging

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from agents import FunctionTool, ToolOutputImage, ToolOutputText, function_tool

from scenesmith.arm_placement.mujoco_renderer import render_scene
from scenesmith.arm_placement.scene_analyzer import SceneDescription
from scenesmith.utils.openai import encode_image_to_base64

console_logger = logging.getLogger(__name__)


class CriticTools:
    """Tool factory for arm placement critic agent.

    Creates read-only tools for scene observation and arm placement evaluation.
    """

    def __init__(
        self,
        output_dir: Path,
        scene_description: SceneDescription,
        get_current_scene_path: Any,  # Callable returning Path.
        get_arm_state: Any,  # Callable returning dict with arm placement info.
    ):
        self.output_dir = output_dir
        self.scene_description = scene_description
        self._get_current_scene_path = get_current_scene_path
        self._get_arm_state = get_arm_state
        self.render_count = 0
        self.last_render_dir: Path | None = None
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create read-only tool closures for critic evaluation."""

        @function_tool
        async def observe_scene() -> list[ToolOutputImage | ToolOutputText]:
            """Take visual snapshots of the current scene from multiple angles.

            You MUST call this first before evaluating the arm placement.
            After calling, you'll see images showing the room and arm from
            front, top, side, and perspective views.

            Returns:
                Images of the scene from multiple viewpoints.
            """
            return self._observe_scene_impl()

        @function_tool
        def get_scene_info() -> str:
            """Get a text description of the room and furniture layout.

            Returns:
                Structured text with room type, dimensions, and furniture list.
            """
            return self._get_scene_info_impl()

        @function_tool
        def get_arm_placement_details() -> str:
            """Get detailed information about the current arm placement.

            Returns the arm's position, orientation, surface it's on,
            what objects are within reach, clearance measurements, and
            collision status.

            Returns:
                Detailed arm placement analysis text.
            """
            return self._get_arm_placement_details_impl()

        return {
            "observe_scene": observe_scene,
            "get_scene_info": get_scene_info,
            "get_arm_placement_details": get_arm_placement_details,
        }

    def _observe_scene_impl(self) -> list[ToolOutputImage | ToolOutputText]:
        """Render scene and return images."""
        console_logger.info("Tool called: observe_scene (critic)")

        scene_path = self._get_current_scene_path()
        self.render_count += 1
        render_dir = self.output_dir / "renders" / f"critic_{self.render_count:03d}"
        self.last_render_dir = render_dir

        image_paths = render_scene(
            scene_xml_path=scene_path,
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

        arm_state = self._get_arm_state()
        status = "with arm placed" if arm_state.get("placed") else "without arm"

        outputs.append(
            ToolOutputText(
                text=f"Scene observed from {len(image_paths)} viewpoints ({status}). "
                "Evaluate the arm placement based on these views."
            )
        )

        console_logger.info(
            f"Returning {len(image_paths)} critic images via ToolOutputImage"
        )
        return outputs

    def _get_scene_info_impl(self) -> str:
        """Return scene description text."""
        console_logger.info("Tool called: get_scene_info (critic)")
        return self.scene_description.to_text()

    def _get_arm_placement_details_impl(self) -> str:
        """Return detailed arm placement analysis."""
        console_logger.info("Tool called: get_arm_placement_details")

        arm_state = self._get_arm_state()
        if not arm_state.get("placed"):
            return "No arm has been placed in the scene yet."

        pos = arm_state["position"]
        rotation = arm_state["rotation"]
        surface = arm_state["surface"]

        # Find objects within reach.
        arm_xy = np.array(pos[:2])
        within_reach = []
        nearby_but_far = []
        for f in self.scene_description.furniture:
            f_xy = np.array(f.position[:2])
            dist = float(np.linalg.norm(arm_xy - f_xy))
            if dist < 0.4:
                within_reach.append(f"{f.name} (dist={dist:.2f}m)")
            elif dist < 0.8:
                nearby_but_far.append(f"{f.name} (dist={dist:.2f}m)")

        # Check collision status by loading the merged scene.
        collision_info = "unknown"
        try:
            scene_path = self._get_current_scene_path()
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            # Check contacts involving arm bodies.
            arm_contacts = []
            for c in range(data.ncon):
                contact = data.contact[c]
                geom1_body = model.geom_bodyid[contact.geom1]
                geom2_body = model.geom_bodyid[contact.geom2]
                body1_name = model.body(geom1_body).name
                body2_name = model.body(geom2_body).name

                # Check if either body is part of the arm.
                is_arm1 = "link" in body1_name.lower() or "omx" in body1_name.lower()
                is_arm2 = "link" in body2_name.lower() or "omx" in body2_name.lower()
                if is_arm1 or is_arm2:
                    other = body2_name if is_arm1 else body1_name
                    if other not in arm_contacts:
                        arm_contacts.append(other)

            if arm_contacts:
                collision_info = f"Contacts detected with: {arm_contacts}"
            else:
                collision_info = "No collisions detected"

        except Exception as e:
            collision_info = f"Could not check collisions: {e}"

        # Find the surface info.
        surface_info = None
        for s in self.scene_description.surfaces:
            if s.body_name == surface or s.name == surface:
                surface_info = s
                break

        surface_details = ""
        if surface_info:
            surface_details = (
                f"\n  Surface details:\n"
                f"    Height: {surface_info.height:.3f}m\n"
                f"    Dimensions: {surface_info.dimensions[0]:.3f} x "
                f"{surface_info.dimensions[1]:.3f}m\n"
                f"    Clearance: {surface_info.clearance:.3f}m"
            )

        return (
            f"Arm Placement Details:\n"
            f"  Surface: {surface}\n"
            f"  Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
            f"  Rotation: {rotation:.1f} degrees\n"
            f"  Reach radius: ~0.4m\n"
            f"{surface_details}\n"
            f"\n  Objects within reach: {within_reach or 'none'}\n"
            f"  Objects nearby (outside reach): {nearby_but_far or 'none'}\n"
            f"\n  Collision status: {collision_info}"
        )


def create_critic_tools(
    output_dir: Path,
    scene_description: SceneDescription,
    get_current_scene_path: Any,
    get_arm_state: Any,
) -> tuple[list[FunctionTool], "CriticTools"]:
    """Create critic tools for arm placement evaluation.

    Args:
        output_dir: Working output directory.
        scene_description: Analyzed scene description.
        get_current_scene_path: Callable returning current scene XML path.
        get_arm_state: Callable returning dict with arm placement state.

    Returns:
        Tuple of (list of FunctionTools, CriticTools instance).
    """
    tools_obj = CriticTools(
        output_dir=output_dir,
        scene_description=scene_description,
        get_current_scene_path=get_current_scene_path,
        get_arm_state=get_arm_state,
    )
    return list(tools_obj.tools.values()), tools_obj
