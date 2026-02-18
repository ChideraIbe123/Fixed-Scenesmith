"""MuJoCo scene analysis for extracting furniture, surfaces, and scene structure.

Parses a MuJoCo XML scene file to identify placement surfaces, furniture items,
and room layout information for the arm placement agent.
"""

import logging

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

console_logger = logging.getLogger(__name__)

# Arm reach radius in meters (approximate for OMX_F).
ARM_REACH_RADIUS = 0.4


@dataclass
class FurnitureInfo:
    """Information about a piece of furniture in the scene."""

    name: str
    position: list[float]  # [x, y, z]
    size: list[float]  # approximate bounding box [x, y, z]
    body_name: str  # MuJoCo body name

    def to_text(self) -> str:
        """Format as text for agent consumption."""
        return (
            f"- {self.name}: position=({self.position[0]:.3f}, "
            f"{self.position[1]:.3f}, {self.position[2]:.3f}), "
            f"size=({self.size[0]:.3f}, {self.size[1]:.3f}, {self.size[2]:.3f})"
        )


@dataclass
class SurfaceInfo:
    """Information about a placeable surface in the scene."""

    name: str
    body_name: str  # MuJoCo body name
    position: list[float]  # center of surface [x, y, z]
    height: float  # z-height of the surface
    dimensions: list[float]  # [width, depth] in meters
    nearby_objects: list[str] = field(default_factory=list)
    clearance: float = 0.0  # approximate free space around edges

    def to_text(self) -> str:
        """Format as text for agent consumption."""
        nearby = ", ".join(self.nearby_objects) if self.nearby_objects else "none"
        return (
            f"- {self.name} (body: {self.body_name}): "
            f"position=({self.position[0]:.3f}, {self.position[1]:.3f}, "
            f"{self.position[2]:.3f}), height={self.height:.3f}m, "
            f"dimensions=({self.dimensions[0]:.3f} x {self.dimensions[1]:.3f})m, "
            f"nearby_objects=[{nearby}], clearance={self.clearance:.3f}m"
        )


@dataclass
class SceneDescription:
    """Complete description of a MuJoCo scene."""

    room_type: str
    room_dimensions: list[float]  # [width, depth, height]
    furniture: list[FurnitureInfo] = field(default_factory=list)
    surfaces: list[SurfaceInfo] = field(default_factory=list)

    def to_text(self) -> str:
        """Format complete scene description as text for agent prompts."""
        lines = [
            f"Room type: {self.room_type}",
            f"Room dimensions: {self.room_dimensions[0]:.1f} x "
            f"{self.room_dimensions[1]:.1f} x {self.room_dimensions[2]:.1f} meters",
            "",
            f"Furniture ({len(self.furniture)} items):",
        ]
        for f in self.furniture:
            lines.append(f.to_text())

        lines.append("")
        lines.append(f"Available surfaces ({len(self.surfaces)} found):")
        for s in self.surfaces:
            lines.append(s.to_text())

        return "\n".join(lines)


def _get_body_aabb(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple:
    """Compute approximate axis-aligned bounding box for a body.

    Uses the body's geoms to estimate the bounding box.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_id: Body ID.

    Returns:
        Tuple of (center [x,y,z], size [sx,sy,sz]).
    """
    geom_ids = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id:
            geom_ids.append(g)

    if not geom_ids:
        pos = data.xpos[body_id]
        return list(pos), [0.1, 0.1, 0.1]

    # Compute bounding box from all geom positions and sizes.
    mins = np.array([1e6, 1e6, 1e6])
    maxs = np.array([-1e6, -1e6, -1e6])

    for g in geom_ids:
        gpos = data.geom_xpos[g]
        gsize = model.geom_size[g]
        gtype = model.geom_type[g]

        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half = gsize[:3]
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.array([gsize[0]] * 3)
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = np.array([gsize[0], gsize[0], gsize[1]])
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            # Use mesh AABB from model.
            mesh_id = model.geom_dataid[g]
            if mesh_id >= 0 and mesh_id < model.nmesh:
                vert_start = model.mesh_vertadr[mesh_id]
                vert_count = model.mesh_vertnum[mesh_id]
                if vert_count > 0:
                    verts = model.mesh_vert[vert_start : vert_start + vert_count]
                    vert_min = np.min(verts, axis=0)
                    vert_max = np.max(verts, axis=0)
                    mins = np.minimum(mins, gpos + vert_min)
                    maxs = np.maximum(maxs, gpos + vert_max)
                    continue
                else:
                    half = np.array([0.1, 0.1, 0.1])
            else:
                half = np.array([0.1, 0.1, 0.1])
        else:
            half = np.array([0.1, 0.1, 0.1])

        mins = np.minimum(mins, gpos - half)
        maxs = np.maximum(maxs, gpos + half)

    center = (mins + maxs) / 2
    size = maxs - mins

    return list(center), list(np.maximum(size, 0.01))


def _get_collision_geom_top_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float | None:
    """Get the world-frame top Z of a single geom.

    Returns None for visual-only geoms (contype=0 and conaffinity=0).
    """
    if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
        return None

    gpos = data.geom_xpos[geom_id]
    gtype = model.geom_type[geom_id]

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id >= 0 and mesh_id < model.nmesh:
            vert_start = model.mesh_vertadr[mesh_id]
            vert_count = model.mesh_vertnum[mesh_id]
            if vert_count > 0:
                verts = model.mesh_vert[vert_start : vert_start + vert_count]
                return float(gpos[2] + np.max(verts[:, 2]))
        return float(gpos[2] + 0.1)
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(gpos[2] + model.geom_size[geom_id][2])
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(gpos[2] + model.geom_size[geom_id][0])
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(gpos[2] + model.geom_size[geom_id][1])
    else:
        return float(gpos[2] + 0.1)


def _is_horizontal_surface(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    min_height: float = 0.3,
    min_area: float = 0.04,
) -> tuple[bool, float, list[float]]:
    """Check if a body has a horizontal surface suitable for arm placement.

    Uses collision geom top-Z percentiles to find the actual flat surface
    height, which is more accurate than the full AABB top (which includes
    visual meshes and decorative elements above the surface).

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_id: Body ID to check.
        min_height: Minimum height above ground for a valid surface.
        min_area: Minimum surface area in m^2.

    Returns:
        Tuple of (is_surface, surface_height, surface_dimensions).
    """
    center, size = _get_body_aabb(model, data, body_id)
    surface_width = size[0]
    surface_depth = size[1]
    surface_area = surface_width * surface_depth

    # Collect top-Z values from collision geoms only (skip visual-only geoms
    # whose vertices can extend well above the physical surface).
    top_zs = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        tz = _get_collision_geom_top_z(model, data, g)
        if tz is not None:
            top_zs.append(tz)

    if top_zs:
        # Use 75th percentile: robust against both low-z parts (legs)
        # and high-z outliers (decorative elements above the surface).
        surface_height = float(np.percentile(top_zs, 75))
    else:
        # Fallback to AABB top if no collision geoms found.
        surface_height = center[2] + size[2] / 2

    # Must be elevated and have enough area.
    if surface_height < min_height or surface_area < min_area:
        return False, 0.0, [0.0, 0.0]

    return True, surface_height, [surface_width, surface_depth]


def analyze_scene(scene_xml_path: Path) -> SceneDescription:
    """Analyze a MuJoCo scene and extract structured information.

    Loads the scene, identifies furniture items and potential placement surfaces,
    computes spatial relationships.

    Args:
        scene_xml_path: Path to the MuJoCo scene XML file.

    Returns:
        SceneDescription with furniture and surface information.
    """
    console_logger.info(f"Analyzing scene: {scene_xml_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Estimate room dimensions from the scene bounds.
    all_positions = data.xpos[1:]  # Skip worldbody.
    if len(all_positions) > 0:
        room_min = np.min(all_positions, axis=0)
        room_max = np.max(all_positions, axis=0)
        room_dims = list(room_max - room_min + 0.5)  # Add margin.
    else:
        room_dims = [5.0, 5.0, 3.0]

    # Detect room type from body names.
    body_names = []
    for i in range(model.nbody):
        name = model.body(i).name
        if name:
            body_names.append(name.lower())
    name_str = " ".join(body_names)

    room_type = "room"
    if "kitchen" in name_str or "counter" in name_str or "stove" in name_str:
        room_type = "kitchen"
    elif "bedroom" in name_str or "bed" in name_str:
        room_type = "bedroom"
    elif "living" in name_str or "sofa" in name_str or "couch" in name_str:
        room_type = "living room"
    elif "bathroom" in name_str or "toilet" in name_str:
        room_type = "bathroom"
    elif "office" in name_str or "desk" in name_str:
        room_type = "office"

    # Extract furniture and surfaces.
    furniture_list = []
    surface_list = []

    for i in range(1, model.nbody):  # Skip world body.
        body_name = model.body(i).name
        if not body_name:
            continue

        # Skip floor/wall/ceiling bodies.
        lower_name = body_name.lower()
        if any(
            skip in lower_name
            for skip in ["floor", "wall", "ceiling", "room", "light"]
        ):
            continue

        center, size = _get_body_aabb(model, data, i)

        furniture_list.append(
            FurnitureInfo(
                name=body_name,
                position=list(center),
                size=list(size),
                body_name=body_name,
            )
        )

        # Check if this could be a placement surface.
        is_surface, surface_height, surface_dims = _is_horizontal_surface(
            model, data, i
        )
        if is_surface:
            # Find nearby objects within arm reach.
            nearby = []
            surface_pos = np.array(center[:2])
            for j in range(1, model.nbody):
                if j == i:
                    continue
                other_name = model.body(j).name
                if not other_name:
                    continue
                other_pos = np.array(data.xpos[j][:2])
                dist = np.linalg.norm(surface_pos - other_pos)
                if dist < ARM_REACH_RADIUS * 2:
                    nearby.append(other_name)

            # Estimate clearance (distance to nearest obstacle at surface level).
            clearance = ARM_REACH_RADIUS  # Default.
            for j in range(1, model.nbody):
                if j == i:
                    continue
                other_center, other_size = _get_body_aabb(model, data, j)
                # Check if other object is at similar height.
                if abs(other_center[2] - center[2]) < 0.5:
                    dist = np.linalg.norm(
                        np.array(center[:2]) - np.array(other_center[:2])
                    )
                    edge_dist = max(
                        0, dist - (size[0] + other_size[0]) / 2
                    )
                    clearance = min(clearance, edge_dist)

            surface_list.append(
                SurfaceInfo(
                    name=body_name,
                    body_name=body_name,
                    position=[center[0], center[1], surface_height],
                    height=surface_height,
                    dimensions=surface_dims,
                    nearby_objects=nearby[:5],  # Limit for readability.
                    clearance=clearance,
                )
            )

    # Sort surfaces by height (prefer countertop-height surfaces).
    surface_list.sort(key=lambda s: abs(s.height - 0.85), reverse=False)

    console_logger.info(
        f"Scene analysis: {len(furniture_list)} furniture items, "
        f"{len(surface_list)} potential surfaces"
    )

    return SceneDescription(
        room_type=room_type,
        room_dimensions=room_dims,
        furniture=furniture_list,
        surfaces=surface_list,
    )


def get_surface_details(
    scene_xml_path: Path, body_name: str
) -> SurfaceInfo | None:
    """Get detailed information about a specific surface.

    Args:
        scene_xml_path: Path to the MuJoCo scene XML.
        body_name: Name of the body to inspect.

    Returns:
        SurfaceInfo if the body is a valid surface, None otherwise.
    """
    scene_desc = analyze_scene(scene_xml_path)
    for surface in scene_desc.surfaces:
        if surface.body_name == body_name:
            return surface
    return None
