"""URDF to MuJoCo MJCF converter for robot arm descriptions.

Converts a URDF file (with STL mesh references) into a standalone MuJoCo MJCF
XML file. Handles package:// URI resolution, mesh copying, joint limits,
mimic joints (via MuJoCo equality constraints), and inertial properties.
"""

import logging
import shutil
import xml.etree.ElementTree as ET

from pathlib import Path

import mujoco
import numpy as np

console_logger = logging.getLogger(__name__)


def _resolve_package_uri(uri: str, package_dir: Path) -> Path:
    """Resolve a package:// URI to an absolute file path.

    Args:
        uri: URI string like "package://open_manipulator_description/meshes/...".
        package_dir: Root directory of the ROS package (parent of meshes/).

    Returns:
        Absolute path to the referenced file.
    """
    if uri.startswith("package://"):
        # Strip "package://package_name/" prefix.
        parts = uri[len("package://") :].split("/", 1)
        if len(parts) == 2:
            return package_dir / parts[1]
    return Path(uri)


def _find_package_dir(urdf_path: Path) -> Path:
    """Find the ROS package root directory from a URDF path.

    Walks up from the URDF file to find the directory containing both
    'urdf/' and 'meshes/' subdirectories.

    Args:
        urdf_path: Path to the URDF file.

    Returns:
        Path to the package root directory.

    Raises:
        FileNotFoundError: If package root cannot be determined.
    """
    # Walk up from URDF location looking for meshes/ sibling.
    current = urdf_path.parent
    for _ in range(5):  # Max 5 levels up.
        if (current / "meshes").is_dir():
            return current
        current = current.parent

    raise FileNotFoundError(
        f"Cannot find package root (directory with meshes/) from {urdf_path}"
    )


def convert_urdf_to_mjcf(
    urdf_path: Path,
    output_dir: Path,
    package_dir: Path | None = None,
) -> Path:
    """Convert a URDF robot description to a standalone MuJoCo MJCF file.

    Parses the URDF XML, resolves mesh references, copies STL files,
    and builds a MuJoCo model using the MjSpec API.

    Args:
        urdf_path: Path to the URDF file.
        output_dir: Directory for output MJCF and copied meshes.
        package_dir: Optional package root override. If None, auto-detected.

    Returns:
        Path to the generated MJCF file.
    """
    if package_dir is None:
        package_dir = _find_package_dir(urdf_path)

    console_logger.info(f"Converting URDF: {urdf_path}")
    console_logger.info(f"Package dir: {package_dir}")

    # Parse URDF.
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.attrib.get("name", "robot")

    # Create output directories.
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "arm_meshes"
    mesh_dir.mkdir(exist_ok=True)

    # Collect all links and joints from URDF.
    links = {}
    for link_elem in root.findall("link"):
        links[link_elem.attrib["name"]] = link_elem

    joints = {}
    joint_order = []
    mimic_joints = {}
    for joint_elem in root.findall("joint"):
        jname = joint_elem.attrib["name"]
        joints[jname] = joint_elem
        joint_order.append(jname)
        mimic_elem = joint_elem.find("mimic")
        if mimic_elem is not None:
            mimic_joints[jname] = {
                "joint": mimic_elem.attrib["joint"],
                "multiplier": float(mimic_elem.attrib.get("multiplier", 1.0)),
                "offset": float(mimic_elem.attrib.get("offset", 0.0)),
            }

    # Build kinematic chain: parent_link -> [(joint, child_link), ...].
    children = {}
    parent_of = {}
    for jname in joint_order:
        jelem = joints[jname]
        parent_link = jelem.find("parent").attrib["link"]
        child_link = jelem.find("child").attrib["link"]
        children.setdefault(parent_link, []).append((jname, child_link))
        parent_of[child_link] = (jname, parent_link)

    # Find root link (no parent).
    all_child_links = {cl for _, cl in [parent_of[k] for k in parent_of]}
    root_links = [ln for ln in links if ln not in parent_of]

    # Copy meshes and track filenames.
    mesh_files = {}  # original_filename -> copied_filename

    def _copy_mesh(uri: str) -> str | None:
        """Copy a mesh file and return the relative path for MJCF."""
        if not uri:
            return None
        src = _resolve_package_uri(uri, package_dir)
        if not src.exists():
            console_logger.warning(f"Mesh file not found: {src}")
            return None
        dst = mesh_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        mesh_files[src.name] = src.name
        return f"arm_meshes/{src.name}"

    def _parse_origin(elem: ET.Element | None) -> tuple[list[float], list[float]]:
        """Parse <origin xyz="..." rpy="..."/> element.

        Returns:
            Tuple of (xyz position, rpy euler angles).
        """
        if elem is None:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        xyz = [float(v) for v in elem.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in elem.attrib.get("rpy", "0 0 0").split()]
        return xyz, rpy

    def _rpy_to_quat(rpy: list[float]) -> list[float]:
        """Convert roll-pitch-yaw to quaternion [w, x, y, z]."""
        r, p, y = rpy
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        cp, sp = np.cos(p / 2), np.sin(p / 2)
        cy, sy = np.cos(y / 2), np.sin(y / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y_q = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [w, x, y_q, z]

    # Build MuJoCo spec.
    spec = mujoco.MjSpec()
    spec.modelname = robot_name

    # Set compiler defaults.
    # degree=False means angles are in radians (URDF convention).
    spec.compiler.degree = False
    spec.compiler.meshdir = str(output_dir)

    # Add default class for geoms.
    main_default = spec.default
    main_default.geom.type = mujoco.mjtGeom.mjGEOM_MESH

    # Add meshes to spec.
    mesh_name_map = {}  # stl_filename -> mjcf_mesh_name

    def _add_mesh_from_geometry(geom_elem: ET.Element) -> str | None:
        """Add mesh from a URDF geometry element and return mesh name."""
        mesh_elem = geom_elem.find("mesh")
        if mesh_elem is None:
            # Handle box geometry.
            box_elem = geom_elem.find("box")
            if box_elem is not None:
                return None  # Box handled separately.
            return None

        filename = mesh_elem.attrib.get("filename", "")
        scale_str = mesh_elem.attrib.get("scale", "1 1 1")
        scale = [float(v) for v in scale_str.split()]

        rel_path = _copy_mesh(filename)
        if rel_path is None:
            return None

        stl_name = Path(rel_path).stem
        if stl_name not in mesh_name_map:
            mj_mesh = spec.add_mesh()
            mj_mesh.name = stl_name
            mj_mesh.file = rel_path
            mj_mesh.scale = scale
            mesh_name_map[stl_name] = stl_name

        return stl_name

    def _add_body_recursive(parent_body, link_name: str) -> None:
        """Recursively add bodies for the kinematic chain."""
        link_elem = links.get(link_name)
        if link_elem is None:
            return

        # Process child joints from this link.
        for jname, child_link_name in children.get(link_name, []):
            jelem = joints[jname]
            jtype = jelem.attrib.get("type", "fixed")

            # Parse joint origin (transform from parent link to child link).
            origin_elem = jelem.find("origin")
            xyz, rpy = _parse_origin(origin_elem)

            # Create child body.
            child_body = parent_body.add_body()
            child_body.name = child_link_name
            child_body.pos = xyz
            if any(v != 0.0 for v in rpy):
                child_body.quat = _rpy_to_quat(rpy)

            # Add inertial properties.
            child_link_elem = links.get(child_link_name)
            if child_link_elem is not None:
                inertial_elem = child_link_elem.find("inertial")
                if inertial_elem is not None:
                    mass_elem = inertial_elem.find("mass")
                    if mass_elem is not None:
                        mass_val = float(mass_elem.attrib["value"])
                        if mass_val > 0:
                            child_body.mass = mass_val

                    inertia_origin = inertial_elem.find("origin")
                    if inertia_origin is not None:
                        ipos = [
                            float(v)
                            for v in inertia_origin.attrib.get("xyz", "0 0 0").split()
                        ]
                        child_body.ipos = ipos

                    inertia_elem = inertial_elem.find("inertia")
                    if inertia_elem is not None:
                        ixx = float(inertia_elem.attrib.get("ixx", "0"))
                        iyy = float(inertia_elem.attrib.get("iyy", "0"))
                        izz = float(inertia_elem.attrib.get("izz", "0"))
                        ixy = float(inertia_elem.attrib.get("ixy", "0"))
                        ixz = float(inertia_elem.attrib.get("ixz", "0"))
                        iyz = float(inertia_elem.attrib.get("iyz", "0"))
                        child_body.fullinertia = [ixx, iyy, izz, ixy, ixz, iyz]

            # Add joint (unless fixed).
            if jtype == "revolute" or jtype == "continuous":
                mj_joint = child_body.add_joint()
                mj_joint.name = jname
                mj_joint.type = mujoco.mjtJoint.mjJNT_HINGE

                axis_elem = jelem.find("axis")
                if axis_elem is not None:
                    mj_joint.axis = [
                        float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()
                    ]

                limit_elem = jelem.find("limit")
                if limit_elem is not None and jtype == "revolute":
                    lower = float(limit_elem.attrib.get("lower", "-3.14159"))
                    upper = float(limit_elem.attrib.get("upper", "3.14159"))
                    mj_joint.limited = True
                    mj_joint.range = [lower, upper]

                dynamics_elem = jelem.find("dynamics")
                if dynamics_elem is not None:
                    damping = float(dynamics_elem.attrib.get("damping", "0"))
                    mj_joint.damping = damping

            elif jtype == "prismatic":
                mj_joint = child_body.add_joint()
                mj_joint.name = jname
                mj_joint.type = mujoco.mjtJoint.mjJNT_SLIDE

                axis_elem = jelem.find("axis")
                if axis_elem is not None:
                    mj_joint.axis = [
                        float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()
                    ]

                limit_elem = jelem.find("limit")
                if limit_elem is not None:
                    lower = float(limit_elem.attrib.get("lower", "0"))
                    upper = float(limit_elem.attrib.get("upper", "0"))
                    mj_joint.limited = True
                    mj_joint.range = [lower, upper]

            # Add visual geoms.
            if child_link_elem is not None:
                for visual_elem in child_link_elem.findall("visual"):
                    geom_origin = visual_elem.find("origin")
                    geom_xyz, geom_rpy = _parse_origin(geom_origin)

                    geometry = visual_elem.find("geometry")
                    if geometry is None:
                        continue

                    box_elem = geometry.find("box")
                    if box_elem is not None:
                        # Small box geometry (e.g., end effector marker).
                        size_str = box_elem.attrib.get("size", "0.01 0.01 0.01")
                        size = [float(v) / 2 for v in size_str.split()]
                        geom = child_body.add_geom()
                        geom.type = mujoco.mjtGeom.mjGEOM_BOX
                        geom.size = size
                        geom.pos = geom_xyz
                        if any(v != 0.0 for v in geom_rpy):
                            geom.quat = _rpy_to_quat(geom_rpy)
                        geom.contype = 0
                        geom.conaffinity = 0

                        # Parse material color.
                        material = visual_elem.find("material")
                        if material is not None:
                            color_elem = material.find("color")
                            if color_elem is not None:
                                rgba = [
                                    float(v)
                                    for v in color_elem.attrib["rgba"].split()
                                ]
                                geom.rgba = rgba
                        continue

                    mesh_name = _add_mesh_from_geometry(geometry)
                    if mesh_name is None:
                        continue

                    geom = child_body.add_geom()
                    geom.meshname = mesh_name
                    geom.type = mujoco.mjtGeom.mjGEOM_MESH
                    geom.pos = geom_xyz
                    if any(v != 0.0 for v in geom_rpy):
                        geom.quat = _rpy_to_quat(geom_rpy)
                    geom.contype = 0
                    geom.conaffinity = 0

                    # Parse material color.
                    material = visual_elem.find("material")
                    if material is not None:
                        color_elem = material.find("color")
                        if color_elem is not None:
                            rgba = [
                                float(v) for v in color_elem.attrib["rgba"].split()
                            ]
                            geom.rgba = rgba

                # Add collision geoms.
                for collision_elem in child_link_elem.findall("collision"):
                    geom_origin = collision_elem.find("origin")
                    geom_xyz, geom_rpy = _parse_origin(geom_origin)

                    geometry = collision_elem.find("geometry")
                    if geometry is None:
                        continue

                    box_elem = geometry.find("box")
                    if box_elem is not None:
                        size_str = box_elem.attrib.get("size", "0.01 0.01 0.01")
                        size = [float(v) / 2 for v in size_str.split()]
                        geom = child_body.add_geom()
                        geom.type = mujoco.mjtGeom.mjGEOM_BOX
                        geom.size = size
                        geom.pos = geom_xyz
                        if any(v != 0.0 for v in geom_rpy):
                            geom.quat = _rpy_to_quat(geom_rpy)
                        geom.group = 3  # Collision group (not rendered by default).
                        continue

                    mesh_name = _add_mesh_from_geometry(geometry)
                    if mesh_name is None:
                        continue

                    geom = child_body.add_geom()
                    geom.meshname = mesh_name
                    geom.type = mujoco.mjtGeom.mjGEOM_MESH
                    geom.pos = geom_xyz
                    if any(v != 0.0 for v in geom_rpy):
                        geom.quat = _rpy_to_quat(geom_rpy)
                    geom.group = 3

            # Recurse into children.
            _add_body_recursive(child_body, child_link_name)

    # Build the tree starting from root link(s).
    # Skip "world" link if it exists (just a fixed reference frame).
    for root_link in root_links:
        if root_link == "world":
            # Start from children of world.
            for jname, child_link in children.get("world", []):
                jelem = joints[jname]
                origin_elem = jelem.find("origin")
                xyz, rpy = _parse_origin(origin_elem)

                root_body = spec.worldbody.add_body()
                root_body.name = child_link
                root_body.pos = xyz
                if any(v != 0.0 for v in rpy):
                    root_body.quat = _rpy_to_quat(rpy)

                # Add visual/collision geoms for root body.
                root_link_elem = links.get(child_link)
                if root_link_elem is not None:
                    # Inertial.
                    inertial_elem = root_link_elem.find("inertial")
                    if inertial_elem is not None:
                        mass_elem = inertial_elem.find("mass")
                        if mass_elem is not None:
                            root_body.mass = float(mass_elem.attrib["value"])
                        inertia_origin = inertial_elem.find("origin")
                        if inertia_origin is not None:
                            ipos = [
                                float(v)
                                for v in inertia_origin.attrib.get(
                                    "xyz", "0 0 0"
                                ).split()
                            ]
                            root_body.ipos = ipos
                        inertia_elem = inertial_elem.find("inertia")
                        if inertia_elem is not None:
                            ixx = float(inertia_elem.attrib.get("ixx", "0"))
                            iyy = float(inertia_elem.attrib.get("iyy", "0"))
                            izz = float(inertia_elem.attrib.get("izz", "0"))
                            ixy = float(inertia_elem.attrib.get("ixy", "0"))
                            ixz = float(inertia_elem.attrib.get("ixz", "0"))
                            iyz = float(inertia_elem.attrib.get("iyz", "0"))
                            root_body.fullinertia = [ixx, iyy, izz, ixy, ixz, iyz]

                    for visual_elem in root_link_elem.findall("visual"):
                        geom_origin = visual_elem.find("origin")
                        geom_xyz, geom_rpy = _parse_origin(geom_origin)
                        geometry = visual_elem.find("geometry")
                        if geometry is None:
                            continue
                        mesh_name = _add_mesh_from_geometry(geometry)
                        if mesh_name is None:
                            continue
                        geom = root_body.add_geom()
                        geom.meshname = mesh_name
                        geom.type = mujoco.mjtGeom.mjGEOM_MESH
                        geom.pos = geom_xyz
                        if any(v != 0.0 for v in geom_rpy):
                            geom.quat = _rpy_to_quat(geom_rpy)
                        geom.contype = 0
                        geom.conaffinity = 0

                        material = visual_elem.find("material")
                        if material is not None:
                            color_elem = material.find("color")
                            if color_elem is not None:
                                rgba = [
                                    float(v)
                                    for v in color_elem.attrib["rgba"].split()
                                ]
                                geom.rgba = rgba

                    for collision_elem in root_link_elem.findall("collision"):
                        geom_origin = collision_elem.find("origin")
                        geom_xyz, geom_rpy = _parse_origin(geom_origin)
                        geometry = collision_elem.find("geometry")
                        if geometry is None:
                            continue
                        mesh_name = _add_mesh_from_geometry(geometry)
                        if mesh_name is None:
                            continue
                        geom = root_body.add_geom()
                        geom.meshname = mesh_name
                        geom.type = mujoco.mjtGeom.mjGEOM_MESH
                        geom.pos = geom_xyz
                        if any(v != 0.0 for v in geom_rpy):
                            geom.quat = _rpy_to_quat(geom_rpy)
                        geom.group = 3

                _add_body_recursive(root_body, child_link)
        else:
            # Non-world root link.
            root_body = spec.worldbody.add_body()
            root_body.name = root_link
            _add_body_recursive(root_body, root_link)

    # Add mimic joint equality constraints.
    for mimic_jname, mimic_info in mimic_joints.items():
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name = f"mimic_{mimic_jname}"
        eq.name1 = mimic_info["joint"]
        eq.name2 = mimic_jname
        # For joint equality, data[0:5] are polynomial coefficients:
        # joint2 = data[0] + data[1]*joint1 + data[2]*joint1^2 + ...
        eq.data[0] = mimic_info["offset"]
        eq.data[1] = mimic_info["multiplier"]
        eq.data[2] = 0.0
        eq.data[3] = 0.0
        eq.data[4] = 0.0

    # Compile and export.
    output_path = output_dir / f"{robot_name}.xml"
    try:
        model = spec.compile()
        xml_string = spec.to_xml()
        with open(output_path, "w") as f:
            f.write(xml_string)
        console_logger.info(
            f"MJCF exported: {output_path} "
            f"({model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms)"
        )
    except Exception as e:
        console_logger.error(f"Failed to compile MuJoCo spec: {e}")
        raise

    # Validate the arm loads independently.
    _validate_arm_mjcf(output_path)

    return output_path


def _validate_arm_mjcf(mjcf_path: Path) -> bool:
    """Validate that the arm MJCF loads and simulates without errors.

    Args:
        mjcf_path: Path to the MJCF file.

    Returns:
        True if validation passed.
    """
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)

        for _ in range(10):
            mujoco.mj_step(model, data)

        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            console_logger.error("Arm simulation produced NaN values")
            return False

        console_logger.info(
            f"Arm validation passed: {model.nbody} bodies, "
            f"{model.njnt} joints, {model.ngeom} geoms"
        )
        return True
    except Exception as e:
        console_logger.error(f"Arm validation failed: {e}")
        return False
