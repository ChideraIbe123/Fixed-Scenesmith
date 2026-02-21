"""Pi0.5 VLA control loop for the OMX-F arm in MuJoCo.

Connects to a remote Pi0.5 DROID policy server, renders VLA camera views,
packs DROID-format observations, receives joint velocity actions, and
integrates them into the arm's qpos.

Supports two modes:
- Kinematic (default): velocity integration → qpos → mj_forward
- Physics (--physics): position actuators → mj_step with contacts/gravity

Pipeline: render cameras → pack observation → send to Pi0.5 → receive
action chunk → integrate velocities → repeat at control_freq Hz.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from scenesmith.arm_placement.mujoco_renderer import (
    _create_scene_option,
    _enhance_scene_xml,
)

console_logger = logging.getLogger(__name__)

# OMX-F arm joint names (5 arm + 2 gripper).
ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5",
    "gripper_joint_1", "gripper_joint_2",
]

# VLA camera names injected by vla_camera.py.
VLA_THIRD_PERSON_CAM = "vla_third_person"
VLA_WRIST_CAM = "vla_wrist"


@dataclass
class VLAControlConfig:
    """Configuration for the VLA control loop."""

    # Remote Pi0.5 policy server.
    server_host: str = "136.113.180.52"
    server_port: int = 8000

    # Task instruction for the VLA model.
    prompt: str = "pick up the object"

    # Control timing.
    control_freq: float = 15.0        # Hz
    open_loop_horizon: int = 8        # Steps between server queries.
    max_steps: int = 600              # Episode length.

    # Action integration.
    action_scale: float = 0.1         # Velocity → qpos scaling factor.

    # Gripper control.
    gripper_threshold: float = 0.5    # Binarization threshold.
    gripper_open_qpos: float = 0.0
    gripper_close_qpos: float = 0.5

    # Test object injection.
    add_test_object: bool = False

    # Physics mode: use mj_step with position actuators for real contacts.
    physics: bool = False

    # Recording.
    record: bool = False
    record_dir: Path = field(default_factory=lambda: Path("renders/vla_episode"))


class VLAControlLoop:
    """Closed-loop VLA control for the OMX-F arm in MuJoCo.

    Renders two VLA camera views (third-person + wrist), packs
    DROID-format observations, queries a remote Pi0.5 policy server
    via websocket, and integrates the returned joint velocity actions
    into the arm's qpos using kinematic control.
    """

    def __init__(
        self,
        scene_xml_path: str | Path,
        config: VLAControlConfig | None = None,
    ):
        self.scene_xml_path = Path(scene_xml_path)
        self.config = config or VLAControlConfig()
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._scene_option: mujoco.MjvOption | None = None
        self._policy = None

        # Joint index caches (populated in _setup).
        self._arm_qpos_indices: list[int] = []       # joint1-5
        self._gripper1_qpos_idx: int = -1
        self._gripper2_qpos_idx: int = -1

        # Physics mode state.
        self._target_qpos: np.ndarray | None = None  # Position targets for actuators.
        self._n_substeps: int = 33  # Physics substeps per control step.

        # Recording state.
        self._frames: list[np.ndarray] = []          # Third-person view.
        self._frames_wrist: list[np.ndarray] = []    # Wrist view.
        self._frames_overview: list[np.ndarray] = [] # Wide overview.
        self._overview_renderer: mujoco.Renderer | None = None
        self._step_count = 0

        # Trajectory recording (joint angles per frame for Blender replay).
        self._traj_joint_angles: list[np.ndarray] = []   # (5,) per frame
        self._traj_gripper_angles: list[float] = []       # scalar per frame
        self._traj_cup_poses: list[np.ndarray] = []       # (7,) per frame
        self._cup_qpos_idx: int = -1  # Start of test_cup freejoint in qpos

    # ------------------------------------------------------------------
    # Test object injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_test_object(scene_xml_path: Path) -> Path:
        """Inject a small red cup near the arm for pickup testing.

        Places a free-body cylinder (cup-like) ~15cm in front of the arm
        base on the table surface. Returns path to the modified XML.
        """
        import os
        import tempfile
        import xml.etree.ElementTree as ET

        tree = ET.parse(scene_xml_path)
        root = tree.getroot()

        # Find arm base position to place object nearby.
        arm_body = None
        for elem in root.iter("body"):
            if elem.get("name") == "omx_f_base":
                arm_body = elem
                break

        if arm_body is None:
            raise ValueError("Arm body 'omx_f_base' not found in scene.")

        arm_pos = [float(v) for v in arm_body.get("pos", "0 0 0").split()]
        arm_quat = [float(v) for v in arm_body.get("quat", "1 0 0 0").split()]

        # Compute "forward" direction from arm yaw.
        from math import atan2, cos, sin
        qw, _qx, _qy, qz = arm_quat
        theta = 2.0 * atan2(qz, qw)
        fwd_x, fwd_y = cos(theta), sin(theta)

        # Place 25cm in front, 3cm above arm base (on table surface).
        obj_x = arm_pos[0] + fwd_x * 0.25
        obj_y = arm_pos[1] + fwd_y * 0.25
        obj_z = arm_pos[2] + 0.03

        console_logger.info(
            f"Injecting test cup at [{obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}]"
        )

        # Add red material for the cup.
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        # Remove previous test material (idempotency).
        for mat in asset.findall("material"):
            if mat.get("name") == "test_cup_red":
                asset.remove(mat)

        mat_elem = ET.SubElement(asset, "material")
        mat_elem.set("name", "test_cup_red")
        mat_elem.set("rgba", "0.85 0.15 0.15 1")

        # Remove previous test object (idempotency).
        worldbody = root.find("worldbody")
        for body in worldbody.findall("body"):
            if body.get("name") == "test_cup":
                worldbody.remove(body)

        # Free-body cup (cylinder + open top approximation).
        cup_body = ET.SubElement(worldbody, "body")
        cup_body.set("name", "test_cup")
        cup_body.set("pos", f"{obj_x:.6f} {obj_y:.6f} {obj_z:.6f}")

        # Freejoint so it can be picked up.
        fj = ET.SubElement(cup_body, "freejoint")
        fj.set("name", "test_cup_joint")

        # Cup body: cylinder shell.
        geom = ET.SubElement(cup_body, "geom")
        geom.set("name", "test_cup_geom")
        geom.set("type", "cylinder")
        geom.set("size", "0.025 0.035")  # radius=2.5cm, half-height=3.5cm
        geom.set("mass", "0.05")
        geom.set("material", "test_cup_red")

        # Extend keyframe qpos to include the new freejoint DOF (3 pos + 4 quat).
        # Without this, mj_resetDataKeyframe silently fails due to size mismatch.
        keyframe = root.find("keyframe")
        if keyframe is not None:
            for key in keyframe.findall("key"):
                qpos_str = key.get("qpos", "")
                if qpos_str:
                    # Freejoint: 3 pos (0,0,0) + 4 quat (1,0,0,0).
                    key.set("qpos", qpos_str + " 0 0 0 1 0 0 0")

        # Write modified XML next to the original.
        scene_dir = scene_xml_path.parent
        fd, tmp_path = tempfile.mkstemp(
            suffix="_with_cup.xml", dir=scene_dir
        )
        os.close(fd)
        tree.write(tmp_path, xml_declaration=True)
        console_logger.info(f"Wrote test scene to {tmp_path}")
        return Path(tmp_path)

    # ------------------------------------------------------------------
    # Actuator injection for physics mode
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_actuators(scene_xml_path: Path) -> Path:
        """Inject position actuators and physics options into the scene XML.

        Adds:
        - <option> with timestep and gravity
        - Position actuators for joint1-5 and gripper_joint_1
        - gripper_joint_2 is handled by the mimic equality constraint

        Returns path to the modified XML.
        """
        import os
        import tempfile
        import xml.etree.ElementTree as ET

        tree = ET.parse(scene_xml_path)
        root = tree.getroot()

        # Add/replace <option> element with physics settings.
        # Must be near the top of <mujoco> (after <compiler>).
        option = root.find("option")
        if option is None:
            option = ET.Element("option")
            # Insert after <compiler> (index 1) or at the start.
            compiler_idx = 0
            for i, child in enumerate(root):
                if child.tag == "compiler":
                    compiler_idx = i + 1
                    break
            root.insert(compiler_idx, option)
        option.set("timestep", "0.002")
        option.set("gravity", "0 0 -9.81")

        # Remove any existing <actuator> block (idempotency).
        for act in root.findall("actuator"):
            root.remove(act)

        # Add <actuator> after <equality> (MuJoCo element ordering).
        actuator = ET.Element("actuator")
        equality_idx = len(root)
        for i, child in enumerate(root):
            if child.tag == "equality":
                equality_idx = i + 1
                break
        root.insert(equality_idx, actuator)

        arm_joints = [
            ("pos_joint1", "joint1", "50", "-6.28 6.28"),
            ("pos_joint2", "joint2", "50", "-6.28 6.28"),
            ("pos_joint3", "joint3", "50", "-6.28 6.28"),
            ("pos_joint4", "joint4", "50", "-6.28 6.28"),
            ("pos_joint5", "joint5", "50", "-6.28 6.28"),
            ("pos_gripper", "gripper_joint_1", "20", "-0.5 0.5"),
        ]

        for act_name, joint_name, kp, ctrlrange in arm_joints:
            elem = ET.SubElement(actuator, "position")
            elem.set("name", act_name)
            elem.set("joint", joint_name)
            elem.set("kp", kp)
            elem.set("ctrlrange", ctrlrange)
            elem.set("ctrllimited", "true")

        console_logger.info(
            f"Injected {len(arm_joints)} position actuators into scene."
        )

        # Write modified XML.
        scene_dir = scene_xml_path.parent
        fd, tmp_path = tempfile.mkstemp(
            suffix="_with_actuators.xml", dir=scene_dir
        )
        os.close(fd)
        tree.write(tmp_path, xml_declaration=True)
        return Path(tmp_path)

    # ------------------------------------------------------------------
    # Overview camera injection for recording
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_overview_camera(scene_xml_path: Path) -> Path:
        """Inject a wide-angle overview camera for recording.

        Places a camera farther back and to the side of the arm,
        giving a full view of the workspace. Does NOT affect VLA
        observation cameras.

        Returns path to the modified XML.
        """
        import os
        import tempfile
        import xml.etree.ElementTree as ET
        from math import atan2, cos, sin

        tree = ET.parse(scene_xml_path)
        root = tree.getroot()

        # Find arm base for positioning.
        arm_body = None
        for elem in root.iter("body"):
            if elem.get("name") == "omx_f_base":
                arm_body = elem
                break

        if arm_body is None:
            raise ValueError("Arm body 'omx_f_base' not found.")

        arm_pos = [float(v) for v in arm_body.get("pos", "0 0 0").split()]
        arm_quat = [float(v) for v in arm_body.get("quat", "1 0 0 0").split()]

        qw, _qx, _qy, qz = arm_quat
        theta = 2.0 * atan2(qz, qw)
        forward = np.array([cos(theta), sin(theta), 0.0])

        # Cross product with up to get "right" direction.
        right = np.array([-sin(theta), cos(theta), 0.0])

        # Place camera: 0.8m behind, 0.3m to the right, 1.0m above.
        cam_pos = (
            np.array(arm_pos)
            - forward * 0.8
            + right * 0.3
            + np.array([0.0, 0.0, 1.0])
        )

        # Look at a point slightly in front of the arm at table height.
        lookat = np.array(arm_pos) + forward * 0.15 + np.array([0.0, 0.0, 0.05])

        look_dir = lookat - cam_pos
        look_dir /= np.linalg.norm(look_dir)

        world_up = np.array([0.0, 0.0, 1.0])
        cam_right = np.cross(look_dir, world_up)
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, look_dir)
        cam_up /= np.linalg.norm(cam_up)

        xyaxes = np.concatenate([cam_right, cam_up])

        # Remove existing overview camera (idempotency).
        worldbody = root.find("worldbody")
        for cam in worldbody.findall("camera"):
            if cam.get("name") == "vla_overview":
                worldbody.remove(cam)

        cam_elem = ET.SubElement(worldbody, "camera")
        cam_elem.set("name", "vla_overview")
        cam_elem.set("pos", f"{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}")
        cam_elem.set("xyaxes", " ".join(f"{v:.6f}" for v in xyaxes))
        cam_elem.set("fovy", "60")

        # Write modified XML in-place.
        tree.write(str(scene_xml_path), xml_declaration=True)
        console_logger.info(f"Injected overview camera into {scene_xml_path}")
        return scene_xml_path

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Load model, create data, connect to policy server."""
        scene_path = self.scene_xml_path
        if self.config.add_test_object:
            scene_path = self._inject_test_object(scene_path)
        if self.config.physics:
            scene_path = self._inject_actuators(scene_path)

        # Add wide overview camera for recording (doesn't affect VLA obs).
        if self.config.record:
            scene_path = self._inject_overview_camera(scene_path)

        console_logger.info("Loading scene with 'fast' quality...")
        self._model = _enhance_scene_xml(scene_path, quality="fast")
        self._data = mujoco.MjData(self._model)

        # Reset to home keyframe if available.
        try:
            home_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_KEY, "home"
            )
            if home_id >= 0:
                mujoco.mj_resetDataKeyframe(self._model, self._data, home_id)
                console_logger.info("Reset to 'home' keyframe.")
        except Exception:
            pass

        mujoco.mj_forward(self._model, self._data)

        # Resolve arm joint qpos indices.
        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        self._arm_qpos_indices = []
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid >= 0:
                self._arm_qpos_indices.append(int(self._model.jnt_qposadr[jid]))
            else:
                console_logger.warning(f"Arm joint '{name}' not found in model.")

        # Gripper joints.
        for gname, attr in [
            ("gripper_joint_1", "_gripper1_qpos_idx"),
            ("gripper_joint_2", "_gripper2_qpos_idx"),
        ]:
            jid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, gname
            )
            if jid >= 0:
                setattr(self, attr, int(self._model.jnt_qposadr[jid]))
            else:
                console_logger.warning(f"Gripper joint '{gname}' not found.")

        # Detect test cup freejoint for trajectory recording.
        cup_jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "test_cup_joint"
        )
        if cup_jid >= 0:
            self._cup_qpos_idx = int(self._model.jnt_qposadr[cup_jid])
            console_logger.info(
                f"Test cup freejoint at qpos index {self._cup_qpos_idx}"
            )

        console_logger.info(
            f"Arm joint indices: {self._arm_qpos_indices}, "
            f"gripper indices: {self._gripper1_qpos_idx}, {self._gripper2_qpos_idx}"
        )

        # Physics mode: initialize targets, set ctrl, warm up.
        if self.config.physics:
            self._n_substeps = int(
                1.0 / (self.config.control_freq * self._model.opt.timestep)
            )
            console_logger.info(
                f"Physics mode: {self._n_substeps} substeps/control step, "
                f"timestep={self._model.opt.timestep}s, "
                f"na={self._model.nu} actuators"
            )

            # Initialize target positions from current qpos (home pose).
            self._target_qpos = np.array(
                [self._data.qpos[idx] for idx in self._arm_qpos_indices],
                dtype=np.float64,
            )

            # Set ctrl to home position targets.
            for i, qpos_idx in enumerate(self._arm_qpos_indices):
                if i < self._model.nu:
                    self._data.ctrl[i] = self._data.qpos[qpos_idx]
            # Gripper ctrl (last actuator).
            if self._model.nu > len(self._arm_qpos_indices):
                self._data.ctrl[len(self._arm_qpos_indices)] = (
                    self._data.qpos[self._gripper1_qpos_idx]
                )

            # Warm up physics to settle arm under gravity.
            console_logger.info("Running physics warmup (200 steps)...")
            for _ in range(200):
                mujoco.mj_step(self._model, self._data)

            warmup_joints = [
                f"{self._data.qpos[idx]:.3f}"
                for idx in self._arm_qpos_indices
            ]
            console_logger.info(
                f"Post-warmup joints: [{', '.join(warmup_joints)}]"
            )

        # Verify VLA cameras exist.
        for cam_name in [VLA_THIRD_PERSON_CAM, VLA_WRIST_CAM]:
            cam_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
            )
            if cam_id < 0:
                raise ValueError(
                    f"VLA camera '{cam_name}' not found in scene XML. "
                    "Run inject_vla_cameras() first."
                )

        # Create renderer at 224×224 (DROID observation size).
        self._renderer = mujoco.Renderer(self._model, height=224, width=224)
        self._scene_option = _create_scene_option()

        # Create larger renderer for overview recording (480×480).
        if self.config.record:
            self._overview_renderer = mujoco.Renderer(
                self._model, height=480, width=480
            )

        # Connect to Pi0.5 policy server.
        self._connect_policy()

    def _connect_policy(self) -> None:
        """Connect (or reconnect) to the Pi0.5 policy server.

        Disables websocket keepalive pings so the first inference
        (which triggers JAX JIT compilation and can take 60-120s)
        doesn't get killed by a ping timeout.
        """
        console_logger.info(
            f"Connecting to Pi0.5 server at "
            f"{self.config.server_host}:{self.config.server_port}..."
        )
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        uri = f"ws://{self.config.server_host}:{self.config.server_port}"
        console_logger.info(f"Waiting for server at {uri}...")
        while True:
            try:
                ws = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,  # Disable keepalive pings.
                )
                break
            except (ConnectionRefusedError, TimeoutError):
                console_logger.info("Still waiting for server...")
                time.sleep(5)

        # Read server metadata (mirrors openpi-client handshake).
        metadata = msgpack_numpy.unpackb(ws.recv())
        console_logger.info(f"Server metadata: {list(metadata.keys())}")

        # Patch a WebsocketClientPolicy-compatible object in place.
        from openpi_client import websocket_client_policy

        policy = object.__new__(websocket_client_policy.WebsocketClientPolicy)
        policy._uri = uri
        policy._packer = msgpack_numpy.Packer()
        policy._api_key = None
        policy._ws = ws
        policy._server_metadata = metadata

        self._policy = policy
        console_logger.info("Connected to Pi0.5 policy server.")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_camera(self, camera_name: str) -> np.ndarray:
        """Render a single camera view as (224, 224, 3) uint8."""
        self._renderer.update_scene(
            self._data, camera=camera_name, scene_option=self._scene_option
        )
        return self._renderer.render().copy()

    # ------------------------------------------------------------------
    # Observation packing
    # ------------------------------------------------------------------

    def _pack_observation(self) -> dict:
        """Pack a DROID-format observation for the Pi0.5 model.

        Returns dict with keys expected by the DROID policy:
        - observation/exterior_image_1_left: (224,224,3) uint8
        - observation/wrist_image_left: (224,224,3) uint8
        - observation/joint_position: (7,) float32 (zero-padded from 5 DOF)
        - observation/gripper_position: (1,) float32
        - prompt: str
        """
        # Render cameras.
        third_person_img = self._render_camera(VLA_THIRD_PERSON_CAM)
        wrist_img = self._render_camera(VLA_WRIST_CAM)

        # Read arm joint positions (5 DOF), zero-pad to 7 for Franka format.
        joint_pos_5 = np.array(
            [self._data.qpos[idx] for idx in self._arm_qpos_indices],
            dtype=np.float32,
        )
        joint_pos_7 = np.zeros(7, dtype=np.float32)
        joint_pos_7[:len(joint_pos_5)] = joint_pos_5

        # Gripper position (single scalar).
        gripper_pos = np.array(
            [self._data.qpos[self._gripper1_qpos_idx]],
            dtype=np.float32,
        )

        return {
            "observation/exterior_image_1_left": third_person_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_pos_7,
            "observation/gripper_position": gripper_pos,
            "prompt": self.config.prompt,
        }

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply a single action to the arm.

        Action is 8-dim (Franka format):
          dims 0-4 → joint1-5 velocities
          dim 7    → gripper command

        In physics mode, integrates velocities into position targets and
        sets actuator ctrl signals, then runs mj_step substeps.
        In kinematic mode, directly sets qpos and calls mj_forward.

        Args:
            action: (8,) float array from Pi0.5 policy.
        """
        action = np.clip(action, -1.0, 1.0)
        dt = 1.0 / self.config.control_freq

        if self.config.physics:
            self._apply_action_physics(action, dt)
        else:
            self._apply_action_kinematic(action, dt)

    def _apply_action_kinematic(self, action: np.ndarray, dt: float) -> None:
        """Kinematic mode: directly set qpos, call mj_forward."""
        # Integrate joint velocities for arm joints 0-4.
        for i, qpos_idx in enumerate(self._arm_qpos_indices):
            if i < len(action):
                velocity = action[i] * self.config.action_scale
                new_val = self._data.qpos[qpos_idx] + velocity * dt

                # Clamp to joint limits.
                jid = self._model.jnt_qposadr.tolist().index(qpos_idx)
                if self._model.jnt_limited[jid]:
                    lo = self._model.jnt_range[jid, 0]
                    hi = self._model.jnt_range[jid, 1]
                    new_val = np.clip(new_val, lo, hi)

                self._data.qpos[qpos_idx] = new_val

        # Gripper: binarize using threshold (action dim 7).
        if len(action) > 7:
            gripper_cmd = action[7]
            if gripper_cmd > self.config.gripper_threshold:
                gripper_val = self.config.gripper_close_qpos
            else:
                gripper_val = self.config.gripper_open_qpos

            # Set both gripper joints (mimic: joint2 = -joint1).
            if self._gripper1_qpos_idx >= 0:
                self._data.qpos[self._gripper1_qpos_idx] = gripper_val
            if self._gripper2_qpos_idx >= 0:
                self._data.qpos[self._gripper2_qpos_idx] = -gripper_val

        # Forward kinematics (no physics step).
        mujoco.mj_forward(self._model, self._data)

    def _apply_action_physics(self, action: np.ndarray, dt: float) -> None:
        """Physics mode: update position targets, set ctrl, run mj_step."""
        # Integrate VLA velocities into position targets.
        for i in range(len(self._arm_qpos_indices)):
            if i < len(action):
                velocity = action[i] * self.config.action_scale
                self._target_qpos[i] += velocity * dt

                # Clamp to joint limits.
                qpos_idx = self._arm_qpos_indices[i]
                jid = self._model.jnt_qposadr.tolist().index(qpos_idx)
                if self._model.jnt_limited[jid]:
                    lo = self._model.jnt_range[jid, 0]
                    hi = self._model.jnt_range[jid, 1]
                    self._target_qpos[i] = np.clip(
                        self._target_qpos[i], lo, hi
                    )

        # Set arm actuator ctrl to position targets.
        for i in range(len(self._arm_qpos_indices)):
            if i < self._model.nu:
                self._data.ctrl[i] = self._target_qpos[i]

        # Gripper: binarize and set gripper actuator ctrl.
        gripper_act_idx = len(self._arm_qpos_indices)  # Last actuator.
        if len(action) > 7 and gripper_act_idx < self._model.nu:
            gripper_cmd = action[7]
            if gripper_cmd > self.config.gripper_threshold:
                gripper_target = self.config.gripper_close_qpos
            else:
                gripper_target = self.config.gripper_open_qpos
            self._data.ctrl[gripper_act_idx] = gripper_target

        # Run physics substeps.
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def _step(self, action: np.ndarray) -> None:
        """Execute one control step: apply action, optionally record."""
        self._apply_action(action)
        self._step_count += 1

        if self.config.record:
            # Record joint trajectory for Blender replay.
            joint_angles = np.array(
                [self._data.qpos[idx] for idx in self._arm_qpos_indices],
                dtype=np.float64,
            )
            self._traj_joint_angles.append(joint_angles)

            gripper_val = (
                float(self._data.qpos[self._gripper1_qpos_idx])
                if self._gripper1_qpos_idx >= 0
                else 0.0
            )
            self._traj_gripper_angles.append(gripper_val)

            if self._cup_qpos_idx >= 0:
                # Freejoint: 3 pos + 4 quat = 7 DOFs.
                cup_pose = self._data.qpos[
                    self._cup_qpos_idx : self._cup_qpos_idx + 7
                ].copy()
                self._traj_cup_poses.append(cup_pose)

            # Record from both VLA cameras.
            self._frames.append(self._render_camera(VLA_THIRD_PERSON_CAM))
            self._frames_wrist.append(self._render_camera(VLA_WRIST_CAM))
            # Record from wide overview camera (larger resolution).
            if self._overview_renderer is not None:
                self._overview_renderer.update_scene(
                    self._data, camera="vla_overview",
                    scene_option=self._scene_option,
                )
                self._frames_overview.append(
                    self._overview_renderer.render().copy()
                )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the VLA control episode.

        Queries the Pi0.5 server every `open_loop_horizon` steps,
        executing the returned action chunk in between. Maintains
        15 Hz control timing.

        Returns:
            Dict with episode stats (steps, duration, recording path).
        """
        self._setup()
        self._step_count = 0
        self._frames = []
        self._frames_wrist = []
        self._frames_overview = []
        self._traj_joint_angles = []
        self._traj_gripper_angles = []
        self._traj_cup_poses = []

        cfg = self.config
        dt = 1.0 / cfg.control_freq
        action_buffer: list[np.ndarray] = []
        buffer_idx = 0

        console_logger.info(
            f"Starting VLA episode: max_steps={cfg.max_steps}, "
            f"freq={cfg.control_freq}Hz, action_scale={cfg.action_scale}, "
            f"horizon={cfg.open_loop_horizon}"
        )
        console_logger.info(f"Prompt: '{cfg.prompt}'")

        t_start = time.time()

        try:
            while self._step_count < cfg.max_steps:
                t_loop = time.time()

                # Re-query server when action buffer is exhausted.
                if buffer_idx >= len(action_buffer):
                    obs = self._pack_observation()

                    # Log observation details on first query.
                    if self._step_count == 0:
                        for k, v in obs.items():
                            if isinstance(v, np.ndarray):
                                console_logger.info(
                                    f"  obs['{k}']: shape={v.shape}, "
                                    f"dtype={v.dtype}, "
                                    f"range=[{v.min()}, {v.max()}]"
                                )
                            else:
                                console_logger.info(
                                    f"  obs['{k}']: {type(v).__name__} = {v!r}"
                                )

                    # Retry with reconnection on transient server drops.
                    for attempt in range(3):
                        try:
                            result = self._policy.infer(obs)
                            break
                        except Exception as e:
                            console_logger.warning(
                                f"Inference attempt {attempt + 1}/3 failed: {e}"
                            )
                            if attempt < 2:
                                console_logger.info("Reconnecting to server...")
                                time.sleep(2)
                                self._connect_policy()
                            else:
                                raise

                    # Result is dict with "actions" key → (N, 8) array.
                    actions = np.asarray(result["actions"], dtype=np.float32)
                    # Take up to open_loop_horizon steps.
                    action_buffer = list(actions[:cfg.open_loop_horizon])
                    buffer_idx = 0

                    console_logger.debug(
                        f"Step {self._step_count}: got {len(action_buffer)} "
                        f"actions from server (shape {actions.shape})"
                    )

                # Execute next action from buffer.
                action = action_buffer[buffer_idx]
                buffer_idx += 1
                self._step(action)

                # Log joint positions periodically.
                if self._step_count % 15 == 0:
                    joint_vals = [
                        f"{self._data.qpos[idx]:.3f}"
                        for idx in self._arm_qpos_indices
                    ]
                    gripper_val = (
                        f"{self._data.qpos[self._gripper1_qpos_idx]:.3f}"
                        if self._gripper1_qpos_idx >= 0
                        else "N/A"
                    )
                    console_logger.info(
                        f"Step {self._step_count}/{cfg.max_steps}: "
                        f"joints=[{', '.join(joint_vals)}] gripper={gripper_val}"
                    )

                # Maintain control frequency.
                elapsed = time.time() - t_loop
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            console_logger.info(
                f"Episode interrupted at step {self._step_count}."
            )

        duration = time.time() - t_start
        console_logger.info(
            f"Episode complete: {self._step_count} steps in {duration:.1f}s "
            f"({self._step_count / max(duration, 0.001):.1f} Hz effective)"
        )

        result = {
            "steps": self._step_count,
            "duration": duration,
            "effective_hz": self._step_count / max(duration, 0.001),
        }

        if self.config.record and self._frames:
            recording_path = self._save_recording()
            result["recording_path"] = str(recording_path)

        return result

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _save_recording(self) -> Path:
        """Save recorded frames as PNGs and an MP4 video.

        Returns:
            Path to the output directory.
        """
        from PIL import Image

        record_dir = Path(self.config.record_dir)
        frames_dir = record_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        for i, frame in enumerate(self._frames):
            path = frames_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(str(path))
            frame_paths.append(path)

        console_logger.info(
            f"Saved {len(frame_paths)} frames to {frames_dir}"
        )

        # Export MP4 videos for each camera angle.
        try:
            import imageio.v3 as iio

            for name, frames in [
                ("third_person", self._frames),
                ("wrist", self._frames_wrist),
                ("overview", self._frames_overview),
            ]:
                if not frames:
                    continue
                video_path = record_dir / f"episode_{name}.mp4"
                iio.imwrite(
                    str(video_path),
                    np.stack(frames),
                    fps=int(self.config.control_freq),
                    codec="libx264",
                )
                console_logger.info(f"Saved {name} video to {video_path}")
        except ImportError:
            console_logger.warning(
                "imageio[ffmpeg] not available, skipping video export."
            )
        except Exception as e:
            console_logger.warning(f"Video export failed: {e}")

        # Save joint trajectory for Blender replay.
        if self._traj_joint_angles:
            # Read arm base pose from the model for Blender rendering.
            arm_body_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "omx_f_base"
            )
            arm_pos = self._model.body_pos[arm_body_id].copy()
            arm_quat = self._model.body_quat[arm_body_id].copy()

            traj_data = {
                "joint_angles": np.array(self._traj_joint_angles),   # (N, 5)
                "gripper_angles": np.array(self._traj_gripper_angles),  # (N,)
                "fps": np.float64(self.config.control_freq),
                "arm_pos": arm_pos,       # (3,) xyz
                "arm_quat": arm_quat,     # (4,) wxyz
            }
            if self._traj_cup_poses:
                traj_data["cup_pose"] = np.array(self._traj_cup_poses)  # (N, 7)

            traj_path = record_dir / "trajectory.npz"
            np.savez(str(traj_path), **traj_data)
            console_logger.info(
                f"Saved trajectory ({len(self._traj_joint_angles)} frames) "
                f"to {traj_path}"
            )

        return record_dir

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up renderer and policy connection."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._overview_renderer is not None:
            self._overview_renderer.close()
            self._overview_renderer = None
        self._policy = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
