"""Robot-mounted camera for capturing the arm's point of view.

Injects a camera element into the robot arm's body hierarchy and provides
methods to capture snapshots, record trajectories, and produce video from
the robot's perspective.
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET

from pathlib import Path

import mujoco
import numpy as np

console_logger = logging.getLogger(__name__)

# Camera mount points on the OMX-F arm.
# Each entry maps to an arm body name with a local camera offset and orientation.
# fovy: field of view in degrees.
CAMERA_MOUNT_POINTS = {
    "end_effector": {
        "body": "link4",
        "pos": [0.0, 0.0, 0.025],  # On forearm, looking forward past wrist and claw.
        "quat": [0.707, 0.0, 0.707, 0.0],  # 90° around Y: maps body +X → camera -Z.
        "fovy": 100,
    },
    "wrist": {
        "body": "link5",
        "pos": [0.0, 0.0, 0.03],
        "quat": [0.707, 0.0, 0.707, 0.0],  # 90° around Y: maps body +X → camera -Z.
        "fovy": 110,
    },
    "base": {
        "body": "link0",
        "pos": [0.0, 0.0, 0.05],
        "quat": [1.0, 0.0, 0.0, 0.0],  # Looking forward from base.
        "fovy": 120,
    },
}


class RobotCamera:
    """Camera mounted on the robot arm for first-person view capture.

    Injects a <camera> element into the arm body via XML manipulation,
    loads the modified model, and provides snapshot/recording methods.
    """

    def __init__(
        self,
        scene_xml_path: str | Path,
        mount_point: str = "end_effector",
        width: int = 640,
        height: int = 480,
    ):
        """Initialize robot camera.

        Args:
            scene_xml_path: Path to scene XML (must include the arm).
            mount_point: Where to attach camera. One of "end_effector",
                "wrist", "base", or a custom body name.
            width: Camera image width.
            height: Camera image height.

        Raises:
            ValueError: If mount_point is not recognized and no matching
                body is found.
        """
        self.scene_xml_path = Path(scene_xml_path)
        self.mount_point = mount_point
        self.width = width
        self.height = height
        self._renderer = None
        self._model = None
        self._data = None

        # Resolve mount config.
        if mount_point in CAMERA_MOUNT_POINTS:
            self._mount_config = CAMERA_MOUNT_POINTS[mount_point]
        else:
            # Custom body name — use defaults.
            self._mount_config = {
                "body": mount_point,
                "pos": [0.0, 0.0, 0.02],
                "quat": [0.707, 0.0, 0.707, 0.0],
                "fovy": 60,
            }

        self._load_model()

    def _load_model(self) -> None:
        """Inject camera into scene XML and load the model."""
        tree = ET.parse(self.scene_xml_path)
        root = tree.getroot()

        target_body = self._mount_config["body"]
        cam_pos = self._mount_config["pos"]
        cam_quat = self._mount_config["quat"]
        cam_fovy = self._mount_config["fovy"]

        # Find the target body in the XML tree.
        body_elem = None
        for elem in root.iter("body"):
            if elem.get("name") == target_body:
                body_elem = elem
                break

        if body_elem is None:
            raise ValueError(
                f"Body '{target_body}' not found in scene XML. "
                f"Available mount points: {list(CAMERA_MOUNT_POINTS.keys())}"
            )

        # Remove any previously injected robot_cam (idempotency).
        for cam in body_elem.findall("camera"):
            if cam.get("name") == "robot_cam":
                body_elem.remove(cam)

        # Inject camera element.
        cam_elem = ET.SubElement(body_elem, "camera")
        cam_elem.set("name", "robot_cam")
        cam_elem.set(
            "pos", f"{cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f}"
        )
        cam_elem.set(
            "quat",
            f"{cam_quat[0]:.6f} {cam_quat[1]:.6f} "
            f"{cam_quat[2]:.6f} {cam_quat[3]:.6f}",
        )
        cam_elem.set("fovy", str(cam_fovy))

        # Write to temp file in scene dir (so meshdir resolves).
        scene_dir = self.scene_xml_path.parent
        fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=scene_dir)
        try:
            os.close(fd)
            tree.write(tmp_path, xml_declaration=True)
            self._model = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._renderer = mujoco.Renderer(
            self._model, height=self.height, width=self.width
        )

    def snapshot(self) -> np.ndarray:
        """Capture a single frame from the robot camera.

        Returns:
            HxWx3 uint8 pixel array.
        """
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera="robot_cam")
        return self._renderer.render()

    def save_snapshot(self, output_path: str | Path) -> Path:
        """Capture and save a single frame as PNG.

        Args:
            output_path: Where to save the PNG image.

        Returns:
            Path to the saved image.
        """
        from PIL import Image

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pixels = self.snapshot()
        img = Image.fromarray(pixels)
        img.save(str(output_path))
        console_logger.info(f"Saved robot camera snapshot to {output_path}")
        return output_path

    def _get_arm_joint_qpos_indices(self) -> list[int]:
        """Find qpos indices for the arm joints (joint1-joint5, gripper).

        Returns:
            List of qpos addresses for arm joints in order.
        """
        arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5",
            "gripper_joint_1", "gripper_joint_2",
        ]
        indices = []
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid >= 0:
                indices.append(int(self._model.jnt_qposadr[jid]))
        return indices

    def record_trajectory(
        self,
        joint_trajectory: list[list[float]],
        output_dir: str | Path,
        fps: int = 30,
    ) -> list[Path]:
        """Record frames along a joint trajectory.

        Sets arm joint positions directly via qpos and captures a frame
        at each step.

        Args:
            joint_trajectory: List of joint angle vectors. Each entry
                has values for [joint1, joint2, joint3, joint4, joint5]
                (gripper joints optional).
            output_dir: Directory to save frame PNGs.
            fps: Frames per second (used for naming, not timing).

        Returns:
            List of paths to saved frame PNGs.
        """
        from PIL import Image

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        arm_qpos_indices = self._get_arm_joint_qpos_indices()

        frame_paths = []
        for i, joint_pos in enumerate(joint_trajectory):
            # Set arm joint positions directly in qpos.
            n = min(len(joint_pos), len(arm_qpos_indices))
            for j in range(n):
                self._data.qpos[arm_qpos_indices[j]] = joint_pos[j]

            mujoco.mj_forward(self._model, self._data)

            pixels = self.snapshot()
            frame_path = output_dir / f"frame_{i:04d}.png"
            img = Image.fromarray(pixels)
            img.save(str(frame_path))
            frame_paths.append(frame_path)

        console_logger.info(
            f"Recorded {len(frame_paths)} trajectory frames to {output_dir}"
        )
        return frame_paths

    def record_orbit(
        self,
        output_dir: str | Path,
        n_frames: int = 60,
        fps: int = 30,
    ) -> list[Path]:
        """Record the robot's POV while cycling arm joints.

        Moves joint1 (base rotation) through a sweep to show what the
        end-effector camera sees during a simple orbit motion.

        Args:
            output_dir: Directory to save frame PNGs.
            n_frames: Number of frames to capture.
            fps: Frames per second (metadata only).

        Returns:
            List of paths to saved frame PNGs.
        """
        trajectory = []
        for i in range(n_frames):
            t = i / n_frames
            angle = np.sin(2 * np.pi * t) * 1.0  # +/- 1 radian sweep.
            # 5 arm joints: [base, shoulder, elbow, wrist_pitch, wrist_roll]
            trajectory.append([angle, 0.0, 0.0, 0.0, 0.0])

        return self.record_trajectory(trajectory, output_dir, fps=fps)

    @staticmethod
    def frames_to_video(
        frame_paths: list[Path],
        output_path: str | Path,
        fps: int = 30,
    ) -> Path:
        """Combine frame PNGs into an MP4 video.

        Uses imageio with ffmpeg backend. Falls back gracefully if
        imageio[ffmpeg] is not installed.

        Args:
            frame_paths: Ordered list of frame PNG paths.
            output_path: Output MP4 file path.
            fps: Video frame rate.

        Returns:
            Path to the saved video.

        Raises:
            ImportError: If imageio or ffmpeg plugin is not available.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import imageio.v3 as iio

            frames = [iio.imread(str(p)) for p in frame_paths]
            iio.imwrite(
                str(output_path),
                np.stack(frames),
                fps=fps,
                codec="libx264",
            )
            console_logger.info(
                f"Saved video ({len(frames)} frames, {fps}fps) to {output_path}"
            )
            return output_path
        except ImportError:
            raise ImportError(
                "Video export requires imageio[ffmpeg]. "
                "Install with: pip install imageio[ffmpeg]"
            )

    def close(self) -> None:
        """Clean up renderer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def capture_robot_view(
    scene_xml_path: str | Path,
    output_path: str | Path,
    mount_point: str = "end_effector",
    width: int = 640,
    height: int = 480,
) -> Path:
    """Convenience function to capture a single robot-camera snapshot.

    Args:
        scene_xml_path: Path to scene XML with arm already placed.
        output_path: Where to save the PNG.
        mount_point: Camera mount location ("end_effector", "wrist", "base").
        width: Image width.
        height: Image height.

    Returns:
        Path to the saved PNG.
    """
    with RobotCamera(
        scene_xml_path=scene_xml_path,
        mount_point=mount_point,
        width=width,
        height=height,
    ) as cam:
        return cam.save_snapshot(output_path)
