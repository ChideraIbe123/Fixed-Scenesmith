"""Hierarchical AI agent for robot arm placement in MuJoCo scenes."""


def __getattr__(name):
    if name == "ArmPlacementAgent":
        from scenesmith.arm_placement.arm_placement_agent import ArmPlacementAgent

        return ArmPlacementAgent
    if name == "RobotCamera":
        from scenesmith.arm_placement.robot_camera import RobotCamera

        return RobotCamera
    if name == "capture_robot_view":
        from scenesmith.arm_placement.robot_camera import capture_robot_view

        return capture_robot_view
    if name == "VLACameraConfig":
        from scenesmith.arm_placement.vla_camera import VLACameraConfig

        return VLACameraConfig
    if name == "inject_vla_cameras":
        from scenesmith.arm_placement.vla_camera import inject_vla_cameras

        return inject_vla_cameras
    if name == "VLAControlConfig":
        from scenesmith.arm_placement.vla_control import VLAControlConfig

        return VLAControlConfig
    if name == "VLAControlLoop":
        from scenesmith.arm_placement.vla_control import VLAControlLoop

        return VLAControlLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArmPlacementAgent",
    "RobotCamera",
    "VLACameraConfig",
    "VLAControlConfig",
    "VLAControlLoop",
    "capture_robot_view",
    "inject_vla_cameras",
]
