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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ArmPlacementAgent", "RobotCamera", "capture_robot_view"]
