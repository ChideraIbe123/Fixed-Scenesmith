"""Hierarchical AI agent for robot arm placement in MuJoCo scenes."""


def __getattr__(name):
    if name == "ArmPlacementAgent":
        from scenesmith.arm_placement.arm_placement_agent import ArmPlacementAgent

        return ArmPlacementAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ArmPlacementAgent"]
