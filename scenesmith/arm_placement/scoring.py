"""Structured scoring for arm placement critique.

Defines the ArmCritiqueWithScores dataclass that the critic agent produces
as structured output, following the same CritiqueWithScores pattern used
throughout SceneSmith.
"""

from dataclasses import dataclass

from scenesmith.agent_utils.scoring import CategoryScore, CritiqueWithScores


@dataclass
class ArmCritiqueWithScores(CritiqueWithScores):
    """Arm placement critique with 5 evaluation categories.

    Categories evaluate the quality of robot arm placement in the scene
    (5 categories, 0-50 total).
    """

    stability: CategoryScore
    """Arm placed on a flat, sturdy surface with secure base contact."""
    reachability: CategoryScore
    """Arm can reach nearby objects for manipulation tasks."""
    clearance: CategoryScore
    """Sufficient room for arm movement without collisions."""
    task_suitability: CategoryScore
    """Placement makes sense for practical manipulation tasks."""
    collision_free: CategoryScore
    """No intersections or penetrations with scene furniture."""

    def get_scores(self) -> list[CategoryScore]:
        """Return all arm placement critique category scores."""
        return [
            self.stability,
            self.reachability,
            self.clearance,
            self.task_suitability,
            self.collision_free,
        ]
