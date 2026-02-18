"""Main agent class for hierarchical robot arm placement in MuJoCo scenes.

Implements the Planner/Designer/Critic pattern from SceneSmith's BaseStatefulAgent
but purpose-built for MuJoCo XML + URDF + STL files instead of Drake SDF + RoomScene.
"""

import copy
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunConfig,
    Runner,
    SQLiteSession,
    function_tool,
)
from openai.types.shared import Reasoning

from scenesmith.arm_placement.mujoco_renderer import render_scene
from scenesmith.arm_placement.scene_analyzer import SceneDescription, analyze_scene
from scenesmith.arm_placement.scoring import ArmCritiqueWithScores
from scenesmith.arm_placement.tools.critic_tools import CriticTools, create_critic_tools
from scenesmith.arm_placement.tools.designer_tools import (
    DesignerTools,
    create_designer_tools,
)
from scenesmith.arm_placement.urdf_to_mjcf import convert_urdf_to_mjcf
from scenesmith.agent_utils.scoring import (
    CritiqueWithScores,
    compute_total_score,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
)
from scenesmith.prompts import ArmPlacementPrompts, prompt_registry

console_logger = logging.getLogger(__name__)


def _log_agent_usage(result: Any, agent_name: str) -> None:
    """Log token usage from an agent run."""
    try:
        usage = result.context_wrapper.usage
        cached = (
            usage.input_tokens_details.cached_tokens
            if usage.input_tokens_details
            else 0
        )
        reasoning = (
            usage.output_tokens_details.reasoning_tokens
            if usage.output_tokens_details
            else 0
        )
        console_logger.info(
            f"[{agent_name}] Token usage: "
            f"input={usage.input_tokens:,}, "
            f"output={usage.output_tokens:,}, "
            f"reasoning={reasoning:,}, "
            f"cached={cached:,}, "
            f"total={usage.total_tokens:,}, "
            f"requests={usage.requests}"
        )
    except Exception:
        pass  # Non-critical logging.


@dataclass
class ArmPlacementConfig:
    """Configuration for the arm placement agent."""

    model: str = "gpt-5.2"
    max_turns: int = 30
    max_critique_rounds: int = 3
    early_finish_min_score: int = 9
    reset_single_category_threshold: int = 2
    reset_total_sum_threshold: int = 5
    designer_max_turns: int = 20
    critic_max_turns: int = 10
    reasoning_effort_planner: str = "low"
    reasoning_effort_designer: str = "high"
    reasoning_effort_critic: str = "high"


@dataclass
class ArmPlacementResult:
    """Result of the arm placement pipeline."""

    success: bool
    scene_xml_path: Path | None = None
    arm_surface: str | None = None
    arm_position: list[float] | None = None
    arm_rotation: float = 0.0
    final_scores: ArmCritiqueWithScores | None = None
    output_dir: Path | None = None
    error: str | None = None


class ArmPlacementAgent:
    """Hierarchical agent for placing a robot arm in a MuJoCo scene.

    Uses the Planner/Designer/Critic pattern:
    - Planner: Orchestrates the workflow with low reasoning effort
    - Designer: Places and adjusts the arm with high reasoning effort
    - Critic: Evaluates placement quality with structured scoring

    Checkpoint management uses XML strings (simpler than RoomScene state dicts).
    """

    def __init__(
        self,
        scene_xml_path: Path,
        arm_urdf_path: Path,
        output_dir: Path,
        cfg: ArmPlacementConfig | None = None,
    ):
        self.scene_xml_path = scene_xml_path
        self.arm_urdf_path = arm_urdf_path
        self.output_dir = output_dir
        self.cfg = cfg or ArmPlacementConfig()

        # Create output directory.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prompt registry.
        self.prompt_registry = prompt_registry

        # Checkpoint state (N-1/N pattern).
        self.previous_scene_checkpoint: str | None = None
        self.scene_checkpoint: str | None = None
        self.previous_checkpoint_scores: CritiqueWithScores | None = None
        self.checkpoint_scores: CritiqueWithScores | None = None
        self.previous_scores: CritiqueWithScores | None = None

        # Agent references (set during place_arm).
        self.designer: Agent | None = None
        self.critic: Agent | None = None
        self.planner: Agent | None = None
        self.designer_tools_obj: DesignerTools | None = None
        self.critic_tools_obj: CriticTools | None = None

        # Sessions.
        self.designer_session = SQLiteSession(
            session_id="designer",
            db_path=self.output_dir / "designer.db",
        )
        self.critic_session = SQLiteSession(
            session_id="critic",
            db_path=self.output_dir / "critic.db",
        )

    async def place_arm(self) -> ArmPlacementResult:
        """Main entry point — runs the full arm placement pipeline.

        1. Convert URDF to MJCF
        2. Analyze scene
        3. Create agents
        4. Run planner (orchestrates designer + critic)
        5. Return result

        Returns:
            ArmPlacementResult with placement details and scores.
        """
        console_logger.info("=" * 60)
        console_logger.info("ARM PLACEMENT AGENT STARTING")
        console_logger.info("=" * 60)

        # Step 1: Convert URDF to MJCF.
        console_logger.info("Step 1: Converting URDF to MJCF...")
        try:
            arm_mjcf_path = convert_urdf_to_mjcf(
                urdf_path=self.arm_urdf_path,
                output_dir=self.output_dir,
            )
        except Exception as e:
            console_logger.error(f"URDF conversion failed: {e}")
            return ArmPlacementResult(
                success=False, error=f"URDF conversion failed: {e}"
            )

        # Step 2: Analyze scene.
        console_logger.info("Step 2: Analyzing scene...")
        try:
            scene_desc = analyze_scene(self.scene_xml_path)
        except Exception as e:
            console_logger.error(f"Scene analysis failed: {e}")
            return ArmPlacementResult(
                success=False, error=f"Scene analysis failed: {e}"
            )

        console_logger.info(f"Scene: {scene_desc.room_type}, "
                          f"{len(scene_desc.furniture)} furniture, "
                          f"{len(scene_desc.surfaces)} surfaces")

        if not scene_desc.surfaces:
            return ArmPlacementResult(
                success=False,
                error="No suitable placement surfaces found in the scene",
            )

        # Step 3: Create agents.
        console_logger.info("Step 3: Creating agents...")
        designer_tools, self.designer_tools_obj = create_designer_tools(
            scene_xml_path=self.scene_xml_path,
            arm_mjcf_path=arm_mjcf_path,
            output_dir=self.output_dir,
            scene_description=scene_desc,
        )

        critic_tools, self.critic_tools_obj = create_critic_tools(
            output_dir=self.output_dir,
            scene_description=scene_desc,
            get_current_scene_path=lambda: self.designer_tools_obj.current_scene_xml,
            get_arm_state=lambda: {
                "placed": self.designer_tools_obj.arm_placed,
                "position": self.designer_tools_obj.arm_position,
                "rotation": self.designer_tools_obj.arm_rotation,
                "surface": self.designer_tools_obj.arm_surface,
            },
        )

        self.designer = self._create_designer_agent(designer_tools)
        self.critic = self._create_critic_agent(critic_tools)
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent(planner_tools, scene_desc)

        # Step 4: Run planner.
        console_logger.info("Step 4: Running planner agent...")
        runner_instruction = self.prompt_registry.get_prompt(
            ArmPlacementPrompts.PLANNER_RUNNER_INSTRUCTION,
        )

        try:
            result = await Runner.run(
                starting_agent=self.planner,
                input=runner_instruction,
                max_turns=self.cfg.max_turns,
            )
            _log_agent_usage(result, "PLANNER")
            log_agent_response(
                response=result.final_output, agent_name="PLANNER (FINAL)"
            )
        except Exception as e:
            console_logger.error(f"Planner execution failed: {e}")
            return ArmPlacementResult(
                success=False, error=f"Planner execution failed: {e}"
            )

        # Step 5: Collect results.
        console_logger.info("Step 5: Collecting results...")
        dt = self.designer_tools_obj

        # Save final scores if available.
        if self.checkpoint_scores is not None:
            scores_dict = scores_to_dict(self.checkpoint_scores)
            scores_path = self.output_dir / "scores.yaml"
            with open(scores_path, "w") as f:
                yaml.dump(scores_dict, f, default_flow_style=False, sort_keys=False)
            console_logger.info(f"Final scores saved to: {scores_path}")

        placement_result = ArmPlacementResult(
            success=dt.arm_placed,
            scene_xml_path=dt.current_scene_xml if dt.arm_placed else None,
            arm_surface=dt.arm_surface,
            arm_position=dt.arm_position,
            arm_rotation=dt.arm_rotation,
            final_scores=(
                self.checkpoint_scores
                if isinstance(self.checkpoint_scores, ArmCritiqueWithScores)
                else None
            ),
            output_dir=self.output_dir,
        )

        console_logger.info("=" * 60)
        console_logger.info("ARM PLACEMENT AGENT COMPLETE")
        if placement_result.success:
            console_logger.info(f"  Surface: {placement_result.arm_surface}")
            console_logger.info(f"  Position: {placement_result.arm_position}")
            console_logger.info(f"  Rotation: {placement_result.arm_rotation}")
            console_logger.info(f"  Output: {placement_result.scene_xml_path}")
        else:
            console_logger.info(f"  Failed: {placement_result.error}")
        console_logger.info("=" * 60)

        return placement_result

    def _create_designer_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the designer agent."""
        return Agent(
            name="arm_placement_designer",
            model=self.cfg.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                ArmPlacementPrompts.DESIGNER_AGENT,
            ),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.cfg.reasoning_effort_designer),
            ),
        )

    def _create_critic_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the critic agent with structured output."""
        return Agent(
            name="arm_placement_critic",
            model=self.cfg.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                ArmPlacementPrompts.CRITIC_AGENT,
            ),
            output_type=ArmCritiqueWithScores,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.cfg.reasoning_effort_critic),
                tool_choice="observe_scene",
            ),
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], scene_desc: SceneDescription
    ) -> Agent:
        """Create the planner agent."""
        return Agent(
            name="arm_placement_planner",
            model=self.cfg.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                ArmPlacementPrompts.PLANNER_AGENT,
                scene_description=scene_desc.to_text(),
                max_critique_rounds=self.cfg.max_critique_rounds,
                reset_single_category_threshold=self.cfg.reset_single_category_threshold,
                reset_total_sum_threshold=self.cfg.reset_total_sum_threshold,
                early_finish_min_score=self.cfg.early_finish_min_score,
            ),
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.cfg.reasoning_effort_planner),
                parallel_tool_calls=False,
            ),
        )

    def _create_planner_tools(self) -> list[FunctionTool]:
        """Create planner tools for coordinating designer and critic."""

        @function_tool
        async def request_initial_design() -> str:
            """Request the designer to create the initial arm placement.

            The designer will analyze the scene and place the arm on the
            best available surface.

            Returns:
                Designer's report of initial placement.
            """
            return await self._request_initial_design_impl()

        @function_tool
        async def request_critique() -> str:
            """Request the critic to evaluate the current arm placement.

            The critic will examine the placement and provide scores across
            5 categories: Stability, Reachability, Clearance, Task Suitability,
            and Collision-Free.

            Returns:
                Critic's evaluation with scores and improvement suggestions.
            """
            return await self._request_critique_impl()

        @function_tool
        async def request_design_change(instruction: str) -> str:
            """Request the designer to adjust the arm placement.

            Based on the critic's feedback, provide clear instructions about
            what to change.

            Args:
                instruction: Specific changes to make based on critique feedback.

            Returns:
                Designer's report of what was changed.
            """
            return await self._request_design_change_impl(instruction)

        @function_tool
        async def reset_scene_to_checkpoint(reason: str) -> str:
            """Reset scene to previous iteration state when changes made it worse.

            Use when designer's changes caused significant score degradation.

            Args:
                reason: Explanation of why you're resetting.

            Returns:
                Confirmation with checkpoint details.
            """
            return self._perform_checkpoint_reset(reason)

        tools = [request_initial_design]
        if self.cfg.max_critique_rounds > 0:
            tools.extend([
                request_critique,
                request_design_change,
                reset_scene_to_checkpoint,
            ])

        return tools

    async def _request_initial_design_impl(self) -> str:
        """Run designer agent for initial arm placement."""
        console_logger.info("Tool called: request_initial_design")

        instruction = self.prompt_registry.get_prompt(
            ArmPlacementPrompts.DESIGNER_INITIAL_INSTRUCTION,
        )

        result = await Runner.run(
            starting_agent=self.designer,
            input=instruction,
            session=self.designer_session,
            max_turns=self.cfg.designer_max_turns,
        )
        _log_agent_usage(result, "DESIGNER (INITIAL)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (INITIAL)"
            )

        return result.final_output

    async def _request_critique_impl(self) -> str:
        """Run critic agent to evaluate current placement."""
        console_logger.info("Tool called: request_critique")

        critique_instruction = self.prompt_registry.get_prompt(
            ArmPlacementPrompts.CRITIC_RUNNER_INSTRUCTION,
        )

        result = await Runner.run(
            starting_agent=self.critic,
            input=critique_instruction,
            session=self.critic_session,
            max_turns=self.cfg.critic_max_turns,
        )
        _log_agent_usage(result, "CRITIC")

        # Parse structured output.
        response = result.final_output_as(CritiqueWithScores)

        # Log critique and scores.
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="ARM PLACEMENT SCORES")

        # Save scores to YAML.
        if self.critic_tools_obj and self.critic_tools_obj.last_render_dir:
            scores_dict = scores_to_dict(response)
            scores_path = self.critic_tools_obj.last_render_dir / "scores.yaml"
            with open(scores_path, "w") as f:
                yaml.dump(scores_dict, f, default_flow_style=False, sort_keys=False)

        # Compute score deltas.
        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )

        # Shift checkpoints (N-1/N pattern).
        self.previous_scene_checkpoint = self.scene_checkpoint
        self.previous_checkpoint_scores = self.checkpoint_scores

        # Save new checkpoint.
        if self.designer_tools_obj:
            self.scene_checkpoint = self.designer_tools_obj.get_current_scene_xml()
        self.checkpoint_scores = response
        self.previous_scores = response

        return response.critique + score_change_msg

    async def _request_design_change_impl(self, instruction: str) -> str:
        """Run designer agent with critique feedback."""
        console_logger.info("Tool called: request_design_change")

        full_instruction = self.prompt_registry.get_prompt(
            ArmPlacementPrompts.DESIGNER_CRITIQUE_INSTRUCTION,
            instruction=instruction,
        )

        result = await Runner.run(
            starting_agent=self.designer,
            input=full_instruction,
            session=self.designer_session,
            max_turns=self.cfg.designer_max_turns,
        )
        _log_agent_usage(result, "DESIGNER (CHANGE)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (CHANGE)"
            )

        return result.final_output

    def _perform_checkpoint_reset(self, reason: str) -> str:
        """Reset scene to previous checkpoint."""
        console_logger.info(f"Resetting to checkpoint. Reason: {reason}")

        if (
            self.previous_scene_checkpoint is None
            or self.previous_checkpoint_scores is None
        ):
            console_logger.warning("No previous checkpoint available.")
            return (
                "ERROR: No previous checkpoint available to reset to. "
                "You must call request_critique() at least twice."
            )

        # Restore scene XML.
        if self.designer_tools_obj:
            self.designer_tools_obj.restore_scene_xml(self.previous_scene_checkpoint)

        # Reset scores.
        self.checkpoint_scores = copy.deepcopy(self.previous_checkpoint_scores)
        self.previous_scores = copy.deepcopy(self.previous_checkpoint_scores)

        # Shift checkpoint.
        self.scene_checkpoint = self.previous_scene_checkpoint

        # Build scores string.
        scores_parts = [
            f"{score.name}={score.grade}"
            for score in self.checkpoint_scores.get_scores()
        ]
        scores_str = ", ".join(scores_parts)

        return (
            f"Scene reset to state from 2 iterations ago.\n"
            f"Checkpoint scores: {scores_str}\n"
            f"Reset reason: {reason}\n"
            "Continue with design improvements from this restored state."
        )
