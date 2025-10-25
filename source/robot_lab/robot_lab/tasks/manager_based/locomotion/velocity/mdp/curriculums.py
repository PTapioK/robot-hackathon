# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def jump_target_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "reach_target_zone",
    command_term_name: str = "jump_target",
    success_threshold: float = 0.7,
) -> torch.Tensor:
    """Curriculum learning for jump target commands with progressive difficulty.

    This curriculum progressively increases jump distance and height variation based on
    the robot's success rate in landing within the target zone.

    Curriculum Stages:
        - Stage 1: 0.3-0.5m horizontal, ±0.05m height (short level jumps)
        - Stage 2: 0.5-0.8m horizontal, ±0.10m height (medium jumps, slight elevation)
        - Stage 3: 0.8-1.2m horizontal, ±0.20m height (long jumps, moderate elevation)
        - Stage 4: 1.2-1.5m horizontal, ±0.30m height (advanced jumps, large elevation)

    Progression occurs when success rate > 70% over the last 50 episodes.

    Args:
        env: The learning environment.
        env_ids: Environment indices for which to update the curriculum.
        reward_term_name: Name of the reward term tracking jump success. Default: "reach_target_zone".
        command_term_name: Name of the jump command term. Default: "jump_target".
        success_threshold: Success rate threshold for progression. Default: 0.7 (70%).

    Returns:
        Current curriculum level tensor.
    """
    # Define curriculum stages
    CURRICULUM_STAGES = [
        {"horizontal_range": (0.3, 0.5), "height_range": (-0.05, 0.05)},   # Stage 1
        {"horizontal_range": (0.5, 0.8), "height_range": (-0.10, 0.10)},   # Stage 2
        {"horizontal_range": (0.8, 1.2), "height_range": (-0.20, 0.20)},   # Stage 3
        {"horizontal_range": (1.2, 1.5), "height_range": (-0.30, 0.30)},   # Stage 4
    ]

    # Initialize curriculum state on first call
    if not hasattr(env, '_jump_curriculum_level'):
        env._jump_curriculum_level = 0
        env._jump_success_history = []
        env._jump_episode_count = 0

    # Only update curriculum at episode boundaries
    if len(env_ids) > 0 and env.common_step_counter % env.max_episode_length == 0:
        # Get success metrics from the reward manager
        episode_sums = env.reward_manager._episode_sums[reward_term_name]

        # Calculate average success for this episode batch
        # Success is measured by the reach_target_zone reward
        avg_reward = torch.mean(episode_sums[env_ids]).item()

        # Track success (normalize by episode length to get average per step)
        success = avg_reward / env.max_episode_length_s

        # Add to history
        env._jump_success_history.append(success)
        env._jump_episode_count += 1

        # Check for progression every 50 episodes
        if len(env._jump_success_history) >= 50:
            # Calculate success rate over last 50 episodes
            recent_success_rate = sum(env._jump_success_history[-50:]) / 50.0

            # Check if we should progress to next stage
            if (recent_success_rate > success_threshold and
                env._jump_curriculum_level < len(CURRICULUM_STAGES) - 1):

                # Progress to next stage
                env._jump_curriculum_level += 1

                print(f"\n{'='*60}")
                print(f"JUMP CURRICULUM PROGRESSION!")
                print(f"Success rate: {recent_success_rate:.2%} (threshold: {success_threshold:.2%})")
                print(f"Advancing to Stage {env._jump_curriculum_level + 1}")
                print(f"New ranges:")
                new_stage = CURRICULUM_STAGES[env._jump_curriculum_level]
                print(f"  Horizontal: {new_stage['horizontal_range']}")
                print(f"  Height: {new_stage['height_range']}")
                print(f"{'='*60}\n")

                # Update command generator ranges
                jump_command = env.command_manager._terms[command_term_name]
                jump_command.horizontal_range = torch.tensor(
                    new_stage['horizontal_range'],
                    device=env.device
                )
                jump_command.height_range = torch.tensor(
                    new_stage['height_range'],
                    device=env.device
                )

                # Reset success history for new stage
                env._jump_success_history = []

    # Return current curriculum level as tensor
    return torch.tensor(env._jump_curriculum_level, device=env.device)
