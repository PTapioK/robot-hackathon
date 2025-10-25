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
    regression_threshold: float = 0.3,
    rollouts_per_stage: int = 20,
) -> torch.Tensor:
    """Anti-forgetting curriculum learning for jump targets with progressive difficulty.

    This curriculum uses a multi-stage sampling strategy to prevent catastrophic forgetting.
    Instead of training only on the current stage, it samples equally from all unlocked stages,
    ensuring the robot maintains performance on easier jumps while learning harder ones.

    Key Features:
        - 10 curriculum stages with gradual progression
        - Samples equally from all unlocked stages (stage 0 to current_stage)
        - Progresses when success rate > 70% across ALL unlocked stages
        - Regresses when success rate <= 30% to prevent overstepping
        - Tracks per-stage statistics to identify weak areas

    Curriculum Stages (20 total):
        - Stage 0:  0.30-0.40m horizontal, ±0.03m height (beginner - very short)
        - Stage 1:  0.40-0.50m horizontal, ±0.05m height (beginner)
        - Stage 2:  0.50-0.60m horizontal, ±0.08m height (beginner+)
        - Stage 3:  0.60-0.70m horizontal, ±0.12m height (novice)
        - Stage 4:  0.70-0.85m horizontal, ±0.16m height (novice+)
        - Stage 5:  0.85-1.00m horizontal, ±0.22m height (intermediate)
        - Stage 6:  1.00-1.15m horizontal, ±0.28m height (intermediate+)
        - Stage 7:  1.15-1.30m horizontal, ±0.35m height (advanced)
        - Stage 8:  1.30-1.45m horizontal, ±0.43m height (advanced+)
        - Stage 9:  1.45-1.60m horizontal, ±0.52m height (skilled)
        - Stage 10: 1.60-1.75m horizontal, ±0.62m height (skilled+)
        - Stage 11: 1.75-1.90m horizontal, ±0.73m height (expert)
        - Stage 12: 1.90-2.05m horizontal, ±0.85m height (expert+)
        - Stage 13: 2.05-2.20m horizontal, ±0.98m height (master)
        - Stage 14: 2.20-2.35m horizontal, ±1.12m height (master+)
        - Stage 15: 2.35-2.50m horizontal, ±1.20m height (elite)
        - Stage 16: 2.50-2.65m horizontal, ±1.30m height (elite+)
        - Stage 17: 2.65-2.80m horizontal, ±1.40m height (superhuman)
        - Stage 18: 2.80-2.90m horizontal, ±1.45m height (superhuman+)
        - Stage 19: 2.90-3.00m horizontal, ±1.50m height (legendary)

    Args:
        env: The learning environment.
        env_ids: Environment indices for which to update the curriculum.
        reward_term_name: Name of the reward term tracking jump success. Default: "reach_target_zone".
        command_term_name: Name of the jump command term. Default: "jump_target".
        success_threshold: Success rate threshold for progression. Default: 0.7 (70%).
        regression_threshold: Success rate threshold for regression. Default: 0.3 (30%).
        rollouts_per_stage: Number of rollouts to collect per stage before evaluation. Default: 20.

    Returns:
        Current curriculum level tensor.
    """
    # Define 20 curriculum stages with gradual progression to superhuman performance
    CURRICULUM_STAGES = [
        {"horizontal_range": (0.30, 0.40), "height_range": (-0.03, 0.03)},   # Stage 0:  Beginner
        {"horizontal_range": (0.40, 0.50), "height_range": (-0.05, 0.05)},   # Stage 1:  Beginner
        {"horizontal_range": (0.50, 0.60), "height_range": (-0.08, 0.08)},   # Stage 2:  Beginner+
        {"horizontal_range": (0.60, 0.70), "height_range": (-0.12, 0.12)},   # Stage 3:  Novice
        {"horizontal_range": (0.70, 0.85), "height_range": (-0.16, 0.16)},   # Stage 4:  Novice+
        {"horizontal_range": (0.85, 1.00), "height_range": (-0.22, 0.22)},   # Stage 5:  Intermediate
        {"horizontal_range": (1.00, 1.15), "height_range": (-0.28, 0.28)},   # Stage 6:  Intermediate+
        {"horizontal_range": (1.15, 1.30), "height_range": (-0.35, 0.35)},   # Stage 7:  Advanced
        {"horizontal_range": (1.30, 1.45), "height_range": (-0.43, 0.43)},   # Stage 8:  Advanced+
        {"horizontal_range": (1.45, 1.60), "height_range": (-0.52, 0.52)},   # Stage 9:  Skilled
        {"horizontal_range": (1.60, 1.75), "height_range": (-0.62, 0.62)},   # Stage 10: Skilled+
        {"horizontal_range": (1.75, 1.90), "height_range": (-0.73, 0.73)},   # Stage 11: Expert
        {"horizontal_range": (1.90, 2.05), "height_range": (-0.85, 0.85)},   # Stage 12: Expert+
        {"horizontal_range": (2.05, 2.20), "height_range": (-0.98, 0.98)},   # Stage 13: Master
        {"horizontal_range": (2.20, 2.35), "height_range": (-1.12, 1.12)},   # Stage 14: Master+
        {"horizontal_range": (2.35, 2.50), "height_range": (-1.20, 1.20)},   # Stage 15: Elite
        {"horizontal_range": (2.50, 2.65), "height_range": (-1.30, 1.30)},   # Stage 16: Elite+
        {"horizontal_range": (2.65, 2.80), "height_range": (-1.40, 1.40)},   # Stage 17: Superhuman
        {"horizontal_range": (2.80, 2.90), "height_range": (-1.45, 1.45)},   # Stage 18: Superhuman+
        {"horizontal_range": (2.90, 3.00), "height_range": (-1.50, 1.50)},   # Stage 19: Legendary
    ]

    # Initialize curriculum state on first call
    if not hasattr(env, '_jump_curriculum_level'):
        env._jump_curriculum_level = 0  # Current highest unlocked stage
        env._jump_stage_rollouts = torch.zeros(len(CURRICULUM_STAGES), dtype=torch.int32, device=env.device)
        env._jump_stage_successes = torch.zeros(len(CURRICULUM_STAGES), device=env.device)
        env._jump_current_stage_assignments = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        env._jump_evaluation_cycle = 0
        env._jump_total_rollouts_this_cycle = 0

        # Store configuration
        env._jump_rollouts_per_stage = rollouts_per_stage
        env._jump_success_threshold = success_threshold
        env._jump_regression_threshold = regression_threshold

    # Only update curriculum at episode boundaries
    if len(env_ids) > 0 and env.common_step_counter % env.max_episode_length == 0:
        # Get success metrics from the reward manager
        episode_sums = env.reward_manager._episode_sums[reward_term_name]

        # Update per-environment statistics (VECTORIZED)
        # Get stage indices for all resetting environments
        stage_indices = env._jump_current_stage_assignments[env_ids]  # [len(env_ids)]

        # Calculate successes for all environments at once
        rewards = episode_sums[env_ids]  # [len(env_ids)]
        successes = rewards / env.max_episode_length_s  # [len(env_ids)]

        # Accumulate rollout counts per stage using scatter_add
        # This adds 1 to the bin corresponding to each environment's stage
        env._jump_stage_rollouts.scatter_add_(
            0,  # dim to scatter on
            stage_indices.long(),  # indices
            torch.ones_like(stage_indices, dtype=torch.int32)  # values to add
        )

        # Accumulate success scores per stage using scatter_add
        env._jump_stage_successes.scatter_add_(
            0,  # dim to scatter on
            stage_indices.long(),  # indices
            successes  # values to add
        )

        env._jump_total_rollouts_this_cycle += len(env_ids)

        # Calculate required rollouts for evaluation
        num_unlocked_stages = env._jump_curriculum_level + 1
        required_rollouts = num_unlocked_stages * rollouts_per_stage

        # Check if we've collected enough data for evaluation
        if env._jump_total_rollouts_this_cycle >= required_rollouts:
            # Calculate overall success rate across all unlocked stages
            total_successes = torch.sum(env._jump_stage_successes[:num_unlocked_stages]).item()
            total_rollouts = torch.sum(env._jump_stage_rollouts[:num_unlocked_stages]).item()
            overall_success_rate = total_successes / max(total_rollouts, 1)

            # Calculate per-stage success rates for logging
            stage_success_rates = []
            for i in range(num_unlocked_stages):
                stage_rollouts = env._jump_stage_rollouts[i].item()
                if stage_rollouts > 0:
                    stage_success_rates.append(
                        env._jump_stage_successes[i].item() / stage_rollouts
                    )
                else:
                    stage_success_rates.append(0.0)

            # Calculate current max ranges across all unlocked stages
            max_horizontal_range = max(CURRICULUM_STAGES[i]['horizontal_range'][1] for i in range(num_unlocked_stages))
            max_height_range = max(CURRICULUM_STAGES[i]['height_range'][1] for i in range(num_unlocked_stages))
            min_height_range = min(CURRICULUM_STAGES[i]['height_range'][0] for i in range(num_unlocked_stages))

            print(f"\n{'='*80}")
            print(f"JUMP CURRICULUM EVALUATION - Cycle {env._jump_evaluation_cycle}")
            print(f"{'='*80}")
            print(f"Overall Success Rate: {overall_success_rate:.2%}")
            print(f"Current Stage: {env._jump_curriculum_level} (unlocked stages: 0-{env._jump_curriculum_level})")
            print(f"Max Horizontal Distance: {max_horizontal_range:.2f}m")
            print(f"Height Range: {min_height_range:.2f}m to +{max_height_range:.2f}m")
            print(f"\nPer-Stage Performance:")
            for i in range(num_unlocked_stages):
                stage = CURRICULUM_STAGES[i]
                rollouts = env._jump_stage_rollouts[i].item()
                success_rate = stage_success_rates[i]
                print(f"  Stage {i}: {success_rate:.2%} ({rollouts} rollouts) "
                      f"- H: {stage['horizontal_range']}, Z: {stage['height_range']}")

            # Decide on progression or regression
            old_level = env._jump_curriculum_level

            if overall_success_rate > success_threshold:
                # Progress to next stage if not at max
                if env._jump_curriculum_level < len(CURRICULUM_STAGES) - 1:
                    env._jump_curriculum_level += 1
                    print(f"\n*** PROGRESSING to Stage {env._jump_curriculum_level} ***")
                    print(f"    Success rate {overall_success_rate:.2%} > threshold {success_threshold:.2%}")
                else:
                    print(f"\n*** MASTERY ACHIEVED - Already at maximum stage {env._jump_curriculum_level} ***")

            elif overall_success_rate <= regression_threshold:
                # Regress to previous stage if not at minimum
                if env._jump_curriculum_level > 0:
                    env._jump_curriculum_level -= 1
                    print(f"\n*** REGRESSING to Stage {env._jump_curriculum_level} ***")
                    print(f"    Success rate {overall_success_rate:.2%} <= threshold {regression_threshold:.2%}")
                else:
                    print(f"\n*** NEEDS MORE TRAINING - Already at minimum stage {env._jump_curriculum_level} ***")

            else:
                print(f"\n*** MAINTAINING Stage {env._jump_curriculum_level} ***")
                print(f"    Success rate {overall_success_rate:.2%} between thresholds")

            print(f"{'='*80}\n")

            # Reset counters for next evaluation cycle
            env._jump_stage_rollouts.zero_()
            env._jump_stage_successes.zero_()
            env._jump_total_rollouts_this_cycle = 0
            env._jump_evaluation_cycle += 1

    # Assign stages to environments on reset (mixed sampling strategy)
    # This happens whenever environments are reset
    if len(env_ids) > 0:
        num_unlocked_stages = env._jump_curriculum_level + 1

        # Sample stages uniformly from all unlocked stages
        # This ensures equal representation and prevents forgetting
        sampled_stages = torch.randint(
            0, num_unlocked_stages,
            (len(env_ids),),
            dtype=torch.int32,
            device=env.device
        )

        # Assign stages to environments
        env._jump_current_stage_assignments[env_ids] = sampled_stages

        # Update jump command ranges for each environment based on assigned stage
        jump_command = env.command_manager._terms[command_term_name]

        # We need to update ranges on a per-environment basis
        # For now, we'll use a mixed approach: set the command ranges to cover all unlocked stages
        # The actual sampling will be done in the command generator
        if not hasattr(jump_command, '_curriculum_stages'):
            jump_command._curriculum_stages = CURRICULUM_STAGES
            jump_command._stage_assignments = env._jump_current_stage_assignments

        # Update the command generator to use stage-specific ranges
        # We'll store the stage assignments in the command generator for use during resampling
        jump_command._stage_assignments = env._jump_current_stage_assignments
        jump_command._curriculum_stages = CURRICULUM_STAGES

    # Return current curriculum level as tensor
    return torch.tensor(env._jump_curriculum_level, device=env.device)
