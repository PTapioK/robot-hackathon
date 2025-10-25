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
    """Optimized curriculum learning for jump targets with minimal Python overhead.

    This curriculum uses vectorized GPU operations and reduced evaluation frequency
    for maximum training throughput. Multi-stage sampling prevents catastrophic forgetting.

    Key Optimizations:
        - Fully vectorized success tracking (scatter_add operations, no loops)
        - Evaluation every 50 iterations (reduced from 10)
        - Simplified success counters (no rolling buffers)
        - Minimal CPU-GPU synchronization

    Key Features:
        - 20 curriculum stages starting from 0.3m jumps
        - Samples equally from all unlocked stages (stage 0 to current_stage)
        - Progresses when success rate > 50% across ALL unlocked stages
        - Regresses when success rate <= 20% to prevent overstepping

    Curriculum Stages (20 total):
        - Stage 0:  0.30-0.40m horizontal, ±0.03m height (beginner jump)
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
    # Define 20 curriculum stages starting from 0.3m jumps
    # Removed Stage 0 (standing) and Stage 1 (0-0.15m) - too small for FSM rewards
    # Robot learns jumping mechanics from the start at 0.3m distance
    CURRICULUM_STAGES = [
        {"horizontal_range": (0.30, 0.40), "height_range": (-0.03, 0.03)},   # Stage 0:  Beginner jump (was Stage 2)
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

    # Initialize curriculum state on first call (PRE-COMPUTE EVERYTHING)
    if not hasattr(env, '_jump_curriculum_level'):
        env._jump_curriculum_level = 0  # Current highest unlocked stage
        env._jump_current_stage_assignments = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        env._jump_evaluation_cycle = 0

        # Simplified tracking: accumulate successes per stage (no rolling buffer)
        env._jump_stage_successes = torch.zeros(len(CURRICULUM_STAGES), device=env.device)
        env._jump_stage_total = torch.zeros(len(CURRICULUM_STAGES), device=env.device)

        # PRE-COMPUTE all stage ranges as GPU tensors (avoid Python lists/dicts)
        env._stage_h_min = torch.tensor([s['horizontal_range'][0] for s in CURRICULUM_STAGES], device=env.device)
        env._stage_h_max = torch.tensor([s['horizontal_range'][1] for s in CURRICULUM_STAGES], device=env.device)
        env._stage_z_min = torch.tensor([s['height_range'][0] for s in CURRICULUM_STAGES], device=env.device)
        env._stage_z_max = torch.tensor([s['height_range'][1] for s in CURRICULUM_STAGES], device=env.device)

        # Store stages list for command manager
        env._curriculum_stages = CURRICULUM_STAGES

        # Store configuration
        env._jump_rollouts_per_stage = rollouts_per_stage
        env._jump_success_threshold = success_threshold
        env._jump_regression_threshold = regression_threshold

        # Logging: Track episodes
        env._jump_total_episodes = 0

    # Update success tracking whenever environments reset
    if len(env_ids) > 0:
        # Get actual success metric from jump command manager (0.0 or 1.0 per env)
        # This is already computed: distance_to_target < success_radius
        jump_command = env.command_manager._terms[command_term_name]
        successes = jump_command.metrics["success_rate"][env_ids]  # [len(env_ids)]

        # Get stage indices for all resetting environments
        stage_indices = env._jump_current_stage_assignments[env_ids]  # [len(env_ids)]

        # Update success tracking (FULLY VECTORIZED using scatter_add - stays on GPU, no loops!)
        env._jump_stage_successes.scatter_add_(0, stage_indices.long(), successes)
        env._jump_stage_total.scatter_add_(0, stage_indices.long(), torch.ones_like(successes))

        # Track total episodes for logging (only actual resets, not every call)
        env._jump_total_episodes += len(env_ids)

    # Increment call counter for logging
    if not hasattr(env, '_jump_call_counter'):
        env._jump_call_counter = 0
    env._jump_call_counter += 1

    # Progress logging every 100 curriculum calls (roughly every few training iterations)
    # With 16K envs, curriculum is called very frequently, so we need a higher threshold
    if env._jump_call_counter % 100 == 0:
        num_unlocked_stages = env._jump_curriculum_level + 1

        # Calculate current success rate across all unlocked stages
        if torch.min(env._jump_stage_total[:num_unlocked_stages]) >= 1:
            stage_success_rates = env._jump_stage_successes[:num_unlocked_stages] / (env._jump_stage_total[:num_unlocked_stages] + 1e-8)
            overall_success = torch.mean(stage_success_rates).item()

            # Check if we have enough data for evaluation (10 episodes per stage minimum)
            min_episodes_per_stage = int(torch.min(env._jump_stage_total[:num_unlocked_stages]).item())
            min_needed = 10

            # Get current and next stage names
            current_stage = env._jump_curriculum_level
            next_stage = min(current_stage + 1, len(CURRICULUM_STAGES) - 1)

            # Data readiness indicator
            data_ready = "✓" if min_episodes_per_stage >= min_needed else f"{min_episodes_per_stage}/{min_needed}"

            print(f"\n[Curriculum] Stage {current_stage} → {next_stage} | "
                  f"Success: {overall_success:.1%} (need {success_threshold:.0%}) | "
                  f"Data: {data_ready}")

    # Evaluate curriculum every 500 calls (balanced frequency for 16K envs)
    if env._jump_call_counter % 500 == 0:
        num_unlocked_stages = env._jump_curriculum_level + 1

        # Check if we have enough data (at least 10 rollouts per unlocked stage)
        min_rollouts = torch.min(env._jump_stage_total[:num_unlocked_stages])
        if min_rollouts >= 10:
            # Calculate success rates (FULLY VECTORIZED - no loops!)
            stage_success_rates = env._jump_stage_successes[:num_unlocked_stages] / (env._jump_stage_total[:num_unlocked_stages] + 1e-8)

            # Overall success rate is average across all unlocked stages
            overall_success_rate = torch.mean(stage_success_rates).item()  # ONE sync point

            # Print detailed statistics before making decision
            print(f"\n{'='*70}")
            print(f"[Curriculum Evaluation]")
            print(f"{'='*70}")

            # Per-stage breakdown
            for i in range(num_unlocked_stages):
                total_eps = int(env._jump_stage_total[i].item())
                successful_eps = int(env._jump_stage_successes[i].item())
                success_rate = stage_success_rates[i].item()

                h_range = CURRICULUM_STAGES[i]['horizontal_range']
                z_range = CURRICULUM_STAGES[i]['height_range']

                # Format stage description
                if h_range[0] == 0 and h_range[1] == 0:
                    stage_desc = "Standing (0m)"
                else:
                    stage_desc = f"{h_range[0]:.1f}-{h_range[1]:.1f}m, ±{z_range[1]:.2f}m"

                status = "✓" if i <= env._jump_curriculum_level else " "
                current_marker = " ← CURRENT" if i == env._jump_curriculum_level else ""

                print(f"  [{status}] Stage {i}: {stage_desc:20s} | "
                      f"{successful_eps:6d}/{total_eps:6d} = {success_rate:5.1%}{current_marker}")

            print(f"\n  Overall Success: {overall_success_rate:.1%} (need {success_threshold:.0%} to advance)")

            # Progression/regression logic (minimal Python-side logic)
            if overall_success_rate > success_threshold and env._jump_curriculum_level < len(CURRICULUM_STAGES) - 1:
                env._jump_curriculum_level += 1
                # Reset counters when advancing
                env._jump_stage_successes.zero_()
                env._jump_stage_total.zero_()

                h_range = CURRICULUM_STAGES[env._jump_curriculum_level]['horizontal_range']
                print(f"\n  🎯 ADVANCING TO STAGE {env._jump_curriculum_level} ({h_range[0]:.1f}-{h_range[1]:.1f}m jumps)!")

            elif overall_success_rate <= regression_threshold and env._jump_curriculum_level > 0:
                env._jump_curriculum_level -= 1
                # Reset counters when regressing
                env._jump_stage_successes.zero_()
                env._jump_stage_total.zero_()
                print(f"\n  ⚠️  REGRESSING TO STAGE {env._jump_curriculum_level}")
            else:
                print(f"\n  → Continuing Stage {env._jump_curriculum_level} (need {success_threshold:.0%} success)")

            print(f"{'='*70}\n")

            env._jump_evaluation_cycle += 1

    # Assign stages to environments on reset (ULTRA-FAST - single randint call)
    if len(env_ids) > 0:
        num_unlocked_stages = env._jump_curriculum_level + 1

        # Sample stages uniformly from all unlocked stages (ONE GPU operation)
        sampled_stages = torch.randint(
            0, num_unlocked_stages,
            (len(env_ids),),
            dtype=torch.int32,
            device=env.device
        )

        # Assign stages to environments (ONE GPU write)
        env._jump_current_stage_assignments[env_ids] = sampled_stages

    # Return current curriculum level as tensor
    return torch.tensor(env._jump_curriculum_level, device=env.device)
