# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Termination functions for locomotion environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def excessive_ground_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    max_contact_phases: int = 2,
) -> torch.Tensor:
    """Terminate episode if feet touch ground more than allowed number of times.

    This ensures clean jumping behavior - the robot should:
    1. Start with both feet on ground (phase 1)
    2. Take off and become airborne
    3. Land with both feet (phase 2)

    If feet touch ground a 3rd time (walking, hopping, stumbling), the episode terminates.
    This prevents walking or multiple hops and enforces single clean jumps.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        max_contact_phases: Maximum allowed ground contact phases. Default: 2 (initial + landing).

    Returns:
        Boolean tensor indicating which environments should terminate. Shape: (num_envs,).
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact forces for all feet
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    contact_threshold = 1.0

    # Initialize contact phase tracker if not exists
    if not hasattr(env, '_ground_contact_phase_count'):
        # Start with phase 1 (initial standing position)
        env._ground_contact_phase_count = torch.ones(env.num_envs, dtype=torch.int32, device=env.device)
        env._was_airborne_last_step = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Current state: check if any feet are in contact NOW
    current_contacts = contact_forces[:, -1, :] > contact_threshold  # [num_envs, num_feet]
    current_grounded = current_contacts.any(dim=1)  # [num_envs]
    current_airborne = ~current_grounded

    # Detect landing transition: was airborne → now grounded
    # This indicates a new ground contact phase
    landing_transition = env._was_airborne_last_step & current_grounded

    # Increment contact phase counter when landing occurs
    env._ground_contact_phase_count[landing_transition] += 1

    # Update airborne state for next timestep
    env._was_airborne_last_step = current_airborne

    # Terminate if exceeded maximum contact phases
    terminate = env._ground_contact_phase_count > max_contact_phases

    return terminate


def successful_landing(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_tolerance: float = 0.1,
    velocity_threshold: float = 0.1,
    stability_time: float = 0.5,
) -> torch.Tensor:
    """Terminate episode successfully when robot lands stably on target.

    Success criteria:
    1. Robot is within position_tolerance (default 10cm) of target horizontally
    2. All joint velocities are below velocity_threshold (robot is stable)
    3. Feet are in contact with ground
    4. Conditions maintained for stability_time seconds

    This is a "time_out=True" termination indicating successful task completion.

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").
        position_tolerance: Maximum distance from target for success in meters. Default: 0.1m.
        velocity_threshold: Maximum joint velocity for stable landing in rad/s. Default: 0.1 rad/s.
        stability_time: How long to maintain conditions before success in seconds. Default: 0.5s.

    Returns:
        Boolean tensor indicating which environments completed successfully. Shape: (num_envs,).
    """
    # Get robot asset
    from isaaclab.assets import Articulation
    asset: Articulation = env.scene[asset_cfg.name]

    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get target position from command manager
    jump_command = env.command_manager._terms[command_name]
    target_pos_w = jump_command.target_pos_w

    # Check if feet are in contact
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    contact_threshold = 1.0
    current_contacts = contact_forces[:, -1, :] > contact_threshold
    feet_on_ground = current_contacts.any(dim=1)

    # Check horizontal distance to target
    horizontal_distance = torch.norm(
        asset.data.root_pos_w[:, :2] - target_pos_w[:, :2],
        dim=1
    )
    on_target = horizontal_distance < position_tolerance

    # Check joint velocities (all joints should be nearly stationary)
    max_joint_vel = torch.max(torch.abs(asset.data.joint_vel), dim=1)[0]
    is_stable = max_joint_vel < velocity_threshold

    # All conditions must be met
    success_conditions_met = feet_on_ground & on_target & is_stable

    # Initialize stability timer if not exists
    if not hasattr(env, '_success_stability_timer'):
        env._success_stability_timer = torch.zeros(env.num_envs, device=env.device)

    # Update stability timer
    # Increment timer where conditions are met, reset where they're not
    env._success_stability_timer = torch.where(
        success_conditions_met,
        env._success_stability_timer + env.step_dt,
        torch.zeros_like(env._success_stability_timer)
    )

    # Success when stability maintained for required duration
    successful = env._success_stability_timer >= stability_time

    return successful


def ground_contact_outside_safe_zones(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance: float = 0.2,
) -> torch.Tensor:
    """Terminate if ground contact outside start/target safe zones (enforces single-jump)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager._terms[command_name]

    # Any body touching ground?
    any_contact = contact_sensor.data.net_forces_w_history[:, :, :, 2].max(dim=1)[0].max(dim=1)[0] > 1.0

    # Current average feet position vs start/target
    feet_avg = asset.data.body_pos_w[:, sensor_cfg.body_ids, :].mean(dim=1)
    near_start = torch.norm(feet_avg - cmd.start_pos_w, dim=1) < tolerance
    near_target = torch.norm(feet_avg - cmd.target_pos_w, dim=1) < tolerance

    return any_contact & ~(near_start | near_target)
