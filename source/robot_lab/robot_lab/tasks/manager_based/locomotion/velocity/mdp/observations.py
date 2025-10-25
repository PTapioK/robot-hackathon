# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


# ==============================================================================
# Jump-Specific Observations
# ==============================================================================


def target_position_rel(env: ManagerBasedRLEnv, command_name: str = "jump_target") -> torch.Tensor:
    """Returns 3D vector from robot base to target in robot body frame.

    This observation helps the robot understand where to jump in its local coordinate system.

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".

    Returns:
        Relative target position in body frame. Shape: (num_envs, 3).
    """
    # Get jump target command (already in body frame)
    target_pos_b = env.command_manager.get_command(command_name)
    return target_pos_b


def target_distance_height(env: ManagerBasedRLEnv, command_name: str = "jump_target") -> torch.Tensor:
    """Returns horizontal distance and height difference to target.

    This observation provides a compact representation of the jump challenge:
    - Horizontal distance: how far to jump (sqrt(dx^2 + dy^2))
    - Height difference: how high/low to jump (target_z - base_z)

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".

    Returns:
        Tensor containing [horizontal_distance, height_difference]. Shape: (num_envs, 2).
    """
    # Get jump target command in body frame
    target_pos_b = env.command_manager.get_command(command_name)

    # Calculate horizontal distance (XY plane)
    horizontal_distance = torch.norm(target_pos_b[:, :2], dim=1, keepdim=True)

    # Height difference is just the Z component
    height_difference = target_pos_b[:, 2:3]

    return torch.cat([horizontal_distance, height_difference], dim=1)


def is_airborne(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")) -> torch.Tensor:
    """Returns binary indicator of whether robot is airborne (all feet off ground).

    This observation helps the robot distinguish between ground contact and flight phases.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").

    Returns:
        Binary indicator: 1.0 if airborne, 0.0 if on ground. Shape: (num_envs, 1).
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get contact forces for feet (sensor configured to monitor feet)
    # Shape: (num_envs, num_feet)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]

    # Check if all feet have zero contact (airborne = all contacts below threshold)
    # Use small threshold to account for numerical noise
    contact_threshold = 1.0  # Newtons
    feet_in_contact = contact_forces > contact_threshold

    # Airborne if no feet are in contact
    airborne = (~feet_in_contact.any(dim=1)).float().unsqueeze(1)

    return airborne


def feet_position_to_target(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns relative position of both feet to target center.

    This observation helps the robot understand foot placement during landing approach.

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").

    Returns:
        Relative foot positions to target: [left_foot_xyz, right_foot_xyz]. Shape: (num_envs, 6).
    """
    # Get robot asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Get target position in world frame from command manager
    # Need to access the command manager's internal target position
    jump_command = env.command_manager._terms[command_name]
    target_pos_w = jump_command.target_pos_w

    # Get feet body indices (assumes body names contain "foot")
    # This should match the foot_link_name pattern in the environment config
    foot_body_names = asset_cfg.body_names
    if foot_body_names is None:
        # Fallback: try to find feet automatically
        foot_body_names = [".*foot.*"]

    foot_body_ids = asset_cfg.body_ids

    # Get feet positions in world frame
    # Shape: (num_envs, num_feet, 3)
    feet_pos_w = asset.data.body_pos_w[:, foot_body_ids, :]

    # Calculate relative position of each foot to target
    # Assuming 2 feet (left and right) for a biped
    # feet_pos_w shape: (num_envs, 2, 3)
    # target_pos_w shape: (num_envs, 3)
    feet_to_target = feet_pos_w - target_pos_w.unsqueeze(1)

    # Flatten to (num_envs, 6) - [left_x, left_y, left_z, right_x, right_y, right_z]
    feet_to_target_flat = feet_to_target.reshape(env.num_envs, -1)

    return feet_to_target_flat
