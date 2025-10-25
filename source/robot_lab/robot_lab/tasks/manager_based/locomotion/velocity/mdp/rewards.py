# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ==============================================================================
# Jump-Specific Rewards (FSM-Based for Performance)
# ==============================================================================

# Global state buffers for jump FSM (per-device, per-env-count)
_JUMP_STATE_BUFFERS = {}


def _get_jump_state_bufs(device: torch.device, num_envs: int) -> dict:
    """Get or create jump state buffers for finite state machine.

    Buffers track jump phases per environment:
    - air_prev: Was airborne in previous step?
    - took_off: Has robot ever been airborne this episode?
    - landed: Has robot completed first landing this episode?

    Args:
        device: Torch device for tensors.
        num_envs: Number of parallel environments.

    Returns:
        Dictionary of state buffers.
    """
    key = (device, int(num_envs))
    if key in _JUMP_STATE_BUFFERS:
        return _JUMP_STATE_BUFFERS[key]

    # Allocate buffers lazily on first access
    bufs = {
        "air_prev": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "took_off": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "landed": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }
    _JUMP_STATE_BUFFERS[key] = bufs
    return bufs


def reset_jump_state_on_termination(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset jump FSM state buffers on episode termination.

    Called automatically by environment on episode reset.

    Args:
        env: The learning environment.
        env_ids: Indices of environments to reset.
    """
    bufs = _get_jump_state_bufs(env.device, env.num_envs)
    bufs["air_prev"][env_ids] = False
    bufs["took_off"][env_ids] = False
    bufs["landed"][env_ids] = False


def _any_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Fast check if any foot is touching ground.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor.

    Returns:
        Boolean tensor indicating any foot contact. Shape: (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Check if any foot has significant force (> 1N threshold)
    forces_z = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    feet_in_contact = forces_z > 1.0
    return feet_in_contact.any(dim=1)


def _world_up_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract upright component (cos of tilt angle) from quaternion.

    Args:
        quat: Quaternion tensor (w, x, y, z format). Shape: (num_envs, 4).

    Returns:
        Cosine of tilt angle (1.0 = perfectly upright). Shape: (num_envs,).
    """
    # Extract quaternion components (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Z-axis component of rotation (upright = 1.0)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return up_z.clamp(-1.0, 1.0)


def jump_landing_win(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    success_radius: float = 0.3,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-time landing reward with shaped falloff inside success radius.

    Fires ONLY on the first landing step after takeoff. Shaped linearly
    from 1.0 (at target center) to 0.0 (at success_radius edge).
    Prevents reward exploitation from "dancing" on target zone.

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".
        success_radius: Radius for shaped reward in meters. Default: 0.3m.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor (0.0 most steps, shaped [0..1] on landing). Shape: (num_envs,).
    """
    bufs = _get_jump_state_bufs(env.device, env.num_envs)

    # Get robot position
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w

    # Get target position
    jump_command = env.command_manager._terms[command_name]
    target_w = jump_command.target_pos_w

    # Calculate horizontal distance to target
    err_xy = target_w[:, :2] - pos_w[:, :2]
    dist = torch.linalg.norm(err_xy, dim=1)

    # Check contact state
    contact_now = _any_foot_contact(env, sensor_cfg)
    air_now = ~contact_now

    # FSM state transitions
    just_took_off = air_now & ~bufs["air_prev"]
    bufs["took_off"] |= just_took_off

    just_landed = contact_now & bufs["air_prev"] & ~bufs["landed"]
    bufs["landed"] |= just_landed

    # Shaped reward: linear falloff inside success_radius
    inside = dist <= success_radius
    shaped = (1.0 - (dist / success_radius)).clamp(min=0.0)

    # Pay reward ONLY on landing step
    reward = torch.where(
        just_landed,
        torch.where(inside, shaped, torch.zeros_like(shaped)),
        torch.zeros_like(dist)
    )

    # Update state for next step
    bufs["air_prev"] = air_now

    return reward


def progress_to_target(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense progress reward based on velocity toward target.

    Computes dot product of velocity with direction to target.
    No history tracking needed - purely instantaneous.

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".
        asset_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor (progress rate in m/s). Shape: (num_envs,).
    """
    # Get robot state
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w
    lin_vel_w = asset.data.root_lin_vel_w

    # Get target position
    jump_command = env.command_manager._terms[command_name]
    target_w = jump_command.target_pos_w

    # Direction to target (normalized)
    err_xy = target_w[:, :2] - pos_w[:, :2]
    dist = torch.linalg.norm(err_xy, dim=1).clamp_min(1e-6)
    dir_xy = err_xy / dist.unsqueeze(1)

    # Velocity component toward target (dot product)
    progress_rate = (dir_xy * lin_vel_w[:, :2]).sum(dim=1)

    # Stage-aware clamping: gentler penalty in Stage 0 (0.3m jumps - still learning)
    # Stage 1+ (0.4m+) penalize moving away from target
    if hasattr(env, '_jump_current_stage_assignments'):
        stage = env._jump_current_stage_assignments
        in_first_stage = stage == 0
        progress_rate = torch.where(
            in_first_stage,
            progress_rate.clamp(min=0.0),  # Don't penalize moving away in Stage 0
            progress_rate                   # Full penalty in Stage 1+
        )

    return progress_rate


def pre_takeoff_ground_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    command_name: str = "jump_target",
) -> torch.Tensor:
    """Penalty for staying on ground before first takeoff.

    Encourages robot to jump quickly rather than walking.
    Applied in ALL stages (including Stage 0) to enforce jumping behavior.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor.
        command_name: Name of the jump target command (for stage detection).

    Returns:
        Penalty tensor (1.0 = penalize, 0.0 = no penalty). Shape: (num_envs,).
    """
    bufs = _get_jump_state_bufs(env.device, env.num_envs)

    # Check if on ground
    contact_now = _any_foot_contact(env, sensor_cfg)

    # Penalty condition: on ground AND haven't taken off yet (ALL STAGES)
    # This forces the robot to jump rather than walk toward the target
    cost = contact_now & (~bufs["took_off"])

    return cost.float()


def upright_on_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-time bonus for landing upright.

    Rewards maintaining upright orientation on first landing.
    Fires ONLY on the first landing step after takeoff.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Reward tensor (0.0 most steps, [0..1] on landing). Shape: (num_envs,).
    """
    bufs = _get_jump_state_bufs(env.device, env.num_envs)

    # Get robot orientation
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w

    # Check contact state
    contact_now = _any_foot_contact(env, sensor_cfg)

    # Detect first landing (same logic as jump_landing_win)
    just_landed = contact_now & bufs["air_prev"] & ~bufs["landed"]

    # Calculate upright bonus (1.0 = perfectly upright, 0.0 = horizontal/inverted)
    up_z = _world_up_from_quat(quat_w)
    bonus = up_z.clamp(min=0.0)

    # Pay bonus ONLY on first landing step
    reward = torch.where(just_landed, bonus, torch.zeros_like(bonus))

    return reward


def reward_airborne_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward for being airborne (both feet off ground).

    Encourages jumping behavior by rewarding flight time.
    Fires continuously while robot is in the air.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor.

    Returns:
        Reward tensor (1.0 when airborne, 0.0 when grounded). Shape: (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check contact for each foot
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    feet_in_contact = contact_forces > 1.0

    # Reward when NO feet in contact (fully airborne)
    airborne = ~feet_in_contact.any(dim=1)

    return airborne.float()


# ==============================================================================
# Jump-Specific Rewards (Legacy - Replaced by FSM-Based Above)
# ==============================================================================


def reach_target_zone(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    tolerance: float = 0.3,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Large positive reward for landing within target zone.

    Uses exponential kernel to reward landing close to the target.
    Only triggered when robot lands (feet make contact after flight).

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".
        tolerance: Success radius in meters. Default: 0.3m.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get robot asset
    asset: RigidObject = env.scene["robot"]

    # Get target position from command manager
    jump_command = env.command_manager._terms[command_name]
    target_pos_w = jump_command.target_pos_w

    # Calculate horizontal distance to target
    distance_to_target = torch.norm(
        asset.data.root_pos_w[:, :2] - target_pos_w[:, :2],
        dim=1
    )

    # Exponential kernel reward based on distance
    reward = torch.exp(-distance_to_target**2 / tolerance**2)

    # Only reward when feet are in contact (landing phase)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0] > 1.0
    any_foot_contact = feet_in_contact.any(dim=1).float()

    reward = reward * any_foot_contact

    return reward


def target_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    std: float = 0.5,
) -> torch.Tensor:
    """Shaped reward for reducing horizontal distance to target.

    Tracks distance reduction over time to guide robot toward target.

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".
        std: Standard deviation for exponential kernel. Default: 0.5.

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get robot asset
    asset: RigidObject = env.scene["robot"]

    # Get target position from command manager
    jump_command = env.command_manager._terms[command_name]
    target_pos_w = jump_command.target_pos_w

    # Calculate current horizontal distance to target
    current_distance = torch.norm(
        asset.data.root_pos_w[:, :2] - target_pos_w[:, :2],
        dim=1
    )

    # Store previous distance (initialize if needed)
    if not hasattr(env, '_prev_target_distance'):
        env._prev_target_distance = current_distance.clone()

    # Calculate progress (reduction in distance)
    progress = env._prev_target_distance - current_distance

    # Update previous distance for next step
    env._prev_target_distance = current_distance.clone()

    # Shaped reward based on progress
    reward = progress / std

    return reward


def flight_phase_quality(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    min_height: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proper takeoff with both feet leaving ground.

    Provides bonus for achieving minimum flight height.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        min_height: Minimum flight height for bonus reward in meters. Default: 0.1m.
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check if all feet are off the ground (airborne)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    contact_threshold = 1.0
    feet_in_contact = contact_forces > contact_threshold
    airborne = ~feet_in_contact.any(dim=1)

    # Get robot height (Z velocity as proxy for flight quality)
    asset: RigidObject = env.scene[asset_cfg.name]
    vertical_velocity = asset.data.root_lin_vel_w[:, 2]

    # Reward being airborne
    reward = airborne.float()

    # Bonus for upward velocity during flight
    reward += torch.clamp(vertical_velocity, min=0.0, max=min_height * 10.0)

    return reward


def landing_stability(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward stable landing with low angular velocity and upright orientation.

    Only applies during landing phase (when feet make contact).

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get robot asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check if feet are in contact (landing phase)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    feet_in_contact = contact_forces > 1.0
    any_foot_contact = feet_in_contact.any(dim=1).float()

    # Reward low angular velocity (stable landing)
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    ang_vel_reward = torch.exp(-ang_vel_penalty)

    # Reward upright orientation
    orientation_reward = 1.0 - torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    # Combine rewards, only during landing
    reward = (ang_vel_reward + orientation_reward) * any_foot_contact

    return reward


def smooth_flight_trajectory(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive rotation during flight phase.

    Encourages controlled flight without tumbling.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").

    Returns:
        Penalty tensor (negative values). Shape: (num_envs,).
    """
    # Get robot asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check if airborne
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    feet_in_contact = contact_forces > 1.0
    airborne = (~feet_in_contact.any(dim=1)).float()

    # Penalize angular velocity during flight
    ang_vel_magnitude = torch.norm(asset.data.root_ang_vel_b, dim=1)
    penalty = ang_vel_magnitude * airborne

    return penalty


def landing_orientation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_orientation: list[float] = [0, 0, 0, 1],
) -> torch.Tensor:
    """Reward landing with correct upright orientation.

    Uses quaternion distance to measure orientation error.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").
        target_orientation: Target quaternion [x, y, z, w]. Default: [0, 0, 0, 1] (upright).

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get robot asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check if feet are in contact (landing phase)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    feet_in_contact = contact_forces > 1.0
    any_foot_contact = feet_in_contact.any(dim=1).float()

    # Calculate orientation error (using projected gravity as simpler measure)
    # Perfect upright = projected_gravity_b = [0, 0, -1]
    # Error is when x and y components are non-zero
    orientation_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    orientation_reward = torch.exp(-orientation_error / 0.5**2)

    # Only reward during landing
    reward = orientation_reward * any_foot_contact

    return reward


def dual_foot_takeoff(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward taking off with both feet simultaneously.

    Ensures proper bipedal jumping by verifying both feet were in contact
    just before becoming airborne. Prevents single-leg hopping.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact forces for all feet (shape: [num_envs, history_len, num_feet])
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    contact_threshold = 1.0

    # Current state: check if feet are in contact NOW
    current_contacts = contact_forces[:, -1, :] > contact_threshold  # [num_envs, num_feet]
    current_airborne = ~current_contacts.any(dim=1)  # [num_envs]

    # Previous state: check if feet were in contact in previous step
    if contact_forces.shape[1] > 1:
        prev_contacts = contact_forces[:, -2, :] > contact_threshold  # [num_envs, num_feet]
        prev_on_ground = prev_contacts.any(dim=1)  # [num_envs]
        prev_both_feet_down = prev_contacts.all(dim=1)  # [num_envs]
    else:
        # First timestep, assume on ground
        prev_on_ground = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        prev_both_feet_down = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # Detect takeoff transition: was on ground, now airborne
    takeoff_transition = prev_on_ground & current_airborne

    # Reward only if BOTH feet were in contact before takeoff
    proper_takeoff = takeoff_transition & prev_both_feet_down

    return proper_takeoff.float()


def dual_foot_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    time_window: float = 0.1,
) -> torch.Tensor:
    """Reward landing with both feet within a short time window.

    Ensures proper bipedal landing by checking that both feet make contact
    within a specified time window. Prevents single-leg landings.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        time_window: Maximum time difference between feet landing in seconds. Default: 0.1s.

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact forces for all feet
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    contact_threshold = 1.0

    # Current state: check which feet are in contact NOW
    current_contacts = contact_forces[:, -1, :] > contact_threshold  # [num_envs, num_feet]
    current_airborne = ~current_contacts.any(dim=1)  # [num_envs]

    # Previous state
    if contact_forces.shape[1] > 1:
        prev_contacts = contact_forces[:, -2, :] > contact_threshold  # [num_envs, num_feet]
        prev_airborne = ~prev_contacts.any(dim=1)  # [num_envs]
    else:
        prev_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Detect landing transition: was airborne, now on ground
    landing_transition = prev_airborne & ~current_airborne

    # Initialize landing time tracker if not exists
    if not hasattr(env, '_foot_landing_times'):
        env._foot_landing_times = torch.full(
            (env.num_envs, len(sensor_cfg.body_ids)),
            -1.0,
            device=env.device
        )
        env._landing_step_counter = 0

    env._landing_step_counter += 1
    current_time = env._landing_step_counter * env.step_dt

    # Update landing times for feet that just touched down (VECTORIZED)
    # Detect new contact for all feet at once
    if contact_forces.shape[1] > 1:
        # new_foot_contact shape: [num_envs, num_feet]
        new_foot_contact = ~prev_contacts & current_contacts
    else:
        new_foot_contact = current_contacts

    # Update landing times for all feet in one operation
    # Where new contact detected, set to current_time, otherwise keep existing time
    env._foot_landing_times = torch.where(
        new_foot_contact,
        torch.full_like(env._foot_landing_times, current_time),
        env._foot_landing_times
    )

    # Reset landing times when airborne (all feet at once)
    env._foot_landing_times[current_airborne] = -1.0

    # Check if both feet have landed recently (within time window)
    both_feet_landed = (env._foot_landing_times > 0).all(dim=1)  # [num_envs]

    # Calculate time difference between first and last foot landing
    time_diff = torch.zeros(env.num_envs, device=env.device)
    valid_mask = both_feet_landed

    if valid_mask.any():
        min_times = env._foot_landing_times[valid_mask].min(dim=1)[0]
        max_times = env._foot_landing_times[valid_mask].max(dim=1)[0]
        time_diff[valid_mask] = max_times - min_times

    # Reward if both feet landed within the time window
    synchronized_landing = both_feet_landed & (time_diff < time_window)

    # Only reward during the landing transition to avoid giving reward every step
    reward = landing_transition & synchronized_landing

    return reward.float()


def penalize_ground_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize horizontal movement while feet are on the ground.

    Prevents walking by applying penalty when robot moves horizontally with feet touching ground.
    Encourages jumping as the only means of horizontal locomotion.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").
        velocity_threshold: Minimum velocity to penalize (m/s). Default: 0.05m/s.

    Returns:
        Penalty tensor (negative values). Shape: (num_envs,).
    """
    # Get robot asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check if feet are in contact with ground
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    feet_in_contact = contact_forces > 1.0
    any_foot_contact = feet_in_contact.any(dim=1)  # [num_envs]

    # Get horizontal velocity magnitude
    horizontal_vel = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # [num_envs]

    # Penalize if moving horizontally while on ground
    # Only apply penalty above threshold to allow small adjustments
    moving = horizontal_vel > velocity_threshold

    # Penalty proportional to velocity (squared for stronger effect)
    penalty = torch.where(
        any_foot_contact & moving,
        -horizontal_vel ** 2,
        torch.zeros_like(horizontal_vel)
    )

    return penalty


def reward_airborne_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_target",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward horizontal progress toward target ONLY when airborne.

    Ensures robot makes progress through jumping, not walking.

    Args:
        env: The learning environment.
        command_name: Name of the jump target command. Default: "jump_target".
        sensor_cfg: Configuration for the contact sensor. Default: SceneEntityCfg("contact_forces").
        asset_cfg: Configuration for the robot asset. Default: SceneEntityCfg("robot").

    Returns:
        Reward tensor. Shape: (num_envs,).
    """
    # Get robot asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get target position from command manager
    jump_command = env.command_manager._terms[command_name]
    target_pos_w = jump_command.target_pos_w

    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Check if airborne (no feet in contact)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    feet_in_contact = contact_forces > 1.0
    airborne = ~feet_in_contact.any(dim=1)  # [num_envs]

    # Calculate horizontal distance to target
    distance_to_target = torch.norm(
        asset.data.root_pos_w[:, :2] - target_pos_w[:, :2],
        dim=1
    )

    # Track previous distance if not initialized
    if not hasattr(env, '_prev_distance_to_target'):
        env._prev_distance_to_target = distance_to_target.clone()

    # Calculate progress (reduction in distance)
    distance_reduction = env._prev_distance_to_target - distance_to_target

    # Only reward progress made while airborne
    reward = torch.where(
        airborne,
        torch.clamp(distance_reduction, min=0.0) * 10.0,  # Scale up the reward
        torch.zeros_like(distance_reduction)
    )

    # Update previous distance for next step
    env._prev_distance_to_target = distance_to_target.clone()

    return reward
