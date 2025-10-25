# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class JumpTargetCommand(CommandTerm):
    """Command generator that generates jump target positions with curriculum learning support.

    This command generator samples 3D target positions (x, y, z) within configurable ranges
    for training a humanoid robot to jump to target landing zones. It supports:
    - Horizontal distance curriculum (progressive jump distance)
    - Height variation (elevated platforms and lowered areas)
    - Visual target markers for debugging
    - Success-based resampling
    """

    cfg: "JumpTargetCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "JumpTargetCommandCfg", env: ManagerBasedEnv):
        """Initialize the jump target command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Store reference to terrain for ground height queries
        self.terrain = env.scene.terrain if hasattr(env.scene, "terrain") else None

        # Create buffers to store the command
        # -- target position in world frame: (x, y, z)
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # -- target position in robot base frame (updated each step)
        self.target_pos_b = torch.zeros_like(self.target_pos_w)
        # -- distance from robot to target (horizontal)
        self.target_distance = torch.zeros(self.num_envs, device=self.device)

        # Get ground reference height from terrain environment origins
        # This is the actual terrain height where robots spawn, not the robot's trunk height
        if self.terrain is not None and hasattr(self.terrain, "env_origins"):
            self.ground_height = self.terrain.env_origins[:, 2].clone()
        else:
            # Fallback: assume flat ground at z=0
            self.ground_height = torch.zeros(self.num_envs, device=self.device)

        # Initialize curriculum ranges (can be updated by curriculum function)
        self.horizontal_range = torch.tensor(cfg.horizontal_range, device=self.device)
        self.height_range = torch.tensor(cfg.height_range, device=self.device)

        # Metrics
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["landing_distance"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "JumpTargetCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHorizontal range: {self.horizontal_range.cpu().tolist()}\n"
        msg += f"\tHeight range: {self.height_range.cpu().tolist()}\n"
        msg += f"\tSuccess radius: {self.cfg.success_radius}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired jump target position in base frame. Shape is (num_envs, 3)."""
        return self.target_pos_b

    def _update_metrics(self):
        """Update metrics for landing accuracy."""
        # Calculate horizontal distance to target
        distance_to_target = torch.norm(
            self.robot.data.root_pos_w[:, :2] - self.target_pos_w[:, :2],
            dim=1
        )
        self.metrics["landing_distance"] = distance_to_target

        # Check if within success radius
        success = distance_to_target < self.cfg.success_radius
        self.metrics["success_rate"] = success.float()

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample jump target positions for the given environments.

        This method supports curriculum learning with per-environment stage assignments.
        If curriculum stages are assigned, each environment samples from its assigned stage's ranges.
        Otherwise, it falls back to the global horizontal_range and height_range.

        Args:
            env_ids: Environment indices for which to resample commands.
        """
        # Get current robot positions for the resampling environments
        robot_pos_w = self.robot.data.root_pos_w[env_ids].clone()

        # Extract robot's current yaw angle from quaternion (x, y, z, w format)
        robot_quat = self.robot.data.root_quat_w[env_ids]
        robot_yaw = torch.atan2(
            2.0 * (robot_quat[:, 3] * robot_quat[:, 2] + robot_quat[:, 0] * robot_quat[:, 1]),
            1.0 - 2.0 * (robot_quat[:, 1]**2 + robot_quat[:, 2]**2)
        )

        # Sample relative angle within ±40 degrees from robot's forward direction
        relative_angle = torch.empty(len(env_ids), device=self.device).uniform_(
            -0.6981,  # -40 degrees in radians
            0.6981    # +40 degrees in radians
        )

        # Combine robot yaw with relative angle for world-frame target direction
        theta = robot_yaw + relative_angle

        # Check if curriculum stages are assigned (for anti-forgetting curriculum)
        if hasattr(self, '_stage_assignments') and hasattr(self, '_curriculum_stages'):
            # Per-environment sampling from assigned curriculum stages (VECTORIZED)

            # Get stage indices for all environments being reset
            stage_indices = self._stage_assignments[env_ids]  # [len(env_ids)]

            # Pre-compute stage ranges as tensors if not already done
            if not hasattr(self, '_stage_h_min_tensor'):
                # Extract all stage ranges into tensors for fast indexing
                self._stage_h_min_tensor = torch.tensor(
                    [s['horizontal_range'][0] for s in self._curriculum_stages],
                    device=self.device
                )
                self._stage_h_max_tensor = torch.tensor(
                    [s['horizontal_range'][1] for s in self._curriculum_stages],
                    device=self.device
                )
                self._stage_z_min_tensor = torch.tensor(
                    [s['height_range'][0] for s in self._curriculum_stages],
                    device=self.device
                )
                self._stage_z_max_tensor = torch.tensor(
                    [s['height_range'][1] for s in self._curriculum_stages],
                    device=self.device
                )

            # Gather min/max values for assigned stages (vectorized lookup)
            h_min = self._stage_h_min_tensor[stage_indices]  # [len(env_ids)]
            h_max = self._stage_h_max_tensor[stage_indices]  # [len(env_ids)]
            z_min = self._stage_z_min_tensor[stage_indices]  # [len(env_ids)]
            z_max = self._stage_z_max_tensor[stage_indices]  # [len(env_ids)]

            # Sample from ranges (vectorized uniform sampling)
            horizontal_dist = h_min + (h_max - h_min) * torch.rand(len(env_ids), device=self.device)
            target_height = z_min + (z_max - z_min) * torch.rand(len(env_ids), device=self.device)

        else:
            # Fallback: sample from global ranges (backward compatibility)
            horizontal_dist = torch.empty(len(env_ids), device=self.device).uniform_(
                self.horizontal_range[0], self.horizontal_range[1]
            )
            target_height = torch.empty(len(env_ids), device=self.device).uniform_(
                self.height_range[0], self.height_range[1]
            )

        # Ensure minimum distance to prevent spawning inside humanoid
        min_distance = 0.5  # Minimum 0.5m to avoid spawning inside robot
        horizontal_dist = torch.clamp(horizontal_dist, min=min_distance)

        # Calculate target position in world frame
        self.target_pos_w[env_ids, 0] = robot_pos_w[:, 0] + horizontal_dist * torch.cos(theta)
        self.target_pos_w[env_ids, 1] = robot_pos_w[:, 1] + horizontal_dist * torch.sin(theta)
        # Use terrain ground reference height, not robot trunk height
        # Add small offset (0.05m) to ensure target is visible above terrain surface
        self.target_pos_w[env_ids, 2] = self.ground_height[env_ids] + target_height + 0.05

        # Store horizontal distance for metrics
        self.target_distance[env_ids] = horizontal_dist

    def _update_command(self):
        """Update target position in robot base frame."""
        # Transform target position from world to robot base frame
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat

        target_vec_w = self.target_pos_w - self.robot.data.root_pos_w[:, :3]
        self.target_pos_b[:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w),
            target_vec_w
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set visualization markers for jump targets."""
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
            # Set visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update visualization markers with current target positions."""
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return

        # Update target marker positions
        # Create a flat orientation (no rotation)
        zero_rot = torch.zeros(self.num_envs, device=self.device)
        target_quat = quat_from_euler_xyz(zero_rot, zero_rot, zero_rot)

        # Visualize the target
        self.target_visualizer.visualize(
            translations=self.target_pos_w,
            orientations=target_quat
        )


@configclass
class JumpTargetCommandCfg(CommandTermCfg):
    """Configuration for the jump target command generator."""

    class_type: type = JumpTargetCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    horizontal_range: tuple[float, float] = (0.3, 0.5)
    """Range for horizontal jump distance in meters. Default: (0.3, 0.5) for curriculum stage 1."""

    height_range: tuple[float, float] = (-0.05, 0.05)
    """Range for target height relative to starting position in meters.
    Negative values = lowered areas, Positive values = elevated platforms.
    Default: (-0.05, 0.05) for curriculum stage 1."""

    success_radius: float = 0.3
    """Radius tolerance for successful landing in meters. Default: 0.3m."""

    target_visualizer_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="/Visuals/Command/jump_target"
    )
    """Configuration for the target visualization marker. Shows landing zone as a flat box."""
