# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Booster T1 Jumping Navigation Environment Configuration

This environment trains the Booster T1 humanoid robot to jump to target landing zones
with an anti-forgetting curriculum learning system that prevents catastrophic forgetting.

Key Features:
- Jump target command system for position-based navigation
- 20-stage anti-forgetting curriculum from 0.3m to 3.0m jump distances
- Height variation from -1.5m to +1.5m (superhuman performance targets)
- Multi-stage sampling: trains on ALL unlocked stages simultaneously
- Adaptive progression: advances at >70% success, regresses at ≤30% success
- Dual-foot takeoff and landing enforcement for proper bipedal mechanics
- Reduced torque penalties to enable explosive jumping movements
- Jump-specific observations and rewards
- Per-stage performance tracking and detailed logging
"""

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg,
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.booster import BOOSTER_T1_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class BoosterT1JumpEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Booster T1 jumping navigation task."""

    base_link_name = "Trunk"
    foot_link_name = ".*_foot_link"

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # ======================================================================================
        # Scene Configuration
        # ======================================================================================
        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Use rough terrain for jump training
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG

        # ======================================================================================
        # Commands Configuration - Replace velocity with jump targets
        # ======================================================================================
        self.commands = CommandsCfg()
        self.commands.jump_target = mdp.JumpTargetCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 15.0),  # Resample target every 10-15 seconds
            debug_vis=True,  # Enable target visualization
            # Initial ranges (Stage 0) - will be dynamically updated by curriculum
            horizontal_range=(0.3, 0.4),  # Stage 0: very short jumps
            height_range=(-0.03, 0.03),  # Stage 0: nearly level jumps
            success_radius=0.3,  # 30cm tolerance for landing
        )

        # ======================================================================================
        # Observations Configuration
        # ======================================================================================

        # Add jump-specific observations
        self.observations.policy.target_position_rel = ObsTerm(
            func=mdp.target_position_rel,
            params={"command_name": "jump_target"},
        )
        self.observations.policy.target_distance_height = ObsTerm(
            func=mdp.target_distance_height,
            params={"command_name": "jump_target"},
        )
        self.observations.policy.is_airborne = ObsTerm(
            func=mdp.is_airborne,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])},
        )
        self.observations.policy.feet_position_to_target = ObsTerm(
            func=mdp.feet_position_to_target,
            params={
                "command_name": "jump_target",
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            },
        )

        # Remove velocity-tracking observations (not needed for jumping)
        self.observations.policy.velocity_commands = None

        # Keep existing observations with adjusted scales
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None  # Disable - not tracking velocity
        # Keep height_scan enabled for rough terrain navigation

        # ======================================================================================
        # Actions Configuration
        # ======================================================================================
        # Keep action configuration from parent (allows for explosive movements)
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ======================================================================================
        # Rewards Configuration
        # ======================================================================================

        # --- JUMP-SPECIFIC REWARDS (NEW) ---

        self.rewards.reach_target_zone = RewTerm(
            func=mdp.reach_target_zone,
            weight=100.0,
            params={
                "command_name": "jump_target",
                "tolerance": 0.3,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )

        self.rewards.target_progress = RewTerm(
            func=mdp.target_progress,
            weight=2.0,
            params={"command_name": "jump_target", "std": 0.5},
        )

        self.rewards.flight_phase_quality = RewTerm(
            func=mdp.flight_phase_quality,
            weight=5.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "min_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.landing_stability = RewTerm(
            func=mdp.landing_stability,
            weight=8.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.smooth_flight_trajectory = RewTerm(
            func=mdp.smooth_flight_trajectory,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.landing_orientation = RewTerm(
            func=mdp.landing_orientation,
            weight=3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
                "target_orientation": [0, 0, 0, 1],
            },
        )

        self.rewards.dual_foot_takeoff = RewTerm(
            func=mdp.dual_foot_takeoff,
            weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )

        self.rewards.dual_foot_landing = RewTerm(
            func=mdp.dual_foot_landing,
            weight=12.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "time_window": 0.1,  # Both feet must land within 0.1 seconds
            },
        )

        # --- MODIFIED REWARDS (Reduced penalties for jumping) ---

        # General penalties
        self.rewards.is_terminated.weight = -200.0  # Keep termination penalty high

        # Root penalties (adjusted for jumping)
        self.rewards.lin_vel_z_l2.weight = 0  # Don't penalize vertical velocity (needed for jumping!)
        self.rewards.ang_vel_xy_l2.weight = -0.1  # Keep tumbling penalty
        self.rewards.flat_orientation_l2.weight = -0.1  # Reduce (allow some tilt during flight)
        self.rewards.base_height_l2.weight = 0  # Disable (height varies during jump)
        self.rewards.body_lin_acc_l2.weight = 0  # Disable (allow rapid acceleration)
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties (GREATLY reduced for explosive jumping)
        self.rewards.joint_torques_l2.weight = -1e-8  # Reduced from -3e-7 (allow high torques)
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]
        self.rewards.joint_vel_l2.weight = 0  # Disable (allow fast movements)
        self.rewards.joint_acc_l2.weight = -5e-9  # Reduced from -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_Hip_.*", ".*_Knee_.*"]

        # Joint deviation penalties (reduced by 50%)
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.005, [".*_Hip_Yaw", ".*_Hip_Roll"])
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_arms_l1", -0.025, [".*_Shoulder_.*", ".*_Elbow_.*"]
        )
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_torso_l1", -0.05, ["Waist"])

        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_without_cmd.weight = 0
        self.rewards.joint_pos_penalty.weight = -0.5  # Reduced from -1.0
        self.rewards.joint_mirror.weight = 0

        # Action penalties (reduced for jumping)
        self.rewards.action_rate_l2.weight = -0.01  # Reduced from -0.075 (allow quick changes)
        self.rewards.action_mirror.weight = 0

        # Contact sensor penalties
        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0

        # --- DISABLED WALKING-SPECIFIC REWARDS ---
        # These don't apply to jumping task
        self.rewards.track_lin_vel_xy_exp = None  # No velocity tracking
        self.rewards.track_ang_vel_z_exp = None  # No angular velocity tracking
        self.rewards.feet_air_time = None  # We WANT air time for jumping!
        self.rewards.feet_slide = None  # Not relevant for jumping
        self.rewards.feet_contact = None  # Different contact pattern for jumping
        self.rewards.feet_contact_without_cmd = None
        self.rewards.feet_stumble = None
        self.rewards.feet_height = None
        self.rewards.feet_height_body = None

        # Keep upward reward (encourages staying upright)
        self.rewards.upward.weight = 1.0

        # Manually disable zero-weight rewards (can't use parent's method with None values)
        if self.__class__.__name__ == "BoosterT1JumpEnvCfg":
            # Disable rewards with zero weight
            for attr in dir(self.rewards):
                if not attr.startswith("__"):
                    reward_attr = getattr(self.rewards, attr)
                    if reward_attr is not None and not callable(reward_attr) and hasattr(reward_attr, 'weight') and reward_attr.weight == 0:
                        setattr(self.rewards, attr, None)

        # ======================================================================================
        # Terminations Configuration
        # ======================================================================================
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # Add termination for excessive ground contacts (walking/hopping instead of clean jump)
        # Allows max 2 contact phases: (1) initial stance, (2) landing
        # If feet touch ground a 3rd time, episode terminates
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        self.terminations.excessive_ground_contacts = DoneTerm(
            func=mdp.excessive_ground_contacts,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "max_contact_phases": 2,  # Initial stance + landing only
            },
        )

        # Add successful termination when landing perfectly on target
        # Success criteria: within 10cm of target, all joints stable (vel < 0.1 rad/s), maintained for 0.5s
        self.terminations.successful_landing = DoneTerm(
            func=mdp.successful_landing,
            time_out=True,  # This is a success termination, not a failure
            params={
                "command_name": "jump_target",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
                "position_tolerance": 0.1,   # Within 10cm of target
                "velocity_threshold": 0.1,   # Joint velocities < 0.1 rad/s
                "stability_time": 0.5,       # Maintain stability for 0.5 seconds
            },
        )

        # ======================================================================================
        # Curriculum Configuration
        # ======================================================================================

        # Use anti-forgetting jump curriculum with 10 stages
        # This curriculum prevents catastrophic forgetting by sampling from all unlocked stages
        self.curriculum.jump_target_levels = CurrTerm(
            func=mdp.jump_target_curriculum,
            params={
                "reward_term_name": "reach_target_zone",
                "command_term_name": "jump_target",
                "success_threshold": 0.7,           # Progress to next stage at >70% success
                "regression_threshold": 0.3,        # Regress to previous stage at ≤30% success
                "rollouts_per_stage": 20,          # Collect 20 rollouts per unlocked stage before evaluation
            },
        )

        # Disable terrain and velocity curriculums
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        # ======================================================================================
        # Events Configuration
        # ======================================================================================
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ======================================================================================
        # Episode Settings
        # ======================================================================================
        self.episode_length_s = 15.0  # Longer episodes for jump execution
        self.decimation = 4
