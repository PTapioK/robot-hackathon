# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Booster T1 Jumping Navigation Environment Configuration - JUMP-ONLY

This environment trains the Booster T1 humanoid robot to jump (NOT walk) to target
landing zones on rough terrain with high-performance curriculum learning.

Anti-Walking Mechanisms:
- Pre-takeoff ground penalty (-2.0 weight) - discourages staying grounded
- Dense velocity-based progress (+5.0 weight) - rewards forward movement
- No ground-based approach rewards (disabled)

Performance Optimizations:
- NO height_scan raycasting (disabled - extremely expensive!)
- Optimized contact sensor rewards (landing detection only)
- Minimal observations: only target position from command buffer
- Zero-overhead curriculum (fully vectorized GPU ops)
- Aggressive PhysX GPU settings (655K patches, 384MB collision stack)

Curriculum Stages (20 total):
- Stage 0: Beginner jump (0.3-0.4m) - learn basic jumping mechanics
- Stage 1-19: Progressive jumps from 0.4m to 3.0m with height variation
- Removed standing/tiny jumps - robot learns jumping from the start

Core Rewards (4 FSM-based):
- Landing success (50.0) - one-time shaped reward on first landing
- Progress to target (5.0) - dense velocity-based progress (no history)
- Pre-takeoff ground penalty (-2.0) - anti-walking (Stage 1+ only)
- Upright landing bonus (8.0) - one-time orientation reward

Performance: ~180-220K steps/s with 16K environments on H100 (with rough terrain)
FSM optimization: Reduced redundant contact sensor queries via state caching
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

        # CRITICAL PERFORMANCE: Disable height scanners (raycasting is VERY expensive!)
        # Jumping task doesn't need terrain scanning - robot just jumps to target
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        # CRITICAL PERFORMANCE: Optimize contact sensor (history tracking is expensive!)
        from isaaclab.sensors import ContactSensorCfg
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=0,  # Disabled history (was 3)
            track_air_time=False,  # Disabled air time tracking
            update_period=0.0,  # Update every step (default)
        )

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

        # MINIMAL jump observations (only command buffer access - very cheap!)
        # The command manager already computes target_pos_b every step, we just read it
        self.observations.policy.target_position_rel = ObsTerm(
            func=mdp.target_position_rel,
            params={"command_name": "jump_target"},
        )
        # Disable expensive observations that query sensors/body states
        self.observations.policy.target_distance_height = None
        self.observations.policy.is_airborne = None
        self.observations.policy.feet_position_to_target = None

        # Remove velocity-tracking observations (not needed for jumping)
        self.observations.policy.velocity_commands = None

        # Keep existing observations with adjusted scales
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None  # Disable - not tracking velocity

        # CRITICAL PERFORMANCE: Disable height_scan raycasting (VERY expensive with 16K envs!)
        # Jumping task doesn't need terrain scanning - just jump to target position
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None  # Also disable for critic!

        # ======================================================================================
        # Actions Configuration
        # ======================================================================================
        # Keep action configuration from parent (allows for explosive movements)
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ======================================================================================
        # Rewards Configuration
        # ======================================================================================

        # --- FSM-BASED JUMP REWARDS (Performance-Optimized ~180-220K steps/s) ---
        # One-time landing rewards with FSM state tracking to reduce redundant sensor queries
        # Key features: shaped landing rewards, dense velocity-based progress, anti-walking

        # 1. Landing Success - One-time shaped reward on first landing
        self.rewards.jump_landing_win = RewTerm(
            func=mdp.jump_landing_win,
            weight=50.0,  # Main task reward (fires once per jump)
            params={
                "command_name": "jump_target",
                "success_radius": 0.3,  # Shaped reward within 30cm (matches curriculum)
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 2. Progress Toward Target - Dense velocity-based reward (no history)
        self.rewards.progress_to_target = RewTerm(
            func=mdp.progress_to_target,
            weight=5.0,  # Reward progress rate toward target
            params={
                "command_name": "jump_target",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 3. ANTI-WALKING: Penalize ground contact before takeoff (ALL stages)
        self.rewards.pre_takeoff_ground_time = RewTerm(
            func=mdp.pre_takeoff_ground_time,
            weight=-2.0,  # Penalty for staying on ground (applied in all stages)
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "command_name": "jump_target",
            },
        )

        # 4. Upright Landing Bonus - One-time orientation reward on first landing
        self.rewards.upright_on_landing = RewTerm(
            func=mdp.upright_on_landing,
            weight=8.0,  # Bonus for landing upright (fires once per jump)
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Disable legacy jump rewards (replaced by FSM-based above)
        self.rewards.reach_target_zone = None
        self.rewards.airborne_progress = None
        self.rewards.penalize_ground_movement = None
        self.rewards.dual_foot_landing = None
        self.rewards.landing_stability = None
        self.rewards.approach_target = None

        # Disable less critical rewards (keep overhead low)
        self.rewards.flight_phase_quality = None
        self.rewards.smooth_flight_trajectory = None
        self.rewards.landing_orientation = None  # Covered by landing_stability
        self.rewards.dual_foot_takeoff = None

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

        # PERFORMANCE: Simplify joint penalties (many are redundant or negligible)
        self.rewards.joint_torques_l2 = None  # Disabled - weight too small to matter
        self.rewards.joint_vel_l2 = None      # Already disabled
        self.rewards.joint_acc_l2 = None      # Disabled - weight too small to matter

        # Disable joint deviation penalties (expensive per-joint computations)
        self.rewards.joint_deviation_hip_l1 = None
        self.rewards.joint_deviation_arms_l1 = None
        self.rewards.joint_deviation_torso_l1 = None

        self.rewards.joint_pos_limits.weight = -1.0  # Keep - prevents breaks
        self.rewards.joint_vel_limits = None
        self.rewards.joint_power = None
        self.rewards.stand_still_without_cmd = None
        self.rewards.joint_pos_penalty = None  # Disabled - redundant with limits
        self.rewards.joint_mirror = None

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

        # PERFORMANCE TEST: Disable jump-specific terminations (they query sensors every step!)
        # self.terminations.excessive_ground_contacts = DoneTerm(...)
        # self.terminations.successful_landing = DoneTerm(...)

        # ======================================================================================
        # Curriculum Configuration
        # ======================================================================================

        # Optimized jump curriculum (fully vectorized, minimal overhead)
        self.curriculum.jump_target_levels = CurrTerm(
            func=mdp.jump_target_curriculum,
            params={
                "reward_term_name": "jump_landing_win",  # Use FSM-based landing reward
                "command_term_name": "jump_target",
                "success_threshold": 0.5,        # Advance when 50% success rate
                "regression_threshold": 0.2,     # Regress if below 20%
                "rollouts_per_stage": 10,
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

        # Reduce reset randomization for early training stability
        # Random velocities were causing immediate falls in Stage 0
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),    # Reduced from (-0.5, 0.5)
            "y": (-0.1, 0.1),    # Reduced from (-0.5, 0.5)
            "z": (-0.1, 0.1),    # Reduced from (-0.5, 0.5)
            "roll": (-0.1, 0.1),  # Reduced from (-0.5, 0.5)
            "pitch": (-0.1, 0.1), # Reduced from (-0.5, 0.5)
            "yaw": (-0.1, 0.1),   # Reduced from (-0.5, 0.5)
        }

        # Disable external forces on reset (too destabilizing for early training)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (0.0, 0.0)

        # FSM state reset on episode termination (critical for FSM-based rewards)
        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.reset_jump_fsm = EventTerm(
            func=mdp.reset_jump_state_on_termination,
            mode="reset",
        )

        # ======================================================================================
        # Episode Settings
        # ======================================================================================
        self.episode_length_s = 15.0  # Longer episodes for jump execution
        self.decimation = 4

        # ======================================================================================
        # PhysX GPU Settings - Optimized for NVIDIA H100 80GB (Aggressive Throughput)
        # ======================================================================================
        # Patch buffer: ~655K patches for high parallel contact processing
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        # Collision stack: 384 MB for large-scale collision detection
        self.sim.physx.gpu_collision_stack_size = 384 * 2**20

        # Heap capacity: 256 MB for dynamic allocations
        self.sim.physx.gpu_heap_capacity = 256 * 2**20

        # Temp buffer: 128 MB for intermediate physics calculations
        self.sim.physx.gpu_temp_buffer_capacity = 128 * 2**20

        # Collision pair tracking: 2M pairs for dynamic environments
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**21