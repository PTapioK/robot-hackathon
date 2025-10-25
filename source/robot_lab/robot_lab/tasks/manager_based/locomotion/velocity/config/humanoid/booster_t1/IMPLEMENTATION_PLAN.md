# Implementation Plan: Booster T1 Jumping Navigation Task

## Overview
Create a jumping navigation task where the Booster T1 humanoid robot learns to jump to target landing zones with curriculum learning that progressively increases jump distance and height differences.

## Goal
Train the Booster T1 robot to navigate through jumping to reach target boxes at various distances and elevations/depressions, starting from short level jumps and progressing to longer and more challenging jumps through curriculum learning.

## Key Requirements
- Single target box per episode
- Performance-based curriculum progression
- Visual target markers (conceptual target area)
- Targets can be elevated or lowered relative to starting position
- Reduced penalties on high-torque actions to enable explosive jumping movements

## Files to Create/Modify

### New Files to Create
1. **`mdp/commands.py`** - Add `JumpTargetCommand` class
2. **`mdp/observations.py`** - Add jump-specific observations
3. **`mdp/rewards.py`** - Add jump-specific rewards
4. **`mdp/curriculums.py`** - Add `jump_target_levels` curriculum function
5. **`config/humanoid/booster_t1/jump_env_cfg.py`** - New jumping environment configuration

### Files to Modify
- **`mdp/__init__.py`** - Export new functions/classes

## Detailed Implementation

### 1. Command Generator (`mdp/commands.py`)

Create `JumpTargetCommand` class to generate and manage target landing positions:

**Features:**
- Generate random target positions (x, y, z) within curriculum-defined ranges
- Support both elevated platforms (+z) and lowered areas (-z)
- Include visual markers using `VisualizationMarkers` for debugging
- Resample targets on successful landing or episode timeout
- Track current curriculum level

**Key Parameters:**
```python
- horizontal_range: (min_dist, max_dist)  # e.g., (0.3, 0.5) for stage 1
- height_range: (min_height, max_height)  # e.g., (-0.05, 0.05) for stage 1
- success_radius: 0.3  # meters tolerance for landing zone
- visualization_enabled: True
```

**Command Output:**
- Shape: (num_envs, 3) containing [target_x, target_y, target_z] in world frame

---

### 2. Observations (`mdp/observations.py`)

Add observation functions to help robot learn jumping behavior:

#### `target_position_rel(env) -> torch.Tensor`
- Returns 3D vector from robot base to target in robot body frame
- Shape: (num_envs, 3)
- Helps robot understand where to jump

#### `target_distance_height(env) -> torch.Tensor`
- Returns [horizontal_distance, height_difference]
- Shape: (num_envs, 2)
- Horizontal distance: sqrt(dx^2 + dy^2)
- Height difference: target_z - base_z

#### `is_airborne(env) -> torch.Tensor`
- Binary indicator: 1.0 if all feet in air, 0.0 otherwise
- Shape: (num_envs, 1)
- Uses ContactSensor to detect flight phase

#### `feet_position_to_target(env) -> torch.Tensor`
- Relative position of both feet to target center
- Shape: (num_envs, 6) - [left_foot_xyz, right_foot_xyz] relative to target
- Helps robot understand foot placement during landing

---

### 3. Rewards (`mdp/rewards.py`)

#### Jump-Specific Rewards (NEW)

**`reach_target_zone(env, tolerance=0.3) -> torch.Tensor`**
- Large positive reward for landing within target zone
- Exponential kernel: exp(-distance^2 / tolerance^2)
- Only triggered when robot lands (feet make contact after flight)
- Weight: **+100.0**

**`target_progress(env, std=0.5) -> torch.Tensor`**
- Shaped reward for reducing horizontal distance to target
- Tracks distance reduction over time
- Helps guide robot toward target during approach
- Weight: **+2.0**

**`flight_phase_quality(env, min_height=0.1) -> torch.Tensor`**
- Reward proper takeoff with both feet leaving ground
- Bonus for achieving minimum flight height
- Weight: **+5.0**

**`landing_stability(env) -> torch.Tensor`**
- Reward stable landing: low angular velocity + upright orientation
- Combines flat_orientation and low ang_vel_xy
- Only applies during landing phase
- Weight: **+8.0**

**`smooth_flight_trajectory(env) -> torch.Tensor`**
- Penalize excessive rotation during flight phase
- Encourages controlled flight
- Weight: **-0.5**

**`landing_orientation(env, target_orientation=[0,0,0,1]) -> torch.Tensor`**
- Reward landing with correct upright orientation
- Uses quaternion distance
- Weight: **+3.0**

**`energy_efficient_jump(env) -> torch.Tensor`**
- Moderate penalty on joint torques (much lower than walking)
- Allows explosive movements but discourages waste
- Weight: **-1e-8** (vs -3e-7 in walking)

#### Modified Rewards from Walking Config

**Penalties to REDUCE:**
- `joint_torques_l2`: **-3e-7 → -1e-8** (allow high torques for jumping)
- `joint_acc_l2`: **-1.25e-7 → -5e-9** (allow rapid acceleration)
- `action_rate_l2`: **-0.075 → -0.01** (allow quick action changes)

**Penalties to REMOVE/DISABLE:**
- `feet_air_time`: Remove (we WANT air time for jumping)
- `feet_slide`: Remove (not relevant for jumping)
- `track_lin_vel_xy_exp`: Remove (not tracking velocity, tracking position)
- `track_ang_vel_z_exp`: Remove (not relevant)
- Joint deviation penalties: Keep but reduce weights by 50%

**Penalties to KEEP:**
- `is_terminated`: **-200.0** (still critical)
- `flat_orientation_l2`: **-0.2** (but disable during flight phase)
- `ang_vel_xy_l2`: **-0.1** (excessive tumbling bad)

---

### 4. Curriculum Learning (`mdp/curriculums.py`)

Create `jump_target_curriculum(env, env_ids, reward_term_name="reach_target_zone")`:

**Curriculum Stages:**

| Stage | Horizontal Range | Height Range | Description |
|-------|-----------------|--------------|-------------|
| 1 | 0.3 - 0.5m | ±0.05m | Short level jumps |
| 2 | 0.5 - 0.8m | ±0.10m | Medium jumps, slight elevation |
| 3 | 0.8 - 1.2m | ±0.20m | Long jumps, moderate elevation |
| 4 | 1.2 - 1.5m | ±0.30m | Advanced jumps, large elevation |

**Progression Logic:**
- Track success rate over rolling window (last 50 episodes)
- Success = landing within target zone (0.3m tolerance)
- Advance to next stage when success rate > 70%
- Update command generator ranges dynamically
- Store curriculum state in env attributes

**Implementation:**
```python
def jump_target_curriculum(env, env_ids, reward_term_name="reach_target_zone"):
    # Initialize curriculum stages on first call
    if not hasattr(env, '_jump_curriculum_level'):
        env._jump_curriculum_level = 0
        env._jump_success_history = []

    # Check progression every 50 episodes
    if len(env._jump_success_history) >= 50:
        success_rate = sum(env._jump_success_history[-50:]) / 50
        if success_rate > 0.7 and env._jump_curriculum_level < 3:
            env._jump_curriculum_level += 1
            # Update command ranges
            update_jump_command_ranges(env)

    return env._jump_curriculum_level
```

---

### 5. Environment Configuration (`jump_env_cfg.py`)

Create new environment config based on `rough_env_cfg.py`:

**Key Changes:**

#### Commands
```python
# Replace velocity command with jump target command
commands = CommandsCfg()
commands.jump_target = JumpTargetCommandCfg(
    asset_name="robot",
    horizontal_range=(0.3, 0.5),  # Start with stage 1
    height_range=(-0.05, 0.05),
    success_radius=0.3,
    resampling_time_range=(10.0, 15.0),
    debug_vis=True,
)
```

#### Observations
```python
observations.policy.target_position_rel = ObsTerm(func=mdp.target_position_rel)
observations.policy.target_distance_height = ObsTerm(func=mdp.target_distance_height)
observations.policy.is_airborne = ObsTerm(func=mdp.is_airborne)
observations.policy.feet_position_to_target = ObsTerm(func=mdp.feet_position_to_target)

# Remove velocity-tracking observations
observations.policy.velocity_commands = None
observations.policy.base_lin_vel = None  # Already None in rough config
```

#### Rewards (Jump-Specific)
```python
# New jump rewards
rewards.reach_target_zone = RewTerm(func=mdp.reach_target_zone, weight=100.0,
                                     params={"tolerance": 0.3})
rewards.target_progress = RewTerm(func=mdp.target_progress, weight=2.0)
rewards.flight_phase_quality = RewTerm(func=mdp.flight_phase_quality, weight=5.0)
rewards.landing_stability = RewTerm(func=mdp.landing_stability, weight=8.0)
rewards.smooth_flight_trajectory = RewTerm(func=mdp.smooth_flight_trajectory, weight=-0.5)
rewards.landing_orientation = RewTerm(func=mdp.landing_orientation, weight=3.0)

# Modified penalties
rewards.joint_torques_l2.weight = -1e-8  # Reduced from -3e-7
rewards.joint_acc_l2.weight = -5e-9  # Reduced from -1.25e-7
rewards.action_rate_l2.weight = -0.01  # Reduced from -0.075

# Disable walking-specific rewards
rewards.track_lin_vel_xy_exp = None
rewards.track_ang_vel_z_exp = None
rewards.feet_air_time = None
rewards.feet_slide = None
```

#### Curriculum
```python
curriculum.jump_target_levels = CurrTerm(
    func=mdp.jump_target_curriculum,
    params={"reward_term_name": "reach_target_zone"}
)
curriculum.terrain_levels = None  # Start with flat terrain
curriculum.command_levels = None  # Using jump curriculum instead
```

#### Scene/Terrain
```python
# Use flat plane for initial training
scene.terrain.terrain_type = "plane"
scene.terrain.terrain_generator = None

# Add target visualization marker
scene.target_marker = VisualizationMarkersCfg(
    prim_path="/Visuals/TargetMarker",
    markers={
        "target_box": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(0.6, 0.6, 0.1),
        ),
    },
)
```

#### Episode Settings
```python
episode_length_s = 15.0  # Longer episodes for jump execution
decimation = 4
```

---

## Scene Modifications (Optional for Advanced Stages)

### Visual Target Markers
- Use `VisualizationMarkers` to show target landing zone
- Box shape: 0.6m x 0.6m x 0.1m (matching success radius)
- Color coding: green = reachable, red = too far
- Semi-transparent to see through

### Physical Platforms (Future Work)
- Add `RigidObject` platforms at various heights
- Collision enabled for realistic landing
- Multiple platform heights for curriculum progression
- Useful for stage 3-4 training

---

## Training Strategy

### Phase 1: Foundation (Stages 1-2)
- Train on flat terrain with level jumps (0.3-0.5m)
- Focus on basic takeoff and landing mechanics
- Success criteria: 70% landing accuracy within 0.3m
- Expected training time: ~500-1000 episodes

### Phase 2: Distance Progression (Stage 2-3)
- Introduce small height variations (±0.1m)
- Increase jump distances (0.5-1.2m)
- Refine flight trajectory control
- Expected training time: ~1000-2000 episodes

### Phase 3: Height Variation (Stage 3-4)
- Large height differences (±0.2-0.3m)
- Long distance jumps (1.2-1.5m)
- Combined challenges
- Expected training time: ~1500-3000 episodes

### Phase 4: Rough Terrain (Advanced)
- Enable terrain generator with curriculum
- Jumps over obstacles
- Realistic deployment scenarios
- Expected training time: ~2000-5000 episodes

---

## Implementation Order

1. ✓ Create this documentation file
2. ✓ Create `JumpTargetCommand` class with visualization
3. ✓ Add observation functions (`target_position_rel`, etc.)
4. ✓ Implement jump-specific reward functions
5. ✓ Create `jump_target_curriculum` function
6. ✓ Build `jump_env_cfg.py` environment config
7. TODO: Test with single fixed jump distance (disable curriculum)
8. TODO: Verify observations and rewards are working correctly
9. TODO: Enable curriculum and train stage 1
10. TODO: Progressively train through all curriculum stages
11. TODO: Evaluate performance and iterate on reward weights

---

## Key Considerations

### Safety & Stability
- Start with conservative torque limits to prevent instability
- Monitor landing impact forces to avoid termination
- May need to adjust `illegal_contact` termination threshold
- Consider adding soft landing rewards (gradual contact)

### Debugging & Visualization
- Visual debugging essential - always enable target markers
- Log success rate, average landing distance, flight time
- Visualize foot trajectories during flight
- Monitor joint torques during takeoff phase

### Hyperparameter Tuning
- Reward weights may need adjustment based on behavior
- Success radius (0.3m) may be too strict initially
- Consider starting with 0.5m radius, then tighten
- Flight phase detection threshold may need tuning

### Pre-training Considerations
- May benefit from pre-training on walking task
- Transfer learning from locomotion policy possible
- Consider fine-tuning from `rough_env_cfg` checkpoint
- May need to adjust actuator gains for explosive movements

### Alternative Approaches
- Could use position-based actions instead of joint positions
- Hierarchical policy: high-level (jump planning) + low-level (control)
- Separate takeoff and landing sub-policies
- Add intermediate waypoints for very long jumps

---

## Success Metrics

### Performance Metrics
- **Success Rate**: % episodes landing within target radius
- **Landing Accuracy**: Average distance from target center
- **Flight Quality**: Trajectory smoothness, peak height achieved
- **Energy Efficiency**: Average torque during jump
- **Landing Stability**: Angular velocity at landing, orientation error

### Curriculum Progression Metrics
- Time to achieve 70% success rate per stage
- Total training episodes required
- Sample efficiency improvements across stages
- Generalization to unseen jump distances

### Deployment Readiness
- Success rate > 80% on stage 4
- Robust to initial state perturbations
- Handles rough terrain (if phase 4 completed)
- Safe landing mechanics (no falls/terminations)

---

## Future Extensions

1. **Multi-jump Navigation**: Jump through sequence of targets
2. **Obstacle Avoidance**: Jump over barriers, gaps
3. **Dynamic Targets**: Moving landing zones
4. **Vision-based**: Use cameras instead of perfect state
5. **Rough Terrain**: Uneven starting/landing surfaces
6. **Long-range Jumps**: Extend to 2-3m distances
7. **Parkour Skills**: Combine with climbing, vaulting

---

## References

- Base environment: `velocity_env_cfg.py`
- Booster T1 config: `rough_env_cfg.py`
- Curriculum implementation: `mdp/curriculums.py`
- Isaac Lab locomotion tasks: `isaaclab_tasks.manager_based.locomotion.velocity`

---

**Document Created**: 2025-10-24
**Task**: Booster T1 Jumping Navigation with Curriculum Learning
**Status**: Planning Complete - Ready for Implementation
