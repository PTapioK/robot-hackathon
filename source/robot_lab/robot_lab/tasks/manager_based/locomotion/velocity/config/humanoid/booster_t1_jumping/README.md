# Booster T1 Jumping Navigation Task

This directory contains the implementation of a jumping navigation task for the Booster T1 humanoid robot using Isaac Lab.

## Overview

The jumping task trains the Booster T1 robot to jump to target landing zones using reinforcement learning with curriculum learning. The robot progressively learns to jump further and handle height variations.

## Environment ID

```
RobotLab-Isaac-Velocity-Jump-Booster-T1-v0
```

## Quick Start

### Training

To train the jumping agent:

```bash
# Using RSL-RL (default)
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Jump-Booster-T1-v0 --num_envs 4096

# Using CusRL
python scripts/cusrl/train.py --task RobotLab-Isaac-Velocity-Jump-Booster-T1-v0 --num_envs 4096
```

### Playing/Testing

To test a trained policy:

```bash
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Jump-Booster-T1-v0 --num_envs 32
```

## Curriculum Stages

The environment implements a 4-stage curriculum that progressively increases difficulty:

| Stage | Horizontal Distance | Height Variation | Description |
|-------|-------------------|------------------|-------------|
| 1 | 0.3 - 0.5m | ±0.05m | Short level jumps |
| 2 | 0.5 - 0.8m | ±0.10m | Medium jumps with slight elevation changes |
| 3 | 0.8 - 1.2m | ±0.20m | Long jumps with moderate elevation changes |
| 4 | 1.2 - 1.5m | ±0.30m | Advanced jumps with large elevation changes |

**Progression:** The robot automatically advances to the next stage when it achieves >70% success rate over the last 50 episodes.

## Key Features

### Observations

The robot receives the following jump-specific observations:

1. **target_position_rel**: 3D vector from robot to target in body frame
2. **target_distance_height**: Horizontal distance and height difference to target
3. **is_airborne**: Binary indicator of flight phase
4. **feet_position_to_target**: Relative foot positions to landing zone

Plus standard observations (joint positions, velocities, projected gravity, etc.)

### Rewards

**Jump-Specific Rewards:**
- `reach_target_zone` (weight: 100.0): Large reward for landing near target
- `target_progress` (weight: 2.0): Shaped reward for reducing distance to target
- `flight_phase_quality` (weight: 5.0): Reward for proper takeoff and flight
- `landing_stability` (weight: 8.0): Reward for stable landing with low angular velocity
- `smooth_flight_trajectory` (weight: -0.5): Penalize excessive rotation during flight
- `landing_orientation` (weight: 3.0): Reward upright landing orientation

**Modified Penalties:**
- Joint torque penalties reduced by ~10x to allow explosive jumping movements
- Joint acceleration penalties reduced by ~25x
- Action rate penalties reduced by ~7.5x
- Velocity tracking rewards removed (not applicable to position-based jumping)
- Feet air time penalties removed (we want air time for jumping!)

### Commands

Instead of velocity commands, the robot receives jump target positions:
- **horizontal_range**: Distance to jump (curriculum-based)
- **height_range**: Height variation of target (curriculum-based)
- **success_radius**: 0.3m tolerance for successful landing
- **resampling_time_range**: New target every 10-15 seconds

## Implementation Details

### Files Created

1. **`jump_env_cfg.py`**: Main environment configuration
2. **`mdp/commands.py`**: Added `JumpTargetCommand` class
3. **`mdp/observations.py`**: Added 4 jump-specific observation functions
4. **`mdp/rewards.py`**: Added 6 jump-specific reward functions
5. **`mdp/curriculums.py`**: Added `jump_target_curriculum` function

### Key Differences from Walking Task

1. **Command System**: Position-based targets instead of velocity commands
2. **Torque Limits**: Much lower penalties to enable explosive movements
3. **Observations**: Added flight phase and landing zone awareness
4. **Rewards**: Focus on landing accuracy, stability, and flight quality
5. **Terrain**: Flat plane for initial training (can be extended to rough terrain)

## Training Tips

1. **Early Stages**: The robot will learn basic jumping mechanics in Stage 1. This may take 500-1000 episodes.

2. **Curriculum Progression**: Monitor the console for curriculum advancement messages. If the robot gets stuck at a stage, consider:
   - Adjusting reward weights
   - Lowering the success threshold (default: 70%)
   - Increasing training time per stage

3. **Hyperparameter Tuning**: Key parameters to adjust:
   - `success_radius`: Make larger (e.g., 0.5m) if early training is too difficult
   - Reward weights: Balance between landing accuracy and stability
   - Episode length: Default 15s may need adjustment based on jump distances

4. **Visualization**: Enable debug visualization to see target markers:
   ```python
   env.command_manager.get_term("jump_target").set_debug_vis(True)
   ```

## Expected Behavior

### Stage 1 (0.3-0.5m)
- Robot learns to crouch and explode upward
- Short hops to nearby targets
- Focus on landing without falling

### Stage 2 (0.5-0.8m)
- Longer jumps requiring more power
- Introduction to slight height variations
- Improved landing stability

### Stage 3 (0.8-1.2m)
- Significant distance requiring optimized trajectories
- Moderate elevation changes (up to 20cm)
- Refined flight control

### Stage 4 (1.2-1.5m)
- Maximum distance jumps
- Large elevation changes (up to 30cm)
- Mastery of all jumping skills

## Troubleshooting

### Robot falls immediately
- Check that termination conditions are properly configured
- Ensure initial robot pose is stable
- Verify contact sensors are working

### No curriculum progression
- Check success metrics in tensorboard
- Verify `reach_target_zone` reward is being accumulated
- Lower success threshold if necessary

### Robot doesn't leave ground
- Increase reward weight for `flight_phase_quality`
- Decrease torque penalties further
- Check actuator limits allow for sufficient power

## Future Extensions

1. **Rough Terrain**: Enable terrain generator for jumps over obstacles
2. **Dynamic Targets**: Moving landing zones
3. **Multi-Jump Sequences**: Chain multiple jumps together
4. **Vision-Based**: Use cameras instead of perfect state
5. **Parkour Skills**: Add climbing, vaulting, wall jumps

## References

- Base environment: `velocity_env_cfg.py`
- Booster T1 config: `rough_env_cfg.py`
- Implementation plan: `IMPLEMENTATION_PLAN.md`
- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/

---

**Status**: Core implementation complete. Ready for training and testing.
**Created**: 2025-10-24
**Last Updated**: 2025-10-24
