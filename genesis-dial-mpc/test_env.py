import sys, os
sys.path.insert(0, os.path.abspath('genesis-dial-mpc'))

import genesis as gs
import torch
import math
from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig
import jax


def run_sim_test_reset(env, n_steps=10000):
    rng = jax.random.PRNGKey(seed=0)
    # reset all environments explicitly by passing their indices
    rng, rng_reset = jax.random.split(rng)
    reset_env = jax.jit(env.reset)
    state = reset_env(rng_reset)
    print(f"[Test] Initial reset returned: {state}")
    # interval in steps: 5 sec / dt
    reset_interval = int(5.0 / env._config.dt)
    for i in range(n_steps):
        env.scene.step()
        # every reset_interval steps, call reset and log
        if (i + 1) % reset_interval == 0:
            rng, rng_reset = jax.random.split(rng_reset)
            state = reset_env(rng_reset)
            print(f"[Test] Reset at step {i+1}, returned: {state}")


def run_sim_test_step(env, n_steps=1000):
    rng = jax.random.PRNGKey(seed=0)
    # Reset the environment to get initial state
    rng, rng_reset = jax.random.split(rng)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    state = reset_env(rng_reset)
    print(f"[Test] Initial state: {state}")
    
    # Track metrics
    total_reward = 0.0
    done_steps = []
    
    # Run for n_steps with random actions
    for i in range(n_steps):
        # Generate random action
        rng, key_act = jax.random.split(rng)
        action = jax.random.uniform(
            key_act, shape=(len(env.motor_dofs),), minval=-1.0, maxval=1.0
        )
        
        # Step the environment
        state = step_env(state, action)
        
        # Track metrics
        total_reward += state.reward
        if state.done > 0:
            done_steps.append(i)
            print(f"[Test] Episode done at step {i}, resetting...")
            rng, rng_reset = jax.random.split(rng)
            state = reset_env(rng_reset)
        
        # Print progress periodically
        if (i + 1) % 100 == 0:
            print(f"[Test] Step {i+1}, Reward: {state.reward}, Total reward: {total_reward}")
    
    print(f"[Test] Completed {n_steps} steps")
    print(f"[Test] Total reward: {total_reward}")
    print(f"[Test] Done steps: {done_steps}")
    return state

def run_deterministic_test(env, n_steps=100, seed=0):
    """
    Run a deterministic test with fixed seed and action sequence.
    This can be used to compare results with Brax.
    
    Args:
        env: The environment to test
        n_steps: Number of steps to run
        seed: Random seed for initialization
    
    Returns:
        A dictionary of recorded metrics
    """
    # Initialize with fixed random seed
    rng = jax.random.PRNGKey(seed=seed)
    rng, rng_reset = jax.random.split(rng)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    state = reset_env(rng_reset)
    print(f"[Test] Initial state:")
    print(f"  - Initial position: {state.pipeline_state.x.pos[0]}")
    print(f"  - Initial rotation: {state.pipeline_state.x.rot[0]}")
    
    # Prepare arrays to store metrics
    rewards = []
    positions = []
    rotations = []
    velocities = []
    ang_velocities = []
    foot_positions = []
    
    # Define a fixed action sequence (sine wave patterns)
    def get_action(step_idx):
        # Generate deterministic sine wave actions for each joint
        # with different frequencies and amplitudes
        actions = []
        for i in range(len(env.motor_dofs)):
            # Different frequency and phase for each joint
            freq = 0.1 + (i * 0.05)
            phase = i * 0.5
            amplitude = 0.5
            
            # Sine wave action
            actions.append(amplitude * math.sin(freq * step_idx + phase))
        
        return jax.numpy.array(actions)
    
    # Run steps with the deterministic action sequence
    for i in range(n_steps):
        # Get deterministic action for this step
        action = get_action(i)
        
        # Step the environment
        state = step_env(state, action)
        
        # Record metrics
        rewards.append(float(state.reward))
        positions.append(state.pipeline_state.x.pos[0].copy())  # Base position
        rotations.append(state.pipeline_state.x.rot[0].copy())  # Base rotation
        velocities.append(state.pipeline_state.xd.vel[0].copy())  # Base velocity
        ang_velocities.append(state.pipeline_state.xd.ang[0].copy())  # Base angular velocity
        foot_positions.append(state.pipeline_state.site_xpos[1:5].copy())  # Foot positions
        
        # Print progress every 10 steps
        if (i + 1) % 10 == 0:
            print(f"[Test] Step {i+1}:")
            print(f"  - State: {state}")
    
    # Compute summary metrics
    avg_reward = sum(rewards) / len(rewards)
    final_position = positions[-1]
    avg_velocity = sum([jax.numpy.linalg.norm(v) for v in velocities]) / len(velocities)
    
    print("\n[Test] Deterministic test results:")
    print(f"  - Average reward: {avg_reward}")
    print(f"  - Final position: {final_position}")
    print(f"  - Average velocity magnitude: {avg_velocity}")
    
    # Return metrics that can be compared with Brax
    results = {
        "rewards": rewards,
        "positions": positions,
        "rotations": rotations,
        "velocities": velocities,
        "ang_velocities": ang_velocities,
        "foot_positions": foot_positions,
        "avg_reward": avg_reward,
        "final_position": final_position,
        "avg_velocity": avg_velocity,
    }
    
    return results

if __name__ == '__main__':

    cfg = UnitreeGo2EnvConfig(
        dt=0.02,
        timestep=0.02,
        backend='cpu',
        leg_control='torque',
    )

    env = UnitreeGo2Env(cfg)

    gs.tools.run_in_another_thread(fn=run_sim_test_reset, args=(env, 10))
    
    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

    env.scene.viewer.start() 