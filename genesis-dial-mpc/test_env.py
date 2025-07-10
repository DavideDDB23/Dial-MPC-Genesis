import sys, os
sys.path.insert(0, os.path.abspath('genesis-dial-mpc'))

import genesis as gs
import torch
import math
from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig
import jax
import numpy as np

def run_sim_test_reset(env, n_steps=10000):
    rng = jax.random.PRNGKey(seed=0)
    
    # Create JIT-compiled reset function
    reset_env_jit = jax.jit(env.reset)
    
    # Initial reset
    rng, rng_reset = jax.random.split(rng)
    state = reset_env_jit(rng_reset)
    print(f"[Test] Initial reset returned: {state}")
    
    # Apply the initial state to Genesis
    apply_state_to_genesis(env, state.pipeline_state)
    
    # Interval in steps: 3 sec / dt
    reset_interval = int(3.0 / env._config.dt)
    
    for i in range(n_steps):
        env.scene.step()
        
        # Every reset_interval steps, call reset and apply to Genesis
        if (i + 1) % reset_interval == 0:
            rng, rng_reset = jax.random.split(rng_reset)
            state = reset_env_jit(rng_reset)
            print(f"[Test] Reset at step {i+1}, returned: {state}")
            
            # Apply the reset state to Genesis
            apply_state_to_genesis(env, state.pipeline_state)

def apply_state_to_genesis(env, pipeline_state):
    """Apply JAX pipeline state to Genesis simulation."""
    
    # Convert JAX arrays to numpy for Genesis
    q = np.array(pipeline_state.q)
    qd = np.array(pipeline_state.qd)
    
    # Set base position and orientation
    env.robot.set_pos(q[:3], zero_velocity=True)
    env.robot.set_quat(q[3:7], zero_velocity=True)
    
    # Set joint positions (skip base pos/quat)
    joint_positions = q[7:]
    env.robot.set_dofs_position(
        position=joint_positions,
        dofs_idx_local=env.motor_dofs,
        zero_velocity=True,
    )
    
    # Set base velocity
    base_lin_vel = qd[:3]
    base_ang_vel = qd[3:6]
#    env.robot.set_vel(base_lin_vel)
#    env.robot.set_ang(base_ang_vel)
    
    # Set joint velocities
    joint_velocities = qd[6:]
    env.robot.set_dofs_velocity(
        velocity=joint_velocities,
        dofs_idx_local=env.motor_dofs,
    )

if __name__ == '__main__':
    cfg = UnitreeGo2EnvConfig(
        dt=0.02,
        timestep=0.02,
        backend='cpu',
        leg_control='torque',
    )

    env = UnitreeGo2Env(cfg)

    gs.tools.run_in_another_thread(fn=run_sim_test_reset, args=(env, 100000))
    
    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
    env.scene.viewer.start()