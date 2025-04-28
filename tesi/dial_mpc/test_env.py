import sys, os
sys.path.insert(0, os.path.abspath('tesi'))

import genesis as gs
import torch
import math
from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig


def run_sim_test_reset(env, n_steps=10000):
    # reset all environments explicitly by passing their indices
    envs_idx = torch.arange(env._config.n_envs)
    state = env.reset(envs_idx)
    print(f"[Test] Initial reset returned: {state}")
    # interval in steps: 5 sec / dt
    reset_interval = int(5.0 / env._config.dt)
    for i in range(n_steps):
        env.scene.step()
        # every reset_interval steps, call reset and log
        if (i + 1) % reset_interval == 0:
            state = env.reset(envs_idx)
            print(f"[Test] Reset at step {i+1}, returned: {state}")

def run_sim_test_step(env, n_steps=10000):
    # reset all environments
    envs_idx = torch.arange(env._config.n_envs)
    state = env.reset(envs_idx)
    print(f"[Test] Initial state.obs shape: {state.obs.shape}, reward: {state.reward}")
    # step through environment using zero actions
    action_dim = len(env.motor_dofs)
    for i in range(n_steps):
        action = torch.zeros(env._config.n_envs, action_dim)
        state = env.step(state, action, envs_idx)
        # check for NaNs in observation
        if torch.isnan(state.obs).any():
            print(f"[Test] NaN in obs at step {i}")
            break
        # periodic logging
        if (i + 1) % 100 == 0:
            print(f"[Test] Step {i+1}, obs {tuple(state.obs.shape)}, reward {state.reward}, done {state.done}")

def run_sim(env, n_steps=10000):
    # reset the scene and run simulation steps
    env.scene.reset()
    for i in range(n_steps):
        env.scene.step()

def run_sim_test_full(env, n_steps=1000):
    """Comprehensive test: reset, step, and observation validity over multiple steps."""
    envs_idx = torch.arange(env._config.n_envs)
    # initial reset
    state = env.reset(envs_idx)
    print(f"[Full Test] After reset: obs shape {state.obs.shape}, reward {state.reward}, done {state.done}")
    # stepping loop
    action_dim = len(env.motor_dofs)
    for i in range(n_steps):
        action = torch.zeros(env._config.n_envs, action_dim)
        state = env.step(state, action, envs_idx)
        # check for NaNs
        if torch.isnan(state.obs).any():
            print(f"[Full Test] NaN detected in obs at step {i}")
            return
        # periodic logging
        if (i + 1) % max(1, n_steps // 10) == 0:
            print(f"[Full Test] Step {i+1}: obs shape {state.obs.shape}, reward {state.reward}, done {state.done}")
    print("[Full Test] Completed without NaNs and consistent shapes.")

def run_sim_test_random_actions(env, n_steps=1000):
    """Test environment by stepping with random actions in [-1,1]."""
    envs_idx = torch.arange(env._config.n_envs)
    state = env.reset(envs_idx)
    print(f"[Random Test] After reset: obs shape {state.obs.shape}, reward {state.reward}, done {state.done}")
    action_dim = len(env.motor_dofs)
    for i in range(n_steps):
        # sample uniform random actions in [-1,1]
        action = torch.rand(env._config.n_envs, action_dim) * 2.0 - 1.0
        state = env.step(state, action, envs_idx)
        if torch.isnan(state.obs).any():
            print(f"[Random Test] NaN in obs at step {i}")
            return
        if (i + 1) % max(1, n_steps // 10) == 0:
            print(f"[Random Test] Step {i+1}: obs shape {state.obs.shape}, reward {state.reward}, done {state.done}")
    print("[Random Test] Completed without NaNs and valid shapes.")

def run_sim_test_constant_torque(env, n_steps=10000, torque=1.0):
    """Test environment by stepping with constant torque."""
    envs_idx = torch.arange(env._config.n_envs)
    state = env.reset(envs_idx)
    print(f"[Constant Torque Test] After reset: obs shape {state.obs.shape}, reward {state.reward}, done {state.done}")
    action_dim = len(env.motor_dofs)
    for i in range(n_steps):
        action = torch.full((env._config.n_envs, action_dim), torque)
        state = env.step(state, action, envs_idx)
        if torch.isnan(state.obs).any():
            print(f"[Constant Torque Test] NaN in obs at step {i}")
            return
        if (i + 1) % max(1, n_steps // 10) == 0:
            print(f"[Constant Torque Test] Step {i+1}: obs shape {state.obs.shape}, reward {state.reward}, done {state.done}")
    print("[Constant Torque Test] Completed without NaNs and valid shapes.")

def run_sim_test_obs_state_consistency(env, n_steps=1000):
    """Verify obs qpos matches sim_state qpos for zero-action steps."""
    envs_idx = torch.arange(env._config.n_envs)
    # reset
    state = env.reset(envs_idx)
    action_dim = len(env.motor_dofs)
    start = 6 + action_dim  # vel_tar (3) + ang_vel_tar (3) + ctrl (action_dim)
    for i in range(n_steps):
        action = torch.zeros(env._config.n_envs, action_dim)
        state = env.step(state, action, envs_idx)
        # extract sim_state qpos
        rigid = state.sim_state.solvers_state[env._rigid_solver_idx]
        qpos_full = torch.as_tensor(rigid.qpos)
        qpos = qpos_full[envs_idx]
        # extract obs qpos slice
        end = start + qpos.shape[-1]
        obs_qpos = state.obs[:, start:end]
        if not torch.allclose(obs_qpos, qpos, atol=1e-5):
            print(f"[Obs-State Test] Mismatch at step {i}: obs {obs_qpos}, sim {qpos}")
            return
    print("[Obs-State Test] qpos in obs matches sim_state qpos for all steps.")

def run_sim_test_action_effect(env, n_steps=1000, act_val=1.0):
    """Multi-step test: verify that each action passed to step appears in the obs control slice."""
    envs_idx = torch.arange(env._config.n_envs)
    # reset
    state = env.reset(envs_idx)
    action_dim = len(env.motor_dofs)
    start = 3 + 3  # obs layout: vel_tar(3), ang_vel_tar(3), ctrl
    for i in range(n_steps):
        # constant action every step
        action = torch.full((env._config.n_envs, action_dim), act_val)
        # compute expected control from current state
        if env._config.leg_control == 'position':
            expected = env.act2joint(action)
        else:
            expected = env.act2tau(action, state.sim_state, envs_idx)
        # step env
        state = env.step(state, action, envs_idx)
        # extract control slice from obs
        ctrl_obs = state.obs[:, start:start + action_dim]
        if not torch.allclose(ctrl_obs, expected, atol=1e-5):
            print(f"[Action Test] Step {i}: mismatch ctrl_obs={ctrl_obs}, expected={expected}")
            return
        # periodic confirmation
        if (i + 1) % max(1, n_steps // 10) == 0:
            print(f"[Action Test] Step {i+1}: control matches expected.")
    print("[Action Test] All steps control matches expected.")

def run_sim_test_sine_action(env, n_steps=1000, amp=1.0, freq=1.0):
    """Apply sine-wave action on first joint to observe base movement."""
    envs_idx = torch.arange(env._config.n_envs)
    state = env.reset(envs_idx)
    action_dim = len(env.motor_dofs)
    for i in range(n_steps):
        # compute phase and sine action
        phase = 2 * math.pi * freq * env._config.dt * i
        action = torch.zeros(env._config.n_envs, action_dim)
        action[:, 0] = amp * math.sin(phase)
        # step environment
        state = env.step(state, action, envs_idx)
        # read base position
        rigid = state.sim_state.solvers_state[env._rigid_solver_idx]
        qpos_full = torch.as_tensor(rigid.qpos)
        base_pos = qpos_full[envs_idx][:, :3]
        if (i + 1) % max(1, n_steps // 10) == 0:
            print(f"[Sine Test] Step {i+1}: action[0]={action[0,0]:.3f}, base_pos {base_pos}")
    print("[Sine Test] Completed sine-action test.")

def run_sim_test_sine_joint(env, n_steps=1000, amp=1.0, freq=1.0):
    """Switch to position control and apply sine-wave targets to first joint, observing qpos change."""
    # override to position control
    env._config.leg_control = 'position'
    envs_idx = torch.arange(env._config.n_envs)
    state = env.reset(envs_idx)
    action_dim = len(env.motor_dofs)
    # monitor joint angle index 0 in qpos slice
    for i in range(n_steps):
        phase = 2 * math.pi * freq * env._config.dt * i
        action = torch.zeros(env._config.n_envs, action_dim)
        action[:, 0] = amp * math.sin(phase)
        state = env.step(state, action, envs_idx)
        # extract qpos from sim_state
        rigid = state.sim_state.solvers_state[env._rigid_solver_idx]
        qpos_full = torch.as_tensor(rigid.qpos)
        qpos = qpos_full[envs_idx]
        angle = qpos[:, 7]  # first joint angle
        if (i + 1) % max(1, n_steps // 10) == 0:
            # convert Tensors to Python floats for formatting
            target_val = action[0, 0].item() if isinstance(action[0,0], torch.Tensor) else float(action[0,0])
            angle_val = angle[0].item() if isinstance(angle[0], torch.Tensor) else float(angle[0])
            print(f"[SinePos Test] Step {i+1}: target={target_val:.3f}, actual_angle={angle_val:.3f}")
    print("[SinePos Test] Completed position-control sine test.")

if __name__ == '__main__':

    # configure the environment (dt, timestep, backend, number of envs)
    cfg = UnitreeGo2EnvConfig(
        dt=0.02,
        timestep=0.02,
        backend='cpu',
        n_envs=1,
        leg_control='torque',  # use torque control instead of position
    )

    # create the environment
    env = UnitreeGo2Env(cfg)

    # start random-action simulation test
    # start constant-torque simulation test using env.step
    # start constant-torque simulation test using env.step
    # start obs-state consistency test
    # start action-effect test to verify ctrl in obs matches action every step    
    # start sine-action simulation test to observe base movement
    # start sine-position simulation test to move joints
    gs.tools.run_in_another_thread(fn=run_sim_test_sine_joint, args=(env, 1000, 0.5, 0.5))
    
    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

    # start the interactive viewer (blocks until closed)
    env.scene.viewer.start() 