import sys, os
sys.path.insert(0, os.path.abspath('tesi'))

import genesis as gs
import torch
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

def run_sim(env, n_steps=10000):
    # reset the scene and run simulation steps
    env.scene.reset()
    for i in range(n_steps):
        env.scene.step()

if __name__ == '__main__':

    # configure the environment (dt, timestep, backend, number of envs)
    cfg = UnitreeGo2EnvConfig(
        dt=0.02,
        timestep=0.02,
        backend='cpu',
        n_envs=1,
    )

    # create the environment
    env = UnitreeGo2Env(cfg)

    # run simulation in a separate thread so the viewer remains responsive
    gs.tools.run_in_another_thread(fn=run_sim_test_reset, args=(env, 10000))

    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

    # start the interactive viewer (blocks until closed)
    env.scene.viewer.start() 