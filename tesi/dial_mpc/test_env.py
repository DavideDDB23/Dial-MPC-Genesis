import sys, os
sys.path.insert(0, os.path.abspath('tesi'))

import genesis as gs
import torch
from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig


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
    gs.tools.run_in_another_thread(fn=run_sim, args=(env, 10000))

    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

    # start the interactive viewer (blocks until closed)
    env.scene.viewer.start() 