from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

from functools import partial

import genesis as gs
import torch

from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.scene import Scene
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

from dial_mpc.config.base_env_config import BaseEnvConfig


class BaseEnv:
    def __init__(self, config: BaseEnvConfig):
        assert config.dt % config.timestep == 0.0, "timestep must be divisible by dt"
        self._config = config
        n_frames = int(config.dt / config.timestep)
        
        if config.backend == "cpu":
            gs.init(backend=gs.cpu)
            torch.set_default_device("cpu")
        elif config.backend == "mps":
            gs.init(backend=gs.gpu)
            torch.set_default_device("mps")

         # create scene
        self.scene: Scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=config.dt, substeps=n_frames),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / config.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options = gs.options.VisOptions(
                background_color=(0.15, 0.25, 0.35),
                ambient_light=(0.3, 0.3, 0.3),
                lights=[
                    {
                        "type": "directional",
                        "dir": (0.0, 0.0, -1.0),
                        "color": (0.6, 0.6, 0.6),
                        "intensity": 5.0,
                    },
                ],
            ),
            rigid_options=gs.options.RigidOptions(
                dt=config.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=True,
        )

        # add an infinite plane at z=0 as the ground
        plane = gs.morphs.Plane(
            pos=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            fixed=True,
        )
        self.scene.add_entity(morph=plane)

        self.create_robot()

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        # build batched simulation for multiple environments
        self.scene.build(n_envs=config.n_envs) 

    #    self.physical_joint_range2 = self.model.jnt_range[1:]
        self.physical_joint_range = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.joint_range = self.physical_joint_range
    #    self.joint_torque_range2 = self.model.actuator_ctrlrange
        lower_torque, upper_torque = self.robot.get_dofs_force_range(self.motor_dofs)
        self.joint_torque_range = torch.stack([lower_torque, upper_torque], dim=1)

         # number of everything
    #    self._nv2 = self.model.nv
    #    self._nq2 = self.model.nq

        self._nv = self.robot.n_dofs   
        self._nq = self.robot.n_qs   

    def create_robot(self):
        """
        Add the robot to the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError
