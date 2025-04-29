from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

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
        self.scene.build(n_envs=config.n_envs, env_spacing=(1, 1)) 
        
        self.physical_joint_range = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.joint_range = self.physical_joint_range
        
        lower_torque, upper_torque = self.robot.get_dofs_force_range(self.motor_dofs)
        self.joint_torque_range = torch.stack([lower_torque, upper_torque], dim=1)

        self._nv = self.robot.n_dofs   
        self._nq = self.robot.n_qs
        
        #verificare se è corretto, forse usando tau direttamete da act2tau, non serve questo
        self.robot.set_dofs_kp([self._config.kp] * len(self.motor_dofs), self.motor_dofs)
        self.robot.set_dofs_kv([self._config.kd] * len(self.motor_dofs), self.motor_dofs)

    def create_robot(self):
        """
        Add the robot to the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError

    def act2joint(self, act: torch.Tensor) -> torch.Tensor:
        # act: (n_envs, n_actions) or (n_actions,) -> joint positions
        # normalize to [0,1]
        act_normalized = (act * self._config.action_scale + 1.0) / 2.0
        # joint_range: (n_actions, 2)
        lower = self.joint_range[:, 0]
        upper = self.joint_range[:, 1]
        # compute targets, broadcasting over batch
        joint_targets = lower + act_normalized * (upper - lower)
        # clamp to physical_joint_range
        phys_lower = self.physical_joint_range[:, 0]
        phys_upper = self.physical_joint_range[:, 1]
        joint_targets = torch.clamp(joint_targets, phys_lower, phys_upper)
        return joint_targets

    def act2tau(self, act: torch.Tensor, sim_state, envs_idx: torch.Tensor = None) -> torch.Tensor:
        """Convert normalized action into torque using PD on joint errors, querying Genesis SimState."""
        # desired joint positions (n_envs, n_actions)
        joint_target = self.act2joint(act)
        # get dynamics state from RigidSolver
        rigid = sim_state.solvers_state[self._rigid_solver_idx]
        qpos_full = torch.as_tensor(rigid.qpos)     # (n_envs, n_qs)
        qvel_full = torch.as_tensor(rigid.dofs_vel)  # (n_envs, n_dofs)

        qpos = qpos_full[envs_idx]
        qvel = qvel_full[envs_idx]
        # extract joint angles and velocities (skip free-joint dims)
        q = qpos[:, 7:7 + joint_target.shape[-1]]
        qd = qvel[:, 6:6 + joint_target.shape[-1]]
        # PD control
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd
        return tau

