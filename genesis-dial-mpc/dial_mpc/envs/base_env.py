from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import sys, os
sys.path.insert(0, os.path.abspath('genesis-dial-mpc'))

import jax
import jax.numpy as jnp
from functools import partial

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.scene import Scene
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import torch

from dial_mpc.envs.base import *

from dial_mpc.config.base_env_config import BaseEnvConfig


class BaseEnv(PipelineEnv):
    def __init__(self, config: BaseEnvConfig):
        assert jnp.allclose(config.dt % config.timestep, 0.0), "timestep must be divisible by dt"
        self._config = config
        n_frames = int(config.dt / config.timestep)
        
        gs.init(backend=gs.cpu)
        torch.set_default_device("cpu")
        
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
        self.scene.build() 

        # joint limit definitions
        joint_limits = self.robot.get_dofs_limit(self.motor_dofs)
        self.physical_joint_range = jnp.asarray(torch.stack(joint_limits, dim=1))
        self.joint_range = self.physical_joint_range
        
        lower_torque, upper_torque = self.robot.get_dofs_force_range(self.motor_dofs)
        self.joint_torque_range = jnp.asarray(torch.stack([lower_torque, upper_torque], dim=1))
        
        self._nv = self.robot.n_dofs   
        self._nq = self.robot.n_qs

        if self._config.leg_control == "position":
            self.robot.set_dofs_kp([self._config.kp] * len(self.motor_dofs), self.motor_dofs)
            self.robot.set_dofs_kv([self._config.kd] * len(self.motor_dofs), self.motor_dofs)


    def create_robot(self):
        """
        Add the robot to the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        act_normalized = (
            act * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )  # scale to joint range
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        joint_target = self.act2joint(act)

        q = pipline_state.qpos[7:]
        q = q[: len(joint_target)]
        qd = pipline_state.qvel[6:]
        qd = qd[: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau
