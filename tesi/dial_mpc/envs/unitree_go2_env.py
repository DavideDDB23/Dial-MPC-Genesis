from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

from functools import partial

import mujoco
import genesis.utils.mjcf as mjcf
import torch
import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.utils import *
from dial_mpc.utils.io_utils import get_model_path


@dataclass
class UnitreeGo2EnvConfig(BaseEnvConfig):
    kp: Union[float, torch.Tensor] = 30.0
    kd: Union[float, torch.Tensor] = 0.0
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "trot"
    n_envs: int = 1

@dataclass
class State:
    sim_state: Any
    obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    info: Dict[str, torch.Tensor]

class UnitreeGo2Env(BaseEnv):
    def __init__(self, config: UnitreeGo2EnvConfig):
        super().__init__(config)

        self._foot_radius = 0.0175

        self._gait = config.gait
        self._gait_phase = {
            "stand": torch.tensor(4),
            "walk": torch.tensor([0.0, 0.5, 0.75, 0.25]),
            "trot": torch.tensor([0.0, 0.5, 0.5, 0.0]),
            "canter": torch.tensor([0.0, 0.33, 0.33, 0.66]),
            "gallop": torch.tensor([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": torch.tensor([1.0, 1.0, 0.0]),
            "walk": torch.tensor([0.75, 1.0, 0.08]),
            "trot": torch.tensor([0.45, 2, 0.08]),
            "canter": torch.tensor([0.4, 4, 0.06]),
            "gallop": torch.tensor([0.3, 3.5, 0.10]),
        }

        self._torso_idx = self.robot.base_link.idx

        self._init_q = torch.tensor(self.model.keyframe("home").qpos)
        self._default_pose = self.model.keyframe("home").qpos[7:]
        
        self.joint_range = torch.tensor(
            [
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
            ]
        )

        feet_site = [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ]
        feet_site_id = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        # store as PyTorch tensor of int indices
        self._feet_site_id = torch.tensor(feet_site_id, dtype=torch.int32)
        
        # find index of the RigidSolver in the simulation solver list
        self._rigid_solver_idx = None
        for idx, solver in enumerate(self.scene.sim.solvers):
            if isinstance(solver, RigidSolver):
                self._rigid_solver_idx = idx
                break
        assert self._rigid_solver_idx is not None, "RigidSolver not found in sim.solvers"
        
    ''' # test: retrieve feet site world positions via MuJoCo Data
        mj_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, mj_data)
        # site_xpos is a (n_site, 3) numpy array
        feet_pos = mj_data.site_xpos[self._feet_site_id.tolist()]
        self._feet_site_pos = torch.as_tensor(feet_pos, dtype=torch.float32)
        print("Feet site positions (world):", self._feet_site_pos)
    '''

    def create_robot(self):
        dof_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        
        model_path = get_model_path('unitree_go2', 'mjx_go2_force.xml')
        self.model = mjcf.parse_mjcf(model_path)
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(model_path)))

        self.motor_dofs: list[int] = [self.robot.get_joint(name).dof_idx_local for name in dof_names]

    def pipeline_init(self, init_q: torch.Tensor, init_qvel: torch.Tensor, envs_idx: torch.Tensor = None):
        """
        Initialize the simulation pipeline state in Genesis, analogous to Brax's pipeline_init.
        """
        # environment indices for batched envs; reset only these indices
        b = envs_idx.numel()
        # split init_q into base position, orientation, and joint positions
        base_pos = init_q[:3]
        base_quat = init_q[3:7]
        joint_pos = init_q[7:]
        # set base pose
        self.robot.set_pos(
            base_pos.unsqueeze(0).repeat(b, 1),
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_quat(
            base_quat.unsqueeze(0).repeat(b, 1),
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # set joint positions
        self.robot.set_dofs_position(
            position=joint_pos.unsqueeze(0).repeat(b, 1),
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # zero joint velocities
        self.robot.zero_all_dofs_velocity(envs_idx)
        # step the scene once to compute forward kinematics, contacts, etc.
        self.scene.step()
        # return the fresh simulation state
        return self.scene.get_state()

    def reset(self, envs_idx: torch.Tensor = None) -> State:
        """Reset whole or part of the batch of environments."""
        b = envs_idx.numel()
        # reinitialize selected envs
        sim_state = self.pipeline_init(
            self._init_q,
            torch.zeros(self._nv),
            envs_idx,
        )
        # build initial info per env
        state_info = {
            'pos_tar': torch.tensor([0.282, 0.0, 0.3]).unsqueeze(0).repeat(b, 1),
            'vel_tar': torch.zeros(b, 3),
            'ang_vel_tar': torch.zeros(b, 3),
            'yaw_tar': torch.zeros(b),
            'step': torch.zeros(b, dtype=torch.int64),
            'z_feet': torch.zeros(b, 4),
            'z_feet_tar': torch.zeros(b, 4),
            'randomize_target': torch.full((b,), self._config.randomize_tasks, dtype=torch.bool),
            'last_contact': torch.zeros(b, 4, dtype=torch.bool),
            'feet_air_time': torch.zeros(b, 4),
        }
        # compute initial observation for selected envs
        obs = self._get_obs(sim_state, state_info, envs_idx)
        # reward and done flags
        reward = torch.zeros(b)
        done = torch.zeros(b)
        print("[DEBUG] obs:", obs)
        # return state
        return State(sim_state, obs, reward, done, {}, state_info)

    def _get_obs(self, sim_state, state_info: Dict[str, torch.Tensor], envs_idx: torch.Tensor = None) -> torch.Tensor:
        # fetch the RigidSolverState dynamically
        rigid = sim_state.solvers_state[self._rigid_solver_idx]
        # full positions and velocities
        qpos_full = torch.as_tensor(rigid.qpos)
        qvel_full = torch.as_tensor(rigid.dofs_vel)
        # use only selected envs if provided
        qpos = qpos_full[envs_idx]
        qvel = qvel_full[envs_idx]
        # target commands
        vel_tar = state_info["vel_tar"]
        ang_tar = state_info["ang_vel_tar"]
        # control values same shape as joint positions
        ctrl = torch.zeros_like(qpos[:, 7:])
        # body frame velocities
        base_lin = qvel[:, :3]
        base_ang = qvel[:, 3:6]
        base_quat = qpos[:, 3:7]
        vb = global_to_body_velocity(base_lin, base_quat)
        ab = global_to_body_velocity(base_ang * (torch.pi / 180.0), base_quat)
        # joint positions and velocities
        joint_pos = qpos[:, 7:]
        joint_vel = qvel[:, 6:]
        # concat in Brax order
        obs = torch.cat([
            vel_tar,
            ang_tar,
            ctrl,
            joint_pos,
            vb,
            ab,
            joint_vel,
        ], dim=-1)
        return obs

    def sample_command(self, envs_idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample linear and angular velocity commands for environments that needs it.
        Returns two tensors of shape (lenght of envs_idx, 3): (lin_vel_cmd, ang_vel_cmd).
        """
        # hardcoded ranges (min,max) for linear and angular velocity
        min_lin_x, max_lin_x = -1.5, 1.5  # [m/s]
        min_lin_y, max_lin_y = -0.5, 0.5  # [m/s]
        min_ang_z, max_ang_z = -1.5, 1.5  # [rad/s]
        # sample uniformly using gs_rand_float
        lin_x = gs_rand_float(min_lin_x, max_lin_x, len(envs_idx))
        lin_y = gs_rand_float(min_lin_y, max_lin_y, len(envs_idx))
        ang_z = gs_rand_float(min_ang_z, max_ang_z, len(envs_idx))
        # stack into velocity vectors
        lin_vel_cmd = torch.stack([lin_x, lin_y, torch.zeros(len(envs_idx))], dim=1)
        ang_vel_cmd = torch.stack([
            torch.zeros(len(envs_idx)),
            torch.zeros(len(envs_idx)),
            ang_z
        ], dim=1)
        return lin_vel_cmd, ang_vel_cmd