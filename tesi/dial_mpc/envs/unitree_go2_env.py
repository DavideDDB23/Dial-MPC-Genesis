from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

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
            "stand": torch.zeros(4),
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
        Initialize the simulation pipeline state in Genesis
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
        # return the simulation state
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
        # compute initial observation for selected envs (zero control)
        obs = self._get_obs(sim_state, state_info, ctrl=None, envs_idx=envs_idx)
        # reward and done flags
        reward = torch.zeros(b)
        done = torch.zeros(b)
        # return state
        return State(sim_state, obs, reward, done, {}, state_info)
    
    def step(self, state: State, action: torch.Tensor, envs_idx: torch.Tensor = None) -> State:

        # apply control
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            self.robot.control_dofs_position(
                position=joint_targets,
                dofs_idx_local=self.motor_dofs,
                envs_idx=envs_idx,
            )
            ctrl = joint_targets
        else:
            tau = self.act2tau(action, state.sim_state, envs_idx)
            self.robot.control_dofs_force(
                force=tau,
                dofs_idx_local=self.motor_dofs,
                envs_idx=envs_idx,
            )
            ctrl = tau

        # step simulation
        self.scene.step()
        new_sim_state = self.scene.get_state()

        obs = self._get_obs(new_sim_state, state.info, ctrl, envs_idx)

        # switch to new target if randomize_target is True
        def dont_randomize():
            batch_size = envs_idx.numel()
            return (
                torch.tensor([self._config.default_vx, self._config.default_vy, 0.0], device=envs_idx.device).unsqueeze(0).expand(batch_size, -1),
                torch.tensor([0.0, 0.0, self._config.default_vyaw], device=envs_idx.device).unsqueeze(0).expand(batch_size, -1),
            )

        # Check which environments need target randomization
        should_randomize = (state.info["randomize_target"]) & (state.info["step"] % 500 == 0)
        
        # Get default values for all environments
        default_vel_tar, default_ang_vel_tar = dont_randomize()
        
        # Initialize with default values
        vel_tar = default_vel_tar.clone()
        ang_vel_tar = default_ang_vel_tar.clone()
        
        # If any environment needs randomization, get random values and update only those
        if should_randomize.any():
            random_vel_tar, random_ang_vel_tar = self.sample_command(envs_idx[should_randomize])
            vel_tar[should_randomize] = random_vel_tar
            ang_vel_tar[should_randomize] = random_ang_vel_tar
        
        # Calculate ramped values based on step count
        state.info["vel_tar"] = torch.minimum(
            vel_tar * state.info["step"].unsqueeze(-1) * self._config.dt / self._config.ramp_up_time, 
            vel_tar
        )
        state.info["ang_vel_tar"] = torch.minimum(
            ang_vel_tar * state.info["step"].unsqueeze(-1) * self._config.dt / self._config.ramp_up_time,
            ang_vel_tar,
        )

        # extract rigid state
        rigid = new_sim_state.solvers_state[self._rigid_solver_idx]
        qpos_full = torch.as_tensor(rigid.qpos)
        qvel_full = torch.as_tensor(rigid.dofs_vel)
        qpos = qpos_full[envs_idx]
        qvel = qvel_full[envs_idx]

        # gait reward
        geoms_pos = self.rigid_solver.get_geoms_pos(self._feet_site_id, envs_idx=envs_idx)
        z_feet = geoms_pos[..., 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self._config.dt
        )
        # ensure z_feet_tar shape matches (batch, n_legs)
        b, n_legs = z_feet.shape
        if z_feet_tar.ndim == 1:
            # shape (n_legs,) -> (batch, n_legs)
            z_feet_tar = z_feet_tar.unsqueeze(0).repeat(b, 1)
        elif z_feet_tar.ndim == 2 and z_feet_tar.shape == (n_legs, b):
            # transpose if got shape (n_legs, batch)
            z_feet_tar = z_feet_tar.T
        elif z_feet_tar.ndim == 2 and z_feet_tar.shape != (b, n_legs):
            raise ValueError(f"Unexpected z_feet_tar shape {z_feet_tar.shape}, expected {(b, n_legs)}")
        reward_gaits = -torch.sum(((z_feet_tar - z_feet) / 0.05) ** 2, dim=-1)
        foot_contact_z = z_feet - self._foot_radius
        contact = foot_contact_z < 1e-3
        contact_filt_mm = contact | state.info["last_contact"]
        state.info["feet_air_time"] = state.info["feet_air_time"] + self._config.dt

        # position reward
        base_pos = qpos[:, :3]
        base_quat = qpos[:, 3:7]

        # upright reward
        vec_tar = torch.tensor([0.0, 0.0, 1.0], device=base_pos.device, dtype=base_pos.dtype)
        vec = gs_rotate(vec_tar, base_quat)
        reward_upright = -torch.sum((vec - vec_tar) ** 2, dim=-1)

        # yaw orientation reward
        yaw_tar = (
            state.info["yaw_tar"] + state.info["ang_vel_tar"][..., 2] * self._config.dt * state.info["step"]
        )
        yaw = gs_quat2euler(base_quat)[..., 2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -torch.square(torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw)))

        # velocity reward
        base_lin = qvel[:, :3]
        base_ang = qvel[:, 3:6] * (torch.pi / 180.0)
        vb = global_to_body_velocity(base_lin, base_quat)
        ab = global_to_body_velocity(base_ang, base_quat)
        reward_vel = -torch.sum((vb[:, :2] - state.info["vel_tar"][..., :2]) ** 2, dim=-1)
        reward_ang_vel = -torch.sum((ab[..., 2] - state.info["ang_vel_tar"][..., 2]) ** 2, dim=-1)

        # height reward
        reward_height = -torch.sum((base_pos[:, 2] - state.info["pos_tar"][..., 2]) ** 2, dim=-1)

        # total reward
        reward = (
            reward_gaits * 0.1
            + reward_upright * 0.5
            + reward_yaw * 0.3
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_height * 1.0
        )

        # done condition
        up = torch.tensor([0.0, 0.0, 1.0], device=base_pos.device)
        rot_up = gs_rotate(up, base_quat)  # (b,3)
        done = (rot_up * up).sum(dim=-1) < 0
        joint_angles = qpos[:, 7:]
        done |= torch.any(joint_angles < self.joint_range[:, 0], dim=-1)
        done |= torch.any(joint_angles > self.joint_range[:, 1], dim=-1)
        done |= base_pos[:, 2] < 0.18
        done = done.to(torch.float32)

        # update info
        state.info["step"] += 1
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["feet_air_time"] = state.info["feet_air_time"] * (~contact_filt_mm)
        state.info["last_contact"] = contact

        # return new State
        return State(
            sim_state=new_sim_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            info=state.info,
        )

    def _get_obs(self, sim_state, state_info: Dict[str, torch.Tensor], ctrl: torch.Tensor = None, envs_idx: torch.Tensor = None) -> torch.Tensor:
        # fetch the RigidSolverState
        rigid = sim_state.solvers_state[self._rigid_solver_idx]
        # full positions and velocities
        qpos_full = torch.as_tensor(rigid.qpos)
        qvel_full = torch.as_tensor(rigid.dofs_vel)
        # use only selected envs if provided
        qpos = qpos_full[envs_idx]
        qvel = qvel_full[envs_idx]
        # default zero control if not provided
        if ctrl is None:
            ctrl = torch.zeros_like(qpos[:, 7:])
        # body frame velocities
        base_lin = qvel[:, :3]
        base_ang = qvel[:, 3:6]
        base_quat = qpos[:, 3:7]
        vb = global_to_body_velocity(base_lin, base_quat)
        ab = global_to_body_velocity(base_ang * (torch.pi / 180.0), base_quat)
        joint_vel = qvel[:, 6:]
        # concat in Brax order
        obs = torch.cat([
            state_info["vel_tar"],
            state_info["ang_vel_tar"],
            ctrl,
            qpos,
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
        lin_vel_x = [-1.5, 1.5]  
        lin_vel_y = [-0.5, 0.5]  
        ang_vel_yaw = [-1.5, 1.5]
        
        # sample uniformly using gs_rand_float
        device = envs_idx.device
        batch_size = len(envs_idx)
        
        lin_x = gs_rand_float(lin_vel_x[0], lin_vel_x[1], batch_size, device)
        lin_y = gs_rand_float(lin_vel_y[0], lin_vel_y[1], batch_size, device)
        ang_z = gs_rand_float(ang_vel_yaw[0], ang_vel_yaw[1], batch_size, device)
        
        # Create velocity commands in same format
        lin_vel_cmd = torch.stack([lin_x, lin_y, torch.zeros(batch_size, device=device)], dim=1)
        ang_vel_cmd = torch.stack([
            torch.zeros(batch_size, device=device),
            torch.zeros(batch_size, device=device),
            ang_z
        ], dim=1)
        
        return lin_vel_cmd, ang_vel_cmd