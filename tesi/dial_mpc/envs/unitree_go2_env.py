from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

from functools import partial

import mujoco
import genesis.utils.mjcf as mjcf
import torch
import genesis as gs

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
