import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.scene import Scene
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs: int = num_envs
        self.num_obs: int = obs_cfg["num_obs"] #48
        self.num_privileged_obs = None # for policy_runner
        self.num_actions: int = env_cfg["num_actions"]#12
        self.num_commands: int = command_cfg["num_commands"]#3

        self.simulate_action_latency: bool = True  # there is a 1 step latency on real robot
        self.dt: float = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length: int = math.ceil(env_cfg["episode_length_s"] / self.dt) # 20/0.02 = 1000 steps

        self.env_cfg: dict = env_cfg
        self.obs_cfg: dict = obs_cfg
        self.reward_cfg: dict = reward_cfg
        self.command_cfg: dict = command_cfg

        self.obs_scales: dict = obs_cfg["obs_scales"]
        self.reward_scales: dict = reward_cfg["reward_scales"]
        
        # Camera and recording related variables
        self.headless: bool = not show_viewer
        self._recording: bool = False
        self._recorded_frames: list = []

        # create scene
        self.scene: Scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat) # inverse of the initial orientation quaternion of the robot's base
        self.robot: RigidEntity  = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                links_to_keep=self.env_cfg['links_to_keep'],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

    
        self._set_camera()

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs: list[int] = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]# mapping from joint names to indices

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs) #apply kp to each joint
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)# apply kd to each joint

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)# buffer to store episode length for each environment
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        self.extras = dict()  # extra information for logging
        ## not used for now ##
        # self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        # self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        # for i in range(self.dof_pos_limits.shape[0]):
        #     # soft limits
        #     m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
        #     r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
        #     self.dof_pos_limits[i, 0] = (
        #         m - 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
        #     )
        #     self.dof_pos_limits[i, 1] = (
        #         m + 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
        #     )

        def find_link_indices(names):
            """Finds the indices of the links in the robot that match the given names."""
            link_indices = list()

            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_link_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        ) # not used for now
        self.penalized_contact_link_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        ) # indices of the links that we want to penalize for contact forces
        self.feet_link_indices = find_link_indices(
            self.env_cfg['feet_link_names']
        ) # indices of feet links

        ### foot related stuff ###
        self.feet_link_indices_world_frame = [i+1 for i in self.feet_link_indices]

        ##  gait control ##
        self.foot_positions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        ) #not used for now
        self.foot_quaternions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 4, device=self.device, dtype=gs.tc_float,
        )# not used for now
        self.foot_velocities = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )
    
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions): #actions come from neural net
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]) #clippimng to prevent extreme values
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions  #simulate action latency
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos # transform normalized policy actions into target joint positions
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)# send target positions to robot's PD controllers
        self.scene.step() # advance the simulation by one step

        # update buffers
        self.episode_length_buf += 1 # increment episode length
        self.base_pos[:] = self.robot.get_pos() # get fresh robot base position
        self.base_quat[:] = self.robot.get_quat() # get fresh robot base orientation
        self.base_euler = quat_to_xyz( 
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        ) # get fresh robot base orientation in euler angles (roll, pitch, yaw)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat) # transform world lin vel to robot's base frame
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)# transform world ang vel to robot's base frame
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat) # project gravity vector to robot's base frame
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)# update joint positions
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)# update joint velocities
        self.foot_positions[:] = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame)# update foot positions
        self.foot_quaternions[:] = self.rigid_solver.get_links_quat(self.feet_link_indices_world_frame) # update foot orientations
        self.foot_velocities[:] = self.rigid_solver.get_links_vel(self.feet_link_indices_world_frame)# update foot velocities
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )# net force applied on each links due to direct external contacts, shape (num_envs, num_links, 3)
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        ) # will contain the indices of all environments that have reached a multiple of 200 timesteps, and thus need resampling
        self._resample_commands(envs_idx) # resample commands for those environments

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length# if we reached 1000 steps, we need to reset the environment
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]# if pitch(forward/backward tilt) is greater than x degrees, we need to reset the environment
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]# if roll(side-to-side tilt) is greater than x degrees, we need to reset the environment

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()# environments that have reached the max episode length
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0 # used by PPO to handle terminated vs. truncated episodes differently

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())# reset the environments that need to be reset

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3, the robot's linear velocity in its base frame(3d)
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3, the robot's angular velocity in its base frame(3d)
                self.projected_gravity,  # 3, gravity vector in the robot's base frame, indicating its orientation
                self.commands * self.commands_scale,  # 3, target velocities the robot should achieve
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12, current joint angles relative to default
                self.dof_vel * self.obs_scales["dof_vel"],  # 12, current joint velocities 
                self.actions,  # 12 # previous actions issued by the policy
            ],
            axis=-1,
        )

        # Render for recording if enabled
        self._render_headless()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions---------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(
                self.commands[:, :2] - self.base_lin_vel[:, :2]
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return torch.exp(-ang_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.base_pos[:, 2]
        base_height_target = self.reward_cfg['base_height_target']
        return torch.square(base_height - base_height_target)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.link_contact_forces[:, self.penalized_contact_link_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
    
            dim=1,
        )
    def _reward_absolute_lin_vel(self):
        # Reward absolute linear velocity (encourages speed in any direction)
        # We take the norm of the horizontal velocity (x and y components)
        absolute_velocity = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return absolute_velocity

    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
    #     out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
    #     return torch.sum(out_of_limits, dim=1)
    
    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(
    #         torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
    #     )


    # ------------ Camera and recording functions ----------------
    def _set_camera(self):
        '''Set camera position and direction for recording'''
        self._floating_camera = self.scene.add_camera(
            pos=np.array([0, -1, 1]),
            lookat=np.array([0, 0, 0]),
            fov=40,
            GUI=False,
        )

    def _render_headless(self):
        '''Render frames for recording when in headless mode'''
        if self._recording and len(self._recorded_frames) < 150:
            robot_pos = np.array(self.base_pos[0].cpu())
            self._floating_camera.set_pose(
                pos=robot_pos + np.array([-1, -1, 0.5]), 
                lookat=robot_pos + np.array([0, 0, -0.1])
            )
            frame, _, _, _ = self._floating_camera.render()
            self._recorded_frames.append(frame)

    def get_recorded_frames(self):
        '''Return the recorded frames and reset recording state'''
        if len(self._recorded_frames) == 150:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None

    def start_recording(self, record_internal=True):
        '''Start recording frames'''
        self._recorded_frames = []
        self._recording = True
        if not record_internal:
            self._floating_camera.start_recording()

    def stop_recording(self, save_path=None):
        '''Stop recording and optionally save to a file'''
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self._floating_camera.stop_recording(save_path, fps=int(1 / self.dt))