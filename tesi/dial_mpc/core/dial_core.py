import os
import time
from dataclasses import dataclass
import importlib
import sys

sys.path.insert(0, os.path.abspath('tesi'))

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import art
import emoji

import torch
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import genesis as gs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.envs.unitree_go2_env import UnitreeGo2EnvConfig, UnitreeGo2Env

plt.style.use("science")

def rollout_batch(env, initial_state, us_batch):
        """
        Performs parallel rollouts of Nsample trajectories using Genesis batch processing.
        
        Args:
            env: Environment
            initial_state: Initial state for all environments
            us_batch: Action tensor of shape (Nsample, Hsample, nu)
            
        Returns:
            rewards: Reward tensor of shape (Nsample, Hsample)
            pipeline_states: Object containing state data for all batches
        """
        Nsample = us_batch.shape[0]
        Hsample = us_batch.shape[1]
        nu = us_batch.shape[2]
        
        # Prepare data structures to collect results
        all_rewards = []
        all_q = []
        all_qd = []
        all_x_pos = []
        
        # Prepare a batch of Nsample environments
        envs_idx = torch.arange(Nsample)
        state_batch = initial_state
        
        # Collect reward
        # Hsample simulation steps
        for t in range(Hsample):
            # Apply actions for this timestep to all environments in parallel
            actions_t = us_batch[:, t]  # Shape: (Nsample, nu)
            state_batch = env.step(state_batch, actions_t, envs_idx=envs_idx)
            
            # Collect reward
            all_rewards.append(state_batch.reward)
            
            # Extract state data
            rigid = state_batch.sim_state.solvers_state[env._rigid_solver_idx]
            
            # Collect positions and velocities
            qpos = torch.tensor(rigid.qpos)[envs_idx]
            qvel = torch.tensor(rigid.dofs_vel)[envs_idx]
            
            all_q.append(qpos)
            all_qd.append(qvel)
            all_x_pos.append(qpos[:, :3])
        
        # Convert lists to tensors
        rewards = torch.stack(all_rewards, dim=1)  # Shape: (Nsample, Hsample)
        
        # Create structure for pipeline_states
        class PipelineStates:
            def __init__(self, q, qd, x_pos):
                self.q = torch.stack(q, dim=1)      # Shape: (Nsample, Hsample, n_q)
                self.qd = torch.stack(qd, dim=1)    # Shape: (Nsample, Hsample, n_v)
                self.x = type('', (), {})()
                self.x.pos = torch.stack(x_pos, dim=1)  # Shape: (Nsample, Hsample, 3)
        
        pipeline_states = PipelineStates(all_q, all_qd, all_x_pos)
        
        return rewards, pipeline_states


def softmax_update(weights, Y0s, sigma, mu_0t):
    """
    Performs softmax update of control trajectories.
    
    Args:
        weights: Softmax weights for each trajectory of shape (Nsample,)
        Y0s: Control trajectories tensor of shape (Nsample, Hnode+1, nu)
        sigma: Noise scale
        mu_0t: Current mean trajectory (not used in our case)
        
    Returns:
        mu_0tm1: New mean trajectory
        sigma: Noise scale (unchanged)
    """
    # Ensure tensors have the same type (float32)
    weights = weights.to(torch.float32)
    Y0s = Y0s.to(torch.float32)
    
    # Calculate weighted averages (same functionality as jnp.einsum)
    mu_0tm1 = torch.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = len(env.motor_dofs)

        self.n_envs = args.Nsample + 1

        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = torch.log(torch.tensor(sigma1 / sigma0)) / args.Ndiffuse
        self.sigmas = A * torch.exp(B * torch.arange(args.Ndiffuse))
        self.sigma_control = (
            args.horizon_diffuse_factor ** torch.arange(args.Hnode + 1).flip(0)
        )

        # Parameters for node-action interpolation
        self.ctrl_dt = 0.02
        self.step_us = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)
        
        self.envs_idx = torch.arange(args.Nsample)

    def node2u(self, nodes):
        """
        Converts control nodes to actions using spline interpolation.
        
        Args:
            nodes: Nodes tensor of shape (Hnode+1, nu) or (batch, Hnode+1, nu)
            
        Returns:
            us: Actions tensor of shape (Hsample+1, nu) or (batch, Hsample+1, nu)
        """
        # Convert to numpy for SciPy
        if nodes.ndim == 2:
            # Single trajectory
            nodes_np = nodes.cpu().numpy()
            step_nodes_np = self.step_nodes.cpu().numpy()
            step_us_np = self.step_us.cpu().numpy()
            
            # Interpolate for each control dimension
            us_np = np.zeros((len(step_us_np), nodes_np.shape[1]))
            for i in range(nodes_np.shape[1]):
                spline = InterpolatedUnivariateSpline(step_nodes_np, nodes_np[:, i], k=2)
                us_np[:, i] = spline(step_us_np)
            
            # Convert back to PyTorch
            us = torch.from_numpy(us_np).to(nodes.device)
            return us
        else:
            # Batch of trajectories
            batch_size = nodes.shape[0]
            us_list = []
            for b in range(batch_size):
                us_list.append(self.node2u(nodes[b]))
            return torch.stack(us_list)

    def u2node(self, us):
        """
        Converts actions to control nodes using spline interpolation.
        
        Args:
            us: Actions tensor of shape (Hsample+1, nu) or (batch, Hsample+1, nu)
            
        Returns:
            nodes: Nodes tensor of shape (Hnode+1, nu) or (batch, Hnode+1, nu)
        """
        # Convert to numpy for SciPy
        if us.ndim == 2:
            # Single trajectory
            us_np = us.cpu().numpy()
            step_nodes_np = self.step_nodes.cpu().numpy()
            step_us_np = self.step_us.cpu().numpy()
            
            # Interpolate for each control dimension
            nodes_np = np.zeros((len(step_nodes_np), us_np.shape[1]))
            for i in range(us_np.shape[1]):
                spline = InterpolatedUnivariateSpline(step_us_np, us_np[:, i], k=2)
                nodes_np[:, i] = spline(step_nodes_np)
            
            # Convert back to PyTorch
            nodes = torch.from_numpy(nodes_np).to(us.device)
            return nodes
        else:
            # Batch of trajectories
            batch_size = us.shape[0]
            nodes_list = []
            for b in range(batch_size):
                nodes_list.append(self.u2node(us[b]))
            return torch.stack(nodes_list)

    def reverse_once(self, state, Ybar_i, noise_scale):
        """
        Performs one iteration of the "reverse diffusion" process, sampling control trajectories
        and updating the mean trajectory based on obtained rewards.
        
        Args:
            state: Current environment state
            Ybar_i: Current mean trajectory of shape (Hnode+1, nu)
            noise_scale: Noise scale for this iteration
            
        Returns:
            Ybar: New mean trajectory
            info: Dictionary with diffusion information
        """
        Ybar_i = Ybar_i.to(torch.float32)
        noise_scale = noise_scale.to(torch.float32)

        eps_Y = torch.randn(
            (self.args.Nsample, self.args.Hnode + 1, self.nu), dtype=torch.float32
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        
        # we can't change the first control
        Y0s[:, 0] = Ybar_i[0]
        
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Ybar_i_expanded = Ybar_i.unsqueeze(0)  # shape: (1, Hnode+1, nu)
        Y0s = torch.cat([Y0s, Ybar_i_expanded], dim=0)
        Y0s = torch.clamp(Y0s, -1.0, 1.0)
        
        # Convert control nodes to actions
        us = self.node2u(Y0s)  # shape: (Nsample+1, Hsample+1, nu)
        
        envs_idx = torch.arange(Y0s.shape[0])
        state_batch = self.env.reset(envs_idx=envs_idx)
        
        # esitimate mu_0tm1
        rewss, pipeline_statess = rollout_batch(self.env, state_batch, us)
        rew_Ybar_i = rewss[-1].mean()
        qss = pipeline_statess.q.to(torch.float32)
        qdss = pipeline_statess.qd.to(torch.float32)
        xss = pipeline_statess.x.pos.to(torch.float32)
        rews = rewss.mean(dim=-1)
        logp0 = (rews - rew_Ybar_i) / rews.std(dim=-1) / self.args.temp_sample
        
        weights = torch.nn.functional.softmax(logp0, dim=0)
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)
                
        # Calculate weighted averages 
        weights = weights.to(torch.float32)
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)
        qbar = torch.einsum("n,nij->ij", weights, qss)
        qdbar = torch.einsum("n,nij->ij", weights, qdss)
        xbar = torch.einsum("n,nij->ij", weights, xss)
        
        # Prepare info
        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": new_noise_scale,
        }
        
        return Ybar, info

    def reverse(self, state, YN):
        """
        Performs the complete reverse diffusion process, executing multiple iterations of reverse_once
        with gradually decreasing noise.
        
        Args:
            state: Current environment state
            YN: Initial trajectory (generally all zeros) of shape (Hnode+1, nu)
            
        Returns:
            Yi: Optimized trajectory
        """
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                Yi, info = self.reverse_once(
                    state, Yi, self.sigmas[i] * torch.ones(self.args.Hnode + 1)
                )
                if torch.is_tensor(Yi): 
                    Yi = Yi.contiguous()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{info['rews'].mean().item():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    def shift(self, Y):
        """
        Shifts the trajectory one step forward, setting the last control to zero.
        
        Args:
            Y: Control trajectory of shape (Hnode+1, nu)
            
        Returns:
            Y_shifted: Shifted trajectory
        """
        # Convert control nodes to actions
        u = self.node2u(Y)
        
        # Shift actions forward by one step
        u_shifted = torch.roll(u, -1, dims=0)
        
        # Set last action to zero
        u_shifted[-1] = torch.zeros(self.nu)
        
        # Convert back to nodes
        Y_shifted = self.u2node(u_shifted)
        
        return Y_shifted

    def shift_Y_from_u(self, u, n_step):
        """
        Shifts the action trajectory by n_step steps and converts it to nodes.
        
        Args:
            u: Action trajectory of shape (Hsample+1, nu)
            n_step: Number of steps to shift
            
        Returns:
            Y: Shifted node trajectory
        """
        # Shift actions forward by n_step steps
        u_shifted = torch.roll(u, -n_step, dims=0)
        
        # Set last n_step actions to zero
        u_shifted[-n_step:] = torch.zeros_like(u_shifted[-n_step:])
        
        # Convert to nodes
        Y = self.u2node(u_shifted)
        
        return Y

def main():
    """
    Main function that executes the DIAL-MPC algorithm
    """
    art.tprint("LeCAR @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    torch.manual_seed(dial_config.seed)

    env_config = load_dataclass_from_dict(
        UnitreeGo2EnvConfig, config_dict, convert_list_to_array=True
    )
    
    env_config.n_envs = dial_config.Nsample + 1

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = UnitreeGo2Env(env_config)
    mbdpi = MBDPI(dial_config, env)

    state_init = env.reset(envs_idx=torch.tensor([0]))

    YN = torch.zeros([dial_config.Hnode + 1, mbdpi.nu])
    Y0 = YN

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step
            state = env.step(state, Y0[0], envs_idx=torch.tensor([0]))
            rollout.append(state)
            rews.append(state.reward.item())
            us.append(Y0[0].clone())

            # update Y0
            Y0 = mbdpi.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Starting DIAL-MPC optimization")

            t0 = time.time()
            
            traj_diffuse_factors = (
                mbdpi.sigma_control * dial_config.traj_diffuse_factor ** torch.arange(n_diffuse)[:, None]
            )
            for i in range(n_diffuse):
                Y0, info = mbdpi.reverse_once(state, Y0, traj_diffuse_factors[i])
                infos.append(info)
            
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward.item():.2e}", "freq": f"{freq:.2f}"})

    rew = sum(rews) / len(rews)
    print(f"mean reward = {rew:.2e}")

    # Create output directory if it doesn't exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save rollout
    data = []
    xdata = []
    for i, state in enumerate(rollout):
        rigid = state.sim_state.solvers_state[env._rigid_solver_idx]
        # Select only the first environment's state
        qpos = torch.tensor(rigid.qpos)[0] 
        qvel = torch.tensor(rigid.dofs_vel)[0]
        
        data_i = torch.cat([
            torch.tensor([i]).float(),
            qpos.flatten(), 
            qvel.flatten(),
            torch.tensor(us[i])
        ])
        data.append(data_i)
        
        if i < len(infos):
            xdata.append(infos[i]["xbar"][-1])
    
    data_tensor = torch.stack(data)
    xdata_tensor = torch.stack(xdata)
    
    np.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data_tensor.numpy())
    np.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata_tensor.numpy())
    
    print(f"Results saved in {dial_config.output_dir}")

if __name__ == "__main__":
    main()