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
from scipy.interpolate import InterpolatedUnivariateSpline
import functools

from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig

plt.style.use("science")

def rollout_us(step_env, state, us):
    """
    Rollout a sequence of controls through the environment.
    
    Args:
        step_env: The environment step function
        state: Initial state (State object)
        us: Sequence of controls to apply (batch_size, horizon, action_dim)
        
    Returns:
        rewards: Rewards for each step (batch_size, horizon)
        pipeline_states: Pipeline states for each step (batch_size, horizon)
    """
    batch_size = us.shape[0]
    horizon = us.shape[1]
    
    # Initialize lists to store results
    rewards = []
    pipeline_states = []
    
    # Create environment indices for the batch
    envs_idx = torch.arange(batch_size)
    
    # Rollout through the horizon
    current_state = state
    for t in range(horizon):
        # Get current control for all environments
        current_u = us[:, t]
        
        # Step the environment
        current_state = step_env(current_state, current_u, envs_idx)
        
        # Store results
        rewards.append(current_state.reward)
        pipeline_states.append(current_state.sim_state)
    
    # Stack results along the horizon dimension
    rewards = torch.stack(rewards, dim=1)  # (batch_size, horizon)
    pipeline_states = pipeline_states  # List of pipeline states
    
    return rewards, pipeline_states

def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = torch.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = len(env.motor_dofs)
        
        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        # Initialize sigma values for diffusion
        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = torch.log(torch.tensor(sigma1 / sigma0)) / args.Ndiffuse
        self.sigmas = A * torch.exp(B * torch.arange(args.Ndiffuse))
        self.sigma_control = (
            args.horizon_diffuse_factor ** torch.arange(args.Hnode + 1).flip(0)
        )

        # Setup control and node timesteps
        self.ctrl_dt = 0.02
        self.step_us = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # Setup functions for control and node conversion
        self.rollout_us = rollout_us
        self.node2u = self.node2u
        self.u2node = self.u2node

    def node2u(self, nodes):
        """
        Convert node values to control values using spline interpolation.
        
        Args:
            nodes: Node values (batch_size, n_nodes, action_dim)
            
        Returns:
            us: Control values (batch_size, n_steps, action_dim)
        """
        batch_size = nodes.shape[0]
        n_nodes = nodes.shape[1]
        action_dim = nodes.shape[2]
        
        # Reshape for spline interpolation
        nodes_flat = nodes.reshape(-1, n_nodes)
        
        # Create spline for each batch and action dimension
        us = []
        for i in range(batch_size * action_dim):
            spline = InterpolatedUnivariateSpline(
                self.step_nodes.numpy(), 
                nodes_flat[i].numpy(), 
                k=2
            )
            u = spline(self.step_us.numpy())
            us.append(u)
        
        # Reshape back to original dimensions
        us = torch.tensor(us).reshape(batch_size, action_dim, -1).transpose(1, 2)
        return us

    def u2node(self, us):
        """
        Convert control values to node values using spline interpolation.
        
        Args:
            us: Control values (batch_size, n_steps, action_dim)
            
        Returns:
            nodes: Node values (batch_size, n_nodes, action_dim)
        """
        batch_size = us.shape[0]
        n_steps = us.shape[1]
        action_dim = us.shape[2]
        
        # Reshape for spline interpolation
        us_flat = us.reshape(-1, n_steps)
        
        # Create spline for each batch and action dimension
        nodes = []
        for i in range(batch_size * action_dim):
            spline = InterpolatedUnivariateSpline(
                self.step_us.numpy(), 
                us_flat[i].numpy(), 
                k=2
            )
            node = spline(self.step_nodes.numpy())
            nodes.append(node)
        
        # Reshape back to original dimensions
        nodes = torch.tensor(nodes).reshape(batch_size, action_dim, -1).transpose(1, 2)
        return nodes

    def reverse_once(self, state, Ybar_i, noise_scale):
        """
        Perform one step of the reverse diffusion process.
        
        Args:
            state: Current state of the environment
            Ybar_i: Current node values (batch_size, n_nodes, action_dim)
            noise_scale: Scale of noise to add (n_nodes,)
            
        Returns:
            Ybar: Updated node values (batch_size, n_nodes, action_dim)
            info: Dictionary containing rewards, states, and other information
        """
        batch_size = Ybar_i.shape[0]
        
        # Sample from q_i
        eps_Y = torch.randn(
            (self.args.Nsample, self.args.Hnode + 1, self.nu),
            device=Ybar_i.device
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        # We can't change the first control
        Y0s[:, 0] = Ybar_i[0]
        # Append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = torch.cat([Y0s, Ybar_i.unsqueeze(0)], dim=0)
        Y0s = torch.clamp(Y0s, -1.0, 1.0)
        
        # Convert Y0s to us
        us = self.node2u(Y0s)
        
        # Estimate mu_0tm1
        rews, pipeline_states = self.rollout_us(self.env.step, state, us)
        rew_Ybar_i = rews[-1].mean()
        
        # Extract states from pipeline
        qss = []
        qdss = []
        xss = []
        for ps in pipeline_states:
            # Get rigid solver state
            rigid = ps.solvers_state[self.env._rigid_solver_idx]
            # Extract positions and velocities
            qpos = torch.as_tensor(rigid.qpos)
            qvel = torch.as_tensor(rigid.dofs_vel)
            # Get foot positions
            feet_pos = self.env.rigid_solver.get_geoms_pos(
                self.env._feet_site_id, 
                envs_idx=torch.arange(batch_size)
            )
            qss.append(qpos)
            qdss.append(qvel)
            xss.append(feet_pos)
        
        # Stack states
        qss = torch.stack(qss, dim=0)  # (n_samples, batch_size, n_dofs)
        qdss = torch.stack(qdss, dim=0)  # (n_samples, batch_size, n_dofs)
        xss = torch.stack(xss, dim=0)  # (n_samples, batch_size, n_feet, 3)
        
        # Compute rewards and weights
        rews = rews.mean(dim=-1)
        logp0 = (rews - rew_Ybar_i) / rews.std(dim=-1) / self.args.temp_sample
        weights = torch.nn.functional.softmax(logp0, dim=0)
        
        # Update Ybar and noise scale
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)
        
        # Update with reward
        Ybar = torch.einsum('n,nij->ij', weights, Y0s)
        qbar = torch.einsum('n,nij->ij', weights, qss)
        qdbar = torch.einsum('n,nij->ij', weights, qdss)
        xbar = torch.einsum('n,nijk->ijk', weights, xss)
        
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
        Perform the reverse diffusion process.
        
        Args:
            state: Initial state of the environment
            YN: Initial node values (batch_size, n_nodes, action_dim)
            
        Returns:
            Yi: Final node values after reverse diffusion
        """
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                # Compute noise scale for current step
                noise_scale = self.sigmas[i] * torch.ones(self.args.Hnode + 1)
                
                # Perform one step of reverse diffusion
                Yi, rews = self.reverse_once(state, Yi, noise_scale)
                
                # Update progress bar
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({
                    "rew": f"{rews.mean():.2e}",
                    "freq": f"{freq:.2f}"
                })
        
        return Yi

    def shift(self, Y):
        """
        Shift the control sequence by one step.
        
        Args:
            Y: Node values (batch_size, n_nodes, action_dim)
            
        Returns:
            Y: Shifted node values (batch_size, n_nodes, action_dim)
        """
        # Convert nodes to control values
        u = self.node2u(Y)
        
        # Shift control sequence by one step
        u = torch.roll(u, -1, dims=0)
        # Set last control to zero
        u[-1] = torch.zeros(self.nu)
        
        # Convert back to node values
        Y = self.u2node(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        """
        Shift the control sequence by n steps.
        
        Args:
            u: Control values (batch_size, n_steps, action_dim)
            n_step: Number of steps to shift
            
        Returns:
            Y: Shifted node values (batch_size, n_nodes, action_dim)
        """
        # Shift control sequence by n steps
        u = torch.roll(u, -n_step, dims=0)
        # Set last n_step controls to zero
        u[-n_step:] = torch.zeros_like(u[-n_step:])
        
        # Convert to node values
        Y = self.u2node(u)
        return Y


def main():
    art.tprint("DDB @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
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

    # Load DIAL-MPC configuration
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    torch.manual_seed(dial_config.seed)

    # Load environment configuration
    env_config = load_dataclass_from_dict(
        UnitreeGo2EnvConfig, 
        config_dict, 
        convert_list_to_array=True
    )
    
    # Set n_envs to Nsample + 1 for proper vectorization
    env_config.n_envs = dial_config.Nsample + 1

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = UnitreeGo2Env(env_config)
    mbdpi = MBDPI(dial_config, env)

    # Initialize environment
    state_init =  env.reset(torch.arange(env_config.n_envs))

    # Initialize control sequence
    YN = torch.zeros([dial_config.Hnode + 1, mbdpi.nu])

    Y0 = YN

    # Main control loop
    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # Forward single step
            state = env.step(state, Y0[0], torch.arange(env_config.n_envs))
            rollout.append(state.sim_state)
            rews.append(state.reward)
            us.append(Y0[0])

            # Update Y0
            Y0 = mbdpi.shift(Y0)

            t0 = time.time()
            # Compute diffusion factors
            traj_diffuse_factors = (
                mbdpi.sigma_control * dial_config.traj_diffuse_factor ** torch.arange(dial_config.Ndiffuse)
            ).unsqueeze(1)
            
            # Iterate over diffusion factors
            for factor in traj_diffuse_factors:
                Y0, info = mbdpi.reverse_once(state, Y0, factor)
            
            # Update progress bar
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({
                "rew": f"{state.reward.mean():.2e}",
                "freq": f"{freq:.2f}"
            })

    # Compute mean reward
    rew = torch.tensor(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # Create result directory if it doesn't exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print("Processing rollout for visualization")

    # Save the rollout
    data = []
    xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        # Extract rigid solver state
        rigid = pipeline_state.solvers_state[env._rigid_solver_idx]
        # Get positions and velocities
        qpos = torch.as_tensor(rigid.qpos)
        qvel = torch.as_tensor(rigid.dofs_vel)
        # Get foot positions
        feet_pos = env.rigid_solver.get_geoms_pos(
            env._feet_site_id,
            envs_idx=torch.arange(env_config.n_envs)
        )
        
        # Get current control
        ctrl = us[i]  # This is the control applied at step i
        
        data.append(
            torch.cat([
                torch.tensor([i]),
                qpos,
                qvel,
                ctrl
            ])
        )
        xdata.append(feet_pos)
    
    # Convert to tensors and save
    data = torch.stack(data)
    xdata = torch.stack(xdata)
    
    # Save with descriptive names
    torch.save(data, os.path.join(dial_config.output_dir, f"{timestamp}_states.pt"))
    torch.save(xdata, os.path.join(dial_config.output_dir, f"{timestamp}_predictions.pt"))


if __name__ == "__main__":
    main()
