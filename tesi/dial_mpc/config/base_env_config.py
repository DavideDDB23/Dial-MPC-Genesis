from dataclasses import dataclass


@dataclass
class BaseEnvConfig:
    randomize_tasks: bool = False  # Whether to randomize the task.
    # P gain, or a list of P gains for each joint.
    kp: float = 30.0
    # D gain, or a list of D gains for each joint.
    kd: float = 1.0
    # dt
    dt: float = 0.02
    # timestep of the underlying simulator step. user is responsible for making sure it matches their model.
    timestep: float = 0.02
    backend: str = "cpu"  # hardware & Genesis backend
    # control method for the joints, either "torque" or "position"
    leg_control: str = "torque"
    action_scale: float = 1.0  # scale of the action space.
    n_envs: int = 1