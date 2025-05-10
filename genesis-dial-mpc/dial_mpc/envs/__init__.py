from .unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig

# Mapping from environment name to its configuration class
_configs = {
    "unitree_go2_walk": UnitreeGo2EnvConfig,
}

# Mapping from environment name to its environment class
_envs = {
    "unitree_go2_walk": UnitreeGo2Env,
}

def register_config(name: str, config: type):
    _configs[name] = config


def get_config(name: str):
    try:
        return _configs[name]
    except KeyError:
        raise ValueError(f"Environment config '{name}' not found.")


def register_environment(name: str, env_class: type):
    _envs[name] = env_class


def get_environment(name: str, config):
    try:
        env_class = _envs[name]
    except KeyError:
        raise ValueError(f"Environment '{name}' not found.")
    return env_class(config)
