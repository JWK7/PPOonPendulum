from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()

register(
    id="Pendulum-v1-custom",
    entry_point="gym.envs.classic_control:CustomPendulumEnv",
    reward_threshold=200,
)
