import argparse
from typing import Dict, Any, Hashable

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from toy_mp.envs.conveyor_portal.env import ConveyorPortalEnv
from toy_mp.rllib.callbacks import ToyMetricsCallback
from toy_mp.rllib.multiagent_wrapper import SequentialPhaseMAEnv

from .models.shared_critic_model_tf import PhaseActorSharedCriticTFModel


def make_multi_agent_env(env_config: Dict):
    env = ConveyorPortalEnv.from_yaml(env_config["env_yaml"])
    return SequentialPhaseMAEnv(env)


def make_single_agent_env(env_config: Dict):
    return ConveyorPortalEnv.from_yaml(env_config["env_yaml"])


def phase_policy_mapping_independent(agent_id: Hashable, episode: Any, **kwargs) -> str:
    return {
        "phase1": "pi_phase1",
        "phase2": "pi_phase2",
        "phase3": "pi_phase3",
    }.get(str(agent_id), "pi_phase1")


def build_multiagent_config_independent(env_yaml: str, framework: str, num_gpus: int, num_workers: int) -> PPOConfig:
    register_env("toy_phase_ma", lambda cfg: make_multi_agent_env(cfg))

    dummy = make_multi_agent_env({"env_yaml": env_yaml})
    obs_space = dummy.observation_space
    act_space = dummy.action_space

    config = (
        PPOConfig()
        .environment(env="toy_phase_ma", env_config={"env_yaml": env_yaml})
        .framework(framework=framework)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=num_workers)
        .callbacks(ToyMetricsCallback)
        .training(train_batch_size=4000)
        .multi_agent(
            policies={
                "pi_phase1": (None, obs_space, act_space, {}),
                "pi_phase2": (None, obs_space, act_space, {}),
                "pi_phase3": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=phase_policy_mapping_independent,
            policies_to_train=["pi_phase1", "pi_phase2", "pi_phase3"],
        )
    )
    return config



def phase_policy_mapping_fully_shared(agent_id: Hashable, episode: Any, **kwargs) -> str:
    return "pi_shared"


def build_multiagent_config_fully_shared(env_yaml: str, framework: str, num_gpus: int, num_workers: int) -> PPOConfig:
    register_env("toy_phase_ma", lambda cfg: make_multi_agent_env(cfg))
    
    dummy = make_multi_agent_env({"env_yaml": env_yaml})
    obs_space = dummy.observation_space
    act_space = dummy.action_space

    config = (
        PPOConfig()
        .environment(env="toy_phase_ma", env_config={"env_yaml": env_yaml})
        .framework(framework=framework)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=num_workers)
        .callbacks(ToyMetricsCallback)
        .training(train_batch_size=4000)
        .multi_agent(
            policies={"pi_shared": (None, obs_space, act_space, {})},
            policy_mapping_fn=phase_policy_mapping_fully_shared,
            policies_to_train=["pi_shared"],
        )
    )
    return config



def phase_policy_mapping_shared_critic(agent_id: Hashable, episode: Any, **kwargs) -> str:
    return {
        "phase1": "pi_phase1",
        "phase2": "pi_phase2",
        "phase3": "pi_phase3",
    }.get(str(agent_id), "pi_phase1")


def build_multiagent_config_three_actors_shared_critic(
    env_yaml: str,
    framework: str,
    num_gpus: int = 0,
    num_workers: int = 0,
) -> PPOConfig:
    register_env("toy_phase_ma", lambda cfg: make_multi_agent_env(cfg))

    dummy = make_multi_agent_env({"env_yaml": env_yaml})
    obs_space = dummy.observation_space
    act_space = dummy.action_space

    model_cfg = {"custom_model": PhaseActorSharedCriticTFModel}

    return (
        PPOConfig()
        .environment(env="toy_phase_ma", env_config={"env_yaml": env_yaml})
        .framework(framework=framework)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=num_workers)
        .callbacks(ToyMetricsCallback)
        .training(train_batch_size=4000)
        .multi_agent(
            policies={
                "pi_phase1": (None, obs_space, act_space, {"model": model_cfg}),
                "pi_phase2": (None, obs_space, act_space, {"model": model_cfg}),
                "pi_phase3": (None, obs_space, act_space, {"model": model_cfg}),
            },
            policy_mapping_fn=phase_policy_mapping_shared_critic,
            policies_to_train=["pi_phase1", "pi_phase2", "pi_phase3"],
        )
    )



def build_singleagent_config_monolithic(env_yaml: str, framework: str, num_gpus: int, num_workers: int) -> PPOConfig:
    register_env("toy_single", lambda cfg: make_single_agent_env(cfg))
    config = (
        PPOConfig()
        .environment(env="toy_single", env_config={"env_yaml": env_yaml})
        .framework(framework=framework)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=num_workers)
        .callbacks(ToyMetricsCallback)
        .training(train_batch_size=4000)
    )
    return config

