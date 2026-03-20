from __future__ import annotations

import argparse
from typing import Dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from toy_mp.envs.conveyor_portal.env import ConveyorPortalEnv
from toy_mp.rllib.callbacks import ToyMetricsCallback
from toy_mp.rllib.multiagent_wrapper import SequentialPhaseMAEnv
from toy_mp.rllib.policy_mapping import phase_policy_mapping_fn


def make_env(env_config: Dict):
    env = ConveyorPortalEnv.from_yaml(env_config["env_yaml"])
    return SequentialPhaseMAEnv(env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-yaml", required=True, help="Path to MapSpec YAML.")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--local-mode", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--framework", choices=["tf2"], default="tf2")
    ap.add_argument("--num-gpus", type=int, default=0)
    args = ap.parse_args()

    ray.init(local_mode=args.local_mode, include_dashboard=False)
    register_env("toy_phase_ma", lambda cfg: make_env(cfg))

    dummy = make_env({"env_yaml": args.env_yaml})
    obs_space = dummy.observation_space
    act_space = dummy.action_space

    config = (
        PPOConfig()
        .environment(env="toy_phase_ma", env_config={"env_yaml": args.env_yaml})
        .framework(framework=args.framework)
        .resources(num_gpus=args.num_gpus)
        .env_runners(num_env_runners=args.num_workers)
        .callbacks(ToyMetricsCallback)
        .training(train_batch_size=4000)
        .multi_agent(
            policies={
                "pi_phase1": (None, obs_space, act_space, {}),
                "pi_phase2": (None, obs_space, act_space, {}),
                "pi_phase3": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=phase_policy_mapping_fn,
            policies_to_train=["pi_phase1", "pi_phase2", "pi_phase3"],
        )
    )

    algo = config.build()
    for i in range(args.iters):
        result = algo.train()
        # Updated key name for modern RLlib versions
        reward_mean = result.get('env_runners/episode_reward_mean', 
                                result.get('episode_reward_mean', 'N/A'))
        print(f"iter={i} reward_mean={reward_mean}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
