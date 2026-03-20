import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs:", gpus)


import argparse, json, time
from dataclasses import asdict, dataclass
from pathlib import Path

import ray

from toy_mp.rllib.trainers import (
    build_multiagent_config_independent,
    build_multiagent_config_fully_shared,
    build_multiagent_config_three_actors_shared_critic,
    build_singleagent_config_monolithic
)

@dataclass
class RunMeta:
    variant: str
    env_yaml: str
    framework: str
    seed: int
    iters: int
    created_at_unix: float


VARIANTS = {
    "multiagent_config_independent": build_multiagent_config_independent,
    "multiagent_config_fully_shared": build_multiagent_config_fully_shared,
    "multiagent_config_three_actors_shared_critic": build_multiagent_config_three_actors_shared_critic,
    "singleagent_config_monolithic": build_singleagent_config_monolithic,
}

DEFAULT_VARIANTS = ["singleagent_config_monolithic"]


def train_one(variant: str, env_yaml: str, framework: str, seed: int, iters: int, out_dir: Path) -> None:
    cfg = VARIANTS[variant](env_yaml, framework=framework, num_workers=0, num_gpus=0).debugging(seed=seed)
    algo = cfg.build()

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = RunMeta(variant, env_yaml, framework, seed, iters, time.time())
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")

    with (out_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for i in range(iters):
            r = algo.train()
            row = {
                "iter": i,
                "episode_reward_mean": r.get("episode_reward_mean"),
                "episodes_total": r.get("episodes_total"),
                "timesteps_total": r.get("timesteps_total"),
                "env_runners": r.get("env_runners", {}),
            }
            f.write(json.dumps(row) + "\n")

    algo.stop()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-yaml", required=True)
    ap.add_argument("--output-dir", default="results/runs")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--framework", choices=["tf"], default="tf")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    args = ap.parse_args()

    ray.init(address="auto", include_dashboard=False)

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(args.output_dir) / ts

    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"Unknown variant: {v}. Options: {list(VARIANTS.keys())}")

    for v in args.variants:
        for s in range(args.seeds):
            run_dir = base / v / f"seed_{s}"
            print(f"[run] variant={v} seed={s} -> {run_dir}")
            train_one(v, args.env_yaml, args.framework, s, args.iters, run_dir)

    ray.shutdown()
    print(f"Done. Runs written under: {base}")

if __name__ == "__main__":
    main()
