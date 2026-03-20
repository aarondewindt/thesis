from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

def load_histories(runs_dir: Path) -> Dict[str, List[List[dict]]]:
    out: Dict[str, List[List[dict]]] = {}
    for variant_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        out[variant_dir.name] = []
        for seed_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            hist = seed_dir / "history.jsonl"
            if not hist.exists():
                continue
            rows = [json.loads(l) for l in hist.read_text(encoding="utf-8").splitlines() if l.strip()]
            out[variant_dir.name].append(rows)
    return out

def extract(hist: List[dict], key: str) -> List[float]:
    vals = []
    for r in hist:
        v = r.get(key)
        vals.append(float(v) if v is not None else float("nan"))
    return vals

def mean_std(curves: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not curves:
        return [], []
    T = min(len(c) for c in curves)
    means, stds = [], []
    for t in range(T):
        xs = [c[t] for c in curves if not math.isnan(c[t])]
        if not xs:
            means.append(float("nan"))
            stds.append(float("nan"))
            continue
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
        means.append(m)
        stds.append(v ** 0.5)
    return means, stds

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    data = load_histories(runs_dir)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Learning curves (episode_reward_mean)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean episode reward")

    for variant, seed_hists in data.items():
        curves = [extract(h, "episode_reward_mean") for h in seed_hists]
        m, s = mean_std(curves)
        if not m:
            continue
        x = list(range(len(m)))
        ax.plot(x, m, label=variant)
        ax.fill_between(x, [mi - si for mi, si in zip(m, s)], [mi + si for mi, si in zip(m, s)], alpha=0.2)

    ax.legend()
    out_path = Path(args.out) if args.out else (runs_dir / "learning_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
