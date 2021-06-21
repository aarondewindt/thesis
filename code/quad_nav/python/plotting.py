from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib


def main():
    plt.style.use("ggplot")
    results_dir_path = Path(__file__).parent.parent
    plot_path = Path(__file__).parent.parent.parent.parent / "reports" / "preliminary_thesis" / "figures"

    fig, axs = plt.subplots(2, 1, figsize=(6, 4))
    plt.xlabel("Iteration")
    plt.ylabel("Mean reward")
    axs[0].set_ylabel("Mean reward")

    dfs_alpha = {
        "$\\alpha=0.1$": pd.read_csv(results_dir_path / "results_1_9.csv"),
        # "$\\alpha=0.01$": pd.read_csv(results_dir_path / "results_01_9.csv"),
        "$\\alpha=0.001$": pd.read_csv(results_dir_path / "results_001_9.csv"),
    }

    dfs_gamma = {
        "$\\gamma=0.0$": pd.read_csv(results_dir_path / "results_01_00.csv"),
        "$\\gamma=0.1$": pd.read_csv(results_dir_path / "results_01_01.csv"),
        # "$\\gamma=0.5$": pd.read_csv(results_dir_path / "results_01_05.csv"),
        "$\\gamma=0.9$": pd.read_csv(results_dir_path / "results_01_09.csv"),
        # "$\\gamma=1.0$": pd.read_csv(results_dir_path / "results_01_10.csv"),
    }

    for label, df in dfs_alpha.items():
        mean = df.mean(axis=1).rolling(100).mean()
        mean.plot(label=label, ax=axs[0])

    for label, df in dfs_gamma.items():
        mean = df.mean(axis=1).rolling(100).mean()
        mean.plot(label=label, ax=axs[1])

    plt.tight_layout()
    axs[0].legend()
    axs[1].legend()

    tikzplotlib.clean_figure()
    tikzplotlib.save(plot_path / "qr_rewards.tex", extra_axis_parameters=[
        'width=0.9\\textwidth',
        'height=0.3\\textwidth'])

    plt.show()


if __name__ == '__main__':
    main()
