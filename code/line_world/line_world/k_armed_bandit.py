from random import random, randint, seed as set_seed, gauss
from time import time
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import Optional
import numpy as np


results_path = Path(__file__).with_suffix(".i.pickle")


def k_armed_bandit(fig_dir_path: Optional[Path]):
    n_episodes = 1000
    n_problems = 2000
    eps_options = [0, 0.01, 0.1]
    k = 10
    seed = 0

    if results_path.exists():
        with results_path.open("rb") as f:
            results = pickle.load(f)
    else:
        results = [None] * len(eps_options)

        for i, eps in enumerate(eps_options):
            rewards = [None] * n_episodes
            optimal_action = [None] * n_episodes

            agents = [Agent(
                eps,
                environment=Environment(
                    k,
                    seed + h
                )
            ) for h in range(n_problems)]

            for j in range(n_episodes):
                rewards[j] = 0
                optimal_action[j] = 0
                for agent in agents:
                    reward, chose_optimal_action = agent.run_episode()
                    rewards[j] += reward
                    optimal_action[j] += int(chose_optimal_action)
                    # reward_sum = agent.environment.perform_action(agent.greedy_action())
                rewards[j] /= n_problems
                optimal_action[j] /= n_problems / 100
            results[i] = rewards, optimal_action

        with results_path.open("wb") as f:
            pickle.dump(results, f)

    # Reward average figure.
    plt.figure(figsize=(5, 4), dpi=300, facecolor='w', edgecolor='k')
    fig, (ax_1, ax_2) = plt.subplots(num=1, nrows=2, ncols=1)

    linewidths = [2, 1, 0.5]
    for i, (rewards, optimal_action) in enumerate(results):
        ax_1.plot(rewards, color=f"C{i}", linewidth=linewidths[i], label=f"$\epsilon = {eps_options[i]}$")
        ax_2.plot(optimal_action, color=f"C{i}", linewidth=linewidths[i], label=f"$\epsilon = {eps_options[i]}$")

    # ax_1.set_xlabel("episode")
    ax_1.set_ylabel("average reward")
    ax_1.get_xaxis().set_visible(False)
    ax_1.legend()

    ax_2.set_xlabel("episode")
    ax_2.set_ylabel("% optimal policy")

    plt.tight_layout()
    if fig_dir_path:
        plt.savefig(str(fig_dir_path / "10_armed_bandit.pdf"))
    else:
        plt.show()


def k_bandit_violin(fig_dir_path: Optional[Path]):
    k = 5
    np.random.seed(19690801)
    fig = plt.figure(num=None, figsize=(5, 3), dpi=300, facecolor='w', edgecolor='k')
    data = [np.random.normal(mu, 1, 100000) for mu in np.random.normal(0, 1, k)]
    plt.violinplot(data, showextrema=False, showmeans=True)
    # plt.title("Example k-armed bandit reward distribution")
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.tight_layout()

    if fig_dir_path:
        pdf_path = fig_dir_path / "k_armed_bandit_violin.pdf"
        plt.savefig(str(pdf_path))
    else:
        plt.show()


class Agent:
    """
    k-armed bandit agent. Learns by averaging the rewards received from the environment.
    """
    def __init__(self, eps: float, environment: 'Environment'):
        self.eps = eps
        self.q_table = [0.0] * environment.k
        self.visit_count = [0] * environment.k
        self.environment = environment

    def greedy_action(self):
        """Greedy action taken by the agent"""
        return self.q_table.index(max(self.q_table))

    def run_episode(self):
        """
        Run an episode.
        :returns: Tuple with two elements, the first one being the reward it received
                  and the second one a boolean on whether it tool the optimal action or not.
        """
        # Choose actions
        if self.eps > random():
            # Random action
            action = randint(0, self.environment.k - 1)
        else:
            # Greedy action
            action = self.q_table.index(max(self.q_table))

        # Perform action
        reward = self.environment.perform_action(action)

        # Update visit count and calculate alpha
        self.visit_count[action] += 1
        alpha = 1 / self.visit_count[action]

        # Learn
        self.q_table[action] += alpha * (reward - self.q_table[action])

        return reward, action == self.environment.optimal_action


class Environment:
    """
    k-armed bandit environment. The true value of each action is selected based on a
    zero mean, unit variance normal distribution and the rewards returned by the
    environment are selected with a unit variance normal distribution whose mean is
    the action's true value.

    :param k: Number armes
    :param seed: Random generator seed.
    """
    def __init__(self, k: int, seed: int=None):
        self.k = k
        """Number of possible actions"""
        set_seed(seed or time())

        self.reward_mean = [gauss(0, 1) for _ in range(k)]
        """Rewards for each action"""

        self.optimal_action = self.reward_mean.index(max(self.reward_mean))
        """Optimal action in this environment"""

    def perform_action(self, action):
        """Performs the action."""
        return gauss(self.reward_mean[action], 1)


if __name__ == '__main__':
    fig_dir_path = Path("/media/stuff/stuff/repos/thesis/reports/preliminary_thesis/figures")
    k_armed_bandit(fig_dir_path)
    k_bandit_violin(fig_dir_path)
