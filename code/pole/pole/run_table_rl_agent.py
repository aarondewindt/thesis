from pole import Pole, TableRLAgent, Season
from pole.filter import filter_signal
import matplotlib.pyplot as plt
from time import perf_counter
from itertools import product
import numpy as np
from math import radians

# Configure pole simulation
pole = Pole(0.500, 0.100)
pole.dt = 0.01

# # Configure RL agent
min_theta = radians(-4)
max_theta = radians(4)
min_theta_dot = -5
max_theta_dot = 5
min_torque = -1
max_torque = 1
n_theta = 41
n_theta_dot = 41
n_torque = 41
agent = TableRLAgent(
    min_theta=min_theta,
    max_theta=max_theta,
    min_theta_dot=min_theta_dot,
    max_theta_dot=max_theta_dot,
    min_torque=min_torque,
    max_torque=max_torque,
    n_theta=n_theta,
    n_theta_dot=n_theta_dot,
    n_torque=n_torque,
    epsilon=0.9,
    gamma=0.9,
    alpha=1,
    n_bootstrapping=1,
)

# Connect the agent to the environment.
agent.set_environment(pole)

# Create season and choose inputs.
season = Season(agent)
season_inputs = {
    "eps":             [0.7, 0.8, 0.9, 1.0],
    "gamma":           [0.7, 0.9,  0.7, 0.7],
    "alpha":           [1, 0.8,  1, 0],
    "n_episodes":      [5000, 1, 5000, 1000]
}

# Loop through
reward_sum = []
learn_rate_sum = []
for i in range(len(season_inputs["eps"])):
    # Setup agent
    agent.eps = season_inputs["eps"][i]
    agent.gamma = season_inputs["gamma"][i]
    agent.alpha = season_inputs["alpha"][i]

    # Run season
    t0 = perf_counter()
    season.run(season_inputs["n_episodes"][i], 1, 500)
    print("dt", perf_counter() - t0)

    # Plot results
    season_data = season.get_data_log()
    for idx, episode_data in season_data.items():
        plt.figure()
        plt.suptitle(f"S{i}E{idx}")
        for j, value in enumerate(episode_data):
            plt.subplot(3, 2, j+1)
            plt.grid()
            plt.plot(episode_data.time, episode_data[value])
            plt.title(value)

        plt.tight_layout()

    scalar_data = season.get_scalar_data()
    reward_sum.extend(scalar_data['reward_sum'])
    learn_rate_sum.extend(scalar_data['learn_rate_sum'])

plt.figure()
plt.plot(reward_sum, ".", markersize=2)
plt.plot(filter_signal(reward_sum, wn=0.02))
plt.title("Reward sum")

plt.figure()
plt.plot(learn_rate_sum, ".", markersize=2)
plt.plot(filter_signal(learn_rate_sum, wn=0.01))
plt.title("learn_rate sum")


def foo(min_x, max_x, n_x, idx):
    return idx * (max_x - min_x) / (n_x - 1) + min_x


thetas = [foo(min_theta, max_theta, n_theta, i) for i in range(n_theta)]
theta_dots = [foo(min_theta_dot, max_theta_dot, n_theta_dot, i) for i in range(n_theta_dot)]
actions = np.ones((len(thetas), len(theta_dots))) * 1e10

plt.figure()
for (idx_theta, theta), (idx_theta_dot, theta_dot) in product(enumerate(thetas), enumerate(theta_dots)):
    actions[idx_theta, idx_theta_dot] = agent.choose_ideal_torque(theta, theta_dot)

ax = plt.gca()
im = plt.imshow(actions)
ax.set_xticks(np.arange(len(theta_dots)))
ax.set_yticks(np.arange(len(thetas)))
ax.set_yticklabels([f'{theta:.3}' for theta in thetas])
ax.set_xticklabels([f'{theta_dot:.3}' for theta_dot in theta_dots])
cbar = ax.figure.colorbar(im, ax=ax)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
# cbar.add_lines([foo(min_torque, max_torque, n_torque, i) for i in range(n_torque)])
plt.tight_layout()
plt.show()
