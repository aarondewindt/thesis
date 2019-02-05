from pole import Pole, TableRLAgent, Season
from pole.filter import filter_signal
import matplotlib.pyplot as plt
from time import perf_counter
from itertools import product
import numpy as np
from math import radians

pole = Pole(0.500, 0.100)

pole.dt = 0.01

min_theta = radians(-4)
max_theta = radians(4)
min_theta_dot = -5
max_theta_dot = 5
min_torque = -0.5
max_torque = 0.5
n_theta = 21
n_theta_dot = 21
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
    n_r=1,
)

agent.set_environment(pole)

season = Season(agent)

episodes_per_season = 400
season_inputs = {
    "eps":             [0.7, 0.8, 0.9, 1.0],
    "gamma":           [0.90, 0.9,  0.9, 0.9],
    "alpha":           [1.0, 1.0,  1.0, 0],
    "n_episodes":      [10000, 3000, 1000, 1000]
}
reward_sum = []
for i in range(len(season_inputs["eps"])):
    # Setup agent
    agent.eps = season_inputs["eps"][i]
    agent.gamma = season_inputs["gamma"][i]
    agent.alpha = season_inputs["alpha"][i]

    # Run season
    t0 = perf_counter()
    season.run(season_inputs["n_episodes"][i], 1, 100)
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

plt.figure()
plt.plot(reward_sum)
plt.plot(filter_signal(reward_sum))
plt.title("Reward sum")


def foo(min_x, max_x, n_x, idx):
    return idx * (max_x - min_x) / (n_x - 1) + min_x


thetas = [foo(min_theta, max_theta, n_theta, i) for i in range(n_theta)]
theta_dots = [foo(min_theta_dot, max_theta_dot, n_theta_dot, i) for i in range(n_theta_dot)]
actions = np.ones((len(thetas), len(theta_dots))) * 1e10

print(thetas)

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
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# cbar.add_lines([foo(min_torque, max_torque, n_torque, i) for i in range(n_torque)])
plt.tight_layout()
plt.show()


