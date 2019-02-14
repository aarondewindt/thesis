from pole import Pole, TableRLAgent, Season
from pole.filter import filter_signal
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
import xarray as xr
from math import radians

# Configure pole simulation
pole = Pole(0.500, 0.100)
pole.dt = 0.01

# # Configure RL agent
min_theta = radians(-180)
max_theta = radians(180)
min_theta_dot = -5
max_theta_dot = 5
min_torque = -0.18
max_torque = 0.18
n_theta = 81
n_theta_dot = 41
n_torque = 5
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
    "eps":             [0.7, 1.0],
    "gamma":           [0.9, 0.9],
    "alpha":           [1.0, 0],
    "n_episodes":      [100000, 2]
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
    season.run(season_inputs["n_episodes"][i], 1, 10000)
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
plt.plot(reward_sum, ".", markersize=1, alpha=0.2)
plt.plot(filter_signal(reward_sum, wn=0.02))
plt.title("Reward sum")

plt.figure()
plt.plot(learn_rate_sum, ".", markersize=1, alpha=0.2)
plt.plot(filter_signal(learn_rate_sum, wn=0.01))
plt.title("learn_rate sum")

plt.figure()
agent.ideal_torque.plot()

plt.figure()
xr.apply_ufunc(np.log10, agent.visit_count).plot()
plt.show()
