from pole import Pole, TileCodingAgent, Season
from pole.filter import filter_signal
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from math import radians

from cw.context import time_it


# Configure pole simulation
pole = Pole(0.500, 0.100)
pole.dt = 0.01


agent = TileCodingAgent(
    center=[0, 0, 0],
    tile_size=[radians(1.5), 0.5, 0.1],
    tilings=50,
    default_weight=0.0,
    random_offsets=True,
    min_action=-0.3,
    max_action=0.3,
    n_actions=11,
    epsilon=0.9,
    gamma=0.9,
    alpha=0.2,
)

print(agent.actions)

# Connect the agent to the environment.
agent.set_environment(pole)
# Create season and choose inputs.
season = Season(agent)
season_inputs = {
    "eps":             [0.7],
    "gamma":           [0.9],
    "alpha":           [0.9],
    "n_episodes":      [5]
}

# Loop through
reward_sum = []
learn_rate_sum = []
for i in range(len(season_inputs["eps"])):
    # Setup agent
    agent.epsilon = season_inputs["eps"][i]
    agent.gamma = season_inputs["gamma"][i]
    agent.alpha = season_inputs["alpha"][i]

    # Run season
    with time_it("season run"):
        season.run(season_inputs["n_episodes"][i], 1, 1000)

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

# plt.figure()
# agent.ideal_torque.plot()
#
# plt.figure()
# xr.apply_ufunc(np.log10, agent.visit_count).plot()
plt.show()
