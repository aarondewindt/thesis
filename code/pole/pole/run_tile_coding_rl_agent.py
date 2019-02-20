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

config = 1
configs = [
    (1, 5, [radians(4.44), 0.244, 0.02]),
    (20, 5, [radians(10), 0.5, 0.08]),
    (1, 5, [radians(4.44), 0.244, 0.02]),
]

agent = TileCodingAgent(
    center=[0, 0, 0],
    tile_size=configs[config][2],
    tilings=configs[config][0],
    default_weight=0.0,
    random_offsets=True,
    min_action=-0.18,
    max_action=0.18,
    n_actions=configs[config][1],
    epsilon=0.9,
    gamma=0.9,
    alpha=0.2,
    vc_min_theta=radians(-180),
    vc_max_theta=radians(180),
    vc_n_theta=256,
    vc_min_theta_dot=-20,
    vc_max_theta_dot=20,
    vc_n_theta_dot=256,
)

print(agent.actions)

# Connect the agent to the environment.
agent.set_environment(pole)
# Create season and choose inputs.
season = Season(agent)
season_inputs = {
    "eps":             [0.7, 1.0],
    "gamma":           [0.7, 0],
    "alpha":           [0.2, 0],
    "n_episodes":      [10000, 2],
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

plt.figure()
agent.get_greedy_action().plot()

plt.figure()
plt.title("Tile update count")
xr.apply_ufunc(np.log10, agent.get_update_count()).plot()

plt.figure()
plt.title("Visit count")
xr.apply_ufunc(np.log10, agent.get_visit_count()).plot()

plt.show()
