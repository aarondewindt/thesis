from pole import Pole, PIDAgent, Season
import matplotlib.pyplot as plt
from time import perf_counter

pole = Pole(0.500, 0.100)

agent = PIDAgent(0.5, 0, .02)
agent.set_environment(pole)

season = Season(agent)

t0 = perf_counter()
season.run(100, 3, 1000)
print("dt", perf_counter() - t0)

season_data = season.get_data_log()

for idx, episode_data in season_data.items():
    plt.figure()
    plt.suptitle(f"Episode {idx}")
    for i, value in enumerate(episode_data):
        plt.subplot(3, 3, i+1)
        plt.plot(episode_data.time, episode_data[value])
        plt.title(value)


plt.figure()
scalar_data = season.get_scalar_data()
# print(scalar_data)
plt.plot(scalar_data['reward_sum'])
plt.title("Reward sum")
plt.show()
