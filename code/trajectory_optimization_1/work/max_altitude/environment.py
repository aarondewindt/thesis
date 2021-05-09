from math import pi
from typing import Sequence, Tuple

import gym
import numba as nb
import numpy as np

from traj1.environments.launcher_v1 import LauncherV1, Stage, AP_NONE, AP_FLIGHT_PATH_CONTROL, AP_PITCH_CONTROL, AP_PITCH_RATE_CONTROL


class Environment(LauncherV1):
    def __init__(self, env_config):
        super().__init__(
            dt=env_config["dt"],
            surface_diameter=env_config["surface_diameter"],
            mu=env_config["mu"],
            stages=env_config["stages"],
            initial_longitude=env_config["initial_longitude"],
            initial_altitude=env_config["initial_altitude"],
            initial_theta_e=env_config["initial_theta_e"],
            gamma_controller_gains=env_config["gamma_controller_gains"],
            theta_controller_gains=env_config["theta_controller_gains"],
            controller_theta_dot_limits=env_config["controller_theta_dot_limits"],
            end_at_apogee=True,
            end_at_ground=False)

        self.autopilot_mode = env_config["autopilot_mode"]
        self.action_space = gym.spaces.Box(low=-pi, high=pi, shape=(1,))  # Flight path angle command
        self.observation_space = gym.spaces.Box(low=-pi-0.1,
                                                high=pi+0.1,
                                                shape=(1,))

    def reset(self):
        self.sim.reset()
        return np.array([self.sim.gamma_e])

    def step(self, action: float):
        self.sim.step((
            True,
            False,
            nb.int32(self.autopilot_mode),
            nb.float64(action)
        ))
        self.sim.reward = self.sim.h / 1000
        return np.array([self.sim.gamma_e]), self.sim.reward, self.sim.done, {"t": self.sim.t}



