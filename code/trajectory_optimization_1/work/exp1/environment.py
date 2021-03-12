from math import pi
from typing import Sequence, Tuple

import gym
import numba as nb
import numpy as np

from traj1.environments.launcher_v1 import LauncherV1, Stage, AP_FLIGHT_PATH_CONTROL, AP_NONE


class Environment(LauncherV1):
    def __init__(self,
                 dt: float,
                 surface_diameter: float,
                 mu: float,
                 stages: Sequence[Stage],
                 initial_longitude: float,
                 initial_altitude: float,
                 initial_theta_e: float,
                 gamma_controller_gains: Tuple[float, float, float],
                 theta_controller_gains: Tuple[float, float, float],
                 controller_theta_dot_limits: Tuple[float, float]):
        super().__init__(dt, surface_diameter, mu, stages, initial_longitude, initial_altitude, initial_theta_e,
                         gamma_controller_gains, theta_controller_gains, controller_theta_dot_limits,
                         end_at_apogee=False,
                         end_at_ground=False)

        self.action_space = gym.spaces.Box(low=-pi, high=pi, shape=(1,))  # Flight path angle command
        self.observation_space = gym.spaces.Box(low=np.array([-pi, -np.inf], dtype=np.float32),
                                                high=np.array([pi, np.inf], dtype=np.float32),
                                                shape=(2,))

    def step(self, action: float):
        self.sim.step((
            True,
            False,
            nb.int32(AP_NONE),
            nb.float64(action)
        ))
        self.sim.reward = self.sim.h
        return (self.sim.gamma_e, self.sim.h), self.sim.reward, self.sim.done, {}



