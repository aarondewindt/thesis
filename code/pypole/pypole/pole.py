from math import sin, fmod
from typing import Optional

import numpy as np

from cw.simulation import ModuleBase
from cw.constants import g_earth, pi2, pi


class Pole(ModuleBase):
    def __init__(self,
                 mass: float,
                 length: float,
                 theta_terminate: Optional[float]=None,
                 theta_min_reward: Optional[float]=None):
        super().__init__(
            required_states=[
                "agent_torque",
                "theta",
                "theta_dot",
                "theta_dot_dot",
                "torque",
                "reward"
            ]
        )

        self.mass = mass
        self.length = length
        self.theta_terminate = theta_terminate
        self.theta_min_reward = theta_min_reward or pi
        self.state = np.array([0.0, 0.0])
        self.inertia = mass * length * length / 3

    def step(self):
        theta = self.s.theta
        abs_theta = abs(theta)
        
        self.s.torque = self.mass * g_earth * (self.length / 2) * sin(theta) + self.s.agent_torque
        self.s.theta_dot_dot = self.s.torque / self.inertia
        self.s.theta = fmod(fmod(theta + pi, pi2) + pi2, pi2) - pi
        self.s.reward = ((self.theta_min_reward - abs(theta)) / self.theta_min_reward) ** 4

        if self.theta_terminate:
            if abs_theta > self.theta_terminate:
                self.simulation.stop()
