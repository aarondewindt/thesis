from math import pi, radians, atan2, sqrt, sin, cos
from typing import Sequence, Tuple

import gym
import numba as nb
import numpy as np

from traj1.environments.launcher_v1 import LauncherV1, Stage, AP_NONE, AP_FLIGHT_PATH_CONTROL, AP_PITCH_CONTROL, AP_PITCH_RATE_CONTROL
from traj1.environments.launcher_v1.simulation import wrap_angle, clip
from cw.vdom import hyr
from cw.astrodynamics import kepler_to_cartesian

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork


default_config = {
    "dt": 0.05,
    "surface_diameter": 1737.4e3,
    "mu": 4.9048695e12,
    "stages": (
        Stage( 
            dry_mass=1,
            propellant_mass=0.2,
            specific_impulse=80,
            thrust=4*1.7),
    ),
    "initial_altitude": 1,
    "initial_theta_e": radians(90),
    "initial_longitude": radians(90),
    "initial_vie": (0., 0.),
    "initial_kepler": None,
    "gamma_controller_gains": (4, 0, 0.2),
    "theta_controller_gains": (10, 0, 0.0),
    "controller_theta_dot_limits": (-1, 1),
    "end_at_apogee": False,
    "end_at_ground": True,
    "end_at_burnout": True,
    "autopilot_mode": AP_PITCH_RATE_CONTROL,
    "theta_e_random_window": radians(150)
}


class LauncherV1SubOrbital(LauncherV1):
    def __init__(self, env_config=None):
        self.config = default_config.copy()
        self.config.update(env_config or {})

        self._theta_e_random_window = self.config['theta_e_random_window']
        self._initial_theta_e = self.config['initial_theta_e']
        self.autopilot_mode = self.config["autopilot_mode"]

        if (initial_kepler := self.config['initial_kepler']) is not None:
            xii, vii = kepler_to_cartesian(
                a=initial_kepler['a'],
                e=initial_kepler['e'],
                i=0.,
                raan=0.,
                omega=0.,
                true_anomaly=initial_kepler['true_anomaly'],
                mu=self.config["mu"]
            )

            initial_longitude = atan2(xii[1], xii[0])
            initial_altitude = sqrt(xii[0]*xii[0]+xii[1]*xii[1]) - self.config["surface_diameter"]
            tei = np.array(((-sin(initial_longitude), cos(initial_longitude)),
                            (cos(initial_longitude), sin(initial_longitude))), dtype=np.float64)

            vii = (vii[0], vii[1])
            initial_vie = tei @ vii

        else:
            initial_longitude = self.config["initial_longitude"]
            initial_altitude = self.config["initial_altitude"]
            initial_vie = self.config["initial_vie"]

        super().__init__(
            dt=self.config["dt"],
            surface_diameter=self.config["surface_diameter"],
            mu=self.config["mu"],
            stages=self.config["stages"],
            initial_longitude=initial_longitude,
            initial_altitude=initial_altitude,
            initial_theta_e=self.config["initial_theta_e"],
            gamma_controller_gains=self.config["gamma_controller_gains"],
            theta_controller_gains=self.config["theta_controller_gains"],
            controller_theta_dot_limits=self.config["controller_theta_dot_limits"],
            end_at_apogee=self.config["end_at_apogee"],
            end_at_ground=self.config["end_at_ground"],
            end_at_burnout=self.config["end_at_burnout"],
            initial_vie=initial_vie
        )

        if self.config['autopilot_mode'] == AP_PITCH_RATE_CONTROL:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
#             self.action_space = gym.spaces.Discrete(5)
        else:
            self.action_space = gym.spaces.Box(low=-pi, high=pi, shape=(1,))

        self.observation_space = gym.spaces.Box(low=np.array([-pi-0.01, -pi-0.01, 0.], dtype=np.float32),
                                                high=np.array([pi+0.01, pi+0.01, 1.], dtype=np.float32),
                                                shape=(3,))

        # self.observation_space = gym.spaces.Box(low=np.array([-1, -1], dtype=np.float32),
        #                                         high=np.array([1, 1], dtype=np.float32),
        #                                         shape=(2,))

    def observation(self):
        return np.array([wrap_angle(self.sim.gamma_e), wrap_angle(self.sim.theta_e), clip(0., 1., self.sim.h / 1200)])
        # return self.sim.vie_hat

    def reset(self):
        # Randomize initial pitch angle
        if self._theta_e_random_window:
            beta = (pi - self._theta_e_random_window) / 2
            self.sim.initial_theta_e = self.random.uniform(beta, pi - beta)
        else:
            self.sim.initial_theta_e = self._initial_theta_e

        # Reset simulation and return observation
        self.sim.reset()
        return self.observation()

    def step(self, action: float):
        if not np.isfinite(action):
            raise ValueError("Action is not finite")

        self.sim.step((
            False,  # action_engine_on
            False,  # action_drop_stage
            nb.int32(self.autopilot_mode),  # action_autopilot_mode
            nb.float64(action)  # action_autopilot_reference
        ))

        # In some very rare cases the simulation might return nan. It's 1 in billions.
        if all(np.isfinite(self.sim.vie)):
            # The reward is zero until the terminal state, at which point it's between 0 and 1.
            # The reward is based on the maximum altitude the launcher can reach with the current
            # vertical velocity. It is scaled such that the maximum potential altitude is just under 1.
            if self.sim.done:
                self.sim.reward = ((self.sim.h + self.sim.vie[1]**2 / (2 * np.linalg.norm(self.sim.gii))) / 4600)**21
                # self.sim.reward = (self.sim.h + self.sim.vie[1]**2 / (2 * np.linalg.norm(self.sim.gii)))
            else:
                self.sim.reward = 0

            return self.observation(), \
                   self.sim.reward, \
                   self.sim.done, \
                   {"t": self.sim.t}
        else:
            self.sim.done = True
            raise ValueError("Simulation is not finite")

    def _repr_html_(self):
        return hyr({
            "config": self.config,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }).to_html()



