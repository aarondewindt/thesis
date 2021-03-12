from math import pi

from dataclasses import dataclass
from typing import Sequence, Optional, Tuple

import gym
import numpy as np
import numba as nb

from .simulation import Simulation


class Stage:
    def __init__(self,
                 dry_mass: float,
                 propellant_mass: float,
                 specific_impulse: float,
                 thrust: float,
                 n_ignitions: Optional[int] = 1):
        self.dry_mass = dry_mass
        self.propellant_mass = propellant_mass
        self.specific_impulse = specific_impulse
        self.thrust = thrust
        self.n_ignitions = n_ignitions

    def to_tuple(self):
        return (np.float64(self.dry_mass),
                np.float64(self.propellant_mass),
                np.float64(self.specific_impulse),
                np.float64(self.thrust),
                np.int32(-1 if self.n_ignitions is None else self.n_ignitions))


class LauncherV1(gym.Env):
    metadata = {'render.modes': []}

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
                 controller_theta_dot_limits: Tuple[float, float],
                 end_at_apogee: bool,
                 end_at_ground: bool):
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(2),  # Engine command
            gym.spaces.Discrete(2),  # Drop stage command
            gym.spaces.Box(low=-pi, high=pi, shape=(1,)),  # Flight path angle command
        ))
        self.observation_space = gym.spaces.Tuple(())

        self.random = np.random.Generator(np.random.PCG64())

        self.sim = Simulation(
            dt=dt,
            surface_diameter=surface_diameter,
            mu=mu,
            stages=nb.typed.List([stage.to_tuple() for stage in stages]),
            initial_longitude=initial_longitude,
            initial_altitude=initial_altitude,
            initial_theta_e=initial_theta_e,
            gamma_controller_gains=tuple(gamma_controller_gains),
            theta_controller_gains=tuple(theta_controller_gains),
            controller_theta_dot_limits=tuple(controller_theta_dot_limits),
            end_at_apogee=end_at_apogee,
            end_at_ground=end_at_ground,
        )

        self.reset()

    def reset(self):
        self.sim.reset()

    def seed(self, seed=None):
        self.random = np.random.Generator(np.random.PCG64(seed))

    def step(self, action: Tuple[bool, bool, int, float]):
        # Since the observation and rewards may be different between experiments, the step function
        # is left unimplemented to keep configurable by inheriting this class.
        # Example:
        #     self.sim.step((
        #         action[0],  # engine_on: True to ignite the engine or keep burning.
        #         action[1],  # drop_stage: True to drop the current stage.
        #         nb.int32(action[2]),  # autopilot_mode: Choose between AP_NONE, AP_FLIGHT_PATH_CONTROL,
        #                               # AP_PITCH_CONTROL, AP_PITCH_RATE_CONTROL
        #         nb.float64(action[3])  # autopilot_reference: Autopilot reference signal
        #     ))
        #     self.sim.reward = self.sim.h  # Set the reward in the simulation so it can calculate the score.
        #     return (self.sim.gamma_e, self.sim.vii), self.sim.reward, self.sim.done, {}

        raise NotImplementedError("Step function not implemented in base environment class.")

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def sim_states_dict(self):
        return {name: str(getattr(self.sim, name)) for name in state_names}


state_names = (
    "t", "action_engine_on", "action_drop_stage", "action_autopilot_mode", "action_autopilot_reference",
    "ap_comm_gamma_e", "ap_comm_theta_e", "gii", "xii", "vii", "aii", "tei", "vie", "fii_thrust",
    "theta_i", "theta_i_dot", "theta_e", "mass", "mass_dot", "h", "engine_on", "stage_state",
    "stage_idx", "stage_ignitions_left", "gamma_i", "gamma_e", "longitude", "reward", "score", "done"
)
