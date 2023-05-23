from math import pi
from typing import Tuple, Any

# import gym
import numba as nb
import numpy as np
import gymnasium.spaces as gs

from traj2.environments.launcher_v1 import LauncherV1, Stage, AP_NONE, AP_PITCH_CONTROL, AP_FLIGHT_PATH_CONTROL, AP_PITCH_RATE_CONTROL
from traj2.environments.launcher_v1.simulation import wrap_angle
from cw.vdom import hyr

from env_config import EnvConfig

# from ray.rllib.models.tf.fcnet import FullyConnectedNetwork


@nb.jit(nopython=True, cache=True)
def cost_function(target_a, a, e):
    a_error = abs(target_a - a) / target_a
    return -1 / (1 + (a_error*100)**2 + (e * 5)**2) + 1


class LauncherV1Orbital(LauncherV1):
    def __init__(self, env_config: dict[str, Any] | None=None):
        if env_config is None:
            self.config = EnvConfig()
        else:
            self.config = EnvConfig.validate(env_config)
        
        random = np.random.Generator(np.random.PCG64())
        initial_condition = self.config.get_init_conditions(random)

        self.target_e = 0.0
        self.target_a = initial_condition.target_a
        self.target_h = initial_condition.target_h
        self.target_v = initial_condition.target_v
        self.mu = initial_condition.mu
        self.autopilot_mode = AP_PITCH_RATE_CONTROL
        
        super().__init__(
            dt=self.config.dt,
            surface_diameter=self.config.surface_diameter,
            mu=self.config.mu,
            stages=self.config.stages,
            initial_longitude=initial_condition.longitude,
            initial_altitude=initial_condition.altitude,
            initial_theta_e=initial_condition.theta_e,
            gamma_controller_gains=self.config.gamma_controller_gains,
            theta_controller_gains=self.config.theta_controller_gains,
            controller_theta_dot_limits=self.config.controller_theta_dot_limits,
            end_at_apogee=self.config.end_at_apogee,
            end_at_ground=self.config.end_at_ground,
            end_at_burnout=self.config.end_at_burnout,
            initial_vie=initial_condition.vie,
            random=random
        )

        if self.autopilot_mode == AP_PITCH_RATE_CONTROL:
            autopilot_range = 1
        else:
            autopilot_range = 1
            
        self.action_space = gs.Tuple((
            gs.Box(low=-autopilot_range, high=autopilot_range, shape=(1,)),
            gs.Discrete(2)
        ))
        
        self.observation_space = gs.Box(low=np.array([-pi-0.01, -pi-0.01, -np.inf, -np.inf, -np.inf, -1, -1, -np.inf, -np.inf], dtype=np.float32),
                                        high=np.array([pi+0.01, pi+0.01, np.inf, np.inf, np.inf, 2, np.inf, np.inf, np.inf], dtype=np.float32),
                                        shape=(9,))

    def __reduce__(self):
        return LauncherV1Orbital, (self.config,)

    def observation(self):
        return np.array([
            wrap_angle(self.sim.gamma_e), 
            wrap_angle(self.sim.theta_e),
            (self.target_h - self.sim.h) / self.target_h,
            self.sim.vie[0],
            self.sim.vie[1],
            self.sim.eccentricity,
            self.sim.semi_major_axis,
            self.sim.mass - self.config.stages[0].dry_mass,
            self.sim.mass_dot,            
        ], dtype=np.float32)

    def reset(self):
        initial_condition = self.config.get_init_conditions(self.random)

        self.sim.initial_altitude = initial_condition.altitude
        self.sim.initial_longitude = initial_condition.longitude
        self.sim.initial_theta_e = initial_condition.theta_e
        self.sim.initial_vie = np.array(initial_condition.vie)
        self.sim.stages[0] = (
            self.sim.stages[0][0],
            initial_condition.prop_mass,
            self.sim.stages[0][2],
            self.sim.stages[0][3],
            self.sim.stages[0][4],
        )

        self.sim.reset()
        return self.observation()

    def step(self, action: Tuple[float, bool]):
        if not np.isfinite(action[0]):
            raise ValueError("Action is not finite")

        autopilot_reference = nb.float64(action[0])        
        engine_on = self.sim.engine_on and bool(action[1])

        self.sim.step((
            engine_on,  # action_engine_on
            False,  # action_drop_stage
            nb.int32(self.autopilot_mode),  # action_autopilot_mode
            autopilot_reference  # action_autopilot_reference
        ))

        # if self.sim.t > 60:
        #     self.sim.done = True
        
        observation = self.observation()
        
        if all(np.isfinite(self.sim.vie)) and all(np.isfinite(observation)):
            if self.sim.done:
                cost = cost_function(self.target_a, self.sim.semi_major_axis, self.sim.eccentricity)
                self.sim.reward = 1 - cost
            else:
                self.sim.reward = 0 

            if not np.isfinite(self.sim.reward):
                raise ValueError("Reward is not finite.")

            return observation, \
                   self.sim.reward, \
                   self.sim.done, \
                   {"t": self.sim.t}
        else:
            self.sim.done = True
            raise ValueError("Simulation is not finite")

    def _repr_html_(self):
        return hyr({
            "config": self.config.dict(),
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }).to_html()
