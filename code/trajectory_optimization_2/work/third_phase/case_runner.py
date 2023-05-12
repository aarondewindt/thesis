from typing import Sequence, Optional
from pathlib import Path
from math import pi
import time
import os

from scipy.interpolate import interp1d
import gymnasium.spaces as gs
import numpy as np
import numba as nb

from traj2.logger import Logger
from environment import LauncherV1Orbital


@nb.jit(nopython=True, cache=True)
def sigmoid(x):
    return x / np.sqrt(1 + x * x)


@nb.jit(nopython=True, cache=True)
def cost_function(target_a, a, e):
    a_error = abs(target_a - a) / target_a
    return -1 / (1 + (a_error*100)**2 + (e * 5)**2) + 1


class CaseRunner:
    def __init__(self, 
                 env: LauncherV1Orbital, 
                 max_time: float, 
                 n_checkpoints: int,
                 results_path: Optional[Path]=None):
        n_checkpoints = int(n_checkpoints)
        assert n_checkpoints >= 2, "Must have at least two checkpoints."

        self.env = env
        self.max_time = max_time
        self.results_path = results_path

        self.logger = Logger()
        self.logger.register_time_attribute(env.sim, "t")
        self.logger.register(env.sim, "env", [
            "h", "i", "vie", "vie_hat", "reward",
            "gamma_e", "theta_e", "theta_i_dot",
            "ap_comm_gamma_e", "ap_comm_theta_e",
            "action_autopilot_mode", "action_autopilot_reference",
            "vii", "xii", "fii_thrust", "mass", "mass_dot",
            "end_at_apogee", "end_at_ground",
            "semi_major_axis", "eccentricity",
            "action_engine_on", "longitude", 
        ])

        self.last_result = None
        self.n_checkpoints = n_checkpoints

        assert isinstance(env.action_space, gs.Tuple)
        ap_command_action_space = env.action_space.spaces[0]

        assert isinstance(ap_command_action_space, gs.Box)
        ap_low_bound = ap_command_action_space.low[0]
        ap_high_bound = ap_command_action_space.high[0]

        self.bounds = (
            np.array([0.] * (1 + n_checkpoints) + ([ap_low_bound] * n_checkpoints)),
            np.array([max_time] * (1 + n_checkpoints) + [ap_high_bound] * n_checkpoints),
        )
        
        self.ndim = len(self.bounds[0])

    def __reduce__(self):
        return CaseRunner, (self.env, self.max_time, self.n_checkpoints, self.results_path)

    # @staticmethod
    # def cost_function(target_h, h, e):
    #     h_error = abs(target_h - h) / target_h
    #     return (
    #         sigmoid(h_error*5)**2
    #         + sigmoid(e*5)**2
    #     )

    # @staticmethod
    # def cost_function(target_a, a, e):
    #     a_error = abs(target_a - a) / target_a
    #     return (
    #         sigmoid(a_error*100)**2
    #         + sigmoid(e * 5)**2
    #     )

    def __call__(self, case_inputs: np.ndarray):
        def single_case(x: Sequence[float]):
            assert len(x) == 1 + 2 * self.n_checkpoints
            engine_timing = x[:3]
            times = (0, *x[3:self.n_checkpoints+1], self.max_time)
            pitches = x[self.n_checkpoints+1:]

            result = self.run_case_interpolation(engine_timing, times, pitches)

            # final_h = result.env_h.values[-1]
            final_a = result.env_semi_major_axis.values[-1]
            final_eccentricity = result.env_eccentricity.values[-1]
            cost = cost_function(self.env.target_a, final_a, final_eccentricity)
            # cost = abs(self.env.target_h - final_h) / self.env.target_h + final_eccentricity

            self.last_result = result
            return cost
        
        costs = np.empty((case_inputs.shape[0],))
        timestamps = np.empty((case_inputs.shape[0],))

        for i, x in enumerate(case_inputs):
            costs[i] = single_case(x)
            timestamps[i] = time.time()

        if self.results_path is not None:
            log_path = self.results_path / f"{time.time_ns()}_{os.getpid()}.npy"
            log = np.hstack((
                costs.reshape((-1, 1)),
                timestamps.reshape((-1, 1)),
                case_inputs
            ))
            np.save(log_path, log)

        return costs

    def run_case_interpolation(self, engine_timing, times, pitches):
        pitch_interpolator = interp1d(times, pitches, fill_value="extrapolate")
        
        observation = self.env.reset()
        for i in range(int(self.max_time / self.env.sim.integrator.h)):
            time = self.env.sim.t
            pitch_angle = pitch_interpolator(time)

            is_engine_running = (time <= engine_timing[0]) | (engine_timing[1] < time < engine_timing[2])
            observation, reward, done, info = self.env.step((pitch_angle, is_engine_running))
            self.logger.step()
            if done:
                break

        return self.logger.episode_finish()









