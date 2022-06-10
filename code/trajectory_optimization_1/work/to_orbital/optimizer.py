from time import pthread_getcpuclockid
from typing import Sequence
from math import pi

from scipy.interpolate import interp1d

from traj1.logger import Logger
from environment import LauncherV1Orbital


class Optimizer:
    def __init__(self, env: LauncherV1Orbital, max_time: float):
        self.env = env
        self.max_time = max_time

        self.logger = Logger()
        self.logger.register_time_attribute(env.sim, "t")
        self.logger.register(env.sim, "env", [
            "h", "i", "vie", "vie_hat", "reward",
            "gamma_e", "theta_e", "theta_i_dot",
            "ap_comm_gamma_e", "ap_comm_theta_e",
            "action_autopilot_mode", "action_autopilot_reference",
            "vii", "xii", "fii_thrust", "mass", "mass_dot",
            "end_at_apogee", "end_at_ground",
            "semi_major_axis", "eccentricity"
        ])

        self.last_result = None

    def run_case(self, x: Sequence[float]):
        engine_cutoff_time = x[0]
        n_checkpoints = (len(x) - 1) // 2
        times = x[:n_checkpoints]
        pitches = x[n_checkpoints:n_checkpoints+n_checkpoints]

        pitch_interpolator = interp1d(times, pitches, fill_value="extrapolate")
        
        observation = self.env.reset()
        for i in range(int(self.max_time / self.env.sim.integrator.h)):
            time = self.env.sim.t
            pitch_angle = pitch_interpolator(time)
            is_engine_running = time <= engine_cutoff_time
            observation, reward, done, info = self.env.step((pitch_angle, is_engine_running))
            self.logger.step()
            if done:
                break
        result = self.logger.episode_finish()
        self.last_result = result

        reward_sum = sum(result.env_reward.values)

        return -reward_sum







