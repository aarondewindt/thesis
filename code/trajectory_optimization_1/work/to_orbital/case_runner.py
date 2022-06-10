from time import pthread_getcpuclockid
from typing import Sequence
from math import pi

from scipy.interpolate import interp1d

from traj1.logger import Logger
from environment import LauncherV1Orbital


class CaseRunner:
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
            "semi_major_axis", "eccentricity",
            "action_engine_on"
        ])

        self.last_result = None

    def __call__(self, x: Sequence[float]):
        engine_timing = x[:3]
        n_checkpoints = ((len(x) - 3) + 2) // 2
        times = (0, *x[3:n_checkpoints+1], self.max_time)
        pitches = x[n_checkpoints+1:]

        result = self.run_case_interpolation(engine_timing, times, pitches)

        final_h = result.env_h.values[-1]
        final_eccentricity = result.env_eccentricity.values[-1]
        cost = abs(self.env.target_h - final_h) / self.env.target_h + final_eccentricity

        self.last_result = result
        print(cost, x)
        return cost

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









