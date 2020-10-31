from cw.simulation import ModuleBase
from cw.control import PIDController


class PIDAgent(ModuleBase):
    def __init__(self, k_p, k_i, k_d):
        super().__init__(is_discreet=True,
                         target_time_step=0.01,
                         required_states=["agent_torque"])
        self.controller = PIDController(k_p, k_i, k_d)

    def initialize(self, simulation):
        super().initialize(simulation)
        self.controller.reset()

    def step(self):
        self.s.agent_torque = self.controller.step(self.s.t, self.s.theta)
