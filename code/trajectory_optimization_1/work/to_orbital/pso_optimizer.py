import numpy as np
import pyswarms as ps

from case_runner import CaseRunner 


class PSOOptimizer:
    def __init__(self, case_runner: CaseRunner, n_particles: int, pso_options=None) -> None:
        pso_options = pso_options or {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        self.case_runner = case_runner
        self.optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=9, options=pso_options)

    def optimize(self, iters):
        def cost_function(swarm_positions: np.ndarray):
            costs = np.empty((swarm_positions.shape[0],))
            for i, x in enumerate(swarm_positions):
                costs[i] = self.case_runner(x)
            return costs

        # Perform optimization
        return self.optimizer.optimize(cost_function, iters=iters)
