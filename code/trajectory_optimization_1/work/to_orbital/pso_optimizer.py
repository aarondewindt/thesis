import numpy as np
import pyswarms as ps

from case_runner import CaseRunner 


class PSOOptimizer:
    def __init__(self, case_runner: CaseRunner, n_particles: int, pso_options=None) -> None:
        pso_options = pso_options or {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        self.case_runner = case_runner
        self.optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles, 
            dimensions=case_runner.ndim,
            bounds=case_runner.bounds,
            options=pso_options
        )

    def optimize(self, iters, n_processes=1):
        # Perform optimization
        return self.optimizer.optimize(
            self.case_runner, 
            iters=iters, 
            n_processes=n_processes
        )
