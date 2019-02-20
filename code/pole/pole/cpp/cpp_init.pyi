from typing import Sequence, Tuple, Dict, Optional, Union, Iterable
import numpy as np
from os import PathLike


class Pole:
    def __init__(self,  mass: float,  length: float):
        ...

    def reset(self):
        ...

    def act(self,  torque: float):
        ...

    @property
    def theta(self):
        ...

    @theta.setter
    def theta(self,  value):
        ...

    @property
    def theta_dot(self):
        ...

    @theta_dot.setter
    def theta_dot(self,  value):
        ...

    @property
    def time(self):
        ...

    @time.setter
    def time(self,  value):
        ...

    @property
    def dt(self):
        ...

    @dt.setter
    def dt(self,  value):
        ...


class Agent:
    def dummy(self):
        """Dummy function so that the pyi generator doesn't break things."""
        ...


class PIDAgent(Agent):
    def __init__(self,  k_p,  k_i,  k_d):
        ...

    def run_step(self):
        ...

    def run_episode(self,  max_steps):
        ...

    def begin_episode(self):
        ...

    def set_environment(self, pole: Pole):
        ...

    def get_data(self) -> dict:
        ...


class Season:
    def __init__(self, agent: Agent):
        ...

    def clear_all_logs(self):
        ...

    def get_data_log(self):
        ...

    def get_scalar_data(self):
        ...

    def run(self, n_episodes, n_record, max_steps):
        ...


class PolyRLAgent(Agent):
    def __init__(
                self,  
                eps, 
                gamma, 
                alpha, 
                min_action, 
                max_action, 
                action_variance):
        ...

    def run_step(self):
        ...

    def run_episode(self,  max_steps):
        ...

    def begin_episode(self):
        ...

    def set_environment(self, pole: Pole):
        ...

    def get_data(self) -> dict:
        ...

    @property
    def eps(self):
        ...

    @eps.setter
    def eps(self,  value):
        ...

    @property
    def gamma(self):
        ...

    @gamma.setter
    def gamma(self,  value):
        ...

    @property
    def alpha(self):
        ...

    @alpha.setter
    def alpha(self,  value):
        ...

    @property
    def min_action(self):
        ...

    @min_action.setter
    def min_action(self,  value):
        ...

    @property
    def max_action(self):
        ...

    @max_action.setter
    def max_action(self,  value):
        ...

    @property
    def action_variance(self):
        ...

    @action_variance.setter
    def action_variance(self,  value):
        ...


class TableRLAgent(Agent):
    def __init__(self, 
                  min_theta, 
                  max_theta, 
                  min_theta_dot, 
                  max_theta_dot, 
                  min_torque, 
                  max_torque, 
                  n_theta, 
                  n_theta_dot, 
                  n_torque, 
                  epsilon, 
                  gamma, 
                  alpha, 
                  n_bootstrapping):
        ...

    def foo(min_x,  max_x,  n_x,  idx):
        ...

    def run_step(self):
        ...

    def run_episode(self, max_steps):
        ...

    def begin_episode(self):
        ...

    def set_environment(self, pole: Pole):
        ...

    def get_data(self) -> dict:
        ...

    def get_value(self,  theta: float,  theta_dot: float,  torque: float) -> float:
        ...

    def choose_ideal_torque(self,  theta: float,  theta_dot: float) -> float:
        ...

    @property
    def eps(self):
        ...

    @eps.setter
    def eps(self,  value):
        ...

    @property
    def gamma(self):
        ...

    @gamma.setter
    def gamma(self,  value):
        ...

    @property
    def alpha(self):
        ...

    @alpha.setter
    def alpha(self,  value):
        ...

    @property
    def ideal_torque(self):
        ...

    @property
    def visit_count(self):
        ...


class TileCodingAgent(Agent):
    def __init__(
                self,  
                center, 
                tile_size, 
                tilings, 
                default_weight, 
                random_offsets, 
                min_action, 
                max_action, 
                n_actions, 
                epsilon, 
                gamma, 
                alpha, 
                vc_min_theta, 
                vc_max_theta, 
                vc_n_theta, 
                vc_min_theta_dot, 
                vc_max_theta_dot, 
                vc_n_theta_dot) :
        ...

    def foo(min_x,  max_x,  n_x,  idx):
        ...

    def run_step(self):
        ...

    def run_episode(self, max_steps):
        ...

    def begin_episode(self):
        ...

    def set_environment(self, pole: Pole):
        ...

    def get_data(self) -> dict:
        ...

    def get_visit_count(self):
        ...

    def get_greedy_action(self):
        ...

    def get_update_count(self):
        ...

    @property
    def epsilon(self):
        ...

    @epsilon.setter
    def epsilon(self,  value):
        ...

    @property
    def gamma(self):
        ...

    @gamma.setter
    def gamma(self,  value):
        ...

    @property
    def alpha(self):
        ...

    @alpha.setter
    def alpha(self,  value):
        ...

    @property
    def tilings(self):
        ...

