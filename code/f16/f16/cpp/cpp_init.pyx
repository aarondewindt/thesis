from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cython.operator import dereference, postincrement
import xarray as xr
from itertools import product
import numpy as np


def hello_cython():
    print("hello cython")


cdef class Pole:
    cdef c_Pole *thisptr

    def __cinit__(self, mass: float, length: float):
        self.thisptr = new c_Pole(mass, length)

    def reset(self):
        self.thisptr.theta = 0.001
        self.thisptr.theta_dot = 0
        self.thisptr.time = 0

    def act(self, torque: float):
        cdef double reward = 99.0
        cdef bool is_terminal = True
        self.thisptr.act(torque, reward, is_terminal)
        return reward, is_terminal

    @property
    def theta(self):
        return self.thisptr.theta
    @theta.setter
    def theta(self, value):
        self.thisptr.theta = value

    @property
    def theta_dot(self):
        return self.thisptr.theta_dot
    @theta_dot.setter
    def theta_dot(self, value):
        self.thisptr.theta_dot = value

    @property
    def time(self):
        return self.thisptr.time
    @time.setter
    def time(self, value):
        self.thisptr.time = value

    @property
    def dt(self):
        return self.thisptr.dt
    @dt.setter
    def dt(self, value):
        self.thisptr.dt = value

cdef class Agent:
    cdef c_Agent* get_agent_ptr(self):
        return NULL
    
    def dummy(self):
        """Dummy function so that the pyi generator doesn't break things."""

cdef class PIDAgent(Agent):
    cdef c_PIDAgent *thisptr

    def __cinit__(self, k_p, k_i, k_d):
        self.thisptr = new c_PIDAgent(k_p, k_i, k_d)

    cdef c_Agent* get_agent_ptr(self):
        return <c_Agent*>self.thisptr

    def run_step(self):
        return self.thisptr.run_step()

    def run_episode(self, max_steps):
        self.thisptr.run_episode(max_steps)

    def begin_episode(self):
        self.thisptr.begin_episode()

    def set_environment(self, Pole pole: Pole):
        self.thisptr.set_environment(pole.thisptr)

    def get_data(self) -> dict:
        data = self.thisptr.get_data()
        return data_map_to_dataset(data, True)


cdef class Season:
    cdef c_Season *thisptr

    def __cinit__(self, Agent agent: Agent):
        self.thisptr = new c_Season(agent.get_agent_ptr())

    def clear_all_logs(self):
        self.thisptr.clear_all_logs()

    def get_data_log(self):
        log_data = self.thisptr.get_data_log()

        def data_generator():
            it = log_data.begin()
            while it != log_data.end():
                idx = dereference(it).first
                data_map = data_map_to_dataset(dereference(it).second, True)
                yield idx, data_map
                postincrement(it)
        
        return dict(data_generator())
    
    def get_scalar_data(self):
        data = self.thisptr.get_scalar_data()
        return data_map_to_dataset(data, True)

    def run(self, int n_episodes, int n_record, int max_steps):
        with nogil:
            self.thisptr.run(n_episodes, n_record, max_steps)

cdef data_map_to_dataset(dvec_smap_ptr data_map, bool delete_data):
    def data_generator():
        it = data_map.begin()
        while it != data_map.end():
            key = dereference(it).first.decode("utf-8")
            vector = list(dereference(dereference(it).second))
            yield key, vector
            if delete_data:
                del vector
            postincrement(it)

    data_dict = dict(data_generator())
    if "time" in data_dict:
        return xr.Dataset(
            data_vars={key: (("time",), vector) for key, vector in data_dict.items() if key != "time"},
            coords={"time": data_dict["time"]}
        )
    else:
        return xr.Dataset(data_vars={key: (("episode",), vector) for key, vector in data_dict.items()})

cdef class PolyRLAgent(Agent):
    cdef c_PolyRLAgent *thisptr

    def __cinit__(
                self, 
                eps,
                gamma,
                alpha,
                min_action,
                max_action,
                action_variance):
        self.thisptr = new c_PolyRLAgent(eps,
                gamma,
                alpha,
                min_action,
                max_action,
                action_variance)

    cdef c_Agent* get_agent_ptr(self):
        return <c_Agent*>self.thisptr

    def run_step(self):
        return self.thisptr.run_step()

    def run_episode(self, max_steps):
        self.thisptr.run_episode(max_steps)

    def begin_episode(self):
        self.thisptr.begin_episode()

    def set_environment(self, Pole pole: Pole):
        self.thisptr.set_environment(pole.thisptr)

    def get_data(self) -> dict:
        data = self.thisptr.get_data()
        return data_map_to_dataset(data, True)


    @property
    def eps(self):
        return self.thisptr.eps
    @eps.setter
    def eps(self, value):
        self.thisptr.eps = value

    @property
    def gamma(self):
        return self.thisptr.gamma
    @gamma.setter
    def gamma(self, value):
        self.thisptr.gamma = value

    @property
    def alpha(self):
        return self.thisptr.alpha
    @alpha.setter
    def alpha(self, value):
        self.thisptr.alpha = value

    @property
    def min_action(self):
        return self.thisptr.min_action
    @min_action.setter
    def min_action(self, value):
        self.thisptr.min_action = value

    @property
    def max_action(self):
        return self.thisptr.max_action
    @max_action.setter
    def max_action(self, value):
        self.thisptr.max_action = value

    @property
    def action_variance(self):
        return self.thisptr.action_variance
    @action_variance.setter
    def action_variance(self, value):
        self.thisptr.action_variance = value


cdef class TableRLAgent(Agent):
    cdef c_TableRLAgent *thisptr
    cdef list thetas
    cdef list theta_dots
    cdef list torques

    def __cinit__(self,
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
        self.thisptr = new c_TableRLAgent(
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
            n_bootstrapping)

        # Calculates the value for a specific index.
        def foo(min_x, max_x, n_x, idx):
            return idx * (max_x - min_x) / (n_x - 1) + min_x

        self.thetas = [foo(min_theta, max_theta, n_theta, i) for i in range(n_theta)]
        self.theta_dots = [foo(min_theta_dot, max_theta_dot, n_theta_dot, i) for i in range(n_theta_dot)]
        self.torques = [foo(min_torque, max_torque, n_torque, i) for i in range(n_torque)]

    cdef c_Agent* get_agent_ptr(self):
        return <c_Agent*>self.thisptr

    def run_step(self):
        return self.thisptr.run_step()

    def run_episode(self, int max_steps):
        self.thisptr.run_episode(max_steps)

    def begin_episode(self):
        self.thisptr.begin_episode()

    def set_environment(self, Pole pole: Pole):
        self.thisptr.set_environment(pole.thisptr)

    def get_data(self) -> dict:
        data = self.thisptr.get_data()
        return data_map_to_dataset(data, True)

    def get_value(self, theta: float, theta_dot: float, torque: float) -> float:
        return self.thisptr.get_value(theta, theta_dot, torque)

    def choose_ideal_torque(self, theta: float, theta_dot: float) -> float:
        return self.thisptr.choose_ideal_torque(theta, theta_dot)

    @property
    def eps(self):
        return self.thisptr.eps
    @eps.setter
    def eps(self, value):
        self.thisptr.eps = value

    @property
    def gamma(self):
        return self.thisptr.gamma
    @gamma.setter
    def gamma(self, value):
        self.thisptr.gamma = value

    @property
    def alpha(self):
        return self.thisptr.alpha
    @alpha.setter
    def alpha(self, value):
        self.thisptr.alpha = value

    @property
    def ideal_torque(self):
        ideal_torque = np.ones((len(self.thetas), len(self.theta_dots))) * np.nan
        for (idx_theta, theta), (idx_theta_dot, theta_dot) in product(enumerate(self.thetas), enumerate(self.theta_dots)):
            ideal_torque[idx_theta, idx_theta_dot] = self.choose_ideal_torque(theta, theta_dot)

        return xr.DataArray(
            ideal_torque,
            coords={
                'theta': self.thetas,
                'theta_dot': self.theta_dots
            },
            dims=['theta', 'theta_dot']
        )
        

    @property
    def visit_count(self):
        visit_count = np.ones((len(self.thetas), len(self.theta_dots)))
        for idx_theta, idx_theta_dot in product(range(len(self.thetas)), range(len(self.theta_dots))):
            visit_count[idx_theta, idx_theta_dot] = self.thisptr.visit_count[idx_theta][idx_theta_dot]
        return xr.DataArray(
            visit_count,
            coords={
                'theta': self.thetas,
                'theta_dot': self.theta_dots
            },
            dims=['theta', 'theta_dot']
        )
        
cdef class TileCodingAgent(Agent):
    cdef c_TileCodingAgent *thisptr
    cdef readonly list actions
    cdef readonly list thetas
    cdef readonly list theta_dots

    def __cinit__(
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
 
        self.thisptr = new c_TileCodingAgent(
            center, tile_size, tilings, default_weight, random_offsets,
            min_action, max_action, n_actions, epsilon, gamma, alpha,
            vc_min_theta, vc_max_theta, vc_n_theta, vc_min_theta_dot, 
            vc_max_theta_dot, vc_n_theta_dot,)

        self.actions = [self.thisptr.actions[i] for i in range(n_actions)]

        # Calculates the value for a specific index.
        def foo(min_x, max_x, n_x, idx):
            return idx * (max_x - min_x) / (n_x - 1) + min_x

        self.thetas = [foo(vc_min_theta, vc_max_theta, vc_n_theta, i) for i in range(vc_n_theta)]
        self.theta_dots = [foo(vc_min_theta_dot, vc_max_theta_dot, vc_n_theta_dot, i) for i in range(vc_n_theta_dot)]

    cdef c_Agent* get_agent_ptr(self):
        return <c_Agent*>self.thisptr

    def run_step(self):
        return self.thisptr.run_step()

    def run_episode(self, int max_steps):
        self.thisptr.run_episode(max_steps)

    def begin_episode(self):
        self.thisptr.begin_episode()

    def set_environment(self, Pole pole: Pole):
        self.thisptr.set_environment(pole.thisptr)

    def get_data(self) -> dict:
        data = self.thisptr.get_data()
        return data_map_to_dataset(data, True)

    def get_visit_count(self):
        visit_count = np.ones((len(self.thetas), len(self.theta_dots)))
        for idx_theta, idx_theta_dot in product(range(len(self.thetas)), range(len(self.theta_dots))):
            visit_count[idx_theta, idx_theta_dot] = self.thisptr.visit_count[idx_theta][idx_theta_dot]
        return xr.DataArray(
            visit_count,
            coords={
                'theta': self.thetas,
                'theta_dot': self.theta_dots
            },
            dims=['theta', 'theta_dot']
        )

    def get_greedy_action(self):
        # Generate greedy action map and get the pointer.
        cdef double **greedy_action_ptr = self.thisptr.greedy_action_map()

        # Move data from the pointer to a numpy array.
        greedy_action = np.ones((len(self.thetas), len(self.theta_dots)))
        for idx_theta, idx_theta_dot in product(range(len(self.thetas)), range(len(self.theta_dots))):
            greedy_action[idx_theta, idx_theta_dot] = greedy_action_ptr[idx_theta][idx_theta_dot]

        # Create DataArray
        data_array =  xr.DataArray(
            greedy_action,
            coords={
                'theta': self.thetas,
                'theta_dot': self.theta_dots
            },
            dims=['theta', 'theta_dot']
        )

        # Delete pointer and return DataArray.
        free(greedy_action_ptr)
        return data_array;

    def get_update_count(self):
        # Generate greedy action map and get the pointer.
        cdef double **update_count_ptr = self.thisptr.update_count_map()

        # Move data from the pointer to a numpy array.
        update_count = np.ones((len(self.thetas), len(self.theta_dots)))
        for idx_theta, idx_theta_dot in product(range(len(self.thetas)), range(len(self.theta_dots))):
            update_count[idx_theta, idx_theta_dot] = update_count_ptr[idx_theta][idx_theta_dot]

        # Create DataArray
        data_array =  xr.DataArray(
            update_count,
            coords={
                'theta': self.thetas,
                'theta_dot': self.theta_dots
            },
            dims=['theta', 'theta_dot']
        )

        # Delete pointer and return DataArray.
        free(update_count_ptr)
        return data_array;

    @property
    def epsilon(self):
        return self.thisptr.epsilon
    @epsilon.setter
    def epsilon(self, value):
        self.thisptr.epsilon = value

    @property
    def gamma(self):
        return self.thisptr.gamma
    @gamma.setter
    def gamma(self, value):
        self.thisptr.gamma = value

    @property
    def alpha(self):
        return self.thisptr.alpha
    @alpha.setter
    def alpha(self, value):
        self.thisptr.alpha = value

    @property
    def tilings(self):
        return self.thisptr.tilings
