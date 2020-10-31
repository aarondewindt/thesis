from cython.operator import dereference, postincrement
import xarray as xr
import numpy as np
cimport numpy as np


cdef data_map_to_dataset(map[string, vector[f64]] data_map):
    def data_generator():
        it = data_map.begin()
        while it != data_map.end():
            key = dereference(it).first.decode("utf-8")
            vector = list(dereference(it).second)
            yield key, vector
            postincrement(it)

    data_dict = dict(data_generator())
    if "time" in data_dict:
        return xr.Dataset(
            data_vars={key: (("time",), vector) for key, vector in data_dict.items() if key != "time"},
            coords={"time": data_dict["time"]}
        )
    else:
        return xr.Dataset(data_vars={key: (("episode",), vector) for key, vector in data_dict.items()})


cdef class Environment:
    cdef c_Environment *thisptr

    def __cinit__(self, mass: float, length: float, f64 theta_terminate: float, f64 theta_min_reward: float):
        self.thisptr = new c_Environment(mass, length, theta_terminate, theta_min_reward)

    def reset(self, theta: float=0, theta_dot: float=0):
        self.thisptr.reset(theta, theta_dot)

    def step(self, torque: float):
        return self.thisptr.step(torque)

    @property
    def mass(self):
        return self.thisptr.mass
    @mass.setter
    def mass(self, value):
        self.thisptr.mass = value

    @property
    def length(self):
        return self.thisptr.length
    @length.setter
    def length(self, value):
        self.thisptr.length = value

    @property
    def inertia(self):
        return self.thisptr.inertia
    @inertia.setter
    def inertia(self, value):
        self.thisptr.inertia = value

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

    @property
    def theta_min_reward(self):
        return self.thisptr.theta_min_reward
    @theta_min_reward.setter
    def theta_min_reward(self, value):
        self.thisptr.theta_min_reward = value

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


cdef class PIDAgent:
    cdef c_PIDAgent *thisptr

    def __cinit__(self, env: Environment, k_p: float, k_i: float, k_d: float):
            self.thisptr = new c_PIDAgent(dereference(env.thisptr), k_p, k_i, k_d)

    def run_episode(self, max_steps: int):
        self.thisptr.run_episode(max_steps)

    def get_data(self):
            data = self.thisptr.get_data()
            return data_map_to_dataset(data)


cdef class TableAgent:
    cdef c_TableAgent *thisptr
    cdef usize n_action
    cdef usize n_theta
    cdef usize n_theta_dot
    cdef object _actions
    cdef object _thetas
    cdef object _theta_dots

    def __cinit__(self,
                  env: Environment,
                  min_action: float,
                  max_action: float,
                  min_theta: float,
                  max_theta: float,
                  min_theta_dot: float,
                  max_theta_dot: float,
                  n_action: int,
                  n_theta: int,
                  n_theta_dot: int,
                  epsilon: float,
                  gamma: float,
                  alpha: float):
        self.thisptr = new c_TableAgent(dereference(env.thisptr),
                                        min_action,
                                        max_action,
                                        min_theta,
                                        max_theta,
                                        min_theta_dot,
                                        max_theta_dot,
                                        n_action,
                                        n_theta,
                                        n_theta_dot,
                                        epsilon,
                                        gamma,
                                        alpha)
        self.n_action = n_action
        self.n_theta = n_theta
        self.n_theta_dot = n_theta_dot

        delta_action = (max_action - min_action) / n_action
        delta_theta = (max_theta - min_theta) / n_theta
        delta_theta_dot = (max_theta_dot - min_theta_dot) / n_theta_dot

        self._actions = [min_action + delta_action * (i + 0.5) for i in range(n_action)]
        self._thetas = [min_theta + delta_theta * (i + 0.5) for i in range(n_theta)]
        self._theta_dots = [min_theta_dot + delta_theta_dot * (i + 0.5) for i in range(n_theta_dot)]

    @property
    def actions(self):
        return self._actions

    @property
    def thetas(self):
        return self._thetas

    @property
    def theta_dots(self):
        return self._theta_dots

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


    def run_episode(self, max_steps: int):
        return self.thisptr.run_episode(max_steps)

    def get_data(self):
        data = self.thisptr.get_data()
        return data_map_to_dataset(data)

    def get_reward_sum(self):
        return self.thisptr.get_reward_sum()

    def get_values(self):
        data = list(self.thisptr.get_values())
        print(self.n_action, self.n_theta, self.n_theta_dot)
        data = np.array(data).reshape((self.n_action,
                                       self.n_theta,
                                       self.n_theta_dot), order="F")
        return data

    def get_counts(self):
        data = list(self.thisptr.get_counts())
        data = np.array(data).reshape((self.n_action,
                                       self.n_theta,
                                       self.n_theta_dot), order="F")
        return data

    def get_greedy_action_table(self):
            data = list(self.thisptr.get_greedy_action_table())
            data = np.array(data).reshape((self.n_theta,
                                           self.n_theta_dot), order="F")
            return data


cdef class TileCodingAgent:
    cdef c_TileCodingAgent *thisptr

    cdef usize n_theta
    cdef usize n_theta_dot

    cdef object _actions
    cdef object _thetas
    cdef object _theta_dots

    def __cinit__(self, env: Environment,
                        center: list,
                        tile_size: list,
                        tilings: int,
                        default_weight: float,
                        random_offsets: bool,
                        min_action: float,
                        max_action: float,
                        n_actions: int,
                        epsilon: float,
                        gamma: float,
                        alpha: float,
                        vc_min_theta: float,
                        vc_max_theta: float,
                        vc_n_theta: int,
                        vc_min_theta_dot: float,
                        vc_max_theta_dot: float,
                        vc_n_theta_dot: int):
        self.thisptr = new c_TileCodingAgent(dereference(env.thisptr),
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
                                             vc_n_theta_dot)
        self.n_theta = vc_n_theta
        self.n_theta_dot = vc_n_theta_dot

        self._actions = [self.thisptr.actions[i] for i in range(n_actions)]

        # Calculates the value for a specific index.
        def foo(min_x, max_x, n_x, idx):
            return idx * (max_x - min_x) / (n_x - 1) + min_x

        self._thetas = [foo(vc_min_theta, vc_max_theta, vc_n_theta, i) for i in range(vc_n_theta)]
        self._theta_dots = [foo(vc_min_theta_dot, vc_max_theta_dot, vc_n_theta_dot, i) for i in range(vc_n_theta_dot)]

    @property
    def actions(self):
        return self._actions

    @property
    def thetas(self):
        return self._thetas

    @property
    def theta_dots(self):
        return self._theta_dots

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

    def run_episode(self, max_steps: int):
            return self.thisptr.run_episode(max_steps)

    def get_data(self):
        data = self.thisptr.get_data()
        return data_map_to_dataset(data)

    def get_reward_sum(self):
        return self.thisptr.get_reward_sum()

    def get_greedy_action_table(self):
        data = list(self.thisptr.get_greedy_action_table())
        data = np.array(data).reshape((self.n_theta,
                                       self.n_theta_dot), order="F")
        return data

    def get_counts(self):
        data = list(self.thisptr.get_counts())
        data = np.array(data).reshape((self.n_theta,
                                       self.n_theta_dot), order="F")
        return data
