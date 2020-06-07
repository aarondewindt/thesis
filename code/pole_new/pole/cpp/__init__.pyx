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

    def __cinit__(self, mass: float, length: float):
        self.thisptr = new c_Environment(mass, length)

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
    cdef usize q_table_size_theta
    cdef usize q_table_size_theta_dot
    cdef usize q_table_size_torque

    def __cinit__(self,
                  env: Environment,
                  min_theta: float,
                  max_theta: float,
                  min_theta_dot: float,
                  max_theta_dot: float,
                  min_torque: float,
                  max_torque: float,
                  q_table_size_theta: int,
                  q_table_size_theta_dot: int,
                  q_table_size_torque: int,
                  epsilon: float,
                  gamma: float,
                  alpha: float):
            self.thisptr = new c_TableAgent(dereference(env.thisptr),
                                            min_theta,
                                            max_theta,
                                            min_theta_dot,
                                            max_theta_dot,
                                            min_torque,
                                            max_torque,
                                            q_table_size_theta,
                                            q_table_size_theta_dot,
                                            q_table_size_torque,
                                            epsilon,
                                            gamma,
                                            alpha)
            self.q_table_size_theta = q_table_size_theta
            self.q_table_size_theta_dot = q_table_size_theta_dot
            self.q_table_size_torque = q_table_size_torque

    def run_episode(self, max_steps: int):
        self.thisptr.run_episode(max_steps)

    def get_data(self):
        data = self.thisptr.get_data()
        return data_map_to_dataset(data)

    def get_reward_sum(self):
        return self.thisptr.get_reward_sum()

    def get_q_table_data(self):
        q_table_data = list(self.thisptr.get_q_table_data())
        q_table = np.array(q_table_data).reshape((self.q_table_size_torque,
                                                  self.q_table_size_theta,
                                                  self.q_table_size_theta_dot))
        return q_table

