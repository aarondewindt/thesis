from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdint cimport uint64_t, int64_t
from libcpp.string cimport string

ctypedef double f64
ctypedef int64_t i64

cdef extern from "environment.h" namespace "pole":
    cdef cppclass c_Environment "pole::Environment":
        c_Environment(f64 mass, f64 length)
        void reset(f64 theta_, f64 theta_dot_)
        pair[f64, bool] step(f64 torque)
        f64 mass
        f64 length
        f64 inertia
        f64 time
        f64 dt
        f64 theta_min_reward
        f64 theta
        f64 theta_dot

cdef extern from "agent_base.h":
    cdef cppclass c_AgentBase "AgentBase":
        pass

cdef extern from "pid_agent.h" namespace "pole":
    cdef cppclass c_PIDAgent "pole::PIDAgent":
        c_PIDAgent(c_Environment& env, double k_p, double k_i, double k_d)
        bool run_step()
        void run_episode(i64 max_steps)
        void begin_episode()
        map[string, vector[f64]] get_data()
        map[string, f64] get_scalar_data
        f64 get_reward_sum()
