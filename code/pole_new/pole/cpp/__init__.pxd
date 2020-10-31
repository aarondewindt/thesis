from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdint cimport uint64_t, int64_t
from libcpp.string cimport string

ctypedef double f64
ctypedef int64_t i64
ctypedef size_t usize


cdef extern from "environment.h" namespace "pole":
    cdef cppclass c_Environment "pole::Environment":
        c_Environment(f64 mass, f64 length, f64 theta_terminate, f64 theta_min_reward)
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
        i64 run_episode(i64 max_steps)
        void begin_episode()
        map[string, vector[f64]] get_data()
        map[string, f64] get_scalar_data
        f64 get_reward_sum()

cdef extern from "table_agent.h" namespace "pole":
    cdef cppclass c_TableAgent "pole::TableAgent":
        c_TableAgent(c_Environment& env,
                     f64 min_action,
                     f64 max_action,
                     f64 min_theta,
                     f64 max_theta,
                     f64 min_theta_dot,
                     f64 max_theta_dot,
                     usize n_action,
                     usize n_theta,
                     usize n_theta_dot,
                     f64 epsilon,
                     f64 gamma,
                     f64 alpha)
        bool run_step()
        i64 run_episode(i64 max_steps)
        void begin_episode()
        map[string, vector[f64]] get_data()
        map[string, f64] get_scalar_data
        f64 get_reward_sum()
        vector[f64]& get_values()
        vector[f64]& get_counts()
        vector[f64]& get_greedy_action_table()
        f64 epsilon;
        f64 gamma;
        f64 alpha;


cdef extern from "grid_tile_coding_agent.h" namespace "pole":
    cdef cppclass c_TileCodingAgent "pole::TileCodingAgent":
        c_TileCodingAgent(c_Environment& env,
                        vector[f64] center,
                        vector[f64] tile_size,
                        int tilings,
                        double default_weight,
                        bool random_offsets,
                        double min_action,
                        double max_action,
                        int n_actions,
                        double epsilon,
                        double gamma,
                        double alpha,
                        double vc_min_theta,
                        double vc_max_theta,
                        int vc_n_theta,
                        double vc_min_theta_dot,
                        double vc_max_theta_dot,
                        int vc_n_theta_dot)
        bool run_step()
        i64 run_episode(i64 max_steps)
        void begin_episode()
        map[string, vector[f64]] get_data()
        map[string, f64] get_scalar_data
        f64 get_reward_sum()
        vector[f64]& get_greedy_action_table()
        vector[f64]& get_counts()
        f64 epsilon;
        f64 gamma;
        f64 alpha;
        double *actions;
