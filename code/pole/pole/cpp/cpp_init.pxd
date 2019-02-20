from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libc.stdint cimport uint64_t

ctypedef vector[double]* dvec_ptr
ctypedef map[string, dvec_ptr]* dvec_smap_ptr 

cdef extern from "pole.h":
    cdef cppclass c_Pole "Pole":
        c_Pole(double mass, double length)
        void act(double torque, double &reward, bool &is_terminal)
        double theta
        double theta_dot
        double time
        double dt
        double mass
        double length
        double inertia

cdef extern from "agent.h":
    cdef cppclass c_Agent "Agent":
        pass

cdef extern from "pid_agent.h":
    cdef cppclass c_PIDAgent "PIDAgent":
        c_PIDAgent(double k_p, double k_i, double k_d)
        bool run_step()
        void run_episode(long max_steps)
        void begin_episode()
        void set_environment(c_Pole *pole)
        c_Pole* set_environment()
        map[string, dvec_ptr]* get_data()

cdef extern from "season.h":
    cdef cppclass c_Season "Season":
        c_Season(c_Agent *agent)
        map[long, dvec_smap_ptr]* get_data_log()
        map[string, dvec_ptr]* get_scalar_data()
        void clear_all_logs()
        void run(long n_episodes, long n_record, long max_steps) nogil

cdef extern from "poly_rl_agent.h":
    cdef cppclass c_PolyRLAgent "PolyRLAgent":
        c_PolyRLAgent(
            double eps,
            double gamma,
            double alpha,
            double min_action,
            double max_action,
            double action_variance)

        double eps
        double gamma
        double alpha
        double min_action
        double max_action
        double action_variance

        bool run_step()
        void run_episode(long max_steps)
        void begin_episode()
        void set_environment(c_Pole *pole)
        c_Pole* set_environment()
        map[string, dvec_ptr]* get_data()


cdef extern from "table_rl_agent.h":
    cdef cppclass c_TableRLAgent "TableRLAgent":
        c_TableRLAgent(
            double min_theta,
            double max_theta,
            double min_theta_dot,
            double max_theta_dot,
            double min_torque,
            double max_torque,
            int n_theta,
            int n_theta_dot,
            int n_torque,
            double epsilon,
            double gamma,
            double alpha,
            int n_bootstrapping)

        double eps
        double gamma
        double alpha
        unsigned long **visit_count
        double min_theta;
        double max_theta;
        double min_theta_dot;
        double max_theta_dot;
        double min_torque;
        double max_torque;
        int n_theta;
        int n_theta_dot;
        int n_torque;
        int n_bootstrapping;

        bool run_step()
        void run_episode(long max_steps)
        void begin_episode()
        void set_environment(c_Pole *pole)
        c_Pole* set_environment()
        map[string, dvec_ptr]* get_data()
        double get_value(double theta, double theta_dot, double torque)
        double choose_ideal_torque(double theta, double theta_dot)


cdef extern from "tile_coding_agent.h":
    cdef cppclass c_TileCodingAgent "TileCodingAgent":
        c_TileCodingAgent(
            vector[double] center,
            vector[double] tile_size,
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

        double epsilon;
        double gamma;
        double alpha;
        double *actions;
        int n_actions;
        int tilings;
        uint64_t **visit_count

        double **greedy_action_map()
        double **update_count_map()

        bool run_step()
        void run_episode(long max_steps)
        void begin_episode()
        void set_environment(c_Pole *pole)
        c_Pole* set_environment()
        map[string, dvec_ptr]* get_data()
