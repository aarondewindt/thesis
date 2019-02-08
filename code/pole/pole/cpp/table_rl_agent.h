//
// Created by elvieto on 29-1-19.
//

#ifndef POLE_TABLE_RL_AGENT_H
#define POLE_TABLE_RL_AGENT_H

#include "agent.h"
#include "pole.h"
#include <map>


class TableRLAgent : Agent {
public:
    TableRLAgent(
        double min_theta,
        double max_theta,
        double min_theta_dot,
        double max_theta_dot,
        double min_torque,
        double max_torque,
        int q_table_size_theta,
        int q_table_size_theta_dot,
        int q_table_size_torque,
        double epsilon,
        double gamma,
        double alpha,
        int n_bootstrapping);

    ~TableRLAgent();

    bool run_step() override;
    void run_episode(long max_steps) override;
    void begin_episode() override;
    void end_episode() override;

    inline void set_environment(Pole *pole) override {
        this->pole = pole;
    }

    inline Pole *get_environment() override {
        return this->pole;
    }

    std::map<std::string, std::vector<double>*>* get_data() override;
    std::map<std::string, double> get_scalar_data() override;

    double choose_ideal_torque(double theta, double theta_dot);
    double get_value(double theta, double theta_dot, double torque);
    double get_max_value(double theta, double theta_dot);

    void q_table_prelookup(double theta, double theta_dot, double torque,
                           int &theta_idx, int &theta_dot_idx, int &torque_idx);


    // Can be changed
    double eps;
    double gamma;
    double alpha;
    double *rewards;
    int *past_theta_idx;
    int *past_theta_dot_idx;
    int *past_torque_idx;
    double ***q_table;
    unsigned long **visit_count;

    // do not change.
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
private:
    Pole *pole;
    std::map<std::string, std::vector<double>*> data_map;



};


#endif //POLE_TABLE_RL_AGENT_H
