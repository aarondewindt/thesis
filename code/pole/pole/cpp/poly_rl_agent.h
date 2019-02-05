//
// Created by elvieto on 25-11-18.
//

#ifndef POLE_POLY_RL_AGENT_H
#define POLE_POLY_RL_AGENT_H

#include <Eigen/Dense>
#include <vector>
#include "pole.h"
#include "agent.h"


const size_t n_terms = 27;
typedef Eigen::Matrix<double, n_terms, 1> PolyRLColVector;
typedef Eigen::Matrix<double, 1, n_terms> PolyRLRowVector;


class PolyRLAgent : Agent {
public:
    PolyRLAgent(
            double eps,
            double gamma,
            double alpha,
            double min_action,
            double max_action,
            double action_variance);

    explicit PolyRLAgent(
            double eps,
            double gamma,
            double alpha,
            double min_action,
            double max_action,
            double action_variance,
            std::vector<double> *initial_weights);

    bool run_step() override;
    void run_episode(long max_steps) override;
    void begin_episode() override;
    std::map<std::string, std::vector<double>*>* get_data() override;
    std::map<std::string, double> get_scalar_data() override;

    inline void set_environment(Pole *pole) override {
        this->pole = pole;
    }

    inline Pole *get_environment() override {
        return this->pole;
    }

    PolyRLColVector get_x(double s1, double s2, double a);
    double get_greedy_action(double s1, double s2);

    double value(double s1, double s2, double a);

    std::vector<double> get_weights();
    void set_weights(std::vector<double> &weights);

    double eps;
    double gamma;
    double alpha;
    double min_action;
    double max_action;
    double action_variance;

private:
    PolyRLRowVector w;
    Pole *pole;

    double action_t;

    std::map<std::string, std::vector<double>*> data_map;

    void initialize_data_map();
};


#endif //POLE_POLY_RL_AGENT_H
