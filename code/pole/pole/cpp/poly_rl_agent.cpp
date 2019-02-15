//
// Created by elvieto on 25-11-18.
//
// Note: This code will not be used anymore, but it I keeping it for now
//       as reference for some stuff I did here.
//

#include "poly_rl_agent.h"
#include "rand.h"
#include <cmath>
#include <iostream>


PolyRLAgent::PolyRLAgent(
        double eps,
        double gamma,
        double alpha,
        double min_action,
        double max_action,
        double action_variance) :
            pole(nullptr),
            eps(eps),
            gamma(gamma),
            alpha(alpha),
            min_action(min_action),
            max_action(max_action),
            action_variance(action_variance),
            action_t(NAN)
            {
    w.setZero();
//    w[2] = 5;
    initialize_data_map();
}

PolyRLAgent::PolyRLAgent(
        double eps,
        double gamma,
        double alpha,
        double min_action,
        double max_action,
        double action_variance,
        std::vector<double> *initial_weights) :
            w(initial_weights->data()),
            pole(nullptr),
            eps(eps),
            gamma(gamma),
            alpha(alpha),
            min_action(min_action),
            max_action(max_action),
            action_variance(action_variance),
            action_t(NAN) {
    initialize_data_map();
}

void PolyRLAgent::initialize_data_map() {
    data_map["time"] = new std::vector<double>();
    data_map["theta"] = new std::vector<double>();
    data_map["theta_dot"] = new std::vector<double>();
    data_map["action"] = new std::vector<double>();
    data_map["reward"] = new std::vector<double>();
    data_map["delta_v"] = new std::vector<double>();
}


PolyRLColVector PolyRLAgent::get_x(double s1, double s2, double a) {
    PolyRLColVector x;
    x <<
      1,
            a,
            a*a,
            s2,
            a*s2,
            a*a*s2,
            s2*s2,
            a*s2*s2,
            a*a*s2*s2,
            s1,
            a*s1,
            a*a*s1,
            s1*s2,
            a*s1*s2,
            a*a*s1*s2,
            s1*s2*s2,
            a*s1*s2*s2,
            a*a*s1*s2*s2,
            s1*s1,
            a*s1*s1,
            a*a*s1*s1,
            s1*s1*s2,
            a*s1*s1*s2,
            a*a*s1*s1*s2,
            s1*s1*s2*s2,
            a*s1*s1*s2*s2,
            a*a*s1*s1*s2*s2
            ;
    return x;
}

std::vector<double> PolyRLAgent::get_weights() {
    return std::vector<double>(w.data(), w.data() + w.rows() * w.cols());
}

void PolyRLAgent::set_weights(std::vector<double> &weights) {
    w = PolyRLColVector(weights.data());
}

bool PolyRLAgent::run_step() {
    alpha *= 0.99;
    if (pole == nullptr) {
        throw "No pole set";
    }
    // Get greedy action.
    double action;

    // Add randomize the action if exploring.
    if (frand(0, 1) > eps) {
        action = frand(min_action, max_action);
    } else {
        action = get_greedy_action(pole->theta, pole->theta_dot);
    }

//    action = 0.0;

    // Store values for time step t.
    double theta_t = pole->theta;
    double theta_dot_t = pole->theta_dot;
    PolyRLColVector x_t = get_x(theta_t, theta_dot_t, action_t);

    // Perform action
    double reward;
    bool is_terminal;
    pole->act(action, reward, is_terminal);

    // Calculate target value output.
    double u_t = reward + gamma * value(pole->theta, pole->theta_dot, action);

    // Value update
    PolyRLRowVector delta_v = PolyRLColVector::Zero();

    if (!std::isnan(action_t)) {
        // Calculate value update
        delta_v = alpha * (u_t - value(theta_t, theta_dot_t, action_t)) * x_t;

        // Update t+1 weights.
        PolyRLRowVector w_t1 = w + delta_v;
        w = w_t1;
    }

    // Store the t+1 action for the next iteration.
    action_t = action;

    double norm = delta_v.norm();

    // Log results.
    data_map["time"]->push_back(pole->time);
    data_map["theta"]->push_back(pole->theta);
    data_map["theta_dot"]->push_back(pole->theta_dot);
    data_map["action"]->push_back(action);
    data_map["reward"]->push_back(reward);
    data_map["delta_v"]->push_back(norm);

    return is_terminal;
}

double PolyRLAgent::get_greedy_action(double s1, double s2) {
    // TODO: Think of something better.
    double delta_action = (max_action - min_action) / 100;
    double action = min_action;

    double value;
    double max_value_action = frand(min_action, max_action);
    double max_value = this->value(s1, s2, action);
    while (action < max_action) {
        value = this->value(s1, s2, action);
//        std::printf("val %f  %f  %f\n", action, value, max_value);
        if (value > max_value) {
            max_value_action = action;
            max_value = value;
        }
        action += delta_action;
    }
    return max_value_action;
}

double PolyRLAgent::value(double s1, double s2, double a) {
    return w * get_x(s1, s2, a);

}

void PolyRLAgent::run_episode(long max_steps) {
    begin_episode();
    for (;max_steps--;){
        if (run_step())
            break;
    }
}

void PolyRLAgent::begin_episode() {
    if (pole == nullptr) {
        throw "No pole set";
    }

    // Clear data map.
    for (auto const& item : data_map)
    {
        item.second->clear();
    }
}

std::map<std::string, std::vector<double>*>* PolyRLAgent::get_data() {
    auto data_map_copy = new std::map<std::string, std::vector<double>*>();

    for (auto const& item : data_map) {
        auto new_data = new std::vector<double>(*(item.second));
        (*data_map_copy)[item.first] = new_data;

    }

    return data_map_copy;
}

std::map<std::string, double> PolyRLAgent::get_scalar_data() {
    std::map<std::string, double> scalar_map;
    double temp = 0;
    for (auto& reward : (*data_map["reward"])) {
        temp += reward;
    }
    scalar_map["reward_sum"] = temp;
    return scalar_map;
}
