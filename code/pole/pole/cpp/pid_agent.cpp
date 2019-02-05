//
// Created by elvieto on 24-11-18.
//

#include "pid_agent.h"
#include <cmath>
#include <map>
#include <printf.h>


PIDAgent::PIDAgent(double k_p, double k_i, double k_d) :
    k_p(k_p), k_i(k_i), k_d(k_d),
    integral(0), previous_error(NAN),
    pole(nullptr),
    data_map(std::map<std::string, std::vector<double>*>()) {

    // Initialize data map.
    data_map["time"] = new std::vector<double>();
    data_map["theta"] = new std::vector<double>();
    data_map["theta_dot"] = new std::vector<double>();
    data_map["error"] = new std::vector<double>();
    data_map["integral"] = new std::vector<double>();
    data_map["derivative"] = new std::vector<double>();
    data_map["torque"] = new std::vector<double>();
    data_map["reward"] = new std::vector<double>();
}

PIDAgent::~PIDAgent(){
    // Clear the memory used by the data vectors.
    for (auto const& item : data_map) {
        delete item.second;
    }
}

void PIDAgent::begin_episode() {
    if (pole == nullptr) {
        throw "No pole set";
    }
    integral = 0;
    previous_error = NAN;

    // Clear data map.
    for (auto const& item : data_map)
    {
        item.second->clear();
    }
}

std::map<std::string, std::vector<double>*>* PIDAgent::get_data() {
    auto data_map_copy = new std::map<std::string, std::vector<double>*>();

    for (auto const& item : data_map) {
        auto new_data = new std::vector<double>(*(item.second));
        (*data_map_copy)[item.first] = new_data;

    }

    return data_map_copy;
}

std::map<std::string, double> PIDAgent::get_scalar_data() {
    std::map<std::string, double> scalar_map;
    double temp = 0;
    for (auto& reward : (*data_map["reward"])) {
        temp += reward;
    }
    scalar_map["reward_sum"] = temp;
    return scalar_map;
}

bool PIDAgent::run_step(){
    if (pole == nullptr) {
        throw "No pole set";
    }

    // Calculate error.
    double error = -pole->theta;
    double derivative;
    double torque;

    // Calculate integral and derivative.
    if (std::isnan(previous_error)) {
        integral = 0;
        derivative = 0;
    } else {
        integral += (previous_error + error) / 2 * pole->dt;
        derivative = (error - previous_error) / pole->dt;
    }

    previous_error = error;

    // Calculate torque and run single step.
    torque = k_p * error + k_i * integral + k_d * derivative;
    double reward;
    bool is_terminal;
    pole->act(torque, reward, is_terminal);

    // Log state.
    data_map["time"]->push_back(pole->time);
    data_map["theta"]->push_back(pole->theta);
    data_map["theta_dot"]->push_back(pole->theta_dot);
    data_map["error"]->push_back(error);
    data_map["integral"]->push_back(integral);
    data_map["derivative"]->push_back(derivative);
    data_map["torque"]->push_back(torque);
    data_map["reward"]->push_back(reward);

    return is_terminal;
}

void PIDAgent::run_episode(long max_steps) {
    begin_episode();
    for (;max_steps--;){
        if (run_step())
            break;
    }
}
