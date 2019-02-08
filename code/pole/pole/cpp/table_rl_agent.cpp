//
// Created by elvieto on 29-1-19.
//

#include "table_rl_agent.h"
#include <cmath>
#include <math.h>
#include "rand.h"
#include <printf.h>


#define GET_IDX(min, interval, value) (int)(std::round(value / interval) - (min / interval))
#define GET_VALUE(min, max, n, idx) idx * ((max - min) / (n - 1)) + min

TableRLAgent::TableRLAgent(
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
        int n_bootstrapping) :
    min_theta(min_theta),
    max_theta(max_theta),
    min_theta_dot(min_theta_dot),
    max_theta_dot(max_theta_dot),
    min_torque(min_torque),
    max_torque(max_torque),
    n_theta(q_table_size_theta),
    n_theta_dot(q_table_size_theta_dot),
    n_torque(q_table_size_torque),
    eps(epsilon),
    gamma(gamma),
    alpha(alpha),
    n_bootstrapping(n_bootstrapping),
    pole(nullptr),
    data_map(std::map<std::string, std::vector<double>*>()) {

    // Initialize Q table. Set all values to 0.
    q_table = new double**[q_table_size_theta];
    for (int i = 0; i < q_table_size_theta; i++) {
        q_table[i] = new double*[q_table_size_theta_dot];
        for (int j = 0; j < q_table_size_theta_dot; j++) {
            q_table[i][j] = new double[q_table_size_torque]();
        }
    }

    // Initialize visit count. Set all values to 0.
    visit_count = new unsigned long*[q_table_size_theta];
    for (int i = 0; i < q_table_size_theta; i++) {
        visit_count[i] = new unsigned long[q_table_size_theta_dot]();
    }

    // Initialize data map.
     data_map["time"] = new std::vector<double>();
     data_map["theta"] = new std::vector<double>();
     data_map["theta_dot"] = new std::vector<double>();
     data_map["error"] = new std::vector<double>();
     data_map["torque"] = new std::vector<double>();
     data_map["reward"] = new std::vector<double>();
     data_map["learn_rate"] = new std::vector<double>();

    // Initialize past rewards and states.
    rewards = new double[n_bootstrapping]();
    past_theta_idx = new int[n_bootstrapping];
    past_theta_dot_idx = new int[n_bootstrapping];
    past_torque_idx = new int[n_bootstrapping];

    // Set past_theta_idx to -1. We can use this to check when we have enough
    // data to start updating values.
    for (int i = 0; i < n_bootstrapping; i++) {
        past_theta_idx[i] = -1;
    }
}

TableRLAgent::~TableRLAgent(){
    // Deallocate the q_table.
    for (int i = 0; i < n_theta; i++) {
        for (int j = 0; j < n_torque; j++) {
            delete q_table[i][j];
        }
        delete q_table[i];
    }
    delete q_table;
    delete rewards;
    delete past_theta_idx;
    delete past_theta_dot_idx;
    delete past_torque_idx;
}

bool TableRLAgent::run_step() {
    if (pole == nullptr) {
        throw "No pole set";
    }

    // Shift past rewards list to make space for the new reward.
    for (int i = 1; i < n_bootstrapping; i++) {
        rewards[i] = rewards[i - 1];
        past_theta_idx[i] = past_theta_idx[i - 1];
        past_theta_dot_idx[i] = past_theta_dot_idx[i - 1];
        past_torque_idx[i] = past_torque_idx[i - 1];
    }

    // Choose an action. Torque to apply on pole.
    double action;
    if (frand(0, 1) > eps) {
        action = GET_VALUE(min_torque, max_torque, n_torque, irand(0, n_torque - 1));
    } else {
        action = choose_ideal_torque(pole->theta, pole->theta_dot);
    }

    {
        // Get the indexes of the current state and chosen action
        int theta_idx, theta_dot_idx, torque_idx;
        q_table_prelookup(pole->theta, pole->theta_dot, action, theta_idx, theta_dot_idx, torque_idx);

        // Store the indexes in the past lists, so they can be used later to update the value table.
        past_theta_idx[0] = theta_idx;
        past_theta_dot_idx[0] = theta_dot_idx;
        past_torque_idx[0] = torque_idx;
        visit_count[theta_idx][theta_dot_idx] += 1;
    }

    // Perform action
    bool is_terminal;
    pole->act(action, *rewards, is_terminal);

    // Default learn_rate value to nan.
    // This way we know on the python side, when we still haven't updated states yet.
    double learn_rate = nan("");

    // If the last value in the past theta array is not a nan we have filled the array
    // and can start updating the values for the state-action pairs n_bootstrapping iterations ago.
    if (past_theta_idx[n_bootstrapping - 1] != -1)  {
        double g = 0.0;
        // Calculate cumulative discounted reward.
        // Items with i=0 are the newest, so they need to be multiplied with gamma^(n-1)
        // Items with index (n-1) are the oldest, so they need to be multiplied with gamma^(0)=1
        for (int i = 0; i < n_bootstrapping; i++) {
            g += rewards[i] * pow(gamma, n_bootstrapping - 1 - i);
        }

        // Add term with value at current state.
        g += pow(gamma, n_bootstrapping) * get_max_value(pole->theta, pole->theta_dot);

        // Update value
        int theta_idx = past_theta_idx[n_bootstrapping - 1];
        int theta_dot_idx = past_theta_dot_idx[n_bootstrapping - 1];
        int torque_idx = past_torque_idx[n_bootstrapping - 1];
        learn_rate = alpha * (g - q_table[theta_idx][theta_dot_idx][torque_idx]);
        q_table[theta_idx][theta_dot_idx][torque_idx] += learn_rate;
    }

    // Log results.
    data_map["time"]->push_back(pole->time);
    data_map["theta"]->push_back(pole->theta);
    data_map["theta_dot"]->push_back(pole->theta_dot);
    data_map["error"]->push_back(-pole->theta);
    data_map["torque"]->push_back(action);
    data_map["reward"]->push_back(*rewards);
    data_map["learn_rate"]->push_back(learn_rate);

    return is_terminal;
}

double TableRLAgent::choose_ideal_torque(double theta, double theta_dot) {
    int theta_idx, theta_dot_idx, _;
    q_table_prelookup(theta, theta_dot, 0.0, theta_idx, theta_dot_idx, _);

    int best_torque_idx = irand(0, n_torque - 1);
    double best_value = q_table[theta_idx][theta_dot_idx][best_torque_idx];
    for (int i = 0; i < n_torque; i++) {
        if (i == best_torque_idx) continue;
        if (q_table[theta_idx][theta_dot_idx][i] > best_value) {
            best_value = q_table[theta_idx][theta_dot_idx][i];
            best_torque_idx = i;
        }
    }

    return GET_VALUE(min_torque, max_torque, n_torque, best_torque_idx);
}

double TableRLAgent::get_value(double theta, double theta_dot, double torque) {
    int theta_idx, theta_dot_idx, torque_idx;
    q_table_prelookup(theta, theta_dot, torque, theta_idx, theta_dot_idx, torque_idx);
    return q_table[theta_idx][theta_dot_idx][torque_idx];
}


double TableRLAgent::get_max_value(double theta, double theta_dot) {
    int theta_idx, theta_dot_idx, _;
    q_table_prelookup(theta, theta_dot, 0.0, theta_idx, theta_dot_idx, _);

    int best_torque_idx = 0;
    double best_value = q_table[theta_idx][theta_dot_idx][0];
    for (int i = 1; i < n_torque; i++) {
        if (q_table[theta_idx][theta_dot_idx][i] > best_value) {
            best_value = q_table[theta_idx][theta_dot_idx][i];
            best_torque_idx = i;
        }
    }

    return q_table[theta_idx][theta_dot_idx][best_torque_idx];
}


void TableRLAgent::q_table_prelookup(double theta, double theta_dot, double torque,
                                     int &theta_idx, int &theta_dot_idx, int &torque_idx) {
    if (theta > max_theta) theta = max_theta;
    if (theta < min_theta) theta = min_theta;
    if (theta_dot > max_theta_dot) theta_dot = max_theta_dot;
    if (theta_dot < min_theta_dot) theta_dot = min_theta_dot;
    if (torque > max_torque) torque = max_torque;
    if (torque < min_torque) torque = min_torque;

    static double interval_theta = (max_theta - min_theta) / (n_theta - 1);
    static double interval_theta_dot = (max_theta_dot - min_theta_dot) / (n_theta_dot - 1);
    static double interval_torque = (max_torque - min_torque) / (n_torque - 1);

    theta_idx = GET_IDX(min_theta, interval_theta, theta);
    theta_dot_idx = GET_IDX(min_theta_dot, interval_theta_dot, theta_dot);
    torque_idx = GET_IDX(min_torque, interval_torque, torque);
}

void TableRLAgent::begin_episode() {
    if (pole == nullptr) {
        throw "No pole set";
    }

    // Clear data map.
    for (auto const& item : data_map)
    {
        item.second->clear();
    }
}

void TableRLAgent::end_episode() {
    // Once the episode has ended there will still be values left in the state history
    // that need to be processed to update the value table.

    // Max possible value with current state.
    double max_value = get_max_value(pole->theta, pole->theta_dot);

    // The items in the back are the oldest ones, so will will start processing from the back.
    // We will do this by reducing the bootstrapping n until it's one.
    // The last one has already been processed in the normal iteration step.
    // So we will start with 'n_bootstrapping - 1'
    for (int n = n_bootstrapping - 1; n >= 0; n--){
        // If we have run less than n_bootstrapping iterations, the last items the the list
        // will be empty, ignore these.
        if (past_theta_idx[n - 1] != -1) {
            // Calculate cumulative discounted reward.
            double g = 0.0;
            for (int i = 0; i < n; i++) {
                g += rewards[i] * pow(gamma, n - 1 - i);
            }

            // Add term with value at current state.
            g += pow(gamma, n) * max_value;

            // Update value
            int theta_idx = past_theta_idx[n - 1];
            int theta_dot_idx = past_theta_dot_idx[n - 1];
            int torque_idx = past_torque_idx[n - 1];
            q_table[theta_idx][theta_dot_idx][torque_idx] += alpha * (g - q_table[theta_idx][theta_dot_idx][torque_idx]);
        }
    }
}

std::map<std::string, std::vector<double>*>* TableRLAgent::get_data() {
    auto data_map_copy = new std::map<std::string, std::vector<double>*>();

    for (auto const& item : data_map) {
        auto new_data = new std::vector<double>(*(item.second));
        (*data_map_copy)[item.first] = new_data;

    }

    return data_map_copy;
}

std::map<std::string, double> TableRLAgent::get_scalar_data() {
    std::map<std::string, double> scalar_map;
    double temp = 0;
    for (auto& reward : (*data_map["reward"])) {temp += reward;}
    scalar_map["reward_sum"] = temp;

    temp = 0;
    for (auto& learn_rate : (*data_map["learn_rate"])) {temp += abs(learn_rate);}
    scalar_map["learn_rate_sum"] = temp;
    return scalar_map;
}

void TableRLAgent::run_episode(long max_steps) {
    begin_episode();
    for (;max_steps--;){
        if (run_step())
            break;
    }
    end_episode();
}

