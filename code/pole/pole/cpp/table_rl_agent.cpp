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
        int n_r) :
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
    n_r(n_r),
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

    // Initialize data map.
     data_map["time"] = new std::vector<double>();
     data_map["theta"] = new std::vector<double>();
     data_map["theta_dot"] = new std::vector<double>();
     data_map["error"] = new std::vector<double>();
     data_map["torque"] = new std::vector<double>();
     data_map["reward"] = new std::vector<double>();
     data_map["g"] = new std::vector<double>();

    // Initialize past rewards and states.
    rewards = new double[n_r]();
    past_theta = new double[n_r];
    past_theta_dot = new double[n_r];
    past_torque = new double[n_r];
    for (int i = 0; i < n_r; i++) {
        past_theta[i] = nan("");
    }

    int _, __, tidx;
    printf("0 %f\n", GET_VALUE(min_torque, max_torque, n_torque, 0));
    printf("%d %f\n", n_torque - 1, GET_VALUE(min_torque, max_torque, n_torque, (n_torque - 1)));
}

TableRLAgent::~TableRLAgent(){
    for (int i = 0; i < n_theta; i++) {
        for (int j = 0; j < n_torque; j++) {
            delete q_table[i][j];
        }
        delete q_table[i];
    }
    delete q_table;
    delete rewards;
}

bool TableRLAgent::run_step() {
//    std::printf("Run step\n");
    if (pole == nullptr) {
        throw "No pole set";
    }

    // Shift past rewards list to make space for the new reward.
    for (int i = (n_r - 1); i > 0; i--) {
        rewards[i] = rewards[i - 1];
        past_theta[i] = past_theta[i - 1];
        past_theta_dot[i] = past_theta_dot[i - 1];
        past_torque[i] = past_torque[i - 1];
    }

    // Choose an action. Torque to apply on pole.
    double action;
    if (frand(0, 1) > eps) {
        action = GET_VALUE(min_torque, max_torque, n_torque, irand(0, n_torque - 1));
    } else {
        action = choose_ideal_torque(pole->theta, pole->theta_dot);
    }

//    std::printf("Action choosen\n");

    // Get current value.
    int theta_idx, theta_dot_idx, torque_idx;
    q_table_prelookup(pole->theta, pole->theta_dot, action, theta_idx, theta_dot_idx, torque_idx);
    double value_0 = q_table[theta_idx][theta_dot_idx][torque_idx];

    past_theta[0] = pole->theta;
    past_theta_dot[0] = pole->theta_dot;
    past_torque[0] = action;

    // Perform action
    bool is_terminal;
    pole->act(action, *rewards, is_terminal);

//    printf("%f\n", *past_rewards);

    double g = nan("");

    // If the last value in the past theta array is not a nan we have filled the array
    // and can start updating the values for the state-action pairs n_r iterations ago.
    if (!isnan(past_theta[n_r - 1])) {
        g = 0.0;
        // Calculate new value
//        for (int i = n_r - 1; i >= 0; i--) {
//           printf("%f ", past_rewards[i] * pow(gamma, n_r - i - 1));
//            g += past_rewards[i] * pow(gamma, n_r - i - 1);
//        }

        g = *rewards + gamma * get_max_value(pole->theta, pole->theta_dot);

        // Update value
        q_table[theta_idx][theta_dot_idx][torque_idx] += alpha * (g - value_0);
    }

//    printf("g=%f\nt=%d\n\n", g, is_terminal);

    // Log results.
    data_map["time"]->push_back(pole->time);
    data_map["theta"]->push_back(pole->theta);
    data_map["theta_dot"]->push_back(pole->theta_dot);
    data_map["error"]->push_back(-pole->theta);
    data_map["torque"]->push_back(action);
    data_map["reward"]->push_back(*rewards);
    data_map["g"]->push_back(g);

//    std::printf("Logged\n");

    return is_terminal;

}

double TableRLAgent::choose_ideal_torque(double theta, double theta_dot) {
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
    if (theta_dot < min_theta_dot) theta_dot = min_theta;
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
    for (auto& reward : (*data_map["reward"])) {
        temp += reward;
    }
    scalar_map["reward_sum"] = temp;
    return scalar_map;
}

void TableRLAgent::run_episode(long max_steps) {
    begin_episode();
    for (;max_steps--;){
        if (run_step())
            break;
    }
}

