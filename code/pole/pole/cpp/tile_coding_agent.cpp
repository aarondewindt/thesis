//
// Created by elvieto on 14-2-19.
//

#include "tile_coding_agent.h"
#include "rand.h"



#define GET_VALUE(min, max, n, idx) idx * ((max - min) / (n - 1)) + min

TileCodingAgent::TileCodingAgent(
        std::vector<double> center,
        std::vector<double> tile_size,
        int tilings,
        double default_weight,
        bool random_offsets,
        double min_action,
        double max_action,
        int n_actions,
        double epsilon,
        double gamma,
        double alpha) :
            tiles(center.data(), tile_size.data(), tilings, default_weight, random_offsets),
            epsilon(epsilon),
            gamma(gamma),
            alpha(alpha),
            n_actions(n_actions),
            tilings(tilings){

    // Create list of actions.
    actions = new double[n_actions];
    for (int i = 0; i < n_actions; i++) {
        actions[i] = GET_VALUE(min_action, max_action, n_actions, i);
    }

    // Initialize the data map with fresh vectors.
    data_map["time"] = new std::vector<double>();
    data_map["theta"] = new std::vector<double>();
    data_map["theta_dot"] = new std::vector<double>();
    data_map["error"] = new std::vector<double>();
    data_map["torque"] = new std::vector<double>();
    data_map["reward"] = new std::vector<double>();
    data_map["learn_rate"] = new std::vector<double>();
}


bool TileCodingAgent::run_step() {
    if (pole == nullptr) {
        throw "No pole set";
    }

    // Choose action
    // Holds the value and tile keys at state_0.
    ValueTileKeys value_keys_0;
    double action;
    if (frand(0, 1) > epsilon) {
        // Choose random action.
        action = actions[irand(0, n_actions)];
        XArray x = {pole->theta, pole->theta_dot, action};
        value_keys_0 = tiles.get_value_and_tile_keys(x);
    } else {
        // Choose greedy action.
        greedy_action(pole->theta, pole->theta_dot, action, value_keys_0);
    }

    // Perform action
    bool is_terminal;
    double reward;
    pole->act(action, reward, is_terminal);

    // Learn
    // We are gonna do Q-learning here.
    // So find the greedy action of the state we are currently in.
    ValueTileKeys value_keys_1;
    double _; // I don't case about the action.
    greedy_action(pole->theta, pole->theta_dot, _, value_keys_1);

    // Calculate the amount with which we need to update the tile weights with.
    double update_value = alpha * (reward + gamma * value_keys_1.value - value_keys_0.value);
    tiles.update_weights(update_value, value_keys_0);

    // Log results
    data_map["time"]->push_back(pole->time);
    data_map["theta"]->push_back(pole->theta);
    data_map["theta_dot"]->push_back(pole->theta_dot);
    data_map["error"]->push_back(-pole->theta);
    data_map["torque"]->push_back(action);
    data_map["reward"]->push_back(reward);
    data_map["learn_rate"]->push_back(update_value);

    return is_terminal;
}

void TileCodingAgent::greedy_action(double theta, double theta_dot,
                                    double &best_action, ValueTileKeys &best_value) {
    // Randomly choose an action as the best one. this will ensure that
    // it will pick one of them if they are all the same.
    best_action = actions[irand(0, n_actions)];
    XArray x = {theta, theta_dot, best_action};
    best_value = tiles.get_value_and_tile_keys(x);

    ValueTileKeys value_keys;

    // Loop through all actions.
    for (int i = 0; i < n_actions; i++) {
        // Get value for action
        x(0, 2) = actions[i];
        value_keys = tiles.get_value_and_tile_keys(x);

        // If it's better then update the state value.
        if (value_keys.value > best_value.value) {
            best_value.value = value_keys.value;
            for (int j = 0; j < tilings; j++) {
                best_value.tile_keys[j] = value_keys.tile_keys[j];
            }
        }
    }
}


void TileCodingAgent::begin_episode() {
    if (pole == nullptr) {
        throw "No pole set";
    }

    // Clear data map.
    for (auto const& item : data_map)
    {
        item.second->clear();
    }
}

std::map<std::string, std::vector<double>*>* TileCodingAgent::get_data() {
    auto data_map_copy = new std::map<std::string, std::vector<double>*>();

    for (auto const& item : data_map) {
        auto new_data = new std::vector<double>(*(item.second));
        (*data_map_copy)[item.first] = new_data;

    }

    return data_map_copy;
}

std::map<std::string, double> TileCodingAgent::get_scalar_data() {
    std::map<std::string, double> scalar_map;
    double temp = 0;
    for (auto& reward : (*data_map["reward"])) {temp += reward;}
    scalar_map["reward_sum"] = temp;

    temp = 0;
    for (auto& learn_rate : (*data_map["learn_rate"])) {temp += abs(learn_rate);}
    scalar_map["learn_rate_sum"] = temp;
    return scalar_map;
}

void TileCodingAgent::run_episode(long max_steps) {
    begin_episode();
    for (;max_steps--;){
        if (run_step())
            break;
    }
    end_episode();
}