//
// Created by elvieto on 14-2-19.
//

#include "tile_coding_agent.h"
#include "rand.h"


#define SATURATE(min, max, variable) if (variable < min) {variable = min;} else if (variable > max) {variable = max;}
#define GET_IDX(min, max, n, value) (int)(std::round(value / ((max - min) / (n - 1))) - (min / ((max - min) / (n - 1))))

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
        double alpha,
        double vc_min_theta,
        double vc_max_theta,
        int vc_n_theta,
        double vc_min_theta_dot,
        double vc_max_theta_dot,
        int vc_n_theta_dot
        ) :
            tiles(center.data(), tile_size.data(), tilings, default_weight, random_offsets),
            epsilon(epsilon),
            gamma(gamma),
            alpha(alpha),
            n_actions(n_actions),
            tilings(tilings),
            vc_min_theta(vc_min_theta),
            vc_max_theta(vc_max_theta),
            vc_n_theta(vc_n_theta),
            vc_min_theta_dot(vc_min_theta_dot),
            vc_max_theta_dot(vc_max_theta_dot),
            vc_n_theta_dot(vc_n_theta_dot) {

    // Create list of actions.
    actions = new double[n_actions];
    for (int i = 0; i < n_actions; i++) {
        actions[i] = GET_VALUE(min_action, max_action, n_actions, i);
    }

    // Initialize visit count. Set all values to 0.
    visit_count = new uint64_t*[vc_n_theta];
    for (int i = 0; i < vc_n_theta; i++) {
        visit_count[i] = new uint64_t[vc_n_theta_dot]();
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

    { // Update visit count.
        double theta = pole->theta;
        double theta_dot = pole->theta_dot;
        SATURATE(vc_min_theta, vc_max_theta, theta);
        SATURATE(vc_min_theta_dot, vc_max_theta_dot, theta_dot);
        visit_count[GET_IDX(vc_min_theta, vc_max_theta, vc_n_theta, theta)]
                   [GET_IDX(vc_min_theta_dot, vc_max_theta_dot, vc_n_theta_dot, theta_dot)] += 1;
    }

    // Choose action
    // Holds the value and tile keys at state_0.
    ValueTileKeys *value_keys_0 = nullptr;
    double action;
    if (frand(0, 1) > epsilon) {
        // Choose random action.
        action = actions[irand(0, n_actions)];
        double x[3] = {pole->theta, pole->theta_dot, action};
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
    ValueTileKeys *value_keys_1 = nullptr;
    double _; // I don't care about the action here.
    greedy_action(pole->theta, pole->theta_dot, _, value_keys_1);

    // Calculate the amount with which we need to update the tile weights with.
    double update_value = alpha * (reward + (gamma * value_keys_1->value) - value_keys_0->value);
    tiles.update_weights(update_value, value_keys_0);

    // Log results
    data_map["time"]->push_back(pole->time);
    data_map["theta"]->push_back(pole->theta);
    data_map["theta_dot"]->push_back(pole->theta_dot);
    data_map["error"]->push_back(-pole->theta);
    data_map["torque"]->push_back(action);
    data_map["reward"]->push_back(reward);
    data_map["learn_rate"]->push_back(update_value);

    delete value_keys_0;
    delete value_keys_1;

    return is_terminal;
}

void TileCodingAgent::greedy_action(double theta, double theta_dot,
                                    double &best_action, ValueTileKeys* &best_value) {
    // Randomly choose an action as the best one. this will ensure that
    // it will pick one of them if they are all the same.
    best_action = actions[irand(0, n_actions)];
    double x[3] = {theta, theta_dot, best_action};
    best_value = tiles.get_value_and_tile_keys(x);

    ValueTileKeys *value_keys;

    // Loop through all actions.
    for (int i = 0; i < n_actions; i++) {
        // Get value for action
        x[2] = actions[i];
        value_keys = tiles.get_value_and_tile_keys(x);

        // If it's better then update the state value.
        if (value_keys->value > best_value->value) {
            delete best_value;
            best_value = value_keys;
            best_action = x[2];
        } else {
            delete value_keys;
        }
    }
}

double** TileCodingAgent::greedy_action_map() {
    // Initialize map first dimension (theta).
    auto map = new double*[vc_n_theta];
    // We'll not use this, but we need to pass it to the greedy_action function.
    ValueTileKeys* best_value_keys = nullptr;

    // The greedy (best) action will be stored here.
    double best_action = 0;

    // Loop through theta.
    for (int i = 0; i < vc_n_theta; i++) {
        double theta = GET_VALUE(vc_min_theta, vc_max_theta, vc_n_theta, i);

        // Initialize map second dimension (theta_dot) and loop through it.
        map[i] = new double[vc_n_theta_dot]();
        for (int j = 0; j < vc_n_theta_dot; j++) {
            double theta_dot = GET_VALUE(vc_min_theta_dot, vc_max_theta_dot, vc_n_theta_dot, j);

            // Get greedy action
            // Delete the value keys object since we don't need it.
            // Store greedy action in map.
            greedy_action(theta, theta_dot, best_action, best_value_keys);
            delete best_value_keys;
            map[i][j] = best_action;
        }
    }
    return map;
}

double **TileCodingAgent::update_count_map() {
    // Initialize map first dimension (theta).
    auto map = new double*[vc_n_theta];
    double x[3];

    // Loop through theta.
    for (int i = 0; i < vc_n_theta; i++) {
        x[0] = GET_VALUE(vc_min_theta, vc_max_theta, vc_n_theta, i); // Theta

        // Initialize map second dimension (theta_dot) and loop through it.
        map[i] = new double[vc_n_theta_dot]();
        for (int j = 0; j < vc_n_theta_dot; j++) {
            x[1] = GET_VALUE(vc_min_theta_dot, vc_max_theta_dot, vc_n_theta_dot, j); // Theta_dot

            // Set update count for state to 0.
            map[i][j] = 0;

            // Loop through all actions.
            for (int k = 0; k < n_actions; k++) {
                x[2] = actions[k];

                // Loop through all tilings.
                for (unsigned short l = 0; l < tilings; l++) {
                    // Get tile info and add it's visit count to the update count map.
                    TileInfo *tile_info = tiles.get_tile_info(tiles.get_tile_key(x, l));
                    map[i][j] += tile_info->visit_count;
                }
            }
        }
    }

    return map;
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
        if (run_step()) {
            break;
        }
    }
    end_episode();
}


