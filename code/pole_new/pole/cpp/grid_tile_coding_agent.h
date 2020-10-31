//
// Created by adewindt on 6/24/20.
//

#ifndef POLE_NEW_GRID_TILE_CODING_AGENT_H
#define POLE_NEW_GRID_TILE_CODING_AGENT_H

#include "rand.h"
#include "grid_tile_coding.h"
#include "agent_base.h"
#include <array>
#include <cmath>


#define SATURATE(min, max, variable) if (variable < min) {variable = min;} else if (variable > max) {variable = max;}
#define GET_IDX(min, max, n, value) (int)(std::round(value / ((max - min) / (n - 1))) - (min / ((max - min) / (n - 1))))

#define GET_VALUE(min, max, n, idx) idx * ((max - min) / (n - 1)) + min


namespace pole {
    class TileCodingAgent : public AgentBase {
    public:
        Environment& env;
        f64 reward_sum;
        GridTileCoding<3, 1> tiles;

        double epsilon;
        double gamma;
        double alpha;
        double *actions;
        int n_actions;
        int tilings;

        uint64_t **visit_count;
        double vc_min_theta;
        double vc_max_theta;
        int vc_n_theta;
        double vc_min_theta_dot;
        double vc_max_theta_dot;
        int vc_n_theta_dot;

        typedef GridTileCoding<3, 1>::ValueTileKeys ValueTileKeys;
        typedef GridTileCoding<3, 1>::TileInfo TileInfo;

        struct LogEntry {
            // Constructor necessary for std::vector.emplace_back()
            inline LogEntry( f64& time, f64& theta, f64& theta_dot, f64& action, f64& delta_v, f64& reward) :
                    time(time), theta(theta), theta_dot(theta_dot), action(action), delta_v(delta_v), reward(reward) {}
            f64 time;
            f64 theta;
            f64 theta_dot;
            f64 action;
            f64 delta_v;
            f64 reward;
        };
        std::vector<LogEntry> log;

    public:
        inline TileCodingAgent(
                Environment& env,
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
                    env(env),
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
                    vc_n_theta_dot(vc_n_theta_dot),
                    reward_sum(0) {

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
        }

        inline void greedy_action(double theta, double theta_dot,
                      double &best_action, ValueTileKeys* &best_value) {
            // Randomly choose an action as the best one. this will ensure that
            // it will pick one of them if they are all the same.
            best_action = actions[usize_rand(0, n_actions)];
            double x[3] = {theta, theta_dot, best_action};
            best_value = tiles.get_value_and_tile_keys(x);

            ValueTileKeys *value_keys;

            // Loop through all actions.
            for (int i = 0; i < n_actions; i++) {
                // Get value for action
                x[2] = actions[i];
                value_keys = tiles.get_value_and_tile_keys(x);

                // If it's better then update the state value.
                if (*value_keys->values > *best_value->values) {
                    delete best_value;
                    best_value = value_keys;
                    best_action = x[2];
                } else {
                    delete value_keys;
                }
            }
        }

        bool step() final {
            { // Update visit count.
                double theta = env.theta;
                double theta_dot = env.theta_dot;
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
                action = actions[usize_rand(0, n_actions)];
                double x[3] = {env.theta, env.theta_dot, action};
                value_keys_0 = tiles.get_value_and_tile_keys(x);
            } else {
                // Choose greedy action.
                greedy_action(env.theta, env.theta_dot, action, value_keys_0);
            }

            // Perform action
            auto [reward, is_terminal] = env.step(action);
            reward_sum += reward;

            // Learn
            // We are gonna do Q-learning here.
            // So find the greedy action of the state we are currently in.
            ValueTileKeys *value_keys_1 = nullptr;
            double _; // I don't care about the action here.
            greedy_action(env.theta, env.theta_dot, _, value_keys_1);

            // Calculate the amount with which we need to update the tile weights with.
            double update_value[1] = {alpha * (reward + (gamma * *value_keys_1->values) - *value_keys_0->values)};
            tiles.update_weights(update_value, value_keys_0);

            // Log
            log.emplace_back(
                    env.time,
                    env.theta,
                    env.theta_dot,
                    action,
                    update_value[0],
                    reward);

            return is_terminal;
        }


        i64 run_episode(i64 max_steps) final {
            // Clear log from previous episode.
            log.clear();
            log.reserve(max_steps);
            reward_sum = 0;

            for (i64 i = 0; i < max_steps; i++) {
                if (step()) {
                    return i;
                }
            }

            return max_steps;
        }

        std::map<std::string, std::vector<f64>> get_data() final {
            // Create datamap
            std::map<std::string, std::vector<f64>> data_map;

            // Create vectors.
            data_map.try_emplace("time");
            data_map.try_emplace("theta");
            data_map.try_emplace("theta_dot");
            data_map.try_emplace("action");
            data_map.try_emplace("reward");

            // Allocate the required memory in each vector.
            for (auto& [key, vector] : data_map ) {
                vector.reserve(log.size());
            }

            // Get a reference to each one of the vectors.
            std::vector<f64>& vector_time = data_map["time"];
            std::vector<f64>& vector_theta = data_map["theta"];
            std::vector<f64>& vector_theta_dot = data_map["theta_dot"];
            std::vector<f64>& vector_action = data_map["action"];
            std::vector<f64>& vector_delta_v = data_map["delta_v"];
            std::vector<f64>& vector_reward = data_map["reward"];

            // Copy data to the vectors.
            for (auto& entry : log) {
                vector_time.push_back(entry.time);
                vector_theta.push_back(entry.theta);
                vector_theta_dot.push_back(entry.theta_dot);
                vector_action.push_back(entry.action);
                vector_delta_v.push_back(entry.delta_v);
                vector_reward.push_back(entry.reward);
            }

            // Return data map. This should result in a move operation.
            return data_map;
        }

        std::map<std::string, f64> get_scalar_data() final {
            std::map<std::string, f64> scalar_data;
            scalar_data["reward_sum"] = reward_sum;
            return scalar_data;
        }

        f64 get_reward_sum() final {
            return reward_sum;
        }

        std::vector<f64> get_greedy_action_table() {
            std::vector<f64> greedy_action_table;
            greedy_action_table.reserve(vc_n_theta * vc_n_theta_dot);

            for (usize idx_theta = 0; idx_theta < vc_n_theta; idx_theta++) {
                double theta = GET_VALUE(vc_min_theta, vc_max_theta, vc_n_theta, idx_theta);
                for (usize idx_theta_dot = 0; idx_theta_dot < vc_n_theta_dot; idx_theta_dot++) {
                    double theta_dot = GET_VALUE(vc_min_theta_dot, vc_max_theta_dot, vc_n_theta_dot, idx_theta_dot);
                    double action;
                    ValueTileKeys *value_keys_1 = nullptr;
                    greedy_action(theta, theta_dot, action, value_keys_1);
                    greedy_action_table.push_back(action);
                }
            }

            return greedy_action_table;
        }

        std::vector<f64> get_counts() {
            // Initialize output table.
            std::vector<f64> count_table;
            count_table.reserve(vc_n_theta * vc_n_theta_dot);

            double x[3] = {0.};

            for (usize idx_theta = 0; idx_theta < vc_n_theta; idx_theta++) {
                x[0] = GET_VALUE(vc_min_theta, vc_max_theta, vc_n_theta, idx_theta); // Theta

                for (int idx_theta_dot = 0; idx_theta_dot < vc_n_theta_dot; idx_theta_dot++) {
                    x[1] = GET_VALUE(vc_min_theta_dot, vc_max_theta_dot, vc_n_theta_dot, idx_theta_dot); // Theta_dot
                    double count = 0;

                    for (int idx_action = 0; idx_action < n_actions; idx_action++) {
                        x[2] = actions[idx_action];

                        for (unsigned short l = 0; l < tilings; l++) {
                            // Get tile info and add it's visit count to the update count map.
                            TileInfo *tile_info = tiles.get_tile_info(tiles.get_tile_key(x, l));
                            count += tile_info->visit_count;
                        }
                    }
                    count_table.push_back(count);
                }
            }

            return count_table;
        }

    };
}

#endif //POLE_NEW_GRID_TILE_CODING_AGENT_H
