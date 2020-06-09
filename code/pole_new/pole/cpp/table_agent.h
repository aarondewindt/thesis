//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_TABLE_AGENT_H
#define POLE_NEW_TABLE_AGENT_H

#include "agent_base.h"
#include "tcb/span.hpp"
#include "rand.h"
#include "table_3d.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iterator>

namespace pole {
    class TableAgent : public AgentBase {
        Environment& env;
        f64 reward_sum;

        const f64& min_action;
        const f64& max_action;
        const usize& n_action;
        const f64& delta_action;

        const f64& min_theta;
        const f64& max_theta;
        const usize& n_theta;
        const f64& min_theta_dot;
        const f64& max_theta_dot;
        const usize& n_theta_dot;

        struct TableItem {
            f64 value;
            u64 count;
            inline friend bool operator<(const TableItem& l, const TableItem& r) {
                return l.value < r.value;
            }
        };

        Table3D<TableItem, f64> table;

        struct LogEntry {
            // Constructor necessary for std::vector.emplace_back()
            inline LogEntry(f64& time, f64& theta, f64& theta_dot, f64& action, f64& reward, f64& delta_v) :
                    time(time), theta(theta), theta_dot(theta_dot),
                    action(action), reward(reward), delta_v(delta_v) {}
            f64 time;
            f64 theta;
            f64 theta_dot;
            f64 action;
            f64 reward;
            f64 delta_v;
        };

        std::vector<LogEntry> log;

    public:
        f64 epsilon;
        f64 gamma;
        f64 alpha;

        inline TableAgent(Environment& env,
                          f64 min_action,
                          f64 max_action,
                          f64 min_theta,
                          f64 max_theta,
                          f64 min_theta_dot,
                          f64 max_theta_dot,
                          usize n_action,
                          usize n_theta,
                          usize n_theta_dot,
                          f64 epsilon,
                          f64 gamma,
                          f64 alpha) :
                env(env),
                reward_sum(0),
                epsilon(epsilon),
                gamma(gamma),
                alpha(alpha),
                table(
                    min_action, max_action, n_action,
                    min_theta, max_theta, n_theta,
                    min_theta_dot, max_theta_dot, n_theta_dot,
                    TableItem({-1, 0lu})),
                min_action(table.min0),
                max_action(table.max0),
                delta_action(table.delta0),
                min_theta(table.min1),
                max_theta(table.max1),
                min_theta_dot(table.min2),
                max_theta_dot(table.max2),
                n_action(table.n0),
                n_theta(table.n1),
                n_theta_dot(table.n2) {}


        inline std::pair<f64, TableItem*> greedy_action(f64 theta, f64 theta_dot) {
            auto values = table(theta, theta_dot);
            auto max_element = std::max_element(values.begin(), values.end());
            usize max_element_idx = std::distance(values.begin(), max_element);
            return std::make_pair(
                    min_action + delta_action * (max_element_idx + 0.5),
                    values.data() + max_element_idx);
        }

        inline std::pair<f64, TableItem*> greedy_action_idx(usize theta, usize theta_dot) {
            auto values = table.get(theta, theta_dot);
            auto max_element = std::max_element(values.begin(), values.end());
            usize max_element_idx = std::distance(values.begin(), max_element);
            return std::make_pair(
                    min_action + delta_action * (max_element_idx + 0.5),
                    values.data() + max_element_idx);
        }

        inline std::pair<f64, TableItem*> evaluate_policy(f64 theta, f64 theta_dot) {
            if (frand(0., 1.) < epsilon) {
                usize action_idx = usize_rand(0, n_action);
                f64 action = min_action + delta_action * (action_idx + 0.5);
                return std::make_pair(
                    action,
                    &table(action, theta, theta_dot));
            } else {
                return greedy_action(theta, theta_dot);
            }
        }

        bool step() final {
            auto [action_t0, item_0] = evaluate_policy(env.theta, env.theta_dot);
            f64& q_t0 = item_0->value;
            item_0->count++;

            auto [reward_t1, is_terminal] = env.step(action_t0);

            auto [action_t1, item_1] = evaluate_policy(env.theta, env.theta_dot);
            f64& q_t1 = item_1->value;

            f64 delta_v = alpha * (reward_t1 + (std::isnan(q_t1) ? 0 : gamma * q_t1)   - q_t0);

            if (std::isnan(q_t1)) {
                q_t0 = delta_v;
            } else {
                q_t0 += delta_v;
            }

            reward_sum += reward_t1;
            log.emplace_back(
                    env.time,
                    env.theta,
                    env.theta_dot,
                    action_t0,
                    reward_t1,
                    delta_v);

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
            data_map.try_emplace("delta_v");

            // Allocate the required memory in each vector.
            for (auto& [key, vector] : data_map ) {
                vector.reserve(log.size());
            }

            // Get a reference to each one of the vectors.
            std::vector<f64>& vector_time = data_map["time"];
            std::vector<f64>& vector_theta = data_map["theta"];
            std::vector<f64>& vector_theta_dot = data_map["theta_dot"];
            std::vector<f64>& vector_action = data_map["action"];
            std::vector<f64>& vector_reward = data_map["reward"];
            std::vector<f64>& vector_delta_v = data_map["delta_v"];

            // Copy data to the vectors.
            for (auto& entry : log) {
                vector_time.push_back(entry.time);
                vector_theta.push_back(entry.theta);
                vector_theta_dot.push_back(entry.theta_dot);
                vector_action.push_back(entry.action);
                vector_reward.push_back(entry.reward);
                vector_delta_v.push_back(entry.delta_v);
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

        std::vector<f64> get_values() {
            std::vector<f64> q_table;
            q_table.reserve(table.n0 * table.n1 * table.n2);

            for (auto& item : table.get_span()) {
                q_table.push_back(item.value);
            }

            return q_table;
        }

        std::vector<f64> get_counts() {
            std::vector<f64> count_table;
            count_table.reserve(table.n0 * table.n1 * table.n2);

            for (auto& item : table.get_span()) {
                count_table.push_back(item.count);
            }

            return count_table;
        }

        std::vector<f64> get_greedy_action_table() {
            std::vector<f64> greedy_action_table;
            greedy_action_table.reserve(table.n0 * table.n1 * table.n2);

            for (usize idx_theta = 0; idx_theta < n_theta; idx_theta++) {
                for (usize idx_theta_dot = 0; idx_theta_dot < n_theta_dot; idx_theta_dot++) {
                    auto [action, _] = greedy_action_idx(idx_theta, idx_theta_dot);
                    greedy_action_table.push_back(action);
                }
            }

            return greedy_action_table;
        }


    };
}

#endif //POLE_NEW_TABLE_AGENT_H
