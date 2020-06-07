//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_TABLE_AGENT_H
#define POLE_NEW_TABLE_AGENT_H

#include "agent_base.h"
#include "tcb/span.hpp"
#include "rand.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iterator>

namespace pole {
    class TableAgent : public AgentBase {
        Environment& env;
        f64 reward_sum;

        const f64 min_theta;
        const f64 max_theta;
        const f64 min_theta_dot;
        const f64 max_theta_dot;
        const f64 min_torque;
        const f64 max_torque;
        const usize q_table_size_theta;
        const usize q_table_size_theta_dot;
        const usize q_table_size_torque;
        const f64 delta_theta;
        const f64 delta_theta_dot;
        const f64 delta_action;

        f64 epsilon;
        f64 gamma;
        f64 alpha;
        std::vector<f64> q_table;

        struct LogEntry {
            // Constructor necessary for std::vector.emplace_back()
            inline LogEntry(f64& time, f64& theta, f64& theta_dot, f64& action, f64& reward, f64& delta_v) :
                    time(time), theta(theta), theta_dot(theta_dot), action(action), reward(reward), delta_v(delta_v) {}
            f64 time;
            f64 theta;
            f64 theta_dot;
            f64 action;
            f64 reward;
            f64 delta_v;
        };
        std::vector<LogEntry> log;

    public:
        inline TableAgent(Environment& env,
                          f64 min_theta,
                          f64 max_theta,
                          f64 min_theta_dot,
                          f64 max_theta_dot,
                          f64 min_torque,
                          f64 max_torque,
                          usize q_table_size_theta,
                          usize q_table_size_theta_dot,
                          usize q_table_size_torque,
                          f64 epsilon,
                          f64 gamma,
                          f64 alpha) :
                env(env),
                reward_sum(0),
                min_theta(min_theta),
                max_theta(max_theta),
                min_theta_dot(min_theta_dot),
                max_theta_dot(max_theta_dot),
                min_torque(min_torque),
                max_torque(max_torque),
                q_table_size_theta(q_table_size_theta),
                q_table_size_theta_dot(q_table_size_theta_dot),
                q_table_size_torque(q_table_size_torque),
                epsilon(epsilon),
                gamma(gamma),
                alpha(alpha),
                q_table(q_table_size_theta * q_table_size_theta_dot * q_table_size_torque, 0),
                delta_theta((max_theta - min_theta) / q_table_size_theta),
                delta_theta_dot((max_theta_dot - min_theta_dot) / q_table_size_theta_dot),
                delta_action((max_torque - min_torque) / q_table_size_torque) {}


        inline f64& q_search(f64 action, f64 theta, f64 theta_dot) {
            return q(
                    static_cast<usize>(action - min_torque / delta_action),
                    static_cast<usize>(theta - min_theta / delta_theta),
                    static_cast<usize>(theta_dot - min_theta_dot / delta_theta_dot)
            );
        }

        inline f64& q(usize action_idx, usize theta_idx, usize theta_dot_idx) {
            static usize c1 = q_table_size_torque;
            static usize c2 = q_table_size_torque + q_table_size_theta_dot;

            if (action_idx > q_table_size_torque) throw std::domain_error("Theta_idx out of bounds");
            if (theta_idx > q_table_size_theta) throw std::domain_error("Theta_idx out of bounds");
            if (theta_dot_idx > q_table_size_theta_dot) throw std::domain_error("Theta_idx out of bounds");

            return q_table[action_idx + c1 * theta_dot_idx + c2 * theta_idx];
        }

        inline tcb::span<f64> q_search(f64 theta, f64 theta_dot) {
            return q(
                    static_cast<usize>(theta - min_theta / delta_theta),
                    static_cast<usize>(theta_dot - min_theta_dot / delta_theta_dot)
            );
        }

        inline tcb::span<f64> q(usize theta_idx, usize theta_dot_idx) {
            static usize c1 = q_table_size_torque;
            static usize c2 = q_table_size_torque + q_table_size_theta_dot;
            static tcb::span<f64> q_table_span(q_table);

            if (theta_idx > q_table_size_theta) throw std::domain_error("Theta_idx out of bounds");
            if (theta_dot_idx > q_table_size_theta_dot) throw std::domain_error("Theta_idx out of bounds");

            return q_table_span.subspan(c1 * theta_dot_idx + c2 * theta_idx, q_table_size_torque);
        }

        inline std::pair<f64, f64*> greedy_action(f64 theta, f64 theta_dot) {
            auto values = q_search(theta, theta_dot);
            auto max_element = std::max_element(values.begin(), values.end());
            usize max_element_idx = std::distance(values.begin(), max_element);
            return std::make_pair(
                    min_torque + delta_action * (max_element_idx + 0.5),
                    values.data() + max_element_idx);
        }

        inline std::pair<f64, f64*> evaluate_policy(f64 theta, f64 theta_dot) {
            if (frand(0., 1.) < epsilon) {
                usize action_idx = usize_rand(0, q_table_size_torque);
                f64 action = min_torque + delta_action * (action_idx + 0.5);
                return std::make_pair(
                    action,
                    &q(action, theta, theta_dot));
            } else {
                return greedy_action(theta, theta_dot);
            }
        }

        bool step() final {
            auto [action_t0, q_t0_ptr] = evaluate_policy(env.theta, env.theta_dot);
            f64& q_t0 = *q_t0_ptr;

            auto [reward_t1, is_terminal] = env.step(action_t0);

            auto [action_t1, q_t1_ptr] = evaluate_policy(env.theta, env.theta_dot);
            f64& q_t1 = *q_t1_ptr;

            f64 delta_v = alpha * (reward_t1 + gamma * q_t1 - q_t0);

            q_t0 += delta_v;

            // Log
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


        void run_episode(i64 max_steps) final {
            // Clear log from previous episode.
            log.clear();
            log.reserve(max_steps);
            reward_sum = 0;

            for (i64 i = 0; i < max_steps; i++) {
                if (step()) {
                    return;
                }
            }
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
    };
}

#endif //POLE_NEW_TABLE_AGENT_H
