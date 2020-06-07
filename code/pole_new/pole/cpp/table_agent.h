//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_TABLE_AGENT_H
#define POLE_NEW_TABLE_AGENT_H

#include "agent_base.h"
#include <array>
#include <cmath>
#include <stdexcept>


namespace pole {
    class TableAgent : public AgentBase {
        Environment& env;
        f64 reward_sum;

        f64 min_theta;
        f64 max_theta;
        f64 min_theta_dot;
        f64 max_theta_dot;
        f64 min_torque;
        f64 max_torque;
        usize q_table_size_theta;
        usize q_table_size_theta_dot;
        usize q_table_size_torque;
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

        inline f64& q(usize theta_idx, usize theta_dot_idx, usize torque_idx) {
            static usize c1 = q_table_size_theta;
            static usize c2 = q_table_size_theta + q_table_size_theta_dot;

            if (theta_idx > q_table_size_theta) throw std::domain_error("Theta_idx out of bounds");
            if (theta_dot_idx > q_table_size_theta_dot) throw std::domain_error("Theta_idx out of bounds");
            if (torque_idx > q_table_size_torque) throw std::domain_error("Theta_idx out of bounds");

            return q_table[theta_idx + c1 * theta_dot_idx + c2 * torque_idx];
        }

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
                q_table(q_table_size_theta * q_table_size_theta_dot * q_table_size_torque, 0)

        {}

        bool step() final {
            f64 action = 0;
            f64 delta_v = 0;

            auto [reward, is_terminal] = env.step(action);

            // Log
            reward_sum += reward;
            log.emplace_back(
                    env.time,
                    env.theta,
                    env.theta_dot,
                    action,
                    reward,
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
