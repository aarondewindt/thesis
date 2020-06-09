//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_PID_AGENT_H
#define POLE_NEW_PID_AGENT_H

#include "agent_base.h"
#include <array>
#include <cmath>

namespace pole {
    class PIDAgent : public AgentBase {
        Environment& env;
        f64 k_p;
        f64 k_i;
        f64 k_d;
        f64 previous_error;
        f64 integral;
        f64 reward_sum;

        struct LogEntry {
            // Constructor necessary for std::vector.emplace_back()
            inline LogEntry( f64& time, f64& theta, f64& theta_dot, f64& error,
                    f64& integral, f64& derivative, f64& torque, f64& reward) :
                    time(time), theta(theta), theta_dot(theta_dot), error(error), integral(integral),
                    derivative(derivative), torque(torque), reward(reward) {}
            f64 time;
            f64 theta;
            f64 theta_dot;
            f64 error;
            f64 integral;
            f64 derivative;
            f64 torque;
            f64 reward;
        };
        std::vector<LogEntry> log;

    public:
        inline PIDAgent(Environment& env, f64 k_p, f64 k_i, f64 k_d) :
            env(env),
            k_p(k_p),
            k_i(k_i),
            k_d(k_d),
            integral(std::nan("")),
            previous_error(std::nan("")),
            reward_sum(0)
            {}

        bool step() final {
            // Calculate error.
            f64 error = -env.theta;
            f64 derivative;
            f64 torque;

            // Calculate integral and derivative.
            if (std::isnan(previous_error)) {
                integral = 0.;
                derivative = 0.;
            } else {
                integral += (previous_error + error) / 2 * env.dt;
                derivative = (error - previous_error) / env.dt;
            }

            previous_error = error;

            // Calculate torque and run single step.
            torque = k_p * error + k_i * integral + k_d * derivative;
            auto [reward, is_terminal] = env.step(torque);

            reward_sum += reward;

            // Log
            log.emplace_back(
                    env.time,
                    env.theta,
                    env.theta_dot,
                    error,
                    integral,
                    derivative,
                    torque,
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
            data_map.try_emplace("error");
            data_map.try_emplace("integral");
            data_map.try_emplace("derivative");
            data_map.try_emplace("torque");
            data_map.try_emplace("reward");

            // Allocate the required memory in each vector.
            for (auto& [key, vector] : data_map ) {
                vector.reserve(log.size());
            }

            // Get a reference to each one of the vectors.
            std::vector<f64>& vector_time = data_map["time"];
            std::vector<f64>& vector_theta = data_map["theta"];
            std::vector<f64>& vector_theta_dot = data_map["theta_dot"];
            std::vector<f64>& vector_error = data_map["error"];
            std::vector<f64>& vector_integral = data_map["integral"];
            std::vector<f64>& vector_derivative = data_map["derivative"];
            std::vector<f64>& vector_torque = data_map["torque"];
            std::vector<f64>& vector_reward = data_map["reward"];

            // Copy data to the vectors.
            for (auto& entry : log) {
                vector_time.push_back(entry.time);
                vector_theta.push_back(entry.theta);
                vector_theta_dot.push_back(entry.theta_dot);
                vector_error.push_back(entry.error);
                vector_integral.push_back(entry.integral);
                vector_derivative.push_back(entry.derivative);
                vector_torque.push_back(entry.torque);
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
    };
}

#endif //POLE_NEW_PID_AGENT_H
