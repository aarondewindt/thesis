//
// Created by elvieto on 3-4-19.
//

#ifndef F16_ENVIRONMENT_H
#define F16_ENVIRONMENT_H

#include <vector>
#include "msgpack11.hpp"


using namespace msgpack11;

namespace f16 {
    class Environment {
    public:
        float a_matrix[16];
        float b_matrix[4];
        float t_end;
        float dt;
        std::vector<float> q_command;
        int current_iteration;

        static const int n_dynamics_state = 4;
        union {
            float x[n_dynamics_state];
            struct {
                float velocity;
                float alpha;
                float theta;
                float q;
            };
        } dynamics_state;

        static const int n_rl_state = 4;
        union {
            float x[n_rl_state];
            struct {
                float alpha;
                float q;
                float e;
                float k_p;
            };
        } rl_state;

        Environment(std::vector<float> const &a_matrix, std::vector<float> const &b_matrix, float dt, float t_end);

        Environment(std::string byte_string);

        ~Environment() = default;

        /// \brief Resets the environment back to it's initial state.
        void reset(std::vector<float> state_min, std::vector<float> state_max);

        /// \brief Generates a new input signal.
        void generate_q_command(std::vector<float>);

        /// \brief Performs the action.
        void perform_action(float k_p, float k_i, float k_d);

        /// \brief Serializes the environment
        std::string dump();
    };
}


#endif //F16_ENVIRONMENT_H
