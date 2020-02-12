//
// Created by elviento on 3-4-19.
//

#include <cstring>
#include "environment.h"
#include "rand.h"


namespace f16 {
    Environment::Environment(
            std::vector<float> const &a_matrix,
            std::vector<float> const &b_matrix,
            float dt,
            float t_end) : dt(dt), current_iteration(0) {

        // Copy matrices
        std::memcpy((void *) a_matrix.data(), this->a_matrix, sizeof(float));
        std::memcpy((void *) b_matrix.data(), this->b_matrix, sizeof(float));

        // Calculate number of iterations, floor it, and use the calculated t_end.
        // This it to make sure t_end is an integer  multiple of dt.
        auto n_iterations = (unsigned int) (t_end / dt);
        this->t_end = n_iterations * dt;

        // Initialize the elevator input vector.
        q_command.resize(n_iterations);
    }

    void Environment::reset(std::vector<float> state_min, std::vector<float> state_max) {

        // Set state initial conditions.
        for (int i = 0; i < n_dynamics_state; i++) {
            dynamics_state.x[i] = random::get(state_min[i], state_max[i]);
        }

        // Set the current iteration back to 0.
        current_iteration = 0;
    }
}

