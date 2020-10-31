//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_ENVIRONMENT_H
#define POLE_NEW_ENVIRONMENT_H

#include <array>
#include <tuple>

#include "types.h"
#include "tcb/span.hpp"
#include "effolkronium/random.hpp"


namespace pole {
    class Environment {
    public:
        f64 mass;
        f64 length;
        f64 inertia;
        f64 time = 0;
        f64 dt = 0.001;
        f64 theta_min_reward;
        f64 theta_terminate;
        f64 theta = 0;
        f64 theta_dot = 0;

        /// Create new environment with a pole with the given mass and length
        ///
        /// \param mass
        /// \param length
        inline Environment(f64 mass, f64 length, f64 theta_terminate, f64 theta_min_reward) :
                mass(mass),
                length(length),
                inertia(mass * length * length / 3),
                theta_terminate(theta_terminate),
                theta_min_reward(theta_min_reward) {}

        /// Reset the environment to its initial state.
        ///
        /// \param theta_ Initial theta
        /// \param theta_dot_ Initial theta_dot
        inline void reset(f64 theta_=0., f64 theta_dot_=0.) {
            time = 0;
            theta = theta_;
            theta_dot = theta_dot_;
        }

        /// Simulate one step with the given torque.
        ///
        /// \param torque Torque to apply on pole.
        /// \return Pair with the reward and boolean indicating whether this is a terminal state.
        inline std::pair<f64, bool> step(f64 torque) {
            // Add torque due to gravity.
            torque += mass * 9.81 * (length / 2) * sin(theta);

            f64 theta_dot_dot = torque / inertia;
            
            // Integrate
            theta_dot += theta_dot_dot * dt;
            theta += theta_dot * dt;
            time += dt;

            theta = fmod(fmod(theta + pi, pi2) + pi2, pi2) - pi;
            f64 theta_abs = fabs(theta);

            return std::make_pair(
                pow((theta_min_reward - theta_abs) / theta_min_reward, 2),
                theta_abs > theta_terminate);


        }

    };
}

#endif //POLE_NEW_ENVIRONMENT_H
