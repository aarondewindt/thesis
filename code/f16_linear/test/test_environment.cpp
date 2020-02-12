//
// Created by elvieto on 9-4-19.
//

#include "catch.hpp"
#include "match_pointer_array.h"
#include <vector>
#include <iostream>

#include "environment.h"

using namespace Catch::literals;

namespace f16 {
    TEST_CASE("Test Environment", "[environment]") {
        std::vector<float> a_matrix = {-8.89398e-02f, -1.07038e+01f, -3.21700e+01f, -3.97652e+00f,
                                       -5.04240e-04f, -5.16701e-02f, 4.70806e-13f, 9.79230e-01f,
                                       0.00000e+00f, 0.00000e+00f, 0.00000e+00f, 1.00000e+00f,
                                       -3.06961e-19f, -6.25562e-01f, 0.00000e+00f, -2.48488e-01f};

        std::vector<float> b_matrix = {-7.10382e-02f, -2.30832e-04f, 0.00000e+00f, -1.54077e-02f};

        Environment env = Environment(
            a_matrix,
            b_matrix,
            0.01,
            10
        );

        REQUIRE(env.t_end == 10_a);
        REQUIRE(env.dt == 0.01_a);
        REQUIRE(env.current_iteration == 0);
        REQUIRE(env.q_command.size() == 1000);
        REQUIRE(env.q_command.capacity() == 1000);

        SECTION("reset") {
            // Resetting states to 1, 2, 3, 4.
            env.current_iteration = 3;
            env.reset({1, 2, 3, 4}, {1, 2, 3, 4});
            REQUIRE_THAT((float*)env.dynamics_state.x, PArrayApprox({1, 2, 3, 4}));
            REQUIRE(env.current_iteration == 0);
        }

        SECTION("generate command") {

        }

    }
}
