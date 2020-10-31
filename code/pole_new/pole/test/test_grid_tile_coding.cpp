//
// Created by adewindt on 6/10/20.
//

#include "catch2/catch.hpp"
#include "match_span.h"
#include "types.h"
#include "table_3d.h"
#include "spancpy.h"
#include "fmt/format.h"
#include <array>

#include "grid_tile_coding.h"

namespace pole::tests {
    SCENARIO("GridTileCoding", "") {
        GIVEN("") {
            const std::array<f64,
            GridTileCoding<2, 1> tiles;
        }
    }
}