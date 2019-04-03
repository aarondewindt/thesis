#include "catch.hpp"
#include "grid_tile_coding.h"
#include "match_pointer_array.h"
#include <vector>


SCENARIO("GridTileCoding can update and query values") {
    GIVEN("A clear GridTileCoding<3, 2> instance") {
        const int tilings = 2;
        const int rank = 3;
        const int n_values = 2;

        typedef GridTileCoding<rank, n_values>::ValueTileKeys ValueTileKeys;
        typedef GridTileCoding<rank, n_values>::TileKeyUnion TileKeyUnion;

        double center_coordinate[rank] = {0, 0, 0};
        double tile_size[rank] = {1, 1, 1};

        GridTileCoding<rank, n_values> gtc(center_coordinate, tile_size, tilings, 0.0, false);

        REQUIRE_THAT(gtc.center_coordinates[0], PArrayApprox({0.0, 0.0, 0.0}, 1e-20, 1e-20));
        REQUIRE_THAT((double*)gtc.min_x, PArrayApprox({-128.0, -128.0, -128.0}));
        REQUIRE_THAT((double*)gtc.max_x, PArrayApprox({127.5, 127.5, 127.5}));

        double x[rank] = {0.5, 0.5, .0};
        WHEN("check values at {0.5, 0.5, .0} ") {
            ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(x);
            THEN("the values must equal {0, 0}") {
                REQUIRE_THAT((double*)value_tiles->values, PArrayApprox({0.0, 0.0}, 1e-20, 1e-20));
            };
        }
    }

}