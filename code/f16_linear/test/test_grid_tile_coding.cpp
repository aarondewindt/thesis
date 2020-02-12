#include "catch.hpp"
#include "grid_tile_coding.h"
#include "match_pointer_array.h"
#include <vector>


SCENARIO("GridTileCoding updates and query values") {
    GIVEN("a clear GridTileCoding<3, 2> instance") {
        const int tilings = 2;
        const int rank = 3;
        const int n_values = 2;

        typedef float TCType;
        typedef GridTileCoding<rank, n_values, TCType>::ValueTileKeys ValueTileKeys;
        typedef GridTileCoding<rank, n_values, TCType>::TileKeyUnion TileKeyUnion;

        TCType center_coordinate[rank] = {0, 0, 0};
        TCType tile_size[rank] = {1, 1, 1};

        GridTileCoding<rank, n_values, TCType> gtc(center_coordinate, tile_size, tilings, 0.0, false);

        REQUIRE_THAT(gtc.center_coordinates[0], PArrayApprox<TCType>({0.0, 0.0, 0.0}, 1e-20, 1e-20));
        REQUIRE_THAT((TCType*)gtc.min_x, PArrayApprox<TCType>({-128.0, -128.0, -128.0}));
        REQUIRE_THAT((TCType*)gtc.max_x, PArrayApprox<TCType>({127.5, 127.5, 127.5}));

        TCType x[rank] = {0.5, 0.5, .0};
        WHEN("check values at {0.5, 0.5, .0} ") {
            ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(x);
            THEN("must equal {0, 0}") {
                REQUIRE_THAT((TCType*)value_tiles->values, PArrayApprox<TCType>({0.0, 0.0}, 1e-20, 1e-20));
            };
        }

        WHEN("set values at {0.5, 0.5, .0} to {5, 10}") {
            TCType values[n_values] = {5, 10};
            ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(x);
            gtc.update_weights(values, value_tiles);
            delete value_tiles;
            value_tiles = gtc.get_value_and_tile_keys(x);
            THEN("must equal {5, 10}") {
                REQUIRE_THAT((TCType*)value_tiles->values, PArrayApprox<TCType>({5.0, 10.0}));
            };
        }
    }

}