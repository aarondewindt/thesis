#include <iostream>
#include "pole.h"
#include "grid_tile_coding.h"
#include <printf.h>
#include <inttypes.h>

typedef GridTileCoding<3>::ValueTileKeys ValueTileKeys;
typedef GridTileCoding<3>::TileKeyUnion TileKeyUnion;

const int tilings = 2;
const int rank = 3;


void print_value_tile_keys(ValueTileKeys *value_tiles) {
    printf("value: %f\n", value_tiles->value);
    auto *tile_keys = (TileKeyUnion*)value_tiles->tile_keys;
    for (int i = 0; i < tilings; i++) {
        printf("0x%016" PRIx64 " - %hu - ", tile_keys[i].tile_key, tile_keys[i].elements.tiling_idx);
        for (int j = 0; j < rank; j++) {
            printf("%i ", tile_keys[i].elements.x_idx[j]);
        }
        printf("\n");
    }
}

int main() {
    double center_coordinate[rank] = {0, 0, 0};
    double tile_size[rank] = {1, 1, 1};

    GridTileCoding<3> gtc(center_coordinate, tile_size, tilings, 0.0, false);

    printf("Center coordinates.\n");
    for (int i = 0; i < tilings; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << gtc.center_coordinates[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n";
    std::cout << "min: " << gtc.min_x << "\nmax: " << gtc.max_x << std::endl;

    double values[rank] = {0.5, 0.5, .0};
    {

        ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(values);
        print_value_tile_keys(value_tiles);
        gtc.update_weights(5, value_tiles);
        delete value_tiles;
    }
    {
        ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(values);
        print_value_tile_keys(value_tiles);
        gtc.update_weights(-6, value_tiles);
        delete value_tiles;
    }
    {
        ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(values);
        print_value_tile_keys(value_tiles);
        gtc.update_weights(5, value_tiles);
        delete value_tiles;
    }
    return 0;
}
