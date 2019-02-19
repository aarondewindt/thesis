#include <iostream>
#include "pole.h"
#include "grid_tile_coding.h"
#include <printf.h>
#include <inttypes.h>

typedef GridTileCoding<4>::XArray XArray;
typedef GridTileCoding<4>::ValueTileKeys ValueTileKeys;
typedef GridTileCoding<4>::TileKeyUnion TileKeyUnion;

int tilings = 10;


void print_value_tile_keys(ValueTileKeys &value_tiles) {
    printf("value: %f\n", value_tiles.value);
    auto *tile_keys = (TileKeyUnion*)value_tiles.tile_keys;
    for (int i = 0; i < tilings; i++) {
        printf("0x%016" PRIx64 " - %hu - ", tile_keys[i].tile_key, tile_keys[i].elements.tiling_idx);
        for (int j = 0; j < 4; j++) {
            printf("%i ", tile_keys[i].elements.x_idx[j]);
        }
        printf("\n");
    }
}

int main() {
    double center_coordinate[4] = {0, 0, 0, 0};
    double tile_size[4] = {1, 1, 1, 1};

    GridTileCoding<4> gtc(center_coordinate, tile_size, tilings, 0.0, true);

    printf("Center coordinates.\n");
    for (int i = 0; i < tilings; i++) {
        std::cout << gtc.center_coordinates[i] << std::endl;
    }
    std::cout << "\n";
    std::cout << "min: " << gtc.min_x << "\nmax: " << gtc.max_x << std::endl;

    {
        XArray values = {0.25, 0.25, .0, 0};
        ValueTileKeys value_tiles = *gtc.get_value_and_tile_keys(values);
        print_value_tile_keys(value_tiles);
        gtc.update_weights(5, value_tiles);
    }
    {
        XArray values = {0, 0, .0, 0};
        ValueTileKeys value_tiles = *gtc.get_value_and_tile_keys(values);
        print_value_tile_keys(value_tiles);
//        gtc.update_weights(-6, value_tiles);
    }
    {
        XArray values = {0.25, 0.25, .0, 0};
        ValueTileKeys value_tiles = *gtc.get_value_and_tile_keys(values);
        print_value_tile_keys(value_tiles);
//        gtc.update_weights(5, value_tiles);
    }
    return 0;
}
