#include <iostream>
#include "pole.h"
#include "grid_tile_coding.h"
#include <printf.h>
#include <inttypes.h>


const int tilings = 2;
const int rank = 3;
const int n_values = 2;

typedef GridTileCoding<3, n_values>::ValueTileKeys ValueTileKeys;
typedef GridTileCoding<3, n_values>::TileKeyUnion TileKeyUnion;

void print_value_tile_keys(ValueTileKeys *value_tiles) {
    printf("value:");
    for (double value : value_tiles->values) {
        printf(" %f", value);
    }
    printf("\n");
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

    GridTileCoding<rank, n_values> gtc(center_coordinate, tile_size, tilings, 0.0, false);

    printf("Center coordinates.\n");
    for (int i = 0; i < tilings; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << gtc.center_coordinates[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "\nmin:";
    for (auto min : gtc.min_x) {
        std::cout << " " << min;
    }
    std::cout << "\nmax:";
    for (auto max : gtc.max_x) {
        std::cout << " " << max;
    }

    std::cout << "\n\n";

    double x[rank] = {0.5, 0.5, .0};
    {
        double values[n_values] = {5, 10};
        ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(x);
        print_value_tile_keys(value_tiles);
        gtc.update_weights(values, value_tiles);
        delete value_tiles;
        printf("\n");
    }
    {
        double values[n_values] = {-6, 3};
        ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(x);
        print_value_tile_keys(value_tiles);
        gtc.update_weights(values, value_tiles);
        delete value_tiles;
        printf("\n");
    }
    {
        double values[n_values] = {5, 9};
        ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(x);
        print_value_tile_keys(value_tiles);
        gtc.update_weights(values, value_tiles);
        delete value_tiles;
        printf("\n");
    }
    {
        ValueTileKeys *value_tiles = gtc.get_value_and_tile_keys(x);
        print_value_tile_keys(value_tiles);
        delete value_tiles;
    }

    return 0;
}
