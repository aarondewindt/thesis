#include <iostream>
#include "pole.h"
#include "grid_tile_coding.h"
#include <printf.h>

int main() {
    double center_coordinate[4] = {0, 0, 0, 0};
    double tile_size[4] = {3, 6, 9, 12};

    typedef GridTileCoding<4, 3>::XArray ValueArray;
    typedef GridTileCoding<4, 3>::XArray GridArray;

    GridTileCoding<4, 3> gtc(center_coordinate, tile_size, 0.0);

    std::cout << gtc.center_coordinate << std::endl;
    std::cout << gtc.tile_size << std::endl;
    std::cout << gtc.tiling_offset << std::endl << std::endl;


    ValueArray values = {4.6, 0.5, -0.5, 7};
    gtc.get_weights(values);

    return 0;
}