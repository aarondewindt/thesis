//
// Created by elviento on 12-2-19.
//

#ifndef POLE_GRID_TILE_CODING_H
#define POLE_GRID_TILE_CODING_H

#include <Eigen/Dense>
#include <unordered_map>
#include <array>
#include <printf.h>
#include <limits>
#include <cmath>


template <int _rank, int _tilings, class T = uint64_t>
class GridTileCoding {
public:
    /// 1D array with the length of rank. Aka the number of input values.
    typedef Eigen::Array<double, 1, _rank> XArray;

    /// 1D array with the length being the number of tilings.
    typedef Eigen::Array<double, 1, _tilings> TArray;

    // The keys we will use to identify each tile is a uint64_t, where the first
    // 7 bytes are used to store the indices in x_c and the last byte to store the
    // tiling index.
    // This union is used to build this key.
    union TileKeyUnion{
        uint64_t tile_key;
        struct {
            int8_t x_idx[7];
            uint8_t tiling_idx;
        } elements;
    };

    struct ValueTileKeys {
        double value;
        uint64_t tile_keys[_tilings];
    };

    struct TileInfo {
        double weight;
        T visit_count;
    };

    /// \param center_coordinate The coordinates of the center tile
    /// \param tile_size The tile size
    /// \param default_weight Initial weight for new tiles.
    GridTileCoding(
            const double *center_coordinate,
            const double *tile_size,
            double default_weight) : default_weight(default_weight)
    {
        // Copy the data to the local arrays.
        auto center_coordinate_data = this->center_coordinate.data();
        auto tile_size_data = this->tile_size.data();
        for (int i = 0; i < _rank; i++) {
            center_coordinate_data[i] = center_coordinate[i];
            tile_size_data[i] = tile_size[i];
        }
        tiling_offset = this->tile_size / tilings;

                     // Center coordinate of the first tiling.
        min_x = this->center_coordinate

                     // Add offset of last tiling to get the center coordinate of the last tiling.
                     + (this->tile_size - tiling_offset)

                     // Get the center coordinates of the lowest tile in the tiling
                     + (this->tile_size * std::numeric_limits<int8_t>::min())

                     // Remove half the tile size to get the lower corner of the tile.
                     // This is the lowest possible value covered by all tilings.
                     - (this->tile_size / 2);

                     // Center coordinate of the first tiling
        max_x = this->center_coordinate

                     // Center coordinate of the last tile of the first tiling
                     + (this->tile_size * std::numeric_limits<int8_t>::max())

                     // Add half a tile size to get the upper corner.
                     // This is the max value.
                     + (this->tile_size / 2);

    }

    /// Returns a ValueTileKey struct containing the value at x and the keys of the
    /// triggered tiles.
    /// \param x State(-action pair)
    /// \return Value
    inline ValueTileKeys get_value_and_tile_keys(XArray x) {
        // Create new value tile key struct.
        ValueTileKeys value_tile_keys;
        value_tile_keys.value = 0;

        // Loop through triggered keys and calculate the value and store tile keys.
        for (uint8_t i = 0; i < tilings; i++) {
            value_tile_keys.tile_keys[i] = get_tile_key(x, i);
            value_tile_keys.value += get_tile_info(value_tile_keys.tile_keys[i]).weight;
        }

        return value_tile_keys;
    }

    /// Update the triggered tiles by the given value.
    ///
    /// \param value
    /// \param value_tile_key
    inline void update_weights(double value, const ValueTileKeys value_tile_key) {
        update_weights(value, value_tile_key.tile_keys);
    }

    /// Update the tiles at x by the given value.
    inline void update_weights(double value, XArray x) {
        for (uint8_t i = 0; i < tilings; i++) {
            TileInfo tile_info = tiles[get_tile_key(x, i)];
            tile_info.weight += value;
            tile_info.visit_count += 1;
        }
    }

    /// Update the triggered tiles by the given value.
    /// \param value
    /// \param tile_keys
    inline void update_weights(double value, const uint64_t *tile_keys) {
        for (int i = 0; i < tilings; i++) {
            TileInfo tile_info = tiles[tile_keys[i]];
            tile_info.weight += value;
            tile_info.visit_count += 1;
        }
    }

    /// Get the weights of the triggered tiles at x.
    /// \param x
    /// \return
    TArray get_weights(XArray x) {
        static TArray weights;

        // Loop through each tiling and get the weight if the target tile on that tiling.
        for (uint8_t i = 0; i < tilings; i++) {
            weights(0, i) = get_tile_info(get_tile_key(x, i)).weight;
        }

        return weights;
    }

    /// Get the tile key for the tile on tiling_idx triggered at x.
    /// \param x Input state.
    /// \param tiling_idx
    /// \return
    inline uint64_t get_tile_key(XArray x, uint8_t tiling_idx) {
        TileKeyUnion tile_key_union;

        // Set tiling index
        tile_key_union.elements.tiling_idx = tiling_idx;

        // Saturate values between min_values and max values.
        x = x.cwiseMax(min_x);
        x = x.cwiseMin(max_x);

        // Calculated the "indices" of the triggered tile.
        // These are still doubles and need to rounded to the nearest integer,
        // but we do that later when inserting them into the union.
        XArray x_idx = ((x - (center_coordinate + (tiling_idx * tiling_offset))) / tile_size);

        // Copy the indices into the union
        for (int i = 0; i < rank; i++) {
            tile_key_union.elements.x_idx[i] = static_cast<int8_t>(std::lround(x_idx(0, i)));
        }

        return tile_key_union.tile_key;
    }

    inline TileInfo get_tile_info(uint64_t tile_key) {
        auto item = tiles.find(tile_key);
        if (item != tiles.end()) {
            return item->second;
        } else {
            TileInfo tile_info = TileInfo({default_weight, 0});
            tiles[tile_key] = tile_info;
            return tile_info;
        }
    }

//    /// Get the weight for a tile by it's tile_key. If the tile has not been
//    /// accessed before create it;s entry in the tiles hash map and set it's value
//    /// to default_value.
//    ///
//    /// \param tile_key Tile key of the target tile.
//    /// \return Weight of the tile
//    inline double get_weight(uint64_t tile_key) {
//        return get_tile_info(tile_key).weight;
//    }

    std::unordered_map<uint64_t, TileInfo> tiles;
    XArray center_coordinate;
    XArray tile_size;
    XArray tiling_offset;
    XArray min_x;
    XArray max_x;
    int rank = _rank;
    int tilings = _tilings;
    double default_weight;
};


#endif //POLE_GRID_TILE_CODING_H
