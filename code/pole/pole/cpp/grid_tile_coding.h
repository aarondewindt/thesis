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
#include "rand.h"
#include <inttypes.h>
#include <vector>


template <int _rank, class T = uint64_t>
class GridTileCoding {
public:
    /// 1D array with the length of rank. Aka the number of input values.
    typedef Eigen::Array<double, 1, _rank> XArray;

    // The keys we will use to identify each tile are of type uint64_t, where the first
    // 6 bytes are used to store the indices in x_c and the last two bytes to store the
    // tiling index.
    // This union is used to build this key.
    typedef uint16_t TilingIdxType;
    typedef int8_t XIdxType;
    union TileKeyUnion{
        uint64_t tile_key;
        struct {
            XIdxType x_idx[6];
            TilingIdxType tiling_idx;
        } elements;
    };

    // Holds the value and the keys of the tiles holding the value.
    struct ValueTileKeys {
        double value;
        uint64_t *tile_keys;

        ~ValueTileKeys() {
            delete tile_keys;
        }
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
            int tilings,
            double default_weight,
            bool random_offsets) : default_weight(default_weight), tilings(tilings)
    {
        // Copy the data to the local arrays.
        XArray cc;
        for (int i = 0; i < _rank; i++) {
            cc(0, i) = center_coordinate[i];
            this->tile_size(0, i) = tile_size[i];
        }

        // Choose the center coordinates for each tiling.
        // The first tiling will be placed at the center, the rest will be around it.
        // We will also look vor the min, max area covered by the tilings.
        center_coordinates = new XArray[tilings];
        center_coordinates[0] = cc;
        min_x = cc;
        max_x = cc;
        for (int i = 1; i < tilings; i++) {
            if (random_offsets){
                // Random offsets.
                center_coordinates[i] = cc + this->tile_size * frand(-0.5, 0.5);
            }else{
                // Uniform offsets.
                center_coordinates[i] = cc + (this->tile_size / tilings) * i;
            }


            // The minimum coordinate will be given by the most positive coordinates.
            // And the other way around for the max coordinate.
            min_x = min_x.cwiseMax(center_coordinates[i]);
            max_x + max_x.cwiseMin(center_coordinates[i]);
        }

        // Now we add/substract the total tiling size to get the extremes of the area
        // covered by the tilings.
        min_x += std::numeric_limits<XIdxType>::min() * this->tile_size - (this->tile_size / 2);
        max_x += std::numeric_limits<XIdxType>::max() * this->tile_size + (this->tile_size / 2);
    }

    ~GridTileCoding() {
        delete center_coordinates;
        for (auto const &item : tiles) {
            delete item.second;
        }
    }

    /// Returns a ValueTileKey struct containing the value at x and the keys of the
    /// triggered tiles.
    /// \param x State(-action pair)
    /// \return Value
    inline ValueTileKeys get_value_and_tile_keys(XArray &x) {
        // Create new value tile key struct.
        ValueTileKeys value_tile_keys;
        value_tile_keys.tile_keys = new uint64_t[tilings];
        value_tile_keys.value = 0;

        // Loop through triggered keys and calculate the value and store tile keys.
        for (TilingIdxType i = 0; i < tilings; i++) {
            value_tile_keys.tile_keys[i] = get_tile_key(x, i);
            value_tile_keys.value += get_tile_info(value_tile_keys.tile_keys[i])->weight;
        }

        return value_tile_keys;
    }

    /// Update the triggered tiles by the given value.
    ///
    /// \param value
    /// \param value_tile_key
    inline void update_weights(double value, ValueTileKeys &value_tile_key) {
        update_weights(value, value_tile_key.tile_keys);
    }

    /// Update the tiles at x by the given value.
    inline void update_weights(double value, XArray &x) {
        for (TilingIdxType i = 0; i < tilings; i++) {
            TileInfo *tile_info = get_tile_info(get_tile_key(x, i));
            tile_info->weight += value;
            tile_info->visit_count += 1;
        }
    }

    /// Update the weights of the triggered tiles with the given value. The
    /// value added to each tile will be divided with the number of tilings.
    /// \param value
    /// \param tile_keys
    inline void update_weights(double value, const uint64_t *tile_keys) {
        value /= tilings;
        for (int i = 0; i < tilings; i++) {
            TileInfo *tile_info = get_tile_info(tile_keys[i]);
            tile_info->weight += value;
            tile_info->visit_count += 1;
        }
    }

    /// Get the tile key for the tile on tiling_idx triggered at x.
    /// \param x Input state.
    /// \param tiling_idx
    /// \return
    inline uint64_t get_tile_key(XArray &x, TilingIdxType tiling_idx) {
        TileKeyUnion tile_key_union;

        // Set tiling index
        tile_key_union.elements.tiling_idx = tiling_idx;

        // Saturate values between min_values and max values.
        x = x.cwiseMax(min_x);
        x = x.cwiseMin(max_x);

        // Calculated the "indices" of the triggered tile.
        // These are still doubles and need to rounded to the nearest integer,
        // but we do that later when inserting them into the union.
        XArray x_idx = ((x - (center_coordinates[tiling_idx])) / tile_size);

        // Copy the indices into the union
        for (int i = 0; i < rank; i++) {
            tile_key_union.elements.x_idx[i] = static_cast<XIdxType>(std::lround(x_idx(0, i)));
        }

        // Set the remaining indices to 0. There are in total 6 indices in the key.
        for (int i = rank; i < 6; i++) {
            tile_key_union.elements.x_idx[i] = 0;
        }

        return tile_key_union.tile_key;
    }

    inline TileInfo* get_tile_info(uint64_t tile_key) {
        auto item = tiles.find(tile_key);
        if (item != tiles.end()) {
            printf("found 0x%016" PRIx64 "\n", tile_key);
            return item->second;
        } else {
            printf("nound 0x%016" PRIx64 "\n", tile_key);
            auto *tile_info = new TileInfo({default_weight, 0});
            tiles[tile_key] = tile_info;
            return tile_info;
        }
    }

    XArray *center_coordinates;
    XArray tile_size;
    XArray min_x;
    XArray max_x;
    int rank = _rank;
    int tilings;
    double default_weight;

private:
    std::unordered_map<uint64_t, TileInfo*> tiles;
};


#endif //POLE_GRID_TILE_CODING_H
