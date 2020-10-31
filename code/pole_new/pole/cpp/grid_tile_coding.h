//
// Created by elviento on 12-2-19.
//

#ifndef POLE_GRID_TILE_CODING_H
#define POLE_GRID_TILE_CODING_H

#include <iostream>
#include <Eigen/Dense>
#include <unordered_map>
#include <array>
#include <printf.h>
#include <limits>
#include <cmath>
#include "rand.h"
#include <vector>
#include <cstring>
#include "types.h"


namespace pole {
    ///
    ///
    /// \tparam _rank
    /// \tparam _n_values
    template <usize _rank, usize _n_values>
    class GridTileCoding {
    public:
        /// 1D array with the length of rank. Aka the number of input values.
        typedef Eigen::Array<f64, 1, _rank> XArray;

        // The keys we will use to identify each tile are of type uint64_t, where the first
        // 6 bytes are used to store the indices in x_c and the last two bytes to store the
        // tiling index.
        // This union is used to build this key.
        typedef u16 TilingIdxType;
        typedef i8 XIdxType;
        union TileKeyUnion{
            u64 tile_key;
            struct {
                XIdxType x_idx[6];
                TilingIdxType tiling_idx;
            } elements;
        };

        // Holds the value and the keys of the tiles holding the value.
        struct ValueTileKeys {
            f64 values[_n_values];
            u64 *tile_keys;

            ValueTileKeys(f64 const *values, i32 tilings) {
                for (usize i = 0; i < _n_values; i++) {
                    this->values[i] = values[i];
                }
                tile_keys = new uint64_t[tilings];
            }

            ValueTileKeys(f64 value, i32 tilings){
                for (usize i = 0; i < _n_values; i++) {
                    this->values[i] = value;
                }
                tile_keys = new u64[tilings];
            }

            ~ValueTileKeys() {
                if (tile_keys != nullptr) {
                    delete [] tile_keys;
                    tile_keys = nullptr;
                }
            }
        };

        struct TileInfo {
            f64 weights[_n_values];
            u64 visit_count;

            TileInfo(f64 weight, u64 visit_count) : visit_count(visit_count) {
                for (usize i = 0; i < _n_values; i++) {
                    this->weights[i] = weight;
                }
            }
        };

        /// \param center_coordinate The coordinates of the center tile
        /// \param tile_size The tile size
        /// \param default_weight Initial weight for new tiles.
        GridTileCoding(
                f64 *center_coordinate,
                f64 *tile_size,
                i32 tilings,
                f64 default_weight,
                bool random_offsets) :
                default_weight(default_weight), tilings(tilings)
        {
            center_coordinates = new f64*[tilings];

            // Copy the data to the local arrays.
            XArray cc;
            XArray tcc;
            XArray ts;

            center_coordinates[0] = new f64[rank];
            for (usize i = 0; i < _rank; i++) {
                this->center_coordinates[0][i] = center_coordinate[i];
                this->tile_size[i] = tile_size[i];
                cc(0, i) = center_coordinate[i];
                ts(0, i) = tile_size[i];
            }

            // Choose the center coordinates for each tiling.
            // The first tiling will be placed at the center, the rest will be around it.
            // We will also look vor the min, max area covered by the tilings.
            XArray temp_min_x = cc;
            XArray temp_max_x = cc;
            for (usize i = 1; i < tilings; i++) {
                if (random_offsets){
                    // Random offsets.
                    XArray offset_factors = XArray::Random() * 0.5;
                    tcc = cc + ts * offset_factors;
                }else{
                    // Uniform offsets.
                    tcc = cc + (ts / tilings) * i;
                }

                this->center_coordinates[i] = new f64[rank];
                memcpy(this->center_coordinates[i], tcc.data(), sizeof(f64) * rank);

                // The minimum coordinate will be given by the most positive coordinates.
                // And the other way around for the max coordinate.
                temp_min_x = temp_min_x.cwiseMax(tcc);
                temp_max_x + temp_max_x.cwiseMin(tcc);
            }

            // Now we add/substract the total tiling size to get the extremes of the area
            // covered by the tilings.
            temp_min_x += std::numeric_limits<XIdxType>::min() * ts - (ts / 2);
            temp_max_x += std::numeric_limits<XIdxType>::max() * ts + (ts / 2);
            memcpy(min_x, temp_min_x.data(), sizeof(f64) * rank);
            memcpy(max_x, temp_max_x.data(), sizeof(f64) * rank);
        }

        ~GridTileCoding() {
            delete center_coordinates;
            for (auto const &item : tiles) {
                delete item.second;
            }
        }

        /// Returns a ValueTileKey struct containing the value at x and the keys of the
        /// triggered tiles.
        ///
        /// \param x State(-action pair)
        /// \return Value
        inline ValueTileKeys* get_value_and_tile_keys(f64* x) {
            // Create new value tile key struct.
            auto *value_tile_keys = new ValueTileKeys(0.0, tilings);

            // Loop through triggered keys and calculate the value and store tile keys.
            for (TilingIdxType i = 0; i < tilings; i++) {
                value_tile_keys->tile_keys[i] = get_tile_key(x, i);
                for (usize j = 0; j < _n_values; j++){
                    value_tile_keys->values[j] += get_tile_info(value_tile_keys->tile_keys[i])->weights[j];
                }
            }

            return value_tile_keys;
        }

        /// Update the triggered tiles by the given value.
        ///
        /// \param value
        /// \param value_tile_key
        inline void update_weights(f64 *values, ValueTileKeys* value_tile_key) {
            update_weights(values, value_tile_key->tile_keys);
        }

        /// Update the tiles at x by the given value.
        inline void update_weights(f64 *values, f64 *x) {
            for (TilingIdxType i = 0; i < tilings; i++) {
                TileInfo *tile_info = get_tile_info(get_tile_key(x, i));
                for (usize j = 0; j < _n_values; j++) {
                    tile_info->weights[j] += values[j] / tilings;
                }
                tile_info->visit_count += 1;
            }
        }

        /// Update the weights of the triggered tiles with the given value. The
        /// value added to each tile will be divided with the number of tilings.
        ///
        /// \param value
        /// \param tile_keys
        inline void update_weights(f64 *values, const u64 *tile_keys) {
            for (int i = 0; i < tilings; i++) {
                TileInfo *tile_info = get_tile_info(tile_keys[i]);
                for (int j = 0; j < _n_values; j++) {
                    tile_info->weights[j] += values[j] / tilings;
                }
                tile_info->visit_count += 1;
            }
        }

        /// Get the tile key for the tile on tiling_idx triggered at x.
        ///
        /// \param x Input state.
        /// \param tiling_idx
        /// \return The tile key for the given tile.
        inline uint64_t get_tile_key(f64 *x, TilingIdxType tiling_idx) {
            TileKeyUnion tile_key_union;

            // Set tiling index
            tile_key_union.elements.tiling_idx = tiling_idx;

            // Loop through each element
            for (usize i = 0; i < rank; i++) {
                // Saturate
                if (x[i] < min_x[i]) {
                    x[i] = min_x[i];
                } else if (x[i] > max_x[i]) {
                    x[i] = max_x[i];
                }

                // Calculate tile index.
                tile_key_union.elements.x_idx[i] = (XIdxType)std::lround((x[i] - center_coordinates[tiling_idx][i]) / tile_size[i]);
            }

            // Set the remaining indices to 0. There are in total 6 indices in the key.
            for (usize i = rank; i < 6; i++) {
                tile_key_union.elements.x_idx[i] = 0;
            }

            return tile_key_union.tile_key;
        }

        inline TileInfo* get_tile_info(u64 tile_key) {
            auto item = tiles.find(tile_key);
            if (item != tiles.end()) {
                return item->second;
            } else {
                auto *tile_info = new TileInfo(default_weight, 0);
                tiles[tile_key] = tile_info;
                return tile_info;
            }
        }

        f64 **center_coordinates;
        f64  tile_size[_rank];
        f64 min_x[_rank];
        f64 max_x[_rank];
        i32 rank = _rank;
        i32 n_values = _n_values;
        i32 tilings;
        f64 default_weight;

    private:
        std::unordered_map<u64, TileInfo*> tiles;
    };
}

#endif //POLE_GRID_TILE_CODING_H
