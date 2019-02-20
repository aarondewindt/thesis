//
// Created by elvieto on 14-2-19.
//

#ifndef POLE_TILE_CODING_AGENT_H
#define POLE_TILE_CODING_AGENT_H

#include "grid_tile_coding.h"
#include "agent.h"
#include <vector>


class TileCodingAgent : Agent {
public:
    typedef GridTileCoding<3>::ValueTileKeys ValueTileKeys;
    typedef GridTileCoding<3>::TileInfo TileInfo;

    TileCodingAgent(
            std::vector<double> center,
            std::vector<double> tile_size,
            int tilings,
            double default_weight,
            bool random_offsets,
            double min_action,
            double max_action,
            int n_actions,
            double epsilon,
            double gamma,
            double alpha,
            double vc_min_theta,
            double vc_max_theta,
            int vc_n_theta,
            double vc_min_theta_dot,
            double vc_max_theta_dot,
            int vc_n_theta_dot

            );

    bool run_step() override;
    void run_episode(long max_steps) override;
    void begin_episode() override;
    inline void end_episode() override {};

    inline void set_environment(Pole *pole) override {
        this->pole = pole;
    }

    inline Pole *get_environment() override {
        return this->pole;
    }

    std::map<std::string, std::vector<double>*>* get_data() override;
    std::map<std::string, double> get_scalar_data() override;


    void greedy_action(double theta, double theta_dot,
                       double &best_action, ValueTileKeys* &best_value);

    double **greedy_action_map();
    double **update_count_map();

    GridTileCoding<3> tiles;
    Pole *pole;
    std::map<std::string, std::vector<double>*> data_map;

    double epsilon;
    double gamma;
    double alpha;
    double *actions;
    int n_actions;
    int tilings;

    uint64_t **visit_count;
    double vc_min_theta;
    double vc_max_theta;
    int vc_n_theta;
    double vc_min_theta_dot;
    double vc_max_theta_dot;
    int vc_n_theta_dot;
};


#endif //POLE_TILE_CODING_AGENT_H
