//
// Created by elvieto on 14-2-19.
//

#ifndef POLE_TILE_CODING_AGENT_H
#define POLE_TILE_CODING_AGENT_H

#include "grid_tile_coding.h"
#include "agent.h"


class TileCodingAgent : Agent {
public:
    typedef GridTileCoding<2>::XArray XArray;

    TileCodingAgent();


    GridTileCoding<2> tiles;

};


#endif //POLE_TILE_CODING_AGENT_H
