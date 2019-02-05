//
// Created by elvieto on 25-11-18.
//

#ifndef POLE_RAND_H
#define POLE_RAND_H

#include <cstdlib>
#include <effolkronium/random.hpp>

// get base random alias which is auto seeded and has static API and internal state
using Random = effolkronium::random_static;

inline double frand(double min, double max)
{
//    double f = (double)rand() / RAND_MAX;
//    return min + f * (max - min);

    return Random::get(min, max);
}

inline double frand()
{
//    double f = (double)rand() / RAND_MAX;
//    return -1 + f * 2;

    return Random::get(-1.0, 1.0);
}

inline int irand(int min, int max) {
    return Random::get(min, max);
}

#endif //POLE_RAND_H
